#pragma once
// Experimental mapped-host transports. No GPU peer access or CPU reduction
// workers. Deliberately separate from the calibrated production dispatcher.
#include "uccl_lite/cpu_router/packet_pipeline.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace uccl_lite::compact_experiment {
using namespace cpu_router;
inline void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
constexpr int kRanks = 8;
struct alignas(128) Word {
  uint32_t value;
};
struct Flags {
  Word ready, done, active;
};
struct alignas(128) Command {
  uint64_t value;
};
struct Binding {
  __nv_bfloat16 const* input;
  __nv_bfloat16* output;
  __nv_bfloat16* host[kRanks];
  Flags* flags[kRanks];
  Command* command;
  int rank, ranks;
};
__device__ inline uint32_t acquire(uint32_t const* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ inline void release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}
// Persistent kernels must bypass L1 for data written by later producers.
__device__ inline uint4 load4(void const* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ inline __nv_bfloat16 load1(__nv_bfloat16 const* p) {
  unsigned short v;
  asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
  return __ushort_as_bfloat16(v);
}
__device__ inline void copy(__nv_bfloat16* to, __nv_bfloat16 const* from,
                            size_t n, size_t tid, size_t stride) {
  if ((reinterpret_cast<uintptr_t>(to) | reinterpret_cast<uintptr_t>(from)) &
      15) {
    for (size_t i = tid; i < n; i += stride) to[i] = load1(from + i);
    return;
  }
  for (size_t i = tid; i < n / 8; i += stride)
    reinterpret_cast<uint4*>(to)[i] = load4(from + 8 * i);
  for (size_t i = n / 8 * 8 + tid; i < n; i += stride) to[i] = load1(from + i);
}
__global__ void miniGather(Binding b, size_t shard, uint32_t epoch) {
  auto grid = cooperative_groups::this_grid();
  if (grid.thread_rank() < b.ranks)
    while (acquire(&b.flags[grid.thread_rank()]->done.value) < epoch - 1)
      __nanosleep(64);
  grid.sync();
  copy(b.host[b.rank], b.input, shard, grid.thread_rank(), grid.size());
  __threadfence_system();
  grid.sync();
  if (grid.thread_rank() == 0) release(&b.flags[b.rank]->ready.value, epoch);
  int source = blockIdx.x % b.ranks;
  if (threadIdx.x == 0 && source != b.rank)
    while (acquire(&b.flags[source]->ready.value) < epoch) __nanosleep(64);
  __syncthreads();
  copy(b.output + source * shard, source == b.rank ? b.input : b.host[source],
       shard, (blockIdx.x / b.ranks) * blockDim.x + threadIdx.x,
       (gridDim.x / b.ranks) * blockDim.x);
  __threadfence_system();
  grid.sync();
  if (grid.thread_rank() == 0) release(&b.flags[b.rank]->done.value, epoch);
}
// op: 0 AG, 1 RS, 2 AR, 3 stop. Each command begins with all inputs ready;
// completion is global and host-blocking. One CTA/GPU is reserved until stop.
__global__ void resident(Binding b, uint32_t initialEpoch) {
  __shared__ uint32_t epoch, op;
  __shared__ size_t count;
  uint32_t previous = initialEpoch;
  if (!threadIdx.x) release(&b.flags[b.rank]->active.value, 1);
  for (;;) {
    if (!threadIdx.x) {
      uint64_t word;
      do {
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                     : "=l"(word)
                     : "l"(&b.command->value)
                     : "memory");
        __nanosleep(64);
      } while (uint32_t(word >> 32) == previous);
      epoch = uint32_t(word >> 32);
      op = uint32_t(word & 3);
      count = size_t((word >> 2) & 0x3fffffff);
    }
    __syncthreads();
    if (op == 3) break;
    size_t shard = count / b.ranks;
    copy(b.host[b.rank], b.input, op == 0 ? shard : count, threadIdx.x,
         blockDim.x);
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) release(&b.flags[b.rank]->ready.value, epoch);
    if (threadIdx.x < b.ranks)
      while (acquire(&b.flags[threadIdx.x]->ready.value) < epoch)
        __nanosleep(64);
    __syncthreads();
    if (op == 0) {
      for (int source = 0; source < b.ranks; ++source)
        copy(b.output + source * shard,
             source == b.rank ? b.input : b.host[source], shard, threadIdx.x,
             blockDim.x);
    } else {
      size_t n = op == 1 ? shard : count;
      for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        size_t index = op == 1 ? b.rank * shard + i : i;
        float sum = 0;
        for (int source = 0; source < b.ranks; ++source)
          sum += __bfloat162float(load1(b.host[source] + index));
        b.output[i] = __float2bfloat16_rn(sum);
      }
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) release(&b.flags[b.rank]->done.value, epoch);
    previous = epoch;
    __syncthreads();
  }
  if (!threadIdx.x) release(&b.flags[b.rank]->active.value, 0);
}
class Transport {
 public:
  Transport(Transport const&) = delete;
  Transport& operator=(Transport const&) = delete;
  Transport(std::vector<PacketPort> ports, size_t capacityBytes,
            int blocks = 16)
      : ports_(std::move(ports)), capacity_(capacityBytes), blocks_(blocks) {
    if (ports_.empty() || ports_.size() > kRanks || capacity_ < 16 ||
        capacity_ % (2 * ports_.size()) || blocks < int(ports_.size()) ||
        blocks > 32 || blocks % ports_.size())
      throw std::invalid_argument("invalid compact transport geometry");
    for (size_t r = 0; r < ports_.size(); ++r) {
      check(cudaSetDevice(ports_[r].device));
      int numa = gpuNumaNode(ports_[r].device);
      host_[r] = RegisteredMemory::pinned(capacity_, numa);
      flags_[r] = RegisteredMemory::pinned(sizeof(Flags), numa);
      std::memset(flags_[r].data(), 0, sizeof(Flags));
      check(cudaStreamCreateWithFlags(&streams_[r], cudaStreamNonBlocking));
      int active, sms;
      check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, miniGather,
                                                          256, 0));
      check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                   ports_[r].device));
      int coop;
      check(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch,
                                   ports_[r].device));
      if (!coop || blocks > active * sms)
        throw std::runtime_error("nonresident cooperative grid");
    }
    command_ = RegisteredMemory::pinned(sizeof(Command),
                                        gpuNumaNode(ports_[0].device));
    std::memset(command_.data(), 0, sizeof(Command));
    for (size_t r = 0; r < ports_.size(); ++r) {
      auto& b = bindings_[r];
      b.input = static_cast<__nv_bfloat16 const*>(ports_[r].input);
      b.output = static_cast<__nv_bfloat16*>(ports_[r].output);
      b.rank = int(r);
      b.ranks = int(ports_.size());
      for (size_t s = 0; s < ports_.size(); ++s) {
        b.host[s] = static_cast<__nv_bfloat16*>(
            host_[s].span().gpuAddress(ports_[r].device));
        b.flags[s] =
            static_cast<Flags*>(flags_[s].span().gpuAddress(ports_[r].device));
      }
      b.command =
          static_cast<Command*>(command_.span().gpuAddress(ports_[r].device));
    }
  }
  ~Transport() {
    // A failed GPU protocol cannot safely be unwound through cudaFree. Tests
    // use a process timeout; ordinary successful lifecycles always stop first.
    if (running_) stop();
    for (size_t r = 0; r < ports_.size(); ++r) {
      cudaSetDevice(ports_[r].device);
      cudaStreamSynchronize(ports_[r].stream);
      cudaStreamDestroy(streams_[r]);
    }
  }
  void gather(size_t count, std::vector<PacketPort> const& ports) {
    validate(0, count);
    if (running_ || mode_ == 2 || ports.size() != ports_.size())
      throw std::logic_error("invalid eager binding");
    for (size_t r = 0; r < ports.size(); ++r) {
      if (ports[r].device != ports_[r].device ||
          ports[r].stream != ports_[r].stream)
        throw std::invalid_argument("compact device/stream changed");
      auto input = reinterpret_cast<uintptr_t>(ports[r].input);
      auto output = reinterpret_cast<uintptr_t>(ports[r].output);
      if (!input || !output ||
          (input < output + count * 2 &&
           output < input + count * 2 / ports.size()))
        throw std::invalid_argument("mini-host requires disjoint input/output");
      check(cudaSetDevice(ports[r].device));
      cudaStreamCaptureStatus status;
      check(cudaStreamIsCapturing(ports[r].stream, &status));
      if (status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("mini-host is eager-only");
    }
    mode_ = 1;
    nextEpoch();
    size_t shard = count / ports.size();
    for (size_t r = 0; r < ports.size(); ++r) {
      check(cudaSetDevice(ports[r].device));
      Binding b = bindings_[r];
      b.input = static_cast<__nv_bfloat16 const*>(ports[r].input);
      b.output = static_cast<__nv_bfloat16*>(ports[r].output);
      void* args[] = {&b, &shard, &epoch_};
      check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(miniGather),
                                        dim3(blocks_), dim3(256), args, 0,
                                        ports[r].stream));
    }
  }
  void start() {
    if (running_ || mode_ == 1)
      throw std::logic_error("invalid resident lifecycle");
    if (epoch_ >= UINT32_MAX - 1)
      throw std::overflow_error("resident epoch exhausted");
    mode_ = 2;
    for (auto const& p : ports_) {
      check(cudaSetDevice(p.device));
      check(cudaStreamSynchronize(p.stream));
    }
    for (size_t r = 0; r < ports_.size(); ++r) {
      check(cudaSetDevice(ports_[r].device));
      resident<<<1, 256, 0, streams_[r]>>>(bindings_[r], epoch_);
      check(cudaGetLastError());
    }
    running_ = true;
    wait(true, 1);
  }
  void run(int op, size_t count) {
    validate(op, count);
    if (!running_) throw std::logic_error("resident not started");
    if (epoch_ >= UINT32_MAX - 1)
      throw std::overflow_error("resident epoch exhausted; stop required");
    publish(op, count);
    wait(false, epoch_);
  }
  void stop() {
    if (!running_) return;
    publish(3, 0);
    wait(true, 0);
    for (size_t r = 0; r < ports_.size(); ++r) {
      check(cudaSetDevice(ports_[r].device));
      check(cudaStreamSynchronize(streams_[r]));
    }
    running_ = false;
  }

 private:
  void validate(int op, size_t count) const {
    if (op < 0 || op > 2 || !count || count % ports_.size() ||
        count > capacity_ / 2 || count >= (1UL << 30))
      throw std::invalid_argument("invalid compact extent/operation");
  }
  void nextEpoch() {
    if (epoch_ == UINT32_MAX)
      throw std::overflow_error("compact epoch exhausted");
    ++epoch_;
  }
  void publish(int op, size_t count) {
    nextEpoch();
    uint64_t word = (uint64_t(epoch_) << 32) | (uint64_t(count) << 2) | op;
    __atomic_store_n(&static_cast<Command*>(command_.data())->value, word,
                     __ATOMIC_RELEASE);
  }
  void wait(bool active, uint32_t value) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (size_t r = 0; r < ports_.size(); ++r) {
      auto* f = static_cast<Flags*>(flags_[r].data());
      auto* p = active ? &f->active.value : &f->done.value;
      size_t spins = 0;
      while (__atomic_load_n(p, __ATOMIC_ACQUIRE) != value) {
        asm volatile("pause");
        if (!(++spins % 65536) && std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error("compact transport timeout");
      }
    }
  }
  std::vector<PacketPort> ports_;
  size_t capacity_;
  int blocks_;
  uint32_t epoch_ = 0;
  bool running_ = false;
  int mode_ = 0;  // One allocation serves either eager or resident lifecycle.
  std::array<RegisteredMemory, kRanks> host_, flags_;
  RegisteredMemory command_;
  std::array<Binding, kRanks> bindings_{};
  std::array<cudaStream_t, kRanks> streams_{};
};
}  // namespace uccl_lite::compact_experiment

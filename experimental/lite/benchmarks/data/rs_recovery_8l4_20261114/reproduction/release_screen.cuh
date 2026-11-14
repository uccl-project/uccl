#pragma once
#include "compact_transport.cuh"
#include "uccl_lite/cpu_router/cpu_executor.hpp"
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sched.h>

namespace uccl_lite::compact_experiment {
struct RingFlags {
  Word progress[128];
};
struct RingBinding {
  __nv_bfloat16 const* input;
  __nv_bfloat16* output;
  __nv_bfloat16* host[2][kRanks];
  RingFlags* flags[kRanks];
  __nv_bfloat16* scratch[2];
  int rank, ranks;
};
__device__ inline uint32_t ringAdd(uint32_t a, uint32_t b) {
  __nv_bfloat162_raw x{uint16_t(a), uint16_t(a >> 16)};
  __nv_bfloat162_raw y{uint16_t(b), uint16_t(b >> 16)};
  __nv_bfloat162_raw out = __hadd2(__nv_bfloat162(x), __nv_bfloat162(y));
  return uint32_t(out.x) | (uint32_t(out.y) << 16);
}
// A host-mediated ring: only adjacent ranks exchange partials. Each hop adds
// one local BF16 contribution and rounds to BF16. Reader credits protect both
// banks across changing extents and asynchronously queued calls.
__global__ void hostRingScatter(RingBinding b, size_t shard, uint32_t base) {
  auto grid = cooperative_groups::this_grid();
  int prev = (b.rank + b.ranks - 1) % b.ranks, next = (b.rank + 1) % b.ranks;
  // Previous-command completion releases both banks at our sole reader.
  if (grid.thread_rank() == 0 && base)
    while (acquire(&b.flags[next]->progress[0].value) < base) __nanosleep(64);
  grid.sync();
  copy(b.host[0][b.rank], b.input + size_t(prev) * shard, shard,
       grid.thread_rank(), grid.size());
  grid.sync();
  if (grid.thread_rank() == 0)
    release(&b.flags[b.rank]->progress[0].value, base + 1);
  for (int step = 0; step < b.ranks - 1; ++step) {
    int slot = step & 1, outputSlot = slot ^ 1;
    uint32_t seq = base + step + 1;
    bool final = step == b.ranks - 2;
    if (grid.thread_rank() == 0) {
      while (acquire(&b.flags[prev]->progress[0].value) < seq) __nanosleep(64);
      // Publishing stage step also acknowledges reading the preceding bank.
      // The spare bank for step zero was freed before this command started.
      if (!final && step)
        while (acquire(&b.flags[next]->progress[0].value) < seq) __nanosleep(64);
    }
    grid.sync();
    int chunk = (b.rank + b.ranks - step - 2) % b.ranks;
    auto const* local = b.input + size_t(chunk) * shard;
    auto const* remote = b.host[slot][prev];
    auto* out = final ? b.output : b.host[outputSlot][b.rank];
    if (shard % 8 == 0 && !((reinterpret_cast<uintptr_t>(b.input) |
                             reinterpret_cast<uintptr_t>(b.output)) &
                            15)) {
      for (size_t i = grid.thread_rank(); i < shard / 8; i += grid.size()) {
        auto a = load4(remote + i * 8), v = load4(local + i * 8);
        reinterpret_cast<uint4*>(out)[i] =
            make_uint4(ringAdd(a.x, v.x), ringAdd(a.y, v.y), ringAdd(a.z, v.z),
                       ringAdd(a.w, v.w));
      }
    } else {
      for (size_t i = grid.thread_rank(); i < shard; i += grid.size())
        out[i] = __float2bfloat16_rn(__bfloat162float(load1(remote + i)) +
                                     __bfloat162float(load1(local + i)));
    }
      grid.sync();
    if (grid.thread_rank() == 0) {
      release(&b.flags[b.rank]->progress[0].value, seq + 1);
    }
  }
}
__global__ void hostRingCtaScatter(RingBinding b, size_t shard, uint32_t base) {

  int prev = (b.rank + b.ranks - 1) % b.ranks, next = (b.rank + 1) % b.ranks;
  // Previous-command completion releases both banks at our sole reader.
  if (threadIdx.x == 0 && base)
    while (acquire(&b.flags[next]->progress[blockIdx.x].value) < base) __nanosleep(64);
  __syncthreads();
  copy(b.host[0][b.rank], b.input + size_t(prev) * shard, shard,
       blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x);
  __syncthreads();
  if (threadIdx.x == 0)
    release(&b.flags[b.rank]->progress[blockIdx.x].value, base + 1);
  for (int step = 0; step < b.ranks - 1; ++step) {
    int slot = step & 1, outputSlot = slot ^ 1;
    uint32_t seq = base + step + 1;
    bool final = step == b.ranks - 2;
    if (threadIdx.x == 0) {
      while (acquire(&b.flags[prev]->progress[blockIdx.x].value) < seq) __nanosleep(64);
      // Publishing stage step also acknowledges reading the preceding bank.
      // The spare bank for step zero was freed before this command started.
      if (!final && step)
        while (acquire(&b.flags[next]->progress[blockIdx.x].value) < seq) __nanosleep(64);
    }
    __syncthreads();
    int chunk = (b.rank + b.ranks - step - 2) % b.ranks;
    auto const* local = b.input + size_t(chunk) * shard;
    auto const* remote = b.host[slot][prev];
    auto* out = final ? b.output : b.host[outputSlot][b.rank];
    if (shard % 8 == 0 && !((reinterpret_cast<uintptr_t>(b.input) |
                             reinterpret_cast<uintptr_t>(b.output)) &
                            15)) {
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < shard / 8; i += blockDim.x * gridDim.x) {
        auto a = load4(remote + i * 8), v = load4(local + i * 8);
        reinterpret_cast<uint4*>(out)[i] =
            make_uint4(ringAdd(a.x, v.x), ringAdd(a.y, v.y), ringAdd(a.z, v.z),
                       ringAdd(a.w, v.w));
      }
    } else {
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < shard; i += blockDim.x * gridDim.x)
        out[i] = __float2bfloat16_rn(__bfloat162float(load1(remote + i)) +
                                     __bfloat162float(load1(local + i)));
    }
      __syncthreads();
    if (threadIdx.x == 0) {
      release(&b.flags[b.rank]->progress[blockIdx.x].value, seq + 1);
    }
  }
}
// Recursive halving pairs the low rank bits first, retaining local-NUMA
// partials until the last exchange on the measured 4+4 rank placement.
__global__ void hostHalvingScatter(RingBinding b, size_t shard, uint32_t base) {
  auto grid = cooperative_groups::this_grid();
  int stages = __ffs(b.ranks) - 1;
  if (grid.thread_rank() == 0 && base)
    for (int j = 0; j < stages; ++j)
      while (acquire(&b.flags[b.rank ^ (1 << j)]->progress[0].value) < base)
        __nanosleep(64);
  grid.sync();
  bool vector = shard % 8 == 0 &&
      !((reinterpret_cast<uintptr_t>(b.input) |
         reinterpret_cast<uintptr_t>(b.output)) & 15);
  size_t unit = vector ? 8 : 1;
  size_t width = shard / unit;
  for (size_t i = grid.thread_rank(); i < width * (b.ranks / 2);
       i += grid.size()) {
    size_t d = ((i / width) << 1) | ((b.rank & 1) ^ 1);
    auto* out = b.host[0][b.rank] + i * unit;
    auto* in = b.input + d * shard + (i % width) * unit;
    if (vector) *reinterpret_cast<uint4*>(out) = load4(in);
    else *out = load1(in);
  }
  grid.sync();
  if (grid.thread_rank() == 0) release(&b.flags[b.rank]->progress[0].value, base + 1);
  for (int j = 0; j < stages; ++j) {
    int peer = b.rank ^ (1 << j);
    uint32_t seq = base + j + 1;
    if (grid.thread_rank() == 0) {
      while (acquire(&b.flags[peer]->progress[0].value) < seq) __nanosleep(64);
      if (j && j + 1 < stages)
        while (acquire(&b.flags[b.rank ^ (1 << (j - 1))]->progress[0].value) < seq)
          __nanosleep(64);
    }
    grid.sync();
    for (size_t i = grid.thread_rank(); i < width * (b.ranks >> (j + 1));
         i += grid.size()) {
      size_t k = i / width, offset = (i % width) * unit;
      size_t d = (k << (j + 1)) | (b.rank & ((1 << (j + 1)) - 1));
      auto* remote = b.host[j & 1][peer] + i * unit;
      auto* local = j ? b.scratch[(j - 1) & 1] + i * unit
                      : b.input + d * shard + offset;
      bool send = (d ^ b.rank) & (1 << (j + 1));
      auto* out = j + 1 == stages ? b.output + offset
          : (send ? b.host[(j + 1) & 1][b.rank] : b.scratch[j & 1])
                + (k >> 1) * shard + offset;
      if (vector) {
        auto a = load4(remote), v = load4(local);
        *reinterpret_cast<uint4*>(out) = make_uint4(
            ringAdd(a.x, v.x), ringAdd(a.y, v.y), ringAdd(a.z, v.z), ringAdd(a.w, v.w));
      } else *out = __float2bfloat16_rn(__bfloat162float(load1(remote)) +
                                      __bfloat162float(load1(local)));
    }
      grid.sync();
    if (grid.thread_rank() == 0) release(&b.flags[b.rank]->progress[0].value, seq + 1);
  }
}
// Eight-rank 4+4 prototype. Stage one collects NUMA-local inputs; stage
// two writes a partial for the paired remote owner and retains its own partial.
__global__ void hostHierScatter(RingBinding b, size_t shard, uint32_t base) {
  auto grid = cooperative_groups::this_grid();
  int group = b.rank & ~3, other = b.rank ^ 4;
  if (grid.thread_rank() == 0 && base)
    for (int r = 0; r < b.ranks; ++r)
      while (acquire(&b.flags[r]->progress[0].value) < base) __nanosleep(64);
  grid.sync();
  // Each local reducer retains two original destination shards on-device.
  // Only the other six destination shards have readers on this NUMA node.
  for (int d = 0; d < b.ranks; ++d)
    if ((d & 3) != (b.rank & 3))
      copy(b.host[0][b.rank] + d * shard, b.input + d * shard, shard,
           grid.thread_rank(), grid.size());
  grid.sync();
  if (grid.thread_rank() == 0) release(&b.flags[b.rank]->progress[0].value, base + 1);
  if (grid.thread_rank() == 0)
    for (int r = group; r < group + 4; ++r)
      while (acquire(&b.flags[r]->progress[0].value) < base + 1) __nanosleep(64);
  grid.sync();
  if (shard % 8 == 0) {
    for (size_t i = grid.thread_rank(); i < shard / 8 * 2; i += grid.size()) {
      size_t offset = i % (shard / 8) * 8;
      int dest = int(i / (shard / 8)) * 4 + (b.rank & 3);
      float sums[8] = {};
      for (int r = group; r < group + 4; ++r) {
        auto v = load4((r == b.rank ? b.input : b.host[0][r]) + dest * shard + offset);
        auto* h = reinterpret_cast<__nv_bfloat16*>(&v);
        for (int j = 0; j < 8; ++j) sums[j] += __bfloat162float(h[j]);
      }
      uint4 v;
      auto* h = reinterpret_cast<__nv_bfloat16*>(&v);
      for (int j = 0; j < 8; ++j) h[j] = __float2bfloat16_rn(sums[j]);
      auto* out = dest == b.rank ? b.scratch[0] : b.host[1][b.rank];
      *reinterpret_cast<uint4*>(out + offset) = v;
    }
  } else {
    for (size_t i = grid.thread_rank(); i < shard * 2; i += grid.size()) {
      int dest = int(i / shard) * 4 + (b.rank & 3);
      float sum = 0;
      for (int r = group; r < group + 4; ++r)
        sum += __bfloat162float(load1((r == b.rank ? b.input : b.host[0][r]) + dest * shard + i % shard));
      (dest == b.rank ? b.scratch[0] : b.host[1][b.rank])[i % shard] = __float2bfloat16_rn(sum);
    }
  }
  grid.sync();
  if (grid.thread_rank() == 0) {
    release(&b.flags[b.rank]->progress[0].value, base + 2);
    while (acquire(&b.flags[other]->progress[0].value) < base + 2) __nanosleep(64);
  }
  grid.sync();
  if (shard % 8 == 0) {
    for (size_t i = grid.thread_rank(); i < shard / 8; i += grid.size()) {
      auto a = load4(b.host[1][other] + i * 8), v = load4(b.scratch[0] + i * 8);
      reinterpret_cast<uint4*>(b.output)[i] = make_uint4(
          ringAdd(a.x, v.x), ringAdd(a.y, v.y), ringAdd(a.z, v.z), ringAdd(a.w, v.w));
    }
  } else {
  for (size_t i = grid.thread_rank(); i < shard; i += grid.size())
    b.output[i] = __float2bfloat16_rn(__bfloat162float(load1(b.host[1][other] + i)) +
                                    __bfloat162float(b.scratch[0][i]));
  }
  grid.sync();
  if (grid.thread_rank() == 0) release(&b.flags[b.rank]->progress[0].value, base + 3);
}
class HostRing {
 public:
  HostRing(std::vector<PacketPort> ports, size_t fullBytes, int blocks, bool halving = false, bool cta = false, bool hier = false)
      : ports_(std::move(ports)), capacity_(fullBytes), blocks_(blocks), halving_(halving), cta_(cta), hier_(hier) {
    if (ports_.size() < 2 || ports_.size() > kRanks || !fullBytes ||
        fullBytes % (ports_.size() * 2) || blocks < 1 || blocks > 128)
      throw std::invalid_argument("invalid host ring geometry");
    if (hier && ports_.size() != 8) throw std::invalid_argument("hier needs eight ranks");
    if (halving && (ports_.size() & (ports_.size() - 1)))
      throw std::invalid_argument("halving needs power-of-two ranks");
    size_t stride = ((fullBytes / (hier ? 1 : (halving ? 2 : ports_.size())) + 15) / 16) * 16;
    for (size_t r = 0; r < ports_.size(); ++r) {
      check(cudaSetDevice(ports_[r].device));
      int node = gpuNumaNode(ports_[r].device);
      memory_[r] = RegisteredMemory::pinned(2 * stride, node);
      flags_[r] = RegisteredMemory::pinned(sizeof(RingFlags), node);
      std::memset(flags_[r].data(), 0, sizeof(RingFlags));
      if (halving || hier) {
        for (auto& p : bindings_[r].scratch) check(cudaMalloc(&p, stride));
      }
      int active, sms;
      check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, hier ? hostHierScatter : (halving ? hostHalvingScatter : (cta ? hostRingCtaScatter : hostRingScatter)), 256, 0));
      check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                   ports_[r].device));
      if (blocks_ > active * sms)
        throw std::invalid_argument("ring grid is not resident");
    }
    for (size_t r = 0; r < ports_.size(); ++r) {
      auto& b = bindings_[r];
      b.rank = int(r);
      b.ranks = int(ports_.size());
      for (size_t s = 0; s < ports_.size(); ++s) {
        auto* p =
            static_cast<char*>(memory_[s].span().gpuAddress(ports_[r].device));
        b.host[0][s] = reinterpret_cast<__nv_bfloat16*>(p);
        b.host[1][s] = reinterpret_cast<__nv_bfloat16*>(p + stride);
        b.flags[s] = static_cast<RingFlags*>(
            flags_[s].span().gpuAddress(ports_[r].device));
      }
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    std::ifstream file("/sys/fs/cgroup/cpuset.cpus.effective");
    std::string list, part;
    std::getline(file, list);
    if (list.empty()) throw std::runtime_error("cannot read cpuset");
    std::istringstream fields(list);
    while (std::getline(fields, part, ',')) {
      auto dash = part.find('-');
      int lo = std::stoi(part), hi = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
      for (int cpu = lo; cpu <= hi && cpu < CPU_SETSIZE; ++cpu) CPU_SET(cpu, &allowed);
    }
    std::vector<int> cpus;
    for (auto const& p : ports_) {
      int node = gpuNumaNode(p.device), selected = -1;
      for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu)
        if (CPU_ISSET(cpu, &allowed) &&
            std::find(cpus.begin(), cpus.end(), cpu) == cpus.end() &&
            std::filesystem::exists("/sys/devices/system/node/node" +
                                    std::to_string(node) + "/cpu" + std::to_string(cpu))) {
          selected = cpu;
          break;
        }
      if (selected < 0) throw std::runtime_error("no local launch CPU");
      cpus.push_back(selected);
      lanes_.push_back(lanes_.size());
    }
    launches_ = std::make_unique<CpuExecutor>(cpus, 128, 2000);

  }
  HostRing(HostRing const&) = delete;
  HostRing& operator=(HostRing const&) = delete;
  ~HostRing() {
    if (launches_) launches_->stop();
    for (auto const& p : ports_) {
      cudaSetDevice(p.device);
      cudaStreamSynchronize(p.stream);
    }
    for (size_t r = 0; r < ports_.size(); ++r) {
      cudaSetDevice(ports_[r].device);
      for (auto* p : bindings_[r].scratch) if (p) cudaFree(p);
    }
  }
  void scatter(size_t count, std::vector<PacketPort> const& ports) {
    if (ports.size() != ports_.size() || !count || count % ports_.size() ||
        count > capacity_ / 2)
      throw std::invalid_argument("invalid ring extent");
    uint32_t steps = hier_ ? 3 : halving_ ? uint32_t(__builtin_ctz(unsigned(ports.size())) + 1) : uint32_t(ports.size());
    if (base_ > UINT32_MAX - steps)
      throw std::overflow_error("ring sequence exhausted");
    for (size_t r = 0; r < ports.size(); ++r) {
      if (ports[r].device != ports_[r].device ||
          ports[r].stream != ports_[r].stream ||
          ports[r].input == ports[r].output)
        throw std::invalid_argument(
            "ring needs fixed streams and out-of-place buffers");
      check(cudaSetDevice(ports[r].device));
      cudaStreamCaptureStatus capture;
      check(cudaStreamIsCapturing(ports[r].stream, &capture));
      if (capture != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("prototype ring is eager-only");
    }
    size_t shard = count / ports.size();
    shard_ = shard;
    for (size_t r = 0; r < ports.size(); ++r) {
      bindings_[r].input = static_cast<__nv_bfloat16 const*>(ports[r].input);
      bindings_[r].output = static_cast<__nv_bfloat16*>(ports[r].output);
      errors_[r] = nullptr;
      done_[r].store(false, std::memory_order_relaxed);
    }
    launches_->enqueueBatch(lanes_.data(), lanes_.size(), launch, this);
    for (size_t r = 0; r < ports.size(); ++r) {
      while (!done_[r].load(std::memory_order_acquire)) asm volatile("pause");
      if (errors_[r]) std::rethrow_exception(errors_[r]);
    }
    base_ += steps;
  }

 private:
  static void launch(void* context, int r) {
    auto& o = *static_cast<HostRing*>(context);
    try {
      check(cudaSetDevice(o.ports_[r].device));
      void* args[] = {&o.bindings_[r], &o.shard_, &o.base_};
      check(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(o.hier_ ? hostHierScatter : (o.halving_ ? hostHalvingScatter : (o.cta_ ? hostRingCtaScatter : hostRingScatter))),
          dim3(o.blocks_), dim3(256), args, 0, o.ports_[r].stream));
    } catch (...) { o.errors_[r] = std::current_exception(); }
    o.done_[r].store(true, std::memory_order_release);
  }
  std::unique_ptr<CpuExecutor> launches_;
  std::vector<size_t> lanes_;
  std::array<std::atomic<bool>, kRanks> done_;
  std::array<std::exception_ptr, kRanks> errors_;
  size_t shard_ = 0;
  std::vector<PacketPort> ports_;
  size_t capacity_;
  int blocks_;
  bool halving_, cta_, hier_;
  uint32_t base_ = 0;
  std::array<RegisteredMemory, kRanks> memory_, flags_;
  std::array<RingBinding, kRanks> bindings_{};
};
}  // namespace uccl_lite::compact_experiment

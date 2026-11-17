#pragma once
// Eager host-packet AllGather for eight MPI ranks on two four-GPU hosts.
// All arithmetic belongs to the caller; this engine only moves immutable
// shards.
#include <linux/mempolicy.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <mpi.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite {
namespace host_gather {
inline void cudaCheck(int line, cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error("host gather CUDA line " + std::to_string(line) +
                             ": " + cudaGetErrorString(e));
}
inline void check(int e) {
  if (e != MPI_SUCCESS) throw std::runtime_error("host gather MPI failure");
}
inline void clearMpiProbeStatus() {
  auto prior = cudaGetLastError();
  if (prior != cudaSuccess && prior != cudaErrorInvalidResourceHandle)
    cudaCheck(__LINE__, prior);
}
struct Sources {
  uint4 const* p[8];
};
static __global__ void consume(Sources sources, uint4* output, size_t shard,
                               size_t offset, size_t length, int first) {
  int source = first + blockIdx.y;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length;
       i += size_t(gridDim.x) * blockDim.x)
    output[size_t(source) * shard + offset + i] = sources.p[source][i];
}
// Experimental consumer: each source points to a full-input packet, not a
// reduced shard. All eight contributions are summed directly into GPU output.
static __global__ void reduceInputs(Sources sources, float4* output,
                                    size_t offset, size_t length) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length;
       i += size_t(gridDim.x) * blockDim.x) {
    float4 sum = make_float4(0, 0, 0, 0);
    for (int source = 0; source < 8; ++source) {
      float4 value = reinterpret_cast<float4 const*>(sources.p[source])[i];
      sum.x += value.x;
      sum.y += value.y;
      sum.z += value.z;
      sum.w += value.w;
    }
    output[offset + i] = sum;
  }
}
class Gather {
  static constexpr int slots = 3;
  struct Slot {
    bool active = false, uploaded = false, local = false, sent = false;
    bool remote = false, acked = false;
    size_t offset = 0, length = 0;
    std::array<MPI_Request, 3> readyRecv{}, readySend{}, ackRecv{}, ackSend{};
    std::array<MPI_Request, 2> netRecv{}, netSend{};
    MPI_Request netReady = MPI_REQUEST_NULL;
    cudaEvent_t up{}, down{};
    Slot() {
      readyRecv.fill(MPI_REQUEST_NULL);
      readySend.fill(MPI_REQUEST_NULL);
      ackRecv.fill(MPI_REQUEST_NULL);
      ackSend.fill(MPI_REQUEST_NULL);
      netRecv.fill(MPI_REQUEST_NULL);
      netSend.fill(MPI_REQUEST_NULL);
    }
  };
  int rank_, local_, group_, node_, peerNode_;
  bool leader_;
  size_t capacity_;
  MPI_Win window_ = MPI_WIN_NULL;
  std::array<char*, 2> host_{}, mapped_{};
  cudaStream_t upload_{}, consume_{};
  std::array<Slot, slots> state_;
  static bool done(MPI_Request& r) {
    int flag;
    check(MPI_Test(&r, &flag, MPI_STATUS_IGNORE));
    return flag;
  }
  template <size_t N>
  static bool done(std::array<MPI_Request, N>& r) {
    int flag;
    check(MPI_Testall(N, r.data(), &flag, MPI_STATUSES_IGNORE));
    return flag;
  }
  bool done(cudaEvent_t event) {
    cudaCheck(__LINE__, cudaSetDevice(local_));
    auto e = cudaEventQuery(event);
    if (e == cudaErrorNotReady) return false;
    cudaCheck(__LINE__, e);
    return true;
  }
  char* packet(int group, int slot, int section = 0) {
    return host_[group] + (size_t(slot) * 6 + section * 2) * capacity_;
  }

 public:
  Gather(int rank, MPI_Comm localComm, size_t maxShard, size_t chunk)
      : rank_(rank),
        local_(rank % 4),
        group_(local_ / 2),
        node_(rank / 4),
        peerNode_(node_ ^ 1),
        leader_(!(local_ & 1)),
        capacity_(std::min(maxShard, chunk)) {
    int n, l, w;
    check(MPI_Comm_size(localComm, &n));
    check(MPI_Comm_rank(localComm, &l));
    check(MPI_Comm_size(MPI_COMM_WORLD, &w));
    if (n != 4 || l != local_ || w != 8 || !capacity_ || capacity_ % 16)
      throw std::runtime_error(
          "host gather requires two contiguous four-GPU hosts");
    size_t page = sysconf(_SC_PAGESIZE),
           bytes = (slots * 6 * capacity_ + page - 1) / page * page;
    MPI_Info info;
    check(MPI_Info_create(&info));
    check(MPI_Info_set(info, "alloc_shared_noncontig", "true"));
    void* raw = nullptr;
    check(MPI_Win_allocate_shared(leader_ ? bytes + page : 0, 1, info,
                                  localComm, &raw, &window_));
    check(MPI_Info_free(&info));
    int present = 0;
    int* model = nullptr;
    check(MPI_Win_get_attr(window_, MPI_WIN_MODEL, &model, &present));
    if (!present || *model != MPI_WIN_UNIFIED)
      throw std::runtime_error("unified shared window required");
    for (int g = 0; g < 2; ++g) {
      MPI_Aint size;
      int disp;
      void* ptr = nullptr;
      check(MPI_Win_shared_query(window_, g * 2, &size, &disp, &ptr));
      host_[g] = reinterpret_cast<char*>(
          (reinterpret_cast<uintptr_t>(ptr) + page - 1) / page * page);
    }
    if (leader_) {
      unsigned long mask = 1UL << group_;
      if (syscall(SYS_mbind, host_[group_], bytes, MPOL_BIND, &mask,
                  8 * sizeof(mask), 0))
        throw std::runtime_error("host gather NUMA bind failed");
      std::memset(host_[group_], 0, bytes);
    }
    check(MPI_Barrier(localComm));
    cudaCheck(__LINE__, cudaSetDevice(local_));
    for (auto& s : state_) {
      cudaCheck(__LINE__,
                cudaEventCreateWithFlags(&s.up, cudaEventDisableTiming));
      cudaCheck(__LINE__,
                cudaEventCreateWithFlags(&s.down, cudaEventDisableTiming));
    }
    for (int g = 0; g < 2; ++g) {
      cudaCheck(__LINE__, cudaHostRegister(host_[g], bytes,
                                           cudaHostRegisterPortable |
                                               cudaHostRegisterMapped));
      cudaCheck(__LINE__,
                cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_[g]),
                                         host_[g], 0));
    }
    cudaCheck(__LINE__,
              cudaStreamCreateWithFlags(&upload_, cudaStreamNonBlocking));
    cudaCheck(__LINE__,
              cudaStreamCreateWithFlags(&consume_, cudaStreamNonBlocking));
    clearMpiProbeStatus();
  }
  ~Gather() {
    cudaSetDevice(local_);
    for (auto& s : state_) {
      cudaEventDestroy(s.up);
      cudaEventDestroy(s.down);
    }
    cudaStreamDestroy(upload_);
    cudaStreamDestroy(consume_);
    for (auto p : host_)
      if (p) cudaHostUnregister(p);
    if (window_ != MPI_WIN_NULL) MPI_Win_free(&window_);
  }
  void run(void const* input, void* output, size_t shardBytes,
           bool reduceAllInputs = false) {
    size_t pieces = (shardBytes + capacity_ - 1) / capacity_, next = 0,
           retired = 0;
    auto gpuConsume = [&](int index, bool remote) {
      // A one-shot reduction waits for all eight inputs; it has no local-only
      // output pass. Existing upload/send/ACK ordering stays unchanged.
      if (reduceAllInputs && !remote) return;
      cudaCheck(__LINE__, cudaSetDevice(local_));
      Slot& s = state_[index];
      Sources sources{};
      int first = 4 * (remote ? peerNode_ : node_);
      for (int g = 0; g < 2; ++g)
        for (int p = 0; p < 2; ++p) {
          char const* src =
              remote
                  ? mapped_[group_] +
                        (size_t(index) * 6 + 2 + g * 2) * capacity_ +
                        p * s.length
                  : mapped_[g] + size_t(index) * 6 * capacity_ + p * s.length;
          int global = first + 2 * g + p;
          if (global == rank_) src = static_cast<char const*>(input) + s.offset;
          sources.p[global] = reinterpret_cast<uint4 const*>(src);
        }
      unsigned blocks = std::min<size_t>(128, (s.length / 16 + 255) / 256);
      // CUDA-aware MPI can leave a sticky invalid-handle probe status.
      // Every CUDA operation owned here is checked directly; clear only that
      // probe status before testing this launch, preserving all other errors.
      clearMpiProbeStatus();
      if (reduceAllInputs) {
        for (int g = 0; g < 2; ++g)
          for (int p = 0; p < 2; ++p) {
            int global = 4 * node_ + 2 * g + p;
            char const* src =
                mapped_[g] + size_t(index) * 6 * capacity_ + p * s.length;
            if (global == rank_)
              src = static_cast<char const*>(input) + s.offset;
            sources.p[global] = reinterpret_cast<uint4 const*>(src);
          }
        reduceInputs<<<blocks, 256, 0, consume_>>>(
            sources, static_cast<float4*>(output), s.offset / 16,
            s.length / 16);
      } else
        consume<<<dim3(blocks, 4), 256, 0, consume_>>>(
            sources, static_cast<uint4*>(output), shardBytes / 16,
            s.offset / 16, s.length / 16, first);
      cudaCheck(__LINE__, cudaGetLastError());
      if (remote) cudaCheck(__LINE__, cudaEventRecord(s.down, consume_));
    };
    while (retired < pieces) {
      for (int index = 0; index < slots; ++index) {
        Slot& s = state_[index];
        if (!s.active) continue;
        int tag = 20000 + index * 8;
        if (!s.uploaded && done(s.up)) {
          for (int p = 0, j = 0; p < 4; ++p)
            if (p != local_)
              check(MPI_Isend(nullptr, 0, MPI_BYTE, node_ * 4 + p, tag,
                              MPI_COMM_WORLD, &s.readySend[j++]));
          s.uploaded = true;
        }
        if (s.uploaded && !s.local && done(s.readyRecv)) {
          gpuConsume(index, false);
          s.local = true;
          if (leader_) {
            for (int g = 0; g < 2; ++g)
              check(MPI_Isend(packet(group_, index), int(2 * s.length),
                              MPI_BYTE, peerNode_ * 4 + 2 * g, tag + 1,
                              MPI_COMM_WORLD, &s.netSend[g]));
          }
          s.sent = true;
        }
        if (s.sent && !s.remote &&
            (leader_ ? done(s.netRecv) : done(s.netReady))) {
          if (leader_)
            check(MPI_Isend(nullptr, 0, MPI_BYTE, rank_ + 1, tag + 2,
                            MPI_COMM_WORLD, &s.netReady));
          gpuConsume(index, true);
          s.remote = true;
        }
        // Leaders acknowledge only after network sends complete. Thus their
        // sibling cannot reuse the shared source while an HCA still reads it.
        if (s.remote && !s.acked && done(s.down) && done(s.netSend)) {
          for (int p = 0, j = 0; p < 4; ++p)
            if (p != local_)
              check(MPI_Isend(nullptr, 0, MPI_BYTE, node_ * 4 + p, tag + 3,
                              MPI_COMM_WORLD, &s.ackSend[j++]));
          s.acked = true;
        }
        if (s.acked && done(s.ackRecv) && done(s.ackSend) &&
            done(s.readySend) && done(s.netReady)) {
          s.active = false;
          ++retired;
        }
      }
      if (next < pieces && !state_[next % slots].active) {
        int index = next % slots, tag = 20000 + index * 8;
        Slot& s = state_[index];
        s.active = true;
        s.uploaded = s.local = s.sent = s.remote = s.acked = false;
        s.offset = next * capacity_;
        s.length = std::min(capacity_, shardBytes - s.offset);
        for (int p = 0, j = 0; p < 4; ++p)
          if (p != local_) {
            check(MPI_Irecv(nullptr, 0, MPI_BYTE, node_ * 4 + p, tag,
                            MPI_COMM_WORLD, &s.readyRecv[j]));
            check(MPI_Irecv(nullptr, 0, MPI_BYTE, node_ * 4 + p, tag + 3,
                            MPI_COMM_WORLD, &s.ackRecv[j++]));
          }
        if (leader_) {
          for (int g = 0; g < 2; ++g)
            check(MPI_Irecv(packet(group_, index, g + 1), int(2 * s.length),
                            MPI_BYTE, peerNode_ * 4 + 2 * g, tag + 1,
                            MPI_COMM_WORLD, &s.netRecv[g]));
        } else
          check(MPI_Irecv(nullptr, 0, MPI_BYTE, rank_ - 1, tag + 2,
                          MPI_COMM_WORLD, &s.netReady));
        cudaCheck(__LINE__, cudaSetDevice(local_));
        cudaCheck(__LINE__, cudaMemcpyAsync(
                                packet(group_, index) + (local_ & 1) * s.length,
                                static_cast<char const*>(input) + s.offset,
                                s.length, cudaMemcpyDeviceToHost, upload_));
        cudaCheck(__LINE__, cudaEventRecord(s.up, upload_));
        ++next;
      }
      int pending;
      check(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &pending,
                       MPI_STATUS_IGNORE));
    }
    clearMpiProbeStatus();
  }
  Gather(Gather const&) = delete;
  Gather& operator=(Gather const&) = delete;
};
}  // namespace host_gather
}  // namespace uccl_lite

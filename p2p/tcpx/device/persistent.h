#pragma once
#include "unpack_descriptor.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>

namespace tcpx {
namespace device {

#define cudaCheckErrors(msg)                                  \
  do {                                                        \
    cudaError_t __err = cudaGetLastError();                   \
    if (__err != cudaSuccess) {                               \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
              cudaGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n");             \
      exit(1);                                                \
    }                                                         \
  } while (0)

// kernel config
constexpr int MAX_INFLIGHT_PER_CHANNEL = 16;
static constexpr int kMaxIovs = MAX_UNPACK_DESCRIPTORS;
static constexpr uint64_t kAbortTailValue = (uint64_t)-2;

struct PersistentKernelConfig {
  uint32_t numThBlocks = 2;  // nChannels
  uint32_t numThPerBlock = 2048;
  uint32_t fifoCap = MAX_INFLIGHT_PER_CHANNEL;
  uint32_t smem_size = 0;  // default is 0
  cudaStream_t stream = nullptr;
};

// IO Vector
struct alignas(8) Iov {
  void* srcs[kMaxIovs];
  void* dsts[kMaxIovs];
  int lens[kMaxIovs];
  int iov_n;
};

struct IovFifo {
  uint64_t head;  // GPU writes finished index
  uint64_t tail;  // CPU posts working index
  struct Iov* iovs;
  int capacity;
};

class IovMultiFifo {
  int num_fifo_;
  int fifo_cap_;
  volatile struct IovFifo** fifo_vec_;  // size: num_fifo
  uint64_t* fifo_slot_idx_;  // size: num_fifo; monotonic increasing index

 public:
  IovMultiFifo(int num_fifo, int fifo_cap)
      : num_fifo_(num_fifo), fifo_cap_(fifo_cap) {
    fifo_vec_ = new IovFifo volatile*[num_fifo_];
    fifo_slot_idx_ = new uint64_t[num_fifo_];
    for (int i = 0; i < num_fifo_; i++) {
      IovFifo* fifo_ptr = nullptr;
      cudaHostAlloc(&fifo_ptr, sizeof(IovFifo), cudaHostAllocMapped);
      cudaCheckErrors("cudaHostAlloc IovFifo failed");

      Iov* iovs_ptr = nullptr;
      cudaHostAlloc(&iovs_ptr, sizeof(Iov) * fifo_cap_, cudaHostAllocMapped);
      cudaCheckErrors("cudaHostAlloc Iov array failed");

      fifo_ptr->iovs = iovs_ptr;
      fifo_ptr->capacity = fifo_cap_;
      fifo_ptr->head = static_cast<uint64_t>(-1);
      fifo_ptr->tail = static_cast<uint64_t>(-1);

      for (int j = 0; j < fifo_cap_; ++j) {
        fifo_ptr->iovs[j].iov_n = -1;
      }

      fifo_vec_[i] = fifo_ptr;
      fifo_slot_idx_[i] = 0;
    }
    __sync_synchronize();
  }

  ~IovMultiFifo() {
    if (fifo_vec_) {
      for (int i = 0; i < num_fifo_; ++i) {
        if (fifo_vec_[i]) {
          if (fifo_vec_[i]->iovs) {
            cudaFreeHost((void*)fifo_vec_[i]->iovs);
          }
          cudaFreeHost((void*)fifo_vec_[i]);
        }
      }
      delete[] fifo_vec_;
    }

    if (fifo_slot_idx_) {
      delete[] fifo_slot_idx_;
    }
  }

  IovFifo** get_fifo_vec();
  std::tuple<uint64_t, volatile struct Iov*> reserve_fifo_slot(int fifo_idx);
  void dispatch_task(int fifo_idx);
  bool check_completion(int fifo_idx, uint64_t slot_idx);
  void abort(int fifo_idx);
};

using KernelFuncType = void (*)(IovFifo**, uint64_t);
class PersistentKernel {
 public:
  explicit PersistentKernel(PersistentKernelConfig const& config);
  ~PersistentKernel();

  bool launch(KernelFuncType kernelFunc);
  uint64_t submit(int fifo_id, Iov const& iov);
  bool is_done(int fifo_idx, uint64_t slot_idx);
  void stop();

  PersistentKernelConfig cfg_;
  IovMultiFifo* fifo_;
  cudaStream_t stream_;
  bool launched_ = false;
};

struct DescriptorInfo {
  int channel_id;
  std::vector<uint64_t> slot_ids;
};

class UnpackerPersistentKernel : public PersistentKernel {
 public:
  using PersistentKernel::PersistentKernel;

  bool launch();
  uint64_t submitDescriptors(int channel_id,
                             tcpx::rx::UnpackDescriptorBlock const& desc_block);
  bool is_done_block(uint64_t desc_id);

 private:
  mutable std::mutex desc_mu_;
  std::unordered_map<uint64_t, DescriptorInfo> desc_id_to_info_;
  std::atomic<uint64_t> next_desc_id_{0};
};

}  // namespace device
}  // namespace tcpx
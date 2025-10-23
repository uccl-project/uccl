#include "persistent.h"
#include <thread>
#include <assert.h>
#include <cuda_pipeline.h>

namespace tcpx {
namespace device {

#define N_THREAD_PER_IOV 8

/*
IovMultiFifo
*/
IovFifo** IovMultiFifo::get_fifo_vec() { return (struct IovFifo**)fifo_vec_; }

std::tuple<uint64_t, volatile struct Iov*> IovMultiFifo::reserve_fifo_slot(
    int fifo_idx) {
  auto slot_idx = fifo_slot_idx_[fifo_idx]++;
  auto fifo = fifo_vec_[fifo_idx];
  auto reserved_iov = fifo->iovs + slot_idx % fifo->capacity;
  return {slot_idx, reserved_iov};
}

void IovMultiFifo::dispatch_task(int fifo_idx) {
  auto fifo = fifo_vec_[fifo_idx];
  auto slot_idx = fifo_slot_idx_[fifo_idx] - 1;
  fifo->tail = slot_idx;
  __sync_synchronize();
}

bool IovMultiFifo::check_completion(int fifo_idx, uint64_t slot_idx) {
  assert(slot_idx < fifo_slot_idx_[fifo_idx]);
  auto fifo = fifo_vec_[fifo_idx];
  return (int64_t)fifo->head >= (int64_t)slot_idx;
}

void IovMultiFifo::abort(int fifo_idx) {
  auto fifo = fifo_vec_[fifo_idx];
  fifo->tail = kAbortTailValue;
  __sync_synchronize();
}

/*
PersistentKernel
*/
PersistentKernel::PersistentKernel(PersistentKernelConfig const& config)
    : cfg_(config), fifo_(nullptr), stream_(nullptr), launched_(false) {
  // create stream if user didn't pass one
  if (cfg_.stream == nullptr) {
    cudaStreamCreate(&stream_);
    cudaCheckErrors("cudaStreamCreate failed");
  } else {
    stream_ = cfg_.stream;
  }

  // create the underlying FIFO manager with numThBlocks FIFOs
  fifo_ = new IovMultiFifo(cfg_.numThBlocks, cfg_.fifoCap);
}

PersistentKernel::~PersistentKernel() {
  if (launched_) stop();  // ensure kernel aborted
  if (cfg_.stream == nullptr && stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    cudaCheckErrors("cudaStreamDestroy failed");
  }
  delete fifo_;
}

bool PersistentKernel::launch(KernelFuncType kernelFunc) {
  if (launched_) return true;

  kernelFunc<<<cfg_.numThBlocks, cfg_.numThPerBlock, cfg_.smem_size, stream_>>>(
      fifo_->get_fifo_vec(), cfg_.fifoCap);
  cudaCheckErrors("kernel launch failed");

  launched_ = true;
  return true;
}

uint64_t PersistentKernel::submit(int fifo_idx, Iov const& iov) {
  auto [slot_idx, gpu_iov] = fifo_->reserve_fifo_slot(fifo_idx);

  int copy_n = iov.iov_n;
  if (copy_n > (int)kMaxIovs) copy_n = kMaxIovs;  // safety clamp

  gpu_iov->iov_n = copy_n;

  // copy arrays
  for (int j = 0; j < copy_n; ++j) {
    gpu_iov->srcs[j] = iov.srcs[j];
    gpu_iov->dsts[j] = iov.dsts[j];
    gpu_iov->lens[j] = iov.lens[j];
  }

  // publish the work to GPU by updating tail
  fifo_->dispatch_task(fifo_idx);
  return slot_idx;
}

bool PersistentKernel::is_done(int fifo_idx, uint64_t slot_idx) {
  return fifo_->check_completion(fifo_idx, slot_idx);
}

void PersistentKernel::stop() {
  for (uint32_t i = 0; i < cfg_.numThBlocks; i++) {
    fifo_->abort(i);
  }
  cudaStreamSynchronize(stream_);
}

/*
UnpackerPersisitentKernel
*/

// -------------------- device helpers --------------------
__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
}

__device__ __forceinline__ void fence_acq_rel_sys() {
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.sys;" ::: "memory");
#else
  asm volatile("membar.sys;" ::: "memory");
#endif
}

__device__ __forceinline__ void st_relaxed_sys(uint64_t* ptr, uint64_t val) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(ptr), "l"(val)
               : "memory");
#else
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(ptr), "l"(val)
               : "memory");
#endif
}

template <typename X, typename Y, typename Z = decltype(X() + Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x + y - 1) / y;
}

// -------------------- memory copy kernel --------------------
__device__ __forceinline__ void copyGlobalMemory(void* dst, void* src,
                                                 int len) {
  uintptr_t src_addr = (uintptr_t)src;
  uintptr_t dst_addr = (uintptr_t)dst;
  int i = 0;

  for (; i + 8 <= len; i += 8) {
    *(uint64_t*)(dst_addr + i) = *(uint64_t*)(src_addr + i);
  }
  if (i < len) {
    for (; i < len; i++) {
      *(uint8_t*)(dst_addr + i) = *(uint8_t*)(src_addr + i);
    }
  }
}

template <typename T, int kCpAsycDepth>
__device__ __forceinline__ void kernelScatteredMemcpy(struct Iov* iov) {
  extern __shared__ uint8_t allocated_smem[];
  auto smem = reinterpret_cast<T*>(allocated_smem);

  constexpr int nthreads_per_iov = N_THREAD_PER_IOV;

  // Each SM is an independent worker.
  int nthreads = blockDim.x;
  int tid = threadIdx.x;
  int iov_n = iov->iov_n;

  // Speedup tricks for 1 iov copy; could be deleted for generality.
  if (iov_n == 1) {
    void** src_addrs = iov->srcs;
    void** dst_addrs = iov->dsts;
    int* iov_lens = iov->lens;

    // Yang: Doing the scattered memcpy here? directly copy to dst ptrs.
    char* src = (char*)src_addrs[0];
    char* dst = (char*)dst_addrs[0];
    int iov_len = iov_lens[0];

    // Make it t-byte aligned to avoid GPU SEGV.
    int num_packs = iov_len / 8;
    int len_per_th = divUp(num_packs, nthreads) * 8;
    int start = len_per_th * tid;
    int end = min(start + len_per_th, iov_len);
    int len = end - start;
    if (len > 0) copyGlobalMemory(dst + start, src + start, len);
    return;
  }

  // Ignoring some non-rounded threads
  if (tid > nthreads_per_iov * iov_n) return;

  int iov_n_per_iter = nthreads / nthreads_per_iov;
  int start_iov = tid / nthreads_per_iov;

  for (int i = start_iov; i < iov_n; i += iov_n_per_iter) {
    // Map each thread to a iov copy.
    int iov_idx = i;
    // Compute local tid within the th group assigned to this iov copy.
    int local_tid = tid % nthreads_per_iov;

    // Retrieve parameters for this copy.
    char* src_ptr = (char*)iov->srcs[iov_idx];
    char* dst_ptr = (char*)iov->dsts[iov_idx];
    int iov_len = iov->lens[iov_idx];
    if (iov_len == 0) return;

    // Copy t-byte chunks first (if possible)
    int num_full = iov_len / sizeof(T);
    T* src_T = (T*)src_ptr;
    T* dst_T = (T*)dst_ptr;

    int write = 0;  // write buffer for ring buffer
    int tail = 0;   // read buffer for ring buffer
    int issued = 0;
    int last_j = 0;

    for (int j = local_tid; j < num_full; j += nthreads_per_iov) {
      void* smem_ptr = &smem[tid + nthreads * write];
      void const* gmem_ptr = &src_T[j];

      // Ref: https://github.com/NVIDIA/libcudacxx/pull/220
      // already include `cache_hint` to avoid l1 cache.
      // potentially use inline asm to avoid l2 cache.
      __pipeline_memcpy_async(smem_ptr, gmem_ptr, sizeof(T));
      __pipeline_commit();

      ++issued;
      last_j = j;

      if (issued == kCpAsycDepth) {
        __pipeline_wait_prior(kCpAsycDepth - 1);
        int dst_pos = j - (kCpAsycDepth - 1) * nthreads_per_iov;
        dst_T[dst_pos] = smem[tid + nthreads * tail];
        tail = (tail + 1) % kCpAsycDepth;
        --issued;
      }

      write = (write + 1) % kCpAsycDepth;
    }

    while (issued) {
      --issued;
      __pipeline_wait_prior(issued);
      int dst_pos = last_j - issued * nthreads_per_iov;
      dst_T[dst_pos] = smem[tid + nthreads * tail];
      tail = (tail + 1) % kCpAsycDepth;
    }

    // Let only one thread in the copy group (e.g. local_tid == 0) copy
    // the tail.
    // Try to avoid this tail by specifying `zfill` in `__pipeline_memcpy_async`
    // not useful potentially due to the increased offset calculation.
    if (local_tid == 0) {
      // Handle the remaining tail bytes (if any)
      int tail_start = num_full * sizeof(T);
      for (int j = tail_start; j < iov_len; j++) {
        dst_ptr[j] = src_ptr[j];
      }
    }
  }
}

template <typename T, int kCpAsycDepth>
__global__ void persistKernel(struct IovFifo** fifo_vec, uint64_t fifoCap) {
  __shared__ uint64_t cached_tail;
  __shared__ uint64_t abort_flag;
  __shared__ struct Iov* cur_iov;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int global_tid = bid * blockDim.x + tid;

  struct IovFifo* fifo = fifo_vec[bid];

  // This impossible print is necessary to make sure other kernels run.
  if (global_tid == -1) {
    printf("Persist kernel: block %d, thread %d\n", bid, tid);
  }

  // Initing per-threadblock variables
  if (tid == 0) {
    abort_flag = 0;
    cached_tail = (uint64_t)-1;
  }
  __syncthreads();

  // We should avoid all thread loading the global memory at the same, as
  // this will cause severe performance drop. while
  // (ld_volatile(&fifo->abort) == 0) {
  while (true) {
    // Each thread block loads new work from CPU.
    if (tid == 0) {
      uint64_t cur_tail;
      do {
        cur_tail = ld_volatile(&fifo->tail);

        if (cur_tail == kAbortTailValue) {
          // The CPU has posted a abort signal.
          abort_flag = 1;
          break;
        }
      } while ((int64_t)cur_tail < (int64_t)(cached_tail + 1));
      // Processing one iov at a time.
      cur_tail = cached_tail + 1;

      cached_tail = cur_tail;
      cur_iov = fifo->iovs + cached_tail % fifoCap;
    }
    __syncthreads();
    if (abort_flag) return;

    kernelScatteredMemcpy<T, kCpAsycDepth>(cur_iov);

    __syncthreads();

    // Post the finished work to the GPU
    if (tid == 0) {
      fence_acq_rel_sys();
      st_relaxed_sys(&fifo->head, cached_tail);
    }
  }
}

bool UnpackerPersistentKernel::launch() {
  return PersistentKernel::launch(persistKernel<float4, 2>);  // float4: 16bytes
}

uint64_t UnpackerPersistentKernel::submitDescriptors(
    int channel_id, tcpx::rx::UnpackDescriptorBlock const& desc_block) {
  constexpr int nthreads_per_iov = N_THREAD_PER_IOV;
  int const block_dim = cfg_.numThPerBlock;
  int const max_iov_per_block = block_dim / nthreads_per_iov;

  std::vector<uint64_t> slot_ids;
  slot_ids.reserve((desc_block.count + max_iov_per_block - 1) /
                   max_iov_per_block);

  int const total_desc = desc_block.count;

  for (int offset = 0; offset < total_desc; offset += max_iov_per_block) {
    int const batch_count = std::min(max_iov_per_block, total_desc - offset);

    Iov iov{};
    iov.iov_n = batch_count;

    for (int j = 0; j < batch_count; ++j) {
      auto const& d = desc_block.descriptors[offset + j];
      iov.srcs[j] = static_cast<void*>(
          static_cast<char*>(desc_block.bounce_buffer) + d.src_off);
      iov.dsts[j] = static_cast<void*>(
          static_cast<char*>(desc_block.dst_buffer) + d.dst_off);
      iov.lens[j] = d.len;
    }

    uint64_t slot_id = submit(channel_id, iov);
    slot_ids.push_back(slot_id);
  }

  uint64_t desc_id = next_desc_id_.fetch_add(1, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> lock(desc_mu_);
    desc_id_to_info_.emplace(desc_id,
                             DescriptorInfo{channel_id, std::move(slot_ids)});
  }

  return desc_id;
}

bool UnpackerPersistentKernel::is_done_block(uint64_t desc_id) {
  int channel_id = -1;
  std::vector<uint64_t> slot_ids;

  {
    std::lock_guard<std::mutex> lock(desc_mu_);
    auto it = desc_id_to_info_.find(desc_id);
    if (it == desc_id_to_info_.end()) {
      return false;
    }
    channel_id = it->second.channel_id;
    slot_ids = it->second.slot_ids;
  }

  while (true) {
    bool all_done = true;
    for (uint64_t slot_id : slot_ids) {
      if (!PersistentKernel::is_done(channel_id, slot_id)) {
        all_done = false;
        break;
      }
    }

    if (all_done) {
      return true;
    }

    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

}  // namespace device
}  // namespace tcpx
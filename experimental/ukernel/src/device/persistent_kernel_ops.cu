#include "fifo/d2c_fifo_device.hpp"
#include "persistent_kernel_ops.h"
// #include "operators/operator.cuh"

namespace UKernel {
namespace Device {

namespace {

constexpr uint32_t kTmaSharedBytes = 1024;

__device__ __forceinline__ uint32_t dtype_bytes(DataType dtype) {
  switch (dtype) {
    case DataType::Fp16:
      return 2;
    case DataType::Fp32:
      return 4;
    case DataType::Fp8:
    default:
      return 1;
  }
}

#if !defined(__HIP_PLATFORM_AMD__) && UKERNEL_ENABLE_TMA && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ void tma_init_barrier(uint64_t* barrier) {
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(barrier_ptr), "r"(1)
               : "memory");
}

__device__ __forceinline__ void tma_expect_bytes(uint64_t* barrier,
                                                 uint32_t bytes) {
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(barrier_ptr), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void tma_wait_load(uint64_t* barrier, int phase) {
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "TMA_WAIT_%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra.uni TMA_DONE_%=;\n"
      "bra.uni TMA_WAIT_%=;\n"
      "TMA_DONE_%=:\n"
      "}\n"
      :
      : "r"(barrier_ptr), "r"(phase)
      : "memory");
}

__device__ __forceinline__ void tma_store_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N = 0>
__device__ __forceinline__ void tma_store_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;\n" : : "n"(N) : "memory");
}

template <int N = 0>
__device__ __forceinline__ void tma_store_read_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" : : "n"(N) : "memory");
}

__device__ __forceinline__ void tma_load_vector_chunk(uint64_t desc_ptr,
                                                      void* smem_dst,
                                                      uint64_t* barrier,
                                                      uint32_t elem_offset) {
  uint32_t shared_dst =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%3, %4, %5, %6}], [%2];\n"
      :
      : "r"(shared_dst), "l"(desc_ptr), "r"(barrier_ptr), "r"(elem_offset),
        "r"(0), "r"(0), "r"(0)
      : "memory");
}

__device__ __forceinline__ void tma_store_vector_chunk(uint64_t desc_ptr,
                                                       void const* smem_src,
                                                       uint32_t elem_offset) {
  uint32_t shared_src =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_src));
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group "
      "[%0, {%2, %3, %4, %5}], [%1];\n"
      :
      : "l"(desc_ptr), "r"(shared_src), "r"(elem_offset), "r"(0), "r"(0),
        "r"(0)
      : "memory");
}
#endif

}  // namespace

template <typename T>
__device__ __forceinline__ T apply_red(ReduceType op, T a, T b) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Max) return a > b ? a : b;
  return a;  // None or unknown
}

template <>
__device__ __forceinline__ __half apply_red<__half>(ReduceType op, __half a,
                                                    __half b) {
  float af = __half2float(a);
  float bf = __half2float(b);
  float rf;
  if (op == ReduceType::Sum)
    rf = af + bf;
  else if (op == ReduceType::Max)
    rf = (af > bf ? af : bf);
  else
    rf = af;
  return __float2half(rf);
}

__device__ void run_copy_register(CollArgs const& a) {
  auto* dst = reinterpret_cast<char*>(a.dst);
  auto* src = reinterpret_cast<char const*>(a.src);

  const uint64_t total = (uint64_t)a.bytes;

  const uint64_t tid = (uint64_t)threadIdx.x;
  const uint64_t nthread = (uint64_t)blockDim.x;

  const uint64_t chunk_size = (total + nthread - 1) / nthread;

  const uint64_t start = tid * chunk_size;
  const uint64_t end = min(start + chunk_size, total);

  for (uint64_t i = start; i < end; ++i) {
    dst[i] = src[i];
  }
}

__device__ void run_copy_tma(CollArgs const& a, DataType dtype,
                             void* tma_scratch, uint64_t* tma_barrier) {
#if !defined(__HIP_PLATFORM_AMD__) && UKERNEL_ENABLE_TMA && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 900)
  if (tma_scratch == nullptr || tma_barrier == nullptr ||
      a.src_tma_desc == nullptr || a.dst_tma_desc == nullptr ||
      a.tma_chunk_elements == 0) {
    run_copy_register(a);
    return;
  }

  uint32_t elem_bytes = dtype_bytes(dtype);
  if (elem_bytes == 0 || (a.bytes % elem_bytes) != 0) {
    run_copy_register(a);
    return;
  }

  uint64_t total_elements = a.bytes / elem_bytes;
  uint32_t chunk_elements = a.tma_chunk_elements;
  uint32_t chunk_bytes = chunk_elements * elem_bytes;
  if (chunk_bytes == 0 || chunk_bytes > kTmaSharedBytes ||
      (total_elements % chunk_elements) != 0) {
    run_copy_register(a);
    return;
  }

  uint64_t src_desc = reinterpret_cast<uint64_t>(a.src_tma_desc);
  uint64_t dst_desc = reinterpret_cast<uint64_t>(a.dst_tma_desc);

  for (uint64_t elem_offset = 0; elem_offset < total_elements;
       elem_offset += chunk_elements) {
    __syncthreads();
    if (threadIdx.x == 0) {
      tma_init_barrier(tma_barrier);
      tma_expect_bytes(tma_barrier, chunk_bytes);
      tma_load_vector_chunk(src_desc, tma_scratch, tma_barrier,
                            static_cast<uint32_t>(elem_offset));
      tma_wait_load(tma_barrier, 0);
      tma_store_vector_chunk(dst_desc, tma_scratch,
                             static_cast<uint32_t>(elem_offset));
      tma_store_commit_group();
      tma_store_wait_group<0>();
      tma_store_read_wait<0>();
    }
    __syncthreads();
  }
#else
  (void)dtype;
  (void)tma_scratch;
  (void)tma_barrier;
  run_copy_register(a);
#endif
}

__device__ void run_copy(CollArgs const& a, DataType dtype, void* tma_scratch,
                         uint64_t* tma_barrier) {
  switch (a.resolved_path) {
    case TransferPath::TmaOp:
      run_copy_tma(a, dtype, tma_scratch, tma_barrier);
      return;
    case TransferPath::Auto:
    case TransferPath::RegisterOp:
    default:
      run_copy_register(a);
      return;
  }
}

template <typename T>
__device__ void run_reduce_inplace(CollArgs const& a) {
  auto* dst = reinterpret_cast<T*>(a.dst);
  auto* src = reinterpret_cast<T const*>(a.src);

  const uint64_t n = (uint64_t)a.bytes / sizeof(T);

  const uint64_t tid = (uint64_t)threadIdx.x;
  const uint64_t nthread = (uint64_t)blockDim.x;

  const uint64_t chunk_size = (n + nthread - 1) / nthread;
  const uint64_t start = tid * chunk_size;
  const uint64_t end = min(start + chunk_size, n);

  const ReduceType rop = a.redType;

  if (rop == ReduceType::None) return;

  for (uint64_t i = start; i < end; ++i) {
    dst[i] = apply_red<T>(rop, dst[i], src[i]);
  }
}

template __device__ void run_reduce_inplace<float>(CollArgs const&);
template __device__ void run_reduce_inplace<__half>(CollArgs const&);
// more
// template __device__ void run_reduce_t<double>(const CollArgs&);
// template __device__ void run_reduce_t<half>(const CollArgs&);

// TODO: using sm id to assign task
__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                     mscclpp::FifoDeviceHandle* d2c_fifo,
                                     CollArgs* d_coll, GemmArgs* d_gemm,
                                     bool* should_stop) {
  extern __shared__ char smem[];
  __align__(128) __shared__ unsigned char tma_scratch[kTmaSharedBytes];
  __align__(8) __shared__ uint64_t tma_barrier;
  (void)d2c_fifo;
  (void)d_gemm;
  (void)smem;

  const uint32_t bid = blockIdx.x;
  auto& fifo = c2d_fifos[bid];  // block => fifo

  while (true) {
    if (should_stop && *should_stop) break;

    Task* task = fifo.poll();
    if (task == nullptr) continue;

    /*
    If we later re-introduce inter-SM task fanout, it should live on top of the
    standalone SmFifo primitive rather than as implicit runtime state here.
    */

    __syncthreads();

    const TaskType ttype = (TaskType)task->type_u8();
    const DataType dtype = (DataType)task->dtype_u8();
    const uint32_t idx = task->args_index();
    const uint32_t block_id = task->block_index();
    const CollArgs a = d_coll[idx];

    // if (threadIdx.x == 0) {
    //   printf("cur block=%u : task block_id=%u args_id=%u type=%d dtype=%d
    //   red=%d bytes=%llu path=%u step=%u chunk=%u\n", bid, block_id, idx,
    //          int(ttype), int(dtype), int(a.redType),
    //          static_cast<unsigned long long>(a.bytes),
    //          static_cast<unsigned>(a.resolved_path), a.step_id, a.chunk_id);
    // }

    switch (ttype) {
      case TaskType::CollCopy: {
        run_copy(a, dtype, tma_scratch, &tma_barrier);
        break;
      }
      case TaskType::CollReduce: {
        if (dtype == DataType::Fp32) {
          run_reduce_inplace<float>(a);
        } else if (dtype == DataType::Fp16) {
          run_reduce_inplace<__half>(a);
        } else {
          // Fp8 TODO:
        }
        break;
      }
      case TaskType::TkGemm: {
        // skip TK operator now.
        // GemmArgs const& ga = d_gemm[idx];
        // TkMatmulGlobals* g = (TkMatmulGlobals*)ga.globals;
        // run_tk_gemm(*g, ga.num_tile_rows, ga.num_tile_cols, smem);
        break;
      }
      default:
        break;
    }

    __threadfence();

    if (threadIdx.x == 0) {
      // Push completion trigger to host
      // if (d2c_fifo) {
      //   mscclpp::ProxyTrigger trig(ttype, dtype, block_id, idx);
      //   d2c_fifo->push(trig);
      // }
      // Pop task from C2D fifo
      fifo.pop();
    }

    __syncthreads();
  }
}

}  // namespace Device
}  // namespace UKernel

#include "bench_kernel.cuh"
#include "bench_utils.hpp"
#include "common.hpp"
#include "d2h_queue_device.cuh"
#include "ep_config.hpp"
#include "ep_configs.cuh"
#include "ep_event.hpp"
#include "ep_proxy_registry.hpp"
#include "ep_runtime.cuh"
#include "ep_util.hpp"
#include "internode.cuh"
#include "internode_ll.cuh"
#include "intranode.cuh"
#include "layout.hpp"
#include "rdma.hpp"
#include "ring_buffer.cuh"
#include "uccl_bench.hpp"
#include "uccl_proxy.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

namespace uccl {
std::unordered_map<int, std::vector<nb::object>> g_proxies_by_dev;

std::unordered_map<int, std::vector<nb::object>>& proxies_by_dev() {
  return g_proxies_by_dev;
}
}  // namespace uccl

#define NUM_MAX_LOCAL_EXPERTS 1024

namespace nb = nanobind;

static std::mutex g_proxies_mu;

struct EventOverlap {};
struct Ctx {
  long num_tokens{0};
  long hidden{0};
};
static std::atomic<long> g_next{1};
static std::mutex g_mu;
static std::unordered_map<long, Ctx> g_ctx;

enum DTypeCode : int {
  kUInt8 = 0,
  kInt8 = 1,
  kInt16 = 2,
  kInt32 = 3,
  kInt64 = 4,
  kFloat16 = 5,
  kBFloat16 = 6,
  kFloat32 = 7,
  kFloat64 = 8,
  kBool = 9,
  kFloat8E4M3 = 10,
};

static cudaDataType_t cuda_dtype_from_code(int dtype) {
  switch (dtype) {
    case kUInt8:
      return CUDA_R_8U;
    case kInt8:
      return CUDA_R_8I;
    case kInt16:
      return CUDA_R_16I;
    case kInt32:
      return CUDA_R_32I;
    case kInt64:
      return CUDA_R_64I;
    case kFloat16:
      return CUDA_R_16F;
    case kBFloat16:
      return CUDA_R_16BF;
    case kFloat32:
      return CUDA_R_32F;
    case kFloat64:
      return CUDA_R_64F;
    case kBool:
      return CUDA_R_8I;
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
    case kFloat8E4M3:
      return CUDA_R_8F_E4M3;
#endif
    default:
      throw std::invalid_argument("Unsupported dtype code for CUDA kernel");
  }
}

static int64_t dtype_size_bytes(int dtype) {
  switch (dtype) {
    case kUInt8:
    case kInt8:
    case kBool:
    case kFloat8E4M3:
      return 1;
    case kInt16:
    case kFloat16:
    case kBFloat16:
      return 2;
    case kInt32:
    case kFloat32:
      return 4;
    case kInt64:
    case kFloat64:
      return 8;
    default:
      throw std::invalid_argument("Unsupported dtype code size");
  }
}

namespace {

enum DLDeviceType : int32_t {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLROCM = 10,
};

enum DLDataTypeCode : uint8_t {
  kDLUInt = 1,
};

struct DLDevice {
  DLDeviceType device_type;
  int32_t device_id;
};

struct DLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensor {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};

struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensor* self);
};

struct RdmaBufferDlpack {
  DLManagedTensor managed{};
  int64_t shape[1]{};
  void* ptr{nullptr};
  std::size_t bytes{0};
  int device_index{0};
  bool is_host_allocated{false};
};

void free_rdma_raw_buffer(void* ptr, int device_index, bool is_host_allocated) {
  if (!ptr) return;
  if (is_host_allocated) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
  CUDA_CHECK(cudaSetDevice(device_index));
  CUDA_CHECK(cudaFree(ptr));
}

void free_rdma_raw_buffer_nothrow(void* ptr, int device_index,
                                  bool is_host_allocated) {
  if (!ptr) return;
  if (is_host_allocated) {
    (void)cudaFreeHost(ptr);
    return;
  }
  (void)cudaSetDevice(device_index);
  (void)cudaFree(ptr);
}

void rdma_buffer_dlpack_deleter(DLManagedTensor* managed) {
  if (!managed) return;
  auto* ctx = static_cast<RdmaBufferDlpack*>(managed->manager_ctx);
  if (!ctx) return;
  free_rdma_raw_buffer_nothrow(ctx->ptr, ctx->device_index,
                               ctx->is_host_allocated);
  delete ctx;
}

void rdma_buffer_capsule_destructor(PyObject* capsule) {
  if (!PyCapsule_IsValid(capsule, "dltensor")) return;
  auto* managed =
      static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
  if (managed && managed->deleter) managed->deleter(managed);
}

nb::object make_rdma_buffer_dlpack_capsule(void* ptr, std::size_t bytes,
                                           int device_index,
                                           bool is_host_allocated) {
  auto* ctx = new RdmaBufferDlpack();
  ctx->ptr = ptr;
  ctx->bytes = bytes;
  ctx->device_index = device_index;
  ctx->is_host_allocated = is_host_allocated;
  ctx->shape[0] = static_cast<int64_t>(bytes);
  ctx->managed.manager_ctx = ctx;
  ctx->managed.deleter = rdma_buffer_dlpack_deleter;
  ctx->managed.dl_tensor.data = ptr;
  ctx->managed.dl_tensor.device = {
    is_host_allocated ? kDLCPU
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
                      : kDLROCM,
#else
                      : kDLCUDA,
#endif
    is_host_allocated ? 0 : device_index,
  };
  ctx->managed.dl_tensor.ndim = 1;
  ctx->managed.dl_tensor.dtype = {kDLUInt, 8, 1};
  ctx->managed.dl_tensor.shape = ctx->shape;
  ctx->managed.dl_tensor.strides = nullptr;
  ctx->managed.dl_tensor.byte_offset = 0;

  PyObject* capsule =
      PyCapsule_New(&ctx->managed, "dltensor", rdma_buffer_capsule_destructor);
  if (!capsule) {
    rdma_buffer_dlpack_deleter(&ctx->managed);
    throw nb::python_error();
  }
  return nb::steal<nb::object>(capsule);
}

std::tuple<nb::object, bool> allocate_rdma_buffer_dlpack(
    std::size_t num_rdma_bytes, int device_index) {
  std::size_t const alloc_bytes = std::max<std::size_t>(num_rdma_bytes, 1);
  bool is_host_allocated = false;
  void* ptr = nullptr;

  CUDA_CHECK(cudaSetDevice(device_index));

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  CUDA_CHECK(hipExtMallocWithFlags(&ptr, alloc_bytes, hipDeviceMallocUncached));
  CUDA_CHECK(cudaMemset(ptr, 0, alloc_bytes));
#elif defined(EFA) || defined(USE_DMABUF)
  CUDA_CHECK(cudaMalloc(&ptr, alloc_bytes));
  CUDA_CHECK(cudaMemset(ptr, 0, alloc_bytes));
#else
  bool const use_host_alloc =
      num_rdma_bytes > 0 &&
      !can_register_gpu_memory_for_rdma(device_index, num_rdma_bytes);
  if (!use_host_alloc) {
    CUDA_CHECK(cudaMalloc(&ptr, alloc_bytes));
    CUDA_CHECK(cudaMemset(ptr, 0, alloc_bytes));
  } else {
    CUDA_CHECK(cudaMallocHost(&ptr, alloc_bytes));
    std::memset(ptr, 0, alloc_bytes);
    is_host_allocated = true;
  }
#endif

  try {
    return {make_rdma_buffer_dlpack_capsule(ptr, alloc_bytes, device_index,
                                            is_host_allocated),
            is_host_allocated};
  } catch (...) {
    free_rdma_raw_buffer(ptr, device_index, is_host_allocated);
    throw;
  }
}

}  // namespace

static std::vector<uint64_t> collect_d2h_channel_addrs_for_device(
    int device_index) {
  std::lock_guard<std::mutex> lk(g_proxies_mu);
  auto it = uccl::g_proxies_by_dev.find(device_index);
  EP_HOST_ASSERT(it != uccl::g_proxies_by_dev.end() && !it->second.empty());

  std::vector<uint64_t> all_addrs;
  // Collect all ring buffer addresses from all proxies
  for (auto& proxy : it->second) {
    // Each proxy now manages multiple ring buffers
    auto proxy_addrs =
        nb::cast<std::vector<uint64_t>>(proxy.attr("get_d2h_channel_addrs")());
    all_addrs.insert(all_addrs.end(), proxy_addrs.begin(), proxy_addrs.end());
  }
  return all_addrs;
}

bool is_sm90_compiled() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // disable sm90 features for HIP to avoid tma stuffs, but AMD supports fp8
  // features
  return true;
#elif !defined(DISABLE_SM90_FEATURES)
  return true;
#else
  return false;
#endif
}

class Buffer {
 public:
  Buffer(int rank, int num_ranks, long num_nvl_bytes, long num_rdma_bytes,
         bool low_latency_mode, bool explicitly_destroy, int num_local_ranks)
      : rank(rank),
        num_ranks(num_ranks),
        num_nvl_bytes(num_nvl_bytes),
        num_rdma_bytes(num_rdma_bytes),
        low_latency_mode(low_latency_mode),
        explicitly_destroy(explicitly_destroy) {
    if (num_local_ranks == -1) num_local_ranks = get_num_max_nvl_peers();
    max_nvl_peers = num_local_ranks;
    {
      cudaGetDevice(&device_index);
      {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        auto it = uccl::g_proxies_by_dev.find(device_index);
        if (it == uccl::g_proxies_by_dev.end() || it->second.empty()) {
          throw std::runtime_error(
              "ep.Buffer: no UcclProxy registered for device " +
              std::to_string(device_index) +
              ". Call uccl.ep.register_proxy(device_index, proxies) "
              "first.");
        }
      }

      {
        CUDA_CHECK(cudaSetDevice(device_index));
        int least_priority = 0, greatest_priority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority,
                                                    &greatest_priority));
        CUDA_CHECK(cudaStreamCreateWithPriority(
            &comm_stream, cudaStreamNonBlocking, greatest_priority));
        auto host_addrs = collect_d2h_channel_addrs_for_device(device_index);
        num_d2h_channel_addrs = static_cast<int>(host_addrs.size());
        if (num_d2h_channel_addrs > 0) {
          CUDA_CHECK(cudaMallocManaged(
              &d_handle_objs, num_d2h_channel_addrs * sizeof(d2hq::D2HHandle)));

          CUDA_CHECK(cudaMallocManaged(
              &d_handles, num_d2h_channel_addrs * sizeof(uint64_t)));

          for (int i = 0; i < num_d2h_channel_addrs; ++i) {
#ifndef USE_MSCCLPP_FIFO_BACKEND
            void* host_ptr = reinterpret_cast<void*>(host_addrs[i]);
            void* dev_ptr = nullptr;
#ifndef USE_GRACE_HOPPER
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&dev_ptr), host_ptr, 0));
#else
            dev_ptr = host_ptr;
#endif
            d_handle_objs[i].init_from_dev_ptr(dev_ptr);
            d_handles[i] = reinterpret_cast<uint64_t>(&d_handle_objs[i]);
#else
            auto* fifo = reinterpret_cast<mscclpp::Fifo*>(host_addrs[i]);
            mscclpp::FifoDeviceHandle h = fifo->deviceHandle();
            d_handle_objs[i].init_from_host_value(h);
            d_handles[i] = reinterpret_cast<uint64_t>(d_handle_objs + i);
#endif
          }

          // Prefetch so the device immediately sees initialized contents
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__) && \
    CUDA_VERSION >= 12000
          // CUDA 12+: cudaMemPrefetchAsync(ptr, count, cudaMemLocation, flags,
          // stream)
          cudaMemLocation loc;
          loc.type = cudaMemLocationTypeDevice;
          loc.id = device_index;
          CUDA_CHECK(cudaMemPrefetchAsync(
              d_handle_objs, num_d2h_channel_addrs * sizeof(d2hq::D2HHandle),
              loc, 0));
          CUDA_CHECK(cudaMemPrefetchAsync(
              d_handles, num_d2h_channel_addrs * sizeof(uint64_t), loc, 0));
#else
          // CUDA 11.x / HIP: cudaMemPrefetchAsync(ptr, count, dstDevice,
          // stream)
          CUDA_CHECK(cudaMemPrefetchAsync(
              d_handle_objs, num_d2h_channel_addrs * sizeof(d2hq::D2HHandle),
              device_index, 0));
          CUDA_CHECK(cudaMemPrefetchAsync(
              d_handles, num_d2h_channel_addrs * sizeof(uint64_t),
              device_index));
#endif
          CUDA_CHECK(cudaDeviceSynchronize());
        }
        // Allocate device memory for IPC base pointers
        CUDA_CHECK(
            cudaMalloc(&d_ipc_rdma_base_ptrs, max_nvl_peers * sizeof(void*)));
        CUDA_CHECK(
            cudaMemset(d_ipc_rdma_base_ptrs, 0, max_nvl_peers * sizeof(void*)));
      }
    }

    int64_t const barrier_signal_bytes = max_nvl_peers * sizeof(int);
    int64_t const buffer_ptr_bytes = max_nvl_peers * sizeof(void*);
    int64_t const barrier_signal_ptr_bytes = max_nvl_peers * sizeof(int*);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                   (num_nvl_bytes <= std::numeric_limits<int64_t>::max() ||
                    num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                   (low_latency_mode ||
                    num_rdma_bytes <= std::numeric_limits<int64_t>::max()));
#else
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                   (num_nvl_bytes <= std::numeric_limits<int>::max() ||
                    num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                   (low_latency_mode ||
                    num_rdma_bytes <= std::numeric_limits<int>::max()));
#endif
    EP_HOST_ASSERT(
        0 <= rank && rank < num_ranks &&
        (num_ranks <= max_nvl_peers * NUM_MAX_RDMA_PEERS || low_latency_mode));
    EP_HOST_ASSERT(num_ranks < max_nvl_peers ||
                   (num_ranks % max_nvl_peers) == 0);
    // if (num_rdma_bytes > 0)
    //   EP_HOST_ASSERT(num_ranks > max_nvl_peers || low_latency_mode);

    rdma_rank = rank / max_nvl_peers;
    nvl_rank = rank % max_nvl_peers;
    num_rdma_ranks = std::max(1, num_ranks / max_nvl_peers);
    num_nvl_ranks = std::min(num_ranks, max_nvl_peers);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
    num_device_sms = prop.multiProcessorCount;

    if (num_nvl_bytes > 0) {
      size_t total_bytes = static_cast<size_t>(num_nvl_bytes) +
                           static_cast<size_t>(barrier_signal_bytes) +
                           static_cast<size_t>(buffer_ptr_bytes) +
                           static_cast<size_t>(barrier_signal_ptr_bytes);

      // Ensure we're on the correct device before memory allocation and IPC
      // handle creation
      CUDA_CHECK(cudaSetDevice(device_index));
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
      // aggressive atomic will work with malloc with uncached memory
      CUDA_CHECK(hipExtMallocWithFlags(&buffer_ptrs[nvl_rank], total_bytes,
                                       hipDeviceMallocUncached));
#else
      CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], total_bytes));
#endif
      CUDA_CHECK(
          cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));

      buffer_ptrs_gpu = reinterpret_cast<void**>(
          static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes +
          barrier_signal_bytes);

      barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(
          static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);

      barrier_signal_ptrs_gpu = reinterpret_cast<int**>(
          static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes +
          barrier_signal_bytes + buffer_ptr_bytes);

      CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0,
                                 barrier_signal_bytes, comm_stream));
    }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    CUDA_CHECK(hipExtMallocWithFlags(&workspace, NUM_WORKSPACE_BYTES,
                                     hipDeviceMallocUncached));
#else
    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
#endif
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));
    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t),
                              cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&moe_recv_counter_mapped),
        const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter,
                              sizeof(int) * NUM_MAX_LOCAL_EXPERTS,
                              cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&moe_recv_expert_counter_mapped),
        const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
      moe_recv_expert_counter[i] = -1;

    if (num_rdma_ranks > 0) {
      CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int),
                                cudaHostAllocMapped));
      CUDA_CHECK(cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&moe_recv_rdma_counter_mapped),
          const_cast<int*>(moe_recv_rdma_counter), 0));
      *moe_recv_rdma_counter = -1;
    }
  }

  std::optional<EventHandle> get_dispatch_layout(
      std::uintptr_t topk_idx_ptr, int num_tokens, int num_topk,
      int num_experts, std::uintptr_t num_tokens_per_rank_ptr,
      std::uintptr_t num_tokens_per_rdma_rank_ptr,
      std::uintptr_t num_tokens_per_expert_ptr,
      std::uintptr_t is_token_in_rank_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
    EP_HOST_ASSERT(topk_idx_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);
    EP_HOST_ASSERT(is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(num_experts > 0);

    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    // NOTE(zhenhuang12): No runime cost. Python now owns the actual tensor
    // allocation stream before passing raw pointers into C++, so this flag is
    // advisory on the C++ side.
    static_cast<void>(allocate_on_comm_stream);

    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    auto* topk_idx = reinterpret_cast<int64_t*>(topk_idx_ptr);
    auto* num_tokens_per_rank = reinterpret_cast<int*>(num_tokens_per_rank_ptr);
    auto* num_tokens_per_rdma_rank =
        num_tokens_per_rdma_rank_ptr == 0
            ? nullptr
            : reinterpret_cast<int*>(num_tokens_per_rdma_rank_ptr);
    auto* num_tokens_per_expert =
        reinterpret_cast<int*>(num_tokens_per_expert_ptr);
    auto* is_token_in_rank = reinterpret_cast<bool*>(is_token_in_rank_ptr);

    uccl::layout::get_dispatch_layout(
        topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank,
        num_tokens_per_expert, is_token_in_rank, num_tokens, num_topk,
        num_ranks, num_experts, comm_stream);

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
    } else {
      stream_wait(compute_stream, comm_stream);
    }
    return event;
  }

  ~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
      destroy();
    } else if (not destroyed) {
      printf(
          "WARNING: destroy() was not called before DeepEP buffer destruction, "
          "which can leak resources.\n");
      fflush(stdout);
    }
  }

  void destroy() {
    EP_HOST_ASSERT(not destroyed);

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    if (num_nvl_bytes > 0) {
      // Barrier
      intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks,
                         comm_stream);
      CUDA_CHECK(cudaDeviceSynchronize());

      // Close remote IPC
      if (is_available()) {
        for (int i = 0; i < num_nvl_ranks; ++i)
          if (i != nvl_rank) CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs[i]));
      }

      // Free local buffer and error flag
      CUDA_CHECK(cudaFree(buffer_ptrs[nvl_rank]));
    }

    if (num_rdma_bytes > 0) {
      for (int i = 0; i < num_nvl_ranks; ++i) {
        if (i != nvl_rank && ipc_rdma_base_ptrs[i] != nullptr) {
          CUDA_CHECK(cudaIpcCloseMemHandle(ipc_rdma_base_ptrs[i]));
        }
      }
    }

    // Free workspace and MoE counter
    CUDA_CHECK(cudaFree(workspace));
    if (d_ipc_rdma_base_ptrs != nullptr) {
      CUDA_CHECK(cudaFree(d_ipc_rdma_base_ptrs));
    }
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));
    // Free D2HHandle device-side arrays if allocated
    if (d_handle_objs) {
      CUDA_CHECK(cudaFree(d_handle_objs));
      d_handle_objs = nullptr;
    }
    if (d_handles) {
      CUDA_CHECK(cudaFree(d_handles));
    }
    if (comm_stream != nullptr) {
      CUDA_CHECK(cudaStreamDestroy(comm_stream));
      comm_stream = nullptr;
    }
    destroyed = true;
    available = false;
  }

  std::tuple<int, std::vector<int>, std::optional<EventHandle>>
  intranode_prepare(std::uintptr_t num_tokens_per_rank_ptr,
                    std::uintptr_t is_token_in_rank_ptr,
                    std::uintptr_t num_tokens_per_expert_ptr, int num_tokens,
                    int num_experts, std::uintptr_t rank_prefix_matrix_ptr,
                    std::uintptr_t channel_prefix_matrix_ptr,
                    int expert_alignment, int num_worst_tokens,
                    uccl::Config const& config,
                    std::optional<EventHandle>& previous_event, bool async,
                    bool allocate_on_comm_stream,
                    std::uintptr_t compute_stream_ptr) {
    EP_HOST_ASSERT(num_tokens > 0);
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);
    EP_HOST_ASSERT(is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);
    EP_HOST_ASSERT(rank_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(channel_prefix_matrix_ptr != 0);

    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    int num_local_experts = num_experts / num_ranks;
    EP_HOST_ASSERT(num_local_experts <= NUM_MAX_LOCAL_EXPERTS);

    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    static_cast<void>(allocate_on_comm_stream);
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    int* num_tokens_per_rank = reinterpret_cast<int*>(num_tokens_per_rank_ptr);
    bool* is_token_in_rank = reinterpret_cast<bool*>(is_token_in_rank_ptr);
    int* num_tokens_per_expert =
        reinterpret_cast<int*>(num_tokens_per_expert_ptr);
    int* rank_prefix_matrix = reinterpret_cast<int*>(rank_prefix_matrix_ptr);
    int* channel_prefix_matrix =
        reinterpret_cast<int*>(channel_prefix_matrix_ptr);

    *moe_recv_counter = -1;
    for (int i = 0; i < num_local_experts; ++i) moe_recv_expert_counter[i] = -1;
    int num_memset_int = num_channels * num_ranks * 4;
    EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <=
                   static_cast<size_t>(num_nvl_bytes));
    uccl::intranode::notify_dispatch(
        num_tokens_per_rank, moe_recv_counter_mapped, num_ranks,
        num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts,
        num_tokens, is_token_in_rank, channel_prefix_matrix, rank_prefix_matrix,
        num_memset_int, expert_alignment, buffer_ptrs_gpu,
        barrier_signal_ptrs_gpu, rank, comm_stream, num_channels);

    int num_recv_tokens = -1;
    std::vector<int> num_recv_tokens_per_expert_list;
    if (num_worst_tokens > 0) {
      num_recv_tokens = num_worst_tokens;
    } else {
      auto start_time = std::chrono::high_resolution_clock::now();
      while (true) {
        num_recv_tokens = static_cast<int>(*moe_recv_counter);
        bool ready = (num_recv_tokens >= 0);
        for (int i = 0; i < num_local_experts and ready; ++i) {
          ready &= moe_recv_expert_counter[i] >= 0;
        }
        if (ready) break;
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time)
                .count() > NUM_CPU_TIMEOUT_SECS) {
          throw std::runtime_error("DeepEP error: CPU recv timeout");
        }
      }
      num_recv_tokens_per_expert_list = std::vector<int>(
          moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
    } else {
      stream_wait(compute_stream, comm_stream);
    }
    return {num_recv_tokens, num_recv_tokens_per_expert_list, event};
  }

  std::optional<EventHandle> intranode_dispatch(
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_element_size,
      std::uintptr_t x_scales_ptr, int num_scales, int scale_token_stride,
      int scale_hidden_stride, std::uintptr_t topk_idx_ptr, int num_topk,
      std::uintptr_t topk_weights_ptr, std::uintptr_t is_token_in_rank_ptr,
      std::uintptr_t rank_prefix_matrix_ptr,
      std::uintptr_t channel_prefix_matrix_ptr, int num_experts,
      int num_worst_tokens, bool cached_mode, uccl::Config const& config,
      int num_recv_tokens, std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_x_scales_ptr, std::uintptr_t recv_topk_idx_ptr,
      std::uintptr_t recv_topk_weights_ptr,
      std::uintptr_t recv_channel_prefix_matrix_ptr,
      std::uintptr_t recv_src_idx_ptr, std::uintptr_t send_head_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
    EP_HOST_ASSERT(x_ptr != 0 && is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_x_ptr != 0 && recv_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_src_idx_ptr != 0 && send_head_ptr != 0);
    EP_HOST_ASSERT(num_tokens > 0 && hidden > 0 && num_recv_tokens >= 0);
    EP_HOST_ASSERT((hidden * x_element_size) % static_cast<int>(sizeof(int4)) ==
                   0);

    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    static_cast<void>(allocate_on_comm_stream);
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }
    if (cached_mode) {
      EP_HOST_ASSERT(rank_prefix_matrix_ptr != 0);
      int num_memset_int = num_channels * num_ranks * 4;
      uccl::intranode::cached_notify_dispatch(
          reinterpret_cast<int*>(rank_prefix_matrix_ptr), num_memset_int,
          buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks,
          comm_stream);
    }

    auto* x = reinterpret_cast<void*>(x_ptr);
    auto* x_scales =
        x_scales_ptr == 0 ? nullptr : reinterpret_cast<float*>(x_scales_ptr);
    auto* topk_idx =
        topk_idx_ptr == 0 ? nullptr : reinterpret_cast<int64_t*>(topk_idx_ptr);
    auto* topk_weights = topk_weights_ptr == 0
                             ? nullptr
                             : reinterpret_cast<float*>(topk_weights_ptr);

    uccl::intranode::dispatch(
        reinterpret_cast<void*>(recv_x_ptr),
        recv_x_scales_ptr == 0 ? nullptr
                               : reinterpret_cast<float*>(recv_x_scales_ptr),
        reinterpret_cast<int*>(recv_src_idx_ptr),
        recv_topk_idx_ptr == 0 ? nullptr
                               : reinterpret_cast<int64_t*>(recv_topk_idx_ptr),
        recv_topk_weights_ptr == 0
            ? nullptr
            : reinterpret_cast<float*>(recv_topk_weights_ptr),
        reinterpret_cast<int*>(recv_channel_prefix_matrix_ptr),
        reinterpret_cast<int*>(send_head_ptr), x, x_scales, topk_idx,
        topk_weights, reinterpret_cast<bool*>(is_token_in_rank_ptr),
        reinterpret_cast<int*>(channel_prefix_matrix_ptr), num_tokens,
        num_worst_tokens,
        static_cast<int>(hidden * x_element_size / sizeof(int4)), num_topk,
        num_experts, num_scales, scale_token_stride, scale_hidden_stride,
        buffer_ptrs_gpu, rank, num_ranks, comm_stream, config.num_sms,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens);

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
    } else {
      stream_wait(compute_stream, comm_stream);
    }
    return event;
  }

  std::optional<EventHandle> intranode_combine(
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_dtype_code,
      int x_element_size, std::uintptr_t topk_weights_ptr, int num_topk,
      std::uintptr_t bias_0_ptr, std::uintptr_t bias_1_ptr,
      std::uintptr_t src_idx_ptr, int num_recv_tokens,
      std::uintptr_t rank_prefix_matrix_ptr,
      std::uintptr_t channel_prefix_matrix_ptr, std::uintptr_t send_head_ptr,
      uccl::Config const& config, std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_topk_weights_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
    EP_HOST_ASSERT(x_ptr != 0 && src_idx_ptr != 0 &&
                   rank_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(channel_prefix_matrix_ptr != 0 && send_head_ptr != 0);
    EP_HOST_ASSERT(recv_x_ptr != 0);
    EP_HOST_ASSERT((hidden * x_element_size) % static_cast<int>(sizeof(int4)) ==
                   0);

    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    static_cast<void>(allocate_on_comm_stream);
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <=
                   static_cast<size_t>(num_nvl_bytes));
    uccl::intranode::cached_notify_combine(
        buffer_ptrs_gpu, reinterpret_cast<int*>(send_head_ptr), num_channels,
        num_recv_tokens, num_channels * num_ranks * 2, barrier_signal_ptrs_gpu,
        rank, num_ranks, comm_stream);

    void* bias_ptrs[2] = {
        bias_0_ptr == 0 ? nullptr : reinterpret_cast<void*>(bias_0_ptr),
        bias_1_ptr == 0 ? nullptr : reinterpret_cast<void*>(bias_1_ptr),
    };
    uccl::intranode::combine(
        cuda_dtype_from_code(x_dtype_code), reinterpret_cast<void*>(recv_x_ptr),
        recv_topk_weights_ptr == 0
            ? nullptr
            : reinterpret_cast<float*>(recv_topk_weights_ptr),
        reinterpret_cast<void*>(x_ptr),
        topk_weights_ptr == 0 ? nullptr
                              : reinterpret_cast<float*>(topk_weights_ptr),
        bias_ptrs[0], bias_ptrs[1], reinterpret_cast<int*>(src_idx_ptr),
        reinterpret_cast<int*>(rank_prefix_matrix_ptr),
        reinterpret_cast<int*>(channel_prefix_matrix_ptr),
        reinterpret_cast<int*>(send_head_ptr), num_tokens, num_recv_tokens,
        hidden, num_topk, buffer_ptrs_gpu, rank, num_ranks, comm_stream,
        config.num_sms, config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens);

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
    } else {
      stream_wait(compute_stream, comm_stream);
    }
    return event;
  }

  std::tuple<int, int, std::vector<int>, std::optional<EventHandle>>
  internode_prepare(std::uintptr_t num_tokens_per_rank_ptr,
                    std::uintptr_t num_tokens_per_rdma_rank_ptr,
                    std::uintptr_t num_tokens_per_expert_ptr,
                    std::uintptr_t is_token_in_rank_ptr, int num_tokens,
                    int hidden, int x_element_size, int num_scales,
                    int num_topk, int num_experts, int expert_alignment,
                    int num_worst_tokens, uccl::Config const& config,
                    std::uintptr_t rdma_channel_prefix_matrix_ptr,
                    std::uintptr_t recv_rdma_rank_prefix_sum_ptr,
                    std::uintptr_t gbl_channel_prefix_matrix_ptr,
                    std::uintptr_t recv_gbl_rank_prefix_sum_ptr,
                    std::optional<EventHandle>& previous_event, bool async,
                    bool allocate_on_comm_stream,
                    std::uintptr_t compute_stream_ptr) {
    nb::gil_scoped_release release;
    EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_rdma_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);
    EP_HOST_ASSERT(is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_rdma_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_gbl_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(num_tokens > 0 && hidden > 0 && num_experts > 0);
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() &&
                   get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    int const num_channels = config.num_sms / 2;
    int const hidden_int4 =
        hidden * x_element_size / static_cast<int>(sizeof(int4));
    int const num_local_experts = num_experts / num_ranks;
    EP_HOST_ASSERT(num_local_experts <= NUM_MAX_LOCAL_EXPERTS);

    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    static_cast<void>(allocate_on_comm_stream);
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    *moe_recv_counter = -1;
    *moe_recv_rdma_counter = -1;
    for (int i = 0; i < num_local_experts; ++i) moe_recv_expert_counter[i] = -1;

    uccl::internode::notify_dispatch(
        reinterpret_cast<int const*>(num_tokens_per_rank_ptr),
        moe_recv_counter_mapped, num_ranks,
        reinterpret_cast<int const*>(num_tokens_per_rdma_rank_ptr),
        moe_recv_rdma_counter_mapped,
        reinterpret_cast<int const*>(num_tokens_per_expert_ptr),
        moe_recv_expert_counter_mapped, num_experts,
        reinterpret_cast<bool const*>(is_token_in_rank_ptr), num_tokens,
        num_worst_tokens, num_channels, hidden_int4, num_scales, num_topk,
        expert_alignment,
        reinterpret_cast<int*>(rdma_channel_prefix_matrix_ptr),
        reinterpret_cast<int*>(recv_rdma_rank_prefix_sum_ptr),
        reinterpret_cast<int*>(gbl_channel_prefix_matrix_ptr),
        reinterpret_cast<int*>(recv_gbl_rank_prefix_sum_ptr), rdma_buffer_ptr,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
        config.num_max_nvl_chunked_recv_tokens, barrier_signal_ptrs_gpu, rank,
        comm_stream,
        config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
        num_nvl_bytes, low_latency_mode, d_handles, num_d2h_channel_addrs,
        atomic_buffer_ptr);

    int num_recv_tokens = -1;
    int num_rdma_recv_tokens = -1;
    std::vector<int> num_recv_tokens_per_expert_list;
    if (num_worst_tokens > 0) {
      num_recv_tokens = num_worst_tokens;
      num_rdma_recv_tokens = num_worst_tokens;
    } else {
      auto start_time = std::chrono::high_resolution_clock::now();
      while (true) {
        num_recv_tokens = static_cast<int>(*moe_recv_counter);
        num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);
        bool ready = (num_recv_tokens >= 0) && (num_rdma_recv_tokens >= 0);
        for (int i = 0; i < num_local_experts && ready; ++i) {
          ready &= moe_recv_expert_counter[i] >= 0;
        }
        if (ready) break;
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time)
                .count() > NUM_CPU_TIMEOUT_SECS) {
          throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
        }
      }
      num_recv_tokens_per_expert_list = std::vector<int>(
          moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
    } else {
      stream_wait(compute_stream, comm_stream);
    }
    return {num_recv_tokens, num_rdma_recv_tokens,
            num_recv_tokens_per_expert_list, event};
  }

  std::optional<EventHandle> internode_dispatch(
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_element_size,
      std::uintptr_t x_scales_ptr, int num_scales, int scale_token_stride,
      int scale_hidden_stride, std::uintptr_t topk_idx_ptr, int num_topk,
      std::uintptr_t topk_weights_ptr, std::uintptr_t is_token_in_rank_ptr,
      std::uintptr_t rdma_channel_prefix_matrix_ptr,
      std::uintptr_t recv_rdma_rank_prefix_sum_ptr,
      std::uintptr_t gbl_channel_prefix_matrix_ptr,
      std::uintptr_t recv_gbl_rank_prefix_sum_ptr, int num_experts,
      int num_worst_tokens, bool cached_mode, int num_rdma_recv_tokens,
      uccl::Config const& config, std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_x_scales_ptr, std::uintptr_t recv_topk_idx_ptr,
      std::uintptr_t recv_topk_weights_ptr, std::uintptr_t recv_src_meta_ptr,
      std::uintptr_t recv_rdma_channel_prefix_matrix_ptr,
      std::uintptr_t recv_gbl_channel_prefix_matrix_ptr,
      std::uintptr_t send_rdma_head_ptr, std::uintptr_t send_nvl_head_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
    EP_HOST_ASSERT(x_ptr != 0 && recv_x_ptr != 0 && is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_rdma_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_gbl_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(num_tokens > 0 && hidden > 0);
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    int const num_channels = config.num_sms / 2;
    int const hidden_int4 =
        hidden * x_element_size / static_cast<int>(sizeof(int4));
    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    static_cast<void>(allocate_on_comm_stream);
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    if (cached_mode) {
      uccl::internode::cached_notify(
          hidden_int4, num_scales, num_topk, num_topk, num_ranks, num_channels,
          0, nullptr,
          reinterpret_cast<int const*>(rdma_channel_prefix_matrix_ptr),
          reinterpret_cast<int const*>(recv_rdma_rank_prefix_sum_ptr), nullptr,
          rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
          buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
          barrier_signal_ptrs_gpu, rank, comm_stream,
          config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4),
                                           num_ranks),
          num_nvl_bytes, true, low_latency_mode, d_handles,
          num_d2h_channel_addrs, atomic_buffer_ptr);
    } else {
      EP_HOST_ASSERT(recv_src_meta_ptr != 0);
      EP_HOST_ASSERT(send_rdma_head_ptr != 0);
      EP_HOST_ASSERT(send_nvl_head_ptr != 0);
      EP_HOST_ASSERT(recv_rdma_channel_prefix_matrix_ptr != 0);
      EP_HOST_ASSERT(recv_gbl_channel_prefix_matrix_ptr != 0);
    }

    uccl::internode::dispatch(
        reinterpret_cast<void*>(recv_x_ptr),
        recv_x_scales_ptr == 0 ? nullptr
                               : reinterpret_cast<float*>(recv_x_scales_ptr),
        recv_topk_idx_ptr == 0 ? nullptr
                               : reinterpret_cast<int64_t*>(recv_topk_idx_ptr),
        recv_topk_weights_ptr == 0
            ? nullptr
            : reinterpret_cast<float*>(recv_topk_weights_ptr),
        cached_mode ? nullptr : reinterpret_cast<void*>(recv_src_meta_ptr),
        reinterpret_cast<void const*>(x_ptr),
        x_scales_ptr == 0 ? nullptr
                          : reinterpret_cast<float const*>(x_scales_ptr),
        topk_idx_ptr == 0 ? nullptr
                          : reinterpret_cast<int64_t const*>(topk_idx_ptr),
        topk_weights_ptr == 0
            ? nullptr
            : reinterpret_cast<float const*>(topk_weights_ptr),
        cached_mode ? nullptr : reinterpret_cast<int*>(send_rdma_head_ptr),
        cached_mode ? nullptr : reinterpret_cast<int*>(send_nvl_head_ptr),
        cached_mode
            ? nullptr
            : reinterpret_cast<int*>(recv_rdma_channel_prefix_matrix_ptr),
        cached_mode
            ? nullptr
            : reinterpret_cast<int*>(recv_gbl_channel_prefix_matrix_ptr),
        reinterpret_cast<int const*>(rdma_channel_prefix_matrix_ptr),
        reinterpret_cast<int const*>(recv_rdma_rank_prefix_sum_ptr),
        reinterpret_cast<int const*>(gbl_channel_prefix_matrix_ptr),
        reinterpret_cast<int const*>(recv_gbl_rank_prefix_sum_ptr),
        reinterpret_cast<bool const*>(is_token_in_rank_ptr), num_tokens,
        num_worst_tokens, hidden_int4, num_scales, num_topk, num_experts,
        scale_token_stride, scale_hidden_stride, rdma_buffer_ptr,
        config.num_max_rdma_chunked_send_tokens,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens, rank, num_ranks, cached_mode,
        comm_stream, num_channels, low_latency_mode, d_handles,
        num_d2h_channel_addrs, atomic_buffer_ptr);

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
    } else {
      stream_wait(compute_stream, comm_stream);
    }
    return event;
  }

  std::optional<EventHandle> internode_combine(
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_dtype_code,
      int x_element_size, std::uintptr_t topk_weights_ptr, int num_topk,
      std::uintptr_t bias_0_ptr, std::uintptr_t bias_1_ptr,
      std::uintptr_t src_meta_ptr, int num_combined_tokens,
      std::uintptr_t is_combined_token_in_rank_ptr,
      std::uintptr_t rdma_channel_prefix_matrix_ptr,
      std::uintptr_t rdma_rank_prefix_sum_ptr,
      std::uintptr_t gbl_channel_prefix_matrix_ptr,
      std::uintptr_t combined_rdma_head_ptr,
      std::uintptr_t combined_nvl_head_ptr, uccl::Config const& config,
      std::uintptr_t combined_x_ptr, std::uintptr_t combined_topk_weights_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
    EP_HOST_ASSERT(x_ptr != 0 && src_meta_ptr != 0 && combined_x_ptr != 0);
    EP_HOST_ASSERT(is_combined_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(rdma_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(combined_rdma_head_ptr != 0);
    EP_HOST_ASSERT(combined_nvl_head_ptr != 0);
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    int const num_channels = config.num_sms / 2;
    int const hidden_int4 =
        hidden * x_element_size / static_cast<int>(sizeof(int4));
    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    static_cast<void>(allocate_on_comm_stream);
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    uccl::internode::cached_notify(
        hidden_int4, 0, 0, num_topk, num_ranks, num_channels,
        num_combined_tokens, reinterpret_cast<int*>(combined_rdma_head_ptr),
        reinterpret_cast<int const*>(rdma_channel_prefix_matrix_ptr),
        reinterpret_cast<int const*>(rdma_rank_prefix_sum_ptr),
        reinterpret_cast<int*>(combined_nvl_head_ptr), rdma_buffer_ptr,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
        config.num_max_nvl_chunked_recv_tokens, barrier_signal_ptrs_gpu, rank,
        comm_stream,
        config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
        num_nvl_bytes, false, low_latency_mode, d_handles,
        num_d2h_channel_addrs, atomic_buffer_ptr);

    void* bias_ptrs[2] = {
        bias_0_ptr == 0 ? nullptr : reinterpret_cast<void*>(bias_0_ptr),
        bias_1_ptr == 0 ? nullptr : reinterpret_cast<void*>(bias_1_ptr),
    };
    uccl::internode::combine(
        cuda_dtype_from_code(x_dtype_code),
        reinterpret_cast<void*>(combined_x_ptr),
        combined_topk_weights_ptr == 0
            ? nullptr
            : reinterpret_cast<float*>(combined_topk_weights_ptr),
        reinterpret_cast<bool const*>(is_combined_token_in_rank_ptr),
        reinterpret_cast<void const*>(x_ptr),
        topk_weights_ptr == 0
            ? nullptr
            : reinterpret_cast<float const*>(topk_weights_ptr),
        bias_ptrs[0], bias_ptrs[1],
        reinterpret_cast<int const*>(combined_rdma_head_ptr),
        reinterpret_cast<int const*>(combined_nvl_head_ptr),
        reinterpret_cast<void const*>(src_meta_ptr),
        reinterpret_cast<int const*>(rdma_channel_prefix_matrix_ptr),
        reinterpret_cast<int const*>(rdma_rank_prefix_sum_ptr),
        reinterpret_cast<int const*>(gbl_channel_prefix_matrix_ptr), num_tokens,
        num_combined_tokens, hidden, num_topk, rdma_buffer_ptr,
        config.num_max_rdma_chunked_send_tokens,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens, rank, num_ranks, comm_stream,
        num_channels, low_latency_mode, d_handles, num_d2h_channel_addrs,
        atomic_buffer_ptr);

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
    } else {
      stream_wait(compute_stream, comm_stream);
    }
    return event;
  }

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                int hidden, int num_experts,
                                std::uintptr_t stream_ptr) {
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = uccl::LowLatencyLayout(rdma_buffer_ptr,
                                         num_max_dispatch_tokens_per_rank,
                                         hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();
    auto [ptr0, ptr_internode0, count0] = clean_meta_0;
    auto [ptr1, ptr_internode1, count1] = clean_meta_1;

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
      auto offset = reinterpret_cast<int64_t>(ptr) -
                    reinterpret_cast<int64_t>(rdma_buffer_ptr);
      EP_HOST_ASSERT(0 <= offset &&
                     offset + num_bytes <= static_cast<size_t>(num_rdma_bytes));
    };
    check_boundary(ptr0, count0 * sizeof(int));
    check_boundary(ptr1, count1 * sizeof(int));

    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    uccl::internode_ll::clean_low_latency_buffer(ptr0, count0, ptr1, count1,
                                                 stream);
  }

  std::tuple<std::optional<EventHandle>, std::optional<std::function<void()>>>
  low_latency_dispatch(std::uintptr_t x_ptr, int x_rows, int x_cols,
                       std::uintptr_t topk_idx_ptr, int topk_rows,
                       int topk_cols, std::uintptr_t packed_recv_x_ptr,
                       std::uintptr_t packed_recv_x_scales_ptr,
                       std::uintptr_t packed_recv_count_ptr,
                       std::uintptr_t packed_recv_src_info_ptr,
                       std::uintptr_t packed_recv_layout_range_ptr,
                       std::uintptr_t cumulative_local_expert_recv_stats_ptr,
                       std::uintptr_t dispatch_wait_recv_cost_stats_ptr,
                       std::uintptr_t compute_stream_ptr,
                       int num_max_dispatch_tokens_per_rank, int num_experts,
                       bool use_fp8, bool round_scale, bool use_ue8m0,
                       bool async, bool return_recv_hook) {
    EP_HOST_ASSERT(low_latency_mode);
    EP_HOST_ASSERT(x_ptr != 0 && topk_idx_ptr != 0);
    EP_HOST_ASSERT(packed_recv_x_ptr != 0 && packed_recv_count_ptr != 0);
    EP_HOST_ASSERT(packed_recv_src_info_ptr != 0 &&
                   packed_recv_layout_range_ptr != 0);
    EP_HOST_ASSERT(x_rows == topk_rows);
    EP_HOST_ASSERT(x_rows <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x_cols % static_cast<int>(sizeof(int4)) == 0 &&
                   x_cols % 128 == 0);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 &&
                   "TMA requires the number of tokens to be multiple of 4");
    if (use_fp8) {
      EP_HOST_ASSERT(packed_recv_x_scales_ptr != 0);
      EP_HOST_ASSERT(x_cols % 512 == 0);
      if (use_ue8m0) EP_HOST_ASSERT(round_scale);
    }

    auto num_tokens = x_rows;
    auto hidden = x_cols;
    auto num_topk = topk_cols;

    uccl::LowLatencyLayout layout(rdma_buffer_ptr,
                                  num_max_dispatch_tokens_per_rank, hidden,
                                  num_ranks, num_experts, atomic_buffer_ptr);
    EP_HOST_ASSERT(layout.total_bytes <=
                   static_cast<std::size_t>(num_rdma_bytes));
    int low_latency_buffer_idx_used = low_latency_buffer_idx;
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook) stream_wait(launch_stream, compute_stream);

    void* x = reinterpret_cast<void*>(x_ptr);
    int64_t* topk_idx = reinterpret_cast<int64_t*>(topk_idx_ptr);
    void* packed_recv_x = reinterpret_cast<void*>(packed_recv_x_ptr);
    void* packed_recv_x_scales =
        packed_recv_x_scales_ptr == 0
            ? nullptr
            : reinterpret_cast<void*>(packed_recv_x_scales_ptr);
    int* packed_recv_count = reinterpret_cast<int*>(packed_recv_count_ptr);
    int* packed_recv_src_info =
        reinterpret_cast<int*>(packed_recv_src_info_ptr);
    int64_t* packed_recv_layout_range =
        reinterpret_cast<int64_t*>(packed_recv_layout_range_ptr);
    int* cumulative_local_expert_recv_stats =
        cumulative_local_expert_recv_stats_ptr == 0
            ? nullptr
            : reinterpret_cast<int*>(cumulative_local_expert_recv_stats_ptr);
    int64_t* dispatch_wait_recv_cost_stats =
        dispatch_wait_recv_cost_stats_ptr == 0
            ? nullptr
            : reinterpret_cast<int64_t*>(dispatch_wait_recv_cost_stats_ptr);

    auto [ptr0, ptr_internode0, count0] = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
      uccl::internode_ll::dispatch(
          packed_recv_x, packed_recv_x_scales, packed_recv_src_info,
          packed_recv_layout_range, packed_recv_count,
          cumulative_local_expert_recv_stats, dispatch_wait_recv_cost_stats,
          buffer.dispatch_rdma_recv_data_buffer,
          buffer.dispatch_rdma_recv_count_buffer,
          buffer.dispatch_rdma_send_buffer, x, topk_idx, ptr0, ptr_internode0,
          count0, num_tokens, hidden, num_max_dispatch_tokens_per_rank,
          num_topk, num_experts, rank, num_ranks, use_fp8, round_scale,
          use_ue8m0, workspace, num_device_sms, launch_stream, phases,
          d_handles, num_d2h_channel_addrs, max_nvl_peers,
          low_latency_buffer_idx_used, d_ipc_rdma_base_ptrs, rdma_buffer_ptr,
          atomic_buffer_ptr, buffer.dispatch_rdma_recv_count_buffer_internode);
    };
    launcher(return_recv_hook
                 ? LOW_LATENCY_SEND_PHASE
                 : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
      stream_wait(compute_stream, launch_stream);
    }

    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
      recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };
    return {event, recv_hook};
  }

  std::tuple<std::optional<EventHandle>, std::optional<std::function<void()>>>
  low_latency_combine(std::uintptr_t x_ptr, int x_dim0, int x_dim1, int x_dim2,
                      std::uintptr_t topk_idx_ptr, int topk_rows, int topk_cols,
                      std::uintptr_t topk_weights_ptr,
                      std::uintptr_t src_info_ptr, int src_info_dim0,
                      int src_info_dim1, std::uintptr_t layout_range_ptr,
                      int layout_range_dim0, int layout_range_dim1,
                      std::uintptr_t combine_wait_recv_cost_stats_ptr,
                      std::uintptr_t compute_stream_ptr,
                      int num_max_dispatch_tokens_per_rank, int num_experts,
                      bool use_logfmt, bool zero_copy, bool async,
                      bool return_recv_hook, std::uintptr_t out_ptr) {
    EP_HOST_ASSERT(low_latency_mode);
    EP_HOST_ASSERT(x_ptr != 0 && topk_idx_ptr != 0 && topk_weights_ptr != 0);
    EP_HOST_ASSERT(src_info_ptr != 0 && layout_range_ptr != 0);
    EP_HOST_ASSERT(out_ptr != 0);

    auto num_local_experts = num_experts / num_ranks;
    EP_HOST_ASSERT(x_dim0 == num_local_experts);
    EP_HOST_ASSERT(x_dim1 == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x_dim2 % static_cast<int>(sizeof(int4)) == 0 &&
                   x_dim2 % 128 == 0);
    EP_HOST_ASSERT(src_info_dim0 == num_local_experts &&
                   src_info_dim1 ==
                       num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(layout_range_dim0 == num_local_experts &&
                   layout_range_dim1 == num_ranks);
    EP_HOST_ASSERT(topk_rows <= num_max_dispatch_tokens_per_rank);

    auto hidden = x_dim2;
    auto num_topk = topk_cols;
    auto num_combined_tokens = topk_rows;

    uccl::LowLatencyLayout layout(rdma_buffer_ptr,
                                  num_max_dispatch_tokens_per_rank, hidden,
                                  num_ranks, num_experts, atomic_buffer_ptr);
    EP_HOST_ASSERT(layout.total_bytes <=
                   static_cast<std::size_t>(num_rdma_bytes));
    int low_latency_buffer_idx_used = low_latency_buffer_idx;
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    auto compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_ptr);
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook) stream_wait(launch_stream, compute_stream);

    void* x = reinterpret_cast<void*>(x_ptr);
    int64_t* topk_idx = reinterpret_cast<int64_t*>(topk_idx_ptr);
    float* topk_weights = reinterpret_cast<float*>(topk_weights_ptr);
    int* src_info = reinterpret_cast<int*>(src_info_ptr);
    int64_t* layout_range = reinterpret_cast<int64_t*>(layout_range_ptr);
    int64_t* combine_wait_recv_cost_stats =
        combine_wait_recv_cost_stats_ptr == 0
            ? nullptr
            : reinterpret_cast<int64_t*>(combine_wait_recv_cost_stats_ptr);
    void* out = reinterpret_cast<void*>(out_ptr);

    auto [ptr0, ptr_internode0, count0] = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
      uccl::internode_ll::combine(
          out, buffer.combine_rdma_recv_data_buffer,
          buffer.combine_rdma_recv_flag_buffer, buffer.combine_rdma_send_buffer,
          x, topk_idx, topk_weights, src_info, layout_range,
          combine_wait_recv_cost_stats, ptr0, ptr_internode0, count0,
          num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank,
          num_topk, num_experts, rank, num_ranks, use_logfmt, workspace,
          num_device_sms, launch_stream, phases, zero_copy, d_handles,
          num_d2h_channel_addrs, max_nvl_peers, low_latency_buffer_idx_used,
          d_ipc_rdma_base_ptrs, rdma_buffer_ptr, atomic_buffer_ptr,
          buffer.combine_rdma_recv_flag_buffer_internode);
    };
    launcher(return_recv_hook
                 ? LOW_LATENCY_SEND_PHASE
                 : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
      stream_wait(compute_stream, launch_stream);
    }

    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
      recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };
    return {event, recv_hook};
  }

  int get_local_device_id() { return device_index; }

  nb::bytes get_local_ipc_handle() const {
    return nb::bytes(
        reinterpret_cast<char const*>(ipc_handles[nvl_rank].reserved),
        CUDA_IPC_HANDLE_SIZE);
  }

  nb::bytes get_local_rdma_ipc_handle() {
    EP_HOST_ASSERT(
        rdma_buffer_ptr != nullptr &&
        "set_rdma_buffer must be called before requesting RDMA IPC handle");
    cudaIpcMemHandle_t h{};
    CUDA_CHECK(cudaIpcGetMemHandle(&h, rdma_buffer_ptr));
    return nb::bytes(reinterpret_cast<char const*>(h.reserved),
                     CUDA_IPC_HANDLE_SIZE);
  }

  nb::bytes get_local_atomics_ipc_handle() {
    EP_HOST_ASSERT(atomic_buffer_ptr != nullptr &&
                   "set_atomic_buffer must be called before requesting "
                   "atomic IPC handle");
    cudaIpcMemHandle_t h{};
    CUDA_CHECK(cudaIpcGetMemHandle(&h, atomic_buffer_ptr));
    return nb::bytes(reinterpret_cast<char const*>(h.reserved),
                     CUDA_IPC_HANDLE_SIZE);
  }

  int get_num_rdma_ranks() const { return num_rdma_ranks; }
  int get_num_max_nvl_peers() const { return NUM_MAX_NVL_PEERS; }
  int get_source_meta_bytes() const {
    return uccl::internode::get_source_meta_bytes();
  }
  int get_rdma_rank() const { return rdma_rank; }
  int get_root_rdma_rank(bool global) const { return global ? nvl_rank : 0; }

  nb::bytes get_local_uccl_shmem_unique_id() const {
    EP_HOST_ASSERT(rdma_rank == 0 and
                   "Only RDMA rank 0 can get UCCL unique ID");
    auto unique_id = internode::get_unique_id();
    return nb::bytes(reinterpret_cast<char const*>(unique_id.data()),
                     unique_id.size());
  }

  void reset_rdma_buffer() {
    CUDA_CHECK(
        cudaMemsetAsync(rdma_buffer_ptr, 0, num_rdma_bytes, comm_stream));
    CUDA_CHECK(cudaStreamSynchronize(comm_stream));
    // printf("RDMA buffer reset done\n");

    if (atomic_buffer_ptr != nullptr) {
      cudaMemset(atomic_buffer_ptr, 0, kAtomicBufferSize);
      printf("Atomic buffer reset done\n");
    }
  }

  void sync(std::vector<int> const& device_ids,
            std::vector<std::optional<nb::bytes>> const& all_gathered_handles,
            std::optional<nb::bytes> const& root_unique_id_opt,
            std::optional<std::vector<std::optional<nb::bytes>>> const&
                all_gathered_rdma_handles_opt = std::nullopt) {
    EP_HOST_ASSERT(not is_available());
    // Sync IPC handles
    if (num_nvl_bytes > 0) {
      EP_HOST_ASSERT(static_cast<std::size_t>(num_ranks) == device_ids.size());
      EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
      for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks;
           ++i) {
        int global_rank = offset + i;
        int local_rank_idx =
            global_rank % max_nvl_peers;  // Map to correct buffer_ptrs index

        EP_HOST_ASSERT(all_gathered_handles[global_rank].has_value());
        nb::bytes const& h = all_gathered_handles[global_rank].value();
        std::string handle_str(static_cast<char const*>(h.data()), h.size());
        EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
        if (global_rank != rank) {
          std::memcpy(ipc_handles[local_rank_idx].reserved, handle_str.c_str(),
                      CUDA_IPC_HANDLE_SIZE);
          // Ensure we're on the correct device before opening IPC handle
          CUDA_CHECK(cudaSetDevice(device_index));
          CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[local_rank_idx],
                                          ipc_handles[local_rank_idx],
                                          cudaIpcMemLazyEnablePeerAccess));
          barrier_signal_ptrs[local_rank_idx] = reinterpret_cast<int*>(
              static_cast<uint8_t*>(buffer_ptrs[local_rank_idx]) +
              num_nvl_bytes);
        } else {
          // This is our own rank - buffer_ptrs[local_rank_idx] should already
          // be set from constructor But let's verify it's not null and the IPC
          // handle matches
          EP_HOST_ASSERT(buffer_ptrs[local_rank_idx] != nullptr);
          EP_HOST_ASSERT(std::memcmp(ipc_handles[local_rank_idx].reserved,
                                     handle_str.c_str(),
                                     CUDA_IPC_HANDLE_SIZE) == 0);
        }
      }

      // Copy all buffer and barrier signal pointers to GPU
      CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs,
                            sizeof(void*) * max_nvl_peers,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs,
                            sizeof(int*) * max_nvl_peers,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Sync NVSHMEM handles and allocate memory
    // NOTE(MaoZiming): drop nvshmem. we directly allocate rdma_buffer_ptr.
    if (num_rdma_bytes > 0) {
      // TODO(MaoZiming): this needs to be allocated by proxy.
      if (!rdma_buffer_ptr) {
        fprintf(stderr,
                "WARNING: rdma_buffer_ptr is not set, allocating %ld bytes "
                "for RDMA buffer.\n",
                num_rdma_bytes);
        std::abort();
      }
      reset_rdma_buffer();
      EP_HOST_ASSERT(all_gathered_rdma_handles_opt.has_value());
      auto const& all_gathered_rdma_handles = *all_gathered_rdma_handles_opt;
      EP_HOST_ASSERT(static_cast<std::size_t>(num_ranks) ==
                     all_gathered_rdma_handles.size());
      for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks;
           ++i) {
        int global_rank = offset + i;
        int local_rank_idx = global_rank % max_nvl_peers;

        if (global_rank == rank) {
          ipc_rdma_base_ptrs[local_rank_idx] = rdma_buffer_ptr;
        } else if (all_gathered_rdma_handles[global_rank].has_value()) {
          nb::bytes const& h = all_gathered_rdma_handles[global_rank].value();
          std::string handle_str(static_cast<char const*>(h.data()), h.size());
          EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
          std::memcpy(rdma_ipc_handles[local_rank_idx].reserved,
                      handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
          CUDA_CHECK(cudaSetDevice(device_index));
          CUDA_CHECK(cudaIpcOpenMemHandle(&ipc_rdma_base_ptrs[local_rank_idx],
                                          rdma_ipc_handles[local_rank_idx],
                                          cudaIpcMemLazyEnablePeerAccess));
        } else {
          // Host-allocated RDMA buffer (cudaMallocHost): no IPC handle
          ipc_rdma_base_ptrs[local_rank_idx] = nullptr;
        }
      }
      if (d_ipc_rdma_base_ptrs != nullptr) {
        CUDA_CHECK(cudaMemcpy(d_ipc_rdma_base_ptrs, ipc_rdma_base_ptrs,
                              sizeof(void*) * max_nvl_peers,
                              cudaMemcpyHostToDevice));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Ready to use
    available = true;
  }

  void set_rdma_buffer(std::uintptr_t addr, bool is_host_ptr = false) {
    if (addr == 0) {
      throw std::invalid_argument("set_rdma_buffer: ptr null");
    }

    void* ptr = reinterpret_cast<void*>(addr);
    if (is_host_ptr) {
      CUDA_CHECK(cudaSetDevice(device_index));
      void* dev_ptr = nullptr;
#ifndef USE_GRACE_HOPPER
      CUDA_CHECK(
          cudaHostGetDevicePointer(reinterpret_cast<void**>(&dev_ptr), ptr, 0));
#else
      dev_ptr = ptr;
#endif
      rdma_buffer_ptr = dev_ptr;
    } else {
      rdma_buffer_ptr = ptr;
    }
  }

  void set_atomic_buffer_ptr(void* ptr) {
    if (ptr == nullptr) {
      throw std::invalid_argument("set_atomic_buffer_ptr: ptr null");
    }
    atomic_buffer_ptr = ptr;
  }

  std::uintptr_t get_local_buffer_ptr(int64_t offset,
                                      bool use_rdma_buffer) const {
    auto base_ptr =
        static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr
                                              : buffer_ptrs[nvl_rank]) +
        offset;
    return reinterpret_cast<std::uintptr_t>(base_ptr);
  }

  int64_t get_local_buffer_nbytes(bool use_rdma_buffer) const {
    return use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
  }

  std::uintptr_t get_comm_stream() const {
    return reinterpret_cast<std::uintptr_t>(comm_stream);
  }

  bool is_available() const { return available; }
  bool is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
  }

 private:
  int rank{0};
  int num_ranks{1};
  long num_nvl_bytes{0};
  long num_rdma_bytes{0};
  bool low_latency_mode{false};
  bool explicitly_destroy{false};
  int device_index{0};
  std::vector<nb::object> proxies_;
  bool available{false};
  void* rdma_buffer_ptr = nullptr;
  void* atomic_buffer_ptr = nullptr;
  int low_latency_buffer_idx = 0;
  void* workspace = nullptr;

  // device / ranks
  int rdma_rank{0}, nvl_rank{0};
  int num_rdma_ranks{1}, num_nvl_ranks{1};
  int num_device_sms{0};
  int max_nvl_peers{0};

  // stream & workspace
  cudaStream_t comm_stream{nullptr};

  cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS]{};
  void* buffer_ptrs[NUM_MAX_NVL_PEERS]{};
  int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS]{};
  void** buffer_ptrs_gpu{nullptr};
  int** barrier_signal_ptrs_gpu{nullptr};
  cudaIpcMemHandle_t rdma_ipc_handles[NUM_MAX_NVL_PEERS]{};
  void* ipc_rdma_base_ptrs[NUM_MAX_NVL_PEERS]{};

  // clang-format would change to int volatile*
  // clang-format off
  // MoE counters (host mapped)
  volatile int* moe_recv_counter = nullptr;
  int* moe_recv_counter_mapped{nullptr};  // device pointer
  volatile int* moe_recv_expert_counter{nullptr};
  int* moe_recv_expert_counter_mapped{nullptr};
  volatile int* moe_recv_rdma_counter{nullptr};
  int* moe_recv_rdma_counter_mapped{nullptr};
  // clang-format on

  bool destroyed = false;

  // Ring buffers
  int num_d2h_channel_addrs{0};
  d2hq::D2HHandle* d_handle_objs{nullptr};
  uint64_t* d_handles{nullptr};

  // IPC base pointers for GPU access (for replacing nvshmemi_get_p2p_ptr)
  void** d_ipc_rdma_base_ptrs{
      nullptr};  // Device pointer to array of IPC base addresses
};

NB_MODULE(ep, m) {
  m.doc() = "Minimal DeepEP-compatible shim with UCCL";

  nb::class_<uccl::Config>(m, "Config")
      .def(nb::init<int, int, int, int, int>(), nb::arg("num_sms") = 20,
           nb::arg("num_max_nvl_chunked_send_tokens") = 6,
           nb::arg("num_max_nvl_chunked_recv_tokens") = 256,
           nb::arg("num_max_rdma_chunked_send_tokens") = 6,
           nb::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def_ro("num_sms", &uccl::Config::num_sms)
      .def_ro("num_max_nvl_chunked_send_tokens",
              &uccl::Config::num_max_nvl_chunked_send_tokens)
      .def_ro("num_max_nvl_chunked_recv_tokens",
              &uccl::Config::num_max_nvl_chunked_recv_tokens)
      .def_ro("num_max_rdma_chunked_send_tokens",
              &uccl::Config::num_max_rdma_chunked_send_tokens)
      .def_ro("num_max_rdma_chunked_recv_tokens",
              &uccl::Config::num_max_rdma_chunked_recv_tokens)
      .def("get_nvl_buffer_size_hint", &uccl::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint",
           &uccl::Config::get_rdma_buffer_size_hint);

  m.def(
      "register_proxy",
      [](int device_index, nb::object proxy) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        auto& vec = uccl::g_proxies_by_dev[device_index];
        if (!vec.empty()) {
          fprintf(stderr,
                  "WARNING: overwriting existing proxies for device %d\n",
                  device_index);
          std::abort();
        }
        vec.push_back(std::move(proxy));
        printf("Registered proxy for device %d\n", device_index);
      },
      nb::arg("device_index"), nb::arg("proxy"));
  m.def(
      "register_proxies",
      [](int device_index, std::vector<nb::object> proxies) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        auto& vec = uccl::g_proxies_by_dev[device_index];
        if (!vec.empty()) {
          fprintf(stderr,
                  "WARNING: overwriting existing proxies for device %d\n",
                  device_index);
          std::abort();
        }
        for (auto& proxy : proxies) {
          vec.push_back(std::move(proxy));
        }
        printf("Registered proxies for device %d\n", device_index);
      },
      nb::arg("device_index"), nb::arg("proxies"));
  m.def(
      "unregister_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        uccl::g_proxies_by_dev.erase(device_index);
      },
      nb::arg("device_index"));
  m.def(
      "has_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        auto it = uccl::g_proxies_by_dev.find(device_index);
        return it != uccl::g_proxies_by_dev.end() && !it->second.empty();
      },
      nb::arg("device_index"));
  m.def("stop_all_registered_proxies", []() {
    std::lock_guard<std::mutex> lk(g_proxies_mu);
    for (auto& kv : uccl::g_proxies_by_dev) {
      for (auto& proxy : kv.second) {
        try {
          proxy.attr("stop")();
        } catch (...) {
        }
      }
    }
    uccl::g_proxies_by_dev.clear();
  });

  m.def("get_oob_ip", &uccl::get_oob_ip, "Get the OOB IP address");

  m.def(
      "get_rdma_buffer",
      [](std::size_t num_rdma_bytes, int device_index) {
        return allocate_rdma_buffer_dlpack(num_rdma_bytes, device_index);
      },
      nb::arg("num_rdma_bytes"), nb::arg("device_index"),
      R"doc(
        Allocate the RDMA scratch buffer outside PyTorch's CUDA allocator and
        return it as a DLPack capsule plus an `is_host_allocated` flag.
      )doc");

  m.def(
      "can_register_rdma_gpu_buffer",
      [](int device_index, std::size_t num_bytes) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(EFA)
        return true;
#else
        CUDA_CHECK(cudaSetDevice(device_index));
        return can_register_gpu_memory_for_rdma(device_index, num_bytes);
#endif
      },
      nb::arg("device_index"), nb::arg("num_bytes"),
      R"doc(
        Return whether the main RDMA scratch buffer can stay in GPU memory for
        this device and buffer size.
      )doc");

  m.def(
      "rdma_buffer_should_use_host_alloc",
      [](int device_index, std::size_t num_bytes) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(EFA)
        return false;
#else
        CUDA_CHECK(cudaSetDevice(device_index));
        return !can_register_gpu_memory_for_rdma(device_index, num_bytes);
#endif
      },
      nb::arg("device_index"), nb::arg("num_bytes") = 4096,
      R"doc(
        Return whether the main RDMA scratch buffer should be host allocated
        for this device and buffer size.
      )doc");

  nb::class_<EventHandle>(m, "EventHandle")
      .def(nb::init<>())
      .def(
          "__init__",
          [](EventHandle* t, std::uintptr_t stream_ptr) {
            new (t) EventHandle(reinterpret_cast<cudaStream_t>(stream_ptr));
          },
          nb::arg("stream_ptr"))
      .def("current_stream_wait", &EventHandle::current_stream_wait,
           nb::arg("stream_ptr"));

  m.def("connect_atomic_buffer", [](UcclProxy& p, Buffer& b) {
    b.set_atomic_buffer_ptr(p.get_atomic_buffer_ptr());
  });

  nb::class_<EventOverlap>(m, "EventOverlap").def(nb::init<>());
  nb::class_<Buffer>(m, "Buffer")
      .def(nb::init<int, int, long, long, bool, bool, int>(), nb::arg("rank"),
           nb::arg("num_ranks"), nb::arg("num_nvl_bytes") = 0,
           nb::arg("num_rdma_bytes") = 0, nb::arg("low_latency_mode") = false,
           nb::arg("explicitly_destroy") = false,
           nb::arg("num_local_ranks") = -1)
      .def("destroy", &Buffer::destroy)
      .def(
          "set_rdma_buffer",
          [](Buffer& self, std::uintptr_t addr, bool is_host_ptr) {
            self.set_rdma_buffer(addr, is_host_ptr);
          },
          nb::arg("addr"), nb::arg("is_host_ptr") = false,
          R"doc(Set RDMA buffer from a raw address. Caller must keep the memory alive.)doc")
      .def("reset_rdma_buffer", &Buffer::reset_rdma_buffer)
      .def("low_latency_dispatch", &Buffer::low_latency_dispatch,
           nb::arg("x_ptr"), nb::arg("x_rows"), nb::arg("x_cols"),
           nb::arg("topk_idx_ptr"), nb::arg("topk_rows"), nb::arg("topk_cols"),
           nb::arg("packed_recv_x_ptr"), nb::arg("packed_recv_x_scales_ptr"),
           nb::arg("packed_recv_count_ptr"),
           nb::arg("packed_recv_src_info_ptr"),
           nb::arg("packed_recv_layout_range_ptr"),
           nb::arg("cumulative_local_expert_recv_stats_ptr") = 0,
           nb::arg("dispatch_wait_recv_cost_stats_ptr") = 0,
           nb::arg("compute_stream_ptr"),
           nb::arg("num_max_dispatch_tokens_per_rank") = 0,
           nb::arg("num_experts") = 1, nb::arg("use_fp8") = true,
           nb::arg("round_scale") = false, nb::arg("use_ue8m0") = false,
           nb::arg("is_async") = false, nb::arg("return_recv_hook") = false)
      .def("get_local_device_id", &Buffer::get_local_device_id)
      .def("get_local_ipc_handle", &Buffer::get_local_ipc_handle)
      .def("get_local_rdma_ipc_handle", &Buffer::get_local_rdma_ipc_handle)
      .def("get_local_atomics_ipc_handle",
           &Buffer::get_local_atomics_ipc_handle)
      .def("get_num_rdma_ranks", &Buffer::get_num_rdma_ranks)
      .def("get_num_max_nvl_peers", &Buffer::get_num_max_nvl_peers)
      .def("get_source_meta_bytes", &Buffer::get_source_meta_bytes)
      .def("get_rdma_rank", &Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank)
      .def("get_local_buffer_ptr", &Buffer::get_local_buffer_ptr,
           nb::arg("offset"), nb::arg("use_rdma_buffer"))
      .def("get_local_buffer_nbytes", &Buffer::get_local_buffer_nbytes,
           nb::arg("use_rdma_buffer"))
      .def("get_comm_stream", &Buffer::get_comm_stream)
      .def("get_local_uccl_shmem_unique_id",
           &Buffer::get_local_uccl_shmem_unique_id)
      .def("sync", &Buffer::sync, nb::arg("device_ids"),
           nb::arg("all_gathered_handles"),
           nb::arg("root_unique_id_opt") = nb::none(),
           nb::arg("all_gathered_rdma_handles") = nb::none())
      .def("is_available", &Buffer::is_available)
      .def(
          "get_dispatch_layout",
          [](Buffer& self, std::uintptr_t topk_idx_ptr, int num_tokens,
             int num_topk, int num_experts,
             std::uintptr_t num_tokens_per_rank_ptr,
             std::uintptr_t num_tokens_per_rdma_rank_ptr,
             std::uintptr_t num_tokens_per_expert_ptr,
             std::uintptr_t is_token_in_rank_ptr, nb::object previous_event,
             bool async, bool allocate_on_comm_stream,
             std::uintptr_t compute_stream_ptr) {
            std::optional<EventHandle> prev;
            if (!previous_event.is_none()) {
              EventHandle ev = nb::cast<EventHandle>(previous_event);
              prev = ev;
            }
            return self.get_dispatch_layout(
                topk_idx_ptr, num_tokens, num_topk, num_experts,
                num_tokens_per_rank_ptr, num_tokens_per_rdma_rank_ptr,
                num_tokens_per_expert_ptr, is_token_in_rank_ptr, prev, async,
                allocate_on_comm_stream, compute_stream_ptr);
          },
          nb::arg("topk_idx_ptr"), nb::arg("num_tokens"), nb::arg("num_topk"),
          nb::arg("num_experts"), nb::arg("num_tokens_per_rank_ptr"),
          nb::arg("num_tokens_per_rdma_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("previous_event") = nb::none(), nb::arg("async") = false,
          nb::arg("allocate_on_comm_stream") = false,
          nb::arg("compute_stream_ptr") = 0)
      .def(
          "intranode_prepare",
          [](Buffer& self, std::uintptr_t num_tokens_per_rank_ptr,
             std::uintptr_t is_token_in_rank_ptr,
             std::uintptr_t num_tokens_per_expert_ptr, int num_tokens,
             int num_experts, std::uintptr_t rank_prefix_matrix_ptr,
             std::uintptr_t channel_prefix_matrix_ptr, int expert_alignment,
             int num_worst_tokens, uccl::Config const& config,
             nb::object previous_event, bool async,
             bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
            std::optional<EventHandle> prev;
            if (!previous_event.is_none()) {
              EventHandle ev = nb::cast<EventHandle>(previous_event);
              prev = ev;
            }
            return self.intranode_prepare(
                num_tokens_per_rank_ptr, is_token_in_rank_ptr,
                num_tokens_per_expert_ptr, num_tokens, num_experts,
                rank_prefix_matrix_ptr, channel_prefix_matrix_ptr,
                expert_alignment, num_worst_tokens, config, prev, async,
                allocate_on_comm_stream, compute_stream_ptr);
          },
          nb::arg("num_tokens_per_rank_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("num_tokens"),
          nb::arg("num_experts"), nb::arg("rank_prefix_matrix_ptr"),
          nb::arg("channel_prefix_matrix_ptr"), nb::arg("expert_alignment"),
          nb::arg("num_worst_tokens"), nb::arg("config"),
          nb::arg("previous_event") = nb::none(), nb::arg("async") = false,
          nb::arg("allocate_on_comm_stream") = false,
          nb::arg("compute_stream_ptr") = 0)
      .def(
          "intranode_dispatch",
          [](Buffer& self, std::uintptr_t x_ptr, int num_tokens, int hidden,
             int x_element_size, std::uintptr_t x_scales_ptr, int num_scales,
             int scale_token_stride, int scale_hidden_stride,
             std::uintptr_t topk_idx_ptr, int num_topk,
             std::uintptr_t topk_weights_ptr,
             std::uintptr_t is_token_in_rank_ptr,
             std::uintptr_t rank_prefix_matrix_ptr,
             std::uintptr_t channel_prefix_matrix_ptr, int num_experts,
             int num_worst_tokens, bool cached_mode, uccl::Config const& config,
             int num_recv_tokens, std::uintptr_t recv_x_ptr,
             std::uintptr_t recv_x_scales_ptr, std::uintptr_t recv_topk_idx_ptr,
             std::uintptr_t recv_topk_weights_ptr,
             std::uintptr_t recv_channel_prefix_matrix_ptr,
             std::uintptr_t recv_src_idx_ptr, std::uintptr_t send_head_ptr,
             nb::object previous_event, bool async,
             bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
            std::optional<EventHandle> prev;
            if (!previous_event.is_none()) {
              EventHandle ev = nb::cast<EventHandle>(previous_event);
              prev = ev;
            }
            return self.intranode_dispatch(
                x_ptr, num_tokens, hidden, x_element_size, x_scales_ptr,
                num_scales, scale_token_stride, scale_hidden_stride,
                topk_idx_ptr, num_topk, topk_weights_ptr, is_token_in_rank_ptr,
                rank_prefix_matrix_ptr, channel_prefix_matrix_ptr, num_experts,
                num_worst_tokens, cached_mode, config, num_recv_tokens,
                recv_x_ptr, recv_x_scales_ptr, recv_topk_idx_ptr,
                recv_topk_weights_ptr, recv_channel_prefix_matrix_ptr,
                recv_src_idx_ptr, send_head_ptr, prev, async,
                allocate_on_comm_stream, compute_stream_ptr);
          },
          nb::arg("x_ptr"), nb::arg("num_tokens"), nb::arg("hidden"),
          nb::arg("x_element_size"), nb::arg("x_scales_ptr"),
          nb::arg("num_scales"), nb::arg("scale_token_stride"),
          nb::arg("scale_hidden_stride"), nb::arg("topk_idx_ptr"),
          nb::arg("num_topk"), nb::arg("topk_weights_ptr"),
          nb::arg("is_token_in_rank_ptr"), nb::arg("rank_prefix_matrix_ptr"),
          nb::arg("channel_prefix_matrix_ptr"), nb::arg("num_experts"),
          nb::arg("num_worst_tokens"), nb::arg("cached_mode"),
          nb::arg("config"), nb::arg("num_recv_tokens"), nb::arg("recv_x_ptr"),
          nb::arg("recv_x_scales_ptr"), nb::arg("recv_topk_idx_ptr"),
          nb::arg("recv_topk_weights_ptr"),
          nb::arg("recv_channel_prefix_matrix_ptr"),
          nb::arg("recv_src_idx_ptr"), nb::arg("send_head_ptr"),
          nb::arg("previous_event") = nb::none(), nb::arg("async") = false,
          nb::arg("allocate_on_comm_stream") = false,
          nb::arg("compute_stream_ptr") = 0)
      .def(
          "intranode_combine",
          [](Buffer& self, std::uintptr_t x_ptr, int num_tokens, int hidden,
             int x_dtype_code, int x_element_size,
             std::uintptr_t topk_weights_ptr, int num_topk,
             std::uintptr_t bias_0_ptr, std::uintptr_t bias_1_ptr,
             std::uintptr_t src_idx_ptr, int num_recv_tokens,
             std::uintptr_t rank_prefix_matrix_ptr,
             std::uintptr_t channel_prefix_matrix_ptr,
             std::uintptr_t send_head_ptr, uccl::Config const& config,
             std::uintptr_t recv_x_ptr, std::uintptr_t recv_topk_weights_ptr,
             nb::object previous_event, bool async,
             bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
            std::optional<EventHandle> prev;
            if (!previous_event.is_none()) {
              EventHandle ev = nb::cast<EventHandle>(previous_event);
              prev = ev;
            }
            return self.intranode_combine(
                x_ptr, num_tokens, hidden, x_dtype_code, x_element_size,
                topk_weights_ptr, num_topk, bias_0_ptr, bias_1_ptr, src_idx_ptr,
                num_recv_tokens, rank_prefix_matrix_ptr,
                channel_prefix_matrix_ptr, send_head_ptr, config, recv_x_ptr,
                recv_topk_weights_ptr, prev, async, allocate_on_comm_stream,
                compute_stream_ptr);
          },
          nb::arg("x_ptr"), nb::arg("num_tokens"), nb::arg("hidden"),
          nb::arg("x_dtype_code"), nb::arg("x_element_size"),
          nb::arg("topk_weights_ptr"), nb::arg("num_topk"),
          nb::arg("bias_0_ptr"), nb::arg("bias_1_ptr"), nb::arg("src_idx_ptr"),
          nb::arg("num_recv_tokens"), nb::arg("rank_prefix_matrix_ptr"),
          nb::arg("channel_prefix_matrix_ptr"), nb::arg("send_head_ptr"),
          nb::arg("config"), nb::arg("recv_x_ptr"),
          nb::arg("recv_topk_weights_ptr"),
          nb::arg("previous_event") = nb::none(), nb::arg("async") = false,
          nb::arg("allocate_on_comm_stream") = false,
          nb::arg("compute_stream_ptr") = 0)
      .def(
          "internode_prepare",
          [](Buffer& self, std::uintptr_t num_tokens_per_rank_ptr,
             std::uintptr_t num_tokens_per_rdma_rank_ptr,
             std::uintptr_t num_tokens_per_expert_ptr,
             std::uintptr_t is_token_in_rank_ptr, int num_tokens, int hidden,
             int x_element_size, int num_scales, int num_topk, int num_experts,
             int expert_alignment, int num_worst_tokens,
             uccl::Config const& config,
             std::uintptr_t rdma_channel_prefix_matrix_ptr,
             std::uintptr_t recv_rdma_rank_prefix_sum_ptr,
             std::uintptr_t gbl_channel_prefix_matrix_ptr,
             std::uintptr_t recv_gbl_rank_prefix_sum_ptr,
             nb::object previous_event, bool async,
             bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
            std::optional<EventHandle> prev;
            if (!previous_event.is_none()) {
              EventHandle ev = nb::cast<EventHandle>(previous_event);
              prev = ev;
            }
            return self.internode_prepare(
                num_tokens_per_rank_ptr, num_tokens_per_rdma_rank_ptr,
                num_tokens_per_expert_ptr, is_token_in_rank_ptr, num_tokens,
                hidden, x_element_size, num_scales, num_topk, num_experts,
                expert_alignment, num_worst_tokens, config,
                rdma_channel_prefix_matrix_ptr, recv_rdma_rank_prefix_sum_ptr,
                gbl_channel_prefix_matrix_ptr, recv_gbl_rank_prefix_sum_ptr,
                prev, async, allocate_on_comm_stream, compute_stream_ptr);
          },
          nb::arg("num_tokens_per_rank_ptr"),
          nb::arg("num_tokens_per_rdma_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("x_element_size"),
          nb::arg("num_scales"), nb::arg("num_topk"), nb::arg("num_experts"),
          nb::arg("expert_alignment"), nb::arg("num_worst_tokens"),
          nb::arg("config"), nb::arg("rdma_channel_prefix_matrix_ptr"),
          nb::arg("recv_rdma_rank_prefix_sum_ptr"),
          nb::arg("gbl_channel_prefix_matrix_ptr"),
          nb::arg("recv_gbl_rank_prefix_sum_ptr"),
          nb::arg("previous_event") = nb::none(), nb::arg("async") = false,
          nb::arg("allocate_on_comm_stream") = false,
          nb::arg("compute_stream_ptr") = 0)
      .def(
          "internode_dispatch",
          [](Buffer& self, std::uintptr_t x_ptr, int num_tokens, int hidden,
             int x_element_size, std::uintptr_t x_scales_ptr, int num_scales,
             int scale_token_stride, int scale_hidden_stride,
             std::uintptr_t topk_idx_ptr, int num_topk,
             std::uintptr_t topk_weights_ptr,
             std::uintptr_t is_token_in_rank_ptr,
             std::uintptr_t rdma_channel_prefix_matrix_ptr,
             std::uintptr_t recv_rdma_rank_prefix_sum_ptr,
             std::uintptr_t gbl_channel_prefix_matrix_ptr,
             std::uintptr_t recv_gbl_rank_prefix_sum_ptr, int num_experts,
             int num_worst_tokens, bool cached_mode, int num_rdma_recv_tokens,
             uccl::Config const& config, std::uintptr_t recv_x_ptr,
             std::uintptr_t recv_x_scales_ptr, std::uintptr_t recv_topk_idx_ptr,
             std::uintptr_t recv_topk_weights_ptr,
             std::uintptr_t recv_src_meta_ptr,
             std::uintptr_t recv_rdma_channel_prefix_matrix_ptr,
             std::uintptr_t recv_gbl_channel_prefix_matrix_ptr,
             std::uintptr_t send_rdma_head_ptr,
             std::uintptr_t send_nvl_head_ptr, nb::object previous_event,
             bool async, bool allocate_on_comm_stream,
             std::uintptr_t compute_stream_ptr) {
            std::optional<EventHandle> prev;
            if (!previous_event.is_none()) {
              EventHandle ev = nb::cast<EventHandle>(previous_event);
              prev = ev;
            }
            return self.internode_dispatch(
                x_ptr, num_tokens, hidden, x_element_size, x_scales_ptr,
                num_scales, scale_token_stride, scale_hidden_stride,
                topk_idx_ptr, num_topk, topk_weights_ptr, is_token_in_rank_ptr,
                rdma_channel_prefix_matrix_ptr, recv_rdma_rank_prefix_sum_ptr,
                gbl_channel_prefix_matrix_ptr, recv_gbl_rank_prefix_sum_ptr,
                num_experts, num_worst_tokens, cached_mode,
                num_rdma_recv_tokens, config, recv_x_ptr, recv_x_scales_ptr,
                recv_topk_idx_ptr, recv_topk_weights_ptr, recv_src_meta_ptr,
                recv_rdma_channel_prefix_matrix_ptr,
                recv_gbl_channel_prefix_matrix_ptr, send_rdma_head_ptr,
                send_nvl_head_ptr, prev, async, allocate_on_comm_stream,
                compute_stream_ptr);
          },
          nb::arg("x_ptr"), nb::arg("num_tokens"), nb::arg("hidden"),
          nb::arg("x_element_size"), nb::arg("x_scales_ptr"),
          nb::arg("num_scales"), nb::arg("scale_token_stride"),
          nb::arg("scale_hidden_stride"), nb::arg("topk_idx_ptr"),
          nb::arg("num_topk"), nb::arg("topk_weights_ptr"),
          nb::arg("is_token_in_rank_ptr"),
          nb::arg("rdma_channel_prefix_matrix_ptr"),
          nb::arg("recv_rdma_rank_prefix_sum_ptr"),
          nb::arg("gbl_channel_prefix_matrix_ptr"),
          nb::arg("recv_gbl_rank_prefix_sum_ptr"), nb::arg("num_experts"),
          nb::arg("num_worst_tokens"), nb::arg("cached_mode"),
          nb::arg("num_rdma_recv_tokens"), nb::arg("config"),
          nb::arg("recv_x_ptr"), nb::arg("recv_x_scales_ptr"),
          nb::arg("recv_topk_idx_ptr"), nb::arg("recv_topk_weights_ptr"),
          nb::arg("recv_src_meta_ptr"),
          nb::arg("recv_rdma_channel_prefix_matrix_ptr"),
          nb::arg("recv_gbl_channel_prefix_matrix_ptr"),
          nb::arg("send_rdma_head_ptr"), nb::arg("send_nvl_head_ptr"),
          nb::arg("previous_event") = nb::none(), nb::arg("async") = false,
          nb::arg("allocate_on_comm_stream") = false,
          nb::arg("compute_stream_ptr") = 0)
      .def(
          "internode_combine",
          [](Buffer& self, std::uintptr_t x_ptr, int num_tokens, int hidden,
             int x_dtype_code, int x_element_size,
             std::uintptr_t topk_weights_ptr, int num_topk,
             std::uintptr_t bias_0_ptr, std::uintptr_t bias_1_ptr,
             std::uintptr_t src_meta_ptr, int num_combined_tokens,
             std::uintptr_t is_combined_token_in_rank_ptr,
             std::uintptr_t rdma_channel_prefix_matrix_ptr,
             std::uintptr_t rdma_rank_prefix_sum_ptr,
             std::uintptr_t gbl_channel_prefix_matrix_ptr,
             std::uintptr_t combined_rdma_head_ptr,
             std::uintptr_t combined_nvl_head_ptr, uccl::Config const& config,
             std::uintptr_t combined_x_ptr,
             std::uintptr_t combined_topk_weights_ptr,
             nb::object previous_event, bool async,
             bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr) {
            std::optional<EventHandle> prev;
            if (!previous_event.is_none()) {
              EventHandle ev = nb::cast<EventHandle>(previous_event);
              prev = ev;
            }
            return self.internode_combine(
                x_ptr, num_tokens, hidden, x_dtype_code, x_element_size,
                topk_weights_ptr, num_topk, bias_0_ptr, bias_1_ptr,
                src_meta_ptr, num_combined_tokens,
                is_combined_token_in_rank_ptr, rdma_channel_prefix_matrix_ptr,
                rdma_rank_prefix_sum_ptr, gbl_channel_prefix_matrix_ptr,
                combined_rdma_head_ptr, combined_nvl_head_ptr, config,
                combined_x_ptr, combined_topk_weights_ptr, prev, async,
                allocate_on_comm_stream, compute_stream_ptr);
          },
          nb::arg("x_ptr"), nb::arg("num_tokens"), nb::arg("hidden"),
          nb::arg("x_dtype_code"), nb::arg("x_element_size"),
          nb::arg("topk_weights_ptr"), nb::arg("num_topk"),
          nb::arg("bias_0_ptr"), nb::arg("bias_1_ptr"), nb::arg("src_meta_ptr"),
          nb::arg("num_combined_tokens"),
          nb::arg("is_combined_token_in_rank_ptr"),
          nb::arg("rdma_channel_prefix_matrix_ptr"),
          nb::arg("rdma_rank_prefix_sum_ptr"),
          nb::arg("gbl_channel_prefix_matrix_ptr"),
          nb::arg("combined_rdma_head_ptr"), nb::arg("combined_nvl_head_ptr"),
          nb::arg("config"), nb::arg("combined_x_ptr"),
          nb::arg("combined_topk_weights_ptr"),
          nb::arg("previous_event") = nb::none(), nb::arg("async") = false,
          nb::arg("allocate_on_comm_stream") = false,
          nb::arg("compute_stream_ptr") = 0)
      .def("clean_low_latency_buffer", &Buffer::clean_low_latency_buffer,
           nb::arg("num_max_dispatch_tokens_per_rank"), nb::arg("hidden"),
           nb::arg("num_experts"), nb::arg("stream_ptr"))
      .def("low_latency_combine", &Buffer::low_latency_combine,
           nb::arg("x_ptr"), nb::arg("x_dim0"), nb::arg("x_dim1"),
           nb::arg("x_dim2"), nb::arg("topk_idx_ptr"), nb::arg("topk_rows"),
           nb::arg("topk_cols"), nb::arg("topk_weights_ptr"),
           nb::arg("src_info_ptr"), nb::arg("src_info_dim0"),
           nb::arg("src_info_dim1"), nb::arg("layout_range_ptr"),
           nb::arg("layout_range_dim0"), nb::arg("layout_range_dim1"),
           nb::arg("combine_wait_recv_cost_stats_ptr") = 0,
           nb::arg("compute_stream_ptr"),
           nb::arg("num_max_dispatch_tokens_per_rank") = 0,
           nb::arg("num_experts") = 1, nb::arg("use_logfmt") = false,
           nb::arg("zero_copy") = false, nb::arg("is_async") = false,
           nb::arg("return_recv_hook") = false, nb::arg("out_ptr"));
  m.def("alloc_cmd_ring", &alloc_cmd_ring);
  m.def("free_cmd_ring", &free_cmd_ring);
  m.def("launch_gpu_issue_kernel", [](int blocks, int threads_per_block,
                                      uintptr_t stream_ptr, uintptr_t rb_ptr) {
    size_t const shmem_bytes = kQueueSize * 2 * sizeof(unsigned long long);
    auto* stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto* rbs = reinterpret_cast<DeviceToHostCmdBuffer*>(rb_ptr);
    auto st = launch_gpu_issue_batched_commands_shim(blocks, threads_per_block,
                                                     shmem_bytes, stream, rbs);
    if (st != cudaSuccess) {
      throw std::runtime_error("Kernel launch failed: " +
                               std::string(cudaGetErrorString(st)));
    }
  });
  m.def("get_low_latency_rdma_size_hint",
        &uccl::get_low_latency_rdma_size_hint);
  m.def("sync_stream", []() {
    auto st = cudaDeviceSynchronize();
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") +
                               cudaGetErrorString(st));
  });
  m.def("set_device", [](int dev) {
    auto st = cudaSetDevice(dev);
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                               cudaGetErrorString(st));
  });
  m.def("get_device", []() {
    int dev;
    auto st = cudaGetDevice(&dev);
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                               cudaGetErrorString(st));
    return dev;
  });
  m.def("check_stream", [](uintptr_t stream_ptr) {
    auto* s = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaError_t st = cudaStreamQuery(s);
    return std::string(cudaGetErrorString(st));
  });
  m.def("is_sm90_compiled", is_sm90_compiled);
  m.def("get_num_proxy_threads", []() { return kNumProxyThs; });
  m.def(
      "stream_query",
      [](uintptr_t stream_ptr) {
        auto* stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        auto st = cudaStreamQuery(stream);
        if (st == cudaSuccess) return std::string("done");
        if (st == cudaErrorNotReady) return std::string("not_ready");
        return std::string("error: ") + cudaGetErrorString(st);
      },
      nb::arg("stream_ptr"));
  m.def("device_reset", []() {
    auto st = cudaDeviceReset();
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaDeviceReset failed: ") +
                               cudaGetErrorString(st));
  });
  nb::class_<Stats>(m, "Stats");
  nb::class_<UcclProxy>(m, "Proxy")
      .def(nb::init<int, uintptr_t, size_t, int, int, int, int, int, int, bool,
                    bool, bool>(),
           nb::arg("thread_idx"), nb::arg("gpu_buffer_addr"),
           nb::arg("total_size"), nb::arg("rank") = 0, nb::arg("node_idx") = -1,
           nb::arg("local_rank") = 0, nb::arg("num_experts") = -1,
           nb::arg("num_ranks") = -1, nb::arg("num_nodes") = 0,
           nb::arg("use_normal_mode") = false, nb::arg("is_intranode") = false,
           nb::arg("gpu_buffer_is_host_allocated") = false)
      .def("start_sender", &UcclProxy::start_sender)
      .def("start_remote", &UcclProxy::start_remote)
      .def("start_local", &UcclProxy::start_local)
      .def("start_dual", &UcclProxy::start_dual)
      .def("stop", &UcclProxy::stop)
      .def("get_listen_port", &UcclProxy::get_listen_port)
      .def("get_atomic_buffer_ptr", &UcclProxy::get_atomic_buffer_ptr)
      .def("set_atomic_buffer_ptr", &UcclProxy::set_atomic_buffer_ptr)
      .def("set_dispatch_recv_data_offset",
           &UcclProxy::set_dispatch_recv_data_offset, nb::arg("offset"))
      .def("calculate_and_set_dispatch_recv_data_offset",
           &UcclProxy::calculate_and_set_dispatch_recv_data_offset,
           nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_experts"))
      .def("get_d2h_channel_addrs", &UcclProxy::get_d2h_channel_addrs)
      .def_prop_ro("thread_idx", &UcclProxy::thread_idx)
      .def_prop_ro("gpu_buffer_addr", &UcclProxy::gpu_buffer_addr)
      .def("avg_rdma_write_us", &UcclProxy::avg_rdma_write_us)
      .def("avg_wr_latency_us", &UcclProxy::avg_wr_latency_us)
      .def(
          "set_peers_meta",
          [](UcclProxy& self, nb::object metas) {
            std::vector<PeerMeta> v;
            if (nb::isinstance<nb::list>(metas)) {
              for (auto obj : nb::cast<nb::list>(metas)) {
                if (nb::isinstance<nb::dict>(obj)) {
                  auto d = nb::cast<nb::dict>(obj);
                  PeerMeta pm;
                  pm.rank = nb::cast<int>(d["rank"]);
                  pm.ptr = static_cast<uintptr_t>(
                      nb::cast<unsigned long long>(d["ptr"]));
                  pm.nbytes = static_cast<size_t>(
                      nb::cast<unsigned long long>(d["nbytes"]));
                  pm.ip = nb::cast<std::string>(d["ip"]);

                  // Handle listen_ports array (always present)
                  auto ports = nb::cast<nb::list>(d["listen_ports"]);
                  size_t port_count =
                      std::min(static_cast<size_t>(nb::len(ports)),
                               static_cast<size_t>(kNumProxyThs));
                  for (size_t i = 0; i < port_count; ++i) {
                    pm.listen_ports[i] = nb::cast<int>(ports[i]);
                  }
                  // Initialize remaining ports to 0 if fewer than kNumProxyThs
                  // provided
                  for (size_t i = port_count; i < kNumProxyThs; ++i) {
                    pm.listen_ports[i] = 0;
                  }

                  v.push_back(std::move(pm));
                } else {
                  v.push_back(nb::cast<PeerMeta>(obj));
                }
              }
            } else {
              // allow passing a dict directly
              auto d = nb::cast<nb::dict>(metas);
              PeerMeta pm;
              pm.rank = nb::cast<int>(d["rank"]);
              pm.ptr = static_cast<uintptr_t>(
                  nb::cast<unsigned long long>(d["ptr"]));
              pm.nbytes = static_cast<size_t>(
                  nb::cast<unsigned long long>(d["nbytes"]));
              pm.ip = nb::cast<std::string>(d["ip"]);

              // Handle listen_ports array (always present)
              auto ports = nb::cast<nb::list>(d["listen_ports"]);
              size_t port_count = std::min(static_cast<size_t>(nb::len(ports)),
                                           static_cast<size_t>(kNumProxyThs));
              for (size_t i = 0; i < port_count; ++i) {
                pm.listen_ports[i] = nb::cast<int>(ports[i]);
              }
              // Initialize remaining ports to 0 if fewer than kNumProxyThs
              // provided
              for (size_t i = port_count; i < kNumProxyThs; ++i) {
                pm.listen_ports[i] = 0;
              }

              v.push_back(std::move(pm));
            }
            self.set_peers_meta(v);
          },
          nb::arg("metas"),
          "Attach peer metadata (list of dicts or PeerMeta objects).")
      .def(
          "set_bench_d2h_channel_addrs",
          [](UcclProxy& self, nb::iterable addrs) {
            std::vector<uintptr_t> v;
            for (nb::handle h : addrs) v.push_back(nb::cast<uintptr_t>(h));
            self.set_bench_d2h_channel_addrs(v);
          },
          nb::arg("addrs"), "Attach ring buffer addresses for benchmarking.");
  // .def_prop_ro("gpu_buffer_addr", &UcclProxy::gpu_buffer_addr);
  nb::class_<EnvInfo>(m, "EnvInfo")
      .def_ro("blocks", &EnvInfo::blocks)
      .def_ro("queue_size", &EnvInfo::queue_size)
      .def_ro("threads_per_block", &EnvInfo::threads_per_block)
      .def_ro("iterations", &EnvInfo::iterations)
      .def_ro("stream_addr", &EnvInfo::stream_addr)
      .def_ro("rbs_addr", &EnvInfo::rbs_addr);
  nb::class_<Bench>(m, "Bench")
      .def(nb::init<>())
      .def("env_info", &Bench::env_info)
      .def("blocks", &Bench::blocks)
      .def("num_proxies", &Bench::num_proxies)
      .def("ring_addr", &Bench::ring_addr)
      .def("timing_start", &Bench::timing_start)
      .def("timing_stop", &Bench::timing_stop)
      .def("is_running", &Bench::is_running)
      .def("launch_gpu_issue_batched_commands",
           &Bench::launch_gpu_issue_batched_commands)
      .def("sync_stream", &Bench::sync_stream)
      .def("sync_stream_interruptible", &Bench::sync_stream_interruptible,
           nb::arg("poll_ms") = 5, nb::arg("timeout_ms") = -1,
           nb::arg("should_abort") = nullptr)
      .def("join_proxies", &Bench::join_proxies)
      .def("print_block_latencies", &Bench::print_block_latencies)
      .def("compute_stats", &Bench::compute_stats)
      .def("print_summary", &Bench::print_summary)
      .def("print_summary_last", &Bench::print_summary_last)
      .def("last_elapsed_ms", &Bench::last_elapsed_ms);

  // MSCCLPP Fifo class - must be registered before BenchFifo which uses it
  // Bind init<int> to avoid narrowing conversion warning (Python int -> C++
  // uint32_t)
  nb::class_<mscclpp::Fifo>(m, "Fifo").def(nb::init<int>(),
                                           nb::arg("size") = 2048);

  // FIFO-based benchmarking classes
  nb::class_<BenchFifo>(m, "BenchFifo")
      .def(nb::init<>())
      .def("env_info", &BenchFifo::env_info)
      .def("blocks", &BenchFifo::blocks)
      .def("num_proxies", &BenchFifo::num_proxies)
      .def("get_fifo", &BenchFifo::get_fifo, nb::rv_policy::reference)
      .def("timing_start", &BenchFifo::timing_start)
      .def("timing_stop", &BenchFifo::timing_stop)
      .def("is_running", &BenchFifo::is_running)
      .def("launch_gpu_issue_batched_commands",
           &BenchFifo::launch_gpu_issue_batched_commands)
      .def("sync_stream", &BenchFifo::sync_stream)
      .def("sync_stream_interruptible", &BenchFifo::sync_stream_interruptible,
           nb::arg("poll_ms") = 5, nb::arg("timeout_ms") = -1,
           nb::arg("should_abort") = nullptr)
      .def("join_proxies", &BenchFifo::join_proxies)
      .def("print_block_latencies", &BenchFifo::print_block_latencies)
      .def("compute_stats", &BenchFifo::compute_stats)
      .def("print_summary", &BenchFifo::print_summary)
      .def("print_summary_last", &BenchFifo::print_summary_last)
      .def("last_elapsed_ms", &BenchFifo::last_elapsed_ms);

  nb::class_<FifoProxy>(m, "FifoProxy")
      .def(nb::init<int, uintptr_t, size_t, int, int, int, bool>(),
           nb::arg("thread_idx"), nb::arg("gpu_buffer_addr"),
           nb::arg("total_size"), nb::arg("rank"), nb::arg("node_idx"),
           nb::arg("local_rank"), nb::arg("is_intranode"))
      .def("set_fifo", &FifoProxy::set_fifo, nb::arg("fifo"))
      .def("set_peers_meta",
           [](FifoProxy& proxy, nb::list meta_list) {
             std::vector<PeerMeta> vec;
             for (nb::handle h : meta_list) {
               if (nb::isinstance<nb::dict>(h)) {
                 auto d = nb::cast<nb::dict>(h);
                 PeerMeta pm;
                 pm.rank = nb::cast<int>(d["rank"]);
                 pm.ptr = nb::cast<uintptr_t>(d["ptr"]);
                 pm.nbytes = nb::cast<size_t>(d["nbytes"]);
                 pm.ip = nb::cast<std::string>(d["ip"]);

                 // Handle listen_ports array (always present)
                 auto ports = nb::cast<nb::list>(d["listen_ports"]);
                 size_t port_count =
                     std::min(static_cast<size_t>(nb::len(ports)),
                              static_cast<size_t>(kNumProxyThs));
                 for (size_t i = 0; i < port_count; ++i) {
                   pm.listen_ports[i] = nb::cast<int>(ports[i]);
                 }
                 // Initialize remaining ports to 0 if fewer than kNumProxyThs
                 // provided
                 for (size_t i = port_count; i < kNumProxyThs; ++i) {
                   pm.listen_ports[i] = 0;
                 }

                 vec.push_back(std::move(pm));
               } else {
                 vec.push_back(nb::cast<PeerMeta>(h));
               }
             }
             proxy.set_peers_meta(vec);
           })
      .def("start_sender", &FifoProxy::start_sender)
      .def("start_remote", &FifoProxy::start_remote)
      .def("stop", &FifoProxy::stop)
      .def("get_listen_port", &FifoProxy::get_listen_port)
      .def("avg_wr_latency_us", &FifoProxy::avg_wr_latency_us)
      .def("processed_count", &FifoProxy::processed_count)
      .def_ro("thread_idx", &FifoProxy::thread_idx);
}

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
#include "peer_copy_manager.hpp"
#include "ring_buffer.cuh"
#include "uccl_bench.hpp"
#include "uccl_proxy.hpp"
#include "rdma.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

namespace uccl {
std::unordered_map<int, std::vector<py::object>> g_proxies_by_dev;

std::unordered_map<int, std::vector<py::object>>& proxies_by_dev() {
  return g_proxies_by_dev;
}
}  // namespace uccl

#define NUM_MAX_LOCAL_EXPERTS 1024

namespace py = pybind11;

static std::mutex g_proxies_mu;

struct EventOverlap {};
struct Ctx {
  long num_tokens{0};
  long hidden{0};
};
static std::atomic<long> g_next{1};
static std::mutex g_mu;
static std::unordered_map<long, Ctx> g_ctx;

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
        proxy.attr("get_d2h_channel_addrs")().cast<std::vector<uint64_t>>();
    all_addrs.insert(all_addrs.end(), proxy_addrs.begin(), proxy_addrs.end());
  }
  return all_addrs;
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
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
        explicitly_destroy(explicitly_destroy),
        comm_stream(at::cuda::getStreamFromPool(/*isHighPriority=*/true)) {
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
              loc, 0, 0));
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

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    fp8_dtype = strcmp(prop.gcnArchName, "gfx942") == 0
                    ? torch::kFloat8_e4m3fnuz
                    : torch::kFloat8_e4m3fn;
#endif
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

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor,
             torch::Tensor, std::optional<EventHandle>>
  get_dispatch_layout(torch::Tensor const& topk_idx, int num_experts,
                      std::optional<EventHandle>& previous_event, bool async,
                      bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
      EP_HOST_ASSERT(previous_event.has_value() and async);
      at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)),
         num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank =
        torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert =
        torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty(
        {num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
    if (is_internode_available())
      num_tokens_per_rdma_rank = torch::empty(
          {num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    uccl::layout::get_dispatch_layout(
        topk_idx.data_ptr<int64_t>(), num_tokens_per_rank.data_ptr<int>(),
        num_tokens_per_rdma_rank.has_value()
            ? num_tokens_per_rdma_rank.value().data_ptr<int>()
            : nullptr,
        num_tokens_per_expert.data_ptr<int>(),
        is_token_in_rank.data_ptr<bool>(), num_tokens, num_topk, num_ranks,
        num_experts, comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
      for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert,
                      is_token_in_rank}) {
        t.record_stream(comm_stream);
        if (allocate_on_comm_stream) t.record_stream(compute_stream);
      }
      for (auto& to : {num_tokens_per_rdma_rank}) {
        to.has_value() ? to->record_stream(comm_stream) : void();
        if (allocate_on_comm_stream)
          to.has_value() ? to->record_stream(compute_stream) : void();
      }
    } else {
      stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank,
            num_tokens_per_expert, is_token_in_rank, event};
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
    destroyed = true;
    available = false;
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>,
             std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::vector<int>, torch::Tensor, torch::Tensor,
             std::optional<torch::Tensor>, torch::Tensor,
             std::optional<torch::Tensor>, torch::Tensor,
             std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::optional<torch::Tensor>, std::optional<EventHandle>>
  internode_dispatch(
      torch::Tensor const& x, std::optional<torch::Tensor> const& x_scales,
      std::optional<torch::Tensor> const& topk_idx,
      std::optional<torch::Tensor> const& topk_weights,
      std::optional<torch::Tensor> const& num_tokens_per_rank,
      std::optional<torch::Tensor> const& num_tokens_per_rdma_rank,
      torch::Tensor const& is_token_in_rank,
      std::optional<torch::Tensor> const& num_tokens_per_expert,
      int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
      std::optional<torch::Tensor> const& cached_rdma_channel_prefix_matrix,
      std::optional<torch::Tensor> const& cached_recv_rdma_rank_prefix_sum,
      std::optional<torch::Tensor> const& cached_gbl_channel_prefix_matrix,
      std::optional<torch::Tensor> const& cached_recv_gbl_rank_prefix_sum,
      int expert_alignment, int num_worst_tokens, uccl::Config const& config,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream) {
    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata
    // from other ranks, which can be quite long. If users of DeepEP need to
    // execute other Python code on other threads, such as KV transfer, their
    // code will get stuck due to GIL unless we release GIL here.
    pybind11::gil_scoped_release release;

    int const num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and
                   get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
      EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
      EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
      EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
      EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
      EP_HOST_ASSERT(num_tokens_per_rank.has_value());
      EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
      EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
      EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() ==
                     torch::kInt32);
      EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() ==
                     torch::kInt32);
      EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() ==
                     torch::kInt32);
      EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() ==
                     torch::kInt32);
    } else {
      EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
      EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
      EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
      EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and
                     cached_rdma_channel_prefix_matrix->is_contiguous());
      EP_HOST_ASSERT(
          cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
          cached_rdma_channel_prefix_matrix->size(1) == num_channels);
      EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and
                     cached_recv_rdma_rank_prefix_sum->is_contiguous());
      EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) ==
                     num_rdma_ranks);
      EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and
                     cached_gbl_channel_prefix_matrix->is_contiguous());
      EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                     cached_gbl_channel_prefix_matrix->size(1) == num_channels);
      EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and
                     cached_recv_gbl_rank_prefix_sum->is_contiguous());
      EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
      EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and
                     num_tokens_per_rank->is_contiguous());
      EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and
                     num_tokens_per_rdma_rank->is_contiguous());
      EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and
                     num_tokens_per_expert->is_contiguous());
      EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
      EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
      EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
      EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <=
                     NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)),
         hidden = static_cast<int>(x.size(1)),
         hidden_int4 =
             static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts =
             cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
         num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int64_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
      num_topk = static_cast<int>(topk_idx->size(1));
      EP_HOST_ASSERT(num_experts > 0);
      EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
      EP_HOST_ASSERT(topk_weights->dim() == 2 and
                     topk_weights->is_contiguous());
      EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and
                     num_tokens == topk_weights->size(0));
      EP_HOST_ASSERT(num_topk == topk_weights->size(1));
      EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
      topk_idx_ptr = topk_idx->data_ptr<int64_t>();
      topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
      EP_HOST_ASSERT(x.element_size() == 1);
      EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or
                     x_scales->scalar_type() == torch::kInt);
      EP_HOST_ASSERT(x_scales->dim() == 2);
      EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
      num_scales =
          x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
      x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
      scale_token_stride = static_cast<int>(x_scales->stride(0));
      scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
      EP_HOST_ASSERT(previous_event.has_value() and async);
      at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
      num_recv_tokens = cached_num_recv_tokens;
      num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
      rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
      recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
      gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
      recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

      // Just a barrier and clean flags
      uccl::internode::cached_notify(
          hidden_int4, num_scales, num_topk, num_topk, num_ranks, num_channels,
          0, nullptr, nullptr, nullptr, nullptr, rdma_buffer_ptr,
          config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
          config.num_max_nvl_chunked_recv_tokens, barrier_signal_ptrs_gpu, rank,
          comm_stream,
          config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4),
                                           num_ranks),
          num_nvl_bytes, true, low_latency_mode, d_handles,
          num_d2h_channel_addrs, atomic_buffer_ptr);
    } else {
      rdma_channel_prefix_matrix =
          torch::empty({num_rdma_ranks, num_channels},
                       dtype(torch::kInt32).device(torch::kCUDA));
      recv_rdma_rank_prefix_sum = torch::empty(
          {num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
      gbl_channel_prefix_matrix = torch::empty(
          {num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
      recv_gbl_rank_prefix_sum =
          torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

      // Send sizes
      *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
      for (int i = 0; i < num_local_experts; ++i)
        moe_recv_expert_counter[i] = -1;
      uccl::internode::notify_dispatch(
          num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped,
          num_ranks, num_tokens_per_rdma_rank->data_ptr<int>(),
          moe_recv_rdma_counter_mapped, num_tokens_per_expert->data_ptr<int>(),
          moe_recv_expert_counter_mapped, num_experts,
          is_token_in_rank.data_ptr<bool>(), num_tokens, num_worst_tokens,
          num_channels, hidden_int4, num_scales, num_topk, expert_alignment,
          rdma_channel_prefix_matrix.data_ptr<int>(),
          recv_rdma_rank_prefix_sum.data_ptr<int>(),
          gbl_channel_prefix_matrix.data_ptr<int>(),
          recv_gbl_rank_prefix_sum.data_ptr<int>(), rdma_buffer_ptr,
          config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
          config.num_max_nvl_chunked_recv_tokens, barrier_signal_ptrs_gpu, rank,
          comm_stream,
          config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4),
                                           num_ranks),
          num_nvl_bytes, low_latency_mode, d_handles, num_d2h_channel_addrs,
          atomic_buffer_ptr);
      // Synchronize total received tokens and tokens per expert
      if (num_worst_tokens > 0) {
        num_recv_tokens = num_worst_tokens;
        num_rdma_recv_tokens = num_worst_tokens;
      } else {
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
          // Read total count
          num_recv_tokens = static_cast<int>(*moe_recv_counter);
          num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

          // Read per-expert count
          bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
          for (int i = 0; i < num_local_experts and ready; ++i)
            ready &= moe_recv_expert_counter[i] >= 0;

          if (ready) break;

          // Timeout check
          if (std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::high_resolution_clock::now() - start_time)
                  .count() > NUM_CPU_TIMEOUT_SECS) {
            printf(
                "Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: "
                "%d\n",
                rank, num_recv_tokens, num_rdma_recv_tokens);
            for (int i = 0; i < num_local_experts; ++i)
              printf("moe_recv_expert_counter[%d]: %d\n", i,
                     moe_recv_expert_counter[i]);
            throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
          }
        }
        num_recv_tokens_per_expert_list =
            std::vector<int>(moe_recv_expert_counter,
                             moe_recv_expert_counter + num_local_experts);
      }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(),
         recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
      recv_src_meta = torch::empty(
          {num_recv_tokens, uccl::internode::get_source_meta_bytes()},
          dtype(torch::kByte).device(torch::kCUDA));
      recv_rdma_channel_prefix_matrix =
          torch::empty({num_rdma_ranks, num_channels},
                       dtype(torch::kInt32).device(torch::kCUDA));
      recv_gbl_channel_prefix_matrix = torch::empty(
          {num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
      send_rdma_head = torch::empty({num_tokens, num_rdma_ranks},
                                    dtype(torch::kInt32).device(torch::kCUDA));
      send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS},
                                   dtype(torch::kInt32).device(torch::kCUDA));
    }

    // Assign pointers
    int64_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
      recv_topk_idx =
          torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
      recv_topk_weights =
          torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
      recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
      recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
      recv_x_scales = x_scales->dim() == 1
                          ? torch::empty({num_recv_tokens}, x_scales->options())
                          : torch::empty({num_recv_tokens, num_scales},
                                         x_scales->options());
      recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    uccl::internode::dispatch(
        recv_x.data_ptr(), recv_x_scales_ptr, recv_topk_idx_ptr,
        recv_topk_weights_ptr,
        cached_mode ? nullptr : recv_src_meta->data_ptr(), x.data_ptr(),
        x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
        cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
        cached_mode ? nullptr
                    : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
        rdma_channel_prefix_matrix.data_ptr<int>(),
        recv_rdma_rank_prefix_sum.data_ptr<int>(),
        gbl_channel_prefix_matrix.data_ptr<int>(),
        recv_gbl_rank_prefix_sum.data_ptr<int>(),
        is_token_in_rank.data_ptr<bool>(), num_tokens, num_worst_tokens,
        hidden_int4, num_scales, num_topk, num_experts, scale_token_stride,
        scale_hidden_stride, rdma_buffer_ptr,
        config.num_max_rdma_chunked_send_tokens,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens, rank, num_ranks, cached_mode,
        comm_stream, num_channels, low_latency_mode, d_handles,
        num_d2h_channel_addrs, atomic_buffer_ptr);
    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
      for (auto& t : {x, is_token_in_rank, recv_x, rdma_channel_prefix_matrix,
                      recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
                      recv_gbl_rank_prefix_sum}) {
        t.record_stream(comm_stream);
        if (allocate_on_comm_stream) t.record_stream(compute_stream);
      }
      for (auto& to :
           {x_scales, topk_idx, topk_weights, num_tokens_per_rank,
            num_tokens_per_rdma_rank, num_tokens_per_expert,
            cached_rdma_channel_prefix_matrix, cached_recv_rdma_rank_prefix_sum,
            cached_gbl_channel_prefix_matrix, cached_recv_gbl_rank_prefix_sum,
            recv_topk_idx, recv_topk_weights, recv_x_scales,
            recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix,
            send_rdma_head, send_nvl_head, recv_src_meta}) {
        to.has_value() ? to->record_stream(comm_stream) : void();
        if (allocate_on_comm_stream)
          to.has_value() ? to->record_stream(compute_stream) : void();
      }
    } else {
      stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
            event};
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>,
             std::optional<EventHandle>>
  internode_combine(torch::Tensor const& x,
                    std::optional<torch::Tensor> const& topk_weights,
                    std::optional<torch::Tensor> const& bias_0,
                    std::optional<torch::Tensor> const& bias_1,
                    torch::Tensor const& src_meta,
                    torch::Tensor const& is_combined_token_in_rank,
                    torch::Tensor const& rdma_channel_prefix_matrix,
                    torch::Tensor const& rdma_rank_prefix_sum,
                    torch::Tensor const& gbl_channel_prefix_matrix,
                    torch::Tensor const& combined_rdma_head,
                    torch::Tensor const& combined_nvl_head,
                    uccl::Config const& config,
                    std::optional<EventHandle>& previous_event, bool async,
                    bool allocate_on_comm_stream) {
    int const num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and
                   src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and
                   is_combined_token_in_rank.is_contiguous() and
                   is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and
                   rdma_channel_prefix_matrix.is_contiguous() and
                   rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and
                   rdma_rank_prefix_sum.is_contiguous() and
                   rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and
                   gbl_channel_prefix_matrix.is_contiguous() and
                   gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and
                   combined_rdma_head.is_contiguous() and
                   combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and
                   combined_nvl_head.is_contiguous() and
                   combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)),
         hidden = static_cast<int>(x.size(1)),
         hidden_int4 =
             static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens =
        static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) ==
                   uccl::internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and
                   rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and
                   gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and
                   combined_rdma_head.size(0) == num_combined_tokens and
                   combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and
                   combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
      EP_HOST_ASSERT(previous_event.has_value() and async);
      at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
      EP_HOST_ASSERT(topk_weights->dim() == 2 and
                     topk_weights->is_contiguous());
      EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
      EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
      num_topk = static_cast<int>(topk_weights->size(1));
      topk_weights_ptr = topk_weights->data_ptr<float>();
      combined_topk_weights = torch::empty({num_combined_tokens, num_topk},
                                           topk_weights->options());
      combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks ==
                   0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <=
                   config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    uccl::internode::cached_notify(
        hidden_int4, 0, 0, num_topk, num_ranks, num_channels,
        num_combined_tokens, combined_rdma_head.data_ptr<int>(),
        rdma_channel_prefix_matrix.data_ptr<int>(),
        rdma_rank_prefix_sum.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
        rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
        buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
        barrier_signal_ptrs_gpu, rank, comm_stream,
        config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
        num_nvl_bytes, false, low_latency_mode, d_handles,
        num_d2h_channel_addrs, atomic_buffer_ptr);
    // Assign bias pointers
    auto bias_opts =
        std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
      if (bias_opts[i].has_value()) {
        auto bias = bias_opts[i].value();
        EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
        EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and
                       bias.size(1) == hidden);
        bias_ptrs[i] = bias.data_ptr();
      }

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    uccl::internode::combine(
        at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
        combined_x.data_ptr(), combined_topk_weights_ptr,
        is_combined_token_in_rank.data_ptr<bool>(), x.data_ptr(),
        topk_weights_ptr, bias_ptrs[0], bias_ptrs[1],
        combined_rdma_head.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
        src_meta.data_ptr(), rdma_channel_prefix_matrix.data_ptr<int>(),
        rdma_rank_prefix_sum.data_ptr<int>(),
        gbl_channel_prefix_matrix.data_ptr<int>(), num_tokens,
        num_combined_tokens, hidden, num_topk, rdma_buffer_ptr,
        config.num_max_rdma_chunked_send_tokens,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens, rank, num_ranks, comm_stream,
        num_channels, low_latency_mode, d_handles, num_d2h_channel_addrs,
        atomic_buffer_ptr);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
      for (auto& t :
           {x, src_meta, is_combined_token_in_rank, rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum, gbl_channel_prefix_matrix, combined_x,
            combined_rdma_head, combined_nvl_head}) {
        t.record_stream(comm_stream);
        if (allocate_on_comm_stream) t.record_stream(compute_stream);
      }
      for (auto& to : {topk_weights, combined_topk_weights, bias_0, bias_1}) {
        to.has_value() ? to->record_stream(comm_stream) : void();
        if (allocate_on_comm_stream)
          to.has_value() ? to->record_stream(compute_stream) : void();
      }
    } else {
      stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>,
             std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor, torch::Tensor, std::optional<EventHandle>>
  intranode_dispatch(
      torch::Tensor const& x, std::optional<torch::Tensor> const& x_scales,
      std::optional<torch::Tensor> const& topk_idx,
      std::optional<torch::Tensor> const& topk_weights,
      std::optional<torch::Tensor> const& num_tokens_per_rank,
      torch::Tensor const& is_token_in_rank,
      std::optional<torch::Tensor> const& num_tokens_per_expert,
      int cached_num_recv_tokens,
      std::optional<torch::Tensor> const& cached_rank_prefix_matrix,
      std::optional<torch::Tensor> const& cached_channel_prefix_matrix,
      int expert_alignment, int num_worst_tokens, uccl::Config const& config,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending,
    // odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
      EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
      EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
      EP_HOST_ASSERT(num_tokens_per_rank.has_value());
      EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
      EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
      EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() ==
                     torch::kInt32);
    } else {
      EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
      EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and
                   is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and
                   is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
      EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and
                     cached_rank_prefix_matrix->is_contiguous());
      EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and
                     cached_rank_prefix_matrix->size(1) == num_ranks);
      EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and
                     cached_channel_prefix_matrix->is_contiguous());
      EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and
                     cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
      EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and
                     num_tokens_per_expert->is_contiguous());
      EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
      EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <=
                     NUM_MAX_LOCAL_EXPERTS);
      EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and
                     num_tokens_per_rank->is_contiguous());
      EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)),
         hidden = static_cast<int>(x.size(1));
    auto num_experts =
             cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
         num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int64_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
      num_topk = static_cast<int>(topk_idx->size(1));
      EP_HOST_ASSERT(num_experts > 0);
      EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
      EP_HOST_ASSERT(topk_weights->dim() == 2 and
                     topk_weights->is_contiguous());
      EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and
                     num_tokens == topk_weights->size(0));
      EP_HOST_ASSERT(num_topk == topk_weights->size(1));
      EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
      topk_idx_ptr = topk_idx->data_ptr<int64_t>();
      topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
      EP_HOST_ASSERT(x.element_size() == 1);
      EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or
                     x_scales->scalar_type() == torch::kInt);
      EP_HOST_ASSERT(x_scales->dim() == 2);
      EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
      num_scales =
          x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
      x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
      scale_token_stride = static_cast<int>(x_scales->stride(0));
      scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
      EP_HOST_ASSERT(previous_event.has_value() and async);
      at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
      num_recv_tokens = cached_num_recv_tokens;
      rank_prefix_matrix = cached_rank_prefix_matrix.value();
      channel_prefix_matrix = cached_channel_prefix_matrix.value();

      // Copy rank prefix matrix and clean flags
      uccl::intranode::cached_notify_dispatch(
          rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu,
          barrier_signal_ptrs_gpu, rank, num_ranks, comm_stream);
    } else {
      rank_prefix_matrix = torch::empty(
          {num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
      channel_prefix_matrix = torch::empty(
          {num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

      // Send sizes
      // Meta information:
      //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
      //  - Size prefix by experts (not used later), shaped as `[num_ranks,
      //  num_local_experts]`
      // NOTES: no more token dropping in this version
      *moe_recv_counter = -1;
      for (int i = 0; i < num_local_experts; ++i)
        moe_recv_expert_counter[i] = -1;
      EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) *
                         sizeof(int) <=
                     static_cast<size_t>(num_nvl_bytes));
      uccl::intranode::notify_dispatch(
          num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped,
          num_ranks, num_tokens_per_expert->data_ptr<int>(),
          moe_recv_expert_counter_mapped, num_experts, num_tokens,
          is_token_in_rank.data_ptr<bool>(),
          channel_prefix_matrix.data_ptr<int>(),
          rank_prefix_matrix.data_ptr<int>(), num_memset_int, expert_alignment,
          buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, comm_stream,
          num_channels);

      if (num_worst_tokens > 0) {
        // No CPU sync, just allocate the worst case
        num_recv_tokens = num_worst_tokens;

        // Must be forward with top-k stuffs
        EP_HOST_ASSERT(topk_idx.has_value());
        EP_HOST_ASSERT(topk_weights.has_value());
      } else {
        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
          // Read total count
          num_recv_tokens = static_cast<int>(*moe_recv_counter);

          // Read per-expert count
          bool ready = (num_recv_tokens >= 0);
          for (int i = 0; i < num_local_experts and ready; ++i)
            ready &= moe_recv_expert_counter[i] >= 0;

          if (ready) break;

          // Timeout check
          if (std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::high_resolution_clock::now() - start_time)
                  .count() > NUM_CPU_TIMEOUT_SECS)
            throw std::runtime_error("DeepEP error: CPU recv timeout");
        }
        num_recv_tokens_per_expert_list =
            std::vector<int>(moe_recv_expert_counter,
                             moe_recv_expert_counter + num_local_experts);
      }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens},
                                     dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_topk_idx = std::optional<torch::Tensor>(),
         recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty(
        {num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    auto send_head = torch::empty({num_tokens, num_ranks},
                                  dtype(torch::kInt32).device(torch::kCUDA));

    // Assign pointers
    int64_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
      recv_topk_idx =
          torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
      recv_topk_weights =
          torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
      recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
      recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
      recv_x_scales = x_scales->dim() == 1
                          ? torch::empty({num_recv_tokens}, x_scales->options())
                          : torch::empty({num_recv_tokens, num_scales},
                                         x_scales->options());
      recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Dispatch
    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +         // Size prefix matrix
            num_channels * num_ranks * sizeof(int) +  // Channel start offset
            num_channels * num_ranks * sizeof(int) +  // Channel end offset
            num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                hidden * recv_x.element_size() +  // Data buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                sizeof(int) +  // Source index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                num_topk * sizeof(int64_t) +  // Top-k index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                num_topk * sizeof(float) +  // Top-k weight buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                sizeof(float) * num_scales  // FP8 scale buffer
        <= static_cast<size_t>(num_nvl_bytes));

    uccl::intranode::dispatch(
        recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(),
        recv_topk_idx_ptr, recv_topk_weights_ptr,
        recv_channel_prefix_matrix.data_ptr<int>(), send_head.data_ptr<int>(),
        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
        is_token_in_rank.data_ptr<bool>(),
        channel_prefix_matrix.data_ptr<int>(), num_tokens, num_worst_tokens,
        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
        num_topk, num_experts, num_scales, scale_token_stride,
        scale_hidden_stride, buffer_ptrs_gpu, rank, num_ranks, comm_stream,
        config.num_sms, config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
      for (auto& t :
           {x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix,
            recv_x, recv_src_idx, recv_channel_prefix_matrix, send_head}) {
        t.record_stream(comm_stream);
        if (allocate_on_comm_stream) t.record_stream(compute_stream);
      }
      for (auto& to : {x_scales, topk_idx, topk_weights, num_tokens_per_rank,
                       num_tokens_per_expert, cached_channel_prefix_matrix,
                       cached_rank_prefix_matrix, recv_topk_idx,
                       recv_topk_weights, recv_x_scales}) {
        to.has_value() ? to->record_stream(comm_stream) : void();
        if (allocate_on_comm_stream)
          to.has_value() ? to->record_stream(compute_stream) : void();
      }
    } else {
      stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            event};
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>,
             std::optional<EventHandle>>
  intranode_combine(torch::Tensor const& x,
                    std::optional<torch::Tensor> const& topk_weights,
                    std::optional<torch::Tensor> const& bias_0,
                    std::optional<torch::Tensor> const& bias_1,
                    torch::Tensor const& src_idx,
                    torch::Tensor const& rank_prefix_matrix,
                    torch::Tensor const& channel_prefix_matrix,
                    torch::Tensor const& send_head, uccl::Config const& config,
                    std::optional<EventHandle>& previous_event, bool async,
                    bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and
                   src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and
                   send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and
                   rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and
                   channel_prefix_matrix.is_contiguous() and
                   channel_prefix_matrix.scalar_type() == torch::kInt32);

    // One channel use two blocks, even-numbered blocks for sending,
    // odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)),
         hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and
                   rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and
                   channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
      EP_HOST_ASSERT(previous_event.has_value() and async);
      at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
      EP_HOST_ASSERT(topk_weights->dim() == 2 and
                     topk_weights->is_contiguous());
      EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
      EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
      num_topk = static_cast<int>(topk_weights->size(1));
      topk_weights_ptr = topk_weights->data_ptr<float>();
      recv_topk_weights =
          torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
      recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <=
                   static_cast<size_t>(num_nvl_bytes));
    uccl::intranode::cached_notify_combine(
        buffer_ptrs_gpu, send_head.data_ptr<int>(), num_channels,
        num_recv_tokens, num_channels * num_ranks * 2, barrier_signal_ptrs_gpu,
        rank, num_ranks, comm_stream);

    // Assign bias pointers
    auto bias_opts =
        std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
      if (bias_opts[i].has_value()) {
        auto bias = bias_opts[i].value();
        EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
        EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and
                       bias.size(1) == hidden);
        bias_ptrs[i] = bias.data_ptr();
      }

    // Combine data
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(
        num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                hidden * x.element_size() +  // Data buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                sizeof(int) +  // Source index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                num_topk * sizeof(float)  // Top-k weight buffer
        <= static_cast<size_t>(num_nvl_bytes));
    uccl::intranode::combine(
        at::cuda::ScalarTypeToCudaDataType(x.scalar_type()), recv_x.data_ptr(),
        recv_topk_weights_ptr, x.data_ptr(), topk_weights_ptr, bias_ptrs[0],
        bias_ptrs[1], src_idx.data_ptr<int>(),
        rank_prefix_matrix.data_ptr<int>(),
        channel_prefix_matrix.data_ptr<int>(), send_head.data_ptr<int>(),
        num_tokens, num_recv_tokens, hidden, num_topk, buffer_ptrs_gpu, rank,
        num_ranks, comm_stream, config.num_sms,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
      event = EventHandle(comm_stream);
      for (auto& t : {x, src_idx, send_head, rank_prefix_matrix,
                      channel_prefix_matrix, recv_x}) {
        t.record_stream(comm_stream);
        if (allocate_on_comm_stream) t.record_stream(compute_stream);
      }
      for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1}) {
        to.has_value() ? to->record_stream(comm_stream) : void();
        if (allocate_on_comm_stream)
          to.has_value() ? to->record_stream(compute_stream) : void();
      }
    } else {
      stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

    return {recv_x, recv_topk_weights, event};
  }

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                int hidden, int num_experts) {
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

    uccl::internode_ll::clean_low_latency_buffer(
        ptr0, count0, ptr1, count1, at::cuda::getCurrentCUDAStream());
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor,
             torch::Tensor, torch::Tensor, std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch(
      torch::Tensor const& x, torch::Tensor const& topk_idx,
      std::optional<torch::Tensor> const& cumulative_local_expert_recv_stats,
      std::optional<torch::Tensor> const& dispatch_wait_recv_cost_stats,
      int num_max_dispatch_tokens_per_rank, int num_experts, bool use_fp8,
      bool round_scale, bool use_ue8m0, bool async, bool return_recv_hook) {
    EP_HOST_ASSERT(low_latency_mode);

    // printf("low_latency_dispatch called\n");

    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and
                   x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and
                   x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
      EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() ==
                     torch::kInt);
      EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and
                     cumulative_local_expert_recv_stats->is_contiguous());
      EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) ==
                     num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
      EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() ==
                     torch::kInt64);
      EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and
                     dispatch_wait_recv_cost_stats->is_contiguous());
      EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)),
         hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;

    // Buffer control
    // TODO(MaoZiming)
    uccl::LowLatencyLayout layout(rdma_buffer_ptr,
                                  num_max_dispatch_tokens_per_rank, hidden,
                                  num_ranks, num_experts, atomic_buffer_ptr);
    EP_HOST_ASSERT(layout.total_bytes <=
                   static_cast<std::size_t>(num_rdma_bytes));
    int low_latency_buffer_idx_used = low_latency_buffer_idx;
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook) stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x =
        torch::empty({num_local_experts,
                      num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                     x.options().dtype(use_fp8 ?
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
                                               fp8_dtype
#else
                                               torch::kFloat8_e4m3fn
#endif
                                               : torch::kBFloat16));
    auto packed_recv_src_info = torch::empty(
        {num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range =
        torch::empty({num_local_experts, num_ranks},
                     torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty(
        {num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and
                   "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
      // TODO: support unaligned cases
      EP_HOST_ASSERT(hidden % 512 == 0);
      if (not use_ue8m0) {
        packed_recv_x_scales =
            torch::empty({num_local_experts, hidden / 128,
                          num_ranks * num_max_dispatch_tokens_per_rank},
                         torch::dtype(torch::kFloat32).device(torch::kCUDA));
      } else {
        EP_HOST_ASSERT(round_scale);
        packed_recv_x_scales =
            torch::empty({num_local_experts, hidden / 512,
                          num_ranks * num_max_dispatch_tokens_per_rank},
                         torch::dtype(torch::kInt).device(torch::kCUDA));
      }
      packed_recv_x_scales =
          torch::transpose(packed_recv_x_scales.value(), 1, 2);
      packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto [ptr0, ptr_internode0, count0] = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
      uccl::internode_ll::dispatch(
          packed_recv_x.data_ptr(), packed_recv_x_scales_ptr,
          packed_recv_src_info.data_ptr<int>(),
          packed_recv_layout_range.data_ptr<int64_t>(),
          packed_recv_count.data_ptr<int>(),
          cumulative_local_expert_recv_stats.has_value()
              ? cumulative_local_expert_recv_stats->data_ptr<int>()
              : nullptr,
          dispatch_wait_recv_cost_stats.has_value()
              ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>()
              : nullptr,
          buffer.dispatch_rdma_recv_data_buffer,
          buffer.dispatch_rdma_recv_count_buffer,
          buffer.dispatch_rdma_send_buffer, x.data_ptr(),
          topk_idx.data_ptr<int64_t>(), ptr0, ptr_internode0, count0,
          num_tokens, hidden, num_max_dispatch_tokens_per_rank, num_topk,
          num_experts, rank, num_ranks, use_fp8, round_scale, use_ue8m0,
          workspace, num_device_sms, launch_stream, phases, d_handles,
          num_d2h_channel_addrs, max_nvl_peers, low_latency_buffer_idx_used,
          d_ipc_rdma_base_ptrs, rdma_buffer_ptr, atomic_buffer_ptr,
          buffer.dispatch_rdma_recv_count_buffer_internode);  // Added IPC base
                                                              // pointers
    };
    launcher(return_recv_hook
                 ? LOW_LATENCY_SEND_PHASE
                 : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
      // NOTES: we must ensure the all tensors will not be deallocated before
      // the stream-wait happens, so in Python API, we must wrap all tensors
      // into the event handle.
      event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
      stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
      recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            event,
            recv_hook};
  }

  std::tuple<torch::Tensor, std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine(
      torch::Tensor const& x, torch::Tensor const& topk_idx,
      torch::Tensor const& topk_weights, torch::Tensor const& src_info,
      torch::Tensor const& layout_range,
      std::optional<torch::Tensor> const& combine_wait_recv_cost_stats,
      int num_max_dispatch_tokens_per_rank, int num_experts, bool use_logfmt,
      bool zero_copy, bool async, bool return_recv_hook,
      std::optional<torch::Tensor> const& out) {
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and
                   x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and
                   topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and
                   x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and
                   layout_range.size(1) == num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
      EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() ==
                     torch::kInt64);
      EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and
                     combine_wait_recv_cost_stats->is_contiguous());
      EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    // TODO(MaoZiming)
    uccl::LowLatencyLayout layout(rdma_buffer_ptr,
                                  num_max_dispatch_tokens_per_rank, hidden,
                                  num_ranks, num_experts, atomic_buffer_ptr);
    EP_HOST_ASSERT(layout.total_bytes <=
                   static_cast<std::size_t>(num_rdma_bytes));
    int low_latency_buffer_idx_used = low_latency_buffer_idx;
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook) stream_wait(launch_stream, compute_stream);

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
      EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
      EP_HOST_ASSERT(out->size(0) == num_combined_tokens and
                     out->size(1) == hidden);
      EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
      combined_x = out.value();
    } else {
      combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Kernel launch
    auto [ptr0, ptr_internode0, count0] = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
      uccl::internode_ll::combine(
          combined_x.data_ptr(), buffer.combine_rdma_recv_data_buffer,
          buffer.combine_rdma_recv_flag_buffer, buffer.combine_rdma_send_buffer,
          x.data_ptr(), topk_idx.data_ptr<int64_t>(),
          topk_weights.data_ptr<float>(), src_info.data_ptr<int>(),
          layout_range.data_ptr<int64_t>(),
          combine_wait_recv_cost_stats.has_value()
              ? combine_wait_recv_cost_stats->data_ptr<int64_t>()
              : nullptr,
          ptr0, ptr_internode0, count0, num_combined_tokens, hidden,
          num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank,
          num_ranks, use_logfmt, workspace, num_device_sms, launch_stream,
          phases, zero_copy, d_handles, num_d2h_channel_addrs, max_nvl_peers,
          low_latency_buffer_idx_used, d_ipc_rdma_base_ptrs, rdma_buffer_ptr,
          atomic_buffer_ptr,
          buffer.combine_rdma_recv_flag_buffer_internode);  // Added IPC base
                                                            // pointers
    };
    launcher(return_recv_hook
                 ? LOW_LATENCY_SEND_PHASE
                 : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
      // NOTES: we must ensure the all tensors will not be deallocated before
      // the stream-wait happens, so in Python API, we must wrap all tensors
      // into the event handle.
      event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
      stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
      recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {combined_x, event, recv_hook};
  }

  int get_local_device_id() { return device_index; }

  pybind11::bytearray get_local_ipc_handle() const {
    return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
  }

  pybind11::bytearray get_local_rdma_ipc_handle() {
    EP_HOST_ASSERT(
        rdma_buffer_ptr != nullptr &&
        "set_rdma_buffer_raw must be called before requesting RDMA IPC handle");
    cudaIpcMemHandle_t h{};
    CUDA_CHECK(cudaIpcGetMemHandle(&h, rdma_buffer_ptr));
    return {h.reserved, CUDA_IPC_HANDLE_SIZE};
  }

  pybind11::bytearray get_local_atomics_ipc_handle() {
    EP_HOST_ASSERT(atomic_buffer_ptr != nullptr &&
                   "set_atomic_buffer_raw must be called before requesting "
                   "atomic IPC handle");
    cudaIpcMemHandle_t h{};
    CUDA_CHECK(cudaIpcGetMemHandle(&h, atomic_buffer_ptr));
    return {h.reserved, CUDA_IPC_HANDLE_SIZE};
  }

  int get_num_rdma_ranks() const { return num_rdma_ranks; }
  int get_rdma_rank() const { return rdma_rank; }
  int get_root_rdma_rank(bool global) const { return global ? nvl_rank : 0; }

  pybind11::bytearray get_local_uccl_shmem_unique_id() const {
    EP_HOST_ASSERT(rdma_rank == 0 and
                   "Only RDMA rank 0 can get UCCL unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<char const*>(unique_id.data()), unique_id.size()};
  }

  torch::Tensor get_next_low_latency_combine_buffer(
      int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
    // printf("get_next_low_latency_combine_buffer called\n");
    uccl::LowLatencyLayout layout(rdma_buffer_ptr,
                                  num_max_dispatch_tokens_per_rank, hidden,
                                  num_ranks, num_experts, nullptr);

    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg /
                                          elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(
        buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(
        buffer.combine_rdma_send_buffer_data_start,
        {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank,
         hidden},
        {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems,
         num_msg_elems, 1},
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
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

  void sync(
      std::vector<int> const& device_ids,
      std::vector<std::optional<pybind11::bytearray>> const&
          all_gathered_handles,
      std::optional<pybind11::bytearray> const& root_unique_id_opt,
      std::optional<std::vector<std::optional<pybind11::bytearray>>> const&
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
        auto handle_str =
            std::string(all_gathered_handles[global_rank].value());
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
          auto handle_str =
              std::string(all_gathered_rdma_handles[global_rank].value());
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

  void set_rdma_buffer_raw(void* ptr) {
    if (ptr == nullptr) {
      throw std::invalid_argument("set_rdma_buffer_raw: ptr null");
    }
    rdma_buffer_ptr = ptr;
  }

  void set_atomic_buffer_ptr(void* ptr) {
    if (ptr == nullptr) {
      throw std::invalid_argument("set_atomic_buffer_ptr: ptr null");
    }
    atomic_buffer_ptr = ptr;
  }

  torch::Tensor get_local_buffer_tensor(pybind11::object const& dtype,
                                        int64_t offset,
                                        bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype =
        torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr =
        static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr
                                              : buffer_ptrs[nvl_rank]) +
        offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    return torch::from_blob(
        base_ptr, num_bytes / element_bytes,
        torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
  }

  torch::Stream get_comm_stream() const { return comm_stream; }

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
  std::vector<py::object> proxies_;
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
  at::cuda::CUDAStream comm_stream;

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

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // support select fp8 dtype for different gcn arch in runtime
  c10::ScalarType fp8_dtype{torch::kFloat8_e4m3fnuz};
#endif
};

PYBIND11_MODULE(ep, m) {
  m.doc() = "Minimal DeepEP-compatible shim with UCCL";

  pybind11::class_<uccl::Config>(m, "Config")
      .def(pybind11::init<int, int, int, int, int>(), py::arg("num_sms") = 20,
           py::arg("num_max_nvl_chunked_send_tokens") = 6,
           py::arg("num_max_nvl_chunked_recv_tokens") = 256,
           py::arg("num_max_rdma_chunked_send_tokens") = 6,
           py::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint", &uccl::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint",
           &uccl::Config::get_rdma_buffer_size_hint);

  m.def(
      "register_proxy",
      [](int device_index, py::object proxy) {
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
      py::arg("device_index"), py::arg("proxy"));
  m.def(
      "register_proxies",
      [](int device_index, std::vector<py::object> proxies) {
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
      py::arg("device_index"), py::arg("proxies"));
  m.def(
      "unregister_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        uccl::g_proxies_by_dev.erase(device_index);
      },
      py::arg("device_index"));
  m.def(
      "has_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        auto it = uccl::g_proxies_by_dev.find(device_index);
        return it != uccl::g_proxies_by_dev.end() && !it->second.empty();
      },
      py::arg("device_index"));
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
      [](int64_t num_rdma_bytes, int device_index) {
        void* ptr;
        bool is_host_allocated = false;
        CUDA_CHECK(cudaSetDevice(device_index));
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        CUDA_CHECK(
            hipExtMallocWithFlags(&ptr, num_rdma_bytes, hipDeviceMallocUncached));
        CUDA_CHECK(cudaMemset(ptr, 0, num_rdma_bytes));
        torch::Tensor tensor = torch::from_blob(
            ptr, {num_rdma_bytes},
            dtype(torch::kUInt8).device(torch::kCUDA));
        return py::make_tuple(tensor, false);
#else
        // Prefer cudaMalloc when GPU memory can be registered (e.g. with
        // nvidia_peermem); fall back to pinned host for NICs where
        // ibv_reg_mr fails on GPU memory (e.g. GH10).
        torch::Tensor tensor;
        if (can_register_gpu_memory_for_atomics(device_index)) {
          CUDA_CHECK(cudaMalloc(&ptr, num_rdma_bytes));
          CUDA_CHECK(cudaMemset(ptr, 0, num_rdma_bytes));
          tensor = torch::from_blob(
              ptr, {num_rdma_bytes},
              dtype(torch::kUInt8).device(torch::kCUDA));
        } else {
          CUDA_CHECK(cudaMallocHost(&ptr, num_rdma_bytes));
          CUDA_CHECK(cudaMemset(ptr, 0, num_rdma_bytes));
          tensor = torch::from_blob(ptr, {num_rdma_bytes},
                                   dtype(torch::kUInt8));
          is_host_allocated = true;
        }
        return py::make_tuple(tensor, is_host_allocated);
#endif
      },
      py::arg("num_rdma_bytes"),
      py::arg("device_index"),
      R"doc(
        Allocate RDMA buffer. Prefers device memory (cudaMalloc) when the NIC
        can register it; otherwise uses pinned host memory (cudaMallocHost).
        Returns (tensor, is_host_allocated).
      )doc");

  py::class_<EventHandle>(m, "EventHandle")
      .def(py::init<>())
      .def("current_stream_wait", &EventHandle::current_stream_wait);

  m.def("connect_atomic_buffer", [](UcclProxy& p, Buffer& b) {
    b.set_atomic_buffer_ptr(p.get_atomic_buffer_ptr());
  });

  py::class_<EventOverlap>(m, "EventOverlap").def(py::init<>());
  py::class_<Buffer>(m, "Buffer")
      .def(py::init<int, int, long, long, bool, bool, int>(), py::arg("rank"),
           py::arg("num_ranks"), py::arg("num_nvl_bytes") = 0,
           py::arg("num_rdma_bytes") = 0, py::arg("low_latency_mode") = false,
           py::arg("explicitly_destroy") = false,
           py::arg("num_local_ranks") = -1)
      .def("destroy", &Buffer::destroy)
      .def(
          "set_rdma_buffer_raw",
          [](Buffer& self, std::uintptr_t addr) {
            self.set_rdma_buffer_raw(reinterpret_cast<void*>(addr));
          },
          py::arg("addr"),
          R"doc(Set RDMA buffer from a raw address. Caller must keep the memory alive.)doc")
      .def("reset_rdma_buffer", &Buffer::reset_rdma_buffer)
      .def("low_latency_dispatch", &Buffer::low_latency_dispatch, py::arg("x"),
           py::arg("topk_idx"),
           py::arg("cumulative_local_expert_recv_stats") = py::none(),
           py::arg("dispatch_wait_recv_cost_stats") = py::none(),
           py::arg("num_max_dispatch_tokens_per_rank") = 0,
           py::arg("num_experts") = 1, py::arg("use_fp8") = true,
           py::arg("round_scale") = false, py::arg("use_ue8m0") = false,
           py::arg("async") = false, py::arg("return_recv_hook") = false)
      .def("get_local_device_id", &Buffer::get_local_device_id)
      .def("get_local_ipc_handle", &Buffer::get_local_ipc_handle)
      .def("get_local_rdma_ipc_handle", &Buffer::get_local_rdma_ipc_handle)
      .def("get_local_atomics_ipc_handle",
           &Buffer::get_local_atomics_ipc_handle)
      .def("get_num_rdma_ranks", &Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank)
      .def("get_local_buffer_tensor", &Buffer::get_local_buffer_tensor)
      .def("get_comm_stream", &Buffer::get_comm_stream)
      .def("get_local_uccl_shmem_unique_id",
           &Buffer::get_local_uccl_shmem_unique_id)
      .def("sync", &Buffer::sync, py::arg("device_ids"),
           py::arg("all_gathered_handles"),
           py::arg("root_unique_id_opt") = py::none(),
           py::arg("all_gathered_rdma_handles") = py::none())
      .def("is_available", &Buffer::is_available)
      .def("get_next_low_latency_combine_buffer",
           &Buffer::get_next_low_latency_combine_buffer)
      .def("get_dispatch_layout", &Buffer::get_dispatch_layout)
      .def("intranode_dispatch", &Buffer::intranode_dispatch)
      .def("intranode_combine", &Buffer::intranode_combine)
      .def("internode_dispatch", &Buffer::internode_dispatch)
      .def("internode_combine", &Buffer::internode_combine)
      .def("clean_low_latency_buffer", &Buffer::clean_low_latency_buffer)
      .def("low_latency_combine", &Buffer::low_latency_combine, py::arg("x"),
           py::arg("topk_idx"), py::arg("topk_weights"), py::arg("src_info"),
           py::arg("layout_range"),
           py::arg("combine_wait_recv_cost_stats") = py::none(),
           py::arg("num_max_dispatch_tokens_per_rank") = 0,
           py::arg("num_experts") = 1, py::arg("use_logfmt") = false,
           py::arg("zero_copy") = false, py::arg("async") = false,
           py::arg("return_recv_hook") = false, py::arg("out") = py::none());
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
      py::arg("stream_ptr"));
  m.def("device_reset", []() {
    auto st = cudaDeviceReset();
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaDeviceReset failed: ") +
                               cudaGetErrorString(st));
  });
  py::class_<Stats>(m, "Stats");
  py::class_<UcclProxy>(m, "Proxy")
      .def(py::init<int, uintptr_t, size_t, int, int, int, int, int, int, bool,
                    bool, bool>(),
           py::arg("thread_idx"), py::arg("gpu_buffer_addr"),
           py::arg("total_size"), py::arg("rank") = 0, py::arg("node_idx") = -1,
           py::arg("local_rank") = 0, py::arg("num_experts") = -1,
           py::arg("num_ranks") = -1, py::arg("num_nodes") = 0,
           py::arg("use_normal_mode") = false, py::arg("is_intranode") = false,
           py::arg("gpu_buffer_is_host_allocated") = false)
      .def("start_sender", &UcclProxy::start_sender)
      .def("start_remote", &UcclProxy::start_remote)
      .def("start_local", &UcclProxy::start_local)
      .def("start_dual", &UcclProxy::start_dual)
      .def("stop", &UcclProxy::stop)
      .def("get_listen_port", &UcclProxy::get_listen_port)
      .def("get_atomic_buffer_ptr", &UcclProxy::get_atomic_buffer_ptr)
      .def("set_atomic_buffer_ptr", &UcclProxy::set_atomic_buffer_ptr)
      .def("set_dispatch_recv_data_offset",
           &UcclProxy::set_dispatch_recv_data_offset, py::arg("offset"))
      .def("calculate_and_set_dispatch_recv_data_offset",
           &UcclProxy::calculate_and_set_dispatch_recv_data_offset,
           py::arg("num_tokens"), py::arg("hidden"), py::arg("num_experts"))
      .def("get_d2h_channel_addrs", &UcclProxy::get_d2h_channel_addrs)
      .def_property_readonly("thread_idx", &UcclProxy::thread_idx)
      .def_property_readonly("gpu_buffer_addr", &UcclProxy::gpu_buffer_addr)
      .def("avg_rdma_write_us", &UcclProxy::avg_rdma_write_us)
      .def("avg_wr_latency_us", &UcclProxy::avg_wr_latency_us)
      .def(
          "set_peers_meta",
          [](UcclProxy& self, py::object metas) {
            std::vector<PeerMeta> v;
            if (py::isinstance<py::list>(metas)) {
              for (auto obj : metas.cast<py::list>()) {
                if (py::isinstance<py::dict>(obj)) {
                  auto d = obj.cast<py::dict>();
                  PeerMeta pm;
                  pm.rank = py::cast<int>(d["rank"]);
                  pm.ptr = static_cast<uintptr_t>(
                      py::cast<unsigned long long>(d["ptr"]));
                  pm.nbytes = static_cast<size_t>(
                      py::cast<unsigned long long>(d["nbytes"]));
                  pm.ip = py::cast<std::string>(d["ip"]);

                  // Handle listen_ports array (always present)
                  auto ports = d["listen_ports"].cast<py::sequence>();
                  size_t port_count =
                      std::min(static_cast<size_t>(py::len(ports)),
                               static_cast<size_t>(kNumProxyThs));
                  for (size_t i = 0; i < port_count; ++i) {
                    pm.listen_ports[i] = ports[i].cast<int>();
                  }
                  // Initialize remaining ports to 0 if fewer than kNumProxyThs
                  // provided
                  for (size_t i = port_count; i < kNumProxyThs; ++i) {
                    pm.listen_ports[i] = 0;
                  }

                  v.push_back(std::move(pm));
                } else {
                  v.push_back(obj.cast<PeerMeta>());
                }
              }
            } else {
              // allow passing a dict directly
              auto d = metas.cast<py::dict>();
              PeerMeta pm;
              pm.rank = py::cast<int>(d["rank"]);
              pm.ptr = static_cast<uintptr_t>(
                  py::cast<unsigned long long>(d["ptr"]));
              pm.nbytes = static_cast<size_t>(
                  py::cast<unsigned long long>(d["nbytes"]));
              pm.ip = py::cast<std::string>(d["ip"]);

              // Handle listen_ports array (always present)
              auto ports = d["listen_ports"].cast<py::sequence>();
              size_t port_count = std::min(static_cast<size_t>(py::len(ports)),
                                           static_cast<size_t>(kNumProxyThs));
              for (size_t i = 0; i < port_count; ++i) {
                pm.listen_ports[i] = ports[i].cast<int>();
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
          py::arg("metas"),
          "Attach peer metadata (list of dicts or PeerMeta objects).")
      .def(
          "set_bench_d2h_channel_addrs",
          [](UcclProxy& self, py::iterable addrs) {
            std::vector<uintptr_t> v;
            for (py::handle h : addrs) v.push_back(h.cast<uintptr_t>());
            self.set_bench_d2h_channel_addrs(v);
          },
          py::arg("addrs"), "Attach ring buffer addresses for benchmarking.");
  // .def_property_readonly("gpu_buffer_addr", &UcclProxy::gpu_buffer_addr);
  py::class_<EnvInfo>(m, "EnvInfo")
      .def_readonly("blocks", &EnvInfo::blocks)
      .def_readonly("queue_size", &EnvInfo::queue_size)
      .def_readonly("threads_per_block", &EnvInfo::threads_per_block)
      .def_readonly("iterations", &EnvInfo::iterations)
      .def_readonly("stream_addr", &EnvInfo::stream_addr)
      .def_readonly("rbs_addr", &EnvInfo::rbs_addr);
  py::class_<Bench>(m, "Bench")
      .def(py::init<>())
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
           py::arg("poll_ms") = 5, py::arg("timeout_ms") = -1,
           py::arg("should_abort") = nullptr)
      .def("join_proxies", &Bench::join_proxies)
      .def("print_block_latencies", &Bench::print_block_latencies)
      .def("compute_stats", &Bench::compute_stats)
      .def("print_summary", &Bench::print_summary)
      .def("print_summary_last", &Bench::print_summary_last)
      .def("last_elapsed_ms", &Bench::last_elapsed_ms);
  py::class_<PeerCopyManager>(m, "PeerCopyManager")
      .def(py::init<int>(), py::arg("src_device") = 0)
      .def("start_for_proxies",
           [](PeerCopyManager& mgr, py::iterable proxy_list) {
             std::vector<UcclProxy*> vec;
             for (py::handle h : proxy_list)
               vec.push_back(h.cast<UcclProxy*>());
             mgr.start_for_proxies(vec);
           })
      .def("stop", &PeerCopyManager::stop);

  // MSCCLPP Fifo class - must be registered before BenchFifo which uses it
  py::class_<mscclpp::Fifo>(m, "Fifo").def(py::init<uint32_t>(),
                                           py::arg("size") = 2048);

  // FIFO-based benchmarking classes
  py::class_<BenchFifo>(m, "BenchFifo")
      .def(py::init<>())
      .def("env_info", &BenchFifo::env_info)
      .def("blocks", &BenchFifo::blocks)
      .def("num_proxies", &BenchFifo::num_proxies)
      .def("get_fifo", &BenchFifo::get_fifo, py::return_value_policy::reference)
      .def("timing_start", &BenchFifo::timing_start)
      .def("timing_stop", &BenchFifo::timing_stop)
      .def("is_running", &BenchFifo::is_running)
      .def("launch_gpu_issue_batched_commands",
           &BenchFifo::launch_gpu_issue_batched_commands)
      .def("sync_stream", &BenchFifo::sync_stream)
      .def("sync_stream_interruptible", &BenchFifo::sync_stream_interruptible,
           py::arg("poll_ms") = 5, py::arg("timeout_ms") = -1,
           py::arg("should_abort") = nullptr)
      .def("join_proxies", &BenchFifo::join_proxies)
      .def("print_block_latencies", &BenchFifo::print_block_latencies)
      .def("compute_stats", &BenchFifo::compute_stats)
      .def("print_summary", &BenchFifo::print_summary)
      .def("print_summary_last", &BenchFifo::print_summary_last)
      .def("last_elapsed_ms", &BenchFifo::last_elapsed_ms);

  py::class_<FifoProxy>(m, "FifoProxy")
      .def(py::init<int, uintptr_t, size_t, int, int, int, bool>(),
           py::arg("thread_idx"), py::arg("gpu_buffer_addr"),
           py::arg("total_size"), py::arg("rank"), py::arg("node_idx"),
           py::arg("local_rank"), py::arg("is_intranode"))
      .def("set_fifo", &FifoProxy::set_fifo, py::arg("fifo"))
      .def("set_peers_meta",
           [](FifoProxy& proxy, py::list meta_list) {
             std::vector<PeerMeta> vec;
             for (py::handle h : meta_list) {
               if (py::isinstance<py::dict>(h)) {
                 auto d = h.cast<py::dict>();
                 PeerMeta pm;
                 pm.rank = d["rank"].cast<int>();
                 pm.ptr = d["ptr"].cast<uintptr_t>();
                 pm.nbytes = d["nbytes"].cast<size_t>();
                 pm.ip = d["ip"].cast<std::string>();

                 // Handle listen_ports array (always present)
                 auto ports = d["listen_ports"].cast<py::sequence>();
                 size_t port_count =
                     std::min(static_cast<size_t>(py::len(ports)),
                              static_cast<size_t>(kNumProxyThs));
                 for (size_t i = 0; i < port_count; ++i) {
                   pm.listen_ports[i] = ports[i].cast<int>();
                 }
                 // Initialize remaining ports to 0 if fewer than kNumProxyThs
                 // provided
                 for (size_t i = port_count; i < kNumProxyThs; ++i) {
                   pm.listen_ports[i] = 0;
                 }

                 vec.push_back(std::move(pm));
               } else {
                 vec.push_back(h.cast<PeerMeta>());
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
      .def_readonly("thread_idx", &FifoProxy::thread_idx);
}

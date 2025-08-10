#include "ep_config.hpp"
#include "ep_event.hpp"
#include "ep_proxy_registry.hpp"
#include "ep_runtime.cuh"
#include "ep_util.hpp"
#include "internode_ll.cuh"
#include <ATen/cuda/CUDAContext.h>
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

static std::vector<uint64_t> collect_ring_addrs_for_device(int device_index) {
  std::lock_guard<std::mutex> lk(g_proxies_mu);
  auto it = uccl::g_proxies_by_dev.find(device_index);
  EP_HOST_ASSERT(it != uccl::g_proxies_by_dev.end() && !it->second.empty());
  std::vector<uint64_t> addrs;
  addrs.reserve(it->second.size());
  for (auto& proxy : it->second) {
    addrs.push_back(proxy.attr("rb_addr").cast<uint64_t>());
  }
  return addrs;
}

class Buffer {
 public:
  Buffer(int rank, int num_ranks, long num_nvl_bytes, long num_rdma_bytes,
         bool low_latency_mode, bool explicitly_destroy)
      : rank(rank),
        num_ranks(num_ranks),
        num_nvl_bytes(num_nvl_bytes),
        num_rdma_bytes(num_rdma_bytes),
        low_latency_mode(low_latency_mode),
        explicitly_destroy(explicitly_destroy),
        comm_stream(at::cuda::getStreamFromPool(/*isHighPriority=*/true)) {
    // TODO(MaoZiming): Initialize UCCL proxy processes.
    {
      printf(
          "Buffer initializing for rank %d, num_ranks %d, num_nvl_bytes %ld, "
          "num_rdma_bytes %ld\n",
          rank, num_ranks, num_nvl_bytes, num_rdma_bytes);
      {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        CUDA_CHECK(cudaGetDevice(&device_index));
        auto it = uccl::g_proxies_by_dev.find(device_index);
        if (it == uccl::g_proxies_by_dev.end() || it->second.empty()) {
          throw std::runtime_error(
              "uccl_ep.Buffer: no UcclProxy registered for device " +
              std::to_string(device_index) +
              ". Call uccl.uccl_ep.register_proxy(device_index, proxies) "
              "first.");
        }
      }

      {
        CUDA_CHECK(cudaSetDevice(device_index));
        auto host_addrs = collect_ring_addrs_for_device(device_index);
        num_ring_addrs = static_cast<int>(host_addrs.size());
        EP_HOST_ASSERT(num_ring_addrs > 0);

        CUDA_CHECK(cudaMallocManaged(&d_ring_addrs,
                                     num_ring_addrs * sizeof(uint64_t)));

        for (int i = 0; i < num_ring_addrs; ++i) {
          void* host_ptr = reinterpret_cast<void*>(host_addrs[i]);
          void* dev_ptr = nullptr;
          CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
          d_ring_addrs[i] = reinterpret_cast<uint64_t>(dev_ptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    int64_t const barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t const buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t const barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                   (num_nvl_bytes <= std::numeric_limits<int>::max() ||
                    num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                   (low_latency_mode ||
                    num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(0 <= rank && rank < num_ranks &&
                   (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS ||
                    low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS ||
                   (num_ranks % NUM_MAX_NVL_PEERS) == 0);
    if (num_rdma_bytes > 0)
      EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS || low_latency_mode);

    rdma_rank = rank / NUM_MAX_NVL_PEERS;
    nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
    num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
    num_device_sms = prop.multiProcessorCount;

    if (num_nvl_bytes > 0) {
      size_t total_bytes = static_cast<size_t>(num_nvl_bytes) +
                           static_cast<size_t>(barrier_signal_bytes) +
                           static_cast<size_t>(buffer_ptr_bytes) +
                           static_cast<size_t>(barrier_signal_ptr_bytes);

      CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], total_bytes));
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

    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));
    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t),
                              cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped,
                                        const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter,
                              sizeof(int) * NUM_MAX_LOCAL_EXPERTS,
                              cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&moe_recv_expert_counter_mapped),
        moe_recv_expert_counter, 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
      moe_recv_expert_counter[i] = -1;

    if (num_rdma_ranks > 0) {
      CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int),
                                cudaHostAllocMapped));
      CUDA_CHECK(cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&moe_recv_rdma_counter_mapped),
          moe_recv_rdma_counter, 0));
      *moe_recv_rdma_counter = -1;
    }
    printf(
        "Buffer created for rank %d, num_ranks %d, num_nvl_bytes %ld, "
        "num_rdma_bytes %ld, low_latency_mode %d\n",
        rank, num_ranks, num_nvl_bytes, num_rdma_bytes, low_latency_mode);
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

    // Free NVSHMEM
#ifndef DISABLE_NVSHMEM
    if (is_available() and num_rdma_bytes > 0) {
      CUDA_CHECK(cudaDeviceSynchronize());
      internode::barrier();
      internode::free(rdma_buffer_ptr);
      internode::finalize();
    }
#endif

    // Free workspace and MoE counter
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

    destroyed = true;
    available = false;
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
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank,
                            hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <=
                   static_cast<std::size_t>(num_rdma_bytes));
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook) stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x = torch::empty(
        {num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank,
         hidden},
        x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
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
    auto next_clean_meta = next_buffer.clean_meta();
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
          topk_idx.data_ptr<int64_t>(), next_clean_meta.first,
          next_clean_meta.second, num_tokens, hidden,
          num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank,
          num_ranks, use_fp8, round_scale, use_ue8m0, workspace, num_device_sms,
          launch_stream, phases, d_ring_addrs,
          num_ring_addrs);  // NOTE(MaoZiming): adding UCCL ring buffers
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
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank,
                            hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <=
                   static_cast<std::size_t>(num_rdma_bytes));
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
    auto next_clean_meta = next_buffer.clean_meta();
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
          next_clean_meta.first, next_clean_meta.second, num_combined_tokens,
          hidden, num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank,
          num_ranks, use_logfmt, workspace, num_device_sms, launch_stream,
          phases, zero_copy, d_ring_addrs,
          num_ring_addrs);  // NOTE(MaoZiming): adding UCCL ring buffers
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

  int get_num_rdma_ranks() const { return num_rdma_ranks; }
  int get_rdma_rank() const { return rdma_rank; }
  int get_root_rdma_rank(bool global) const { return global ? nvl_rank : 0; }

  pybind11::bytearray get_local_uccl_shmem_unique_id() const {
    EP_HOST_ASSERT(rdma_rank == 0 and
                   "Only RDMA rank 0 can get UCCL unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<char const*>(unique_id.data()), unique_id.size()};
  }

  void sync(std::vector<int> const& device_ids,
            std::vector<std::optional<pybind11::bytearray>> const&
                all_gathered_handles,
            std::optional<pybind11::bytearray> const& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());

    // Sync IPC handles
    if (num_nvl_bytes > 0) {
      EP_HOST_ASSERT(static_cast<std::size_t>(num_ranks) == device_ids.size());
      EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
      for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks;
           ++i) {
        EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
        auto handle_str = std::string(all_gathered_handles[offset + i].value());
        EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
        if (offset + i != rank) {
          std::memcpy(ipc_handles[i].reserved, handle_str.c_str(),
                      CUDA_IPC_HANDLE_SIZE);
          CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i],
                                          cudaIpcMemLazyEnablePeerAccess));
          barrier_signal_ptrs[i] = reinterpret_cast<int*>(
              static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
        } else {
          EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved,
                                     handle_str.c_str(),
                                     CUDA_IPC_HANDLE_SIZE) == 0);
        }
      }

      // Copy all buffer and barrier signal pointers to GPU
      CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs,
                            sizeof(void*) * NUM_MAX_NVL_PEERS,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs,
                            sizeof(int*) * NUM_MAX_NVL_PEERS,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Sync NVSHMEM handles and allocate memory
    // NOTE(MaoZiming): drop nvshmem. we directly allocate rdma_buffer_ptr.
    if (num_rdma_bytes > 0) {
#if false
      // Initialize NVSHMEM
      EP_HOST_ASSERT(root_unique_id_opt.has_value());
      std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
      auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
      std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(),
                  root_unique_id_opt->size());
      auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
      auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
      EP_HOST_ASSERT(nvshmem_rank ==
                     internode::init(root_unique_id, nvshmem_rank,
                                     num_nvshmem_ranks, low_latency_mode));
      internode::barrier();

      // Allocate
      rdma_buffer_ptr =
          internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

      // Clean buffer (mainly for low-latency mode)
      CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

      // Barrier
      internode::barrier();
      CUDA_CHECK(cudaDeviceSynchronize());
#else
      rdma_buffer_ptr =
          internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
      CUDA_CHECK(
          cudaMemsetAsync(rdma_buffer_ptr, 0, num_rdma_bytes, comm_stream));
      CUDA_CHECK(cudaStreamSynchronize(comm_stream));
#endif
    }
    // Ready to use
    available = true;
  }

  bool is_available() const { return available; }

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
  int low_latency_buffer_idx = 0;
  void* workspace = nullptr;

  // device / ranks
  int rdma_rank{0}, nvl_rank{0};
  int num_rdma_ranks{1}, num_nvl_ranks{1};
  int num_device_sms{0};

  // stream & workspace
  at::cuda::CUDAStream comm_stream;

  // NVLink (IPC) area
  static constexpr int NUM_MAX_NVL_PEERS = 8;
  static constexpr int NUM_MAX_RDMA_PEERS = 16;
  static constexpr int NUM_WORKSPACE_BYTES = 32 * 1024 * 1024;  // 32 MiB
  static constexpr int NUM_BUFFER_ALIGNMENT_BYTES = 256;

  cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS]{};
  void* buffer_ptrs[NUM_MAX_NVL_PEERS]{};
  int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS]{};
  void** buffer_ptrs_gpu{nullptr};
  int** barrier_signal_ptrs_gpu{nullptr};

  // MoE counters (host mapped)
  int volatile* moe_recv_counter = nullptr;
  int64_t* moe_recv_counter_mapped{nullptr};  // device pointer
  int* moe_recv_expert_counter{nullptr};
  int* moe_recv_expert_counter_mapped{nullptr};
  int* moe_recv_rdma_counter{nullptr};
  int* moe_recv_rdma_counter_mapped{nullptr};

  bool destroyed = false;

  // Ring buffers
  int num_ring_addrs{0};
  uint64_t* d_ring_addrs{nullptr};
};

PYBIND11_MODULE(uccl_ep, m) {
  m.doc() = "Minimal DeepEP-compatible shim with UCCL";
  m.def(
      "register_proxy",
      [](int device_index, py::object proxy) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        uccl::g_proxies_by_dev[device_index].push_back(std::move(proxy));
      },
      py::arg("device_index"), py::arg("proxy"));
  m.def(
      "register_proxies",
      [](int device_index, std::vector<py::object> proxies) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        auto& vec = uccl::g_proxies_by_dev[device_index];
        for (auto& proxy : proxies) {
          vec.push_back(std::move(proxy));
        }
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

  py::class_<EventHandle>(m, "EventHandle")
      .def(py::init<>())
      .def("current_stream_wait", &EventHandle::current_stream_wait);

  py::class_<EventOverlap>(m, "EventOverlap").def(py::init<>());
  py::class_<Buffer>(m, "Buffer")
      .def(py::init<int, int, long, long, bool, bool>(), py::arg("rank"),
           py::arg("num_ranks"), py::arg("num_nvl_bytes") = 0,
           py::arg("num_rdma_bytes") = 0, py::arg("low_latency_mode") = false,
           py::arg("explicitly_destroy") = false)
      .def("destroy", &Buffer::destroy)
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
      .def("get_num_rdma_ranks", &Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank)
      .def("get_local_uccl_shmem_unique_id",
           &Buffer::get_local_uccl_shmem_unique_id)
      .def("sync", &Buffer::sync, py::arg("device_ids"),
           py::arg("all_gathered_handles"),
           py::arg("root_unique_id_opt") = py::none())
      .def("is_available", &Buffer::is_available)
      .def("low_latency_combine", &Buffer::low_latency_combine, py::arg("x"),
           py::arg("topk_idx"), py::arg("topk_weights"), py::arg("src_info"),
           py::arg("layout_range"),
           py::arg("combine_wait_recv_cost_stats") = py::none(),
           py::arg("num_max_dispatch_tokens_per_rank") = 0,
           py::arg("num_experts") = 1, py::arg("use_logfmt") = false,
           py::arg("zero_copy") = false, py::arg("async") = false,
           py::arg("return_recv_hook") = false, py::arg("out") = py::none());
}
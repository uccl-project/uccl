#pragma once

#include "../kernels/backend/api.cuh"
#include "../kernels/elastic/api.hpp"
#include "../uccl/include/shared_buffer.hpp"
#include "../uccl/include/uccl_proxy.hpp"
#include "../utils/event.hpp"
#include "utils.hpp"
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <pybind11/functional.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace deep_ep::elastic {

class ElasticBuffer {
  // Buffer registered for both scaleout and scaleup
  int64_t num_buffer_bytes;
  void* buffer;

  // Destructor settings
  bool explicitly_destroy;
  bool destroyed = false;

  // Workspace
  // NOTES: for all workspace, we must keep them as zeros
  void* workspace;
  void *host_workspace, *mapped_host_workspace;
  std::shared_ptr<layout::WorkspaceLayout> workspace_layout_wo_expert;

  // CUDA streams
  at::cuda::CUDAStream comm_stream;

  // Whether to use deterministic algorithms
  bool deterministic;

  // Whether to use hybrid mode (scale-out with scale-up)
  bool allow_hybrid_mode;

  // Whether to allow multiple reductions
  bool allow_multiple_reduction;

  // Whether to prefer overlapping communication with compute (use more SMs and
  // channels if false)
  bool prefer_overlap_with_compute;

  // Timeout settings
  int num_cpu_timeout_secs;
  int64_t num_gpu_timeout_cycles;

  // NCCL context
  std::shared_ptr<nccl::NCCLSymmetricMemoryContext> nccl_context;

  // UCCL CPU proxy context for no-GPUDirect-RDMA EP data transport
  bool use_uccl_proxy = false;
  bool uccl_use_host_window = false;
  bool uccl_use_shared_window = false;
  bool uccl_proxies_started = false;
  int rank_idx = 0, num_ranks = 0;
  int node_idx = 0, local_rank = 0, local_world_size = 1;
  std::string uccl_shm_id;
  SharedBuffer uccl_window;
  SharedBuffer uccl_atomic_window;
  void* uccl_gpu_window = nullptr;
  void* uccl_host_window = nullptr;
  void* uccl_host_window_device = nullptr;
  std::vector<std::unique_ptr<UcclProxy>> uccl_proxies;
  std::vector<uint64_t> uccl_d2h_channel_addrs_host;
  uint64_t* uccl_d2h_channel_addrs_device = nullptr;
  uint64_t* uccl_signal_shadow_device = nullptr;

  // Direct intra-node host-window descriptor.
  // Default ON when the shared host window is in use: the sender warp
  // writes directly into the peer's host-pinned slice (one PCIe pass)
  // instead of staging via the proxy memcpy (two DDR passes). Latency
  // is the same on idle systems (the proxy memcpy is fully overlapped
  // with PCIe), but DDR bandwidth and proxy CPU are freed for other
  // workloads. Set EP_UCCL_INTRANODE_DIRECT=0 to fall back to the
  // proxy-memcpy path for A/B comparison.
  bool uccl_intranode_direct_enabled() const {
    if (!uccl_use_shared_window) return false;
    if (char const* e = std::getenv("EP_UCCL_INTRANODE_DIRECT")) {
      return std::atoi(e) != 0;
    }
    return true;
  }
  uint64_t uccl_shared_per_rank_bytes() const {
    return uccl_intranode_direct_enabled()
               ? static_cast<uint64_t>(uccl_window.per_rank_size)
               : 0u;
  }
  int uccl_intranode_local_world_size() const {
    return uccl_intranode_direct_enabled() ? local_world_size : 0;
  }
  int uccl_intranode_my_local_rank() const {
    return uccl_intranode_direct_enabled() ? local_rank : -1;
  }
  int uccl_intranode_node_idx() const {
    return uccl_intranode_direct_enabled() ? node_idx : -1;
  }

  // Some EP hybrid mode settings
  static constexpr int kNumMaxChannelsPerSM = 8;
  static constexpr int kNumMaxSMs = 160;
  static constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * kNumMaxSMs;

  static int get_num_buffer_channels(bool const& prefer_overlap_with_compute) {
    int const max_channels_per_sm =
        prefer_overlap_with_compute ? kNumMaxChannelsPerSM : 4;
    return jit::device_runtime->get_num_sms() * max_channels_per_sm;
  }

  static int get_num_channel_token_slots(int const& num_max_tokens_per_rank,
                                         int const& max_num_channels) {
    return max_num_channels *
           math::ceil_div(num_max_tokens_per_rank, max_num_channels);
  }

  // Some Engram storage settings
  int num_engram_entries = 0, engram_hidden = 0;
  int64_t num_engram_storage_bytes = 0;
  int64_t num_engram_recv_bytes = 0;

  // PP settings
  int prev_rank_idx = 0, next_rank_idx = 0;
  int64_t num_max_pp_tensor_bytes = 0;
  int num_max_pp_inflight_tensors = 0;

  // AGRS session settings
  int64_t num_max_agrs_session_bytes = 0;
  int num_max_agrs_per_session = 0;
  int agrs_session_idx = 0;
  bool agrs_in_session = false;

  // AGRS in-session settings
  int64_t agrs_buffer_offset = 0;
  int agrs_buffer_slot_idx = 0;

 public:
  ElasticBuffer(int const& rank_idx, int const& num_ranks,
                int64_t const& nccl_comm, int64_t const& num_buffer_bytes,
                bool const& deterministic, bool const& allow_hybrid_mode,
                bool const& allow_multiple_reduction,
                bool const& prefer_overlap_with_compute, int const& sl_idx,
                int const& num_allocated_qps, int const& num_cpu_timeout_secs,
                int const& num_gpu_timeout_secs, bool const& explicitly_destroy,
                int const& node_idx, int const& local_rank,
                int const& local_world_size, std::string const& uccl_shm_id)
      : num_buffer_bytes(num_buffer_bytes),
        explicitly_destroy(explicitly_destroy),
        comm_stream(get_global_comm_stream()),
        deterministic(deterministic),
        allow_hybrid_mode(allow_hybrid_mode),
        allow_multiple_reduction(allow_multiple_reduction),
        prefer_overlap_with_compute(prefer_overlap_with_compute) {
    this->rank_idx = rank_idx;
    this->num_ranks = num_ranks;
    this->node_idx = node_idx;
    this->local_rank = local_rank;
    this->local_world_size = local_world_size;
    this->uccl_shm_id = uccl_shm_id.empty() ? "default" : uccl_shm_id;
    bool const request_uccl_proxy = get_env<int>("EP_USE_UCCL_PROXY", 0) != 0 or
                                    get_env<int>("UCCL_FORCE_NO_GDR", 0) != 0;
    bool const force_no_gdr = get_env<int>("UCCL_FORCE_NO_GDR", 0) != 0;
    bool const force_host_window = get_env<int>("EP_FORCE_HOST_WINDOW", 0) != 0;
    auto const total_window_bytes =
        layout::WorkspaceLayout::get_num_bytes() + num_buffer_bytes;
    int device_idx = 0;
    if (request_uccl_proxy) {
      CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
      // PCIe/no-NVLink local GPU-window traffic is not a useful fast path
      // for UCCL proxy mode; use the shared host window whenever multiple
      // ranks share a node, while leaving EP_UCCL_FORCE_GPU_WINDOW for
      // explicit diagnostics.
      bool const prefer_shared_host_window =
          local_world_size > 1 &&
          get_env<int>("EP_UCCL_FORCE_GPU_WINDOW", 0) == 0;
      uccl_use_host_window =
          force_no_gdr || force_host_window || prefer_shared_host_window;
      if (!uccl_use_host_window)
        uccl_use_host_window =
            !can_register_gpu_memory_for_rdma(device_idx, total_window_bytes);
      use_uccl_proxy = true;
      setenv("EP_UCCL_PROXY_ACTIVE", "1", 1);
      if (get_env<int>("EP_UCCL_DEBUG", 0)) {
        fprintf(stderr,
                "[ElasticBuffer][UCCL] rank=%d local=%d/%d window=%s "
                "force_no_gdr=%d force_host=%d prefer_shared=%d\n",
                rank_idx, local_rank, local_world_size,
                uccl_use_host_window ? "host" : "gpu",
                static_cast<int>(force_no_gdr),
                static_cast<int>(force_host_window),
                static_cast<int>(prefer_shared_host_window));
      }
    } else {
      use_uccl_proxy = false;
      setenv("EP_UCCL_PROXY_ACTIVE", "0", 1);
    }

    // Init NCCL runtime
    static constexpr int kBufferAlignment = 16;
    this->nccl_context = std::make_shared<nccl::NCCLSymmetricMemoryContext>(
        nccl_comm, num_ranks, rank_idx, total_window_bytes, kBufferAlignment,
        allow_hybrid_mode, sl_idx, num_allocated_qps);

    // Timeout
    this->num_cpu_timeout_secs = num_cpu_timeout_secs;
    this->num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs);
    this->num_gpu_timeout_cycles *= jit::device_runtime->get_clock_rate();

    // Assign workspaces and buffers
    workspace = this->nccl_context->mapped_window_ptr;
    if (use_uccl_proxy) {
      EP_HOST_ASSERT(get_env<int>("EP_FORCE_NO_NVLINK", 0) != 0 and
                     "UCCL proxy mode requires LITE_EP_NVLINK=0");
      EP_HOST_ASSERT(local_world_size > 0);
      EP_HOST_ASSERT(local_rank >= 0 and local_rank < local_world_size);
      if (uccl_use_host_window) {
        uccl_use_shared_window = local_world_size > 1;
        if (uccl_use_shared_window) {
          uccl_window = allocate_shared_buffer(
              ("deepepv2_window_" + this->uccl_shm_id).c_str(),
              total_window_bytes, local_rank, local_world_size, device_idx,
              layout::WorkspaceLayout::get_num_bytes());
            constexpr size_t kHostPageSize = 4096;
            size_t const atomic_per_rank_bytes =
              ((static_cast<size_t>(kAtomicBufferSize) + kHostPageSize - 1) /
               kHostPageSize) *
              kHostPageSize;
          uccl_atomic_window = allocate_shared_buffer(
              ("deepepv2_atomic_" + this->uccl_shm_id).c_str(),
              atomic_per_rank_bytes, local_rank, local_world_size, device_idx,
              kAtomicBufferSize);
          workspace = uccl_window.my_device_ptr();
        } else {
          CUDA_RUNTIME_CHECK(cudaHostAlloc(
              &uccl_host_window, total_window_bytes, cudaHostAllocMapped));
          std::memset(uccl_host_window, 0, total_window_bytes);
          CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&uccl_host_window_device,
                                                      uccl_host_window, 0));
          workspace = uccl_host_window_device;
        }
      } else {
        CUDA_RUNTIME_CHECK(cudaMalloc(&uccl_gpu_window, total_window_bytes));
        workspace = uccl_gpu_window;
      }

      int const num_nodes = std::max(1, num_ranks / local_world_size);
      auto const proxy_buffer_addr =
          uccl_use_host_window
              ? (uccl_use_shared_window
                     ? reinterpret_cast<uintptr_t>(uccl_window.my_host_ptr())
                     : reinterpret_cast<uintptr_t>(uccl_host_window))
              : reinterpret_cast<uintptr_t>(workspace);
      auto const shared_rdma_base =
          uccl_use_shared_window
              ? reinterpret_cast<uintptr_t>(uccl_window.mmap_ptr)
              : uintptr_t{0};
      auto const shared_atomic_base =
          uccl_use_shared_window
              ? reinterpret_cast<uintptr_t>(uccl_atomic_window.mmap_ptr)
              : uintptr_t{0};
      auto const proxy_nbytes = uccl_use_shared_window
                                    ? uccl_window.per_rank_size
                                    : total_window_bytes;
      auto const shared_rdma_per_rank =
          uccl_use_shared_window ? uccl_window.per_rank_size : size_t{0};
      auto const shared_atomic_per_rank =
          uccl_use_shared_window ? uccl_atomic_window.per_rank_size : size_t{0};
      uccl_proxies.reserve(kNumProxyThs);
      for (int i = 0; i < kNumProxyThs; ++i) {
        auto proxy = std::make_unique<UcclProxy>(
            i, proxy_buffer_addr, proxy_nbytes, rank_idx, node_idx, local_rank,
            0, num_ranks, num_nodes, true, false, uccl_use_host_window,
            shared_rdma_base, shared_rdma_per_rank, shared_atomic_base,
            shared_atomic_per_rank, local_world_size, false);
        auto addrs = proxy->get_d2h_channel_addrs();
        uccl_d2h_channel_addrs_host.insert(uccl_d2h_channel_addrs_host.end(),
                                           addrs.begin(), addrs.end());
        uccl_proxies.emplace_back(std::move(proxy));
      }
      CUDA_RUNTIME_CHECK(
          cudaMalloc(&uccl_d2h_channel_addrs_device,
                     uccl_d2h_channel_addrs_host.size() * sizeof(uint64_t)));
      CUDA_RUNTIME_CHECK(cudaMemcpy(
          uccl_d2h_channel_addrs_device, uccl_d2h_channel_addrs_host.data(),
          uccl_d2h_channel_addrs_host.size() * sizeof(uint64_t),
          cudaMemcpyHostToDevice));
      CUDA_RUNTIME_CHECK(cudaMalloc(
          &uccl_signal_shadow_device,
          layout::WorkspaceLayout::kNumGinBarrierTags *
              layout::WorkspaceLayout::kNumMaxRanks * sizeof(uint64_t)));
      CUDA_RUNTIME_CHECK(cudaMemset(
          uccl_signal_shadow_device, 0,
          layout::WorkspaceLayout::kNumGinBarrierTags *
              layout::WorkspaceLayout::kNumMaxRanks * sizeof(uint64_t)));
    }
    workspace_layout_wo_expert = std::make_shared<layout::WorkspaceLayout>(
        workspace, nccl_context->num_scaleout_ranks,
        nccl_context->num_scaleup_ranks, 0);
    buffer = static_cast<uint8_t*>(workspace) +
             layout::WorkspaceLayout::get_num_bytes();
    CUDA_RUNTIME_CHECK(
        cudaMemset(workspace, 0, layout::WorkspaceLayout::get_num_bytes()));

    // Allocate host workspaces
    CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace,
                                      layout::WorkspaceLayout::get_num_bytes(),
                                      cudaHostAllocMapped));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
    std::memset(host_workspace, 0, layout::WorkspaceLayout::get_num_bytes());

    // We should call a barrier at the end
    // The barrier should be called by Python `dist.barrier`
    // NOTES: do not call our barrier, as the workspace is not ready yet
  }

  ~ElasticBuffer() noexcept(false) {
    if (not explicitly_destroy) destroy();

    if (not destroyed) {
      printf(
          "`destroy()` is not called before DeepEP elastic buffer destruction, "
          "which can leak resources.\n");
      fflush(stdout);
    }
  }

  void destroy() {
    EP_HOST_ASSERT(not destroyed);

    // Finish all works on all GPUs. UCCL mode uses Python-side CPU group
    // synchronization because the native GPU barrier still depends on Gin
    // signal semantics.
    if (use_uccl_proxy) {
      CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    } else {
      barrier(true, true);
    }

    if (uccl_proxies_started) {
      for (auto& proxy : uccl_proxies) proxy->stop();
      uccl_proxies_started = false;
    }

    if (uccl_d2h_channel_addrs_device != nullptr) {
      CUDA_RUNTIME_CHECK(cudaFree(uccl_d2h_channel_addrs_device));
      uccl_d2h_channel_addrs_device = nullptr;
    }
    if (uccl_signal_shadow_device != nullptr) {
      CUDA_RUNTIME_CHECK(cudaFree(uccl_signal_shadow_device));
      uccl_signal_shadow_device = nullptr;
    }
    if (uccl_gpu_window != nullptr) {
      CUDA_RUNTIME_CHECK(cudaFree(uccl_gpu_window));
      uccl_gpu_window = nullptr;
    }
    if (uccl_host_window != nullptr) {
      CUDA_RUNTIME_CHECK(cudaFreeHost(uccl_host_window));
      uccl_host_window = nullptr;
      uccl_host_window_device = nullptr;
    }
    // Deallocate host workspaces
    CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));

    if (use_uccl_proxy && uccl_use_shared_window) {
      free_shared_buffer(uccl_atomic_window);
      free_shared_buffer(uccl_window);
    }

    // Destroy NCCL context
    nccl_context->finalize();

    // Cannot use anymore
    destroyed = true;
  }

  torch::Stream get_comm_stream() const { return comm_stream; }

  bool is_uccl_proxy_active() const { return use_uccl_proxy; }

  std::tuple<int, int> get_physical_domain_size() const {
    return {nccl_context->num_rdma_ranks, nccl_context->num_nvl_ranks};
  }

  std::tuple<int, int> get_logical_domain_size() const {
    return {nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks};
  }

  std::tuple<uint64_t, int64_t, std::vector<int>, std::vector<uint64_t>,
             pybind11::bytes>
  get_uccl_local_metadata() const {
    EP_HOST_ASSERT(use_uccl_proxy);
    std::vector<int> ports;
    ports.reserve(uccl_proxies.size());
    for (auto const& proxy : uccl_proxies)
      ports.push_back(proxy->get_listen_port());
    std::string cuda_ipc_handle_bytes;
    if (!uccl_use_host_window) {
      cudaIpcMemHandle_t handle;
      CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&handle, workspace));
      cuda_ipc_handle_bytes.assign(reinterpret_cast<char const*>(&handle),
                                   sizeof(handle));
    }
    return {
        uccl_use_host_window
            ? (uccl_use_shared_window
                   ? reinterpret_cast<uint64_t>(uccl_window.my_host_ptr())
                   : reinterpret_cast<uint64_t>(uccl_host_window))
            : reinterpret_cast<uint64_t>(workspace),
        uccl_use_host_window
            ? (uccl_use_shared_window
                   ? static_cast<int64_t>(uccl_window.per_rank_size)
                   : static_cast<int64_t>(
                         layout::WorkspaceLayout::get_num_bytes() +
                         num_buffer_bytes))
            : static_cast<int64_t>(layout::WorkspaceLayout::get_num_bytes() +
                                   num_buffer_bytes),
        ports, uccl_d2h_channel_addrs_host,
        pybind11::bytes(cuda_ipc_handle_bytes)};
  }

  void sync_uccl_peers(std::vector<int> const& ranks,
                       std::vector<uint64_t> const& ptrs,
                       std::vector<int64_t> const& nbytes,
                       std::vector<std::string> const& ips,
                       std::vector<std::vector<int>> const& listen_ports,
                       std::vector<std::string> const& cuda_ipc_handles) {
    EP_HOST_ASSERT(use_uccl_proxy);
    EP_HOST_ASSERT(not uccl_proxies_started);
    EP_HOST_ASSERT(ranks.size() == static_cast<size_t>(num_ranks));
    EP_HOST_ASSERT(ptrs.size() == ranks.size());
    EP_HOST_ASSERT(nbytes.size() == ranks.size());
    EP_HOST_ASSERT(ips.size() == ranks.size());
    EP_HOST_ASSERT(listen_ports.size() == ranks.size());
    EP_HOST_ASSERT(cuda_ipc_handles.size() == ranks.size());

    std::vector<PeerMeta> peers(num_ranks);
    for (size_t i = 0; i < ranks.size(); ++i) {
      EP_HOST_ASSERT(ranks[i] >= 0 and ranks[i] < num_ranks);
      auto& peer = peers[ranks[i]];
      peer.rank = ranks[i];
      peer.ptr = static_cast<uintptr_t>(ptrs[i]);
      peer.nbytes = static_cast<size_t>(nbytes[i]);
      peer.ip = ips[i];
      if (cuda_ipc_handles[i].size() == sizeof(cudaIpcMemHandle_t)) {
        peer.has_cuda_ipc = true;
        std::memcpy(&peer.cuda_ipc_handle, cuda_ipc_handles[i].data(),
                    sizeof(cudaIpcMemHandle_t));
      }
      EP_HOST_ASSERT(listen_ports[i].size() == kNumProxyThs);
      for (int j = 0; j < kNumProxyThs; ++j)
        peer.listen_ports[j] = listen_ports[i][j];
    }

    for (auto& proxy : uccl_proxies) {
      proxy->set_peers_meta(peers);
      proxy->start_dual();
    }
    for (auto& proxy : uccl_proxies) proxy->wait_until_ready();
    uccl_proxies_started = true;
  }

  // ReSharper disable once CppMemberFunctionMayBeStatic
  void barrier(bool const& use_comm_stream, bool const& with_cpu_sync) const {
    auto const compute_stream = at::cuda::getCurrentCUDAStream();
    auto const stream = use_comm_stream ? comm_stream : compute_stream;
    if (use_comm_stream) stream_wait(comm_stream, compute_stream);

    // Wait all streams to finish on this GPU
    if (with_cpu_sync) CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    // Launch GPU barrier
    launch_barrier(
        nccl_context->dev_comm, nccl_context->window, workspace,
        nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
        nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
        num_gpu_timeout_cycles, nccl_context->is_scaleup_nvlink,
        uccl_d2h_channel_addrs_device,
        static_cast<int>(uccl_d2h_channel_addrs_host.size()),
        uccl_signal_shadow_device, uccl_shared_per_rank_bytes(),
        uccl_intranode_local_world_size(), uccl_intranode_my_local_rank(),
        uccl_intranode_node_idx(), stream);

    // Let CPU wait
    if (with_cpu_sync) CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    // Compute stream should also wait for the barrier
    if (use_comm_stream) stream_wait(compute_stream, comm_stream);
  }

  void engram_write(torch::Tensor const& storage) {
    // Ensure previous fetch are finished
    barrier(false, true);

    auto const compute_stream = at::cuda::getCurrentCUDAStream();

    // Check storage
    auto const [num_entries, hidden] = get_shape<2>(storage);
    EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());
    num_engram_entries = num_entries, engram_hidden = hidden;

    // Write storage and ensure the received buffer is aligned
    EP_HOST_ASSERT(storage.nbytes() <= num_buffer_bytes);
    CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(buffer, storage.data_ptr(), storage.nbytes(),
                        cudaMemcpyDeviceToDevice, compute_stream));
    num_engram_storage_bytes = math::align<int64_t>(storage.nbytes(), 32);
    num_engram_recv_bytes = num_buffer_bytes - num_engram_storage_bytes;

    // Ensure data is visible for all ranks
    barrier(false, true);
  }

  std::function<torch::Tensor()> engram_fetch(torch::Tensor const& indices,
                                              int num_qps) const {
    auto const [num_tokens] = get_shape<1>(indices);
    EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);
    EP_HOST_ASSERT(indices.is_cuda() and indices.is_contiguous());
    EP_HOST_ASSERT(num_tokens * engram_hidden * sizeof(nv_bfloat16) <=
                   num_engram_recv_bytes);

    // Calculate a QP count
    if (num_qps == 0) num_qps = nccl_context->num_allocated_qps;

    // Return tensor from the raw buffer
    EP_HOST_ASSERT(num_engram_entries > 0);
    auto const fetched = torch::from_blob(
        math::advance_ptr(buffer, num_engram_storage_bytes),
        {num_tokens, engram_hidden},
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

    // Last issued Gin requests
    auto const last_gin_requests = torch::empty(
        {nccl_context->num_ranks * num_qps, sizeof(ncclGinRequest_t)},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    // Launch the fetch kernel
    launch_engram_fetch(
        nccl_context->dev_comm, nccl_context->window, buffer,
        fetched.data_ptr(), indices.data_ptr<int>(),
        static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
        num_engram_entries, engram_hidden, num_tokens, nccl_context->num_ranks,
        num_qps, at::cuda::getCurrentCUDAStream());

    return [=, this]() {
      // Wait for all RDMA gets to complete
      launch_engram_fetch_wait(
          static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
          nccl_context->dev_comm, nccl_context->window, nccl_context->num_ranks,
          num_qps, at::cuda::getCurrentCUDAStream());
      return fetched;
    };
  }

  void pp_set_config(int64_t const& num_max_tensor_bytes,
                     int const& num_max_inflight_tensors) {
    // Flush previous operations
    barrier(false, true);

    EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);
    EP_HOST_ASSERT(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2 <=
                   num_buffer_bytes);
    this->prev_rank_idx =
        (nccl_context->rank_idx + nccl_context->num_ranks - 1) %
        nccl_context->num_ranks;
    this->next_rank_idx =
        (nccl_context->rank_idx + 1) % nccl_context->num_ranks;
    this->num_max_pp_tensor_bytes =
        math::align<int64_t>(num_max_tensor_bytes, 32);
    this->num_max_pp_inflight_tensors = num_max_inflight_tensors;
  }

  void pp_send(torch::Tensor const& x, int const& dst_rank_idx,
               int const& num_sms) const {
    EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and
                   num_max_pp_inflight_tensors > 0);
    EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and
                   x.nbytes() <= num_max_pp_tensor_bytes);
    EP_HOST_ASSERT(dst_rank_idx == prev_rank_idx or
                   dst_rank_idx == next_rank_idx);

    launch_pp_send(nccl_context->dev_comm, nccl_context->window, x.data_ptr(),
                   x.nbytes(), buffer, workspace, nccl_context->rank_idx,
                   dst_rank_idx, nccl_context->num_ranks,
                   num_max_pp_tensor_bytes, num_max_pp_inflight_tensors,
                   num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
                   num_gpu_timeout_cycles,
                   jit::device_runtime->get_num_smem_bytes(),
                   at::cuda::getCurrentCUDAStream());
  }

  void pp_recv(torch::Tensor const& x, int const& src_rank_idx,
               int const& num_sms) const {
    EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and
                   num_max_pp_inflight_tensors > 0);
    EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and
                   x.nbytes() <= num_max_pp_tensor_bytes);
    EP_HOST_ASSERT(src_rank_idx == prev_rank_idx or
                   src_rank_idx == next_rank_idx);

    launch_pp_recv(nccl_context->dev_comm, nccl_context->window, x.data_ptr(),
                   x.nbytes(), buffer, workspace, nccl_context->rank_idx,
                   src_rank_idx, nccl_context->num_ranks,
                   num_max_pp_tensor_bytes, num_max_pp_inflight_tensors,
                   num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
                   num_gpu_timeout_cycles,
                   jit::device_runtime->get_num_smem_bytes(),
                   at::cuda::getCurrentCUDAStream());
  }

  void agrs_set_config(int64_t const& num_max_session_bytes,
                       int const& new_num_max_agrs_per_session) {
    // Flush previous operations
    barrier(true, true);

    EP_HOST_ASSERT(nccl_context->num_ranks > 1);
    EP_HOST_ASSERT(num_max_session_bytes > 0 and
                   new_num_max_agrs_per_session > 0);
    EP_HOST_ASSERT(num_max_session_bytes <= num_buffer_bytes);
    EP_HOST_ASSERT(new_num_max_agrs_per_session <=
                   layout::WorkspaceLayout::kNumMaxInflightAGRS);
    EP_HOST_ASSERT(nccl_context->num_nvl_ranks == nccl_context->num_ranks);
    this->num_max_agrs_session_bytes =
        math::align<int64_t>(num_max_session_bytes, 32);
    this->num_max_agrs_per_session = new_num_max_agrs_per_session;
  }

  void create_agrs_session() {
    EP_HOST_ASSERT(not agrs_in_session);
    agrs_in_session = true;
    agrs_buffer_offset = 0;
    agrs_buffer_slot_idx = 0;
    agrs_session_idx += 1;
  }

  void destroy_agrs_session() {
    // Must be in a session
    EP_HOST_ASSERT(agrs_in_session);
    agrs_in_session = false;

    // Wait compute stream
    stream_wait(comm_stream, at::cuda::getCurrentCUDAStream());

    // Notify that the buffer is now available & Wait for the buffer to be ready
    // NOTES: self-wait is guaranteed by in-stream order
    std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
    std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
    for (int i = 0; i < nccl_context->num_ranks - 1; ++i) {
      auto const dst_rank_idx =
          (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
      write_ptrs[i] = static_cast<int*>(nccl_context->get_sym_ptr(
          workspace_layout_wo_expert->get_agrs_session_signal_ptr(
              nccl_context->rank_idx),
          dst_rank_idx));
      wait_ptrs[i] =
          workspace_layout_wo_expert->get_agrs_session_signal_ptr(dst_rank_idx);
    }
    cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs,
                                        agrs_session_idx);
  }

  std::vector<torch::Tensor> agrs_get_inplace_tensor(
      std::vector<int64_t> const& num_bytes_list) const {
    EP_HOST_ASSERT(num_bytes_list.size() >= 1);
    EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and
                   num_max_agrs_per_session > 0 and agrs_in_session);

    std::vector<torch::Tensor> out;
    out.reserve(num_bytes_list.size());
    int64_t offset = agrs_buffer_offset;
    for (auto const& num_bytes : num_bytes_list) {
      EP_HOST_ASSERT(num_bytes % 32 == 0);
      EP_HOST_ASSERT(
          offset + num_bytes * nccl_context->num_ranks <=
              num_max_agrs_session_bytes and
          agrs_buffer_slot_idx < num_max_agrs_per_session and
          "Not enough session buffer size. Did you forget to flush session?");
      out.push_back(torch::from_blob(
          math::advance_ptr(buffer,
                            offset + num_bytes * nccl_context->rank_idx),
          {num_bytes},
          torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)));
      offset += num_bytes * nccl_context->num_ranks;
    }
    return out;
  }

  std::pair<std::vector<torch::Tensor>, std::function<void()>> all_gather(
      std::vector<torch::Tensor> const& tensors) {
    int const num_tensors = tensors.size();
    EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and
                   num_max_agrs_per_session > 0 and agrs_in_session);
    EP_HOST_ASSERT(num_tensors >= 1);

    int num_copies = 0;
    std::vector<int64_t> offset(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
      auto const& x = tensors[i];
      EP_HOST_ASSERT(x.is_contiguous());
      EP_HOST_ASSERT(x.is_cuda() and x.nbytes() % 32 == 0);

      auto const x_offset = math::ptr_diff(x.data_ptr(), buffer);
      bool const is_inplace =
          0 <= x_offset and x_offset < num_max_agrs_session_bytes;
      offset[i] = agrs_buffer_offset;
      num_copies += nccl_context->num_ranks - is_inplace;
      agrs_buffer_offset += x.nbytes() * nccl_context->num_ranks;
      EP_HOST_ASSERT(
          not is_inplace or
          x.data_ptr() ==
              math::advance_ptr(
                  buffer, offset[i] + x.nbytes() * nccl_context->rank_idx));
    }
    EP_HOST_ASSERT(
        agrs_buffer_offset <= num_max_agrs_session_bytes and
        agrs_buffer_slot_idx < num_max_agrs_per_session and
        "Not enough session buffer size. Did you forget to flush session?");

    // Wait compute stream
    auto const compute_stream = at::cuda::getCurrentCUDAStream();
    stream_wait(comm_stream, compute_stream);

    // Send data to all ranks
    std::vector<size_t> sizes(num_copies);
    std::vector<void*> dst_ptrs(num_copies), src_ptrs(num_copies);
    int count = 0;
    for (int i = 0; i < nccl_context->num_ranks; ++i) {
      for (int j = 0; j < num_tensors; ++j) {
        auto const& x = tensors[j];
        auto const dst_rank_idx =
            (nccl_context->rank_idx + i) % nccl_context->num_ranks;
        void* src_ptr = x.data_ptr();
        void* dst_ptr = nccl_context->get_sym_ptr(
            math::advance_ptr(buffer,
                              offset[j] + x.nbytes() * nccl_context->rank_idx),
            dst_rank_idx);
        if (src_ptr != dst_ptr) {
          src_ptrs[count] = src_ptr;
          dst_ptrs[count] = dst_ptr;
          sizes[count] = x.nbytes();
          count += 1;
        }
      }
    }
    cudaMemcpyAttributes attrs = {
        .srcAccessOrder = cudaMemcpySrcAccessOrderStream,
        .flags = cudaMemcpyFlagPreferOverlapWithCompute};
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
    CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(),
                                            sizes.data(), num_copies, attrs,
                                            comm_stream));
#else
    CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(),
                                            sizes.data(), num_copies, attrs,
                                            nullptr, comm_stream));
#endif

    // Wait for data from other ranks
    int const current_session = agrs_session_idx;
    int const slot_idx = agrs_buffer_slot_idx;
    agrs_buffer_slot_idx += 1;
    std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
    std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
    for (int i = 0; i < nccl_context->num_ranks - 1; ++i) {
      auto const dst_rank_idx =
          (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
      write_ptrs[i] = nccl_context->get_sym_ptr(
          workspace_layout_wo_expert->get_agrs_recv_signal_ptr(
              slot_idx, nccl_context->rank_idx),
          dst_rank_idx);
      wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_recv_signal_ptr(
          slot_idx, dst_rank_idx);
    }
    cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs,
                                        current_session);

    // Build output tensors eagerly
    std::vector<torch::Tensor> out(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
      auto shape = tensors[i].sizes().vec();
      shape.insert(shape.begin(), nccl_context->num_ranks);
      out[i] = torch::from_blob(math::advance_ptr(buffer, offset[i]), shape,
                                tensors[i].options());
    }

    // Return tensors and a handle to wait for data arrival
    auto const event = EventHandle(comm_stream);
    auto handle = [=, this]() {
      EP_HOST_ASSERT(compute_stream == at::cuda::getCurrentCUDAStream());
      EP_HOST_ASSERT(agrs_in_session and
                     current_session == this->agrs_session_idx);
      stream_wait(compute_stream, event);
    };
    return {std::move(out), std::move(handle)};
  }

  torch::cuda::CUDAStream stream_control_prologue(
      std::optional<EventHandle> const& previous_event,
      bool const& allocate_on_comm_stream,
      bool const& async_with_compute_stream) const {
    // Allocate all tensors on communication stream if set
    // NOTES: do not allocate tensors upfront!
    auto const compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(comm_stream);

    // Assertion for safety
    // `previous_event` implicitly means the overlapping computation kernels are
    // launched first, in order not to use the memory on the compute stream, we
    // must allocate on the communication stream. If you launch the
    // communication kernels firstly, then `previous_event` must be unnecessary.
    if (previous_event.has_value()) EP_HOST_ASSERT(allocate_on_comm_stream);

    // Wait previous tasks to finish
    if (previous_event.has_value()) {
      stream_wait(comm_stream, previous_event.value());
    } else {
      stream_wait(comm_stream, compute_stream);
    }
    return compute_stream;
  }

  void stream_control_before_epilogue(
      std::optional<EventHandle> const& previous_event_before_epilogue) const {
    if (previous_event_before_epilogue.has_value())
      stream_wait(comm_stream, previous_event_before_epilogue.value());
  }

  std::optional<EventHandle> stream_control_epilogue(
      std::vector<std::optional<torch::Tensor>> const& tensors,
      at::cuda::CUDAStream const& compute_stream,
      bool const& allocate_on_comm_stream,
      bool const& async_with_compute_stream) const {
    // Ensure memory access safety between two streams
    std::optional<EventHandle> event;
    if (async_with_compute_stream) {
      event = EventHandle(comm_stream);

      // NOTES: this environment only applies to V2 APIs
      if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
        event->tensors_to_record = tensors;
      } else {
        for (auto& t : tensors)
          if (t.has_value()) {
            t->record_stream(compute_stream);
            t->record_stream(comm_stream);
          }
      }
    } else {
      stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

    // The CUDA event marking the finishing
    return event;
  }

  static int64_t get_dispatch_buffer_size(
      int const& num_max_tokens_per_rank, int const& hidden,
      int const& num_sf_packs, int const& num_topk, int const& elem_size,
      int const& num_scaleout_ranks, int const& num_scaleup_ranks,
      bool const& is_scaleup_nvlink,
      int const& max_num_channels = kNumMaxChannels) {
    auto const num_ranks = num_scaleup_ranks * num_scaleout_ranks;
    auto const token_layout =
        get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);

    if (num_scaleout_ranks == 1) {
      // Direct dispatch
      auto const send_buffer_layout = layout::BufferLayout<false>(
          token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);
      auto const recv_buffer_layout = layout::BufferLayout<false>(
          token_layout, num_ranks, num_max_tokens_per_rank);
      return send_buffer_layout.get_num_bytes() +
             recv_buffer_layout.get_num_bytes();
    } else {
      // Hybrid dispatch
      auto const scaleup_recv_buffer = layout::BufferLayout<false>(
          token_layout, num_scaleup_ranks,
          num_scaleout_ranks * num_max_tokens_per_rank);
      auto const scaleout_send_buffer =
          layout::BufferLayout<false>(token_layout, 1, num_max_tokens_per_rank);
      auto const scaleout_recv_buffer = layout::BufferLayout<false>(
          token_layout, num_scaleout_ranks,
          get_num_channel_token_slots(num_max_tokens_per_rank,
                                      max_num_channels));
      return scaleup_recv_buffer.get_num_bytes() +
             scaleout_send_buffer.get_num_bytes() +
             scaleout_recv_buffer.get_num_bytes();
    }
  }

  static int64_t get_combine_buffer_size(
      int const& num_max_tokens_per_rank, int const& hidden,
      int const& num_topk, int const& num_scaleout_ranks,
      int const& num_scaleup_ranks, bool const& is_scaleup_nvlink,
      bool const& allow_multiple_reduction,
      int const& max_num_channels = kNumMaxChannels) {
    auto const num_ranks = num_scaleup_ranks * num_scaleout_ranks;
    auto const token_layout =
        get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

    if (num_scaleout_ranks == 1) {
      // Direct combine
      auto const num_tokens_in_layout =
          allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk;
      auto const send_buffer_layout = layout::BufferLayout<false>(
          token_layout, is_scaleup_nvlink ? 0 : num_ranks,
          // For single reduction cases, the maximum number of received tokens
          // is `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the
          // bad case of `do_expand=True`)
          num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));
      auto const recv_buffer_layout = layout::BufferLayout<false>(
          token_layout, num_tokens_in_layout, num_max_tokens_per_rank);
      return send_buffer_layout.get_num_bytes() +
             recv_buffer_layout.get_num_bytes();
    } else {
      // Hybrid combine
      int const num_tokens_in_scaleup_layout =
          allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk)
                                   : num_topk;
      int const num_tokens_in_scaleout_layout =
          allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk)
                                   : num_topk;
      auto const scaleup_recv_buffer = layout::BufferLayout<false>(
          token_layout, num_tokens_in_scaleup_layout,
          num_scaleout_ranks * num_max_tokens_per_rank);
      auto const scaleout_recv_buffer = layout::BufferLayout<false>(
          token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
      auto const scaleout_send_buffer = layout::BufferLayout<false>(
          token_layout, allow_multiple_reduction ? 1 : num_topk,
          /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
          num_scaleout_ranks * get_num_channel_token_slots(
                                   num_max_tokens_per_rank, max_num_channels));
      return scaleup_recv_buffer.get_num_bytes() +
             scaleout_send_buffer.get_num_bytes() +
             scaleout_recv_buffer.get_num_bytes();
    }
  }

  static int64_t calculate_buffer_size(
      int64_t const& nccl_comm, int const& num_max_tokens_per_rank,
      int const& hidden, int num_topk, bool const& use_fp8_dispatch,
      bool const& allow_hybrid_mode, bool const& allow_multiple_reduction,
      bool const& prefer_overlap_with_compute) {
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

    // The worst case SF bytes must be less than the main part
    EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

    // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to
    // calculate token size
    num_topk = num_topk == 0 ? 32 : num_topk;

    // Topology
    auto const [num_rdma_ranks, num_nvl_ranks] =
        nccl::get_physical_domain_size(nccl_comm);
    auto const [num_scaleout_ranks, num_scaleup_ranks] =
        nccl::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
    auto const is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;
    int const max_num_channels =
        num_scaleout_ranks > 1
            ? get_num_buffer_channels(prefer_overlap_with_compute)
            : kNumMaxChannels;

    // Dispatch size
    auto const elem_size =
        use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
    auto const num_sf_packs =
        use_fp8_dispatch ? math::ceil_div(hidden, 32)
                         : 0;  // An approximation for number of SF packs
    auto const num_dispatch_bytes = get_dispatch_buffer_size(
        num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,
        num_scaleout_ranks, num_scaleup_ranks, is_scaleup_nvlink,
        max_num_channels);

    // Combine layout
    auto const num_combine_bytes = get_combine_buffer_size(
        num_max_tokens_per_rank, hidden, num_topk, num_scaleout_ranks,
        num_scaleup_ranks, is_scaleup_nvlink, allow_multiple_reduction,
        max_num_channels);

    // Return the maximum of those layouts
    return std::max(num_dispatch_bytes, num_combine_bytes);
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>,
             std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::optional<torch::Tensor>, std::vector<int>, torch::Tensor,
             torch::Tensor, torch::Tensor, torch::Tensor,
             std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::optional<EventHandle>>
  dispatch(
      torch::Tensor const& x, std::optional<torch::Tensor> const& sf,
      torch::Tensor const& topk_idx,
      std::optional<torch::Tensor> const& topk_weights,
      std::optional<torch::Tensor> const& cumulative_local_expert_recv_stats,
      std::optional<int> const& cached_num_recv_tokens,
      std::optional<std::vector<int>> const&
          cached_num_recv_tokens_per_expert_list,
      std::optional<torch::Tensor> const&
          cached_psum_num_recv_tokens_per_scaleup_rank,
      std::optional<torch::Tensor> const&
          cached_psum_num_recv_tokens_per_expert,
      std::optional<torch::Tensor> const& cached_dst_buffer_slot_idx,
      std::optional<torch::Tensor> const& cached_token_metadata_at_forward,
      std::optional<torch::Tensor> const& cached_channel_linked_list,
      int const& num_max_tokens_per_rank, int const& num_experts,
      int const& expert_alignment, int const& num_sms, int const& num_qps,
      std::optional<EventHandle> const& previous_event,
      std::optional<EventHandle> const& previous_event_before_epilogue,
      bool const& async_with_compute_stream,
      bool const& allocate_on_comm_stream, bool const& do_handle_copy,
      bool const& do_cpu_sync, bool const& do_expand,
      bool const& use_tma_aligned_col_major_sf) const {
    // Check SM count
    EP_HOST_ASSERT(num_sms > 0);

    // Cached mode must have responding handles
    bool const cached_mode = cached_num_recv_tokens.has_value();
    if (cached_mode) {
      EP_HOST_ASSERT(cached_num_recv_tokens.has_value());
      EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());
      EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_scaleup_rank.has_value());
      EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());
      EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());

      // Hybrid kernels require more
      if (nccl_context->num_scaleout_ranks > 1) {
        EP_HOST_ASSERT(cached_token_metadata_at_forward.has_value());
        EP_HOST_ASSERT(cached_channel_linked_list.has_value());
      }
    }

    // Check data tensor
    auto const [num_tokens, hidden] = get_shape<2>(x);
    auto const num_hidden_bytes = hidden * static_cast<int>(x.element_size());
    auto const num_local_experts = num_experts / nccl_context->num_ranks;
    EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

    // Check SF stuffs
    int num_sf_packs = 0;
    void* sf_ptr = nullptr;
    int sf_token_stride = 0, sf_hidden_stride = 0;
    if (sf.has_value()) {
      // SF must be FP32 or packed UE8M0x4
      auto const [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());
      EP_HOST_ASSERT(num_tokens == num_tokens_);
      EP_HOST_ASSERT(sf->is_cuda());
      EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
      num_sf_packs = num_sf_packs_;
      sf_ptr = sf->data_ptr();
      sf_token_stride = sf->stride(0);
      sf_hidden_stride = sf->stride(1);
    }

    // Check top-k stuffs
    auto const [num_tokens_, num_topk] = get_shape<2>(topk_idx);
    EP_HOST_ASSERT(num_tokens == num_tokens_);
    EP_HOST_ASSERT(topk_idx.scalar_type() ==
                   c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());

    // Weights are optional for training backward
    float* topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
      auto const [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());
      EP_HOST_ASSERT(num_tokens == num_tokens__);
      EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
      topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // Expert receiving counter
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value()) {
      auto const [num_local_experts_] =
          get_shape<1>(cumulative_local_expert_recv_stats.value());
      EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                     cumulative_local_expert_recv_stats->is_contiguous());
      EP_HOST_ASSERT(num_local_experts == num_local_experts_);
      cumulative_local_expert_recv_stats_ptr =
          cumulative_local_expert_recv_stats->data_ptr<int>();
    }

    // Stream control
    // All new tensor allocations should happen after this
    auto const compute_stream = stream_control_prologue(
        previous_event, allocate_on_comm_stream, async_with_compute_stream);

    // The number of received tokens per expert
    // This is useful for expanding mode
    EP_HOST_ASSERT(num_experts % nccl_context->num_ranks == 0);
    auto psum_num_recv_tokens_per_expert =
        cached_psum_num_recv_tokens_per_expert.value_or(torch::Tensor());
    if (cached_mode) {
      auto const& [num_local_experts_] =
          get_shape<1>(psum_num_recv_tokens_per_expert);
      EP_HOST_ASSERT(num_local_experts == num_local_experts_);
      EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_cuda() and
                     psum_num_recv_tokens_per_expert.is_contiguous());
      EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() ==
                     torch::kInt);
    } else {
      // NOTES: for expand mode, the input is exclusive prefix sum, while for
      // non-expand, it is inclusive
      psum_num_recv_tokens_per_expert =
          torch::empty({num_local_experts + 1},
                       at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
    }

    // The prefix sum tensor of number of received tokens from each rank
    // Will also be used in combine as the dispatch handle
    auto psum_num_recv_tokens_per_scaleup_rank =
        cached_psum_num_recv_tokens_per_scaleup_rank.value_or(torch::Tensor());
    if (cached_mode) {
      auto const [num_scaleup_ranks] =
          get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
      EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);
      EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and
                     psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
      EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() ==
                     torch::kInt);
    } else {
      psum_num_recv_tokens_per_scaleup_rank =
          torch::empty({nccl_context->num_scaleup_ranks},
                       at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
    }

    // Decide number of channels by shared memory consumption
    // Only for hybrid version
    int num_channels_per_sm = 1, num_channels = 1;
    int const num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
    if (nccl_context->num_scaleout_ranks > 1) {
      auto const dispatch_token_layout = get_dispatch_token_layout(
          hidden, x.element_size(), num_sf_packs, num_topk);
      auto const combine_token_layout =
          get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
      EP_HOST_ASSERT(num_sms <= kNumMaxSMs);
      num_channels_per_sm = std::min<int>(
          (num_smem_bytes -
           get_num_notify_smem_bytes(nccl_context->num_ranks, num_experts)) /
              dispatch_token_layout.get_num_bytes<true>(),
          32 - kNumNotifyWarps);
      num_channels_per_sm = std::min<int>(
          num_smem_bytes / combine_token_layout.get_num_bytes<true>(),
          num_channels_per_sm);
      num_channels_per_sm = std::min<int>(
          /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);
      if (not prefer_overlap_with_compute)
        num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
      // Multi-node + multi-GPU-per-node configs (UCCL proxy mode) hit a
      // proxy WR-rate ceiling: every channel that exists fires one
      // "finish" PUT_VALUE per cross-rank peer per dispatch, regardless
      // of how many tokens it carried. With small `num_max_tokens_per_rank`
      // (e.g. 128) and many SMs (58 × 3 channels/SM = 174 channels),
      // ~88% of those signals come from channels that processed 0 or 1
      // tokens, but each still costs ~1 µs of proxy-thread `ibv_post_send`.
      // Capping channels-per-SM to 1 in this regime moves dispatch on
      // 2n×4g L4 from ~3 ms to ~1.9 ms (−36%) without hurting GPU-side
      // parallelism (the kernel's TMA warps were already idle most of
      // the time on under-loaded channels). Single-node configs are
      // not signal-bound (intra-node PUT_VALUE goes through host shm
      // memcpy, not the NIC) so they keep the original wide channel
      // layout to maximise GPU-side parallelism.
      bool const multi_node_multi_gpu = use_uccl_proxy and
                                        local_world_size > 1 and
                                        (num_ranks / local_world_size) > 1;
      if (multi_node_multi_gpu)
        num_channels_per_sm = std::min<int>(num_channels_per_sm, 1);
      // Explicit env override always wins (clamp to [1, kNumMaxChannelsPerSM]).
      if (char const* env = std::getenv("EP_NUM_CHANNELS_PER_SM")) {
        num_channels_per_sm =
            std::max(1, std::min(atoi(env), kNumMaxChannelsPerSM));
      }
      num_channels = num_sms * num_channels_per_sm;
      if (get_env<int>("EP_BUFFER_DEBUG"))
        printf(
            "Elastic buffer uses %d channels per SM "
            "(multi_node_multi_gpu=%d)\n",
            num_channels_per_sm, static_cast<int>(multi_node_multi_gpu));
    }

    // Non-hybrid mode handles
    std::optional<torch::Tensor> deterministic_rank_count_buffer = std::nullopt;
    auto dst_buffer_slot_idx =
        cached_dst_buffer_slot_idx.value_or(torch::Tensor());
    if (nccl_context->num_scaleout_ranks == 1) {
      if (cached_mode) {
        auto const [num_tokens__, num_topk_] =
            get_shape<2>(dst_buffer_slot_idx);
        EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk_);
        EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and
                       dst_buffer_slot_idx.is_contiguous());
        EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
      } else if (deterministic) {
        auto const prologue_num_sms = jit::device_runtime->get_num_sms();

        // Allocate new tensors
        deterministic_rank_count_buffer =
            torch::empty({prologue_num_sms, nccl_context->num_scaleup_ranks},
                         torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        dst_buffer_slot_idx =
            torch::empty({num_tokens, num_topk},
                         torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

        // Launch a kernel to preprocess the destination slot indices
        launch_dispatch_deterministic_prologue(
            topk_idx.data_ptr<topk_idx_t>(),
            deterministic_rank_count_buffer->data_ptr<int>(),
            dst_buffer_slot_idx.data_ptr<int>(), num_tokens,
            num_max_tokens_per_rank, num_experts, num_topk,
            nccl_context->scaleup_rank_idx, nccl_context->num_scaleup_ranks,
            prologue_num_sms, jit::device_runtime->get_num_smem_bytes(),
            comm_stream);
      } else {
        // Allocate a new tensor
        dst_buffer_slot_idx =
            torch::empty({num_tokens, num_topk},
                         torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
      }
    }

    // Hybrid mode handles
    std::optional<torch::Tensor> token_metadata_at_forward, channel_linked_list;
    int *token_metadata_at_forward_ptr = nullptr,
        *channel_linked_list_ptr = nullptr;
    if (nccl_context->num_scaleout_ranks > 1) {
      EP_HOST_ASSERT(not deterministic);

      // The token destination slot idx during forward
      // `[i, j, k, l]` means: from channel i from scale-out peer k, the j-th
      // token's index in the l-th rank buffer NOTES: Used primarily for cached
      // mode
      // TODO: May make it a linked list to remove the redundant info in
      // `token_metadata_at_forward`
      auto const num_max_tokens_per_channel =
          math::ceil_div(num_max_tokens_per_rank, num_channels);
      if (cached_mode) {
        auto const [num_channels_, num_scaleout_ranks_,
                    num_max_tokens_per_channel_, num_topk_] =
            get_shape<4>(dst_buffer_slot_idx);
        EP_HOST_ASSERT(
            num_channels == num_channels_ and
            nccl_context->num_scaleout_ranks == num_scaleout_ranks_ and
            num_max_tokens_per_channel == num_max_tokens_per_channel_ and
            num_topk == num_topk_);
        EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and
                       dst_buffer_slot_idx.is_contiguous());
        EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
      } else {
        dst_buffer_slot_idx = torch::empty(
            {num_channels, nccl_context->num_scaleout_ranks,
             num_max_tokens_per_channel, num_topk},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt));
      }

      // The token metadata during forward
      // `[i, j]` means: in channel i, the j-th forwarded token's metadata
      // Info contains:
      //   - Scaleout rank index and source token index in the original rank (0)
      //   - Whether the token is the last one in the chunk (1)
      //   - cached top-k scaleup peer indices (top-k)
      //   - each selections' destination slot indices (top-k)
      auto const num_max_forwarded_tokens =
          nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1;
      auto const num_forward_metadata_dims = 2 + num_topk * 2;
      if (cached_mode) {
        token_metadata_at_forward = cached_token_metadata_at_forward;
        auto const [num_channels_, num_max_forwarded_tokens_,
                    num_forward_metadata_dims_] =
            get_shape<3>(token_metadata_at_forward.value());
        EP_HOST_ASSERT(num_channels == num_channels_ and
                       num_max_forwarded_tokens == num_max_forwarded_tokens_ and
                       num_forward_metadata_dims == num_forward_metadata_dims_);
        EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and
                       token_metadata_at_forward->is_contiguous());
        EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);
      } else {
        token_metadata_at_forward = torch::empty(
            {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt));
      }
      token_metadata_at_forward_ptr =
          token_metadata_at_forward->data_ptr<int>();

      // Per-scaleup-peer-per-channel linked list
      // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's
      // index in the combine's input
      if (cached_mode) {
        channel_linked_list = cached_channel_linked_list;
        auto const [num_channels__, d1_, d2_] =
            get_shape<3>(channel_linked_list.value());
        channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
        EP_HOST_ASSERT(num_channels == num_channels__);
        EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks *
                                      num_max_tokens_per_channel +
                                  1);
        EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);
        EP_HOST_ASSERT(channel_linked_list->is_cuda() and
                       channel_linked_list->is_contiguous());
        EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
      } else {
        channel_linked_list = torch::empty(
            // Index 0 of the list means the starting item
            {num_channels,
             nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1,
             nccl_context->num_scaleup_ranks},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt));
      }
      channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
    }

    // Clone `topk_idx` for saving in the handle (to prevent users'
    // modification)
    auto copied_topk_idx = std::optional<torch::Tensor>();
    topk_idx_t* copied_topk_idx_ptr = nullptr;
    if (do_handle_copy and not cached_mode) {
      copied_topk_idx = torch::empty_like(topk_idx);
      copied_topk_idx_ptr = copied_topk_idx->data_ptr<topk_idx_t>();
    }

    // Check buffer size
    int const dispatch_buffer_channels =
        nccl_context->num_scaleout_ranks > 1 ? num_channels : kNumMaxChannels;
    EP_HOST_ASSERT(get_dispatch_buffer_size(
                       num_max_tokens_per_rank, hidden, num_sf_packs, num_topk,
                       x.element_size(), nccl_context->num_scaleout_ranks,
                       nccl_context->num_scaleup_ranks,
                       nccl_context->is_scaleup_nvlink,
                       dispatch_buffer_channels) <= num_buffer_bytes);

    // Ready and clean host workspace for this round
    auto const host_workspace_layout = layout::WorkspaceLayout(
        host_workspace, nccl_context->num_scaleout_ranks,
        nccl_context->num_scaleup_ranks, num_experts);
    std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(),
                nccl_context->num_scaleup_ranks, 0);
    std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(),
                num_local_experts, 0);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // Do dispatch into the buffers (with SM limitation)
    EP_HOST_ASSERT(num_sms <= jit::device_runtime->get_num_sms());
    launch_dispatch(
        x.data_ptr(), sf_ptr, topk_idx.data_ptr<topk_idx_t>(), topk_weights_ptr,
        copied_topk_idx_ptr, cumulative_local_expert_recv_stats_ptr,
        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
        psum_num_recv_tokens_per_expert.data_ptr<int>(),
        dst_buffer_slot_idx.data_ptr<int>(), token_metadata_at_forward_ptr,
        num_tokens, num_max_tokens_per_rank, hidden, x.element_size(),
        num_sf_packs, sf_token_stride, sf_hidden_stride, num_experts, num_topk,
        expert_alignment, nccl_context->dev_comm, nccl_context->window, buffer,
        workspace, mapped_host_workspace, nccl_context->scaleout_rank_idx,
        nccl_context->scaleup_rank_idx, nccl_context->num_scaleout_ranks,
        nccl_context->num_scaleup_ranks, nccl_context->is_scaleup_nvlink,
        num_sms, num_channels_per_sm, num_smem_bytes, num_qps,
        num_gpu_timeout_cycles, cached_mode, deterministic, do_cpu_sync,
        uccl_d2h_channel_addrs_device,
        static_cast<int>(uccl_d2h_channel_addrs_host.size()),
        uccl_signal_shadow_device, uccl_shared_per_rank_bytes(),
        uccl_intranode_local_world_size(), uccl_intranode_my_local_rank(),
        uccl_intranode_node_idx(), comm_stream);

    // Received token counters
    int num_recv_tokens = 0, num_expanded_tokens = 0;
    int counter_scaleup_rank_idx = 0, counter_local_expert_idx = 0;
    std::vector<int> num_recv_tokens_per_expert_list;

    // Assign these values according to modes
    if (cached_mode) {
      // Cached mode
      // TODO: support to expand for MoE training backward with cached handles
      // from non-expanding forward, which requires maintaining the same
      // expanding order between forward and backward
      EP_HOST_ASSERT(not do_expand and "Cannot do expand with cached mode");
      EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");
      num_recv_tokens = cached_num_recv_tokens.value();
      num_recv_tokens_per_expert_list =
          cached_num_recv_tokens_per_expert_list.value();
    } else if (do_cpu_sync) {
      // Non-cached mode with sync
      auto const start_cpu_time = std::chrono::high_resolution_clock::now();
      while (true) {
        bool ready = true;

        // Read number of received tokens from each scaleup rank
        while (counter_scaleup_rank_idx < nccl_context->num_scaleup_ranks and
               ready) {
          auto const count = math::encode_decode_positive(
              host_workspace_layout.get_scaleup_rank_count_ptr<
                  false>()[counter_scaleup_rank_idx]);
          if ((ready = math::is_decoded_positive_ready(count))) {
            num_recv_tokens += count;
            ++counter_scaleup_rank_idx;
          }
        }

        // Read expert counts
        while (counter_local_expert_idx < num_local_experts and ready) {
          auto const count = math::encode_decode_positive(
              host_workspace_layout.get_scaleup_expert_count_ptr<
                  false>()[counter_local_expert_idx]);
          if ((ready = math::is_decoded_positive_ready(count))) {
            num_recv_tokens_per_expert_list.push_back(count);
            num_expanded_tokens += count;
            ++counter_local_expert_idx;
          }
        }

        // Ready and do next steps
        auto const get_buffer_info = [&]() {
          std::stringstream ss;
          ss << "CPU side received count (scaleup: "
             << nccl_context->scaleup_rank_idx << "): ";
          for (int i = 0;
               i < nccl_context->num_scaleup_ranks + num_local_experts; ++i) {
            ss << host_workspace_layout
                      .get_scaleup_rank_expert_count_ptr<false>()[i];
            ss << (i == nccl_context->num_scaleup_ranks - 1 ? " # " : " ");
          }
          return ss.str();
        };
        if (ready) {
          if (get_env<int>("EP_BUFFER_DEBUG"))
            printf("%s\n", get_buffer_info().c_str());
          break;
        }

        // Timeout checks
        auto const now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now -
                                                             start_cpu_time)
                .count() > num_cpu_timeout_secs)
          throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());
      }
    } else {
      // Non-cached mode without CPU sync, allocate with the worst case
      num_recv_tokens = num_max_tokens_per_rank * nccl_context->num_ranks;
      num_expanded_tokens = nccl_context->num_ranks * num_max_tokens_per_rank *
                            std::min(num_topk, num_local_experts);
      num_expanded_tokens += (expert_alignment - 1) * num_local_experts;
      num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
    }

    // Allocate received tensors
    // `recv_src_metadata` includes source token indices and buffer slot indices
    auto const num_allocated_tokens =
        do_expand ? num_expanded_tokens : num_recv_tokens;
    auto recv_x = torch::empty({num_allocated_tokens, hidden}, x.options());
    auto recv_sf = std::optional<torch::Tensor>();
    auto recv_topk_idx = std::optional<torch::Tensor>();
    auto recv_topk_weights = std::optional<torch::Tensor>();
    auto recv_src_metadata =
        torch::empty({num_recv_tokens, num_topk + 2},
                     torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

    // Optional tensors
    void* recv_sf_ptr = nullptr;
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
    if (sf.has_value()) {
      if (not use_tma_aligned_col_major_sf) {
        recv_sf_token_stride = num_sf_packs, recv_sf_hidden_stride = 1;
      } else {
        // TMA-aligned layout for the next GEMM input
        recv_sf_token_stride = 1, recv_sf_hidden_stride = math::align(
                                      num_allocated_tokens, kNumAlignedSFPacks);
      }
      recv_sf = torch::empty_strided(
          {num_allocated_tokens, num_sf_packs},
          {recv_sf_token_stride, recv_sf_hidden_stride}, sf->options());
      recv_sf_ptr = recv_sf->data_ptr();
    }
    if (not do_expand) {
      recv_topk_idx =
          torch::empty({num_allocated_tokens, num_topk}, topk_idx.options());
      recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
    }
    if (topk_weights.has_value()) {
      recv_topk_weights =
          do_expand
              ? torch::empty({num_allocated_tokens}, topk_weights->options())
              : torch::empty({num_allocated_tokens, num_topk},
                             topk_weights->options());
      recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Process prefix sum, in expanding mode, it is also atomic counters
    if (do_expand) {
      // Slice and exclusive part and do atomic additions into inclusive
      EP_HOST_ASSERT(not cached_mode);
      psum_num_recv_tokens_per_expert =
          psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);
    } else if (not cached_mode) {
      // Slice the inclusive part (and will not be used in the epilogue)
      psum_num_recv_tokens_per_expert =
          psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);
    }
    EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.size(0) ==
                   num_local_experts);

    // Launch copy kernels with full SMs
    stream_control_before_epilogue(previous_event_before_epilogue);
    launch_dispatch_copy_epilogue(
        buffer, workspace,
        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
        psum_num_recv_tokens_per_expert.data_ptr<int>(), recv_x.data_ptr(),
        recv_sf_ptr, recv_topk_idx_ptr, recv_topk_weights_ptr,
        recv_src_metadata.data_ptr<int>(), channel_linked_list_ptr,
        num_recv_tokens, num_max_tokens_per_rank, num_hidden_bytes,
        num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride, num_experts,
        num_topk, nccl_context->scaleout_rank_idx,
        nccl_context->scaleup_rank_idx, nccl_context->num_scaleout_ranks,
        nccl_context->num_scaleup_ranks, jit::device_runtime->get_num_sms(),
        jit::device_runtime->get_num_smem_bytes(), num_channels, do_expand,
        cached_mode, comm_stream);

    // Stream control
    auto const event = stream_control_epilogue(
        {x, sf, topk_idx, topk_weights, recv_x, recv_sf, recv_topk_idx,
         recv_topk_weights, cumulative_local_expert_recv_stats, copied_topk_idx,
         psum_num_recv_tokens_per_scaleup_rank, psum_num_recv_tokens_per_expert,
         recv_src_metadata, deterministic_rank_count_buffer,
         dst_buffer_slot_idx, token_metadata_at_forward, channel_linked_list},
        compute_stream, allocate_on_comm_stream, async_with_compute_stream);

    return {recv_x,
            recv_sf,
            recv_topk_idx,
            recv_topk_weights,
            copied_topk_idx,
            num_recv_tokens_per_expert_list,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            recv_src_metadata,
            dst_buffer_slot_idx,
            token_metadata_at_forward,
            channel_linked_list,
            event};
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>,
             std::optional<EventHandle>>
  combine(torch::Tensor const& x,
          std::optional<torch::Tensor> const& topk_weights,
          std::optional<torch::Tensor> const& bias_0,
          std::optional<torch::Tensor> const& bias_1,
          torch::Tensor const& src_metadata,
          torch::Tensor const& combined_topk_idx,
          torch::Tensor const& psum_num_recv_tokens_per_scaleup_rank,
          std::optional<torch::Tensor> const& token_metadata_at_forward,
          std::optional<torch::Tensor> const& channel_linked_list,
          int const& num_experts, int const& num_max_tokens_per_rank,
          int const& num_sms, int const& num_qps,
          std::optional<EventHandle> const& previous_event,
          std::optional<EventHandle> const& previous_event_before_epilogue,
          bool const& async_with_compute_stream,
          bool const& allocate_on_comm_stream,
          bool const& use_expanded_layout) const {
    // Check SM count
    EP_HOST_ASSERT(num_sms > 0);

    // Check data
    auto const [num_tokens, hidden] = get_shape<2>(x);
    EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
    EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);

    // Check tensors at dispatch
    auto const [num_combined_tokens, num_topk] =
        get_shape<2>(combined_topk_idx);
    auto const [num_scaleup_ranks] =
        get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
    EP_HOST_ASSERT(combined_topk_idx.is_cuda() and
                   combined_topk_idx.is_contiguous());
    EP_HOST_ASSERT(combined_topk_idx.scalar_type() ==
                   c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);
    EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and
                   psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
    EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() ==
                   torch::kInt);
    EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);

    // Check metadata
    // For reduction mode, `num_tokens_` means the number of unexpanded tokens
    auto const [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);
    EP_HOST_ASSERT(num_reduced_tokens ==
                   (use_expanded_layout ? num_reduced_tokens : num_tokens));
    EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);
    EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());
    EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);

    // Check optional tensors
    if (use_expanded_layout) {
      // Reduction should be done with SwiGLU
      EP_HOST_ASSERT(not topk_weights.has_value());
    } else if (topk_weights.has_value()) {
      auto const [num_tokens__, num_topk__] =
          get_shape<2>(topk_weights.value());
      EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk__);
      EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
      EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
    }

    auto const bias_opts = std::vector({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
      if (bias_opts[i].has_value()) {
        auto bias = bias_opts[i].value();
        EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
        EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and
                       bias.size(1) == hidden);
        bias_ptrs[i] = bias.data_ptr();
      }
    }

    // Stream control
    // All new tensor allocations should happen after this
    auto const compute_stream = stream_control_prologue(
        previous_event, allocate_on_comm_stream, async_with_compute_stream);

    // Optional configs and metadata for hybrid combine
    int num_channels = 1;
    int* token_metadata_at_forward_ptr = nullptr;
    int* channel_linked_list_ptr = nullptr;
    if (nccl_context->num_scaleout_ranks > 1) {
      // The token metadata during forward
      auto const [num_channels_, d1, d2] =
          get_shape<3>(token_metadata_at_forward.value());
      auto const num_max_tokens_per_channel =
          math::ceil_div(num_max_tokens_per_rank, num_channels_);
      num_channels = num_channels_;
      token_metadata_at_forward_ptr =
          token_metadata_at_forward->data_ptr<int>();
      EP_HOST_ASSERT(d1 == nccl_context->num_scaleout_ranks *
                                   num_max_tokens_per_channel +
                               1);
      EP_HOST_ASSERT(d2 == 2 + num_topk * 2);
      EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and
                     token_metadata_at_forward->is_contiguous());
      EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);

      // Per-scaleup-peer-per-channel linked list
      auto const [num_channels__, d1_, d2_] =
          get_shape<3>(channel_linked_list.value());
      channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
      EP_HOST_ASSERT(num_channels == num_channels__);
      EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks *
                                    num_max_tokens_per_channel +
                                1);
      EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);
      EP_HOST_ASSERT(channel_linked_list->is_cuda() and
                     channel_linked_list->is_contiguous());
      EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
    }

    // Check buffer size
    int const combine_buffer_channels =
        nccl_context->num_scaleout_ranks > 1 ? num_channels : kNumMaxChannels;
    EP_HOST_ASSERT(
        get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk,
            nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
            nccl_context->is_scaleup_nvlink, allow_multiple_reduction,
            combine_buffer_channels) <= num_buffer_bytes);

    // Push data into remote buffers
    // NOTES: we don't use `num_hidden_bytes` due to enable later quantization
    // possibility
    auto const reduce_buffer = launch_combine(
        x.data_ptr(),
        topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
        src_metadata.data_ptr<int>(),
        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
        token_metadata_at_forward_ptr, channel_linked_list_ptr,
        nccl_context->dev_comm, nccl_context->window, buffer, workspace,
        num_reduced_tokens, num_max_tokens_per_rank, hidden, num_experts,
        num_topk, num_qps, num_gpu_timeout_cycles,
        nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
        nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
        nccl_context->is_scaleup_nvlink, num_sms,
        jit::device_runtime->get_num_smem_bytes(), num_channels,
        use_expanded_layout, allow_multiple_reduction,
        uccl_d2h_channel_addrs_device,
        static_cast<int>(uccl_d2h_channel_addrs_host.size()),
        uccl_signal_shadow_device, uccl_shared_per_rank_bytes(),
        uccl_intranode_local_world_size(), uccl_intranode_my_local_rank(),
        uccl_intranode_node_idx(), comm_stream);

    // Allocate output tensors
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
      combined_topk_weights = torch::empty({num_combined_tokens, num_topk},
                                           topk_weights->options());
      combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Combine pushed data
    stream_control_before_epilogue(previous_event_before_epilogue);
    launch_combine_reduce_epilogue(
        combined_x.data_ptr(), combined_topk_weights_ptr,
        combined_topk_idx.data_ptr<topk_idx_t>(), num_combined_tokens,
        num_max_tokens_per_rank, hidden, num_experts, num_topk, reduce_buffer,
        bias_ptrs[0], bias_ptrs[1], nccl_context->num_scaleout_ranks,
        nccl_context->num_scaleup_ranks, nccl_context->scaleout_rank_idx,
        nccl_context->scaleup_rank_idx, jit::device_runtime->get_num_sms(),
        jit::device_runtime->get_num_smem_bytes(), use_expanded_layout,
        allow_multiple_reduction, comm_stream);

    // Stream control
    auto const event = stream_control_epilogue(
        {x, topk_weights, bias_0, bias_1, src_metadata, combined_topk_idx,
         combined_x, combined_topk_weights,
         psum_num_recv_tokens_per_scaleup_rank, token_metadata_at_forward,
         channel_linked_list},
        compute_stream, allocate_on_comm_stream, async_with_compute_stream);
    return {combined_x, combined_topk_weights, event};
  }
};

static void register_apis(pybind11::module_& m) {
  pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")
      .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool,
                          int, int, int, int, bool, int, int, int,
                          std::string>())
      .def("destroy", &ElasticBuffer::destroy)
      .def("get_comm_stream", &ElasticBuffer::get_comm_stream)
      .def("is_uccl_proxy_active", &ElasticBuffer::is_uccl_proxy_active)
      .def("get_physical_domain_size", &ElasticBuffer::get_physical_domain_size)
      .def("get_logical_domain_size", &ElasticBuffer::get_logical_domain_size)
      .def("get_uccl_local_metadata", &ElasticBuffer::get_uccl_local_metadata)
      .def("sync_uccl_peers", &ElasticBuffer::sync_uccl_peers)
      .def("barrier", &ElasticBuffer::barrier)
      .def("engram_write", &ElasticBuffer::engram_write)
      .def("engram_fetch", &ElasticBuffer::engram_fetch)
      .def("pp_set_config", &ElasticBuffer::pp_set_config)
      .def("pp_send", &ElasticBuffer::pp_send)
      .def("pp_recv", &ElasticBuffer::pp_recv)
      .def("create_agrs_session", &ElasticBuffer::create_agrs_session)
      .def("destroy_agrs_session", &ElasticBuffer::destroy_agrs_session)
      .def("agrs_set_config", &ElasticBuffer::agrs_set_config)
      .def("agrs_get_inplace_tensor", &ElasticBuffer::agrs_get_inplace_tensor)
      .def("all_gather", &ElasticBuffer::all_gather)
      .def("dispatch", &ElasticBuffer::dispatch)
      .def("combine", &ElasticBuffer::combine);
  m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);

  // NCCL communicator handle
  m.def("get_local_nccl_unique_id", &nccl::get_local_unique_id);
  m.def("create_nccl_comm", &nccl::create_nccl_comm);
  m.def("destroy_nccl_comm", &nccl::destroy_nccl_comm);

  // Communication domain utilities
  m.def("get_physical_domain_size", &nccl::get_physical_domain_size);
  m.def("get_logical_domain_size", &nccl::get_logical_domain_size);
}

}  // namespace deep_ep::elastic

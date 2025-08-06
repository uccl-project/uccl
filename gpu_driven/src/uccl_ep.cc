#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

// TODO
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
namespace py = pybind11;

static std::mutex g_proxies_mu;
static std::unordered_map<int, std::vector<py::object>> g_proxies_by_dev;

struct EventOverlap {};
struct Ctx {
  long num_tokens{0};
  long hidden{0};
};
static std::atomic<long> g_next{1};
static std::mutex g_mu;
static std::unordered_map<long, Ctx> g_ctx;

struct EventHandle {
  std::shared_ptr<torch::Event> event;

  EventHandle() {
    event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(at::cuda::getCurrentCUDAStream());
  }

  explicit EventHandle(at::cuda::CUDAStream const& stream) {
    event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(stream);
  }

  EventHandle(EventHandle const&) = default;

  void current_stream_wait() const {
    at::cuda::getCurrentCUDAStream().unwrap().wait(*event);
  }
};

class Buffer {
 public:
  Buffer(int rank, int num_ranks, long num_nvl_bytes, long num_rdma_bytes,
         bool low_latency_mode, bool explicitly_destroy)
      : rank_(rank),
        num_ranks_(num_ranks),
        num_nvl_bytes_(num_nvl_bytes),
        num_rdma_bytes_(num_rdma_bytes),
        low_latency_mode_(low_latency_mode),
        explicitly_destroy_(explicitly_destroy) {
    // TODO(MaoZiming)
    std::lock_guard<std::mutex> lk(g_proxies_mu);
    cudaGetDevice(&device_index_);
    auto it = g_proxies_by_dev.find(device_index_);
    if (it == g_proxies_by_dev.end() || it->second.empty()) {
      throw std::runtime_error(
          "uccl_ep.Buffer: no UcclProxy registered for device " +
          std::to_string(device_index_) +
          ". Call uccl.uccl_ep.register_proxy(device_index_, proxies) first.");
    }
    proxies_ = it->second;
  }

  void destroy() {}

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor,
             torch::Tensor, torch::Tensor, std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch(
      torch::Tensor const& x, torch::Tensor const& topk_idx,
      std::optional<torch::Tensor> const& cumulative_local_expert_recv_stats,
      std::optional<torch::Tensor> const& dispatch_wait_recv_cost_stats,
      int num_max_dispatch_tokens_per_rank, int num_experts, bool use_fp8,
      bool round_scale, bool use_ue8m0, bool async, bool return_recv_hook) {
    // TODO(MaoZiming)
    TORCH_CHECK(x.dim() == 2, "x must be [num_tokens, hidden]");
    auto const num_tokens = x.size(0);
    auto const hidden = x.size(1);

    const int64_t world = std::max<int64_t>(1, num_ranks_);
    const int64_t local_E =
        std::max<int64_t>(1, static_cast<int64_t>(num_experts) / world);

    auto const recv_tokens =
        world * static_cast<int64_t>(num_max_dispatch_tokens_per_rank);
    auto recv_opts = x.options();
    torch::Tensor recv_x =
        torch::empty({local_E, recv_tokens, hidden}, recv_opts);

    std::optional<torch::Tensor> opt_scales;
    if (use_fp8) {
      auto scale_cols = std::max<int64_t>(1, hidden / 128);
      opt_scales.emplace(torch::empty({local_E, recv_tokens, scale_cols},
                                      x.options().dtype(torch::kFloat)));
    }

    torch::Tensor recv_count = torch::zeros(
        {local_E},
        torch::TensorOptions().device(x.device()).dtype(torch::kInt));
    {
      std::vector<int32_t> vals = {static_cast<int32_t>(rank_),
                                   static_cast<int32_t>(num_tokens)};
      torch::Tensor src_info = torch::tensor(
          vals, torch::TensorOptions().device(x.device()).dtype(torch::kInt));
      std::vector<int32_t> lrvals = {0, static_cast<int32_t>(num_tokens)};
      torch::Tensor layout_range = torch::tensor(
          lrvals, torch::TensorOptions().device(x.device()).dtype(torch::kInt));
      std::optional<EventHandle> evt;
      if (async) evt.emplace();
      std::optional<std::function<void()>> hook;
      if (return_recv_hook) hook.emplace([]() { no - op });

      return {recv_x,       opt_scales, recv_count, src_info,
              layout_range, evt,        hook};
    }
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
    // TODO(MaoZiming)
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D or 3D");
    auto const hidden = x.size(x.dim() - 1);
    auto const num_combined = topk_idx.size(0);

    torch::Tensor combined =
        out.has_value() ? *out
                        : torch::zeros({num_combined, hidden}, x.options());

    std::optional<EventHandle> evt;
    if (async) evt.emplace();

    std::optional<std::function<void()>> hook;
    if (return_recv_hook) hook.emplace([]() { no - op hook });

    return {combined, evt, hook};
  }

  int get_local_device_id() {
    // TODO(MaoZiming)
    return device_index_;
  }

  py::object get_local_ipc_handle() const {
    // TODO(MaoZiming)
    return py::none();
  }

  int get_num_rdma_ranks() const {
    // TODO(MaoZiming)
    return 1;
  }

  int get_rdma_rank() const {
    // TODO(MaoZiming)
    return 0;
  }

  py::object get_local_nvshmem_unique_id() const {
    // TODO(MaoZiming)
    return py::bytes("nvshmem-uid");
  }

  int get_root_rdma_rank(bool* with_low_latency) const {
    // TODO(MaoZiming)
    return 0;
  }

  void sync(py::object device_ids, py::object ipc_handles,
            py::object root_unique_id) {
    // TODO(MaoZiming)
    available_ = true;
  }

  bool is_available() const { return available_; }

 private:
  int rank_{0};
  int num_ranks_{1};
  long num_nvl_bytes_{0};
  long num_rdma_bytes_{0};
  bool low_latency_mode_{false};
  bool explicitly_destroy_{false};
  int device_index_{0};
  std::vector<py::object> proxies_;
  bool available_{false};
};

PYBIND11_MODULE(uccl_ep, m) {
  m.doc() = "Minimal DeepEP-compatible shim without libtorch linkage";
  m.def(
      "register_proxy",
      [](int device_index, py::object proxy) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        g_proxies_by_dev[device_index].push_back(std::move(proxy));
      },
      py::arg("device_index"), py::arg("proxy"));
  m.def(
      "register_proxies",
      [](int device_index, std::vector<py::object> proxies) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        auto& vec = g_proxies_by_dev[device_index];
        for (auto& proxy : proxies) {
          vec.push_back(std::move(proxy));
        }
      },
      py::arg("device_index"), py::arg("proxies"));
  m.def(
      "unregister_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        g_proxies_by_dev.erase(device_index);
      },
      py::arg("device_index"));
  m.def(
      "has_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxies_mu);
        auto it = g_proxies_by_dev.find(device_index);
        return it != g_proxies_by_dev.end() && !it->second.empty();
      },
      py::arg("device_index"));
  m.def("stop_all_registered_proxies", []() {
    std::lock_guard<std::mutex> lk(g_proxies_mu);
    for (auto& kv : g_proxies_by_dev) {
      for (auto& proxy : kv.second) {
        try {
          proxy.attr("stop")();
        } catch (...) {
        }
      }
    }
    g_proxies_by_dev.clear();
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
      .def("get_local_nvshmem_unique_id", &Buffer::get_local_nvshmem_unique_id)
      .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank,
           py::arg("with_low_latency"))
      .def("sync", &Buffer::sync, py::arg("device_ids"), py::arg("ipc_handles"),
           py::arg("root_unique_id"))
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
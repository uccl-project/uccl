#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <atomic>
#include <mutex>
#include <unordered_map>
namespace py = pybind11;

static std::mutex g_proxy_mu;
static std::unordered_map<int, py::object> g_proxy_by_dev;

struct EventOverlap {};
struct Ctx {
  long num_tokens{0};
  long hidden{0};
};
static std::atomic<long> g_next{1};
static std::mutex g_mu;
static std::unordered_map<long, Ctx> g_ctx;

class Buffer {
 public:
  Buffer(py::object group, long num_nvl_bytes, long num_rdma_bytes,
         bool low_latency_mode, int num_qps_per_rank,
         bool allow_nvlink_for_low_latency_mode, bool allow_mnnvl,
         bool explicitly_destroy)
      : group_(std::move(group)), device_index_(-1) {
    std::lock_guard<std::mutex> lk(g_proxy_mu);
    auto device_index = group_.attr("rank")().cast<int>();
    auto it = g_proxy_by_dev.find(device_index);
    if (it == g_proxy_by_dev.end() || it->second.is_none()) {
      throw std::runtime_error(
          "uccl_ep.Buffer: no UcclProxy registered for device " +
          std::to_string(device_index) +
          ". Call uccl.uccl_ep.register_proxy(device_index, proxy) first.");
    }
    proxy_ = it->second;
    device_index_ = device_index;
  }

  void destroy() {}

  py::tuple low_latency_dispatch(
      py::object x, py::object topk_idx, int num_max_dispatch_tokens_per_rank,
      int num_experts,
      py::object cumulative_local_expert_recv_stats = py::none(),
      py::object dispatch_wait_recv_cost_stats = py::none(),
      bool use_fp8 = true, bool round_scale = false, bool use_ue8m0 = false,
      bool async_finish = false, bool return_recv_hook = false) {
    py::object torch = py::module::import("torch");
    auto shape = x.attr("shape").cast<py::tuple>();
    long const num_tokens = shape[0].cast<long>();
    long const hidden = shape[1].cast<long>();
    py::object dev = x.attr("device");
    py::object dtype = x.attr("dtype");
    py::object recv_x =
        torch.attr("empty")(py::make_tuple(0, hidden), py::arg("device") = dev,
                            py::arg("dtype") = dtype);

    long world = group_.attr("size")().cast<long>();
    if (world <= 0) world = 1;
    long local_E = std::max<long>(1, num_experts / world);
    py::object recv_count =
        torch.attr("zeros")(py::make_tuple(local_E), py::arg("device") = dev,
                            py::arg("dtype") = torch.attr("int32"));
    int rank = group_.attr("rank")().cast<int>();
    py::object src_info = torch.attr("tensor")(
        py::make_tuple(rank, num_tokens), py::arg("device") = dev,
        py::arg("dtype") = torch.attr("int32"));

    long h = g_next.fetch_add(1);
    {
      std::lock_guard<std::mutex> lk(g_mu);
      g_ctx.emplace(h, Ctx{num_tokens, hidden});
    }
    py::tuple handle = py::make_tuple(py::int_(h));
    py::object event = py::cast(EventOverlap{});
    py::object hook;
    if (return_recv_hook) {
      hook = py::cpp_function([]() { py::print("Dispatch hook is invoked"); });
    } else {
      hook = py::none();
    }
    return py::make_tuple(py::make_tuple(recv_x, recv_count), src_info, handle,
                          event, hook);
  }

  py::tuple low_latency_combine(
      py::object x, py::object topk_idx, py::object topk_weights,
      py::object handle, bool use_logfmt = false, bool zero_copy = false,
      bool async_finish = false, bool return_recv_hook = false,
      py::object out = py::none(),
      py::object combine_wait_recv_cost_stats = py::none()) {
    py::object torch = py::module::import("torch");
    auto shape = x.attr("shape").cast<py::tuple>();
    py::object dev = x.attr("device");
    py::object dtype = x.attr("dtype");
    long h = handle.cast<py::tuple>()[0].cast<long>();
    Ctx c{};
    {
      std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_ctx.find(h);
      if (it == g_ctx.end()) throw std::runtime_error("invalid handle");
      c = it->second;
    }
    py::object combined;
    if (out.is_none()) {
      combined = torch.attr("zeros")(py::make_tuple(c.num_tokens, c.hidden),
                                     py::arg("device") = dev,
                                     py::arg("dtype") = dtype);
    } else {
      combined = out;
    }
    py::object event = py::cast(EventOverlap{});
    py::object hook;
    if (return_recv_hook) {
      hook = py::cpp_function([]() { py::print("Combine is invoked"); });
    } else {
      hook = py::none();
    }
    return py::make_tuple(combined, event, hook);
  }

 private:
  py::object group_;
  int device_index_{-1};
  py::object proxy_;
};

PYBIND11_MODULE(uccl_ep, m) {
  m.doc() = "Minimal DeepEP-compatible shim without libtorch linkage";
  m.def(
      "register_proxy",
      [](int device_index, py::object proxy) {
        std::lock_guard<std::mutex> lk(g_proxy_mu);
        g_proxy_by_dev[device_index] = std::move(proxy);
      },
      py::arg("device_index"), py::arg("proxy"));
  m.def(
      "unregister_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxy_mu);
        g_proxy_by_dev.erase(device_index);
      },
      py::arg("device_index"));
  m.def(
      "has_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxy_mu);
        auto it = g_proxy_by_dev.find(device_index);
        return it != g_proxy_by_dev.end() && !it->second.is_none();
      },
      py::arg("device_index"));
  m.def("stop_all_registered_proxies", []() {
    std::lock_guard<std::mutex> lk(g_proxy_mu);
    for (auto& kv : g_proxy_by_dev) {
      try {
        kv.second.attr("stop")();
      } catch (...) {
      }
    }
    g_proxy_by_dev.clear();
  });

  py::class_<EventOverlap>(m, "EventOverlap").def(py::init<>());
  py::class_<Buffer>(m, "Buffer")
      .def(py::init<py::object, long, long, bool, int, bool, bool, bool>(),
           py::arg("group"), py::arg("num_nvl_bytes") = 0,
           py::arg("num_rdma_bytes") = 0, py::arg("low_latency_mode") = false,
           py::arg("num_qps_per_rank") = 24,
           py::arg("allow_nvlink_for_low_latency_mode") = true,
           py::arg("allow_mnnvl") = false,
           py::arg("explicitly_destroy") = false)
      .def("destroy", &Buffer::destroy)
      .def("low_latency_dispatch", &Buffer::low_latency_dispatch, py::arg("x"),
           py::arg("topk_idx"), py::arg("num_max_dispatch_tokens_per_rank"),
           py::arg("num_experts"),
           py::arg("cumulative_local_expert_recv_stats") = py::none(),
           py::arg("dispatch_wait_recv_cost_stats") = py::none(),
           py::arg("use_fp8") = true, py::arg("round_scale") = false,
           py::arg("use_ue8m0") = false, py::arg("async_finish") = false,
           py::arg("return_recv_hook") = false)
      .def("low_latency_combine", &Buffer::low_latency_combine, py::arg("x"),
           py::arg("topk_idx"), py::arg("topk_weights"), py::arg("handle"),
           py::arg("use_logfmt") = false, py::arg("zero_copy") = false,
           py::arg("async_finish") = false, py::arg("return_recv_hook") = false,
           py::arg("out") = py::none(),
           py::arg("combine_wait_recv_cost_stats") = py::none());
}
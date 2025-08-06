#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace py = pybind11;

// ---- Global device->proxy registry (process-local) ----
static std::mutex g_proxy_mu;
static std::unordered_map<int, py::object> g_proxy_by_dev;

struct Ctx {
  long num_tokens{0};
  long hidden{0};
};

static std::atomic<long> g_next{1};
static std::mutex g_mu;
static std::unordered_map<long, Ctx> g_ctx;

class Buffer {
 public:
  Buffer(py::object group, int device_index, long bytes, bool llm, int qps,
         bool explicitly_destroy)
      : group_(std::move(group)), device_index_(device_index) {
    std::lock_guard<std::mutex> lk(g_proxy_mu);
    auto it = g_proxy_by_dev.find(device_index_);
    if (it == g_proxy_by_dev.end() || it->second.is_none()) {
      throw std::runtime_error(
          "uccl_ep.Buffer: no UcclProxy registered for device " +
          std::to_string(device_index_) +
          ". Call uccl.uccl_ep.register_proxy(device_index, proxy) first.");
    }
    proxy_ = it->second;
  }

  void destroy() {}

  // ... your low_latency_dispatch / low_latency_combine unchanged ...
  py::tuple low_latency_dispatch(py::object x, py::object topk_idx,
                                 long num_tokens, long num_experts,
                                 bool use_fp8 = false, bool round_scale = false,
                                 bool use_ue8m0 = false,
                                 py::object cum_stats = py::none(),
                                 bool async_finish = false,
                                 bool return_recv_hook = true) {
    auto hidden = x.attr("shape").cast<py::tuple>()[1].cast<long>();
    py::object torch = py::module::import("torch");
    py::object recv_x = torch.attr("empty")(
        py::make_tuple(0, hidden), py::arg("device") = x.attr("device"),
        py::arg("dtype") = x.attr("dtype"));
    long world = group_.attr("size")().cast<long>();
    long local_E = std::max<long>(1, num_experts / std::max<long>(1, world));
    py::object recv_count = torch.attr("zeros")(
        py::make_tuple(local_E), py::arg("device") = x.attr("device"),
        py::arg("dtype") = torch.attr("int32"));
    long h = g_next.fetch_add(1);
    {
      std::lock_guard<std::mutex> lk(g_mu);
      g_ctx.emplace(h, Ctx{num_tokens, hidden});
    }
    py::object none = py::none();
    py::object hook =
        py::cpp_function([]() { py::print("Dispatch hook is invoked"); });
    return py::make_tuple(recv_x, recv_count, py::int_(h), none,
                          return_recv_hook ? hook : py::none());
  }

  py::tuple low_latency_combine(py::object recv_x, py::object topk_idx,
                                py::object topk_weights, py::object handle,
                                bool use_logfmt = false, bool zero_copy = false,
                                bool async_finish = false,
                                bool return_recv_hook = true) {
    long h = handle.cast<long>();
    Ctx c{};
    {
      std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_ctx.find(h);
      if (it == g_ctx.end()) throw std::runtime_error("invalid handle");
      c = it->second;
    }
    py::object torch = py::module::import("torch");
    py::object combined =
        torch.attr("zeros")(py::make_tuple(c.num_tokens, c.hidden),
                            py::arg("device") = recv_x.attr("device"),
                            py::arg("dtype") = recv_x.attr("dtype"));
    py::object none = py::none();
    py::object hook =
        py::cpp_function([]() { py::print("Combine is invoked"); });
    return py::make_tuple(combined, none, return_recv_hook ? hook : py::none());
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
      py::arg("device_index"), py::arg("proxy"),
      "Register a running proxy object for a CUDA device.");

  m.def(
      "unregister_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxy_mu);
        g_proxy_by_dev.erase(device_index);
      },
      py::arg("device_index"),
      "Remove the proxy mapping for a CUDA device (does not stop it).");

  m.def(
      "has_proxy",
      [](int device_index) {
        std::lock_guard<std::mutex> lk(g_proxy_mu);
        auto it = g_proxy_by_dev.find(device_index);
        return it != g_proxy_by_dev.end() && !it->second.is_none();
      },
      py::arg("device_index"));

  m.def(
      "stop_all_registered_proxies",
      []() {
        std::lock_guard<std::mutex> lk(g_proxy_mu);
        for (auto& kv : g_proxy_by_dev) {
          auto& proxy = kv.second;
          try {
            proxy.attr("stop")();
          } catch (...) {
          }
        }
        g_proxy_by_dev.clear();
      },
      "Calls .stop() on all registered proxies and clears the map.");

  py::class_<Buffer>(m, "Buffer")
      .def(py::init<py::object, int, long, bool, int, bool>(), py::arg("group"),
           py::arg("device_index"), py::arg("bytes"),
           py::arg("low_latency_mode") = true, py::arg("num_qps_per_rank") = 1,
           py::arg("explicitly_destroy") = true)
      .def("destroy", &Buffer::destroy)
      .def("low_latency_dispatch", &Buffer::low_latency_dispatch, py::arg("x"),
           py::arg("topk_idx"), py::arg("num_tokens"), py::arg("num_experts"),
           py::arg("use_fp8") = false, py::arg("round_scale") = false,
           py::arg("use_ue8m0") = false,
           py::arg("cumulative_local_expert_recv_stats") = py::none(),
           py::arg("async_finish") = false, py::arg("return_recv_hook") = true)
      .def("low_latency_combine", &Buffer::low_latency_combine,
           py::arg("recv_x"), py::arg("topk_idx"), py::arg("topk_weights"),
           py::arg("handle"), py::arg("use_logfmt") = false,
           py::arg("zero_copy") = false, py::arg("async_finish") = false,
           py::arg("return_recv_hook") = true);
}
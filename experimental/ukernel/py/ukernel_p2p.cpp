#include "../include/config.h"
#include "../include/gpu_rt.h"
#include "../src/transport/communicator.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace UKernel {
namespace Transport {
namespace Python {

namespace {

torch::Tensor tensor_from_python(nb::handle obj, char const* arg_name) {
  PyObject* py_obj = obj.ptr();
  if (!THPVariable_Check(py_obj)) {
    throw std::invalid_argument(std::string(arg_name) +
                                " must be a torch.Tensor");
  }
  return THPVariable_Unpack(py_obj);
}

PreferredTransport parse_transport(std::string const& value) {
  if (value == "auto") return PreferredTransport::Auto;
  if (value == "ipc") return PreferredTransport::Ipc;
  if (value == "tcp") return PreferredTransport::Tcp;
  if (value == "rdma") return PreferredTransport::Rdma;
  throw std::invalid_argument("unsupported transport: " + value);
}

}  // namespace

class Communicator {
 public:
  Communicator(int gpu_id, int rank, int world_size, std::string exchanger_ip,
               int exchanger_port, std::string transport = "auto",
               int local_id = -1)
      : comm_(std::make_shared<UKernel::Transport::Communicator>(
            gpu_id, rank, world_size,
            std::make_shared<UKernel::Transport::CommunicatorConfig>(
                UKernel::Transport::CommunicatorConfig{
                    exchanger_ip,
                    exchanger_port,
                    local_id,
                    "default",
                    parse_transport(transport),
                }))) {
    GPU_RT_CHECK(gpuSetDevice(gpu_id));
  }

  ~Communicator() {
    std::vector<uint32_t> ids;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto const& it : pinned_tensors_) ids.push_back(it.first);
      pinned_tensors_.clear();
    }
    for (uint32_t id : ids) {
      comm_->dereg_mr(id);
      comm_->dereg_ipc(id);
    }
  }

  int rank() const { return comm_->rank(); }
  int world_size() const { return comm_->world_size(); }

  bool connect_peer(int peer_rank) { return comm_->connect(peer_rank); }
  bool accept_peer(int peer_rank) { return comm_->accept(peer_rank); }

  bool reg_rdma(uint32_t buffer_id, nb::handle tensor, bool publish = true) {
    if (buffer_id == 0)
      throw std::invalid_argument("buffer_id must be non-zero");
    torch::Tensor t = tensor_from_python(tensor, "tensor");
    if (!t.is_cuda())
      throw std::invalid_argument("reg_rdma requires CUDA tensor");
    if (!t.is_contiguous())
      throw std::invalid_argument("reg_rdma requires contiguous tensor");
    size_t total_bytes =
        static_cast<size_t>(t.numel()) * static_cast<size_t>(t.element_size());
    if (total_bytes == 0)
      throw std::invalid_argument("reg_rdma requires non-empty tensor");
    void* ptr = t.data_ptr();
    if (!comm_->reg_mr(buffer_id, ptr, total_bytes, publish)) return false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      pinned_tensors_[buffer_id] = std::move(t);
      buffer_sizes_[buffer_id] = total_bytes;
    }
    return true;
  }

  bool unreg_rdma(uint32_t buffer_id) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      pinned_tensors_.erase(buffer_id);
      buffer_sizes_.erase(buffer_id);
    }
    return comm_->dereg_mr(buffer_id);
  }

  bool reg_ipc(uint32_t buffer_id, nb::handle tensor, bool publish = true) {
    if (buffer_id == 0)
      throw std::invalid_argument("buffer_id must be non-zero");
    torch::Tensor t = tensor_from_python(tensor, "tensor");
    if (!t.is_cuda())
      throw std::invalid_argument("reg_ipc requires CUDA tensor");
    if (!t.is_contiguous())
      throw std::invalid_argument("reg_ipc requires contiguous tensor");
    size_t total_bytes =
        static_cast<size_t>(t.numel()) * static_cast<size_t>(t.element_size());
    if (total_bytes == 0)
      throw std::invalid_argument("reg_ipc requires non-empty tensor");
    void* ptr = t.data_ptr();
    if (!comm_->reg_ipc(buffer_id, ptr, total_bytes, publish)) return false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      pinned_tensors_[buffer_id] = std::move(t);
      buffer_sizes_[buffer_id] = total_bytes;
    }
    return true;
  }

  bool unreg_ipc(uint32_t buffer_id) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      pinned_tensors_.erase(buffer_id);
      buffer_sizes_.erase(buffer_id);
    }
    return comm_->dereg_ipc(buffer_id);
  }

  bool wait_ipc(int peer_rank, uint32_t buffer_id) {
    return comm_->wait_ipc(peer_rank, buffer_id);
  }

  bool wait_mr(int peer_rank, uint32_t buffer_id) {
    return comm_->wait_mr(peer_rank, buffer_id);
  }

  // ── New async API ──

  uint64_t send_put_async(int peer, uint32_t local_buf, size_t off = 0,
                          size_t len = 0, uint32_t remote_buf = 0,
                          size_t remote_off = 0) {
    if (len == 0) {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = buffer_sizes_.find(local_buf);
      if (it != buffer_sizes_.end()) len = it->second;
    }
    return comm_->send_put_async(peer, local_buf, off, remote_buf, remote_off,
                                 len);
  }

  uint64_t send_signal_async(int peer, uint64_t tag) {
    return comm_->send_signal_async(peer, tag);
  }

  uint64_t wait_signal_async(int peer, uint64_t tag) {
    return comm_->wait_signal_async(peer, tag);
  }

  std::vector<unsigned> poll_py(std::vector<unsigned> const& rids) {
    std::vector<unsigned> rids_copy = rids;
    size_t n = comm_->poll(rids_copy.data(), rids_copy.size());
    rids_copy.resize(n);
    return rids_copy;
  }

  // ── Blocking convenience wrappers ──

  void send(int peer, uint32_t src_buf, uint32_t dst_buf, size_t dst_off) {
    uint64_t rid = send_put_async(peer, src_buf, 0, buffer_size(src_buf),
                                  dst_buf, dst_off);
    if (rid == 0) throw std::runtime_error("send_put_async returned 0");
    wait_one(static_cast<unsigned>(rid));
  }

  void signal(int peer, uint64_t tag) {
    uint64_t rid = send_signal_async(peer, tag);
    if (rid == 0) throw std::runtime_error("send_signal_async returned 0");
    wait_one(static_cast<unsigned>(rid));
  }

  void wait_data(int peer, uint64_t tag, uint32_t recv_buf, size_t off = 0,
                 size_t len = 0) {
    if (len == 0) len = buffer_size(recv_buf);
    uint64_t rid = comm_->wait_signal_async(peer, tag, recv_buf, off, len);
    if (rid == 0)
      throw std::runtime_error("wait_signal_async(data) returned 0");
    wait_one(static_cast<unsigned>(rid));
  }

  // ── Inquiry ──

  std::string peer_transport(int peer_rank) const {
    switch (comm_->peer_transport_kind(peer_rank)) {
      case PeerTransportKind::Ipc:
        return "ipc";
      case PeerTransportKind::Tcp:
        return "tcp";
      case PeerTransportKind::Rdma:
        return "rdma";
      default:
        return "unknown";
    }
  }

  bool same_host(int peer_rank) const { return comm_->same_host(peer_rank); }

  bool barrier(std::string const& barrier_namespace = "default",
               int timeout_ms = -1) {
    return comm_->barrier(barrier_namespace, timeout_ms);
  }

 private:
  void wait_one(unsigned rid) {
    while (true) {
      CompletionResult r[1];
      size_t n = comm_->try_complete(r, 1);
      if (n == 1 && r[0].rid == rid) return;
      std::this_thread::yield();
    }
  }

  size_t buffer_size(uint32_t buffer_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = buffer_sizes_.find(buffer_id);
    if (it == buffer_sizes_.end())
      throw std::invalid_argument("buffer_id " + std::to_string(buffer_id) +
                                  " not registered");
    return it->second;
  }

  std::shared_ptr<UKernel::Transport::Communicator> comm_;
  std::unordered_map<uint32_t, torch::Tensor> pinned_tensors_;
  std::unordered_map<uint32_t, size_t> buffer_sizes_;
  mutable std::mutex mu_;
};

}  // namespace Python
}  // namespace Transport
}  // namespace UKernel

NB_MODULE(TORCH_EXTENSION_NAME, m) {
  using UKernel::Transport::Python::Communicator;

  nb::class_<Communicator>(m, "Communicator")
      .def(nb::init<int, int, int, std::string, int, std::string, int>(),
           nb::arg("gpu_id"), nb::arg("rank"), nb::arg("world_size"),
           nb::arg("exchanger_ip") = "127.0.0.1",
           nb::arg("exchanger_port") = 6979, nb::arg("transport") = "auto",
           nb::arg("local_id") = -1)
      .def_prop_ro("rank", &Communicator::rank)
      .def_prop_ro("world_size", &Communicator::world_size)
      .def("connect_peer", &Communicator::connect_peer, nb::arg("peer_rank"))
      .def("accept_peer", &Communicator::accept_peer, nb::arg("peer_rank"))
      .def("reg_rdma", &Communicator::reg_rdma, nb::arg("buffer_id"),
           nb::arg("tensor"), nb::arg("publish") = true)
      .def("unreg_rdma", &Communicator::unreg_rdma, nb::arg("buffer_id"))
      .def("reg_ipc", &Communicator::reg_ipc, nb::arg("buffer_id"),
           nb::arg("tensor"), nb::arg("publish") = true)
      .def("unreg_ipc", &Communicator::unreg_ipc, nb::arg("buffer_id"))
      .def("wait_mr", &Communicator::wait_mr, nb::arg("peer_rank"),
           nb::arg("buffer_id"))
      .def("wait_ipc", &Communicator::wait_ipc, nb::arg("peer_rank"),
           nb::arg("buffer_id"))
      .def("send_put_async", &Communicator::send_put_async, nb::arg("peer"),
           nb::arg("local_buf"), nb::arg("off") = 0, nb::arg("len") = 0,
           nb::arg("remote_buf") = 0, nb::arg("remote_off") = 0)
      .def("send_signal_async", &Communicator::send_signal_async,
           nb::arg("peer"), nb::arg("tag"))
      .def("wait_signal_async", &Communicator::wait_signal_async,
           nb::arg("peer"), nb::arg("tag"))
      .def("poll", &Communicator::poll_py, nb::arg("rids"))
      .def("signal", &Communicator::signal, nb::arg("peer"), nb::arg("tag"))
      .def("wait_data", &Communicator::wait_data, nb::arg("peer"),
           nb::arg("tag"), nb::arg("recv_buf"), nb::arg("off") = 0,
           nb::arg("len") = 0)
      .def("peer_transport", &Communicator::peer_transport,
           nb::arg("peer_rank"))
      .def("same_host", &Communicator::same_host, nb::arg("peer_rank"))
      .def("barrier", &Communicator::barrier,
           nb::arg("barrier_namespace") = "default", nb::arg("timeout_ms") = -1)
      .def("send", &Communicator::send, nb::arg("peer_rank"),
           nb::arg("local_buffer_id"), nb::arg("remote_buffer_id") = 0,
           nb::arg("remote_offset") = 0);
}

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
  if (value == "uccl") return PreferredTransport::Uccl;
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

  uint64_t isend(int peer_rank, uint32_t local_buffer_id,
                 size_t offset = 0, size_t len = 0,
                 uint32_t remote_buffer_id = 0, size_t remote_offset = 0) {
    if (len == 0) {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = buffer_sizes_.find(local_buffer_id);
      if (it != buffer_sizes_.end()) len = it->second;
    }
    return comm_->isend(peer_rank, local_buffer_id, offset, len,
                        remote_buffer_id, remote_offset);
  }

  uint64_t irecv(int peer_rank, uint32_t local_buffer_id,
                 size_t offset = 0, size_t len = 0) {
    if (len == 0) {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = buffer_sizes_.find(local_buffer_id);
      if (it != buffer_sizes_.end()) len = it->second;
    }
    return comm_->irecv(peer_rank, local_buffer_id, offset, len);
  }

  bool poll(uint64_t req) { return comm_->poll(static_cast<unsigned>(req)); }
  void release(uint64_t req) { comm_->release(static_cast<unsigned>(req)); }

  bool wait_finish(uint64_t req) {
    return comm_->wait_finish(static_cast<unsigned>(req));
  }

  bool wait_finish_multi(std::vector<uint64_t> reqs) {
    std::vector<unsigned> unsigned_reqs(reqs.size());
    for (size_t i = 0; i < reqs.size(); ++i) {
      unsigned_reqs[i] = static_cast<unsigned>(reqs[i]);
    }
    return comm_->wait_finish(unsigned_reqs);
  }

  std::string peer_transport(int peer_rank) const {
    switch (comm_->peer_transport_kind(peer_rank)) {
      case PeerTransportKind::Ipc:  return "ipc";
      case PeerTransportKind::Uccl: return "uccl";
      case PeerTransportKind::Tcp:  return "tcp";
      case PeerTransportKind::Rdma: return "rdma";
      default:                      return "unknown";
    }
  }

  bool same_host(int peer_rank) const { return comm_->same_host(peer_rank); }

  bool barrier(std::string const& barrier_namespace = "default",
               int timeout_ms = -1) {
    return comm_->barrier(barrier_namespace, timeout_ms);
  }

  void send(int peer_rank, uint32_t local_buffer_id,
            uint32_t remote_buffer_id = 0, size_t remote_offset = 0) {
    uint64_t req = isend(peer_rank, local_buffer_id, 0, 0,
                         remote_buffer_id, remote_offset);
    if (req == 0) throw std::runtime_error("isend returned 0");
    wait_finish(req);
  }

  void recv(int peer_rank, uint32_t local_buffer_id) {
    uint64_t req = irecv(peer_rank, local_buffer_id, 0, 0);
    if (req == 0) throw std::runtime_error("irecv returned 0");
    wait_finish(req);
  }

 private:
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
      .def("isend", &Communicator::isend, nb::arg("peer_rank"),
           nb::arg("local_buffer_id"), nb::arg("offset") = 0,
           nb::arg("len") = 0, nb::arg("remote_buffer_id") = 0,
           nb::arg("remote_offset") = 0)
      .def("irecv", &Communicator::irecv, nb::arg("peer_rank"),
           nb::arg("local_buffer_id"), nb::arg("offset") = 0,
           nb::arg("len") = 0)
      .def("poll", &Communicator::poll, nb::arg("req"))
      .def("release", &Communicator::release, nb::arg("req"))
      .def("wait_finish", &Communicator::wait_finish, nb::arg("req"))
      .def("wait_finish_multi", &Communicator::wait_finish_multi,
           nb::arg("reqs"))
      .def("peer_transport", &Communicator::peer_transport,
           nb::arg("peer_rank"))
      .def("same_host", &Communicator::same_host, nb::arg("peer_rank"))
      .def("barrier", &Communicator::barrier,
           nb::arg("barrier_namespace") = "default", nb::arg("timeout_ms") = -1)
      .def("send", &Communicator::send, nb::arg("peer_rank"),
           nb::arg("local_buffer_id"), nb::arg("remote_buffer_id") = 0,
           nb::arg("remote_offset") = 0)
      .def("recv", &Communicator::recv, nb::arg("peer_rank"),
           nb::arg("local_buffer_id"));
}

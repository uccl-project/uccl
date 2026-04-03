#include "../include/gpu_rt.h"
#include "../include/config.h"
#include "../src/transport/communicator.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace UKernel {
namespace Transport {
namespace Python {

namespace {

torch::Tensor tensor_from_python(nb::handle obj, char const* arg_name) {
  PyObject* py_obj = obj.ptr();
  if (!THPVariable_Check(py_obj)) {
    throw std::invalid_argument(std::string(arg_name) + " must be a torch.Tensor");
  }
  return THPVariable_Unpack(py_obj);
}

PreferredTransport parse_transport(std::string const& value) {
  if (value == "auto") return PreferredTransport::Auto;
  if (value == "ipc") return PreferredTransport::Ipc;
  if (value == "uccl") return PreferredTransport::Uccl;
  if (value == "tcp") return PreferredTransport::Tcp;
  throw std::invalid_argument("unsupported transport: " + value);
}

}  // namespace

class Communicator {
 public:
  Communicator(int gpu_id, int rank, int world_size,
               std::string exchanger_ip, int exchanger_port,
               std::string transport = "auto", int local_id = -1)
      : comm_(std::make_shared<UKernel::Transport::Communicator>(
            gpu_id, rank, world_size,
            std::make_shared<UKernel::Transport::CommunicatorConfig>(
                UKernel::Transport::CommunicatorConfig{
                    exchanger_ip, exchanger_port, local_id,
                    parse_transport(transport),
                }))) {
    GPU_RT_CHECK(gpuSetDevice(gpu_id));
  }

  int rank() const { return comm_->rank(); }
  int world_size() const { return comm_->world_size(); }

  bool connect_to(int peer_rank) { return comm_->connect_to(peer_rank); }
  bool accept_from(int peer_rank) { return comm_->accept_from(peer_rank); }
  bool connect_bidir(int peer_rank) { return comm_->connect_bidir(peer_rank); }

  uint64_t isend(int peer_rank, nb::handle tensor,
                 size_t offset = 0, size_t len = 0) {
    torch::Tensor t = tensor_from_python(tensor, "tensor");
    if (!t.is_cuda()) {
      throw std::invalid_argument("isend requires a CUDA tensor");
    }
    if (!t.is_contiguous()) {
      throw std::invalid_argument("isend requires a contiguous tensor");
    }
    size_t elem_bytes = static_cast<size_t>(t.element_size());
    size_t total_bytes = static_cast<size_t>(t.numel()) * elem_bytes;
    if (len == 0) len = total_bytes;
    if (offset + len > total_bytes) {
      throw std::invalid_argument("isend offset+len exceeds tensor size");
    }
    return comm_->isend(peer_rank, t.data_ptr(), offset, len,
                        0, 0, true);
  }

  uint64_t irecv(int peer_rank, nb::handle tensor,
                 size_t offset = 0, size_t len = 0) {
    torch::Tensor t = tensor_from_python(tensor, "tensor");
    if (!t.is_cuda()) {
      throw std::invalid_argument("irecv requires a CUDA tensor");
    }
    if (!t.is_contiguous()) {
      throw std::invalid_argument("irecv requires a contiguous tensor");
    }
    size_t elem_bytes = static_cast<size_t>(t.element_size());
    size_t total_bytes = static_cast<size_t>(t.numel()) * elem_bytes;
    if (len == 0) len = total_bytes;
    if (offset + len > total_bytes) {
      throw std::invalid_argument("irecv offset+len exceeds tensor size");
    }
    return comm_->irecv(peer_rank, t.data_ptr(), offset, len, true);
  }

  bool poll(uint64_t req) { return comm_->poll(static_cast<unsigned>(req)); }
  void release(uint64_t req) { comm_->release(static_cast<unsigned>(req)); }
  bool wait_finish(uint64_t req) { return comm_->wait_finish(static_cast<unsigned>(req)); }

  bool wait_finish_multi(std::vector<uint64_t> reqs) {
    std::vector<unsigned> unsigned_reqs(reqs.size());
    for (size_t i = 0; i < reqs.size(); ++i) {
      unsigned_reqs[i] = static_cast<unsigned>(reqs[i]);
    }
    return comm_->wait_finish(unsigned_reqs);
  }

  std::string peer_transport(int peer_rank) const {
    switch (comm_->peer_transport_kind(peer_rank)) {
      case PeerTransportKind::Ipc: return "ipc";
      case PeerTransportKind::Uccl: return "uccl";
      case PeerTransportKind::Tcp: return "tcp";
      default: return "unknown";
    }
  }

  bool same_host(int peer_rank) const { return comm_->same_host(peer_rank); }

  void send(int peer_rank, nb::handle tensor) {
    uint64_t req = isend(peer_rank, tensor);
    wait_finish(req);
    release(req);
  }

  void recv(int peer_rank, nb::handle tensor) {
    uint64_t req = irecv(peer_rank, tensor);
    wait_finish(req);
    release(req);
  }

 private:
  std::shared_ptr<UKernel::Transport::Communicator> comm_;
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
           nb::arg("exchanger_port") = 6979,
           nb::arg("transport") = "auto",
           nb::arg("local_id") = -1)
      .def_prop_ro("rank", &Communicator::rank)
      .def_prop_ro("world_size", &Communicator::world_size)
      .def("connect_to", &Communicator::connect_to, nb::arg("peer_rank"))
      .def("accept_from", &Communicator::accept_from, nb::arg("peer_rank"))
      .def("connect_bidir", &Communicator::connect_bidir, nb::arg("peer_rank"))
      .def("isend", &Communicator::isend,
           nb::arg("peer_rank"), nb::arg("tensor"),
           nb::arg("offset") = 0, nb::arg("len") = 0)
      .def("irecv", &Communicator::irecv,
           nb::arg("peer_rank"), nb::arg("tensor"),
           nb::arg("offset") = 0, nb::arg("len") = 0)
      .def("poll", &Communicator::poll, nb::arg("req"))
      .def("release", &Communicator::release, nb::arg("req"))
      .def("wait_finish", &Communicator::wait_finish, nb::arg("req"))
      .def("wait_finish_multi", &Communicator::wait_finish_multi, nb::arg("reqs"))
      .def("peer_transport", &Communicator::peer_transport, nb::arg("peer_rank"))
      .def("same_host", &Communicator::same_host, nb::arg("peer_rank"))
      .def("send", &Communicator::send,
           nb::arg("peer_rank"), nb::arg("tensor"))
      .def("recv", &Communicator::recv,
           nb::arg("peer_rank"), nb::arg("tensor"));
}

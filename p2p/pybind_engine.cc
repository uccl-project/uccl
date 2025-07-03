#include "engine.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(p2p, m) {
  m.doc() = "P2P Engine - High-performance RDMA-based peer-to-peer transport";

  // Endpoint class binding
  py::class_<Endpoint>(m, "Endpoint")
      .def(py::init<uint32_t, uint32_t>(), "Create a new Engine instance",
           py::arg("local_gpu_idx"), py::arg("num_cpus"))
      .def(
          "connect",
          [](Endpoint& self, std::string const& remote_ip_addr,
             int remote_gpu_idx) {
            uint64_t conn_id;
            bool success =
                self.connect(remote_ip_addr, remote_gpu_idx, conn_id);
            return py::make_tuple(success, conn_id);
          },
          "Connect to a remote server", py::arg("remote_ip_addr"),
          py::arg("remote_gpu_idx"))
      .def(
          "accept",
          [](Endpoint& self) {
            std::string remote_ip_addr;
            int remote_gpu_idx;
            uint64_t conn_id;
            bool success = self.accept(remote_ip_addr, remote_gpu_idx, conn_id);
            return py::make_tuple(success, remote_ip_addr, remote_gpu_idx,
                                  conn_id);
          },
          "Accept an incoming connection")
      .def(
          "reg",
          [](Endpoint& self, uint64_t ptr, size_t size) {
            uint64_t mr_id;
            bool success =
                self.reg(reinterpret_cast<void const*>(ptr), size, mr_id);
            return py::make_tuple(success, mr_id);
          },
          "Register a data buffer", py::arg("ptr"), py::arg("size"))
      .def(
          "send",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            return self.send(conn_id, mr_id, reinterpret_cast<void const*>(ptr),
                             size);
          },
          "Send a data buffer", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("size"))
      .def(
          "recv",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t max_size) {
            size_t recv_size;
            bool success =
                self.recv(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                          max_size, recv_size);
            return py::make_tuple(success, recv_size);
          },
          "Receive a key-value buffer", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("max_size"))
      .def("join_group", &Endpoint::join_group,
           "Join a rendezvous group: publish discovery info, wait for peers, "
           "and fully-connect",
           py::arg("discovery_uri"), py::arg("group_name"),
           py::arg("world_size"), py::arg("my_rank"), py::arg("remote_gpu_idx"))
      .def(
          "conn_id_of_rank", &Endpoint::conn_id_of_rank,
          "Get the connection ID for a given peer rank (or UINT64_MAX if none)",
          py::arg("rank"))
      .def_static("CreateAndJoin", &Endpoint::CreateAndJoin,
                  "Create an Endpoint and immediately join a rendezvous group",
                  py::arg("discovery_uri"), py::arg("group_name"),
                  py::arg("world_size"), py::arg("my_rank"),
                  py::arg("local_gpu_idx"), py::arg("num_cpus"),
                  py::arg("remote_gpu_idx"))
      .def("__repr__", [](Endpoint const& e) { return "<UCCL P2P Endpoint>"; });
}
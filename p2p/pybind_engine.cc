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
             int remote_gpu_idx, int remote_port) {
            uint64_t conn_id;
            bool success = self.connect(remote_ip_addr, remote_gpu_idx, conn_id,
                                        remote_port);
            return py::make_tuple(success, conn_id);
          },
          "Connect to a remote server", py::arg("remote_ip_addr"),
          py::arg("remote_gpu_idx"), py::arg("remote_port") = -1)
      .def(
          "connect",
          [](Endpoint& self, py::bytes metadata) {
            uint64_t conn_id;
            bool success = self.connect(metadata, conn_id);
            return py::make_tuple(success, conn_id);
          },
          "Connect to a remote server with endpoint-metadata blob",
          py::arg("metadata"))
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
          "read",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            return self.read(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                             size);
          },
          "Read data into a local buffer", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("size"))
      .def(
          "recv",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t max_size) {
            size_t recv_size;
            bool success =
                self.recv(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                          max_size, &recv_size);
            return py::make_tuple(success, recv_size);
          },
          "Receive a key-value buffer", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("max_size"))
      .def(
          "sendv",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> data_ptr_v, std::vector<size_t> size_v,
             size_t num_iovs) {
            std::vector<void const*> data_v;
            data_v.reserve(data_ptr_v.size());
            for (uint64_t ptr : data_ptr_v) {
              data_v.push_back(reinterpret_cast<void const*>(ptr));
            }
            return self.sendv(conn_id, mr_id_v, data_v, size_v, num_iovs);
          },
          "Send multiple data buffers", py::arg("conn_id"), py::arg("mr_id_v"),
          py::arg("data_ptr_v"), py::arg("size_v"), py::arg("num_iovs"))
      .def(
          "recvv",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> data_ptr_v, std::vector<size_t> max_size_v,
             size_t num_iovs) {
            std::vector<void*> data_v;
            data_v.reserve(data_ptr_v.size());
            for (uint64_t ptr : data_ptr_v) {
              data_v.push_back(reinterpret_cast<void*>(ptr));
            }
            std::vector<size_t> recv_size_v(num_iovs);
            bool success = self.recvv(conn_id, mr_id_v, data_v, max_size_v,
                                      recv_size_v, num_iovs);
            return py::make_tuple(success, recv_size_v);
          },
          "Receive multiple data buffers", py::arg("conn_id"),
          py::arg("mr_id_v"), py::arg("data_ptr_v"), py::arg("max_size_v"),
          py::arg("num_iovs"))
      .def(
          "send_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            uint64_t transfer_id;
            bool success = self.send_async(conn_id, mr_id,
                                           reinterpret_cast<void const*>(ptr),
                                           size, &transfer_id);
            return py::make_tuple(success, transfer_id);
          },
          "Send data asynchronously", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("size"))
      .def(
          "recv_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            uint64_t transfer_id;
            bool success =
                self.recv_async(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                size, &transfer_id);
            return py::make_tuple(success, transfer_id);
          },
          "Receive data asynchronously", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("size"))
      .def(
          "poll_async",
          [](Endpoint& self, uint64_t transfer_id) {
            bool is_done;
            bool success = self.poll_async(transfer_id, &is_done);
            return py::make_tuple(success, is_done);
          },
          "Poll the status of an asynchronous transfer", py::arg("transfer_id"))
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
      .def(
          "get_endpoint_metadata",
          [](Endpoint& self) {
            std::vector<uint8_t> metadata = self.get_endpoint_metadata();
            return py::bytes(reinterpret_cast<const char*>(metadata.data()),
                             metadata.size());
          },
          "Return endpoint metadata as a list of bytes")
      .def("__repr__", [](Endpoint const& e) { return "<UCCL P2P Endpoint>"; });
}
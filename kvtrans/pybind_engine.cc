#include "engine.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(kvtrans_engine, m) {
  m.doc() = "KVTrans Engine - High-performance RDMA-based key-value transport";

  // Endpoint class binding
  py::class_<Endpoint>(m, "Endpoint")
      .def(py::init<uint32_t, uint32_t>(), "Create a new Engine instance",
           py::arg("gpu_idx"), py::arg("ncpus"))
      .def(
          "connect",
          [](Endpoint& self, std::string const& ip_addr, int remote_gpu_idx) {
            uint64_t conn_id;
            bool success = self.connect(ip_addr, remote_gpu_idx, conn_id);
            return py::make_tuple(success, conn_id);
          },
          "Connect to a remote server", py::arg("ip_addr"),
          py::arg("remote_gpu_idx"))
      .def(
          "accept",
          [](Endpoint& self) {
            std::string ip_addr;
            int remote_gpu_idx;
            uint64_t conn_id;
            bool success = self.accept(ip_addr, remote_gpu_idx, conn_id);
            return py::make_tuple(success, ip_addr, remote_gpu_idx, conn_id);
          },
          "Accept an incoming connection")
      .def(
          "reg_kv",
          [](Endpoint& self, py::buffer buffer) {
            py::buffer_info info = buffer.request();
            uint64_t mr_id;
            bool success =
                self.reg_kv(info.ptr, info.size * info.itemsize, mr_id);
            return py::make_tuple(success, mr_id);
          },
          "Register a key-value buffer", py::arg("buffer"))
      .def(
          "send_kv",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id,
             py::buffer buffer) {
            py::buffer_info info = buffer.request();
            return self.send_kv(conn_id, mr_id, info.ptr,
                                info.size * info.itemsize);
          },
          "Send a key-value buffer", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("buffer"))
      .def(
          "recv_kv",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id,
             size_t max_size) {
            std::vector<uint8_t> data(max_size);
            size_t actual_size = max_size;
            bool success =
                self.recv_kv(conn_id, mr_id, data.data(), actual_size);
            if (success) {
              data.resize(actual_size);
              return py::make_tuple(
                  success,
                  py::bytes(reinterpret_cast<char*>(data.data()), actual_size));
            } else {
              return py::make_tuple(false, py::bytes());
            }
          },
          "Receive a key-value buffer", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("max_size"))
      .def("__repr__", [](Endpoint const& e) { return "<KVTrans Endpoint>"; });

  // Utility functions
  m.def(
      "create_buffer",
      [](size_t size, uint8_t fill_value = 0) {
        std::string buffer(size, static_cast<char>(fill_value));
        return py::bytearray(buffer);
      },
      "Create a buffer for testing", py::arg("size"),
      py::arg("fill_value") = 0);

  m.def(
      "buffer_to_string",
      [](py::buffer buffer) {
        py::buffer_info info = buffer.request();
        return std::string(static_cast<char*>(info.ptr),
                           info.size * info.itemsize);
      },
      "Convert buffer to string", py::arg("buffer"));
}
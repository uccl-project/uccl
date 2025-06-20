#include "engine.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(kvtrans_engine, m) {
  m.doc() = "KVTrans Engine - High-performance RDMA-based key-value transport";

  // Engine class binding
  py::class_<Engine>(m, "Engine")
      .def(py::init<std::string const&, uint32_t, uint32_t, uint16_t>(),
           "Create a new Engine instance", py::arg("if_name"), py::arg("ncpus"),
           py::arg("nconn_per_cpu"), py::arg("listen_port"))
      .def(
          "connect",
          [](Engine& self, std::string const& ip_addr, uint16_t port) {
            int conn_id;
            bool success = self.connect(ip_addr, port, conn_id);
            return py::make_tuple(success, conn_id);
          },
          "Connect to a remote server", py::arg("ip_addr"), py::arg("port"))
      .def(
          "accept",
          [](Engine& self) {
            std::string ip_addr;
            uint16_t port;
            int conn_id;
            bool success = self.accept(ip_addr, port, conn_id);
            return py::make_tuple(success, ip_addr, port, conn_id);
          },
          "Accept an incoming connection")
      .def(
          "reg_kv",
          [](Engine& self, int conn_id, py::buffer buffer) {
            py::buffer_info info = buffer.request();
            uint64_t kv_id;
            bool success = self.reg_kv(conn_id, info.ptr,
                                       info.size * info.itemsize, kv_id);
            return py::make_tuple(success, kv_id);
          },
          "Register a key-value buffer", py::arg("conn_id"), py::arg("buffer"))
      .def(
          "send_kv",
          [](Engine& self, uint64_t kv_id, py::buffer buffer) {
            py::buffer_info info = buffer.request();
            return self.send_kv(kv_id, info.ptr, info.size * info.itemsize);
          },
          "Send a key-value buffer", py::arg("kv_id"), py::arg("buffer"))
      .def(
          "recv_kv",
          [](Engine& self, uint64_t kv_id, size_t max_size) {
            std::vector<uint8_t> data(max_size);
            size_t actual_size = max_size;
            bool success = self.recv_kv(kv_id, data.data(), actual_size);
            if (success) {
              data.resize(actual_size);
              return py::make_tuple(
                  success,
                  py::bytes(reinterpret_cast<char*>(data.data()), actual_size));
            } else {
              return py::make_tuple(false, py::bytes());
            }
          },
          "Receive a key-value buffer", py::arg("kv_id"), py::arg("max_size"))
      .def("__repr__", [](Engine const& e) { return "<KVTrans Engine>"; });

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
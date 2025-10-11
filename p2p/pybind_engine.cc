#include "engine.h"
#include "tensor.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct InsidePythonGuard {
  InsidePythonGuard() { inside_python = true; }
  ~InsidePythonGuard() { inside_python = false; }
};

PYBIND11_MODULE(p2p, m) {
  m.doc() = "P2P Engine - High-performance RDMA-based peer-to-peer transport";

  m.def("get_oob_ip", &get_oob_ip, "Get the OOB IP address");

  m.def(
      "reg_mem",
      [](uint64_t addr, size_t size, bool is_device, int gpu_id) {
        uint64_t mem_id;
        reg_mem(reinterpret_cast<void*>(addr), size, mem_id, is_device, gpu_id);
        return mem_id;
      },
      "Reg the memory with RDMA capabilities", py::arg("addr"), py::arg("size"),
      py::arg("is_device"), py::arg("gpu_id"));

  m.def(
      "dereg_mem", [](uint64_t mem_id) { dereg_mem(mem_id); },
      "Dereg the memory associated RDMA resources", py::arg("mem_id"));

  // Endpoint class binding
  py::class_<Endpoint>(m, "Endpoint")
      .def(py::init<uint32_t, uint32_t>(), "Create a new Engine instance",
           py::arg("local_gpu_idx"), py::arg("num_cpus"))
      .def(
          "connect",
          [](Endpoint& self, std::string const& remote_ip_addr,
             int remote_gpu_idx, int remote_port) {
            uint64_t conn_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.connect(remote_ip_addr, remote_gpu_idx,
                                     remote_port, conn_id);
            }
            return py::make_tuple(success, conn_id);
          },
          "Connect to a remote server", py::arg("remote_ip_addr"),
          py::arg("remote_gpu_idx"), py::arg("remote_port") = -1)
      .def(
          "get_metadata",
          [](Endpoint& self) {
            std::vector<uint8_t> metadata = self.get_metadata();
            return py::bytes(reinterpret_cast<char const*>(metadata.data()),
                             metadata.size());
          },
          "Return endpoint metadata as a list of bytes")
      .def_static(
          "parse_metadata",
          [](py::bytes metadata_bytes) {
            std::string buf = metadata_bytes;
            std::vector<uint8_t> metadata(buf.begin(), buf.end());
            auto [ip, port, gpu_idx] = Endpoint::parse_metadata(metadata);
            return py::make_tuple(ip, port, gpu_idx);
          },
          "Parse endpoint metadata to extract IP address, port, and GPU index",
          py::arg("metadata"))
      .def(
          "accept",
          [](Endpoint& self) {
            bool success;
            std::string remote_ip_addr;
            int remote_gpu_idx;
            uint64_t conn_id;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.accept(remote_ip_addr, remote_gpu_idx, conn_id);
            }
            return py::make_tuple(success, remote_ip_addr, remote_gpu_idx,
                                  conn_id);
          },
          "Accept an incoming connection")
      .def(
          "send",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.send(conn_id, mr_id,
                                  reinterpret_cast<void const*>(ptr), size);
            }
            return success;
          },
          "Send a data buffer, optionally using metadata (serialized FifoItem)",
          py::arg("conn_id"), py::arg("mr_id"), py::arg("ptr"), py::arg("size"))
      .def(
          "recv",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.recv(conn_id, mr_id, reinterpret_cast<void*>(ptr), size);
            }
            return success;
          },
          "Receive a key-value buffer", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("size"))
      .def(
          "send_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            uint64_t transfer_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.send_async(conn_id, mr_id,
                                        reinterpret_cast<void const*>(ptr),
                                        size, &transfer_id);
            }
            return py::make_tuple(success, transfer_id);
          },
          "Send data asynchronously", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("size"))
      .def(
          "recv_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            uint64_t transfer_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.recv_async(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, &transfer_id);
            }
            return py::make_tuple(success, transfer_id);
          },
          "Receive data asynchronously", py::arg("conn_id"), py::arg("mr_id"),
          py::arg("ptr"), py::arg("size"))
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
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.sendv(conn_id, mr_id_v, data_v, size_v, num_iovs);
            }
            return success;
          },
          "Send multiple data buffers", py::arg("conn_id"), py::arg("mr_id_v"),
          py::arg("data_ptr_v"), py::arg("size_v"), py::arg("num_iovs"))
      .def(
          "recvv",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> data_ptr_v, std::vector<size_t> size_v,
             size_t num_iovs) {
            std::vector<void*> data_v;
            data_v.reserve(data_ptr_v.size());
            for (uint64_t ptr : data_ptr_v) {
              data_v.push_back(reinterpret_cast<void*>(ptr));
            }
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.recvv(conn_id, mr_id_v, data_v, size_v, num_iovs);
            }
            return success;
          },
          "Receive multiple data buffers asynchronously", py::arg("conn_id"),
          py::arg("mr_id_v"), py::arg("data_ptr_v"), py::arg("size_v"),
          py::arg("num_iovs"))
      .def(
          "sendv_async",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> data_ptr_v, std::vector<size_t> size_v,
             size_t num_iovs) {
            std::vector<void const*> data_v;
            data_v.reserve(data_ptr_v.size());
            for (uint64_t ptr : data_ptr_v) {
              data_v.push_back(reinterpret_cast<void const*>(ptr));
            }
            uint64_t transfer_id;
            bool success = self.sendv_async(conn_id, mr_id_v, data_v, size_v,
                                            num_iovs, &transfer_id);
            return py::make_tuple(success, transfer_id);
          },
          "Send multiple data buffers asynchronously", py::arg("conn_id"),
          py::arg("mr_id_v"), py::arg("data_ptr_v"), py::arg("size_v"),
          py::arg("num_iovs"))
      .def(
          "recvv_async",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> data_ptr_v, std::vector<size_t> size_v,
             size_t num_iovs) {
            std::vector<void*> data_v;
            data_v.reserve(data_ptr_v.size());
            for (uint64_t ptr : data_ptr_v) {
              data_v.push_back(reinterpret_cast<void*>(ptr));
            }
            uint64_t transfer_id;
            bool success = self.recvv_async(conn_id, mr_id_v, data_v, size_v,
                                            num_iovs, &transfer_id);

            return py::make_tuple(success, transfer_id);
          },
          "Receive multiple data buffers asynchronously", py::arg("conn_id"),
          py::arg("mr_id_v"), py::arg("data_ptr_v"), py::arg("size_v"),
          py::arg("num_iovs"))
      .def(
          "read",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, py::bytes meta_blob) {
            std::string buf = meta_blob;
            if (buf.size() != sizeof(uccl::FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            uccl::FifoItem item;
            uccl::deserialize_fifo_item(buf.data(), &item);
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.read(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, item);
            }
            return success;
          },
          "RDMA-READ into a local buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          py::arg("conn_id"), py::arg("mr_id"), py::arg("ptr"), py::arg("size"),
          py::arg("meta"))
      .def(
          "read_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, py::bytes meta_blob) {
            std::string buf = meta_blob;
            if (buf.size() != sizeof(uccl::FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            uccl::FifoItem item;
            uccl::deserialize_fifo_item(buf.data(), &item);
            uint64_t transfer_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.read_async(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, item, &transfer_id);
            }
            return py::make_tuple(success, transfer_id);
          },
          "RDMA-READ into a local buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          py::arg("conn_id"), py::arg("mr_id"), py::arg("ptr"), py::arg("size"),
          py::arg("meta"))
      .def(
          "readv",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             py::list meta_blob_v, size_t num_iovs) {
            if (mr_id_v.size() != num_iovs || ptr_v.size() != num_iovs ||
                size_v.size() != num_iovs || py::len(meta_blob_v) != num_iovs) {
              throw std::runtime_error(
                  "All input vectors/lists must have length num_iovs");
            }
            std::vector<uccl::FifoItem> item_v;
            item_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              std::string buf = py::cast<py::bytes>(meta_blob_v[i]);
              if (buf.size() != sizeof(uccl::FifoItem))
                throw std::runtime_error(
                    "meta must be exactly 64 bytes (serialized FifoItem)");
              uccl::FifoItem item;
              uccl::deserialize_fifo_item(buf.data(), &item);
              item_v.push_back(item);
            }
            std::vector<void*> data_v;
            data_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_v.push_back(reinterpret_cast<void*>(ptr_v[i]));
            }
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.readv(conn_id, mr_id_v, data_v, size_v, item_v,
                              num_iovs);
            }
            return ok;
          },
          "RDMA-READ into multiple local buffers using metadata from "
          "advertisev(); "
          "`meta_blob_v` is a list of 64-byte serialized FifoItem returned by "
          "the peer",
          py::arg("conn_id"), py::arg("mr_id_v"), py::arg("ptr_v"),
          py::arg("size_v"), py::arg("meta_blob_v"), py::arg("num_iovs"))
      .def(
          "write",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, py::bytes meta_blob) {
            std::string buf = meta_blob;
            if (buf.size() != sizeof(uccl::FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            uccl::FifoItem item;
            uccl::deserialize_fifo_item(buf.data(), &item);
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.write(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                   size, item);
            }
            return success;
          },
          "RDMA-WRITE into a remote buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          py::arg("conn_id"), py::arg("mr_id"), py::arg("ptr"), py::arg("size"),
          py::arg("meta"))
      .def(
          "write_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, py::bytes meta_blob) {
            std::string buf = meta_blob;
            if (buf.size() != sizeof(uccl::FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            uccl::FifoItem item;
            uccl::deserialize_fifo_item(buf.data(), &item);
            uint64_t transfer_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.write_async(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                   size, item, &transfer_id);
            }
            return py::make_tuple(success, transfer_id);
          },
          "RDMA-WRITE into a remote buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          py::arg("conn_id"), py::arg("mr_id"), py::arg("ptr"), py::arg("size"),
          py::arg("meta"))
      .def(
          "writev",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             py::list meta_blob_v, size_t num_iovs) {
            if (mr_id_v.size() != num_iovs || ptr_v.size() != num_iovs ||
                size_v.size() != num_iovs || py::len(meta_blob_v) != num_iovs) {
              throw std::runtime_error(
                  "All input vectors/lists must have length num_iovs");
            }
            std::vector<uccl::FifoItem> item_v;
            item_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              std::string buf = py::cast<py::bytes>(meta_blob_v[i]);
              if (buf.size() != sizeof(uccl::FifoItem))
                throw std::runtime_error(
                    "meta must be exactly 64 bytes (serialized FifoItem)");
              uccl::FifoItem item;
              uccl::deserialize_fifo_item(buf.data(), &item);
              item_v.push_back(item);
            }
            std::vector<void*> data_v;
            data_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_v.push_back(reinterpret_cast<void*>(ptr_v[i]));
            }
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.writev(conn_id, mr_id_v, data_v, size_v, item_v,
                               num_iovs);
            }
            return ok;
          },
          "RDMA-WRITE into multiple remote buffers using metadata from "
          "advertisev(); "
          "`meta_blob_v` is a list of 64-byte serialized FifoItem returned by "
          "the peer",
          py::arg("conn_id"), py::arg("mr_id_v"), py::arg("ptr_v"),
          py::arg("size_v"), py::arg("meta_blob_v"), py::arg("num_iovs"))
      .def(
          "advertise",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id,
             uint64_t ptr,  // raw pointer passed from Python
             size_t size) {
            char
                serialized[sizeof(uccl::FifoItem)]{};  // 64-byte scratch buffer
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.advertise(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, serialized);
            }
            /* return (success, bytes) — empty bytes when failed */
            return py::make_tuple(
                ok, ok ? py::bytes(serialized, sizeof(uccl::FifoItem))
                       : py::bytes());
          },
          "Expose a registered buffer for the peer to RDMA-READ or RDMA-WRITE",
          py::arg("conn_id"), py::arg("mr_id"), py::arg("ptr"), py::arg("size"))
      .def(
          "advertisev",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             size_t num_iovs) {
            std::vector<char*> serialized_vec(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              serialized_vec[i] = new char[sizeof(uccl::FifoItem)];
              memset(serialized_vec[i], 0, sizeof(uccl::FifoItem));
            }
            std::vector<void*> data_v;
            data_v.reserve(ptr_v.size());
            for (uint64_t ptr : ptr_v) {
              data_v.push_back(reinterpret_cast<void*>(ptr));
            }
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.advertisev(conn_id, mr_id_v, data_v, size_v,
                                   serialized_vec, num_iovs);
            }

            py::list py_bytes_list;
            for (size_t i = 0; i < num_iovs; ++i) {
              py_bytes_list.append(
                  py::bytes(serialized_vec[i], sizeof(uccl::FifoItem)));
            }
            for (size_t i = 0; i < num_iovs; ++i) {
              delete[] serialized_vec[i];
            }
            return py::make_tuple(ok, py_bytes_list);
          },
          "Expose multiple registered buffers for the peer to RDMA-READ or "
          "RDMA-WRITE",
          py::arg("conn_id"), py::arg("mr_id_v"), py::arg("ptr_v"),
          py::arg("size_v"), py::arg("num_iovs"))
      // IPC-specific functions for local connections via Unix Domain Sockets
      .def(
          "connect_local",
          [](Endpoint& self, int remote_gpu_idx) {
            uint64_t conn_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.connect_local(remote_gpu_idx, conn_id);
            }
            return py::make_tuple(success, conn_id);
          },
          "Connect to a local process via Unix Domain Socket",
          py::arg("remote_gpu_idx"))
      .def(
          "accept_local",
          [](Endpoint& self) {
            int remote_gpu_idx;
            uint64_t conn_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.accept_local(remote_gpu_idx, conn_id);
            }
            return py::make_tuple(success, remote_gpu_idx, conn_id);
          },
          "Accept an incoming local connection via Unix Domain Socket")
      .def(
          "send_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.send_ipc(conn_id, reinterpret_cast<void*>(ptr), size);
            }
            return success;
          },
          "Send data via IPC (Inter-Process Communication) using CUDA/HIP "
          "memory handles",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"))
      .def(
          "recv_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.recv_ipc(conn_id, reinterpret_cast<void*>(ptr), size);
            }
            return success;
          },
          "Receive data via IPC (Inter-Process Communication) using CUDA/HIP "
          "memory handles",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"))
      .def(
          "send_ipc_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            uint64_t transfer_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.send_ipc_async(conn_id,
                                            reinterpret_cast<void const*>(ptr),
                                            size, &transfer_id);
            }
            return py::make_tuple(success, transfer_id);
          },
          "Send data asynchronously via IPC using CUDA/HIP memory handles",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"))
      .def(
          "recv_ipc_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            uint64_t transfer_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.recv_ipc_async(
                  conn_id, reinterpret_cast<void*>(ptr), size, &transfer_id);
            }
            return py::make_tuple(success, transfer_id);
          },
          "Receive data asynchronously via IPC using CUDA/HIP memory handles",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"))
      .def(
          "write_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size,
             py::bytes info_blob) {
            std::string buf = info_blob;
            CHECK_EQ(buf.size(), sizeof(Endpoint::IpcTransferInfo))
                << "IpcTransferInfo size mismatch";
            Endpoint::IpcTransferInfo info;
            std::memcpy(&info, buf.data(), sizeof(info));
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.write_ipc(
                  conn_id, reinterpret_cast<void const*>(ptr), size, info);
            }
            return success;
          },
          "Write data via one-sided IPC using IpcTransferInfo",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"), py::arg("info"))
      .def(
          "read_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size,
             py::bytes info_blob) {
            std::string buf = info_blob;
            CHECK_EQ(buf.size(), sizeof(Endpoint::IpcTransferInfo))
                << "IpcTransferInfo size mismatch";
            Endpoint::IpcTransferInfo info;
            std::memcpy(&info, buf.data(), sizeof(info));
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.read_ipc(conn_id, reinterpret_cast<void*>(ptr),
                                      size, info);
            }
            return success;
          },
          "Read data via one-sided IPC using IpcTransferInfo",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"), py::arg("info"))
      .def(
          "write_ipc_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size,
             py::bytes info_blob) {
            std::string buf = info_blob;
            CHECK_EQ(buf.size(), sizeof(Endpoint::IpcTransferInfo))
                << "IpcTransferInfo size mismatch";
            Endpoint::IpcTransferInfo info;
            std::memcpy(&info, buf.data(), sizeof(info));
            uint64_t transfer_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.write_ipc_async(conn_id,
                                             reinterpret_cast<void const*>(ptr),
                                             size, info, &transfer_id);
            }
            return success;
          },
          "Write data asynchronously via one-sided IPC using IpcTransferInfo",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"), py::arg("info"))
      .def(
          "read_ipc_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size,
             py::bytes info_blob) {
            std::string buf = info_blob;
            CHECK_EQ(buf.size(), sizeof(Endpoint::IpcTransferInfo))
                << "IpcTransferInfo size mismatch";
            Endpoint::IpcTransferInfo info;
            std::memcpy(&info, buf.data(), sizeof(info));
            uint64_t transfer_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.read_ipc_async(conn_id, reinterpret_cast<void*>(ptr),
                                      size, info, &transfer_id);
            }
            return py::make_tuple(success, transfer_id);
          },
          "Read data asynchronously via one-sided IPC using IpcTransferInfo",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"), py::arg("info"))
      .def(
          "advertise_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            char serialized[sizeof(Endpoint::IpcTransferInfo)]{};
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.advertise_ipc(
                  conn_id, reinterpret_cast<void*>(ptr), size, serialized);
            }
            return py::make_tuple(success,
                                  py::bytes(serialized, sizeof(serialized)));
          },
          "Advertise memory for IPC access and return IpcTransferInfo",
          py::arg("conn_id"), py::arg("ptr"), py::arg("size"))
      .def(
          "advertisev_ipc",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> ptr_v,
             std::vector<size_t> size_v) {
            size_t num_iovs = ptr_v.size();
            CHECK_EQ(size_v.size(), num_iovs) << "Size vector mismatch";

            std::vector<void*> addr_v(num_iovs);
            std::vector<char*> out_buf_v(num_iovs);
            std::vector<std::string> buffers(num_iovs);

            for (size_t i = 0; i < num_iovs; ++i) {
              addr_v[i] = reinterpret_cast<void*>(ptr_v[i]);
              buffers[i].resize(sizeof(Endpoint::IpcTransferInfo));
              out_buf_v[i] = buffers[i].data();
            }

            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.advertisev_ipc(conn_id, addr_v, size_v, out_buf_v,
                                            num_iovs);
            }

            std::vector<py::bytes> result_v;
            for (size_t i = 0; i < num_iovs; ++i) {
              result_v.push_back(py::bytes(buffers[i]));
            }

            return py::make_tuple(success, result_v);
          },
          "Advertise multiple memory regions for IPC access",
          py::arg("conn_id"), py::arg("ptr_v"), py::arg("size_v"))
      .def(
          "poll_async",
          [](Endpoint& self, uint64_t transfer_id) {
            bool is_done;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;

              success = self.poll_async(transfer_id, &is_done);
            }
            return py::make_tuple(success, is_done);
          },
          "Poll the status of an asynchronous transfer", py::arg("transfer_id"))
      .def(
          "conn_id_of_rank", &Endpoint::conn_id_of_rank,
          "Get the connection ID for a given peer rank (or UINT64_MAX if none)",
          py::arg("rank"))
      .def("__repr__", [](Endpoint const& e) { return "<UCCL P2P Endpoint>"; });
}
#include "endpoint_wrapper.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;

namespace {
struct InsidePythonGuard {
  InsidePythonGuard() { inside_python = true; }
  ~InsidePythonGuard() { inside_python = false; }
};

struct XferDesc {
  void const* addr;
  size_t size;
  uint64_t mr_id;
  std::vector<uint32_t> lkeys;
  std::vector<uint32_t> rkeys;
  std::string ipc_info;  // serialized IpcTransferInfo (empty if unavailable)
};

struct XferHandle {
  uint64_t conn_id;
  std::string op_name;
  uint64_t transfer_id;
};

std::vector<uint8_t> serialize_xfer_descs(
    std::vector<XferDesc> const& xfer_desc_v) {
  size_t total_size = sizeof(size_t);
  for (auto const& desc : xfer_desc_v) {
    assert(desc.lkeys.size() == desc.rkeys.size());
    total_size += sizeof(uint64_t) + sizeof(size_t) + sizeof(size_t) +
                  desc.lkeys.size() * sizeof(uint32_t) * 2 + sizeof(size_t) +
                  desc.ipc_info.size();
  }

  std::vector<uint8_t> result(total_size);
  uint8_t* p = result.data();
  auto emit = [&p](void const* src, size_t n) {
    std::memcpy(p, src, n);
    p += n;
  };

  size_t num_descs = xfer_desc_v.size();
  emit(&num_descs, sizeof(size_t));
  for (auto const& desc : xfer_desc_v) {
    uint64_t addr = reinterpret_cast<uint64_t>(desc.addr);
    size_t keys_count = desc.lkeys.size();
    emit(&addr, sizeof(uint64_t));
    emit(&desc.size, sizeof(size_t));
    emit(&keys_count, sizeof(size_t));
    emit(desc.lkeys.data(), keys_count * sizeof(uint32_t));
    emit(desc.rkeys.data(), keys_count * sizeof(uint32_t));
    size_t ipc_len = desc.ipc_info.size();
    emit(&ipc_len, sizeof(size_t));
    if (ipc_len > 0) {
      emit(desc.ipc_info.data(), ipc_len);
    }
  }
  return result;
}

std::vector<XferDesc> deserialize_xfer_descs(
    std::vector<uint8_t> const& serialized_data) {
  if (serialized_data.empty()) return {};

  uint8_t const* p = serialized_data.data();
  uint8_t const* end = p + serialized_data.size();
  auto consume = [&p, end](void* dst, size_t n) {
    if (p + n > end)
      throw std::runtime_error("Invalid serialized XferDesc data");
    std::memcpy(dst, p, n);
    p += n;
  };

  size_t num_descs;
  consume(&num_descs, sizeof(size_t));

  std::vector<XferDesc> xfer_desc_v;
  xfer_desc_v.reserve(num_descs);
  for (size_t i = 0; i < num_descs; ++i) {
    XferDesc desc;
    uint64_t addr;
    size_t keys_count;
    consume(&addr, sizeof(uint64_t));
    desc.addr = reinterpret_cast<void const*>(addr);
    consume(&desc.size, sizeof(size_t));
    consume(&keys_count, sizeof(size_t));
    desc.lkeys.resize(keys_count);
    consume(desc.lkeys.data(), keys_count * sizeof(uint32_t));
    desc.rkeys.resize(keys_count);
    consume(desc.rkeys.data(), keys_count * sizeof(uint32_t));
    if (p < end) {
      size_t ipc_len;
      consume(&ipc_len, sizeof(size_t));
      if (ipc_len > 0) {
        desc.ipc_info.resize(ipc_len);
        consume(desc.ipc_info.data(), ipc_len);
      }
    }
    xfer_desc_v.push_back(std::move(desc));
  }
  return xfer_desc_v;
}

// Helper function to extract pointer and size from PyTorch tensor
std::pair<void const*, size_t> get_tensor_info(nb::object tensor_obj) {
  // Get data pointer using Python's data_ptr() method
  if (!nb::hasattr(tensor_obj, "data_ptr")) {
    throw std::runtime_error("Tensor does not have data_ptr() method");
  }
  nb::object data_ptr_result = tensor_obj.attr("data_ptr")();
  uint64_t ptr = nb::cast<uint64_t>(data_ptr_result);

  // Get number of elements
  if (!nb::hasattr(tensor_obj, "numel")) {
    throw std::runtime_error("Tensor does not have numel() method");
  }
  nb::object numel_result = tensor_obj.attr("numel")();
  int64_t numel = nb::cast<int64_t>(numel_result);

  // Get element size
  if (!nb::hasattr(tensor_obj, "element_size")) {
    throw std::runtime_error("Tensor does not have element_size() method");
  }
  nb::object element_size_result = tensor_obj.attr("element_size")();
  int64_t element_size = nb::cast<int64_t>(element_size_result);

  size_t total_size = static_cast<size_t>(numel * element_size);
  return std::make_pair(reinterpret_cast<void const*>(ptr), total_size);
}
}  // namespace

NB_MODULE(p2p, m) {
  m.doc() = "P2P Engine - High-performance RDMA-based peer-to-peer transport";

  m.def("get_oob_ip", &uccl::get_oob_ip, "Get the OOB IP address");

  // Register XferHandle type
  nb::class_<XferHandle>(m, "XferHandle")
      .def_ro("conn_id", &XferHandle::conn_id)
      .def_ro("op_name", &XferHandle::op_name)
      .def_ro("transfer_id", &XferHandle::transfer_id);

  // Register XferDesc type
  nb::class_<XferDesc>(m, "XferDesc")
      .def(nb::init<>())
      .def_prop_rw(
          "addr",
          [](XferDesc const& d) { return reinterpret_cast<uint64_t>(d.addr); },
          [](XferDesc& d, uint64_t v) {
            d.addr = reinterpret_cast<void const*>(v);
          })
      .def_rw("size", &XferDesc::size)
      .def_rw("mr_id", &XferDesc::mr_id)
      .def_rw("lkeys", &XferDesc::lkeys)
      .def_rw("rkeys", &XferDesc::rkeys)
      .def("__repr__", [](XferDesc const& d) {
        return "<XferDesc addr=" +
               std::to_string(reinterpret_cast<uint64_t>(d.addr)) +
               " size=" + std::to_string(d.size) +
               " mr_id=" + std::to_string(d.mr_id) + ">";
      });

  // Endpoint class binding
  nb::class_<Endpoint>(m, "Endpoint")
      .def("__init__",
           [](Endpoint *self, uint32_t local_gpu_idx, uint32_t num_cpus) {
             nb::gil_scoped_release release;
             InsidePythonGuard guard;
             new (self) Endpoint(local_gpu_idx, num_cpus);
           },
           nb::arg("local_gpu_idx"), nb::arg("num_cpus"))
      .def("__init__",
           [](Endpoint *self, uint32_t num_cpus) {
             nb::gil_scoped_release release;
             InsidePythonGuard guard;
             new (self) Endpoint(num_cpus);
           },
           nb::arg("num_cpus"))
      .def(
          "start_passive_accept",
          [](Endpoint& self) { return self.start_passive_accept(); },
          "Start a background thread for accepting.")
      .def(
          "connect",
          [](Endpoint& self, std::string const& remote_ip_addr,
             int remote_gpu_idx, int remote_port) {
            uint64_t conn_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.connect(remote_ip_addr, remote_gpu_idx,
                                     remote_port, conn_id);
            }
            return nb::make_tuple(success, conn_id);
          },
          "Connect to a remote server", nb::arg("remote_ip_addr"),
          nb::arg("remote_gpu_idx"), nb::arg("remote_port") = -1)
      .def(
          "add_remote_endpoint",
          [](Endpoint& self, nb::bytes metadata_bytes) {
            uint64_t conn_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              std::string buf(metadata_bytes.c_str(), metadata_bytes.size());
              std::vector<uint8_t> metadata(buf.begin(), buf.end());
              success = self.add_remote_endpoint(metadata, conn_id);
            }
            return nb::make_tuple(success, conn_id);
          },
          "Add remote endpoint - connect only once per remote endpoint.",
          nb::arg("metadata_bytes"))
      .def(
          "get_metadata",
          [](Endpoint& self) {
            std::vector<uint8_t> metadata = self.get_metadata();
            return nb::bytes(reinterpret_cast<char const*>(metadata.data()),
                             metadata.size());
          },
          "Return endpoint metadata as a list of bytes")
      .def_static(
          "parse_metadata",
          [](nb::bytes metadata_bytes) {
            std::string buf(metadata_bytes.c_str(), metadata_bytes.size());
            std::vector<uint8_t> metadata(buf.begin(), buf.end());
            auto [ip, port, gpu_idx] = Endpoint::parse_metadata(metadata);
            return nb::make_tuple(ip, port, gpu_idx);
          },
          "Parse endpoint metadata to extract IP address, port, and GPU index",
          nb::arg("metadata"))
      .def(
          "accept",
          [](Endpoint& self) {
            bool success;
            std::string remote_ip_addr;
            int remote_gpu_idx;
            uint64_t conn_id;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.accept(remote_ip_addr, remote_gpu_idx, conn_id);
            }
            return nb::make_tuple(success, remote_ip_addr, remote_gpu_idx,
                                  conn_id);
          },
          "Accept an incoming connection")
      .def(
          "reg",
          [](Endpoint& self, uint64_t ptr, size_t size) {
            uint64_t mr_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.reg(reinterpret_cast<void const*>(ptr), size, mr_id);
            }
            return nb::make_tuple(success, mr_id);
          },
          "Register a data buffer", nb::arg("ptr"), nb::arg("size"))
      .def(
          "regv",
          [](Endpoint& self, std::vector<uintptr_t> const& ptrs,
             std::vector<size_t> const& sizes) {
            if (ptrs.size() != sizes.size())
              throw std::runtime_error("ptrs and sizes must match");

            std::vector<void const*> data_v;
            data_v.reserve(ptrs.size());
            for (auto p : ptrs)
              data_v.push_back(reinterpret_cast<void const*>(p));

            std::vector<uint64_t> mr_ids;
            bool ok;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.regv(data_v, sizes, mr_ids);
            }
            return nb::make_tuple(ok, nb::cast(mr_ids));
          },
          nb::arg("ptrs"), nb::arg("sizes"),
          "Batch-register multiple memory regions and return [ok, mr_id_list]")
      .def(
          "register_memory",
          [](Endpoint& self, nb::list tensor_list) {
            std::vector<void const*> ptrs;
            std::vector<size_t> sizes;
            size_t list_len = nb::len(tensor_list);
            ptrs.reserve(list_len);
            sizes.reserve(list_len);

            for (size_t i = 0; i < list_len; ++i) {
              nb::object tensor_obj = tensor_list[i];
              if (!nb::hasattr(tensor_obj, "data_ptr")) {
                throw std::runtime_error(
                    "Object at index " + std::to_string(i) +
                    " is not a tensor (missing data_ptr() method)");
              }
              auto [ptr, size] = get_tensor_info(tensor_obj);
              ptrs.push_back(ptr);
              sizes.push_back(size);
            }

            std::vector<XferDesc> xfer_desc_v;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;

              std::vector<uint64_t> mr_id_v;
              if (!self.regv(ptrs, sizes, mr_id_v)) {
                return std::vector<XferDesc>{};
              }

              xfer_desc_v.reserve(list_len);
              for (size_t i = 0; i < list_len; i++) {
                auto mhandle = self.get_mhandle(mr_id_v[i]);
                assert(mhandle != nullptr);
                XferDesc xfer_desc;
                xfer_desc.addr = ptrs[i];
                xfer_desc.size = sizes[i];
                xfer_desc.mr_id = mr_id_v[i];
                for (size_t j = 0; j < kNICContextNumber; j++) {
                  auto mr = mhandle->mr_array.getKeyByContextID(j);
                  assert(mr != nullptr);
                  xfer_desc.lkeys.push_back(mr->lkey);
                  xfer_desc.rkeys.push_back(mr->rkey);
                }
                // Also generate IPC handle for intra-node transfers
                {
                  constexpr size_t kIpcAlign = 1ul << 20;
                  IpcTransferInfo ipc_info = {};
                  ipc_info.size = sizes[i];
                  ipc_info.operation = 1;
                  auto addr_val = reinterpret_cast<uintptr_t>(ptrs[i]);
                  auto addr_aligned = addr_val & ~(kIpcAlign - 1);
                  ipc_info.offset = addr_val - addr_aligned;
                  auto err = gpuIpcGetMemHandle(
                      &ipc_info.handle, reinterpret_cast<void*>(addr_aligned));
                  if (err != gpuSuccess) {
                    throw std::runtime_error(gpuGetErrorString(err));
                  }
                  xfer_desc.ipc_info.assign(
                      reinterpret_cast<char const*>(&ipc_info),
                      sizeof(ipc_info));
                }
                xfer_desc_v.push_back(std::move(xfer_desc));
              }
            }
            return xfer_desc_v;
          },
          "Register memory for a list of PyTorch tensors and return transfer "
          "descriptors",
          nb::arg("tensor_list"))
      .def(
          "get_serialized_descs",
          [](Endpoint& /*self*/, nb::list desc_list) {
            std::vector<XferDesc> xfer_desc_v;
            size_t list_len = nb::len(desc_list);
            xfer_desc_v.reserve(list_len);
            for (size_t i = 0; i < list_len; ++i) {
              xfer_desc_v.push_back(nb::cast<XferDesc const&>(desc_list[i]));
            }

            auto serialized = serialize_xfer_descs(xfer_desc_v);
            return nb::bytes(reinterpret_cast<const char*>(serialized.data()),
                             serialized.size());
          },
          "Serialize transfer descriptors to bytes for network transmission",
          nb::arg("desc_list"))
      .def(
          "deserialize_descs",
          [](Endpoint& /*self*/, nb::bytes serialized_bytes) {
            std::string buf(serialized_bytes.c_str(), serialized_bytes.size());
            std::vector<uint8_t> serialized_data(buf.begin(), buf.end());
            return deserialize_xfer_descs(serialized_data);
          },
          "Deserialize bytes to transfer descriptors",
          nb::arg("serialized_bytes"))
      .def(
          "transfer",
          [](Endpoint& self, uint64_t conn_id, std::string const& op_name,
             nb::list local_desc_list, nb::list remote_desc_list) {
            size_t n = nb::len(local_desc_list);
            if (n != nb::len(remote_desc_list)) {
              throw std::runtime_error(
                  "Local and remote descriptors must have the same size");
            }
            bool is_write = (op_name == "write");
            if (!is_write && op_name != "read") {
              throw std::runtime_error("Invalid op_name: " + op_name);
            }

            Conn* conn = self.get_conn(conn_id);
            if (conn == nullptr) {
              throw std::runtime_error("Invalid conn_id");
            }
            bool is_local = conn->is_local_;

            uint64_t transfer_id;
            bool success;

            if (is_local) {
              // IPC path for intra-node
              if (n == 1) {
                auto const& ldesc =
                    nb::cast<XferDesc const&>(local_desc_list[0]);
                auto const& rdesc =
                    nb::cast<XferDesc const&>(remote_desc_list[0]);
                CHECK(!rdesc.ipc_info.empty())
                    << "Remote descriptor has no IPC info for local transfer";
                IpcTransferInfo info;
                std::memcpy(&info, rdesc.ipc_info.data(), sizeof(info));
                {
                  nb::gil_scoped_release release;
                  InsidePythonGuard guard;
                  if (is_write) {
                    success = self.write_ipc_async(
                        conn_id, ldesc.addr, ldesc.size, info, &transfer_id);
                  } else {
                    success = self.read_ipc_async(
                        conn_id, const_cast<void*>(ldesc.addr), ldesc.size,
                        info, &transfer_id);
                  }
                }
              } else {
                std::vector<IpcTransferInfo> info_v;
                std::vector<size_t> size_v;
                info_v.reserve(n);
                size_v.reserve(n);

                // Collect write ptrs (const) and read ptrs (mutable) together
                std::vector<void const*> wdata_v;
                std::vector<void*> rdata_v;
                if (is_write)
                  wdata_v.reserve(n);
                else
                  rdata_v.reserve(n);

                for (size_t i = 0; i < n; ++i) {
                  auto const& ldesc =
                      nb::cast<XferDesc const&>(local_desc_list[i]);
                  auto const& rdesc =
                      nb::cast<XferDesc const&>(remote_desc_list[i]);
                  CHECK(!rdesc.ipc_info.empty())
                      << "Remote descriptor has no IPC info for local transfer";
                  IpcTransferInfo info;
                  std::memcpy(&info, rdesc.ipc_info.data(), sizeof(info));
                  info_v.push_back(info);
                  size_v.push_back(ldesc.size);
                  if (is_write)
                    wdata_v.push_back(ldesc.addr);
                  else
                    rdata_v.push_back(const_cast<void*>(ldesc.addr));
                }
                {
                  nb::gil_scoped_release release;
                  InsidePythonGuard guard;
                  if (is_write) {
                    success = self.writev_ipc_async(conn_id, wdata_v, size_v,
                                                    info_v, n, &transfer_id);
                  } else {
                    success = self.readv_ipc_async(conn_id, rdata_v, size_v,
                                                   info_v, n, &transfer_id);
                  }
                }
              }
            } else {
              // RDMA path for inter-node
              if (n == 1) {
                auto const& ldesc =
                    nb::cast<XferDesc const&>(local_desc_list[0]);
                auto const& rdesc =
                    nb::cast<XferDesc const&>(remote_desc_list[0]);

                FifoItem fifo_item;
                fifo_item.addr = reinterpret_cast<uint64_t>(rdesc.addr);
                fifo_item.size = rdesc.size;
                std::memcpy(fifo_item.padding, rdesc.rkeys.data(),
                            rdesc.rkeys.size() * sizeof(uint32_t));

                {
                  nb::gil_scoped_release release;
                  InsidePythonGuard guard;
                  if (is_write) {
                    success = self.write_async(
                        conn_id, ldesc.mr_id, const_cast<void*>(ldesc.addr),
                        ldesc.size, fifo_item, &transfer_id);
                  } else {
                    success = self.read_async(
                        conn_id, ldesc.mr_id, const_cast<void*>(ldesc.addr),
                        ldesc.size, fifo_item, &transfer_id);
                  }
                }
              } else {
                std::vector<FifoItem> slot_item_v;
                std::vector<uint64_t> mr_id_v;
                std::vector<void*> src_v;
                std::vector<size_t> size_v;
                slot_item_v.reserve(n);
                mr_id_v.reserve(n);
                src_v.reserve(n);
                size_v.reserve(n);

                for (size_t i = 0; i < n; ++i) {
                  auto const& ldesc =
                      nb::cast<XferDesc const&>(local_desc_list[i]);
                  auto const& rdesc =
                      nb::cast<XferDesc const&>(remote_desc_list[i]);

                  FifoItem fifo_item;
                  fifo_item.addr = reinterpret_cast<uint64_t>(rdesc.addr);
                  fifo_item.size = rdesc.size;
                  std::memcpy(fifo_item.padding, rdesc.rkeys.data(),
                              rdesc.rkeys.size() * sizeof(uint32_t));

                  slot_item_v.push_back(fifo_item);
                  mr_id_v.push_back(ldesc.mr_id);
                  src_v.push_back(const_cast<void*>(ldesc.addr));
                  size_v.push_back(ldesc.size);
                }

                {
                  nb::gil_scoped_release release;
                  InsidePythonGuard guard;
                  if (is_write) {
                    success = self.writev_async(conn_id, mr_id_v, src_v, size_v,
                                                slot_item_v, n, &transfer_id);
                  } else {
                    success = self.readv_async(conn_id, mr_id_v, src_v, size_v,
                                               slot_item_v, n, &transfer_id);
                  }
                }
              }
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Start a transfer and return (success, transfer_id)",
          nb::arg("conn_id"), nb::arg("op_name"), nb::arg("local_desc_list"),
          nb::arg("remote_desc_list"))
      .def(
          "dereg",
          [](Endpoint& self, uint64_t mr_id) {
            bool ok;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.dereg(mr_id);
            }
            return ok;
          },
          "Deregister a memory region", nb::arg("mr_id"))
      .def(
          "send",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.send(conn_id, mr_id,
                                  reinterpret_cast<void const*>(ptr), size);
            }
            return success;
          },
          "Send a data buffer, optionally using metadata (serialized FifoItem)",
          nb::arg("conn_id"), nb::arg("mr_id"), nb::arg("ptr"), nb::arg("size"))
      .def(
          "recv",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.recv(conn_id, mr_id, reinterpret_cast<void*>(ptr), size);
            }
            return success;
          },
          "Receive a key-value buffer", nb::arg("conn_id"), nb::arg("mr_id"),
          nb::arg("ptr"), nb::arg("size"))
      .def(
          "send_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.send_async(conn_id, mr_id,
                                        reinterpret_cast<void const*>(ptr),
                                        size, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Send data asynchronously", nb::arg("conn_id"), nb::arg("mr_id"),
          nb::arg("ptr"), nb::arg("size"))
      .def(
          "recv_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size) {
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.recv_async(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Receive data asynchronously", nb::arg("conn_id"), nb::arg("mr_id"),
          nb::arg("ptr"), nb::arg("size"))
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
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.sendv(conn_id, mr_id_v, data_v, size_v, num_iovs);
            }
            return success;
          },
          "Send multiple data buffers", nb::arg("conn_id"), nb::arg("mr_id_v"),
          nb::arg("data_ptr_v"), nb::arg("size_v"), nb::arg("num_iovs"))
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
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.recvv(conn_id, mr_id_v, data_v, size_v, num_iovs);
            }
            return success;
          },
          "Receive multiple data buffers asynchronously", nb::arg("conn_id"),
          nb::arg("mr_id_v"), nb::arg("data_ptr_v"), nb::arg("size_v"),
          nb::arg("num_iovs"))
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
            return nb::make_tuple(success, transfer_id);
          },
          "Send multiple data buffers asynchronously", nb::arg("conn_id"),
          nb::arg("mr_id_v"), nb::arg("data_ptr_v"), nb::arg("size_v"),
          nb::arg("num_iovs"))
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

            return nb::make_tuple(success, transfer_id);
          },
          "Receive multiple data buffers asynchronously", nb::arg("conn_id"),
          nb::arg("mr_id_v"), nb::arg("data_ptr_v"), nb::arg("size_v"),
          nb::arg("num_iovs"))
      .def(
          "read",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, nb::bytes meta_blob) {
            std::string buf(meta_blob.c_str(), meta_blob.size());
            if (buf.size() != sizeof(FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            FifoItem item;
            deserialize_fifo_item(buf.data(), &item);
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.read(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, item);
            }
            return success;
          },
          "RDMA-READ into a local buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          nb::arg("conn_id"), nb::arg("mr_id"), nb::arg("ptr"), nb::arg("size"),
          nb::arg("meta"))
      .def(
          "read_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, nb::bytes meta_blob) {
            std::string buf(meta_blob.c_str(), meta_blob.size());
            if (buf.size() != sizeof(FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            FifoItem item;
            deserialize_fifo_item(buf.data(), &item);
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.read_async(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, item, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "RDMA-READ into a local buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          nb::arg("conn_id"), nb::arg("mr_id"), nb::arg("ptr"), nb::arg("size"),
          nb::arg("meta"))
      .def(
          "readv",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             nb::list meta_blob_v, size_t num_iovs) {
            if (mr_id_v.size() != num_iovs || ptr_v.size() != num_iovs ||
                size_v.size() != num_iovs || nb::len(meta_blob_v) != num_iovs) {
              throw std::runtime_error(
                  "All input vectors/lists must have length num_iovs");
            }
            std::vector<FifoItem> item_v;
            item_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              nb::bytes _b = nb::cast<nb::bytes>(meta_blob_v[i]);
              std::string buf(_b.c_str(), _b.size());
              if (buf.size() != sizeof(FifoItem))
                throw std::runtime_error(
                    "meta must be exactly 64 bytes (serialized FifoItem)");
              FifoItem item;
              deserialize_fifo_item(buf.data(), &item);
              item_v.push_back(item);
            }
            std::vector<void*> data_v;
            data_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_v.push_back(reinterpret_cast<void*>(ptr_v[i]));
            }
            bool ok;
            {
              nb::gil_scoped_release release;
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
          nb::arg("conn_id"), nb::arg("mr_id_v"), nb::arg("ptr_v"),
          nb::arg("size_v"), nb::arg("meta_blob_v"), nb::arg("num_iovs"))
      .def(
          "readv_async",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             nb::list meta_blob_v, size_t num_iovs) {
            if (mr_id_v.size() != num_iovs || ptr_v.size() != num_iovs ||
                size_v.size() != num_iovs || nb::len(meta_blob_v) != num_iovs) {
              throw std::runtime_error(
                  "All input vectors/lists must have length num_iovs");
            }
            std::vector<FifoItem> item_v;
            item_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              nb::bytes _b = nb::cast<nb::bytes>(meta_blob_v[i]);
              std::string buf(_b.c_str(), _b.size());
              if (buf.size() != sizeof(FifoItem))
                throw std::runtime_error(
                    "meta must be exactly 64 bytes (serialized FifoItem)");
              FifoItem item;
              deserialize_fifo_item(buf.data(), &item);
              item_v.push_back(item);
            }
            std::vector<void*> data_v;
            data_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_v.push_back(reinterpret_cast<void*>(ptr_v[i]));
            }
            uint64_t transfer_id;
            bool ok;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.readv_async(conn_id, mr_id_v, data_v, size_v, item_v,
                                    num_iovs, &transfer_id);
            }
            return nb::make_tuple(ok, transfer_id);
          },
          "RDMA-READ into multiple local buffers asynchronously using metadata "
          "from advertisev(); "
          "`meta_blob_v` is a list of 64-byte serialized FifoItem returned by "
          "the peer",
          nb::arg("conn_id"), nb::arg("mr_id_v"), nb::arg("ptr_v"),
          nb::arg("size_v"), nb::arg("meta_blob_v"), nb::arg("num_iovs"))
      .def(
          "write",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, nb::bytes meta_blob) {
            std::string buf(meta_blob.c_str(), meta_blob.size());
            if (buf.size() != sizeof(FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            FifoItem item;
            deserialize_fifo_item(buf.data(), &item);
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.write(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                   size, item);
            }
            return success;
          },
          "RDMA-WRITE into a remote buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          nb::arg("conn_id"), nb::arg("mr_id"), nb::arg("ptr"), nb::arg("size"),
          nb::arg("meta"))
      .def(
          "write_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, nb::bytes meta_blob) {
            std::string buf(meta_blob.c_str(), meta_blob.size());
            if (buf.size() != sizeof(FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            FifoItem item;
            deserialize_fifo_item(buf.data(), &item);
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.write_async(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                   size, item, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "RDMA-WRITE into a remote buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          nb::arg("conn_id"), nb::arg("mr_id"), nb::arg("ptr"), nb::arg("size"),
          nb::arg("meta"))
      .def(
          "writev",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             nb::list meta_blob_v, size_t num_iovs) {
            if (mr_id_v.size() != num_iovs || ptr_v.size() != num_iovs ||
                size_v.size() != num_iovs || nb::len(meta_blob_v) != num_iovs) {
              throw std::runtime_error(
                  "All input vectors/lists must have length num_iovs");
            }
            std::vector<FifoItem> item_v;
            item_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              nb::bytes _b = nb::cast<nb::bytes>(meta_blob_v[i]);
              std::string buf(_b.c_str(), _b.size());
              if (buf.size() != sizeof(FifoItem))
                throw std::runtime_error(
                    "meta must be exactly 64 bytes (serialized FifoItem)");
              FifoItem item;
              deserialize_fifo_item(buf.data(), &item);
              item_v.push_back(item);
            }
            std::vector<void*> data_v;
            data_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_v.push_back(reinterpret_cast<void*>(ptr_v[i]));
            }
            bool ok;
            {
              nb::gil_scoped_release release;
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
          nb::arg("conn_id"), nb::arg("mr_id_v"), nb::arg("ptr_v"),
          nb::arg("size_v"), nb::arg("meta_blob_v"), nb::arg("num_iovs"))
      .def(
          "writev_async",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             nb::list meta_blob_v, size_t num_iovs) {
            if (mr_id_v.size() != num_iovs || ptr_v.size() != num_iovs ||
                size_v.size() != num_iovs || nb::len(meta_blob_v) != num_iovs) {
              throw std::runtime_error(
                  "All input vectors/lists must have length num_iovs");
            }
            std::vector<FifoItem> item_v;
            item_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              nb::bytes _b = nb::cast<nb::bytes>(meta_blob_v[i]);
              std::string buf(_b.c_str(), _b.size());
              if (buf.size() != sizeof(FifoItem))
                throw std::runtime_error(
                    "meta must be exactly 64 bytes (serialized FifoItem)");
              FifoItem item;
              deserialize_fifo_item(buf.data(), &item);
              item_v.push_back(item);
            }
            std::vector<void*> data_v;
            data_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_v.push_back(reinterpret_cast<void*>(ptr_v[i]));
            }
            uint64_t transfer_id;
            bool ok;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.writev_async(conn_id, mr_id_v, data_v, size_v, item_v,
                                     num_iovs, &transfer_id);
            }
            return nb::make_tuple(ok, transfer_id);
          },
          "RDMA-WRITE into multiple remote buffers asynchronously using "
          "metadata from advertisev(); "
          "`meta_blob_v` is a list of 64-byte serialized FifoItem returned by "
          "the peer",
          nb::arg("conn_id"), nb::arg("mr_id_v"), nb::arg("ptr_v"),
          nb::arg("size_v"), nb::arg("meta_blob_v"), nb::arg("num_iovs"))
      .def(
          "advertise",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id,
             uint64_t ptr,  // raw pointer passed from Python
             size_t size) {
            char serialized[sizeof(FifoItem)]{};  // 64-byte scratch buffer
            bool ok;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.advertise(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, serialized);
            }
            /* return (success, bytes) — empty bytes when failed */
            return nb::make_tuple(
                ok, ok ? nb::bytes(serialized, sizeof(FifoItem)) : nb::bytes("", 0));
          },
          "Expose a registered buffer for the peer to RDMA-READ or RDMA-WRITE",
          nb::arg("conn_id"), nb::arg("mr_id"), nb::arg("ptr"), nb::arg("size"))
      .def(
          "advertisev",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             size_t num_iovs) {
            std::vector<char*> serialized_vec(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              serialized_vec[i] = new char[sizeof(FifoItem)];
              memset(serialized_vec[i], 0, sizeof(FifoItem));
            }
            std::vector<void*> data_v;
            data_v.reserve(ptr_v.size());
            for (uint64_t ptr : ptr_v) {
              data_v.push_back(reinterpret_cast<void*>(ptr));
            }
            bool ok;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.advertisev(conn_id, mr_id_v, data_v, size_v,
                                   serialized_vec, num_iovs);
            }

            nb::list py_bytes_list;
            for (size_t i = 0; i < num_iovs; ++i) {
              py_bytes_list.append(
                  nb::bytes(serialized_vec[i], sizeof(FifoItem)));
            }
            for (size_t i = 0; i < num_iovs; ++i) {
              delete[] serialized_vec[i];
            }
            return nb::make_tuple(ok, py_bytes_list);
          },
          "Expose multiple registered buffers for the peer to RDMA-READ or "
          "RDMA-WRITE",
          nb::arg("conn_id"), nb::arg("mr_id_v"), nb::arg("ptr_v"),
          nb::arg("size_v"), nb::arg("num_iovs"))
      // IPC-specific functions for local connections via Unix Domain Sockets
      .def(
          "connect_local",
          [](Endpoint& self, int remote_gpu_idx) {
            uint64_t conn_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.connect_local(remote_gpu_idx, conn_id);
            }
            return nb::make_tuple(success, conn_id);
          },
          "Connect to a local process via Unix Domain Socket",
          nb::arg("remote_gpu_idx"))
      .def(
          "accept_local",
          [](Endpoint& self) {
            int remote_gpu_idx;
            uint64_t conn_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.accept_local(remote_gpu_idx, conn_id);
            }
            return nb::make_tuple(success, remote_gpu_idx, conn_id);
          },
          "Accept an incoming local connection via Unix Domain Socket")
      .def(
          "send_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.send_ipc(conn_id, reinterpret_cast<void*>(ptr), size);
            }
            return success;
          },
          "Send data via IPC (Inter-Process Communication) using CUDA/HIP "
          "memory handles",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"))
      .def(
          "recv_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.recv_ipc(conn_id, reinterpret_cast<void*>(ptr), size);
            }
            return success;
          },
          "Receive data via IPC (Inter-Process Communication) using CUDA/HIP "
          "memory handles",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"))
      .def(
          "send_ipc_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.send_ipc_async(conn_id,
                                            reinterpret_cast<void const*>(ptr),
                                            size, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Send data asynchronously via IPC using CUDA/HIP memory handles",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"))
      .def(
          "recv_ipc_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.recv_ipc_async(
                  conn_id, reinterpret_cast<void*>(ptr), size, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Receive data asynchronously via IPC using CUDA/HIP memory handles",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"))
      .def(
          "write_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size,
             nb::bytes info_blob) {
            std::string buf(info_blob.c_str(), info_blob.size());
            CHECK_EQ(buf.size(), sizeof(IpcTransferInfo))
                << "IpcTransferInfo size mismatch";
            IpcTransferInfo info;
            std::memcpy(&info, buf.data(), sizeof(info));
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.write_ipc(
                  conn_id, reinterpret_cast<void const*>(ptr), size, info);
            }
            return success;
          },
          "Write data via one-sided IPC using IpcTransferInfo",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"), nb::arg("info"))
      .def(
          "read_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size,
             nb::bytes info_blob) {
            std::string buf(info_blob.c_str(), info_blob.size());
            CHECK_EQ(buf.size(), sizeof(IpcTransferInfo))
                << "IpcTransferInfo size mismatch";
            IpcTransferInfo info;
            std::memcpy(&info, buf.data(), sizeof(info));
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.read_ipc(conn_id, reinterpret_cast<void*>(ptr),
                                      size, info);
            }
            return success;
          },
          "Read data via one-sided IPC using IpcTransferInfo",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"), nb::arg("info"))
      .def(
          "write_ipc_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size,
             nb::bytes info_blob) {
            std::string buf(info_blob.c_str(), info_blob.size());
            CHECK_EQ(buf.size(), sizeof(IpcTransferInfo))
                << "IpcTransferInfo size mismatch";
            IpcTransferInfo info;
            std::memcpy(&info, buf.data(), sizeof(info));
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.write_ipc_async(conn_id,
                                             reinterpret_cast<void const*>(ptr),
                                             size, info, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Write data asynchronously via one-sided IPC using IpcTransferInfo",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"), nb::arg("info"))
      .def(
          "read_ipc_async",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size,
             nb::bytes info_blob) {
            std::string buf(info_blob.c_str(), info_blob.size());
            CHECK_EQ(buf.size(), sizeof(IpcTransferInfo))
                << "IpcTransferInfo size mismatch";
            IpcTransferInfo info;
            std::memcpy(&info, buf.data(), sizeof(info));
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.read_ipc_async(conn_id, reinterpret_cast<void*>(ptr),
                                      size, info, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Read data asynchronously via one-sided IPC using IpcTransferInfo",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"), nb::arg("info"))
      .def(
          "writev_ipc",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> ptr_v,
             std::vector<size_t> size_v, nb::list info_v) {
            size_t num_iovs = ptr_v.size();
            CHECK_EQ(size_v.size(), num_iovs) << "writev_ipc: size_v mismatch";
            CHECK_EQ(nb::len(info_v), num_iovs) << "writev_ipc: info_v mismatch";

            std::vector<void const*> data_ptrs(num_iovs);
            std::vector<IpcTransferInfo> infos(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_ptrs[i] = reinterpret_cast<void const*>(ptr_v[i]);
              nb::bytes b = nb::cast<nb::bytes>(info_v[i]);
              std::string buf(b.c_str(), b.size());
              CHECK_EQ(buf.size(), sizeof(IpcTransferInfo))
                  << "IpcTransferInfo size mismatch at index " << i;
              std::memcpy(&infos[i], buf.data(), sizeof(infos[i]));
            }
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.writev_ipc(conn_id, data_ptrs, size_v, infos, num_iovs);
            }
            return success;
          },
          "Write multiple buffers via one-sided IPC using IpcTransferInfo",
          nb::arg("conn_id"), nb::arg("ptr_v"), nb::arg("size_v"),
          nb::arg("info_v"))
      .def(
          "readv_ipc",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> ptr_v,
             std::vector<size_t> size_v, nb::list info_v) {
            size_t num_iovs = ptr_v.size();
            CHECK_EQ(size_v.size(), num_iovs) << "readv_ipc: size_v mismatch";
            CHECK_EQ(nb::len(info_v), num_iovs) << "readv_ipc: info_v mismatch";

            std::vector<void*> data_ptrs(num_iovs);
            std::vector<IpcTransferInfo> infos(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_ptrs[i] = reinterpret_cast<void*>(ptr_v[i]);
              nb::bytes b = nb::cast<nb::bytes>(info_v[i]);
              std::string buf(b.c_str(), b.size());
              CHECK_EQ(buf.size(), sizeof(IpcTransferInfo))
                  << "IpcTransferInfo size mismatch at index " << i;
              std::memcpy(&infos[i], buf.data(), sizeof(infos[i]));
            }
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.readv_ipc(conn_id, data_ptrs, size_v, infos, num_iovs);
            }
            return success;
          },
          "Read multiple buffers via one-sided IPC using IpcTransferInfo",
          nb::arg("conn_id"), nb::arg("ptr_v"), nb::arg("size_v"),
          nb::arg("info_v"))
      .def(
          "writev_ipc_async",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> ptr_v,
             std::vector<size_t> size_v, nb::list info_v) {
            size_t num_iovs = ptr_v.size();
            CHECK_EQ(size_v.size(), num_iovs)
                << "writev_ipc_async: size_v mismatch";
            CHECK_EQ(nb::len(info_v), num_iovs)
                << "writev_ipc_async: info_v mismatch";

            std::vector<void const*> data_ptrs(num_iovs);
            std::vector<IpcTransferInfo> infos(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_ptrs[i] = reinterpret_cast<void const*>(ptr_v[i]);
              nb::bytes b = nb::cast<nb::bytes>(info_v[i]);
              std::string buf(b.c_str(), b.size());
              CHECK_EQ(buf.size(), sizeof(IpcTransferInfo))
                  << "IpcTransferInfo size mismatch at index " << i;
              std::memcpy(&infos[i], buf.data(), sizeof(infos[i]));
            }
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.writev_ipc_async(conn_id, data_ptrs, size_v, infos,
                                              num_iovs, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Write multiple buffers asynchronously via one-sided IPC",
          nb::arg("conn_id"), nb::arg("ptr_v"), nb::arg("size_v"),
          nb::arg("info_v"))
      .def(
          "readv_ipc_async",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> ptr_v,
             std::vector<size_t> size_v, nb::list info_v) {
            size_t num_iovs = ptr_v.size();
            CHECK_EQ(size_v.size(), num_iovs)
                << "readv_ipc_async: size_v mismatch";
            CHECK_EQ(nb::len(info_v), num_iovs)
                << "readv_ipc_async: info_v mismatch";

            std::vector<void*> data_ptrs(num_iovs);
            std::vector<IpcTransferInfo> infos(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_ptrs[i] = reinterpret_cast<void*>(ptr_v[i]);
              nb::bytes b = nb::cast<nb::bytes>(info_v[i]);
              std::string buf(b.c_str(), b.size());
              CHECK_EQ(buf.size(), sizeof(IpcTransferInfo))
                  << "IpcTransferInfo size mismatch at index " << i;
              std::memcpy(&infos[i], buf.data(), sizeof(infos[i]));
            }
            uint64_t transfer_id;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.readv_ipc_async(conn_id, data_ptrs, size_v, infos,
                                             num_iovs, &transfer_id);
            }
            return nb::make_tuple(success, transfer_id);
          },
          "Read multiple buffers asynchronously via one-sided IPC",
          nb::arg("conn_id"), nb::arg("ptr_v"), nb::arg("size_v"),
          nb::arg("info_v"))
      .def(
          "advertise_ipc",
          [](Endpoint& self, uint64_t conn_id, uint64_t ptr, size_t size) {
            char serialized[sizeof(IpcTransferInfo)]{};
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.advertise_ipc(
                  conn_id, reinterpret_cast<void*>(ptr), size, serialized);
            }
            return nb::make_tuple(success,
                                  nb::bytes(serialized, sizeof(serialized)));
          },
          "Advertise memory for IPC access and return IpcTransferInfo",
          nb::arg("conn_id"), nb::arg("ptr"), nb::arg("size"))
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
              buffers[i].resize(sizeof(IpcTransferInfo));
              out_buf_v[i] = buffers[i].data();
            }

            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.advertisev_ipc(conn_id, addr_v, size_v, out_buf_v,
                                            num_iovs);
            }

            nb::list result_v;
            for (size_t i = 0; i < num_iovs; ++i) {
              result_v.append(nb::bytes(buffers[i].data(), buffers[i].size()));
            }

            return nb::make_tuple(success, result_v);
          },
          "Advertise multiple memory regions for IPC access",
          nb::arg("conn_id"), nb::arg("ptr_v"), nb::arg("size_v"))
      .def(
          "poll_async",
          [](Endpoint& self, uint64_t transfer_id) {
            bool is_done;
            bool success;
            {
              nb::gil_scoped_release release;
              InsidePythonGuard guard;

              success = self.poll_async(transfer_id, &is_done);
            }
            return nb::make_tuple(success, is_done);
          },
          "Poll the status of an asynchronous transfer", nb::arg("transfer_id"))
      .def(
          "conn_id_of_rank", &Endpoint::conn_id_of_rank,
          "Get the connection ID for a given peer rank (or UINT64_MAX if none)",
          nb::arg("rank"))
      .def("__repr__", [](Endpoint const& e) { return "<UCCL P2P Endpoint>"; });
}
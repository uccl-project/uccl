// SPDX-License-Identifier: Apache-2.0
// Out-of-place NCCL reference with the same Python enqueue boundary as PCIeCCL.
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <nccl.h>

namespace py = pybind11;

namespace {
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}

class Reference {
 public:
  Reference(int rank, int world, py::bytes const& bytes) {
    std::string value = bytes;
    if (value.size() != sizeof(ncclUniqueId))
      throw std::invalid_argument("invalid NCCL unique ID");
    ncclUniqueId id;
    std::memcpy(&id, value.data(), sizeof(id));
    check(ncclCommInitRank(&comm_, world, id, rank));
  }

  void allReduce(uintptr_t input, uintptr_t output, size_t count, int dtype,
                 uintptr_t stream) {
    if (!comm_) throw std::runtime_error("closed NCCL reference");
    if (dtype != 0 && dtype != 1)
      throw std::invalid_argument("unsupported dtype");
    check(ncclAllReduce(reinterpret_cast<void const*>(input),
                        reinterpret_cast<void*>(output), count,
                        dtype == 0 ? ncclBfloat16 : ncclFloat, ncclSum, comm_,
                        reinterpret_cast<cudaStream_t>(stream)));
  }

  void close() {
    if (comm_) {
      check(ncclCommDestroy(comm_));
      comm_ = nullptr;
    }
  }

 private:
  ncclComm_t comm_ = nullptr;
};
}  // namespace

PYBIND11_MODULE(_uccl_lite_nccl_reference, module) {
  module.def("create_unique_id", []() {
    ncclUniqueId id;
    check(ncclGetUniqueId(&id));
    return py::bytes(reinterpret_cast<const char*>(&id), sizeof(id));
  });
  module.def("version", []() {
    int version;
    check(ncclGetVersion(&version));
    return version;
  });
  py::class_<Reference>(module, "Communicator")
      .def(py::init<int, int, py::bytes const&>())
      .def("all_reduce", &Reference::allReduce)
      .def("close", &Reference::close);
}

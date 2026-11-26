#pragma once
#include <cuda_runtime_api.h>

namespace uccl_lite::cpu_router {
// Preserve the existing C++ ABI spelling while sharing the plain call data
// independently of the optional router implementation.
struct PacketPort {
  void const* input;
  void* output;
  cudaStream_t stream;
  int device;
};
}  // namespace uccl_lite::cpu_router
namespace uccl_lite {
using DevicePort = cpu_router::PacketPort;
}  // namespace uccl_lite

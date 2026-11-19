#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <cuda_runtime_api.h>

namespace uccl_lite {

// Explicit experimental two-host/four-GPU FP32 GPU hierarchy. No automatic
// admission or NCCL/MPI dependency. Operations use NCCL element-count
// semantics.
enum class NetworkOperation { AllGather = 1, ReduceScatter = 2, AllReduce = 3 };
using NetworkId = std::array<uint8_t, 128>;

class NetworkCollectives {
 public:
  static NetworkId createUniqueId();
  // Construction does not allocate GPU buffers, register memory or start
  // workers.
  NetworkCollectives(NetworkId id, int rank, int device, int rail,
                     size_t capacity);
  ~NetworkCollectives();
  NetworkCollectives(NetworkCollectives const&) = delete;
  NetworkCollectives& operator=(NetworkCollectives const&) = delete;
  // Collective initialization with idle streams, outside capture and timing.
  void prepare();
  // Serialized host calls; one prepared large communicator per process.
  // Caller ensures ordering against other backends.
  // Output and input reuse are safe after completion on the supplied stream.
  // Supports positive counts with complete extents divisible by 128 bytes,
  // through 1 GiB. Large calls use the retained GPU ring/pipeline. Legal
  // in-place and 4-byte aligned pointers are supported.
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream);
  // Wait for submitted work and retire outstanding transport completions.
  void synchronize();
  bool initialized() const;
  bool idle() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uccl_lite

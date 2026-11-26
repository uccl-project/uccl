#pragma once
#include "uccl_lite/collectives.h"

namespace uccl_lite {
// Experimental FP32 peer-packet executor. Its registry entry describes the
// currently qualified layout; it does not claim arbitrary-rank support.
class PeerPacketCollectives {
 public:
  static NetworkId createUniqueId();
  // Construction does not allocate GPU buffers, register memory or start
  // workers.
  PeerPacketCollectives(NetworkId id, int rank, int device, int rail,
                        size_t capacity);
  PeerPacketCollectives(NetworkId id, int rank, int device, int rail,
                        size_t capacity, std::array<int, 2> numaNodes);
  ~PeerPacketCollectives();
  PeerPacketCollectives(PeerPacketCollectives const&) = delete;
  PeerPacketCollectives& operator=(PeerPacketCollectives const&) = delete;
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

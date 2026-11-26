#pragma once
#include "uccl_lite/collectives.h"
#include "uccl_lite/direct_plan.hpp"
#include <string>
#include <vector>

namespace uccl_lite {
struct DirectExchangeRoute {
  int peer = -1;
  bool cudaIpc = false;
};
// General direct exchange: actual participants, no fixed host/GPU partition.
// Experimental executor, NOT automatically admitted by this class. A caller
// must qualify the exact workload against its baseline before choosing it.
// FP32/BF16 reductions accumulate in logical source order in FP32, round once.
// Calls are host-serialized and synchronous, outside external CUDA capture.
// Resources/registrations are created only by prepare. No MPI/NCCL dependency.
class DirectExchangeCollectives {
 public:
  static NetworkId createUniqueId();
  // Read-only distributed-transport capability gate; creates no GPU buffers,
  // registrations or connections. Exact performance still requires
  // qualification.
  static bool supportsCurrentDevice(ScalarType dtype);
  DirectExchangeCollectives(NetworkId id, int rank, int ranks, int device,
                            size_t capacity, ScalarType dtype,
                            size_t chunkBytes = 256 * 1024);
  ~DirectExchangeCollectives();
  DirectExchangeCollectives(DirectExchangeCollectives const&) = delete;
  DirectExchangeCollectives& operator=(DirectExchangeCollectives const&) =
      delete;
  void prepare();
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream);
  void synchronize();
  bool initialized() const;
  std::vector<DirectExchangeRoute> const& routes() const;
  // Includes actual host/UUID ownership, selected NIC, evidence provenance,
  // chunk and source-order contract. No performance/admission claim.
  std::string const& identity() const;
  DirectTraffic traffic(NetworkOperation op, size_t count) const;
  // Collective, idle, private-buffer calibration. Never called from enqueue.
  std::shared_ptr<DirectModelCalibration const> calibrateModel();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uccl_lite

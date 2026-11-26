#pragma once
#include "planner.hpp"
#include <array>

namespace uccl_lite {
enum class DirectStepKind { StageInput, Transfer };
struct DirectStep {
  DirectStepKind kind;
  int target;
};
// Shared by the actual executor and model. Transfer-to-self is a local copy.
std::vector<DirectStep> directExchangeSteps(Collective op, int ranks, int rank);
struct DirectCallSignature {
  size_t count;
  int op;
};
struct DirectStatus {
  char message[512]{};
};
struct DirectTraffic {
  size_t chunks = 0, gpuMemoryBytes = 0, peerPayloadBytes = 0,
         networkPayloadBytes = 0, controlPayloadBytes = 0;
  size_t inputCopies = 0, selfCopies = 0, outputCopies = 0,
         reductionLaunches = 0, callerStreamSynchronizations = 0,
         transportFlushes = 0, agreements = 0;
};
// Row-major directed IPC/IB matrix, diagonal false. Counts exclude packet
// headers, protocol implementation traffic and GPU cache-line amplification.
DirectTraffic directExchangeTraffic(Workload const& workload, size_t ranks,
                                    size_t chunkBytes,
                                    std::vector<bool> const& ipc);
struct DirectPathResource {
  std::string resource;
  bool physicalCapacityBound = false;
};
struct DirectPathModel {
  bool cudaIpc = false;
  // Operational path plus any shared PCIe/NUMA/NIC bounds. GPU memory is
  // counted separately by the generator, never supplied again in this list.
  std::vector<DirectPathResource> resources;
};
struct DirectModelOverheads {
  // Additional costs beyond the operational-resource latencies. These must
  // have a compatible measurement scope; absence must not become zero.
  double entryUs = -1, copyUs = -1, streamSyncUs = -1, transferFlushUs = -1,
         visibilityUs = -1, reductionUs = -1, publicationUs = -1,
         retirementUs = -1;
};
struct DirectOperationResources {
  std::vector<std::string> inputCopy, selfCopy, gather, reduction;
};
struct DirectModelObservation {
  std::string resource, scope, rejection;
  size_t bytes = 0;
  size_t trafficFactor = 1;
  std::vector<double> microseconds;
};
struct DirectModelCalibration {
  Topology topology;
  std::vector<DirectPathModel> paths;
  DirectOperationResources operations;
  DirectModelOverheads overheads;
  size_t chunkBytes = 0;
  ScalarType dtype = ScalarType::Float32;
  std::string identity;
  std::vector<std::string> configurationIdentities;
  std::vector<DirectModelObservation> observations;
  std::vector<std::string> unmodeledResources;
};
// Operation-specific host-completion services include dispatch/synchronization;
// GPU memory remains a shared effective working-set service envelope. Neither
// is claimed to be a physical peak or a transferable cold-memory measurement.
PlanEstimate estimateDirectModel(DirectModelCalibration const& calibration,
                                 Workload const& workload);

// Bulk-synchronous advisory graph. Per-rank submission is serial; transfers
// on independent ranks overlap and shared directed resources contend. Global
// publication/retirement boundaries conservatively drain each phase, rather
// than pretending to model individual TCP progress or SM arbitration.
// A composed topology uses unique logical device tokens and globally scoped
// hardware/resource IDs. No native ordinal is inferred across hosts.
PlanCandidate directExchangeModel(
    Topology const& topology, Workload const& workload, size_t chunkBytes,
    std::vector<DirectPathModel> const& paths,
    DirectModelOverheads const& overheads = {},
    DirectOperationResources const* operations = nullptr);
}  // namespace uccl_lite

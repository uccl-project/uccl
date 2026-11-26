#pragma once

#include "collective_types.hpp"
#include "topology.hpp"
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace uccl_lite {

enum class Algorithm { Direct, Coordinator, Ring, Tree, Hierarchical };
enum class Transport { Peer, Host, Network };
enum class Protocol { LoadStore, BulkCopy, Packet };

// Capability admission is independent of performance. Registry entries belong
// to implementations, not GPU model names or a measured machine's rank count.
struct Capability {
  std::vector<Collective> collectives;
  std::vector<ScalarType> dtypes;
  size_t minRanks = 1, maxRanks = 1;
  size_t minBytes = 1, maxBytes = std::numeric_limits<size_t>::max();
  size_t extentMultiple = 1;
  unsigned alignment = 1;
  bool inPlace = false, capture = false;
  bool allPairsPeer = false, mappedHost = false;
};

struct ResourceDemand {
  // Link resource ID, memory:<hardware node ID>, or nic:<name>:<port>.
  std::string resource;
  size_t bytes = 0;
  // An advertised physical capacity is only a bandwidth bound; end-to-end
  // measured demands carry latency. Bounds alone cannot produce an estimate.
  bool physicalCapacityBound = false;
};
struct PlanStage {
  // Demands run concurrently. Demands sharing a resource are summed; distinct
  // resources may overlap. Stages run serially. A pipeline must explicitly
  // describe its prologue, steady-state groups, and drain in this
  // representation.
  std::vector<ResourceDemand> demands;
  double overheadUs = -1;  // Unknown until supplied by the execution model.
  size_t repetitions = 1;
};
struct PlanTask {
  std::string id;
  std::vector<ResourceDemand> demands;
  // Indices into ExecutionPlan::tasks; arbitrary acyclic order is accepted.
  // An empty-demand task is an explicit dependency/known-delay node.
  std::vector<size_t> dependencies;
  double overheadUs = -1;
};
struct ExecutionPlan {
  // Stable implementation + parameter identity, also part of calibration keys.
  std::string id;
  Algorithm algorithm = Algorithm::Direct;
  Transport transport = Transport::Peer;
  Protocol protocol = Protocol::LoadStore;
  std::vector<int> rankOrder;
  std::vector<std::vector<int>> groups;
  // Required CUDA accessor -> peer edges. Sparse executors declare exactly
  // the edges they use; capability validation does not infer a full mesh.
  std::vector<std::pair<int, int>> peerEdges;
  unsigned channels = 1;
  size_t chunkBytes = 0;
  std::vector<PlanStage> stages;
  // Alternative to serial stages, not an additional charge. Resource-only
  // fluid model: ready tasks start independently, shared bandwidth is allocated
  // with max-min fairness in fractions of isolated progress. Startup latency
  // is max operational resource latency + overhead and holds no bandwidth.
  // This is an advisory model, not a claim about hardware arbitration or SMs.
  std::vector<PlanTask> tasks;
};
struct PlanCandidate {
  // Resource demands are instantiated for this exact call and topology. An
  // algorithm registration must generate a fresh candidate when either changes.
  Workload workload;
  std::string topologyIdentity;
  Capability capability;
  ExecutionPlan plan;
};
struct PlanEstimate {
  bool eligible = false, estimated = false;
  double microseconds = std::numeric_limits<double>::infinity();
  std::string reason;
  struct TaskTiming {
    double startUs = 0, finishUs = 0;
  };
  std::vector<TaskTiming> taskTimings;
  double serialUs = -1, dependencyOnlyUs = -1, resourceLoadUs = -1;
};
struct PlanChoice {
  std::optional<size_t> index;
  std::vector<PlanEstimate> candidates;
};

// CPU-only and side-effect-free. This does not qualify an experimental router
// for automatic admission; exact workload/timing evidence is checked by the
// runtime before an executor is enabled. Missing/advertised resource data
// cannot supply operational predictions unless the caller explicitly requests
// advertised modeling (latency must still be supplied). Physical-capacity-only
// demands can additionally constrain a measured model without claiming latency;
// those bounds cannot produce a prediction by themselves.
PlanChoice choosePlan(Topology const& topology, Workload const& workload,
                      bool capture,
                      std::vector<PlanCandidate> const& candidates,
                      bool allowAdvertised = false);

struct ResourceCalibration {
  std::string resource;
  Performance performance;
  int repetitions = 0;
};
struct HardwareCalibration {
  std::string topologyIdentity;
  // Includes probe/build identity, workload, placement and timing scope.
  std::string identity;
  std::vector<ResourceCalibration> resources;
};
// Apply a validated local probe result atomically. Stale or malformed records
// leave topology untouched. Calibration is separate from collective timings.
void applyHardwareCalibration(Topology& topology,
                              HardwareCalibration const& calibration);

}  // namespace uccl_lite

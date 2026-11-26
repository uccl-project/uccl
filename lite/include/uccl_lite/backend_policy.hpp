#pragma once

#include "collective_types.hpp"
#include "topology.hpp"
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace uccl_lite {

enum class RouterMode { Auto, Off, Force };

struct TimingRange {
  double minimum = 0, median = 0, maximum = 0;
};

struct Measurement {
  std::string topologyIdentity, fallbackIdentity;
  Workload workload;
  TimingRange router, fallback;
  int repetitions = 0;
  bool correct = false;
};

struct BackendDecision {
  bool cpuRouter = false;
  std::string reason;
  double measuredSpeedup = 0;
  double routerMicroseconds = 0, fallbackMicroseconds = 0;
};

// CPU-only selection logic. Never benchmarks, allocates GPU memory, or starts
// workers.
class BackendPolicy {
 public:
  RouterMode mode = RouterMode::Auto;
  double minimumSpeedup = 1.03;
  std::string fallbackIdentity;
  std::vector<Measurement> measurements;

  BackendDecision select(Topology const& topology,
                         Workload const& workload) const;
};

char const* collectiveName(Collective collective);
Collective parseCollective(std::string const& name);
RouterMode parseRouterMode(std::string const& name);

// Archive reader for offline comparisons and integrity checks. Historical
// records never enable automatic routing on a live host.
std::vector<Measurement> loadL4ReferenceProfile(std::string const& path);
std::vector<Measurement> loadRouterProfile(std::string const& path);
void saveRouterProfile(std::string const& path,
                       std::vector<Measurement> const& measurements);

}  // namespace uccl_lite

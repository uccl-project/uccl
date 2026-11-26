#pragma once
#include "collective_types.hpp"
#include "network_calibration.hpp"

namespace uccl_lite {
struct NetworkCachedLayout {
  bool inPlace = false;
  unsigned alignment = 4;
};
enum class NetworkCacheCandidate { Hierarchy, DirectExchange };
struct NetworkPlanRecord {
  NetworkCacheCandidate candidate = NetworkCacheCandidate::Hierarchy;
  Workload workload;
  std::vector<NetworkCachedLayout> layouts;
  bool candidateCorrect = false, useHierarchy = false, useDirect = false;
  std::vector<double> candidateSamples, baselineSamples;
  std::vector<int> rankOrder, rails;
  std::vector<std::string> configurationIdentities;
};
struct NetworkPlanCache {
  // Exact collective context: participating hardware/transport identities,
  // implementation, native baseline/configuration and capacity. Eager only.
  std::string identity;
  std::optional<NetworkProfile> calibration;
  std::vector<NetworkPlanRecord> records;
};
bool sameNetworkRequest(NetworkPlanRecord const& a, NetworkPlanRecord const& b);
std::string encodeNetworkPlanCache(NetworkPlanCache const& cache);
NetworkPlanCache decodeNetworkPlanCache(std::string const& data);
std::optional<NetworkPlanCache> loadNetworkPlanCache(std::string const& path);
// Exclusive update lock + atomic replacement; merge only identical contexts.
// Malformed caches reject without replacement. Missing caches start empty.
void updateNetworkPlanCache(std::string const& path,
                            NetworkPlanCache const& context,
                            NetworkPlanRecord const& record);
void eraseNetworkPlanCacheRecord(std::string const& path,
                                 std::string const& identity,
                                 NetworkPlanRecord const& record);
}  // namespace uccl_lite

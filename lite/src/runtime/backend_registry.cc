#include "uccl_lite/runtime/collective_backend.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace uccl_lite::runtime {
bool BackendSpec::acceptsLayout(std::vector<int> const& layout) const {
  if (layout.empty() || layout.size() > 32 ||
      std::any_of(layout.begin(), layout.end(),
                  [](int n) { return n <= 0 || n > 32; }))
    return false;
  if (maxHosts) {
    if (minHosts < 1 || maxHosts < minHosts || maxHosts > 32 ||
        minTotalRanks < 1 || maxTotalRanks < minTotalRanks ||
        maxTotalRanks > 32 || minLocalRanks < 1 ||
        maxLocalRanks < minLocalRanks || maxLocalRanks > 32 ||
        layout.size() < size_t(minHosts) || layout.size() > size_t(maxHosts))
      return false;
    int total = std::accumulate(layout.begin(), layout.end(), 0);
    return total >= minTotalRanks && total <= maxTotalRanks &&
           std::all_of(layout.begin(), layout.end(), [&](int n) {
             return n >= minLocalRanks && n <= maxLocalRanks;
           });
  }
  if (layout == ranksPerHost) return true;
  return layout.size() == 1 && minLocalRanks > 0 &&
         layout.front() >= minLocalRanks && layout.front() <= maxLocalRanks;
}
int BackendSpec::ranks() const {
  return std::accumulate(ranksPerHost.begin(), ranksPerHost.end(), 0);
}
int BackendSpec::elementBytes() const {
  switch (dtype) {
    case ScalarType::Float32:
      return 4;
    case ScalarType::BFloat16:
      return 2;
    default:
      throw std::invalid_argument("unsupported runtime backend dtype");
  }
}
int BackendSpec::hostForRank(int rank) const {
  if (rank < 0) return -1;
  for (size_t host = 0; host < ranksPerHost.size(); ++host) {
    if (rank < ranksPerHost[host]) return int(host);
    rank -= ranksPerHost[host];
  }
  return -1;
}
int BackendSpec::localRank(int rank) const {
  int host = hostForRank(rank);
  if (host < 0) return -1;
  for (int i = 0; i < host; ++i) rank -= ranksPerHost[i];
  return rank;
}
std::vector<BackendRegistration> const& availableBackends() {
  static std::vector<BackendRegistration> const backends = [] {
    std::vector<BackendRegistration> result;
#ifdef UCCL_LITE_REGISTER_PEER_PACKET
    result.push_back(peerPacketBackend());
#endif
#ifdef UCCL_LITE_REGISTER_HOST_PACKET
    result.push_back(hostPacketBackend());
#endif
#ifdef UCCL_LITE_REGISTER_PEER_STAGED
    result.push_back(peerStagedBackend());
    result.push_back(peerStagedBackend(ScalarType::Float32));
#endif
#ifdef UCCL_LITE_REGISTER_PREPARED_LOCAL
    result.push_back(preparedLocalBackend());
    result.push_back(preparedLocalBackend(ScalarType::Float32));
#endif
#ifdef UCCL_LITE_REGISTER_NETWORK_HIERARCHY
    result.push_back(networkHierarchyBackend());
#endif
#ifdef UCCL_LITE_REGISTER_PREPARED_DISTRIBUTED
    result.push_back(preparedDistributedBackend(ScalarType::Float32));
    result.push_back(preparedDistributedBackend(ScalarType::BFloat16));
#endif
#ifdef UCCL_LITE_REGISTER_DIRECT_EXCHANGE
    result.push_back(directExchangeBackend(ScalarType::Float32));
    result.push_back(directExchangeBackend(ScalarType::BFloat16));
#endif
    return result;
  }();
  return backends;
}
}  // namespace uccl_lite::runtime

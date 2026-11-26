#include "peer_fixture.hpp"
#include <iostream>
#include <stdexcept>
using namespace uccl_lite;
int main() {
  try {
    Topology t;
    t.devices = {3, 0, 2, 1};
    t.identity = "test";
    t.peerAccess.resize(16, true);
    auto tuning = syntheticPeerModel(t);
    Workload small{Collective::AllReduce, ScalarType::BFloat16, 1024, false,
                   16};
    auto large = small;
    large.bytes = 64UL << 20;
    auto a = selectPeerPlan(t, small, tuning),
         b = selectPeerPlan(t, large, tuning);
    if (!a.modeled || !b.modeled || a.family != PeerFamily::Coordinator ||
        b.family != PeerFamily::Bulk)
      throw std::runtime_error(
          "measured resource model did not choose small/large families");
    Topology pair;
    pair.devices = {1, 0};
    pair.identity = "pair";
    pair.peerAccess.resize(4, true);
    auto pairTuning = syntheticPeerModel(pair);
    // On this symmetric two-rank fixture coordinator and RS+AG move the
    // same PCIe volume; staging adds phases without reducing the bottleneck.
    if (selectPeerPlan(pair, large, pairTuning).family !=
        PeerFamily::Coordinator)
      throw std::runtime_error(
          "two-rank model incorrectly inherited four-rank crossover");
    for (int ranks = 1; ranks <= kPeerRankCapacity; ++ranks) {
      Topology variable;
      variable.identity = "variable-ranks";
      for (int i = 0; i < ranks; ++i) variable.devices.push_back(i);
      variable.peerAccess.resize(ranks * ranks, true);
      auto model = syntheticPeerModel(variable);
      auto workload = small;
      workload.bytes = size_t(16 * ranks) * 64;
      auto choice = selectPeerPlan(variable, workload, model);
      if (!choice.modeled ||
          (ranks == 1 && choice.family != PeerFamily::Direct))
        throw std::runtime_error(
            "valid rank extent was not admitted by capabilities");
      if (ranks == 3) {
        model.validationFamily = PeerFamily::Bulk;
        workload.bytes = 1024;
        bool invalid = false;
        try {
          selectPeerPlan(variable, workload, model);
        } catch (std::invalid_argument const&) {
          invalid = true;
        }
        if (!invalid)
          throw std::runtime_error("non-divisible three-rank AR was staged");
      }
    }
    tuning.overheadUs = {{-1, -1, -1}};
    if (selectPeerPlan(t, large, tuning).family != PeerFamily::Direct)
      throw std::runtime_error("unknown overhead must retain direct baseline");
    tuning.validationFamily = PeerFamily::Bulk;
    auto odd = small;
    odd.bytes = 514;
    bool rejected = false;
    try {
      selectPeerPlan(t, odd, tuning);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("bulk AR accepted indivisible shards");
    tuning.validationFamily.reset();
    if (selectPeerPlan(t, odd, tuning).family != PeerFamily::Direct)
      throw std::runtime_error("direct baseline must support odd AR count");
    t.peerAccess[1] = false;
    rejected = false;
    try {
      selectPeerPlan(t, small, tuning);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("unavailable physical peers accepted");
    std::cout << "Peer planner: resource-driven family selection, unknown "
                 "overhead, odd extents and peer capability passed\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

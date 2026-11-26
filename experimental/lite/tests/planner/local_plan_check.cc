#include "peer_fixture.hpp"
#include "uccl_lite/local_plan.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace uccl_lite;
void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  try {
    Topology t;
    t.identity = "six-rank-synthetic";
    t.devices = {5, 2, 0, 4, 1, 3};
    t.peerAccess.assign(36, true);
    syntheticPeerModel(t);
    Workload w{Collective::AllReduce, ScalarType::BFloat16, 1000, true, 2};
    auto estimate = [&](Topology const& topology) {
      return choosePlan(topology, w, false, {stagedPeerCandidate(topology, w)})
          .candidates.at(0);
    };
    auto fast = estimate(t);
    require(fast.eligible && fast.estimated &&
                std::abs(fast.microseconds - 0.52) < 1e-10,
            "six-rank staging/shared-wire traffic estimate");
    for (size_t r = 0; r < 6; ++r) t.hardware[r].memory.bandwidthGBs = 1;
    auto slow = estimate(t);
    require(slow.estimated && std::abs(slow.microseconds - 9) < 1e-10,
            "GPU memory bandwidth must constrain staging and reduction reads");
    t.peerAccess[1] = false;
    require(!estimate(t).eligible, "missing peer edge admitted");
    t.peerAccess[1] = true;
    t.hardware[0].memory = {};
    require(estimate(t).eligible && !estimate(t).estimated,
            "unknown memory rate fabricated");
    require(qualifiedLocalBackend({100, 101, 102}, {90, 91, 92}, true) ==
                LocalBackend::Host,
            "robust gain rejected");
    require(qualifiedLocalBackend({100, 101, 102}, {90, 91, 101}, true) ==
                LocalBackend::Peer,
            "overlap admitted");
    require(qualifiedLocalBackend({100, 101, 102}, {90, 91, 92}, false) ==
                LocalBackend::Peer,
            "incorrect host admitted");
    require(qualifiedLocalBackend({100, 101, 102}, {NAN, 1, 1}, true) ==
                LocalBackend::Peer,
            "invalid host sample admitted");
    require(qualifiedLocalBackend({100, 100, 100}, {98, 98, 98}, true) ==
                LocalBackend::Peer,
            "small gain admitted");
    bool rejected = false;
    try {
      qualifiedLocalBackend({0, 1, 2}, {}, true);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(rejected, "invalid baseline accepted");
    std::array<std::vector<double>, 2> baseline{
        {{100, 101, 102}, {70, 71, 72}}};
    std::array<std::vector<double>, 2> alternatives{
        {{90, 91, 92}, {80, 81, 82}}};
    require(qualifiedFallbackAlternative(baseline, alternatives,
                                         {true, true}) == LocalBackend::Peer,
            "alternatives were compared with an unrelated fallback block");
    require(
        qualifiedFallbackAlternative(baseline, alternatives, {false, true}) ==
            LocalBackend::Fallback,
        "incorrect candidate replaced caller fallback");
    baseline[1] = {100, 101, 102};
    require(qualifiedFallbackAlternative(baseline, alternatives,
                                         {true, true}) == LocalBackend::Host,
            "fastest independently qualified alternative not selected");
    require(
        qualifiedFallbackAlternative(std::array<std::vector<double>, 2>{},
                                     std::array<std::vector<double>, 2>{},
                                     {false, false}) == LocalBackend::Fallback,
        "unmeasured alternatives replaced fallback");
    Topology packet;
    packet.identity = "packet-layout";
    packet.devices = {2, 3, 0, 1};
    packet.numaNodes = {1, 1, 0, 0};
    packet.mappedHost.assign(4, true);
    packet.peerAccess.assign(16, true);
    Workload fp32{Collective::AllReduce, ScalarType::Float32, 1024, true, 4};
    require(peerPacketEligible(packet, fp32),
            "valid retained peer layout rejected");
    auto interleaved = packet;
    interleaved.numaNodes = {1, 0, 1, 0};
    require(
        peerPacketEligible(interleaved, fp32) &&
            pairedNumaRankOrder(interleaved) == std::vector<int>({1, 3, 0, 2}),
        "interleaved NUMA placement did not generate execution mapping");
    auto sparseNodes = interleaved;
    sparseNodes.numaNodes = {9, 3, 9, 3};
    require(pairedNumaRankOrder(sparseNodes) == std::vector<int>({1, 3, 0, 2}),
            "physical NUMA IDs were confused with execution groups");
    auto unbalanced = packet;
    unbalanced.numaNodes = {0, 0, 0, 1};
    require(!peerPacketEligible(unbalanced, fp32),
            "unbalanced NUMA groups admitted");
    unbalanced.numaNodes = {0, 0, 0, 0};
    require(!peerPacketEligible(unbalanced, fp32), "one NUMA group admitted");
    unbalanced.numaNodes = {0, 0, 1, -1};
    require(!peerPacketEligible(unbalanced, fp32),
            "unknown NUMA node admitted");
    interleaved.peerAccess[1] = false;
    require(!peerPacketEligible(interleaved, fp32),
            "mapping bypassed peer capability");
    auto tail = fp32;
    tail.bytes = 1028;
    require(!peerPacketEligible(packet, tail),
            "specialized peer accepted unsupported extent");
    auto bf16 = fp32;
    bf16.dtype = ScalarType::BFloat16;
    require(!peerPacketEligible(packet, bf16),
            "specialized peer admitted wrong dtype");
    auto unaligned = fp32;
    unaligned.alignment = 2;
    require(!choosePlan(packet, unaligned, false,
                        {stagedPeerCandidate(packet, unaligned)})
                 .candidates[0]
                 .eligible,
            "FP32 accepted 2-byte alignment");
    require(choosePlan(packet, fp32, false, {stagedPeerCandidate(packet, fp32)})
                .candidates[0]
                .eligible,
            "generic FP32 peer rejected");
    std::cout << "PASS: typed peer capability, NUMA specialization and paired "
                 "fallback admission\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

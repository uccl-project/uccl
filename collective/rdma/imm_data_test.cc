#include "transport.h"
#include <cstdint>
#include <iostream>

using namespace uccl;

int main() {
  constexpr uint32_t kTargetMask = 0xFFu << IMMDataEQDS::kPULL_TARGET;
  uint32_t failures = 0;
  auto check = [&](uint32_t before, uint32_t target) {
    // Retransmission reads a previously encoded network-order immediate,
    // updates only the pull target, and writes it back to the work request.
    uint32_t wire = htonl(before);
    IMMDataEQDS imm(ntohl(wire));
    imm.SetTarget(target);
    wire = htonl(imm.GetImmData());
    IMMDataEQDS received(ntohl(wire));
    IMMDataEQDS original(before);
    if (received.GetTarget() != (target & 0xFFu) ||
        (received.GetImmData() & ~kTargetMask) != (before & ~kTargetMask) ||
        received.GetFID() != original.GetFID() ||
        received.GetRID() != original.GetRID() ||
        received.GetCSN() != original.GetCSN() ||
        received.GetHINT() != original.GetHINT()) {
      if (++failures <= 4) {
        std::cerr << "target update failed: old=" << original.GetTarget()
                  << " new=" << target << " received=" << received.GetTarget()
                  << '\n';
      }
    }
  };

  for (uint32_t old_target = 0; old_target < 256; ++old_target) {
    for (uint32_t new_target = 0; new_target < 256; ++new_target) {
      // Vary every other field as well as the pull target.
      uint32_t payload =
          ((old_target * 2654435761u) ^ (new_target * 2246822519u)) &
          ~kTargetMask;
      check(payload | (old_target << IMMDataEQDS::kPULL_TARGET), new_target);
    }
  }
  // Zero clears a previous target; high input bits must be ignored.
  check(UINT32_MAX, 0);
  check(UINT32_MAX, 0x100);
  check(0, UINT32_MAX);
  if (failures) {
    std::cerr << failures << " immediate-data checks failed\n";
    return 1;
  }
  std::cout << "PASS: 65,536 target transitions and 3 boundary cases; "
               "other immediate fields preserved\n";
  return 0;
}

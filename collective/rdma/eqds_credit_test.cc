#include "transport.h"
#include <cstdint>
#include <iostream>
#include <random>

using namespace uccl;
using namespace uccl::eqds;

int main() {
  uint32_t failures = 0;
  uint32_t cases = 0;
  auto check = [&](bool condition, char const* message) {
    if (!condition && ++failures <= 8) std::cerr << message << '\n';
  };
  auto spend = [&](uint32_t pull, uint32_t spec, bool speculating,
                   uint32_t bytes) {
    ++cases;
    EQDSCC cc;
    cc.credit_pull_ = pull;
    cc.credit_spec_ = spec;
    cc.in_speculating_ = speculating;
    uint32_t available = pull + (speculating ? spec : 0);
    check(cc.credit() == available, "Reported credit includes unusable bytes");
    bool accepted = cc.spend_credit(bytes);
    check(accepted == (bytes <= available), "Incorrect admission decision");
    if (accepted) {
      check(bytes <= available && cc.credit() == available - bytes,
            "Successful spend did not charge the full chunk");
      check(cc.credit_pull_ <= pull && cc.credit_spec_ <= spec,
            "A credit counter increased or underflowed");
      check(
          static_cast<uint64_t>(pull) + spec ==
              static_cast<uint64_t>(cc.credit_pull_) + cc.credit_spec_ + bytes,
          "Total credit is not conserved");
      if (!speculating)
        check(cc.credit_spec_ == spec, "Spent disabled speculative credit");
    } else {
      check(cc.credit_pull_ == pull && cc.credit_spec_ == spec,
            "Rejected spend changed credit counters");
    }
  };

  uint32_t const edges[] = {0,
                            1,
                            PULL_QUANTUM - 1,
                            PULL_QUANTUM,
                            PULL_QUANTUM + 1,
                            65536,
                            EQDSCC::kEQDSMaxCwnd};
  for (bool speculating : {false, true}) {
    for (auto pull : edges) {
      for (auto spec : edges) {
        for (auto bytes : edges) spend(pull, spec, speculating, bytes);
        spend(pull, spec, speculating, UINT32_MAX);
      }
    }
  }
  std::mt19937 random(1092);
  for (int i = 0; i < 10000; ++i) {
    uint32_t pull = random() % (EQDSCC::kEQDSMaxCwnd + 1);
    uint32_t spec = random() % (EQDSCC::kEQDSMaxCwnd + 1);
    bool speculating = random() & 1;
    uint32_t available = pull + (speculating ? spec : 0);
    spend(pull, spec, speculating, random() % (available + 2));
    spend(pull, spec, speculating, available);
    spend(pull, spec, speculating, available + 1);
  }

  // Leaving speculation does not itself add pull credit. The next credit
  // packet must resume sending; duplicate grants must not add more credit.
  EQDSCC cc;
  cc.stop_speculating();
  check(cc.credit() == 0, "Stopping speculation left unusable credit visible");
  check(!cc.spend_credit(1), "Sent a byte without usable credit");
  for (int i = 0; i < 512; ++i) {
    PullQuanta next = cc.pull_ + 1;
    check(cc.handle_pull(next), "New credit grant was ignored");
    check(cc.credit() == PULL_QUANTUM, "Incorrect granted byte count");
    check(!cc.handle_pull(next), "Duplicate grant was accepted");
    check(cc.spend_credit(PULL_QUANTUM), "Could not spend a complete grant");
    check(cc.credit() == 0, "Grant was not fully charged");
    check(!cc.spend_credit(1), "Overspent a grant");
  }

  // The production pull-target computation must subtract only usable credit.
  // Repeating it without a state change must not invent additional demand.
  SubUcclFlow flow(0, 50e9);
  auto& target_cc = flow.pcb.eqds_cc;
  flow.backlog_bytes_ = 4 * PULL_QUANTUM;
  check(target_cc.compute_pull_target(&flow, 4096) == EQDSCC::INIT_PULL_QUANTA,
        "Speculative credit did not cover initial backlog");
  target_cc.stop_speculating();
  for (int i = 0; i < 10; ++i) {
    check(target_cc.compute_pull_target(&flow, 4096) ==
              EQDSCC::INIT_PULL_QUANTA + 4,
          "Pull target subtracted disabled speculative credit");
  }
  target_cc.credit_pull_ = PULL_QUANTUM + 1;
  check(target_cc.compute_pull_target(&flow, 4096) ==
            EQDSCC::INIT_PULL_QUANTA + 3,
        "Pull target did not account for usable pull credit");

  if (failures) {
    std::cerr << failures << " EQDS credit checks failed\n";
    return 1;
  }
  std::cout << "PASS: " << cases
            << " spend cases, 512 grant/drain cycles, "
               "and production pull-target transitions\n";
  return 0;
}

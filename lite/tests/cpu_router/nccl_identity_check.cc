#include "../../benchmarks/cpu_router/nccl_identity.hpp"
#include <iostream>
#include <stdexcept>

int main() {
  using uccl_lite::benchmark::ncclFallbackIdentity;
  int checks = 0;
  auto require = [&](bool ok) {
    ++checks;
    if (!ok) throw std::runtime_error("NCCL configuration identity mismatch");
  };
  require(ncclFallbackIdentity(23007, {}) == "nccl:23007");
  auto disabled = ncclFallbackIdentity(23007, {"env:NCCL_P2P_DISABLE=1"});
  require(disabled != ncclFallbackIdentity(23007, {}));
  require(disabled != ncclFallbackIdentity(23007, {"env:NCCL_P2P_DISABLE=0"}));
  require(disabled != ncclFallbackIdentity(22807, {"env:NCCL_P2P_DISABLE=1"}));
  require(ncclFallbackIdentity(23007, {"a", "b"}) ==
          ncclFallbackIdentity(23007, {"b", "a"}));
  require(ncclFallbackIdentity(23007, {"ab", "c"}) !=
          ncclFallbackIdentity(23007, {"a", "bc"}));
  require(disabled != ncclFallbackIdentity(23007, {"env:NCCL_P2P_DISABLE=1",
                                                   "user-file:NCCL_PROTO=LL"}));
  require(ncclFallbackIdentity(23007, {"user-file:A"}) !=
          ncclFallbackIdentity(23007, {"user-file:B"}));
  std::cout << "NCCL identity: " << checks
            << " checks passed; transport/config changes invalidate evidence\n";
}

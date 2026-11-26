#include "uccl_lite/network_calibration.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace uccl_lite;
void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class F>
void reject(F action) {
  bool bad = false;
  try {
    action();
  } catch (std::exception const&) {
    bad = true;
  }
  require(bad, "invalid group evidence accepted");
}
int main() {
  std::string file = "build/network-group-check-" + std::to_string(getpid());
  try {
    NetworkProfile p;
    p.hosts = {{{"h0", "t0", "s0"}, {"h1", "t1", "s1"}}};
    p.probeIdentity = "probe";
    p.groupControlIdentities = {"mpi0", "mpi1"};
    NetworkGroupObservation group;
    group.label = "complete";
    group.iterations = 1;
    for (int peer = 0; peer < 2; ++peer)
      for (int side = 0; side < 2; ++side) {
        NetworkPathObservation path;
        path.iterations = 1;
        path.key = {side,
                    1 - side,
                    "gpu" + std::to_string(side * 2 + peer),
                    "gpu" + std::to_string((1 - side) * 2 + peer),
                    "nic",
                    "nic",
                    1,
                    1};
        path.sizes = {{16, {2, 2, 2}}, {1024, {20, 20, 20}}};
        p.paths.push_back(path);
        group.flows.push_back(path.key);
      }
    for (size_t bytes : {16UL, 1024UL}) {
      NetworkGroupObservation::Size size;
      size.bytes = bytes;
      for (int sample = 0; sample < 3; ++sample) {
        size.senderMicroseconds.push_back(std::vector<double>(4, 5));
        size.cohortMicroseconds.push_back(std::vector<double>(4, 10));
        size.cohortMicroseconds.back()[sample] = 100;
      }
      group.sizes.push_back(size);
    }
    p.groups.push_back(group);
    auto result = estimateNetworkGroup(p, p.hosts, group.flows, 1024);
    require(result && result->microseconds == 100 &&
                result->payloadGBs == double(4096) / 100 / 1000,
            "group did not aggregate each call's slowest rank");
    auto reordered = group.flows;
    std::reverse(reordered.begin(), reordered.end());
    require(bool(estimateNetworkGroup(p, p.hosts, reordered, 1024)),
            "flow order changed group identity");
    reordered.pop_back();
    require(!estimateNetworkGroup(p, p.hosts, reordered, 1024),
            "subset reused group performance");
    reordered = group.flows;
    reordered[0].sourceNic = "other";
    require(!estimateNetworkGroup(p, p.hosts, reordered, 1024),
            "different sharing reused group performance");
    require(!estimateNetworkGroup(p, p.hosts, group.flows, 512),
            "group extent interpolated");
    auto hosts = p.hosts;
    hosts[1].transportIdentity = "changed";
    require(!estimateNetworkGroup(p, hosts, group.flows, 1024),
            "stale remote identity matched group");
    saveNetworkProfile(file, p);
    auto loaded = loadNetworkProfile(file);
    require(loaded.groups[0].sizes[1].senderMicroseconds ==
                    group.sizes[1].senderMicroseconds &&
                loaded.groups[0].sizes[1].cohortMicroseconds ==
                    group.sizes[1].cohortMicroseconds &&
                loaded.groupControlIdentities == p.groupControlIdentities,
            "group raw/provenance roundtrip changed");
    auto bad = p;
    auto bootstrapProfile = p;
    bootstrapProfile.groupScope = kNetworkBootstrapGroupProbeScope;
    bootstrapProfile.groupControlIdentities = {"bootstrap0", "bootstrap1"};
    saveNetworkProfile(file, bootstrapProfile);
    auto bootstrapLoaded = loadNetworkProfile(file);
    require(bootstrapLoaded.groupScope == kNetworkBootstrapGroupProbeScope &&
                bootstrapLoaded.groupControlIdentities ==
                    bootstrapProfile.groupControlIdentities &&
                estimateNetworkGroup(bootstrapLoaded, bootstrapLoaded.hosts,
                                     group.flows, 1024)
                        ->microseconds == 100,
            "bootstrap cohort scope/provenance roundtrip changed");
    bad.groupScope = "unknown-control-scope";
    reject([&] { saveNetworkProfile(file, bad); });
    bad = p;
    bad.groups.push_back(group);
    reject([&] { saveNetworkProfile(file, bad); });
    bad = p;
    bad.groups[0].sizes[0].cohortMicroseconds[0][0] = 1;
    reject([&] { saveNetworkProfile(file, bad); });
    bad = p;
    bad.groups[0].sizes[0].senderMicroseconds[0].pop_back();
    reject([&] { saveNetworkProfile(file, bad); });
    bad = p;
    bad.groupControlIdentities[0].clear();
    reject([&] { saveNetworkProfile(file, bad); });
    auto failed = p;
    failed.groups[0].rejection = "later registration failed";
    failed.groups[0].sizes[0].senderMicroseconds.pop_back();
    failed.groups[0].sizes[0].cohortMicroseconds.pop_back();
    saveNetworkProfile(file, failed);
    loaded = loadNetworkProfile(file);
    require(loaded.groups[0].sizes[0].senderMicroseconds.size() == 2 &&
                !estimateNetworkGroup(loaded, loaded.hosts, group.flows, 1024),
            "partial negative evidence lost or admitted");
    // Reconstruct a legacy path-only v1 record and prove it does not invent
    // groups.
    p.groups.clear();
    p.groupControlIdentities = {};
    saveNetworkProfile(file, p);
    std::ifstream in(file);
    std::string text((std::istreambuf_iterator<char>(in)), {});
    in.close();
    text.replace(0, std::string("uccl-lite-network-profile-v2").size(),
                 "uccl-lite-network-profile-v1");
    text.resize(text.find(std::string("\"") + kNetworkGroupProbeScope));
    {
      std::ofstream out(file);
      out << text;
    }
    require(loadNetworkProfile(file).groups.empty(),
            "legacy isolated profile gained concurrent evidence");
    auto legacy = loadNetworkProfile(file);
    legacy.pathScope = kLegacyNetworkProbeScope;
    saveNetworkProfile(file, legacy);
    legacy = loadNetworkProfile(file);
    require(
        !estimateNetworkPath(legacy, legacy.hosts, legacy.paths[0].key, 1024),
        "legacy placement protocol reused as current evidence");
    std::remove(file.c_str());
    std::cout << "PASS: exact concurrent flow/extent matching, max-cohort "
                 "samples, v1 compatibility and partial negatives\n";
  } catch (std::exception const& error) {
    std::remove(file.c_str());
    std::cerr << error.what() << '\n';
    return 1;
  }
}

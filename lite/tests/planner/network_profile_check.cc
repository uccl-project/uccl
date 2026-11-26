#include "uccl_lite/network_calibration.hpp"
#include <cmath>
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
  bool rejected = false;
  try {
    action();
  } catch (std::exception const&) {
    rejected = true;
  }
  require(rejected, "malformed network evidence accepted");
}
int main() {
  std::string file = "build/network-profile-check-" + std::to_string(getpid());
  try {
    NetworkProfile p;
    p.hosts = {{{"a", "topology:a", "transport:a"},
                {"b", "topology:b", "transport:b"}}};
    p.probeIdentity = "probe-build";
    NetworkPathObservation path;
    path.key = {0, 1, "gpuA", "gpuB", "mlx5_1", "mlx5_0", 1, 1};
    path.iterations = 1;
    path.sizes = {
        {16, {9, 10, 11}}, {1024, {19, 20, 21}}, {65536, {99, 100, 101}}};
    p.paths.push_back(path);
    auto estimate = estimateNetworkPath(p, p.hosts, path.key, 1024);
    require(estimate && estimate->microseconds == 20 &&
                estimate->lowerBytes == 1024 && estimate->upperBytes == 1024,
            "exact sample median changed");
    estimate = estimateNetworkPath(p, p.hosts, path.key, 8192);
    require(estimate && std::abs(estimate->microseconds -
                                 (20 + 80.0 * 7168 / 64512)) < 1e-10,
            "interpolated path cost incorrect");
    require(!estimateNetworkPath(p, p.hosts, path.key, 8) &&
                !estimateNetworkPath(p, p.hosts, path.key, 65537),
            "extrapolated unmeasured extent");
    auto hosts = p.hosts;
    hosts[1].transportIdentity += "-new";
    require(!estimateNetworkPath(p, hosts, path.key, 1024),
            "stale remote software accepted");
    hosts = p.hosts;
    hosts[0].topologyIdentity += "-new";
    require(!estimateNetworkPath(p, hosts, path.key, 1024),
            "stale source topology accepted");
    auto key = path.key;
    key.sourceNic = "mlx5_0";
    require(!estimateNetworkPath(p, p.hosts, key, 1024),
            "different physical NIC collapsed");
    key = path.key;
    key.destinationGpu = "other-gpu";
    require(!estimateNetworkPath(p, p.hosts, key, 1024),
            "different remote GPU collapsed");
    key = path.key;
    key.sourceHost = 1;
    key.destinationHost = 0;
    require(!estimateNetworkPath(p, p.hosts, key, 1024),
            "direction inferred from reverse evidence");
    NetworkPathObservation failure;
    failure.key = key;
    failure.rejection = "registration unsupported";
    p.paths.push_back(failure);
    require(!estimateNetworkPath(p, p.hosts, key, 1024),
            "rejected path estimated");
    saveNetworkProfile(file, p);
    auto loaded = loadNetworkProfile(file);
    require(
        loaded.paths[0].sizes[2].microseconds == path.sizes[2].microseconds &&
            loaded.paths[1].rejection == failure.rejection,
        "network profile lost raw/negative evidence");
    auto bad = p;
    bad.paths.push_back(path);
    reject([&] { saveNetworkProfile(file, bad); });
    require(loadNetworkProfile(file).paths.size() == 2,
            "failed replacement damaged valid profile");
    bad = p;
    bad.paths[0].sizes[0].microseconds[0] = NAN;
    reject([&] { saveNetworkProfile(file, bad); });
    bad = p;
    bad.paths[0].iterations = 2;
    reject([&] { saveNetworkProfile(file, bad); });
    bad = p;
    bad.paths[0].sizes[2].microseconds = {1, 2, 3};
    require(!estimateNetworkPath(bad, bad.hosts, path.key, 8192),
            "nonmonotonic samples interpolated");
    {
      std::ofstream out(file, std::ios::app);
      out << "extra";
    }
    reject([&] { loadNetworkProfile(file); });
    {
      std::ofstream out(file);
      out << "uccl-lite-network-profile-v1\n\"different-scope\"\n\"id\"\n";
    }
    reject([&] { loadNetworkProfile(file); });
    std::remove(file.c_str());
    std::cout
        << "PASS: exact network endpoint/software/scope matching, bounded "
           "interpolation, atomic persistence and negative evidence\n";
  } catch (std::exception const& error) {
    std::remove(file.c_str());
    std::cerr << error.what() << '\n';
    return 1;
  }
}

// Preserve a real profile's observations while making explicit negative
// fixtures.
#include "uccl_lite/network_calibration.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::invalid_argument(
          "INPUT OUTPUT stale|missing|isolated|group-subset|group-data");
    auto profile = uccl_lite::loadNetworkProfile(argv[1]);
    std::string mode = argv[3];
    if (mode == "stale")
      profile.hosts[1].transportIdentity += ";test-stale-remote";
    else if (mode == "missing")
      profile.paths.erase(profile.paths.begin());
    else if (mode == "isolated") {
      profile.groups.clear();
      profile.groupControlIdentities = {};
    } else if (mode == "group-subset") {
      for (auto& group : profile.groups) {
        if (group.flows.size() < 3)
          throw std::invalid_argument(
              "subset fixture requires at least three flows");
        group.flows.pop_back();
        for (auto& size : group.sizes) {
          for (auto& row : size.senderMicroseconds) row.pop_back();
          for (auto& row : size.cohortMicroseconds) row.pop_back();
        }
      }
    } else if (mode == "group-data") {
      profile.groups.at(0).sizes.at(0).cohortMicroseconds.at(0).at(0) += 1;
    } else
      throw std::invalid_argument("invalid fixture mode");
    uccl_lite::saveNetworkProfile(argv[2], profile);
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

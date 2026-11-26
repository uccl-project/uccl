// Report stored measurements. This is not live endpoint validation/admission.
#include "uccl_lite/network_calibration.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace uccl_lite;
std::string csv(std::string const& text) {
  std::string result = "\"";
  for (char c : text) {
    result += c;
    if (c == '"') result += c;
  }
  return result + '"';
}
int main(int argc, char** argv) {
  try {
    bool groups = argc == 3 && std::string(argv[2]) == "--groups";
    if (argc != 2 && !groups)
      throw std::invalid_argument("network_profile_report PROFILE [--groups]");
    auto profile = loadNetworkProfile(argv[1]);
    std::cerr << "Stored GPU/NIC measurements only; no live validation or "
                 "best-variant envelope\n";
    std::cout << std::setprecision(17);
    if (groups) {
      std::cout << "label,flows,bytes_per_flow,samples,max_cohort_us,aggregate_"
                   "payload_GBs,rejection,flow_set,scope,probe_identity\n";
      for (auto const& group : profile.groups) {
        std::ostringstream flowSet;
        for (auto const& k : group.flows)
          flowSet << profile.hosts[k.sourceHost].hostname << '/' << k.sourceGpu
                  << '/' << k.sourceNic << ':' << k.sourcePort << "->"
                  << profile.hosts[k.destinationHost].hostname << '/'
                  << k.destinationGpu << '/' << k.destinationNic << ':'
                  << k.destinationPort << ';';
        auto row = [&](size_t bytes, size_t samples) {
          auto estimate =
              estimateNetworkGroup(profile, profile.hosts, group.flows, bytes);
          std::cout << csv(group.label) << ',' << group.flows.size() << ','
                    << bytes << ',' << samples << ',';
          if (estimate)
            std::cout << estimate->microseconds << ',' << estimate->payloadGBs;
          else
            std::cout << ',';
          std::cout << ',' << csv(group.rejection) << ',' << csv(flowSet.str())
                    << ',' << csv(profile.groupScope) << ','
                    << csv(profile.probeIdentity) << '\n';
        };
        if (group.sizes.empty()) row(0, 0);
        for (auto const& size : group.sizes)
          row(size.bytes, size.senderMicroseconds.size());
      }
      return 0;
    }
    std::cout
        << "source_host,source_gpu,source_nic,source_port,destination_host,"
           "destination_gpu,destination_nic,destination_port,bytes,samples,"
           "median_us,payload_GBs,rejection,scope,probe_identity\n";
    for (auto const& path : profile.paths) {
      auto const& k = path.key;
      auto prefix = [&] {
        std::cout << csv(profile.hosts[k.sourceHost].hostname) << ','
                  << csv(k.sourceGpu) << ',' << csv(k.sourceNic) << ','
                  << k.sourcePort << ','
                  << csv(profile.hosts[k.destinationHost].hostname) << ','
                  << csv(k.destinationGpu) << ',' << csv(k.destinationNic)
                  << ',' << k.destinationPort << ',';
      };
      if (!path.rejection.empty()) {
        prefix();
        std::cout << "0,0,0,0," << csv(path.rejection) << ','
                  << csv(profile.pathScope) << ',' << csv(profile.probeIdentity)
                  << '\n';
      }
      for (auto const& observation : path.sizes) {
        auto samples = observation.microseconds;
        std::sort(samples.begin(), samples.end());
        size_t n = samples.size();
        double us = n % 2 ? samples[n / 2]
                          : samples[n / 2 - 1] / 2 + samples[n / 2] / 2;
        prefix();
        std::cout << observation.bytes << ',' << samples.size() << ',' << us
                  << ',' << double(observation.bytes) / us / 1000 << ",\"\","
                  << csv(profile.pathScope) << ',' << csv(profile.probeIdentity)
                  << '\n';
      }
    }
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

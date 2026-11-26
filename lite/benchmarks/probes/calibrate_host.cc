#include "uccl_lite/hardware_calibration.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
  try {
    if (argc < 3)
      throw std::invalid_argument("usage: calibrate_host PROFILE GPU [GPU...]");
    std::vector<int> devices;
    for (int i = 2; i < argc; ++i) devices.push_back(std::stoi(argv[i]));
    auto topology = uccl_lite::inspectTopology(devices);
    auto profile = uccl_lite::probeHostResources(topology);
    uccl_lite::saveHardwareProfile(argv[1], profile);
    auto restored = uccl_lite::loadHardwareProfile(argv[1]);
    uccl_lite::applyHardwareCalibration(topology, restored.calibration);
    std::cout << "CPU memcpy, source/destination mbind verified; workers "
                 "pinned to destination NUMA; 32/64 MiB per worker; 2 warmups, "
                 "5 iterations, 3 repetitions; common host interval; no "
                 "baseline or winner envelope\n";
    std::cout << "All path rates count one-way copied payload bytes; local "
                 "read+write traffic is twice the payload. Worker CPU IDs are "
                 "part of each resource.\n";
    for (auto const& r : restored.calibration.resources)
      std::cout << r.resource << " fitted_GBs=" << r.performance.bandwidthGBs
                << " intercept_us=" << r.performance.latencyUs << '\n';
    std::cout << "Physical NUMA-link and generic memory-controller bandwidth "
                 "remain unknown.\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#include "uccl_lite/topology.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument("usage: topology DEVICE [DEVICE...]");
    std::vector<int> devices;
    for (int i = 1; i < argc; ++i) devices.push_back(std::stoi(argv[i]));
    auto t = uccl_lite::inspectTopology(devices);
    for (size_t rank = 0; rank < devices.size(); ++rank)
      std::cout << "rank=" << rank << " device=" << devices[rank]
                << " pci=" << t.gpuPciIds[rank] << " numa=" << t.numaNodes[rank]
                << '\n';
    for (auto const& link : t.links)
      std::cout << "resource=" << link.resource
                << " advertised_GBs=" << link.performance.bandwidthGBs << '\n';
    for (auto const& port : t.networkPorts)
      std::cout << "nic=" << port.name << " port=" << port.port
                << " numa=" << port.numa << " active=" << port.active
                << " advertised_GBs=" << port.transmit.bandwidthGBs << '\n';
    std::cout << "Memory bandwidth and effective link throughput require "
                 "calibration.\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

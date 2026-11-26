#include "uccl_lite/hardware_calibration.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc < 3)
      throw std::invalid_argument(
          "usage: calibrate_hardware PROFILE DEVICE [DEVICE...]");
    std::vector<int> devices;
    for (int i = 2; i < argc; ++i) devices.push_back(std::stoi(argv[i]));
    auto t = uccl_lite::inspectTopology(devices);
    for (size_t r = 0; r < devices.size(); ++r)
      std::cout << "device=" << devices[r] << " gpu=" << t.gpuNames[r]
                << " pci=" << t.gpuPciIds[r] << " numa=" << t.numaNodes[r]
                << '\n';
    std::cout << "driver=" << t.driverVersion
              << " CUDA-runtime=" << t.cudaRuntimeVersion << '\n';
    auto profile = uccl_lite::probeGpuResources(t);
    uccl_lite::saveHardwareProfile(argv[1], profile);
    auto loaded = uccl_lite::loadHardwareProfile(argv[1]);
    uccl_lite::applyHardwareCalibration(t, loaded.calibration);
    std::cout << "GPU resource calibration; uint4; 64 MiB large/16 byte small; "
                 "2 warmups, 5 iterations, 3 repetitions; GPU-event scope; no "
                 "baseline or per-size winner envelope\n";
    std::cout
        << "Memory rates count reads+writes; peer rates count useful one-way "
           "bytes. Rates and intercepts fit the two measured sizes.\n";
    for (auto const& r : profile.calibration.resources)
      std::cout << r.resource << " GB/s=" << r.performance.bandwidthGBs
                << " intercept_us=" << r.performance.latencyUs << '\n';
    std::cout << "Persisted " << profile.observations.size()
              << " raw observation series; physical bridges and NIC rates "
                 "remain unmeasured\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

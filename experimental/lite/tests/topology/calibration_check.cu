#include "uccl_lite/hardware_calibration.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
using namespace uccl_lite;
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
int main() {
  try {
    auto topology = inspectTopology({0, 1, 2, 3});
    check(cudaSetDevice(0));
    auto status = cudaDeviceEnablePeerAccess(1, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled)
      cudaGetLastError();
    else
      check(status);
    check(cudaSetDevice(3));
    auto profile = probeGpuResources(topology);
    int device = -1;
    check(cudaGetDevice(&device));
    if (device != 3) throw std::runtime_error("probe changed caller device");
    check(cudaSetDevice(0));
    status = cudaDeviceEnablePeerAccess(1, 0);
    if (status != cudaErrorPeerAccessAlreadyEnabled)
      throw std::runtime_error("probe removed existing peer access");
    cudaGetLastError();
    // This edge was not enabled by the test before the probe.
    status = cudaDeviceEnablePeerAccess(2, 0);
    if (status != cudaSuccess)
      throw std::runtime_error("probe leaked owned peer access");
    check(cudaDeviceDisablePeerAccess(2));
    check(cudaDeviceDisablePeerAccess(1));
    saveHardwareProfile("build/calibration-check.profile", profile);
    auto loaded = loadHardwareProfile("build/calibration-check.profile");
    applyHardwareCalibration(topology, loaded.calibration);
    if (profile.calibration.resources.size() !=
            topology.devices.size() + topology.paths.size() ||
        profile.observations.size() != 2 * profile.calibration.resources.size())
      throw std::runtime_error("incomplete calibration sweep");
    for (auto const& link : topology.links)
      if (link.performance.evidence == Evidence::Measured)
        throw std::runtime_error(
            "end-to-end probe falsely qualified a physical link");
    for (auto const& path : topology.paths)
      if (path.performance.evidence != Evidence::Measured ||
          !std::isfinite(path.performance.bandwidthGBs))
        throw std::runtime_error("measured path was not restored");
    for (auto const& o : loaded.observations)
      if (o.microseconds.size() != 3 || o.iterations != 5 || o.warmups != 2)
        throw std::runtime_error("measurement scope was lost");
    std::cout << "PASS: four-GPU verified resource sweep, raw persistence, "
                 "device restoration, and peer-access ownership\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

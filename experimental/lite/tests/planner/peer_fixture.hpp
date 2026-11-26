#pragma once
#include "uccl_lite/peer_plan.hpp"

// Synthetic bandwidths exercise plan decisions, never performance claims or
// calibration admission on a real machine. Preserve the caller's peer matrix.
inline uccl_lite::PeerTuning syntheticPeerModel(uccl_lite::Topology& t) {
  using namespace uccl_lite;
  t.hardware.clear();
  t.links.clear();
  t.paths.clear();
  t.calibrationIdentity.clear();
  t.resourceCalibrationIdentities.clear();
  t.gpuNodes.clear();
  t.networkPorts.clear();
  size_t n = t.devices.size();
  for (size_t r = 0; r < n; ++r) {
    t.gpuNodes.push_back(r);
    t.hardware.push_back({"synthetic-gpu" + std::to_string(r),
                          HardwareKind::Gpu,
                          t.devices[r],
                          0,
                          {100, 0, Evidence::Measured}});
  }
  t.hardware.push_back({"synthetic-root", HardwareKind::PciBridge});
  for (size_t r = 0; r < n; ++r) {
    t.links.push_back(
        {r, n, "up" + std::to_string(r), {10, 0, Evidence::Measured}});
    t.links.push_back(
        {n, r, "down" + std::to_string(r), {10, 0, Evidence::Measured}});
  }
  t.identity += ";synthetic-model-correctness-only";
  PeerTuning tuning;
  tuning.automaticCalibration = false;
  tuning.overheadUs = {{2, 1, 10}};
  return tuning;
}

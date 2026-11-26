#include "uccl_lite/hardware_calibration.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace uccl_lite;
namespace {
void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
template <class F>
void rejects(F f, char const* message) {
  bool rejected = false;
  try {
    f();
  } catch (std::exception const&) {
    rejected = true;
  }
  require(rejected, message);
}
}  // namespace
int main() {
  try {
    Topology t;
    t.identity = "hardware + policy knobs";
    t.hardwareIdentity = "hardware";
    t.hardware = {{"gpu", HardwareKind::Gpu}, {"peer", HardwareKind::Gpu}};
    t.paths = {{0, 1, PathKind::PeerRead, "peer-read:test", {}}};
    HardwareProfile p;
    p.calibration = {t.hardwareIdentity,
                     "synthetic probe \"quoted\"\nidentity",
                     {{"memory:gpu", {100, 2, Evidence::Measured}, 3},
                      {"peer-read:test", {20, 4, Evidence::Measured}, 3}}};
    p.observations = {
        {"memory:gpu", "test-copy", "synthetic", 1024, 2, 5, {2.1, 2.2, 2.3}},
        {"peer-read:test",
         "test-read",
         "synthetic",
         1024,
         2,
         5,
         {4.1, 4.2, 4.3}}};
    std::string file = "build/hardware-profile-check.profile";
    saveHardwareProfile(file, p);
    auto loaded = loadHardwareProfile(file);
    require(loaded.calibration.identity == p.calibration.identity &&
                loaded.observations[0].microseconds ==
                    p.observations[0].microseconds,
            "profile round trip lost metadata or samples");
    applyHardwareCalibration(t, loaded.calibration);
    require(
        t.identity == "hardware + policy knobs" &&
            t.hardware[0].memory.bandwidthGBs == 100 &&
            t.paths[0].performance.bandwidthGBs == 20,
        "matching hardware calibration must not overwrite backend identity");
    auto stale = loaded;
    stale.calibration.topologyIdentity = "other hardware";
    rejects([&] { applyHardwareCalibration(t, stale.calibration); },
            "stale profile admitted");
    auto invalid = p;
    invalid.observations.clear();
    rejects([&] { saveHardwareProfile(file, invalid); },
            "profile without raw observations accepted");
    require(loadHardwareProfile(file).observations.size() == 2,
            "failed save destroyed prior profile");
    invalid = p;
    invalid.observations[0].microseconds[0] = -1;
    rejects([&] { saveHardwareProfile(file, invalid); },
            "negative observation accepted");
    invalid = p;
    invalid.observations[0].microseconds.push_back(2);
    rejects([&] { saveHardwareProfile(file, invalid); },
            "repetition mismatch accepted");
    invalid = p;
    invalid.calibration.resources.push_back(invalid.calibration.resources[0]);
    rejects([&] { saveHardwareProfile(file, invalid); },
            "duplicate resource accepted");
    {
      std::ofstream out(file, std::ios::app);
      out << "unexpected data\n";
    }
    rejects([&] { loadHardwareProfile(file); }, "trailing data accepted");
    {
      std::ofstream out(file);
      out << "uccl-lite-hardware-profile-v1\n\"id\"\n";
    }
    rejects([&] { loadHardwareProfile(file); }, "truncated profile accepted");
    saveHardwareProfile(file, p);
    std::cout << "Hardware profiles: atomic replacement, identity, raw "
                 "samples, corruption and path calibration passed\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

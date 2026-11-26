#include "uccl_lite/hardware_calibration.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <stdexcept>
#include <unistd.h>

namespace uccl_lite {
namespace {
void validate(HardwareProfile const& p) {
  if (p.calibration.identity.empty() ||
      p.calibration.topologyIdentity.empty() || p.calibration.resources.empty())
    throw std::invalid_argument("incomplete hardware profile");
  std::set<std::string> resources, observed;
  std::map<std::string, int> repetitions;
  for (auto const& r : p.calibration.resources) {
    if (r.resource.empty() || !resources.insert(r.resource).second ||
        r.repetitions < 3 || r.performance.evidence != Evidence::Measured ||
        !std::isfinite(r.performance.bandwidthGBs) ||
        r.performance.bandwidthGBs <= 0 ||
        !std::isfinite(r.performance.latencyUs) || r.performance.latencyUs < 0)
      throw std::invalid_argument("invalid hardware profile rate");
    repetitions[r.resource] = r.repetitions;
  }
  for (auto const& o : p.observations) {
    if (!resources.count(o.resource) || o.operation.empty() ||
        o.timingScope.empty() || !o.bytes || o.warmups < 0 ||
        o.iterations < 1 || o.microseconds.size() < 3 ||
        o.microseconds.size() != size_t(repetitions[o.resource]))
      throw std::invalid_argument("invalid hardware probe observation");
    observed.insert(o.resource);
    for (double us : o.microseconds)
      if (!std::isfinite(us) || us <= 0)
        throw std::invalid_argument("invalid probe duration");
  }
  if (observed != resources)
    throw std::invalid_argument("hardware profile is missing raw observations");
}
struct Temporary {
  std::string path;
  explicit Temporary(std::string const& target) {
    auto name = target + ".tmp.XXXXXX";
    std::vector<char> buffer(name.begin(), name.end());
    buffer.push_back(0);
    int fd = mkstemp(buffer.data());
    if (fd < 0)
      throw std::runtime_error("cannot create temporary hardware profile");
    path = buffer.data();
    if (close(fd)) {
      std::remove(path.c_str());
      throw std::runtime_error("cannot close temporary hardware profile");
    }
  }
  ~Temporary() {
    if (!path.empty()) std::remove(path.c_str());
  }
};
}  // namespace
void saveHardwareProfile(std::string const& path, HardwareProfile const& p) {
  validate(p);
  Temporary temporary(path);
  std::ofstream out(temporary.path);
  if (!out) throw std::runtime_error("cannot write hardware profile: " + path);
  out << "uccl-lite-hardware-profile-v1\n"
      << std::setprecision(17) << std::quoted(p.calibration.topologyIdentity)
      << '\n'
      << std::quoted(p.calibration.identity) << '\n'
      << p.calibration.resources.size() << ' ' << p.observations.size() << '\n';
  for (auto const& r : p.calibration.resources)
    out << std::quoted(r.resource) << ' ' << r.performance.bandwidthGBs << ' '
        << r.performance.latencyUs << ' ' << r.repetitions << '\n';
  for (auto const& o : p.observations) {
    out << std::quoted(o.resource) << ' ' << std::quoted(o.operation) << ' '
        << std::quoted(o.timingScope) << ' ' << o.bytes << ' ' << o.warmups
        << ' ' << o.iterations << ' ' << o.microseconds.size();
    for (double us : o.microseconds) out << ' ' << us;
    out << '\n';
  }
  out.close();
  if (!out) throw std::runtime_error("incomplete hardware profile write");
  if (std::rename(temporary.path.c_str(), path.c_str()))
    throw std::runtime_error("cannot replace hardware profile");
  temporary.path.clear();
}
HardwareProfile loadHardwareProfile(std::string const& path) {
  std::ifstream in(path);
  std::string magic;
  std::getline(in, magic);
  if (magic != "uccl-lite-hardware-profile-v1")
    throw std::invalid_argument("unsupported hardware profile");
  HardwareProfile p;
  size_t rates = 0, observations = 0;
  if (!(in >> std::quoted(p.calibration.topologyIdentity) >>
        std::quoted(p.calibration.identity) >> rates >> observations) ||
      rates > 100000 || observations > 200000)
    throw std::invalid_argument("invalid hardware profile dimensions");
  for (size_t i = 0; i < rates; ++i) {
    ResourceCalibration r;
    if (!(in >> std::quoted(r.resource) >> r.performance.bandwidthGBs >>
          r.performance.latencyUs >> r.repetitions))
      throw std::invalid_argument("truncated hardware rate");
    r.performance.evidence = Evidence::Measured;
    p.calibration.resources.push_back(std::move(r));
  }
  size_t totalSamples = 0;
  for (size_t i = 0; i < observations; ++i) {
    ProbeObservation o;
    size_t count = 0;
    if (!(in >> std::quoted(o.resource) >> std::quoted(o.operation) >>
          std::quoted(o.timingScope) >> o.bytes >> o.warmups >> o.iterations >>
          count) ||
        count > 100000)
      throw std::invalid_argument("invalid hardware observation dimensions");
    if (count > 1000000 - totalSamples)
      throw std::invalid_argument("hardware profile sample budget exceeded");
    totalSamples += count;
    o.microseconds.resize(count);
    for (auto& us : o.microseconds)
      if (!(in >> us))
        throw std::invalid_argument("truncated hardware observation");
    p.observations.push_back(std::move(o));
  }
  in >> std::ws;
  if (!in.eof()) throw std::invalid_argument("trailing hardware profile data");
  validate(p);
  return p;
}
}  // namespace uccl_lite

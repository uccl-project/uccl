#include "uccl_lite/network_calibration.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
void validate(NetworkProfile const& p) {
  if (p.groupScope != kNetworkGroupProbeScope &&
      p.groupScope != kNetworkBootstrapGroupProbeScope)
    throw std::invalid_argument("unsupported network cohort timing scope");
  if (p.pathScope != kNetworkProbeScope &&
      p.pathScope != kLegacyNetworkProbeScope)
    throw std::invalid_argument("unsupported network path timing scope");
  if (p.probeIdentity.empty() || p.paths.empty() || p.paths.size() > 4096 ||
      p.hosts[0].hostname == p.hosts[1].hostname)
    throw std::invalid_argument("invalid network profile identity");
  for (auto const& h : p.hosts)
    if (h.hostname.empty() || h.topologyIdentity.empty() ||
        h.transportIdentity.empty())
      throw std::invalid_argument("missing network endpoint identity");
  std::set<NetworkPathKey> keys;
  size_t totalSamples = 0;
  for (auto const& path : p.paths) {
    auto const& k = path.key;
    if (k.sourceHost < 0 || k.sourceHost > 1 ||
        k.destinationHost != 1 - k.sourceHost || k.sourceGpu.empty() ||
        k.destinationGpu.empty() || k.sourceNic.empty() ||
        k.destinationNic.empty() || k.sourcePort < 1 || k.destinationPort < 1 ||
        !keys.insert(k).second || path.warmups < 0 || path.iterations < 1 ||
        path.iterations > 10000 || path.repetitions < 3 ||
        path.repetitions > 10000 ||
        (path.rejection.empty() ? path.sizes.size() < 2
                                : !path.sizes.empty()) ||
        path.sizes.size() > 64)
      throw std::invalid_argument("invalid network path observation");
    std::set<size_t> sizes;
    for (auto const& observation : path.sizes) {
      if (observation.microseconds.size() > 1000000 - totalSamples)
        throw std::invalid_argument("network profile sample budget exceeded");
      totalSamples += observation.microseconds.size();
      if (!observation.bytes || observation.bytes > (1UL << 30) ||
          !sizes.insert(observation.bytes).second ||
          observation.microseconds.size() !=
              size_t(path.iterations) * path.repetitions)
        throw std::invalid_argument(
            "network samples do not cover declared repetitions");
      for (double value : observation.microseconds)
        if (!std::isfinite(value) || value <= 0)
          throw std::invalid_argument("invalid network duration");
    }
  }
  if (p.groups.size() > 4096)
    throw std::invalid_argument("too many network groups");
  if (!p.groups.empty() && (p.groupControlIdentities[0].empty() ||
                            p.groupControlIdentities[1].empty()))
    throw std::invalid_argument("missing concurrent control provenance");
  std::set<std::vector<NetworkPathKey>> groups;
  for (auto const& group : p.groups) {
    auto flows = group.flows;
    std::sort(flows.begin(), flows.end());
    if (group.label.empty() || flows.size() < 2 || flows.size() > 64 ||
        std::adjacent_find(flows.begin(), flows.end()) != flows.end() ||
        !groups.insert(flows).second || group.warmups < 0 ||
        group.iterations < 1 || group.iterations > 10000 ||
        group.repetitions < 3 || group.repetitions > 10000 ||
        (group.rejection.empty() && group.sizes.size() < 2) ||
        group.sizes.size() > 64)
      throw std::invalid_argument("invalid concurrent network group");
    for (auto const& flow : flows)
      if (!keys.count(flow))
        throw std::invalid_argument(
            "network group references an unknown endpoint path");
    std::set<size_t> sizes;
    for (auto const& o : group.sizes) {
      size_t expected = size_t(group.iterations) * group.repetitions;
      if (!o.bytes || o.bytes > (1UL << 30) || !sizes.insert(o.bytes).second ||
          (group.rejection.empty() ? o.senderMicroseconds.size() != expected
                                   : o.senderMicroseconds.size() > expected))
        throw std::invalid_argument("incomplete concurrent network samples");
      if (o.cohortMicroseconds.size() != o.senderMicroseconds.size())
        throw std::invalid_argument("missing cohort durations");
      for (size_t s = 0; s < o.senderMicroseconds.size(); ++s) {
        auto const& sample = o.senderMicroseconds[s];
        auto const& cohort = o.cohortMicroseconds[s];
        if (sample.size() != flows.size() || cohort.size() != flows.size() ||
            sample.size() > (1000000 - totalSamples) / 2)
          throw std::invalid_argument(
              "concurrent sender sample extent invalid");
        totalSamples += 2 * sample.size();
        for (size_t r = 0; r < sample.size(); ++r)
          if (!std::isfinite(sample[r]) || sample[r] <= 0 ||
              !std::isfinite(cohort[r]) || cohort[r] < sample[r])
            throw std::invalid_argument(
                "invalid concurrent sender/cohort duration");
      }
    }
  }
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  size_t n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
}  // namespace
std::string encodeNetworkProfile(NetworkProfile const& p) {
  validate(p);
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "uccl-lite-network-profile-v2\n"
      << std::quoted(p.pathScope) << '\n'
      << std::quoted(p.probeIdentity) << '\n'
      << std::setprecision(17);
  for (auto const& h : p.hosts)
    out << std::quoted(h.hostname) << ' ' << std::quoted(h.topologyIdentity)
        << ' ' << std::quoted(h.transportIdentity) << '\n';
  out << p.paths.size() << '\n';
  for (auto const& path : p.paths) {
    auto const& k = path.key;
    out << k.sourceHost << ' ' << k.destinationHost << ' '
        << std::quoted(k.sourceGpu) << ' ' << std::quoted(k.destinationGpu)
        << ' ' << std::quoted(k.sourceNic) << ' '
        << std::quoted(k.destinationNic) << ' ' << k.sourcePort << ' '
        << k.destinationPort << ' ' << path.warmups << ' ' << path.iterations
        << ' ' << path.repetitions << ' ' << std::quoted(path.rejection) << ' '
        << path.sizes.size() << '\n';
    for (auto const& o : path.sizes) {
      out << o.bytes << ' ' << o.microseconds.size();
      for (double value : o.microseconds) out << ' ' << value;
      out << '\n';
    }
  }
  out << std::quoted(p.groupScope) << ' '
      << std::quoted(p.groupControlIdentities[0]) << ' '
      << std::quoted(p.groupControlIdentities[1]) << ' ' << p.groups.size()
      << '\n';
  for (auto const& group : p.groups) {
    out << std::quoted(group.label) << ' ' << group.warmups << ' '
        << group.iterations << ' ' << group.repetitions << ' '
        << std::quoted(group.rejection) << ' ' << group.flows.size() << ' '
        << group.sizes.size() << '\n';
    for (auto const& k : group.flows)
      out << k.sourceHost << ' ' << k.destinationHost << ' '
          << std::quoted(k.sourceGpu) << ' ' << std::quoted(k.destinationGpu)
          << ' ' << std::quoted(k.sourceNic) << ' '
          << std::quoted(k.destinationNic) << ' ' << k.sourcePort << ' '
          << k.destinationPort << '\n';
    for (auto const& o : group.sizes) {
      out << o.bytes << ' ' << o.senderMicroseconds.size();
      for (size_t s = 0; s < o.senderMicroseconds.size(); ++s)
        for (size_t r = 0; r < group.flows.size(); ++r)
          out << ' ' << o.senderMicroseconds[s][r] << ' '
              << o.cohortMicroseconds[s][r];
      out << '\n';
    }
  }
  auto data = out.str();
  if (data.size() > (64UL << 20))
    throw std::invalid_argument("network profile too large");
  return data;
}
void saveNetworkProfile(std::string const& file, NetworkProfile const& p) {
  validate(p);
  std::string pattern = file + ".tmp.XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back(0);
  int fd = mkstemp(name.data());
  if (fd < 0)
    throw std::runtime_error("cannot create network profile temporary");
  close(fd);
  try {
    std::ofstream out(name.data());
    out << encodeNetworkProfile(p);
    out.close();
    if (!out || std::rename(name.data(), file.c_str()))
      throw std::runtime_error("cannot publish network profile");
  } catch (...) {
    std::remove(name.data());
    throw;
  }
}
NetworkProfile loadNetworkProfile(std::string const& file) {
  struct stat info {};
  if (stat(file.c_str(), &info) || !S_ISREG(info.st_mode) || info.st_size < 0 ||
      info.st_size > (64 << 20))
    throw std::invalid_argument("network profile missing or too large");
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read network profile");
  std::string data(size_t(info.st_size), '\0');
  in.read(data.data(), data.size());
  if (size_t(in.gcount()) != data.size())
    throw std::runtime_error("network profile changed during read");
  char extra;
  if (in.get(extra) || !in.eof())
    throw std::runtime_error("network profile read extent changed");
  return decodeNetworkProfile(data);
}
NetworkProfile decodeNetworkProfile(std::string const& data) {
  if (data.size() > (64UL << 20))
    throw std::invalid_argument("network profile too large");
  std::istringstream in(data);
  in.imbue(std::locale::classic());
  std::string magic, scope;
  NetworkProfile p;
  if (!(in >> magic >> std::quoted(scope) >> std::quoted(p.probeIdentity)) ||
      (magic != "uccl-lite-network-profile-v1" &&
       magic != "uccl-lite-network-profile-v2") ||
      (scope != kNetworkProbeScope && scope != kLegacyNetworkProbeScope))
    throw std::invalid_argument("unsupported network profile scope/version");
  p.pathScope = scope;
  for (auto& h : p.hosts)
    if (!(in >> std::quoted(h.hostname) >> std::quoted(h.topologyIdentity) >>
          std::quoted(h.transportIdentity)))
      throw std::invalid_argument("truncated network endpoint identity");
  size_t paths = 0, total = 0;
  if (!(in >> paths) || paths > 4096)
    throw std::invalid_argument("invalid network path count");
  for (size_t i = 0; i < paths; ++i) {
    NetworkPathObservation path;
    auto& k = path.key;
    size_t sizes = 0;
    if (!(in >> k.sourceHost >> k.destinationHost >> std::quoted(k.sourceGpu) >>
          std::quoted(k.destinationGpu) >> std::quoted(k.sourceNic) >>
          std::quoted(k.destinationNic) >> k.sourcePort >> k.destinationPort >>
          path.warmups >> path.iterations >> path.repetitions >>
          std::quoted(path.rejection) >> sizes) ||
        sizes > 64)
      throw std::invalid_argument("truncated network path");
    for (size_t j = 0; j < sizes; ++j) {
      NetworkObservation o;
      size_t count = 0;
      if (!(in >> o.bytes >> count) || count > 1000000 - total)
        throw std::invalid_argument("network profile sample budget exceeded");
      total += count;
      o.microseconds.resize(count);
      for (double& value : o.microseconds)
        if (!(in >> value))
          throw std::invalid_argument("truncated network samples");
      path.sizes.push_back(std::move(o));
    }
    p.paths.push_back(std::move(path));
  }
  if (magic == "uccl-lite-network-profile-v2") {
    size_t groups = 0;
    if (!(in >> std::quoted(scope) >>
          std::quoted(p.groupControlIdentities[0]) >>
          std::quoted(p.groupControlIdentities[1]) >> groups) ||
        (scope != kNetworkGroupProbeScope &&
         scope != kNetworkBootstrapGroupProbeScope) ||
        groups > 4096)
      throw std::invalid_argument("invalid concurrent network profile scope");
    p.groupScope = scope;
    for (size_t i = 0; i < groups; ++i) {
      NetworkGroupObservation group;
      size_t flows = 0, sizes = 0;
      if (!(in >> std::quoted(group.label) >> group.warmups >>
            group.iterations >> group.repetitions >>
            std::quoted(group.rejection) >> flows >> sizes) ||
          flows < 2 || flows > 64 || sizes > 64)
        throw std::invalid_argument("invalid concurrent network dimensions");
      group.flows.resize(flows);
      for (auto& k : group.flows)
        if (!(in >> k.sourceHost >> k.destinationHost >>
              std::quoted(k.sourceGpu) >> std::quoted(k.destinationGpu) >>
              std::quoted(k.sourceNic) >> std::quoted(k.destinationNic) >>
              k.sourcePort >> k.destinationPort))
          throw std::invalid_argument("truncated concurrent network flow");
      for (size_t s = 0; s < sizes; ++s) {
        NetworkGroupObservation::Size o;
        size_t samples = 0;
        if (!(in >> o.bytes >> samples) ||
            samples > (1000000 - total) / (2 * flows))
          throw std::invalid_argument(
              "concurrent network sample budget exceeded");
        total += 2 * samples * flows;
        o.senderMicroseconds.resize(samples, std::vector<double>(flows));
        o.cohortMicroseconds.resize(samples, std::vector<double>(flows));
        for (size_t s = 0; s < samples; ++s)
          for (size_t r = 0; r < flows; ++r)
            if (!(in >> o.senderMicroseconds[s][r] >>
                  o.cohortMicroseconds[s][r]))
              throw std::invalid_argument(
                  "truncated concurrent network sample");
        group.sizes.push_back(std::move(o));
      }
      p.groups.push_back(std::move(group));
    }
  }
  in >> std::ws;
  if (!in.eof()) throw std::invalid_argument("trailing network profile data");
  validate(p);
  return p;
}
std::optional<NetworkPathEstimate> estimateNetworkPath(
    NetworkProfile const& p, std::array<NetworkHostIdentity, 2> const& current,
    NetworkPathKey const& key, size_t bytes) {
  validate(p);
  if (p.pathScope != kNetworkProbeScope) return {};
  for (size_t h = 0; h < 2; ++h)
    if (current[h].hostname != p.hosts[h].hostname ||
        current[h].topologyIdentity != p.hosts[h].topologyIdentity ||
        current[h].transportIdentity != p.hosts[h].transportIdentity)
      return {};
  auto path = std::find_if(p.paths.begin(), p.paths.end(),
                           [&](auto const& x) { return x.key == key; });
  if (path == p.paths.end() || !path->rejection.empty() || !bytes) return {};
  NetworkObservation const *lower = nullptr, *upper = nullptr;
  for (auto const& observation : path->sizes) {
    if (observation.bytes <= bytes &&
        (!lower || observation.bytes > lower->bytes))
      lower = &observation;
    if (observation.bytes >= bytes &&
        (!upper || observation.bytes < upper->bytes))
      upper = &observation;
  }
  if (!lower || !upper) return {};
  double low = median(lower->microseconds), high = median(upper->microseconds);
  // A noisy/nonmonotonic interval is not a defensible interpolated model.
  if (lower != upper && high <= low) return {};
  double us = lower == upper
                  ? low
                  : low + (high - low) * (double(bytes - lower->bytes) /
                                          double(upper->bytes - lower->bytes));
  return NetworkPathEstimate{us, double(bytes) / us / 1000, lower->bytes,
                             upper->bytes};
}
std::vector<double> networkGroupSamples(NetworkGroupObservation const& group,
                                        size_t bytes) {
  std::vector<double> result;
  if (!group.rejection.empty()) return result;
  for (auto const& o : group.sizes)
    if (o.bytes == bytes) {
      for (auto const& sample : o.cohortMicroseconds) {
        if (sample.size() != group.flows.size() || sample.empty())
          throw std::invalid_argument("invalid concurrent sender dimensions");
        for (double value : sample)
          if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("invalid concurrent sender sample");
        result.push_back(*std::max_element(sample.begin(), sample.end()));
      }
    }
  return result;
}
std::optional<NetworkPathEstimate> estimateNetworkGroup(
    NetworkProfile const& p, std::array<NetworkHostIdentity, 2> const& current,
    std::vector<NetworkPathKey> flows, size_t bytes) {
  validate(p);
  for (size_t h = 0; h < 2; ++h)
    if (current[h].hostname != p.hosts[h].hostname ||
        current[h].topologyIdentity != p.hosts[h].topologyIdentity ||
        current[h].transportIdentity != p.hosts[h].transportIdentity)
      return {};
  std::sort(flows.begin(), flows.end());
  for (auto const& group : p.groups) {
    auto candidate = group.flows;
    std::sort(candidate.begin(), candidate.end());
    if (candidate != flows) continue;
    auto samples = networkGroupSamples(group, bytes);
    if (samples.empty()) return {};
    double us = median(std::move(samples));
    return NetworkPathEstimate{us, double(bytes) * flows.size() / us / 1000,
                               bytes, bytes};
  }
  return {};
}
}  // namespace uccl_lite

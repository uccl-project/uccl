#include "uccl_lite/backend_policy.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace uccl_lite {
namespace {
constexpr char const* kReference = "reference:l4x8-no-p2p-20261012";
constexpr char const* kProfileHeader = "uccl-lite-router-profile-v2";
constexpr char const* kLegacyProfileHeader = "uccl-lite-router-profile-v1";

bool valid(TimingRange const& t) {
  return std::isfinite(t.minimum) && std::isfinite(t.median) &&
         std::isfinite(t.maximum) && t.minimum > 0 && t.minimum <= t.median &&
         t.median <= t.maximum;
}
void validate(Measurement const& m) {
  unsigned alignment = m.workload.alignment;
  if (m.topologyIdentity.empty() || m.fallbackIdentity.empty() ||
      !m.workload.bytes || !valid(m.router) || !valid(m.fallback) ||
      m.repetitions < 3 || !m.correct ||
      m.workload.dtype != ScalarType::BFloat16 || alignment > 256 ||
      (alignment && (alignment & (alignment - 1))))
    throw std::invalid_argument(
        "invalid or unvalidated CPU-router measurement");
}
bool sameWorkload(Workload const& a, Workload const& b) {
  return a.collective == b.collective && a.dtype == b.dtype &&
         a.bytes == b.bytes && a.inPlace == b.inPlace &&
         a.alignment == b.alignment;
}
bool capable(Topology const& t, Workload const& w) {
  size_t n = t.devices.size();
  if (!n || n > 32 || !t.avx2 || w.dtype != ScalarType::BFloat16 || !w.bytes ||
      w.bytes % 2 || t.numaNodes.size() != n || t.mappedHost.size() != n ||
      (w.alignment && w.alignment < 2) || t.gpuNames.size() != n ||
      t.peerAccess.size() != n * n ||
      std::set<int>(t.devices.begin(), t.devices.end()).size() != n)
    return false;
  if (w.collective != Collective::AllReduce && w.bytes % (2 * n)) return false;
  for (size_t r = 0; r < n; ++r) {
    auto cpus = t.permittedCpusPerNode.find(t.numaNodes[r]);
    if (!t.mappedHost[r] || t.numaNodes[r] < 0 ||
        cpus == t.permittedCpusPerNode.end() || cpus->second < 1)
      return false;
  }
  return true;
}
BackendDecision evaluate(Measurement const& m, double threshold,
                         char const* source) {
  validate(m);
  double gain = m.fallback.median / m.router.median;
  bool wins = gain >= threshold && m.router.maximum < m.fallback.minimum;
  return {wins,
          std::string(source) +
              (wins ? ": measured gain exceeds margin; ranges do not overlap"
                    : ": no stable gain; keep existing backend"),
          gain};
}
std::vector<std::string> split(std::string const& line) {
  std::vector<std::string> fields;
  std::istringstream input(line);
  std::string field;
  while (std::getline(input, field, ',')) fields.push_back(field);
  return fields;
}
}  // namespace

char const* collectiveName(Collective op) {
  switch (op) {
    case Collective::AllReduce:
      return "all_reduce";
    case Collective::AllGather:
      return "all_gather";
    case Collective::ReduceScatter:
      return "reduce_scatter";
  }
  throw std::invalid_argument("unknown collective");
}
Collective parseCollective(std::string const& name) {
  if (name == "all_reduce" || name == "allreduce") return Collective::AllReduce;
  if (name == "all_gather" || name == "allgather") return Collective::AllGather;
  if (name == "reduce_scatter" || name == "reducescatter")
    return Collective::ReduceScatter;
  throw std::invalid_argument("unknown collective: " + name);
}
RouterMode parseRouterMode(std::string const& name) {
  if (name == "auto") return RouterMode::Auto;
  if (name == "off" || name == "fallback") return RouterMode::Off;
  if (name == "force" || name == "cpu-router") return RouterMode::Force;
  throw std::invalid_argument("backend must be auto, off, or force");
}

BackendDecision BackendPolicy::select(Topology const& topology,
                                      Workload const& workload) const {
  if (!std::isfinite(minimumSpeedup) || minimumSpeedup < 1)
    throw std::invalid_argument(
        "minimum speedup must be finite and at least one");
  if (mode == RouterMode::Off) return {false, "CPU-router disabled by caller"};
  if (!capable(topology, workload)) {
    if (mode == RouterMode::Force)
      throw std::invalid_argument(
          "CPU-router cannot execute this topology/dtype/workload");
    return {false,
            "unsupported topology/dtype/workload; keep existing backend"};
  }
  if (mode == RouterMode::Force)
    return {true, "CPU-router explicitly forced; no performance claim"};
  // Only an exact local calibration may enable the router. A historical
  // machine description cannot capture PCIe contention, memory configuration,
  // CPU model, or every NCCL setting that affects the crossover.
  if (!topology.identity.empty())
    for (auto it = measurements.rbegin(); it != measurements.rend(); ++it)
      if (sameWorkload(it->workload, workload) &&
          it->topologyIdentity == topology.identity &&
          it->fallbackIdentity == fallbackIdentity)
        return evaluate(*it, minimumSpeedup, "local calibration");
  return {
      false,
      "no matching measurement; keep existing backend (calibrate to opt in)"};
}

std::vector<Measurement> loadL4ReferenceProfile(std::string const& path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open reference profile: " + path);
  std::string line;
  std::getline(input, line);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line !=
      "collective,mode,bytes,backend,median_us,min_us,max_us,bus_bw_gbs,alg_bw_"
      "gbs")
    throw std::invalid_argument("unexpected reference CSV header");
  using Key = std::tuple<Collective, bool, size_t>;
  std::map<Key, Measurement> pairs;
  std::set<std::tuple<Key, std::string>> seen;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    auto f = split(line);
    if (f.size() != 9)
      throw std::invalid_argument("malformed reference CSV row");
    if (f[3] != "kernel" && f[3] != "nccl") continue;
    if (f[1] != "in_place" && f[1] != "out_of_place")
      throw std::invalid_argument("unknown placement in reference CSV");
    Workload workload{parseCollective(f[0]), ScalarType::BFloat16,
                      std::stoull(f[2]), f[1] == "in_place"};
    Key key{workload.collective, workload.inPlace, workload.bytes};
    if (!seen.emplace(key, f[3]).second)
      throw std::invalid_argument("duplicate reference row");
    auto& m = pairs[key];
    m.topologyIdentity = kReference;
    m.fallbackIdentity = "nccl:22705";
    m.workload = workload;
    m.repetitions = 3;
    m.correct = true;  // The committed matched experiment checked all
                       // backend/size pairs.
    TimingRange t{std::stod(f[5]), std::stod(f[4]), std::stod(f[6])};
    (f[3] == "kernel" ? m.router : m.fallback) = t;
  }
  if (pairs.size() != 144)
    throw std::invalid_argument("incomplete eight-L4 reference profile");
  std::vector<Measurement> output;
  for (auto const& pair : pairs) {
    validate(pair.second);
    output.push_back(pair.second);
  }
  return output;
}

void saveRouterProfile(std::string const& path,
                       std::vector<Measurement> const& measurements) {
  for (auto const& m : measurements) validate(m);
  std::ofstream output(path);
  if (!output)
    throw std::runtime_error("cannot write calibration profile: " + path);
  output << kProfileHeader << '\n' << std::setprecision(17);
  for (auto const& m : measurements)
    output << std::quoted(m.topologyIdentity) << ' '
           << std::quoted(m.fallbackIdentity) << ' '
           << collectiveName(m.workload.collective) << ' ' << m.workload.bytes
           << ' ' << m.workload.inPlace << ' ' << m.workload.alignment << ' '
           << m.repetitions << ' ' << m.correct << ' ' << m.router.minimum
           << ' ' << m.router.median << ' ' << m.router.maximum << ' '
           << m.fallback.minimum << ' ' << m.fallback.median << ' '
           << m.fallback.maximum << '\n';
  output.flush();
  if (!output) throw std::runtime_error("failed to write calibration profile");
}

std::vector<Measurement> loadRouterProfile(std::string const& path) {
  std::ifstream input(path);
  std::string line;
  if (!input || !std::getline(input, line) ||
      (line != kProfileHeader && line != kLegacyProfileHeader))
    throw std::invalid_argument("missing or unsupported CPU-router profile: " +
                                path);
  bool legacy = line == kLegacyProfileHeader;
  std::vector<Measurement> output;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    Measurement m;
    std::string op, extra;
    int placement, correct;
    std::istringstream row(line);
    if (!(row >> std::quoted(m.topologyIdentity) >>
          std::quoted(m.fallbackIdentity) >> op >> m.workload.bytes >>
          placement) ||
        (!legacy && !(row >> m.workload.alignment)) ||
        !(row >> m.repetitions >> correct >> m.router.minimum >>
          m.router.median >> m.router.maximum >> m.fallback.minimum >>
          m.fallback.median >> m.fallback.maximum) ||
        row >> extra || placement < 0 || placement > 1 || correct != 1)
      throw std::invalid_argument("malformed CPU-router calibration row");
    m.workload.collective = parseCollective(op);
    m.workload.inPlace = placement;
    m.correct = correct;
    if (m.topologyIdentity == kReference)
      throw std::invalid_argument(
          "load historical evidence with loadL4ReferenceProfile");
    validate(m);
    output.push_back(m);
  }
  return output;
}
}  // namespace uccl_lite

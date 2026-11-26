#include "uccl_lite/peer_profile.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
constexpr char magic[] = "uccl-lite-peer-plans-v1";
constexpr char scope[] =
    "host-completion;warmups=2;calls=3;repetitions=3;guards=full;baseline="
    "direct";
bool same(Workload const& a, Workload const& b) {
  return a.collective == b.collective && a.dtype == b.dtype &&
         a.bytes == b.bytes && a.inPlace == b.inPlace &&
         a.alignment == b.alignment;
}
double median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  return x[x.size() / 2];
}
void validate(PeerPlanRecord const& r) {
  if (r.topologyIdentity.empty() || r.executorIdentity.empty() ||
      r.calibrationIdentity.empty() || !r.correct || !r.workload.bytes ||
      r.workload.bytes % 2 || r.workload.dtype != ScalarType::BFloat16 ||
      r.workload.inPlace || r.workload.alignment < 2 ||
      r.workload.alignment > 256 ||
      (r.workload.alignment & (r.workload.alignment - 1)) ||
      int(r.workload.collective) < 0 || int(r.workload.collective) > 2)
    throw std::invalid_argument("invalid peer plan record");
  qualifiedPeerFamily(r.timings);
}
struct Lock {
  int fd = -1;
  explicit Lock(std::string const& path) {
    fd = open((path + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("cannot open peer plan lock");
    if (flock(fd, LOCK_EX)) {
      close(fd);
      fd = -1;
      throw std::runtime_error("cannot lock peer plans");
    }
  }
  ~Lock() {
    if (fd >= 0) close(fd);
  }
};
struct Temporary {
  std::string name;
  explicit Temporary(std::string const& path) {
    auto text = path + ".tmp.XXXXXX";
    std::vector<char> chars(text.begin(), text.end());
    chars.push_back(0);
    int fd = mkstemp(chars.data());
    if (fd < 0) throw std::runtime_error("cannot create peer plan temporary");
    name = chars.data();
    if (close(fd)) {
      std::remove(name.c_str());
      throw std::runtime_error("cannot close peer plan temporary");
    }
  }
  ~Temporary() {
    if (!name.empty()) std::remove(name.c_str());
  }
};
}  // namespace
PeerFamily qualifiedPeerFamily(
    std::array<std::vector<double>, 3> const& timings) {
  if (timings[0].size() != 3)
    throw std::invalid_argument(
        "peer qualification requires three direct baseline repetitions");
  for (auto const& samples : timings) {
    if (!samples.empty() && samples.size() != 3)
      throw std::invalid_argument("invalid peer repetition count");
    for (auto us : samples)
      if (!std::isfinite(us) || us <= 0)
        throw std::invalid_argument("invalid peer timing");
  }
  double baseline = median(timings[0]), best = baseline;
  PeerFamily winner = PeerFamily::Direct;
  for (size_t i = 1; i < timings.size(); ++i) {
    if (timings[i].empty()) continue;
    double candidate = median(timings[i]);
    if (candidate < best && candidate * 1.03 < baseline &&
        *std::max_element(timings[i].begin(), timings[i].end()) <
            *std::min_element(timings[0].begin(), timings[0].end())) {
      best = candidate;
      winner = static_cast<PeerFamily>(i);
    }
  }
  return winner;
}
std::vector<PeerPlanRecord> loadPeerPlanRecords(std::string const& path) {
  if (!std::filesystem::exists(path)) return {};
  std::ifstream in(path);
  std::string header, timingScope;
  std::getline(in, header);
  std::getline(in, timingScope);
  if (header != magic || timingScope != scope)
    throw std::invalid_argument("unsupported peer plan format or timing scope");
  size_t count = 0;
  if (!(in >> count) || count > 100000)
    throw std::invalid_argument("invalid peer record count");
  std::vector<PeerPlanRecord> result;
  for (size_t i = 0; i < count; ++i) {
    PeerPlanRecord r;
    int op = 0, dtype = 0;
    if (!(in >> std::quoted(r.topologyIdentity) >>
          std::quoted(r.executorIdentity) >>
          std::quoted(r.calibrationIdentity) >> op >> dtype >>
          r.workload.bytes >> r.workload.inPlace >> r.workload.alignment >>
          r.correct))
      throw std::invalid_argument("truncated peer plan");
    r.workload.collective = static_cast<Collective>(op);
    r.workload.dtype = static_cast<ScalarType>(dtype);
    for (auto& values : r.timings) {
      size_t n = 0;
      if (!(in >> n) || (n != 0 && n != 3))
        throw std::invalid_argument("invalid peer sample extent");
      values.resize(n);
      for (auto& x : values)
        if (!(in >> x)) throw std::invalid_argument("truncated peer samples");
    }
    validate(r);
    result.push_back(std::move(r));
  }
  in >> std::ws;
  if (!in.eof()) throw std::invalid_argument("trailing peer plan data");
  return result;
}
void updatePeerPlanRecord(std::string const& path,
                          PeerPlanRecord const& record) {
  validate(record);
  Lock lock(path);
  auto records = loadPeerPlanRecords(path);
  records.erase(
      std::remove_if(records.begin(), records.end(),
                     [&](PeerPlanRecord const& r) {
                       return r.topologyIdentity == record.topologyIdentity &&
                              r.executorIdentity == record.executorIdentity &&
                              same(r.workload, record.workload);
                     }),
      records.end());
  records.push_back(record);
  Temporary temp(path);
  std::ofstream out(temp.name);
  out << magic << '\n'
      << scope << '\n'
      << records.size() << '\n'
      << std::setprecision(17);
  for (auto const& r : records) {
    out << std::quoted(r.topologyIdentity) << ' '
        << std::quoted(r.executorIdentity) << ' '
        << std::quoted(r.calibrationIdentity) << ' '
        << int(r.workload.collective) << ' ' << int(r.workload.dtype) << ' '
        << r.workload.bytes << ' ' << r.workload.inPlace << ' '
        << r.workload.alignment << ' ' << r.correct << '\n';
    for (auto const& values : r.timings) {
      out << values.size();
      for (auto x : values) out << ' ' << x;
      out << '\n';
    }
  }
  out.close();
  if (!out || std::rename(temp.name.c_str(), path.c_str()))
    throw std::runtime_error("cannot replace peer plans");
  temp.name.clear();
}
std::optional<PeerSelection> matchPeerPlanRecord(
    Topology const& t, Workload const& w, std::string const& executor,
    std::vector<PeerPlanRecord> const& records) {
  PeerTuning tuning;
  tuning.automaticCalibration = false;
  auto candidates = peerCandidates(t, w, tuning);
  auto capability = choosePlan(t, w, false, candidates);
  if (candidates.empty() || !capability.candidates[0].eligible)
    return std::nullopt;
  for (auto it = records.rbegin(); it != records.rend(); ++it) {
    if (it->topologyIdentity != t.identity ||
        it->executorIdentity != executor || !same(it->workload, w))
      continue;
    validate(*it);
    for (size_t i = 0; i < candidates.size(); ++i)
      if (it->timings[i].empty() == capability.candidates[i].eligible)
        throw std::invalid_argument(
            "peer record does not cover exactly the eligible families");
    auto family = qualifiedPeerFamily(it->timings);
    size_t index = size_t(family);
    PeerSelection result{family, std::move(candidates[index].plan), false,
                         "persisted exact local qualification"};
    result.qualified = true;
    result.localTimings = it->timings;
    result.calibrationIdentity = it->calibrationIdentity;
    result.fromCache = true;
    return result;
  }
  return std::nullopt;
}
}  // namespace uccl_lite

#include "uccl_lite/network_cache.hpp"
#include "uccl_lite/local_plan.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <locale>
#include <numeric>
#include <sstream>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
constexpr size_t limit = 64UL << 20;
constexpr char magic[] = "uccl-lite-network-plans-v1";
constexpr char portfolioMagic[] = "uccl-lite-network-plans-v2";
constexpr char scope[] =
    "eager;host-max-backend-completion;rounds=3;warmups=2;timed=3;abba;guards="
    "full";
void validate(NetworkPlanRecord const& r) {
  auto const& w = r.workload;
  int op = int(w.collective);
  bool direct = r.candidate == NetworkCacheCandidate::DirectExchange;
  size_t ranks = r.layouts.size(), scalar = scalarBytes(w.dtype);
  if (r.candidate != NetworkCacheCandidate::Hierarchy && !direct)
    throw std::invalid_argument("unknown cached network candidate");
  if (op < 0 || op > 2 || !w.bytes || w.bytes > (1UL << 30) || w.inPlace ||
      (w.dtype != ScalarType::Float32 && w.dtype != ScalarType::BFloat16) ||
      w.alignment != scalar || ranks < 2 || ranks > 32 ||
      w.bytes %
          (scalar * (w.collective == Collective::AllReduce ? 1 : ranks)) ||
      r.candidateSamples.size() != 18 || r.baselineSamples.size() != 18)
    throw std::invalid_argument(
        "invalid network cached workload or evidence extent");
  size_t count =
      w.bytes / (scalar * (w.collective == Collective::AllReduce ? 1 : ranks));
  for (size_t rank = 0; rank < ranks; ++rank) {
    auto l = r.layouts[rank];
    if (l.alignment < scalar || l.alignment > 256 ||
        (l.alignment & (l.alignment - 1)) ||
        (l.inPlace && w.collective != Collective::AllReduce &&
         (rank * count * scalar) % l.alignment))
      throw std::invalid_argument("invalid cached rank buffer layout");
  }
  if (direct) {
    if (r.useHierarchy || !r.rankOrder.empty() || !r.rails.empty() ||
        r.configurationIdentities.size() != ranks)
      throw std::invalid_argument("invalid direct cache execution metadata");
    for (auto const& id : r.configurationIdentities)
      if (id.empty() || id.size() > (1UL << 20) ||
          id.find('\0') != std::string::npos)
        throw std::invalid_argument("invalid direct configuration identity");
  } else {
    if (r.useDirect || !r.configurationIdentities.empty() || ranks != 8 ||
        w.dtype != ScalarType::Float32 || w.bytes % 128 ||
        r.rankOrder.size() != 8 || r.rails.size() != 8)
      throw std::invalid_argument("invalid hierarchy cache execution metadata");
    auto order = r.rankOrder;
    std::sort(order.begin(), order.end());
    for (int rank = 0; rank < 8; ++rank)
      if (order[rank] != rank || r.rails[rank] < 0 || r.rails[rank] >= 8)
        throw std::invalid_argument("invalid cached execution rank or rail");
  }
  for (auto const* samples : {&r.candidateSamples, &r.baselineSamples})
    for (double us : *samples)
      if (!std::isfinite(us) || us <= 0)
        throw std::invalid_argument("invalid cached duration");
  bool selected =
      qualifiedLocalBackend(r.baselineSamples, r.candidateSamples,
                            r.candidateCorrect) == LocalBackend::Host;
  if (selected != (direct ? r.useDirect : r.useHierarchy))
    throw std::invalid_argument("cached decision contradicts its evidence");
}
void pathValid(std::string const& path) {
  if (path.empty() || path.find('\0') != std::string::npos)
    throw std::invalid_argument("invalid network cache path");
}
struct Lock {
  int fd = -1;
  explicit Lock(std::string const& path) {
    pathValid(path);
    fd = open((path + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("cannot open network cache lock");
    if (flock(fd, LOCK_EX)) {
      close(fd);
      fd = -1;
      throw std::runtime_error("cannot lock network cache");
    }
  }
  ~Lock() {
    if (fd >= 0) close(fd);
  }
};
void save(std::string const& path, NetworkPlanCache const& cache) {
  auto data = encodeNetworkPlanCache(cache);
  auto pattern = path + ".tmp.XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back(0);
  int fd = mkstemp(name.data());
  if (fd < 0) throw std::runtime_error("cannot create network cache temporary");
  try {
    size_t written = 0;
    while (written < data.size()) {
      auto n = write(fd, data.data() + written, data.size() - written);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("network cache write failed");
      written += size_t(n);
    }
    if (fsync(fd)) throw std::runtime_error("network cache sync failed");
    int status = close(fd);
    fd = -1;
    if (status || std::rename(name.data(), path.c_str()))
      throw std::runtime_error("cannot publish network cache");
  } catch (...) {
    if (fd >= 0) close(fd);
    std::remove(name.data());
    throw;
  }
}
}  // namespace
bool sameNetworkRequest(NetworkPlanRecord const& a,
                        NetworkPlanRecord const& b) {
  if (a.candidate != b.candidate ||
      a.workload.collective != b.workload.collective ||
      a.workload.dtype != b.workload.dtype ||
      a.workload.bytes != b.workload.bytes ||
      a.workload.inPlace != b.workload.inPlace ||
      a.workload.alignment != b.workload.alignment ||
      a.layouts.size() != b.layouts.size())
    return false;
  for (size_t r = 0; r < a.layouts.size(); ++r)
    if (a.layouts[r].inPlace != b.layouts[r].inPlace ||
        a.layouts[r].alignment != b.layouts[r].alignment)
      return false;
  return true;
}
std::string encodeNetworkPlanCache(NetworkPlanCache const& c) {
  if (c.identity.empty() || c.identity.size() > (1UL << 20) ||
      c.records.size() > 4096)
    throw std::invalid_argument("invalid network cache context");
  std::ostringstream out;
  out.imbue(std::locale::classic());
  bool version2 =
      std::any_of(c.records.begin(), c.records.end(), [](auto const& r) {
        return r.candidate == NetworkCacheCandidate::DirectExchange;
      });
  out << (version2 ? portfolioMagic : magic) << '\n'
      << scope << '\n'
      << std::quoted(c.identity) << '\n'
      << std::quoted(c.calibration ? encodeNetworkProfile(*c.calibration)
                                   : std::string{})
      << '\n'
      << c.records.size() << '\n'
      << std::setprecision(17);
  for (size_t i = 0; i < c.records.size(); ++i) {
    auto const& r = c.records[i];
    validate(r);
    for (size_t j = 0; j < i; ++j)
      if (sameNetworkRequest(r, c.records[j]))
        throw std::invalid_argument("duplicate network cache request");
    if (version2)
      out << int(r.candidate) << ' ' << int(r.workload.dtype) << ' ';
    out << int(r.workload.collective) << ' ' << r.workload.bytes << ' '
        << r.candidateCorrect << ' '
        << (r.candidate == NetworkCacheCandidate::DirectExchange
                ? r.useDirect
                : r.useHierarchy);
    if (version2) out << ' ' << r.layouts.size();
    for (auto l : r.layouts) out << ' ' << l.inPlace << ' ' << l.alignment;
    if (r.candidate == NetworkCacheCandidate::DirectExchange) {
      for (auto const& id : r.configurationIdentities)
        out << ' ' << std::quoted(id);
    } else
      for (auto const* values : {&r.rankOrder, &r.rails})
        for (int x : *values) out << ' ' << x;
    for (auto const* values : {&r.candidateSamples, &r.baselineSamples})
      for (double x : *values) out << ' ' << x;
    out << '\n';
  }
  auto data = out.str();
  if (data.size() > limit)
    throw std::invalid_argument("network cache exceeds byte limit");
  return data;
}
NetworkPlanCache decodeNetworkPlanCache(std::string const& data) {
  if (data.size() > limit)
    throw std::invalid_argument("network cache exceeds byte limit");
  std::istringstream in(data);
  in.imbue(std::locale::classic());
  std::string header, timing, profile;
  std::getline(in, header);
  std::getline(in, timing);
  NetworkPlanCache c;
  size_t count = 0;
  bool version2 = header == portfolioMagic;
  if ((!version2 && header != magic) || timing != scope ||
      !(in >> std::quoted(c.identity) >> std::quoted(profile) >> count) ||
      count > 4096)
    throw std::invalid_argument("unsupported or truncated network plan cache");
  if (!profile.empty()) c.calibration = decodeNetworkProfile(profile);
  for (size_t i = 0; i < count; ++i) {
    NetworkPlanRecord r;
    int op = -1, candidate = 0, dtype = int(ScalarType::Float32);
    size_t ranks = 8;
    bool selected = false;
    if (version2 &&
        (!(in >> candidate >> dtype) || candidate < 0 || candidate > 1 ||
         (dtype != int(ScalarType::Float32) &&
          dtype != int(ScalarType::BFloat16))))
      throw std::invalid_argument("invalid cached candidate or dtype");
    if (!(in >> op >> r.workload.bytes >> r.candidateCorrect >> selected) ||
        (version2 && (!(in >> ranks) || ranks < 2 || ranks > 32)))
      throw std::invalid_argument("truncated network cache request");
    r.candidate = NetworkCacheCandidate(candidate);
    r.workload.collective = Collective(op);
    r.workload.dtype = ScalarType(dtype);
    r.workload.alignment = scalarBytes(r.workload.dtype);
    r.layouts.resize(ranks);
    r.candidateSamples.resize(18);
    r.baselineSamples.resize(18);
    for (auto& l : r.layouts)
      if (!(in >> l.inPlace >> l.alignment))
        throw std::invalid_argument("truncated cached layout");
    if (r.candidate == NetworkCacheCandidate::DirectExchange) {
      r.useDirect = selected;
      r.configurationIdentities.resize(ranks);
      for (auto& id : r.configurationIdentities)
        if (!(in >> std::quoted(id)))
          throw std::invalid_argument("truncated direct configuration");
    } else {
      r.useHierarchy = selected;
      r.rankOrder.resize(ranks);
      r.rails.resize(ranks);
      for (auto* values : {&r.rankOrder, &r.rails})
        for (auto& x : *values)
          if (!(in >> x))
            throw std::invalid_argument("truncated cached execution plan");
    }
    for (auto* values : {&r.candidateSamples, &r.baselineSamples})
      for (auto& x : *values)
        if (!(in >> x)) throw std::invalid_argument("truncated cached timing");
    c.records.push_back(std::move(r));
  }
  in >> std::ws;
  if (!in.eof()) throw std::invalid_argument("trailing network cache data");
  (void)encodeNetworkPlanCache(
      c);  // shared validation, including duplicate keys
  return c;
}
std::optional<NetworkPlanCache> loadNetworkPlanCache(std::string const& path) {
  pathValid(path);
  struct stat info {};
  if (stat(path.c_str(), &info)) {
    if (errno == ENOENT) return {};
    throw std::runtime_error("cannot inspect network cache");
  }
  if (!S_ISREG(info.st_mode) || info.st_size < 0 ||
      size_t(info.st_size) > limit)
    throw std::invalid_argument("network cache is not a bounded regular file");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open network cache");
  std::string data(size_t(info.st_size), '\0');
  in.read(data.data(), data.size());
  if (size_t(in.gcount()) != data.size())
    throw std::runtime_error("network cache truncated during read");
  char extra;
  if (in.get(extra) || !in.eof())
    throw std::runtime_error("network cache read extent changed");
  return decodeNetworkPlanCache(data);
}
void updateNetworkPlanCache(std::string const& path,
                            NetworkPlanCache const& context,
                            NetworkPlanRecord const& record) {
  validate(record);
  Lock lock(path);
  auto old = loadNetworkPlanCache(path);
  NetworkPlanCache c =
      old && old->identity == context.identity ? std::move(*old) : context;
  // A direct-only writer may have loaded its context before another writer
  // published hierarchy resources. Absence is not a request to erase them.
  if (context.calibration) c.calibration = context.calibration;
  c.records.erase(std::remove_if(c.records.begin(), c.records.end(),
                                 [&](auto const& r) {
                                   return sameNetworkRequest(r, record);
                                 }),
                  c.records.end());
  c.records.push_back(record);
  if (c.records.size() > 4096)
    c.records.erase(c.records.begin(),
                    c.records.begin() + (c.records.size() - 4096));
  save(path, c);
}
void eraseNetworkPlanCacheRecord(std::string const& path,
                                 std::string const& identity,
                                 NetworkPlanRecord const& record) {
  Lock lock(path);
  auto cache = loadNetworkPlanCache(path);
  if (!cache || cache->identity != identity) return;
  cache->records.erase(
      std::remove_if(
          cache->records.begin(), cache->records.end(),
          [&](auto const& r) { return sameNetworkRequest(r, record); }),
      cache->records.end());
  save(path, *cache);
}
}  // namespace uccl_lite

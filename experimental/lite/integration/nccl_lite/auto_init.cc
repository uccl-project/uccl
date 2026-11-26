#include "auto_init.hpp"
#include "auto_manifest.hpp"
#include <arpa/inet.h>
#include <net/if.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <ifaddrs.h>
#include <unistd.h>

namespace uccl_lite::integration {
namespace {
uint64_t hash(std::string const& text) {
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : text) h = (h ^ c) * 1099511628211ULL;
  return h;
}
void checked(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void checked(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error("native automatic-init exchange failed");
}
template <class T>
T field(ncclConfig_t const* config, size_t offset, T fallback) {
  T value = fallback;
  if (config && config->size >= offset + sizeof(T))
    std::memcpy(&value, reinterpret_cast<char const*>(config) + offset,
                sizeof(T));
  return value;
}
std::string completionAddress() {
  // Use the explicitly selected bootstrap interface, otherwise the OS default
  // IPv4 route. Never reinterpret either NCCL's or MSCCL++'s opaque ID.
  std::string interface;
  if (auto name = std::getenv("MSCCLPP_SOCKET_IFNAME")) interface = name;
  if (interface.empty()) {
    std::ifstream routes("/proc/net/route");
    std::string line;
    std::getline(routes, line);
    while (std::getline(routes, line)) {
      std::istringstream row(line);
      std::string name, destination, gateway;
      unsigned flags = 0;
      if (row >> name >> destination >> gateway >> std::hex >> flags)
        if (destination == "00000000" && (flags & 1)) {
          interface = name;
          break;
        }
    }
  }
  ifaddrs* addresses = nullptr;
  if (getifaddrs(&addresses))
    throw std::runtime_error("bootstrap IPv4 discovery failed");
  std::unique_ptr<ifaddrs, decltype(&freeifaddrs)> owned(addresses,
                                                         freeifaddrs);
  for (auto a = addresses; a; a = a->ifa_next) {
    if (!a->ifa_addr || a->ifa_addr->sa_family != AF_INET ||
        !(a->ifa_flags & IFF_UP) || interface != a->ifa_name)
      continue;
    char text[INET_ADDRSTRLEN];
    auto address = reinterpret_cast<sockaddr_in const*>(a->ifa_addr);
    if (inet_ntop(AF_INET, &address->sin_addr, text, sizeof(text))) return text;
  }
  throw std::runtime_error(
      "automatic completion needs an IPv4 bootstrap interface");
}
struct Wire {
  int rank = 0, ranks = 0, device = 0, pid = 0, ready = 1, supported = 0,
      wanted = 0;
  uint64_t manifest = 0, configuration = 0, implementation = 0;
  char host[256]{}, gpu[16]{}, error[256]{};
  ucclLiteUniqueId id{}, completionId{};
  int completionReady = 0;
  char completionRoot[INET_ADDRSTRLEN]{};
};
struct Exchange {
  AutoInitRecord record;
  Wire local;
  AutoManifest manifest;
  std::vector<Wire> peers;
  void* buffer = nullptr;
  cudaStream_t stream = nullptr;
  ~Exchange() {
    int old = -1;
    cudaGetDevice(&old);
    cudaSetDevice(local.device);
    if (stream) cudaStreamSynchronize(stream);
    if (buffer) cudaFree(buffer);
    if (stream) cudaStreamDestroy(stream);
    if (old >= 0 && old != local.device) cudaSetDevice(old);
  }
};
}  // namespace
bool automaticInitEnabled() {
  auto flag = std::getenv("UCCL_LITE_AUTO_INIT");
  return flag && !std::strcmp(flag, "1");
}
AutoInitRecord autoInitRecord(ncclComm_t comm, ncclConfig_t const* config,
                              bool grouped) {
  AutoInitRecord r;
  r.comm = comm;
  r.grouped = grouped;
  std::ostringstream s;
  s << "native-init-config-v1";
  int blocking =
      field(config, offsetof(ncclConfig_t, blocking), NCCL_CONFIG_UNDEF_INT);
  if (auto env = std::getenv("NCCL_COMM_BLOCKING"))
    if (!std::strcmp(env, "0") || !std::strcmp(env, "1"))
      blocking = env[0] - '0';
  r.blocking = blocking != 0;
  if (config) s << ";size=" << config->size << ";version=" << config->version;
#define INTEGER(name)      \
  s << ';' << #name << '=' \
    << field(config, offsetof(ncclConfig_t, name), NCCL_CONFIG_UNDEF_INT)
  INTEGER(blocking);
  INTEGER(cgaClusterSize);
  INTEGER(minCTAs);
  INTEGER(maxCTAs);
  INTEGER(splitShare);
  INTEGER(trafficClass);
  INTEGER(collnetEnable);
  INTEGER(CTAPolicy);
  INTEGER(shrinkShare);
  INTEGER(nvlsCTAs);
  INTEGER(nChannelsPerNetPeer);
  INTEGER(nvlinkCentricSched);
#undef INTEGER
  for (auto offset :
       {offsetof(ncclConfig_t, netName), offsetof(ncclConfig_t, commName)}) {
    auto text = field<char const*>(
        config, offset, reinterpret_cast<char const*>(NCCL_CONFIG_UNDEF_PTR));
    s << ';' << offset << '=';
    if (text && text != reinterpret_cast<char const*>(NCCL_CONFIG_UNDEF_PTR))
      s << std::strlen(text) << ':' << text;
    else
      s << (text ? "undefined" : "null");
  }
  r.configuration = s.str();
  return r;
}
std::vector<AutoInitResult> prepareNativeInitializations(
    std::vector<AutoInitRecord> const& records, AutoInitApi const& api) {
  std::vector<AutoInitResult> results;
  std::vector<std::unique_ptr<Exchange>> exchanges;
  int previous = -1;
  checked(cudaGetDevice(&previous));
  struct Restore {
    int device;
    ~Restore() { cudaSetDevice(device); }
  } restore{previous};
  for (auto const& record : records) {
    if (!record.comm) continue;
    if (!record.blocking) {
      results.push_back({record.comm, ncclSuccess,
                         "nonblocking initialization remains native"});
      continue;
    }
    auto e = std::make_unique<Exchange>();
    e->record = record;
    checked(api.count(record.comm, &e->local.ranks));
    checked(api.rank(record.comm, &e->local.rank));
    checked(api.device(record.comm, &e->local.device));
    if (e->local.ranks < 1 || e->local.ranks > 32) {
      results.push_back(
          {record.comm, ncclSuccess,
           "rank extent outside prepared runtime; native retained"});
      continue;
    }
    e->local.pid = getpid();
    e->local.configuration = hash(record.configuration);
    e->local.implementation = hash(api.implementation);
    e->local.supported = !record.grouped && api.available();
    auto path = std::getenv("UCCL_LITE_AUTO_MANIFEST");
    e->local.wanted = path && *path;
    checked(cudaSetDevice(e->local.device));
    try {
      // Independent of manifest validity: a caller may immediately abort the
      // native communicator after any initialization result. No rank may
      // return while another still has the internal native exchange in flight.
      if (!e->local.rank) {
        checked(api.uniqueId(&e->local.completionId));
        auto address = completionAddress();
        std::snprintf(e->local.completionRoot, sizeof(e->local.completionRoot),
                      "%s", address.c_str());
        e->local.completionReady = 1;
      }
      if (gethostname(e->local.host, sizeof(e->local.host) - 1))
        throw std::runtime_error("automatic hostname unavailable");
      cudaDeviceProp property{};
      checked(cudaGetDeviceProperties(&property, e->local.device));
      std::memcpy(e->local.gpu, property.uuid.bytes, 16);
      if (e->local.supported && e->local.wanted) {
        e->manifest = loadAutoManifest(path);
        e->local.manifest = hash(e->manifest.contents);
        if (!e->local.rank) checked(api.uniqueId(&e->local.id));
      }
    } catch (std::exception const& error) {
      e->local.ready = 0;
      std::snprintf(e->local.error, sizeof(e->local.error), "%s", error.what());
    }
    e->peers.resize(e->local.ranks);
    checked(cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking));
    checked(cudaMalloc(&e->buffer, sizeof(Wire) * e->peers.size()));
    checked(cudaMemcpyAsync(
        static_cast<char*>(e->buffer) + size_t(e->local.rank) * sizeof(Wire),
        &e->local, sizeof(Wire), cudaMemcpyHostToDevice, e->stream));
    exchanges.push_back(std::move(e));
  }
  // Batch all local ranks/communicators after native group initialization has
  // completed. Never block on the first local rank before enqueueing its peers.
  if (!exchanges.empty()) {
    checked(api.groupStart());
    ncclResult_t status = ncclSuccess;
    for (auto const& e : exchanges) {
      checked(cudaSetDevice(e->local.device));
      auto result = api.gather(
          static_cast<char*>(e->buffer) + size_t(e->local.rank) * sizeof(Wire),
          e->buffer, sizeof(Wire), ncclUint8, e->record.comm, e->stream);
      if (result != ncclSuccess) status = result;
    }
    auto end = api.groupEnd();
    checked(status);
    checked(end);
    for (auto& e : exchanges) {
      checked(cudaSetDevice(e->local.device));
      checked(cudaStreamSynchronize(e->stream));
      checked(cudaMemcpy(e->peers.data(), e->buffer,
                         sizeof(Wire) * e->peers.size(),
                         cudaMemcpyDeviceToHost));
      checked(cudaFree(e->buffer));
      e->buffer = nullptr;
      checked(cudaStreamDestroy(e->stream));
      e->stream = nullptr;
    }
  }
  std::vector<AutoInitCompletion> completions;
  for (auto const& e : exchanges)
    if (e->peers[0].completionReady)
      completions.push_back({e->local.rank, e->local.ranks,
                             e->peers[0].completionId,
                             e->peers[0].completionRoot});
  // Batch CPU rendezvous too: a process may own several ranks initialized
  // inside one native group. None can wait before its local peers start.
  api.complete(completions);
  for (auto& e : exchanges) {
    bool supported = true, requested = false, ready = true, mismatch = false;
    std::ostringstream configuration;
    configuration << "automatic-native-init-v1;abi=" << api.implementation;
    for (size_t rank = 0; rank < e->peers.size(); ++rank) {
      auto const& p = e->peers[rank];
      ready &= p.ready;
      supported &= p.supported;
      mismatch |= p.rank != int(rank) || p.ranks != e->local.ranks ||
                  p.implementation != e->local.implementation;
      requested |= p.wanted;
      configuration << ';' << p.configuration;
      for (size_t earlier = 0; earlier < rank; ++earlier)
        if (!std::strcmp(p.host, e->peers[earlier].host))
          supported &= p.pid != e->peers[earlier].pid &&
                       std::memcmp(p.gpu, e->peers[earlier].gpu, 16) != 0;
    }
    if (!supported) {
      results.push_back(
          {e->record.comm, ncclSuccess,
           "grouped/multiple-rank/busy initialization remains native"});
      continue;
    }
    if (!requested) {
      results.push_back({e->record.comm, ncclSuccess,
                         "no automatic manifest requested; native retained"});
      continue;
    }
    for (auto const& p : e->peers)
      mismatch |= !p.wanted || p.manifest != e->local.manifest;
    if (!ready || mismatch) {
      std::string reason = "automatic manifest/implementation agreement failed";
      for (auto const& p : e->peers)
        if (!p.ready) {
          reason += "; ";
          reason += p.error;
        }
      results.push_back({e->record.comm, ncclInvalidUsage, reason});
      continue;
    }
    std::vector<AutoRequest> materialized;
    try {
      materialized = materializeAutoManifest(e->manifest, e->local.ranks);
    } catch (std::exception const& error) {
      results.push_back({e->record.comm, ncclInvalidArgument, error.what()});
      continue;
    }
    if (materialized.empty()) {
      results.push_back(
          {e->record.comm, ncclSuccess,
           "no exact request fits this rank extent/capacity; native retained"});
      continue;
    }
    std::vector<ucclLiteRequest> requests;
    for (auto const& r : materialized)
      requests.push_back(
          {r.operation, r.count, r.layouts.data(), r.layouts.size(), 0});
    auto identity = configuration.str();
    ucclLiteOptions options{};
    options.structSize = sizeof(options);
    options.version = 1;
    options.flags = e->manifest.flags;
    options.capacityBytes = e->manifest.capacity;
    options.dtype = e->manifest.dtype;
    options.nativeLibrary = api.library.c_str();
    options.configurationIdentity = identity.c_str();
    checked(cudaSetDevice(e->local.device));
    auto result = api.prepare(e->record.comm, &e->peers[0].id, &options,
                              requests.data(), requests.size());
    results.push_back({e->record.comm, result,
                       result == ncclSuccess
                           ? "automatic exact-workload preparation completed"
                           : "automatic preparation failed; caller must handle "
                             "native initialization error"});
  }
  return results;
}
}  // namespace uccl_lite::integration

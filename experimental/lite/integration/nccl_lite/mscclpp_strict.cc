// Strict NCCL ABI compatibility: no native NCCL dependency or fallback.
#include "uccl_lite/bootstrap/control.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include "uccl_lite/topology.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <unistd.h>

namespace {
using uccl_lite::runtime::BackendRegistration;
using uccl_lite::runtime::BackendSpec;
ncclDataType_t dataType(BackendSpec const& spec) {
  return spec.dtype == uccl_lite::ScalarType::BFloat16 ? ncclBfloat16
                                                       : ncclFloat32;
}
thread_local std::string lastError;
thread_local int groupDepth = 0;
std::atomic<bool> live{false};
ncclResult_t fail(char const* message, ncclResult_t code = ncclInvalidUsage) {
  lastError = message;
  std::fprintf(stderr, "UCCL Lite strict: %s (native NCCL is forbidden)\n",
               message);
  return code;
}
bool nativeLoaded() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    auto slash = line.rfind('/');
    if (slash != std::string::npos &&
        line.substr(slash + 1).rfind("libnccl.so", 0) == 0)
      return true;
  }
  return false;
}
uint64_t hash(std::string const& data) {
  uint64_t value = 14695981039346656037ULL;
  for (unsigned char c : data) value = (value ^ c) * 1099511628211ULL;
  return value;
}
std::string uuid(int device) {
  cudaDeviceProp property{};
  if (cudaGetDeviceProperties(&property, device) != cudaSuccess) return {};
  char text[33];
  for (int i = 0; i < 16; ++i)
    std::snprintf(text + 2 * i, 3, "%02x",
                  static_cast<unsigned char>(property.uuid.bytes[i]));
  return text;
}
struct Policy {
  std::map<std::pair<int, size_t>, bool> shapes;
  std::vector<uccl_lite::runtime::LocalRequest> requests;
  BackendRegistration const* backend = nullptr;
  BackendSpec layout;
  std::vector<std::string> hosts, identities, uuids;
  std::vector<int> rankNodes, rankDevices;
  uint64_t digest = 0;
  size_t capacity = 0;
  int node = -1;
  bool rankUuids = false;
};
Policy policy(int ownRank) {
  return [ownRank] {
    Policy result;
    auto path = std::getenv("UCCL_LITE_MSCCLPP_PROFILE");
    auto verified = std::getenv("UCCL_LITE_MSCCLPP_IDENTITY");
    auto candidate = std::getenv("UCCL_LITE_MSCCLPP_CANDIDATE");
    // No implicit performance admission. This build accepts only explicit
    // candidate experiments with freshly matched topology/software identity.
    if (!path || !verified || !candidate || std::strcmp(candidate, "1"))
      return result;
    std::ifstream file(path);
    std::string data((std::istreambuf_iterator<char>(file)), {});
    std::istringstream input(data);
    std::string magic;
    if (!(input >> magic)) return Policy{};
    bool prepared = magic == "UCCL_LITE_RUNTIME_PROFILE_V2";
    bool explicitRanks = magic == "UCCL_LITE_RUNTIME_PROFILE_V3";
    result.rankUuids = magic == "UCCL_LITE_RUNTIME_PROFILE_V4";
    if (magic == "UCCL_LITE_RUNTIME_PROFILE_V1" || prepared || explicitRanks ||
        result.rankUuids) {
      std::string provider;
      int hosts = 0;
      if (!(input >> provider >> hosts) || hosts < 1 || hosts > 32)
        return Policy{};
      std::vector<int> layout(hosts);
      int ranks = 0;
      for (auto& n : layout) {
        if (!(input >> n) || n < 1 || n > 32) return Policy{};
        ranks += n;
      }
      if (ranks > uccl_lite::bootstrap::HostControl::maxRanks) return Policy{};
      for (auto const& backend : uccl_lite::runtime::availableBackends())
        if (provider == backend.spec.name && backend.spec.acceptsLayout(layout))
          result.backend = &backend;
      if (!result.backend) return Policy{};
      if (result.rankUuids && !result.backend->spec.rankUuidProfiles)
        return Policy{};
      if (explicitRanks || result.rankUuids)
        prepared = result.backend->spec.automatic;
      if (result.backend->spec.automatic != prepared) return Policy{};
      result.layout = result.backend->spec;
      result.layout.ranksPerHost = std::move(layout);
    } else {
      for (auto const& backend : uccl_lite::runtime::availableBackends())
        if (magic == backend.spec.legacyProfileMagic) result.backend = &backend;
      if (!result.backend) return Policy{};
      result.layout = result.backend->spec;
    }
    auto const& spec = result.layout;
    result.rankNodes.resize(spec.ranks());
    result.rankDevices.resize(spec.ranks());
    if (result.rankUuids) {
      std::string tag;
      int count = 0;
      if (!(input >> tag >> count) || tag != "participants" ||
          count != spec.ranks() || ownRank >= count)
        return Policy{};
      result.hosts.resize(spec.ranksPerHost.size());
      result.uuids.resize(count);
      result.identities.resize(count);
      std::vector<int> populations(spec.ranksPerHost.size(), 0);
      std::set<std::pair<int, std::string>> unique;
      char hostname[256]{};
      if (gethostname(hostname, sizeof(hostname) - 1)) return Policy{};
      int driver = 0, runtime = 0;
      if (cudaDriverGetVersion(&driver) != cudaSuccess ||
          cudaRuntimeGetVersion(&runtime) != cudaSuccess)
        return Policy{};
      for (int rank = 0; rank < count; ++rank) {
        int node, d, r, v;
        std::string host, gpu, identity;
        if (!(input >> node >> host >> identity >> d >> r >> v >> gpu) ||
            node < 0 || node >= int(populations.size()) || gpu.size() != 32 ||
            gpu.find_first_not_of("0123456789abcdef") != std::string::npos ||
            !unique.emplace(node, gpu).second)
          return Policy{};
        if (!result.hosts[node].empty() && result.hosts[node] != host)
          return Policy{};
        result.hosts[node] = host;
        ++populations[node];
        result.rankNodes[rank] = node;
        result.rankDevices[rank] = -1;
        result.uuids[rank] = gpu;
        result.identities[rank] = identity;
        if (rank == ownRank && host == hostname && identity == verified &&
            d == driver && r == runtime && v == NCCL_VERSION_CODE)
          result.node = node;
      }
      if (populations != spec.ranksPerHost || result.node < 0 ||
          std::set<std::string>(result.hosts.begin(), result.hosts.end())
                  .size() != result.hosts.size())
        return Policy{};
      int visible = 0;
      if (cudaGetDeviceCount(&visible) != cudaSuccess) return Policy{};
      for (int device = 0; device < visible; ++device) {
        auto gpu = uuid(device);
        for (int rank = 0; rank < count; ++rank)
          if (result.rankNodes[rank] == result.node &&
              result.uuids[rank] == gpu)
            result.rankDevices[rank] = device;
      }
    } else {
      if (explicitRanks) {
        std::string tag;
        int ranks = 0;
        if (!(input >> tag >> ranks) || tag != "ranks" || ranks != spec.ranks())
          return Policy{};
        std::set<std::pair<int, int>> participants;
        for (int rank = 0; rank < ranks; ++rank) {
          int node, device;
          if (!(input >> node >> device) || node < 0 ||
              node >= int(spec.ranksPerHost.size()) || device < 0 ||
              device >= spec.ranksPerHost[node] ||
              !participants.emplace(node, device).second)
            return Policy{};
          result.rankNodes[rank] = node;
          result.rankDevices[rank] = device;
        }
      } else {
        for (int rank = 0; rank < spec.ranks(); ++rank) {
          result.rankNodes[rank] = spec.hostForRank(rank);
          result.rankDevices[rank] = spec.localRank(rank);
        }
      }
      result.hosts.resize(spec.ranksPerHost.size());
      result.identities.resize(spec.ranksPerHost.size());
      result.uuids.resize(spec.ranks());
      char hostname[256]{};
      gethostname(hostname, sizeof(hostname) - 1);
      int driver = 0, runtime = 0, nccl = 0;
      cudaDriverGetVersion(&driver);
      cudaRuntimeGetVersion(&runtime);
      nccl = NCCL_VERSION_CODE;
      int offset = 0;
      for (size_t n = 0; n < spec.ranksPerHost.size(); ++n) {
        int d, r, v;
        if (!(input >> result.hosts[n] >> result.identities[n] >> d >> r >> v))
          return Policy{};
        for (int i = 0; i < spec.ranksPerHost[n]; ++i)
          if (!(input >> result.uuids[offset + i])) return Policy{};
        offset += spec.ranksPerHost[n];
        if (result.hosts[n] == hostname && result.identities[n] == verified &&
            driver == d && runtime == r && nccl == v)
          result.node = n;
      }
      if (std::set<std::string>(result.hosts.begin(), result.hosts.end())
                  .size() != result.hosts.size() ||
          result.node < 0)
        return Policy{};
    }
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      std::istringstream row(line);
      int op;
      size_t count;
      std::string extra;
      if (!(row >> op >> count) || op < 1 || op > 3 || !count) return Policy{};
      size_t scale = spec.elementBytes() * (op == 3 ? 1 : spec.ranks());
      if (count > (1UL << 30) / scale ||
          (!spec.extentMultiple || (count * scale) % spec.extentMultiple) ||
          !result.shapes.emplace(std::make_pair(op, count), true).second)
        return Policy{};
      if (prepared) {
        uccl_lite::runtime::LocalRequest request;
        request.operation = uccl_lite::NetworkOperation(op);
        request.count = count;
        for (int rank = 0; rank < spec.ranks(); ++rank) {
          unsigned encoded = 0;
          if (!(row >> encoded) || encoded < 4 || encoded > 513)
            return Policy{};
          unsigned alignment = encoded / 2;
          bool inplace = encoded & 1;
          if (alignment < unsigned(spec.elementBytes()) ||
              (alignment & (alignment - 1)) ||
              (inplace && op != 3 &&
               (size_t(rank) * count * spec.elementBytes()) % alignment))
            return Policy{};
          request.layouts.push_back({inplace, alignment});
        }
        result.requests.push_back(std::move(request));
      }
      if (row >> extra) return Policy{};
      result.capacity = std::max(result.capacity, count * scale);
    }
    result.digest = hash(data);
    return result;
  }();
}
}  // namespace

// Own communicator handle; no native NCCL object is created or inspected.
struct ncclComm {
  int rank, device, ranks;
  Policy policy;
  bool finalized = false;
  ncclResult_t error = ncclSuccess;
  std::mutex mutex;
  std::unique_ptr<uccl_lite::runtime::CollectiveBackend> transport;
  std::map<std::pair<int, size_t>, uint64_t> counts;
};

extern "C" {
// Read-only discovery fingerprint, including GPU-visible ordering, PCIe paths,
// active NIC ports and bandwidth bounds. No transport or worker is created.
char const* uccl_lite_nccl_runtime_topology() {
  static thread_local std::string snapshot;
  try {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
      throw std::runtime_error("CUDA device discovery failed");
    std::vector<int> devices;
    for (int i = 0; i < count; ++i) devices.push_back(i);
    snapshot = uccl_lite::inspectTopology(devices).hardwareIdentity;
    return snapshot.c_str();
  } catch (std::exception const& error) {
    lastError = error.what();
    return nullptr;
  }
}
// Capability discovery is CPU-only and does not construct any transport.
char const* uccl_lite_nccl_runtime_backends() {
  static std::string catalog = [] {
    std::ostringstream text;
    text << '[';
    bool first = true;
    for (auto const& backend : uccl_lite::runtime::availableBackends()) {
      auto const& spec = backend.spec;
      if (!first) text << ',';
      first = false;
      text << "{\"name\":\"" << spec.name << "\",\"profile\":\""
           << spec.legacyProfileMagic << "\",\"dtype\":\""
           << (spec.dtype == uccl_lite::ScalarType::BFloat16 ? "bfloat16"
                                                             : "float32")
           << "\",\"ranks_per_host\":[";
      for (size_t i = 0; i < spec.ranksPerHost.size(); ++i) {
        if (i) text << ',';
        text << spec.ranksPerHost[i];
      }
      text << "],\"min_local_ranks\":" << spec.minLocalRanks
           << ",\"max_local_ranks\":" << spec.maxLocalRanks
           << ",\"min_hosts\":" << spec.minHosts
           << ",\"max_hosts\":" << spec.maxHosts
           << ",\"min_total_ranks\":" << spec.minTotalRanks
           << ",\"max_total_ranks\":" << spec.maxTotalRanks
           << ",\"extent_multiple\":" << spec.extentMultiple
           << ",\"peer_access\":" << (spec.peerAccess ? "true" : "false")
           << ",\"mapped_host\":" << (spec.mappedHost ? "true" : "false")
           << ",\"profile_versions\":[" << (spec.automatic ? 2 : 1) << ",3"
           << (spec.rankUuidProfiles ? ",4]" : "]")
           << ",\"automatic\":" << (spec.automatic ? "true" : "false") << '}';
    }
    text << ']';
    return text.str();
  }();
  return catalog.c_str();
}
ncclResult_t ncclGetVersion(int* version) {
  if (!version) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;  // ABI compatibility version, not native NCCL.
  return ncclSuccess;
}
char const* ncclGetErrorString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return "UCCL Lite success";
    case ncclInvalidArgument:
      return "UCCL Lite invalid argument";
    case ncclInvalidUsage:
      return "UCCL Lite unsupported usage (fallback forbidden)";
    case ncclUnhandledCudaError:
      return "UCCL Lite CUDA error";
    default:
      return "UCCL Lite error";
  }
}
char const* ncclGetLastError(ncclComm_t) { return lastError.c_str(); }
ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
  if (!id) return ncclInvalidArgument;
  try {
    std::ifstream random("/dev/urandom", std::ios::binary);
    if (!random.read(reinterpret_cast<char*>(id), sizeof(*id)))
      return fail("cannot create communicator unique ID", ncclSystemError);
    return ncclSuccess;
  } catch (std::exception const& error) {
    return fail(error.what(), ncclSystemError);
  }
}
ncclResult_t ncclCommInitRank(ncclComm_t* output, int ranks, ncclUniqueId id,
                              int rank) {
  if (!output || ranks < 1 ||
      ranks > uccl_lite::bootstrap::HostControl::maxRanks || rank < 0 ||
      rank >= ranks)
    return fail("invalid communicator extent, one GPU per process",
                ncclInvalidArgument);
  *output = nullptr;
  if (nativeLoaded()) return fail("native NCCL library is loaded");
  if (live.exchange(true))
    return fail("only one live communicator per process is supported");
  try {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cudaGetDevice failed");
    auto p = policy(rank);
    struct Vote {
      uint64_t digest;
      int valid, node, device, pid, peers, mapped;
      char gpu[33]{};
    };
    bool shape = p.backend && ranks == p.layout.ranks();
    int expectedUuid = p.rankUuids ? rank : 0;
    if (shape && !p.rankUuids) {
      for (int host = 0; host < p.rankNodes[rank]; ++host)
        expectedUuid += p.layout.ranksPerHost[host];
      expectedUuid += p.rankDevices[rank];
    }
    bool valid = shape && !p.shapes.empty() && p.node == p.rankNodes[rank] &&
                 device == p.rankDevices[rank] &&
                 uuid(device) == p.uuids[expectedUuid];
    int peerReady = 1, mapped = 0;
    if (valid) {
      cudaDeviceProp property{};
      if (cudaGetDeviceProperties(&property, device) != cudaSuccess)
        valid = false;
      mapped = property.canMapHostMemory;
      if (p.layout.peerAccess)
        for (int other = 0; other < p.layout.ranksPerHost[p.node]; ++other) {
          int access = 1;
          if (other != device &&
              cudaDeviceCanAccessPeer(&access, device, other) != cudaSuccess) {
            cudaGetLastError();
            access = 0;
          }
          peerReady &= access;
        }
    }
    Vote local{p.digest,      int(valid), p.node, device,
               int(getpid()), peerReady,  mapped};
    auto actualGpu = uuid(device);
    std::snprintf(local.gpu, sizeof(local.gpu), "%s", actualGpu.c_str());
    std::vector<Vote> votes(ranks);
    uccl_lite::NetworkId backendId{};
    {
      uccl_lite::bootstrap::HostControl control(
          std::getenv("UCCL_LITE_MSCCLPP_CONTROL_ROOT"), &id, sizeof(id), rank,
          ranks);
      control.allGather(&local, votes.data(), sizeof(local));
      if (!shape)
        throw std::runtime_error("profile/backend communicator mismatch");
      auto const& spec = p.layout;
      for (int i = 0; i < ranks; ++i) {
        if (!votes[i].valid || votes[i].digest != local.digest ||
            votes[i].node != p.rankNodes[i] ||
            (!p.rankUuids && votes[i].device != p.rankDevices[i]) ||
            (p.rankUuids && p.uuids[i] != votes[i].gpu) ||
            (spec.peerAccess && !votes[i].peers) ||
            (spec.mappedHost && !votes[i].mapped))
          throw std::runtime_error(
              "profile/topology/capability consensus failed");
        for (int j = 0; j < i; ++j)
          if (votes[j].node == votes[i].node && votes[i].pid == votes[j].pid)
            throw std::runtime_error("multiple ranks per process");
      }
      int ready = 1;
      if (rank == 0) {
        try {
          backendId = p.backend->uniqueId();
        } catch (std::exception const& error) {
          std::fprintf(stderr, "UCCL Lite backend ID: %s\n", error.what());
          ready = 0;
        }
      }
      control.broadcast(&ready, sizeof(ready));
      if (!ready) throw std::runtime_error("backend bootstrap ID unavailable");
      control.broadcast(backendId.data(), backendId.size());
    }
    auto const& spec = p.layout;
    auto comm = std::make_unique<ncclComm>();
    comm->rank = rank;
    comm->device = device;
    comm->ranks = ranks;
    comm->policy = p;
    uccl_lite::runtime::BackendContext context;
    context.id = backendId;
    context.rank = rank;
    context.ranks = ranks;
    context.device = device;
    context.capacity = p.capacity;
    context.requests = p.requests;
    if (p.rankUuids) {
      for (int i = 0; i < ranks; ++i)
        if (p.rankNodes[i] == p.node && p.rankDevices[i] >= 0)
          context.localDevices.push_back(p.rankDevices[i]);
      std::sort(context.localDevices.begin(), context.localDevices.end());
    } else if (spec.ranksPerHost.size() == 1) {
      // Local providers bind their peer table to logical rank order.
      for (auto const& vote : votes)
        if (vote.node == p.node) context.localDevices.push_back(vote.device);
    } else {
      // Distributed providers resolve logical ownership from live metadata.
      // Keep the hardware inventory stable when application ranks move, so
      // physical resource measurements retain their original device view.
      for (int device = 0; device < spec.ranksPerHost[p.node]; ++device)
        context.localDevices.push_back(device);
    }
    comm->transport = p.backend->create(context);
    comm->transport->prepare();
    std::fprintf(stderr, "UCCL Lite backend: %s\n", spec.name.c_str());
    *output = comm.release();
    std::fprintf(stderr, "UCCL Lite strict: rank=%d native_nccl_loaded=0\n",
                 rank);
    return ncclSuccess;
  } catch (std::exception const& error) {
    live = false;
    return fail(error.what(), ncclSystemError);
  }
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int ranks,
                                    ncclUniqueId id, int rank,
                                    ncclConfig_t* config) {
  if (config && config->blocking == 0)
    return fail("nonblocking initialization unsupported");
  return ncclCommInitRank(comm, ranks, id, rank);
}
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* error) {
  if (!comm || !error) return ncclInvalidArgument;
  *error = comm->error;
  return ncclSuccess;
}
ncclResult_t ncclCommCount(ncclComm_t comm, int* count) {
  if (!comm || !count) return ncclInvalidArgument;
  *count = comm->ranks;
  return ncclSuccess;
}
ncclResult_t ncclCommUserRank(ncclComm_t comm, int* rank) {
  if (!comm || !rank) return ncclInvalidArgument;
  *rank = comm->rank;
  return ncclSuccess;
}
ncclResult_t ncclCommCuDevice(ncclComm_t comm, int* device) {
  if (!comm || !device) return ncclInvalidArgument;
  *device = comm->device;
  return ncclSuccess;
}
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  if (!comm) return ncclInvalidArgument;
  if (cudaDeviceSynchronize() != cudaSuccess) return ncclUnhandledCudaError;
  try {
    comm->transport->synchronize();
  } catch (std::exception const& error) {
    std::fprintf(stderr, "UCCL Lite finalize: %s\n", error.what());
    comm->error = ncclSystemError;
    return comm->error;
  }
  comm->finalized = true;
  return ncclSuccess;
}
// Diagnostic completion boundary for comparisons with the original standalone
// harness, which includes internal send retirement. Ordinary NCCL callers do
// not need this extension; it neither finalizes nor changes routing policy.
ncclResult_t uccl_lite_nccl_synchronize(ncclComm_t comm) {
  if (!comm) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(comm->mutex);
  if (comm->error != ncclSuccess) return comm->error;
  try {
    comm->transport->synchronize();
    return ncclSuccess;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "UCCL Lite diagnostic synchronize: %s\n",
                 error.what());
    comm->error = ncclSystemError;
    return comm->error;
  }
}
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclInvalidArgument;
  for (auto const& [shape, count] : comm->counts)
    std::fprintf(
        stderr, "UCCL Lite strict route: rank=%d op=%d bytes=%zu calls=%llu\n",
        comm->rank, shape.first, shape.second, (unsigned long long)count);
  std::fprintf(stderr, "UCCL Lite strict: rank=%d fallback=0\n", comm->rank);
  delete comm;
  live = false;
  return ncclSuccess;
}
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  // Do not wait on a lost peer. Retain active transports until process exit.
  if (!comm) return ncclInvalidArgument;
  comm->finalized = true;
  if (comm->error == ncclSuccess && comm->transport->idle()) {
    delete comm;
    live = false;
  }
  return ncclSuccess;
}
ncclResult_t ncclGroupStart() {
  ++groupDepth;
  return ncclSuccess;
}
ncclResult_t ncclGroupEnd() {
  if (!groupDepth) return ncclInvalidUsage;
  --groupDepth;
  return ncclSuccess;
}
ncclResult_t ncclMemAlloc(void** pointer, size_t bytes) {
  return cudaMalloc(pointer, bytes) == cudaSuccess ? ncclSuccess
                                                   : ncclUnhandledCudaError;
}
ncclResult_t ncclMemFree(void* pointer) {
  return cudaFree(pointer) == cudaSuccess ? ncclSuccess
                                          : ncclUnhandledCudaError;
}
ncclResult_t ncclCommRegister(ncclComm_t, void*, size_t, void**) {
  return fail("client memory registration unsupported");
}
ncclResult_t ncclCommDeregister(ncclComm_t, void*) {
  return fail("client memory deregistration unsupported");
}
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t*, void*, ncclDataType_t,
                                      ncclScalarResidence_t, ncclComm_t) {
  return fail("custom reduction unsupported");
}
ncclResult_t ncclRedOpDestroy(ncclRedOp_t, ncclComm_t) {
  return fail("custom reduction unsupported");
}
}

namespace {
ncclResult_t invoke(ncclComm_t comm, int op, void const* input, void* output,
                    size_t count, ncclDataType_t dtype, ncclRedOp_t reduction,
                    cudaStream_t stream) {
  if (!comm || comm->finalized) return ncclInvalidArgument;
  if (!count) return ncclSuccess;
  if (groupDepth) return fail("grouped collectives unsupported");
  if (dtype != dataType(comm->policy.layout) ||
      (op != 1 && reduction != ncclSum) ||
      !comm->policy.shapes.count({op, count}))
    return fail(
        "collective dtype/reduction/extent is outside the strict profile");
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
    return ncclUnhandledCudaError;
  if (capture != cudaStreamCaptureStatusNone)
    return fail("external graph capture unsupported by the selected backend");
  std::lock_guard<std::mutex> guard(comm->mutex);
  try {
    if (!comm->transport->initialized())
      return fail("communicator was not prepared");
    comm->transport->enqueue(static_cast<uccl_lite::NetworkOperation>(op),
                             input, output, count, stream);
    size_t scale =
        comm->policy.layout.elementBytes() * (op == 3 ? 1 : comm->ranks);
    ++comm->counts[{op, count * scale}];
    return ncclSuccess;
  } catch (std::exception const& error) {
    comm->error = ncclSystemError;
    return fail(error.what(), ncclSystemError);
  }
}
}  // namespace
extern "C" {
ncclResult_t ncclAllGather(void const* in, void* out, size_t n,
                           ncclDataType_t type, ncclComm_t c, cudaStream_t s) {
  return invoke(c, 1, in, out, n, type, ncclSum, s);
}
ncclResult_t ncclReduceScatter(void const* in, void* out, size_t n,
                               ncclDataType_t type, ncclRedOp_t op,
                               ncclComm_t c, cudaStream_t s) {
  return invoke(c, 2, in, out, n, type, op, s);
}
ncclResult_t ncclAllReduce(void const* in, void* out, size_t n,
                           ncclDataType_t type, ncclRedOp_t op, ncclComm_t c,
                           cudaStream_t s) {
  return invoke(c, 3, in, out, n, type, op, s);
}
ncclResult_t ncclBroadcast(void const*, void*, size_t, ncclDataType_t, int,
                           ncclComm_t, cudaStream_t) {
  return fail("Broadcast unsupported");
}
ncclResult_t ncclBcast(void*, size_t, ncclDataType_t, int, ncclComm_t,
                       cudaStream_t) {
  return fail("Bcast unsupported");
}
ncclResult_t ncclReduce(void const*, void*, size_t, ncclDataType_t, ncclRedOp_t,
                        int, ncclComm_t, cudaStream_t) {
  return fail("Reduce unsupported");
}
ncclResult_t ncclSend(void const*, size_t, ncclDataType_t, int, ncclComm_t,
                      cudaStream_t) {
  return fail("Send unsupported");
}
ncclResult_t ncclRecv(void*, size_t, ncclDataType_t, int, ncclComm_t,
                      cudaStream_t) {
  return fail("Recv unsupported");
}
ncclResult_t ncclCommSplit(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*) {
  return fail("CommSplit unsupported");
}
}

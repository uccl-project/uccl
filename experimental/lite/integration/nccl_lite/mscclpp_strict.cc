// Strict NCCL ABI compatibility: no native NCCL dependency or fallback.
#include "host_control.h"
#ifdef UCCL_LITE_NCCL_LOCAL8
#include "local_strict.h"
#else
#include "uccl_lite/collectives.h"
#endif
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
#include <sstream>
#include <string>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <unistd.h>

namespace {
#ifdef UCCL_LITE_NCCL_LOCAL8
using Backend = uccl_lite::nccl_detail::LocalCollectives;
using BackendOperation = int;
using BackendId = Backend::Id;
constexpr int kHosts = 1, kLocalRanks = 8, kElementBytes = 2;
constexpr ncclDataType_t kDtype = ncclBfloat16;
constexpr char kProfileMagic[] = "UCCL_LITE_LOCAL8_STRICT_V1";
#else
using Backend = uccl_lite::NetworkCollectives;
using BackendOperation = uccl_lite::NetworkOperation;
using BackendId = uccl_lite::NetworkId;
constexpr int kHosts = 2, kLocalRanks = 4, kElementBytes = 4;
constexpr ncclDataType_t kDtype = ncclFloat32;
constexpr char kProfileMagic[] = "UCCL_LITE_MSCCLPP_STRICT_V1";
#endif
size_t extentScale(int op) { return kElementBytes * (op == 3 ? 1 : 8); }
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
  std::array<std::string, kHosts> hosts, identities;
  std::array<std::string, 8> uuids;
  uint64_t digest = 0;
  size_t capacity = 0;
  int node = -1;
};
Policy const& policy() {
  static Policy p = [] {
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
    if (!(input >> magic) || magic != kProfileMagic) return Policy{};
    char hostname[256]{};
    gethostname(hostname, sizeof(hostname) - 1);
    int driver = 0, runtime = 0, nccl = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    nccl = NCCL_VERSION_CODE;
    for (int n = 0; n < kHosts; ++n) {
      int d, r, v;
      if (!(input >> result.hosts[n] >> result.identities[n] >> d >> r >> v))
        return Policy{};
      for (int i = 0; i < kLocalRanks; ++i)
        if (!(input >> result.uuids[kLocalRanks * n + i])) return Policy{};
      if (result.hosts[n] == hostname && result.identities[n] == verified &&
          driver == d && runtime == r && nccl == v)
        result.node = n;
    }
    if ((kHosts == 2 && result.hosts.front() == result.hosts.back()) ||
        result.node < 0)
      return Policy{};
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      std::istringstream row(line);
      int op;
      size_t count;
      std::string extra;
      if (!(row >> op >> count) || row >> extra || op < 1 || op > 3 || !count)
        return Policy{};
      size_t scale = extentScale(op);
      if (count > (1UL << 30) / scale || (count * scale) % 128 ||
          !result.shapes.emplace(std::make_pair(op, count), true).second)
        return Policy{};
      result.capacity = std::max(result.capacity, count * scale);
    }
    result.digest = hash(data);
    return result;
  }();
  return p;
}
}  // namespace

// Own communicator handle; no native NCCL object is created or inspected.
struct ncclComm {
  int rank, device;
  bool finalized = false;
  ncclResult_t error = ncclSuccess;
  std::mutex mutex;
  std::unique_ptr<Backend> transport;
  std::map<std::pair<int, size_t>, uint64_t> counts;
};

extern "C" {
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
    auto bytes = Backend::createUniqueId();
    static_assert(sizeof(*id) == sizeof(bytes));
    std::memcpy(id, bytes.data(), bytes.size());
    return ncclSuccess;
  } catch (std::exception const& error) {
    return fail(error.what(), ncclSystemError);
  }
}
ncclResult_t ncclCommInitRank(ncclComm_t* output, int ranks, ncclUniqueId id,
                              int rank) {
  if (!output || ranks != 8 || rank < 0 || rank >= ranks)
    return fail("requires eight ranks, one GPU per process",
                ncclInvalidArgument);
  *output = nullptr;
  if (nativeLoaded()) return fail("native NCCL library is loaded");
  if (live.exchange(true))
    return fail("only one live communicator per process is supported");
  try {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cudaGetDevice failed");
    auto const& p = policy();
    struct Vote {
      uint64_t digest;
      int valid, node, device, pid;
    };
    Vote local{
        p.digest,
        int(!p.shapes.empty() && p.node == rank / kLocalRanks &&
            device == rank % kLocalRanks && uuid(device) == p.uuids[rank]),
        p.node, device, int(getpid())};
    std::array<Vote, 8> votes{};
    uccl_lite::nccl_detail::HostControl control(
        std::getenv("UCCL_LITE_MSCCLPP_CONTROL_ROOT"), &id, sizeof(id), rank,
        ranks);
    control.allGather(&local, votes.data(), sizeof(local));
    for (int i = 0; i < ranks; ++i) {
      if (!votes[i].valid || votes[i].digest != local.digest ||
          votes[i].node != i / kLocalRanks ||
          votes[i].device != i % kLocalRanks)
        throw std::runtime_error("profile/topology consensus failed");
      for (int j = i / kLocalRanks * kLocalRanks; j < i; ++j)
        if (votes[i].pid == votes[j].pid)
          throw std::runtime_error("multiple ranks per process");
    }
    auto comm = std::make_unique<ncclComm>();
    comm->rank = rank;
    comm->device = device;
    BackendId bytes{};
    std::memcpy(bytes.data(), &id, sizeof(id));
    comm->transport =
        std::make_unique<Backend>(bytes, rank, device, device / 2, p.capacity);
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
  *count = 8;
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
  if (dtype != kDtype || (op != 1 && reduction != ncclSum) ||
      !policy().shapes.count({op, count}))
    return fail(
        "collective dtype/reduction/extent is outside the strict profile");
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
    return ncclUnhandledCudaError;
  if (capture != cudaStreamCaptureStatusNone)
    return fail("network graph capture unsupported");
  std::lock_guard<std::mutex> guard(comm->mutex);
  try {
    if (!comm->transport->initialized()) comm->transport->prepare();
    comm->transport->enqueue(static_cast<BackendOperation>(op), input, output,
                             count, stream);
    ++comm->counts[{op, count * extentScale(op)}];
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

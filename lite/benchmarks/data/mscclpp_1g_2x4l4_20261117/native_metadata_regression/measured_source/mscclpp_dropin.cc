// NCCL ABI adapter using MSCCL++ NativeAlgorithm and UCCL Lite kernels.
#include "uccl_lite/collectives.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <nccl.h>
#include <unistd.h>

namespace {
template <typename F>
F real(char const* name) {
  static void* library = [] {
    char const* path = std::getenv("UCCL_LITE_REAL_NCCL");
    void* handle =
        dlopen(path && *path ? path : "libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      std::fprintf(stderr, "UCCL Lite: %s\n", dlerror());
      std::abort();
    }
    return handle;
  }();
  auto symbol = dlsym(library, name);
  if (!symbol) {
    std::fprintf(stderr, "UCCL Lite: missing %s\n", name);
    std::abort();
  }
  return reinterpret_cast<F>(symbol);
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
  std::array<std::string, 2> hosts, identities;
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
    if (!(input >> magic) || magic != "UCCL_LITE_MSCCLPP_CANDIDATE_V1")
      return Policy{};
    char hostname[256]{};
    gethostname(hostname, sizeof(hostname) - 1);
    int driver = 0, runtime = 0, nccl = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    real<decltype(&ncclGetVersion)>("ncclGetVersion")(&nccl);
    for (int n = 0; n < 2; ++n) {
      int d, r, v;
      if (!(input >> result.hosts[n] >> result.identities[n] >> d >> r >> v))
        return Policy{};
      for (int i = 0; i < 4; ++i)
        if (!(input >> result.uuids[4 * n + i])) return Policy{};
      if (result.hosts[n] == hostname && result.identities[n] == verified &&
          driver == d && runtime == r && nccl == v)
        result.node = n;
    }
    if (result.hosts[0] == result.hosts[1] || result.node < 0) return Policy{};
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
      size_t scale = op == 3 ? 4 : 32;
      if (count > (256UL << 10) / scale || (count * scale) % 128 ||
          !result.shapes.emplace(std::make_pair(op, count), true).second)
        return Policy{};
      result.capacity = std::max(result.capacity, count * scale);
    }
    result.digest = hash(data);
    return result;
  }();
  return p;
}
struct State {
  std::mutex mutex;
  int rank = 0, device = 0;
  bool disabled = false, recorded = false;
  cudaStream_t lastCaller = nullptr;
  cudaEvent_t done = nullptr;
  std::unique_ptr<uccl_lite::NetworkCollectives> network;
  uint64_t calls[4]{};
  ~State() {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (recorded) cudaEventSynchronize(done);
    network.reset();
    if (done) cudaEventDestroy(done);
    cudaSetDevice(previous);
    std::fprintf(
        stderr,
        "UCCL Lite MSCCL++: rank=%d AG=%llu RS=%llu AR=%llu fallback=%llu\n",
        rank, (unsigned long long)calls[1], (unsigned long long)calls[2],
        (unsigned long long)calls[3], (unsigned long long)calls[0]);
  }
};
std::mutex statesMutex;
std::unordered_map<ncclComm_t, std::shared_ptr<State>> states;
thread_local int groupDepth = 0;
struct PendingInit {
  ncclComm_t* comm;
  int ranks;
  ncclUniqueId id;
  int rank;
  int device;
};
thread_local std::vector<PendingInit> pendingInit;
std::shared_ptr<State> stateFor(ncclComm_t comm) {
  std::lock_guard<std::mutex> guard(statesMutex);
  auto it = states.find(comm);
  return it == states.end() ? nullptr : it->second;
}
void remember(ncclComm_t comm, int ranks, ncclUniqueId, int rank) {
  if (ranks != 8 || groupDepth || !std::getenv("UCCL_LITE_MSCCLPP_CANDIDATE"))
    return;
  // Every participating rank votes, including missing/stale profile ranks.
  // No UCCL GPU allocation, transport registration or proxy is created here.
  auto const& p = policy();
  struct Vote {
    uint64_t digest;
    int valid, node, device, pid;
  };
  int device = -1;
  cudaGetDevice(&device);
  Vote local{p.digest,
             int(!p.shapes.empty() && p.node == rank / 4 &&
                 device == rank % 4 && uuid(device) == p.uuids[rank]),
             p.node, device, int(getpid())};
  std::array<Vote, 8> votes{};
  Vote* gpu = nullptr;
  cudaStream_t stream = nullptr;
  bool ok =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
  if (ok) ok = cudaMalloc(&gpu, sizeof(votes)) == cudaSuccess;
  if (ok)
    ok = cudaMemcpyAsync(gpu + rank, &local, sizeof(local),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess;
  if (ok)
    ok = real<decltype(&ncclAllGather)>("ncclAllGather")(
             gpu + rank, gpu, sizeof(Vote), ncclUint8, comm, stream) ==
         ncclSuccess;
  if (ok)
    ok = cudaMemcpyAsync(votes.data(), gpu, sizeof(votes),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
  if (ok) ok = cudaStreamSynchronize(stream) == cudaSuccess;
  if (gpu) cudaFree(gpu);
  if (stream) cudaStreamDestroy(stream);
  for (int i = 0; i < 8 && ok; ++i) {
    ok = votes[i].valid && votes[i].digest == local.digest &&
         votes[i].node == i / 4 && votes[i].device == i % 4;
    for (int j = i / 4 * 4; j < i; ++j) ok = ok && votes[i].pid != votes[j].pid;
  }
  if (!ok) {
    std::fprintf(stderr,
                 "UCCL Lite MSCCL++: rank=%d identity vote rejected (local "
                 "valid=%d node=%d device=%d)\n",
                 rank, local.valid, local.node, local.device);
    return;
  }
  auto s = std::make_shared<State>();
  s->rank = rank;
  s->device = device;
  std::lock_guard<std::mutex> guard(statesMutex);
  states[comm] = s;
}
void rememberOrDefer(ncclComm_t* comm, int ranks, ncclUniqueId id, int rank) {
  if (!groupDepth) {
    remember(*comm, ranks, id, rank);
    return;
  }
  int device = -1;
  cudaGetDevice(&device);
  pendingInit.push_back({comm, ranks, id, rank, device});
}
bool prepare(State& s, ncclComm_t native) {
  if (s.network) return true;
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  uccl_lite::NetworkId id{};
  if (!s.rank) id = uccl_lite::NetworkCollectives::createUniqueId();
  void* gpu = nullptr;
  cudaStream_t stream = nullptr;
  bool ok =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
  if (ok) ok = cudaMalloc(&gpu, id.size()) == cudaSuccess;
  if (ok && !s.rank)
    ok = cudaMemcpyAsync(gpu, id.data(), id.size(), cudaMemcpyHostToDevice,
                         stream) == cudaSuccess;
  if (ok)
    ok = real<decltype(&ncclBroadcast)>("ncclBroadcast")(
             gpu, gpu, id.size(), ncclUint8, 0, native, stream) == ncclSuccess;
  if (ok)
    ok = cudaMemcpyAsync(id.data(), gpu, id.size(), cudaMemcpyDeviceToHost,
                         stream) == cudaSuccess;
  if (ok) ok = cudaStreamSynchronize(stream) == cudaSuccess;
  if (gpu) cudaFree(gpu);
  if (stream) cudaStreamDestroy(stream);
  if (!ok ||
      cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) != cudaSuccess)
    return false;
  s.network = std::make_unique<uccl_lite::NetworkCollectives>(
      id, s.rank, s.device, s.device / 2, policy().capacity);
  s.network->prepare();
  return true;
}
bool disable(State& s) {
  if (!s.disabled && s.recorded && cudaEventSynchronize(s.done) != cudaSuccess)
    return false;
  s.disabled = true;
  return true;
}
template <typename F>
ncclResult_t invoke(ncclComm_t comm, cudaStream_t caller, int op,
                    void const* send, void* recv, size_t count,
                    ncclDataType_t dtype, ncclRedOp_t reduction, F fallback) {
  // Empty collectives do not change ownership or add work to captured graphs.
  if (op && !count) return fallback(caller);
  auto s = stateFor(comm);
  if (!s) return fallback(caller);
  std::lock_guard<std::mutex> guard(s->mutex);
  if (s->disabled) return fallback(caller);
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(caller, &capture) != cudaSuccess)
    return ncclUnhandledCudaError;
  if (capture != cudaStreamCaptureStatusNone) {
    if (s->recorded &&
        cudaStreamWaitEvent(caller, s->done, cudaEventWaitExternal) !=
            cudaSuccess)
      return ncclUnhandledCudaError;
    s->disabled = true;
    return fallback(caller);
  }
  if (groupDepth) {
    if (!disable(*s)) return ncclUnhandledCudaError;
    return fallback(caller);
  }
  bool selected = dtype == ncclFloat32 && (op == 1 || reduction == ncclSum) &&
                  policy().shapes.count({op, count});
  if (!selected && !s->network) return fallback(caller);
  try {
    if (!prepare(*s, comm)) return ncclSystemError;
    if (s->recorded && s->lastCaller != caller &&
        cudaStreamWaitEvent(caller, s->done, 0) != cudaSuccess)
      return ncclUnhandledCudaError;
    ncclResult_t result = ncclSuccess;
    if (selected)
      s->network->enqueue(static_cast<uccl_lite::NetworkOperation>(op), send,
                          recv, count, caller);
    else
      result = fallback(caller);
    if (result != ncclSuccess) return result;
    ++s->calls[selected ? op : 0];
    if (cudaEventRecord(s->done, caller) != cudaSuccess)
      return ncclUnhandledCudaError;
    s->lastCaller = caller;
    s->recorded = true;
    return ncclSuccess;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "UCCL Lite network rank %d: %s\n", s->rank,
                 error.what());
    return ncclSystemError;
  }
}
void forget(ncclComm_t comm, bool aborting) {
  std::shared_ptr<State> s;
  {
    std::lock_guard<std::mutex> guard(statesMutex);
    auto it = states.find(comm);
    if (it == states.end()) return;
    s = it->second;
    states.erase(it);
  }
  if (aborting && s->network &&
      (!s->recorded || cudaEventQuery(s->done) != cudaSuccess))
    (void)new std::shared_ptr<State>(std::move(s));
}
}  // namespace

extern "C" {
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ranks, ncclUniqueId id,
                              int rank) {
  auto r = real<decltype(&ncclCommInitRank)>("ncclCommInitRank")(comm, ranks,
                                                                 id, rank);
  if (r == ncclSuccess) rememberOrDefer(comm, ranks, id, rank);
  return r;
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int ranks,
                                    ncclUniqueId id, int rank,
                                    ncclConfig_t* config) {
  auto r = real<decltype(&ncclCommInitRankConfig)>("ncclCommInitRankConfig")(
      comm, ranks, id, rank, config);
  if (r == ncclSuccess && (!config || config->blocking != 0))
    rememberOrDefer(comm, ranks, id, rank);
  return r;
}
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  forget(comm, false);
  return real<decltype(&ncclCommDestroy)>("ncclCommDestroy")(comm);
}
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  forget(comm, true);
  return real<decltype(&ncclCommAbort)>("ncclCommAbort")(comm);
}
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  auto s = stateFor(comm);
  if (s) {
    std::lock_guard<std::mutex> guard(s->mutex);
    if (!disable(*s)) return ncclUnhandledCudaError;
  }
  return real<decltype(&ncclCommFinalize)>("ncclCommFinalize")(comm);
}
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t* child, ncclConfig_t* config) {
  auto s = stateFor(comm);
  if (s) {
    std::lock_guard<std::mutex> guard(s->mutex);
    if (!disable(*s)) return ncclUnhandledCudaError;
  }
  return real<decltype(&ncclCommSplit)>("ncclCommSplit")(comm, color, key,
                                                         child, config);
}
ncclResult_t ncclAllGather(void const* send, void* recv, size_t count,
                           ncclDataType_t dtype, ncclComm_t comm,
                           cudaStream_t stream) {
  static auto fn = real<decltype(&ncclAllGather)>("ncclAllGather");
  return invoke(
      comm, stream, 1, send, recv, count, dtype, ncclSum,
      [&](cudaStream_t st) { return fn(send, recv, count, dtype, comm, st); });
}
ncclResult_t ncclReduceScatter(void const* send, void* recv, size_t count,
                               ncclDataType_t dtype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
  static auto fn = real<decltype(&ncclReduceScatter)>("ncclReduceScatter");
  return invoke(comm, stream, 2, send, recv, count, dtype, op,
                [&](cudaStream_t st) {
                  return fn(send, recv, count, dtype, op, comm, st);
                });
}
ncclResult_t ncclAllReduce(void const* send, void* recv, size_t count,
                           ncclDataType_t dtype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  static auto fn = real<decltype(&ncclAllReduce)>("ncclAllReduce");
  return invoke(comm, stream, 3, send, recv, count, dtype, op,
                [&](cudaStream_t st) {
                  return fn(send, recv, count, dtype, op, comm, st);
                });
}
ncclResult_t ncclBroadcast(void const* send, void* recv, size_t count,
                           ncclDataType_t dtype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  static auto fn = real<decltype(&ncclBroadcast)>("ncclBroadcast");
  return invoke(comm, stream, 0, send, recv, count, dtype, ncclSum,
                [&](cudaStream_t st) {
                  return fn(send, recv, count, dtype, root, comm, st);
                });
}
ncclResult_t ncclReduce(void const* send, void* recv, size_t count,
                        ncclDataType_t dtype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  static auto fn = real<decltype(&ncclReduce)>("ncclReduce");
  return invoke(comm, stream, 0, send, recv, count, dtype, op,
                [&](cudaStream_t st) {
                  return fn(send, recv, count, dtype, op, root, comm, st);
                });
}
ncclResult_t ncclBcast(void* buffer, size_t count, ncclDataType_t dtype,
                       int root, ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buffer, buffer, count, dtype, root, comm, stream);
}
ncclResult_t ncclAlltoAll(void const* send, void* recv, size_t count,
                          ncclDataType_t dtype, ncclComm_t comm,
                          cudaStream_t stream) {
  static auto fn = real<decltype(&ncclAlltoAll)>("ncclAlltoAll");
  return invoke(
      comm, stream, 0, send, recv, count, dtype, ncclSum,
      [&](cudaStream_t st) { return fn(send, recv, count, dtype, comm, st); });
}
ncclResult_t ncclGather(void const* send, void* recv, size_t count,
                        ncclDataType_t dtype, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  static auto fn = real<decltype(&ncclGather)>("ncclGather");
  return invoke(comm, stream, 0, send, recv, count, dtype, ncclSum,
                [&](cudaStream_t st) {
                  return fn(send, recv, count, dtype, root, comm, st);
                });
}
ncclResult_t ncclScatter(void const* send, void* recv, size_t count,
                         ncclDataType_t dtype, int root, ncclComm_t comm,
                         cudaStream_t stream) {
  static auto fn = real<decltype(&ncclScatter)>("ncclScatter");
  return invoke(comm, stream, 0, send, recv, count, dtype, ncclSum,
                [&](cudaStream_t st) {
                  return fn(send, recv, count, dtype, root, comm, st);
                });
}
ncclResult_t ncclSend(void const* send, size_t count, ncclDataType_t dtype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  static auto fn = real<decltype(&ncclSend)>("ncclSend");
  return invoke(
      comm, stream, 0, send, nullptr, count, dtype, ncclSum,
      [&](cudaStream_t st) { return fn(send, count, dtype, peer, comm, st); });
}
ncclResult_t ncclRecv(void* recv, size_t count, ncclDataType_t dtype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
  static auto fn = real<decltype(&ncclRecv)>("ncclRecv");
  return invoke(
      comm, stream, 0, nullptr, recv, count, dtype, ncclSum,
      [&](cudaStream_t st) { return fn(recv, count, dtype, peer, comm, st); });
}
ncclResult_t ncclGroupStart() {
  auto r = real<decltype(&ncclGroupStart)>("ncclGroupStart")();
  if (r == ncclSuccess) ++groupDepth;
  return r;
}
ncclResult_t ncclGroupEnd() {
  auto r = real<decltype(&ncclGroupEnd)>("ncclGroupEnd")();
  if (groupDepth) --groupDepth;
  if (!groupDepth) {
    auto pending = std::move(pendingInit);
    pendingInit.clear();
    // nccl-tests groups initialization even with one GPU/process. Defer the
    // collective identity vote until native NCCL has completed GroupEnd.
    // Multiple local ranks initialized by one thread remain entirely native.
    if (r == ncclSuccess && pending.size() == 1) {
      auto const& p = pending.front();
      int previous = 0;
      cudaGetDevice(&previous);
      cudaSetDevice(p.device);
      remember(*p.comm, p.ranks, p.id, p.rank);
      cudaSetDevice(previous);
    }
  }
  return r;
}
}

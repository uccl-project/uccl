// NCCL ABI interposition for calibrated, single-node BF16 packet collectives.
// The real NCCL is also a DT_NEEDED dependency: unwrapped API symbols retain
// their implementation when this library is loaded through ctypes/dlopen.
#include "uccl_lite/multiprocess_collectives.h"
#include <algorithm>
#include <array>
#include <cmath>
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
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <nccl.h>
#include <unistd.h>

namespace {
template <typename F>
F real(char const* name) {
  static void* library = [] {
    char const* path = std::getenv("UCCL_LITE_REAL_NCCL");
    void* p =
        dlopen(path && *path ? path : "libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
    if (!p) {
      std::fprintf(stderr, "UCCL Lite: %s\n", dlerror());
      std::abort();
    }
    return p;
  }();
  auto p = dlsym(library, name);
  if (!p) {
    std::fprintf(stderr, "UCCL Lite: missing %s\n", name);
    std::abort();
  }
  return reinterpret_cast<F>(p);
}
uint64_t hash(void const* data, size_t bytes) {
  uint64_t h = 14695981039346656037ULL;
  auto p = static_cast<unsigned char const*>(data);
  for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ULL;
  return h;
}
std::string uuid(int device) {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return {};
  auto id = prop.uuid;
  char result[33];
  for (int i = 0; i < 16; ++i)
    std::snprintf(result + 2 * i, 3, "%02x",
                  static_cast<unsigned char>(id.bytes[i]));
  return result;
}
struct Policy {
  std::map<std::pair<int, size_t>, bool> shapes;
  std::array<std::string, 8> uuids;
  uint64_t digest = 0;
  size_t capacity = 0;
};
Policy const& policy() {
  static Policy p = [] {
    Policy result;
    auto path = std::getenv("UCCL_LITE_DROPIN_PROFILE");
    auto verified = std::getenv("UCCL_LITE_DROPIN_IDENTITY");
    if (!path || !verified) return result;
    std::ifstream file(path);
    std::string data((std::istreambuf_iterator<char>(file)), {});
    std::istringstream input(data);
    std::string magic, identity;
    char const* mode = std::getenv("UCCL_LITE_DROPIN_CALIBRATING");
    bool calibration = mode && std::strcmp(mode, "1") == 0;
    if (!(input >> magic >> identity) ||
        (magic != "UCCL_LITE_DROPIN_V1" &&
         !(calibration && magic == "UCCL_LITE_CANDIDATE_V1")) ||
        identity != verified)
      return Policy{};
    int driver = 0, runtime = 0, nccl = 0, pd = 0, pr = 0, pn = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    real<decltype(&ncclGetVersion)>("ncclGetVersion")(&nccl);
    if (!(input >> pd >> pr >> pn) || driver != pd || runtime != pr ||
        nccl != pn)
      return Policy{};
    for (auto& id : result.uuids)
      if (!(input >> id)) return Policy{};
    calibration = calibration && magic == "UCCL_LITE_CANDIDATE_V1";
    // Rows certify both NCCL in-place and out-of-place layouts, including
    // minimally aligned BF16 pointers. No interpolation or GPU-count policy.
    std::string line;
    while (std::getline(input, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      std::istringstream row(line);
      int op, repetitions;
      size_t count;
      double ncclMs, packetMs;
      std::string extra;
      if (!(row >> op >> count >> ncclMs >> packetMs >> repetitions) ||
          (row >> extra))
        return Policy{};
      if (op < 1 || op > 3 || !count || repetitions < 3 || !(ncclMs > 0) ||
          !(packetMs > 0) || !std::isfinite(ncclMs) ||
          !std::isfinite(packetMs) ||
          (!calibration && packetMs * 1.05 > ncclMs) ||
          count > (256UL << 20) / (op == 3 ? 2 : 16))
        return Policy{};
      if (!result.shapes.emplace(std::make_pair(op, count), true).second)
        return Policy{};
      result.capacity = std::max(result.capacity, count * (op == 3 ? 2 : 16));
    }
    if (!input.eof()) return Policy{};
    result.digest = hash(data.data(), data.size());
    return result;
  }();
  return p;
}
struct State {
  std::mutex mutex;
  std::string id;
  int rank = 0, device = 0;
  bool disabled = false;
  bool recorded = false;
  cudaStream_t lastCaller = nullptr;
  void* packet = nullptr;
  void* scratch = nullptr;
  void* outputScratch = nullptr;
  cudaEvent_t done = nullptr;
  uint64_t calls[4]{};
  ~State() {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (packet) uccl_lite_mp_destroy(packet);
    if (scratch) cudaFree(scratch);
    if (outputScratch) cudaFree(outputScratch);
    if (done) cudaEventDestroy(done);
    cudaSetDevice(previous);
    if (packet)
      std::fprintf(
          stderr,
          "UCCL Lite drop-in: rank=%d AG=%llu RS=%llu AR=%llu fallback=%llu\n",
          rank, (unsigned long long)calls[1], (unsigned long long)calls[2],
          (unsigned long long)calls[3], (unsigned long long)calls[0]);
  }
};
std::mutex statesMutex;
std::unordered_map<ncclComm_t, std::shared_ptr<State>> states;
thread_local int groupDepth = 0;
std::shared_ptr<State> stateFor(ncclComm_t comm) {
  std::lock_guard<std::mutex> guard(statesMutex);
  auto it = states.find(comm);
  return it == states.end() ? nullptr : it->second;
}
// NCCL performs the initial consensus. This is communicator preparation, not
// a router allocation. Reordered GPUs, multiple ranks per GPU/process, remote
// hosts and inconsistent profiles all leave the communicator on NCCL.
void remember(ncclComm_t comm, int ranks, ncclUniqueId id, int rank) {
  if (ranks != 8 || groupDepth || !std::getenv("UCCL_LITE_DROPIN_PROFILE"))
    return;
  struct Vote {
    uint64_t profile, host;
    int pid, device, valid;
  };
  auto const& p = policy();
  int device = -1;
  cudaGetDevice(&device);
  char host[256]{};
  gethostname(host, sizeof(host));
  Vote local{p.digest, hash(host, std::strlen(host)), int(getpid()), device,
             int(!p.shapes.empty() && device == rank &&
                 uuid(device) == p.uuids[rank])};
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
    ok = votes[i].valid && votes[i].profile == local.profile &&
         votes[i].host == local.host && votes[i].device == i;
    for (int j = 0; j < i; ++j) ok = ok && votes[i].pid != votes[j].pid;
  }
  if (!ok) return;
  auto s = std::make_shared<State>();
  s->rank = rank;
  s->device = device;
  s->id = "dropin-" + std::to_string(hash(&id, sizeof(id)));
  std::lock_guard<std::mutex> guard(statesMutex);
  states[comm] = std::move(s);
}
// Called only for a calibrated eager call, before any packet enqueue. Pure
// fallback, grouped calls and capture never construct packet buffers/workers.
bool prepare(State& s) {
  if (s.packet) return true;
  // NCCL calls preceding the first selected shape may have used other
  // streams. Complete them at the explicit preparation boundary.
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  if (cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) !=
          cudaSuccess ||
      cudaMalloc(&s.scratch, policy().capacity) != cudaSuccess ||
      cudaMalloc(&s.outputScratch, policy().capacity) != cudaSuccess)
    return false;
  s.packet = uccl_lite_mp_create(s.id.c_str(), 8, s.rank, policy().capacity);
  return s.packet != nullptr;
}
// Groups and graph replay are executed entirely by NCCL. Once such a call
// appears, retain NCCL ownership for this communicator's remaining lifetime.
// This avoids packet sequence counters inside replays and preserves group
// submission semantics. Only the transition drains prior packet work.
bool disable(State& s) {
  if (!s.disabled && s.packet && cudaEventSynchronize(s.done) != cudaSuccess)
    return false;
  s.disabled = true;
  return true;
}
template <typename F>
ncclResult_t invoke(ncclComm_t comm, cudaStream_t caller, int op,
                    void const* send, void* recv, size_t count,
                    ncclDataType_t dtype, ncclRedOp_t reduction, F fallback) {
  static bool trace = std::getenv("UCCL_LITE_DROPIN_TRACE") != nullptr;
  if (trace && op) {
    static std::mutex traceMutex;
    static std::map<std::string, bool> seen;
    int rank = -1;
    real<decltype(&ncclCommUserRank)>("ncclCommUserRank")(comm, &rank);
    if (rank == 0) {
      auto key = std::to_string(op) + " " + std::to_string(count) + " " +
                 std::to_string(dtype);
      std::lock_guard<std::mutex> guard(traceMutex);
      if (seen.emplace(key, true).second)
        std::fprintf(stderr, "UCCL Lite shape: %s\n", key.c_str());
    }
  }
  auto s = stateFor(comm);
  if (!s) return fallback(caller);
  std::lock_guard<std::mutex> guard(s->mutex);
  if (s->disabled) return fallback(caller);
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(caller, &capture) != cudaSuccess)
    return ncclUnhandledCudaError;
  if (capture != cudaStreamCaptureStatusNone) {
    // Synchronizing an event on the host invalidates global capture. Import
    // the final packet completion as an external event dependency instead.
    // The event stays unchanged after this permanent transition to NCCL.
    if (s->packet && cudaStreamWaitEvent(caller, s->done,
                                         cudaEventWaitExternal) != cudaSuccess)
      return ncclUnhandledCudaError;
    s->disabled = true;
    return fallback(caller);
  }
  if (groupDepth) {
    if (!s->packet) return fallback(caller);
    if (!disable(*s)) return ncclUnhandledCudaError;
    return fallback(caller);
  }
  bool selected = dtype == ncclBfloat16 && (op == 1 || reduction == ncclSum) &&
                  policy().shapes.count({op, count});
  if (!selected && !s->packet) return fallback(caller);
  if (!prepare(*s)) return ncclSystemError;
  cudaStream_t workStream = caller;
  if (s->recorded && s->lastCaller != caller &&
      cudaStreamWaitEvent(caller, s->done, 0) != cudaSuccess)
    return ncclUnhandledCudaError;
  ncclResult_t result;
  if (selected) {
    // Legal in-place AG/RS can overwrite a later input tile. Preserve the
    // complete input first; AR's packet protocol already supports aliasing.
    size_t inputBytes = count * (op == 2 ? 16 : 2);
    size_t outputBytes = count * (op == 1 ? 16 : 2);
    auto a = reinterpret_cast<uintptr_t>(send),
         b = reinterpret_cast<uintptr_t>(recv);
    if ((a & 15) || (op != 3 && a < b + outputBytes && b < a + inputBytes)) {
      if (cudaMemcpyAsync(s->scratch, send, inputBytes,
                          cudaMemcpyDeviceToDevice, workStream) != cudaSuccess)
        return ncclUnhandledCudaError;
      send = s->scratch;
    }
    void* output = (b & 15) ? s->outputScratch : recv;
    result = uccl_lite_mp_enqueue_ordered(s->packet, op, send, output, count,
                                          workStream) == 0
                 ? ncclSuccess
                 : ncclSystemError;
    if (result == ncclSuccess && output != recv &&
        cudaMemcpyAsync(recv, output, outputBytes, cudaMemcpyDeviceToDevice,
                        workStream) != cudaSuccess)
      return ncclUnhandledCudaError;
  } else
    result = fallback(workStream);
  ++s->calls[selected ? op : 0];
  if (result != ncclSuccess) return result;
  if (cudaEventRecord(s->done, workStream) != cudaSuccess)
    return ncclUnhandledCudaError;
  s->recorded = true;
  s->lastCaller = caller;
  return ncclSuccess;
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
  // Fault recovery must not wait forever for a missing peer. A failed job
  // retains these resources until process exit; successful teardown frees all.
  if (aborting && s->packet && cudaEventQuery(s->done) != cudaSuccess)
    (void)new std::shared_ptr<State>(std::move(s));
}
}  // namespace

extern "C" {
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ranks, ncclUniqueId id,
                              int rank) {
  auto r = real<decltype(&ncclCommInitRank)>("ncclCommInitRank")(comm, ranks,
                                                                 id, rank);
  if (r == ncclSuccess) remember(*comm, ranks, id, rank);
  return r;
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int ranks,
                                    ncclUniqueId id, int rank,
                                    ncclConfig_t* config) {
  auto r = real<decltype(&ncclCommInitRankConfig)>("ncclCommInitRankConfig")(
      comm, ranks, id, rank, config);
  if (r == ncclSuccess && (!config || config->blocking != 0))
    remember(*comm, ranks, id, rank);
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
  return r;
}
}

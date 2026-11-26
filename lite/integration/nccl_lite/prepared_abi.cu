#include "auto_init.hpp"
#include "native_fallback.hpp"
#include "network_preflight.hpp"
#include "prepared_abi.h"
#include "qualification.cuh"
#include "uccl_lite/bootstrap/control.hpp"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/runtime/build_identity.hpp"
#include "uccl_lite/runtime/network_identity.hpp"
#include <mscclpp/core.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <dlfcn.h>

extern char** environ;

namespace {
using namespace uccl_lite;
using namespace uccl_lite::runtime;
thread_local unsigned groupDepth = 0, internalDepth = 0;
thread_local std::string lastError;
struct Internal {
  Internal() { ++internalDepth; }
  ~Internal() { --internalDepth; }
};
struct NativeError : std::runtime_error {
  ncclResult_t code;
  NativeError(ncclResult_t c, char const* s) : std::runtime_error(s), code(c) {}
};
struct Api {
  void* library = nullptr;
  Api() {
    auto path = std::getenv("UCCL_LITE_REAL_NCCL");
    library =
        dlopen(path && *path ? path : "libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    if (dlsym(library, "uccl_lite_nccl_runtime_backends"))
      throw std::invalid_argument("strict UCCL is not native NCCL");
  }
  template <class T>
  T get(char const* name) {
    auto p = dlsym(library, name);
    if (!p)
      throw std::runtime_error(std::string("native NCCL symbol missing: ") +
                               name);
    return reinterpret_cast<T>(p);
  }
  void check(ncclResult_t r) {
    if (r != ncclSuccess)
      throw NativeError(
          r, get<decltype(&ncclGetErrorString)>("ncclGetErrorString")(r));
  }
};
Api& api() {
  static Api instance;
  return instance;
}
void jsonString(std::ostream& out, std::string const& text) {
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\')
      out << '\\' << char(c);
    else if (c < 32) {
      char code[7];
      std::snprintf(code, sizeof(code), "\\u%04x", unsigned(c));
      out << code;
    } else
      out << char(c);
  }
  out << '"';
}
struct CudaFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void cudaCheck(cudaError_t r) {
  if (r != cudaSuccess) throw CudaFailure(cudaGetErrorString(r));
}
template <class F>
ncclResult_t boundary(F f) noexcept {
  try {
    lastError.clear();
    return f();
  } catch (NativeError const& e) {
    lastError = e.what();
    return e.code;
  } catch (CudaFailure const& e) {
    lastError = e.what();
    return ncclUnhandledCudaError;
  } catch (std::invalid_argument const& e) {
    lastError = e.what();
    return ncclInvalidArgument;
  } catch (std::logic_error const& e) {
    lastError = e.what();
    return ncclInvalidUsage;
  } catch (std::exception const& e) {
    lastError = e.what();
    return ncclSystemError;
  } catch (...) {
    lastError = "unknown prepared NCCL ABI error";
    return ncclInternalError;
  }
}
struct Event {
  int device;
  cudaEvent_t event = nullptr;
  explicit Event(int d) : device(d) {
    cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  ~Event() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaEventDestroy(event);
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
  }
};
struct Entry {
  CollectiveRequest request;
  PreparedSelection core;
  bool correct = false, admitted = false, poolCorrect = false;
  std::vector<double> nativeSamples, dispatchSamples, poolNativeSamples,
      poolDispatchSamples;
  std::string executorKey;
};
struct State;
std::mutex registryMutex;
std::map<ncclComm_t, std::shared_ptr<State>> registry;
std::map<ncclComm_t, std::string> automaticNotes;
thread_local std::vector<integration::AutoInitRecord> pendingInitializations;
std::atomic<bool> abortedPreparedProcess{false};
std::shared_ptr<State> lookup(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(registryMutex);
  auto it = registry.find(comm);
  return it == registry.end() ? nullptr : it->second;
}
struct State {
  ncclComm_t comm;
  RuntimeContext context;
  ncclDataType_t dtype;
  ScalarType scalar;
  std::shared_ptr<mscclpp::TcpBootstrap> control;
  CallerFallback baseline;
  std::vector<Entry> entries;
  std::map<std::string, std::unique_ptr<CollectiveBackend>> executors;
  std::vector<int> devices;
  std::string preparedIdentity;
  std::shared_ptr<Event> completion;
  std::atomic<bool> preparing{true}, appWork{false};
  bool localNative = false, nativeAgreed = false;
  size_t trial = SIZE_MAX;
  uint64_t sequence = 0, acceleratedCalls = 0, nativeCalls = 0, guardCalls = 0;
  std::atomic<bool> aborted{false};
  struct Vote {
    uint64_t sequence = 0;
    size_t count = 0;
    int operation = 0, dtype = 0, layout = 0, needNative = 0, device = 0;
    int grouped = 0, captured = 0;
  };
  std::array<Vote, 32> votes{};
  State(ncclComm_t c, RuntimeContext ctx, ncclDataType_t d, ScalarType s)
      : comm(c), context(ctx), dtype(d), scalar(s) {}
  template <class T>
  std::vector<T> gather(T const& local) {
    std::vector<T> values(context.ranks);
    values[context.rank] = local;
    control->allGather(values.data(), sizeof(T));
    return values;
  }
  void agree(bool ok, char const* error) {
    auto values = gather(int(ok));
    if (std::find(values.begin(), values.end(), 0) != values.end())
      throw std::runtime_error(error);
  }
  NetworkId id() {
    NetworkId value{};
    if (!context.rank) value = PreparedRuntime::createUniqueId();
    return gather(value).front();
  }
  std::string identity() const {
    std::ostringstream out;
    out << inspectTopology(devices).hardwareIdentity;
    std::set<std::string> flags;
    for (char** item = environ; *item; ++item)
      if (!std::strncmp(*item, "UCCL_LITE_", 10) ||
          !std::strncmp(*item, "MSCCLPP_", 8) ||
          !std::strncmp(*item, "NCCL_", 5) || !std::strncmp(*item, "CUDA_", 5))
        flags.insert(*item);
    for (auto const& flag : flags) out << ';' << flag.size() << ':' << flag;
    bool network = false, hierarchy = false;
    for (auto const& entry : entries) {
      hierarchy |= entry.core.backend == RuntimeBackend::NetworkHierarchy;
      network |= entry.core.backend == RuntimeBackend::NetworkHierarchy ||
                 entry.core.backend == RuntimeBackend::DirectExchange;
    }
    if (network) {
      out << ';' << networkTransportIdentity();
      if (hierarchy)
        if (auto path = std::getenv("UCCL_LITE_NETWORK_PROFILE"))
          out << ';' << encodeNetworkProfile(loadNetworkProfile(path));
    }
    return out.str();
  }
  ncclResult_t native(NetworkOperation op, void const* input, void* output,
                      size_t count, cudaStream_t stream) {
    Internal scope;
    static auto ag = api().get<decltype(&ncclAllGather)>("ncclAllGather");
    static auto rs =
        api().get<decltype(&ncclReduceScatter)>("ncclReduceScatter");
    static auto ar = api().get<decltype(&ncclAllReduce)>("ncclAllReduce");
    if (op == NetworkOperation::AllGather)
      return ag(input, output, count, dtype, comm, stream);
    if (op == NetworkOperation::ReduceScatter)
      return rs(input, output, count, dtype, ncclSum, comm, stream);
    return ar(input, output, count, dtype, ncclSum, comm, stream);
  }
  int layout(NetworkOperation op, void const* input, void* output,
             size_t count) const {
    size_t bytes = scalarBytes(scalar),
           scale = op == NetworkOperation::AllReduce ? 1 : context.ranks;
    if (!input || !output || !count ||
        count > context.capacity / (bytes * scale))
      return -1;
    auto a = reinterpret_cast<uintptr_t>(input),
         b = reinterpret_cast<uintptr_t>(output);
    size_t in = count * bytes *
                (op == NetworkOperation::ReduceScatter ? context.ranks : 1),
           out = count * bytes *
                 (op == NetworkOperation::AllGather ? context.ranks : 1);
    if (a > UINTPTR_MAX - in || b > UINTPTR_MAX - out) return -1;
    bool overlaps = a < b + out && b < a + in;
    bool legal = op == NetworkOperation::AllGather
                     ? a == b + size_t(context.rank) * count * bytes
                 : op == NetworkOperation::ReduceScatter
                     ? b == a + size_t(context.rank) * count * bytes
                     : a == b;
    unsigned alignment = qualification::alignment(input, output);
    if (alignment < bytes || (overlaps && !legal)) return -1;
    return int(2 * alignment + overlaps);
  }
  void beforeNative(cudaStream_t stream) {
    localNative = true;
    if (!appWork || !completion) return;
    cudaStreamCaptureStatus capture;
    cudaCheck(cudaStreamIsCapturing(stream, &capture));
    cudaCheck(cudaStreamWaitEvent(
        stream, completion->event,
        capture == cudaStreamCaptureStatusNone ? 0 : cudaEventWaitExternal));
  }
  // Every global call uses the same fixed-size envelope until native ownership
  // is agreed. P2P marks a local request; the next global envelope propagates
  // it.
  size_t decide(int operation, void const* input, void* output, size_t count,
                ncclDataType_t type, bool sum, cudaStream_t stream) {
    if (aborted) throw std::logic_error("prepared communicator aborted");
    if (nativeAgreed) return SIZE_MAX;
    if (preparing && trial == SIZE_MAX)
      throw std::logic_error("prepared ABI is not ready");
    cudaStreamCaptureStatus capture;
    cudaCheck(cudaStreamIsCapturing(stream, &capture));
    int device = -1;
    cudaCheck(cudaGetDevice(&device));
    Vote local{sequence++,
               count,
               operation,
               int(type),
               -1,
               int(localNative || groupDepth ||
                   capture != cudaStreamCaptureStatusNone || !sum ||
                   type != dtype || operation < 1 || operation > 3),
               device};
    if (!local.needNative)
      local.layout = layout(NetworkOperation(operation), input, output, count);
    local.needNative |= local.layout < 0 || device != context.device;
    local.grouped = groupDepth != 0;
    local.captured = capture != cudaStreamCaptureStatusNone;
    votes[context.rank] = local;
    control->allGather(votes.data(), sizeof(Vote));
    if (aborted) throw std::logic_error("prepared communicator aborted");
    for (int r = 0; r < context.ranks; ++r)
      if (votes[r].grouped != local.grouped ||
          votes[r].captured != local.captured) {
        localNative = nativeAgreed = true;
        throw std::logic_error(
            "prepared ABI handoff requires matching group/capture status "
            "across ranks");
      }
    ++guardCalls;
    bool native = false;
    for (int r = 0; r < context.ranks; ++r) {
      auto const& v = votes[r];
      native |= v.sequence != local.sequence || v.operation != operation ||
                v.count != count || v.dtype != int(type) || v.needNative;
    }
    size_t chosen = SIZE_MAX;
    if (!native)
      for (size_t i = 0; i < entries.size(); ++i) {
        auto const& e = entries[i];
        if ((trial == SIZE_MAX ? !e.admitted : i != trial) ||
            int(e.request.operation) != operation || e.request.count != count)
          continue;
        bool match = true;
        for (int r = 0; r < context.ranks; ++r)
          match &= votes[r].layout == int(2 * e.request.layouts[r].alignment +
                                          e.request.layouts[r].inPlace);
        if (match) {
          chosen = i;
          break;
        }
      }
    if (chosen == SIZE_MAX) {
      if (trial != SIZE_MAX)
        throw std::invalid_argument(
            "ABI trial metadata did not match prepared request");
      localNative = nativeAgreed = true;
    }
    return chosen;
  }
  template <class F>
  ncclResult_t dispatch(int op, void const* in, void* out, size_t count,
                        ncclDataType_t type, bool sum, cudaStream_t stream,
                        F fallback) {
    size_t chosen = decide(op, in, out, count, type, sum, stream);
    if (chosen == SIZE_MAX) {
      beforeNative(stream);
      ++nativeCalls;
      Internal scope;
      return fallback();
    }
    auto& executor = executors.at(entries[chosen].executorKey);
    if (appWork) cudaCheck(cudaStreamWaitEvent(stream, completion->event, 0));
    {
      Internal scope;
      executor->enqueue(NetworkOperation(op), in, out, count, stream);
    }
    cudaCheck(cudaEventRecord(completion->event, stream));
    if (trial == SIZE_MAX) {
      appWork = true;
      ++acceleratedCalls;
    }
    return ncclSuccess;
  }
  BackendContext backendContext() {
    auto c = context.backendContext();
    c.id = id();
    c.localDevices = devices;
    return c;
  }
  std::unique_ptr<CollectiveBackend> create(Entry const& e) {
    auto c = backendContext();
    std::unique_ptr<CollectiveBackend> result;
    Internal scope;
    switch (e.core.backend) {
      case RuntimeBackend::Peer:
        result = peerStagedBackend(scalar).create(c);
        break;
      case RuntimeBackend::Host:
        result = hostPacketBackend().create(c);
        break;
      case RuntimeBackend::PeerPacket:
        result = peerPacketBackend().create(c);
        break;
      case RuntimeBackend::PeerTree: {
        std::vector<int> parents;
        for (auto const& candidate : e.core.candidates)
          if (candidate.backend == RuntimeBackend::PeerTree)
            parents = candidate.peerTreeParents;
        result = peerTreeBackend(scalar, parents).create(c);
        break;
      }
      case RuntimeBackend::DirectExchange:
        result = directExchangeBackend(scalar).create(c);
        break;
      case RuntimeBackend::NetworkHierarchy: {
        std::shared_ptr<NetworkProfile const> profile;
        for (auto const& candidate : e.core.candidates)
          if (candidate.backend == RuntimeBackend::NetworkHierarchy)
            profile = candidate.networkCalibration;
        result = preparedNetworkBackend(c, profile);
        break;
      }
      default:
        throw std::invalid_argument(
            "caller is not an ABI acceleration candidate");
    }
    result->prepare();
    if (e.core.backend == RuntimeBackend::DirectExchange) {
      bool matched = false;
      for (auto const& candidate : e.core.candidates)
        if (candidate.backend == RuntimeBackend::DirectExchange &&
            candidate.configurationIdentities.size() == size_t(context.ranks))
          matched = candidate.configurationIdentities[context.rank] ==
                    result->configurationIdentity();
      agree(matched, "ABI direct configuration differs from qualified plan");
    }
    control->barrier();
    return result;
  }
  void synchronize() {
    if (completion && appWork)
      cudaCheck(cudaEventSynchronize(completion->event));
    for (auto& e : executors) e.second->synchronize();
  }
  void retire() {
    synchronize();
    executors.clear();
    localNative = nativeAgreed = true;
  }
  void qualify();
  ~State() {
    try {
      synchronize();
    } catch (...) {
    }
    executors.clear();
  }
};

// The actual exported entry point is used during candidate timing. Native
// baseline blocks call the original library directly, without routing guards.
ncclResult_t abiCall(State& state, CollectiveRequest const& r, void const* in,
                     void* out, cudaStream_t stream) {
  if (r.operation == NetworkOperation::AllGather)
    return ncclAllGather(in, out, r.count, state.dtype, state.comm, stream);
  if (r.operation == NetworkOperation::ReduceScatter)
    return ncclReduceScatter(in, out, r.count, state.dtype, ncclSum, state.comm,
                             stream);
  return ncclAllReduce(in, out, r.count, state.dtype, ncclSum, state.comm,
                       stream);
}
void State::qualify() {
  bool any = false;
  for (auto const& e : entries) any |= e.core.backend != RuntimeBackend::Caller;
  if (!any) {
    preparing = false;
    localNative = nativeAgreed = true;
    return;
  }
  int identityReady = 1;
  try {
    preparedIdentity = identity();
  } catch (...) {
    identityReady = 0;
  }
  agree(identityReady, "ABI preparation identity unavailable");
  int ready = 1;
  try {
    completion = std::make_shared<Event>(context.device);
  } catch (...) {
    ready = 0;
  }
  agree(ready, "ABI completion allocation failed");
  auto measure = [&](size_t index, std::set<std::string> const& pool) {
    auto& e = entries[index];
    if (e.core.backend == RuntimeBackend::Caller) return;
    bool pooled = !pool.empty();
    auto& nativeSamples = pooled ? e.poolNativeSamples : e.nativeSamples;
    auto& dispatchSamples = pooled ? e.poolDispatchSamples : e.dispatchSamples;
    nativeSamples.clear();
    dispatchSamples.clear();
    qualification::Probe probe;
    ready = 1;
    try {
      probe.allocate(e.request, context.rank, context.ranks, scalar);
    } catch (...) {
      ready = 0;
    }
    agree(ready, "ABI qualification allocation failed");
    bool candidateCorrect = true;
    int seed = 0;
    auto block = [&](bool candidate) {
      bool correct = true;
      for (int iteration = 0; iteration < 5; ++iteration, ++seed) {
        probe.initialize(seed);
        cudaCheck(cudaStreamSynchronize(probe.stream));
        control->barrier();
        auto start = std::chrono::steady_clock::now();
        if (candidate) {
          api().check(abiCall(*this, e.request, probe.input(), probe.output(),
                              probe.stream));
          executors.at(e.executorKey)->synchronize();
        } else {
          api().check(native(e.request.operation, probe.input(), probe.output(),
                             e.request.count, probe.stream));
          cudaCheck(cudaStreamSynchronize(probe.stream));
        }
        double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count();
        auto durations = gather(us);
        if (iteration >= 2)
          (candidate ? dispatchSamples : nativeSamples)
              .push_back(*std::max_element(durations.begin(), durations.end()));
        cudaCheck(
            cudaMemsetAsync(probe.wrong, 0, sizeof(unsigned), probe.stream));
        probe.verify(seed);
        unsigned wrong;
        cudaCheck(cudaMemcpyAsync(&wrong, probe.wrong, sizeof(wrong),
                                  cudaMemcpyDeviceToHost, probe.stream));
        cudaCheck(cudaStreamSynchronize(probe.stream));
        auto errors = gather(wrong);
        correct &= std::all_of(errors.begin(), errors.end(),
                               [](auto n) { return n == 0; });
      }
      return correct;
    };
    for (int round = 0; round < 3; ++round) {
      if (!block(false))
        throw std::runtime_error(
            "native ABI baseline failed full-buffer qualification");
      if (!pooled)
        executors[e.executorKey] = create(e);
      else
        for (auto const& other : entries)
          if (pool.count(other.executorKey) &&
              !executors.count(other.executorKey))
            executors[other.executorKey] = create(other);
      trial = index;
      candidateCorrect &= block(true);
      candidateCorrect &= block(true);
      trial = SIZE_MAX;
      executors.clear();
      control->barrier();
      if (!block(false))
        throw std::runtime_error(
            "native ABI baseline failed after candidate retirement");
    }
    if (pooled)
      e.poolCorrect = candidateCorrect;
    else
      e.correct = candidateCorrect;
    e.admitted = qualifiedLocalBackend(nativeSamples, dispatchSamples,
                                       candidateCorrect) == LocalBackend::Host;
  };
  for (size_t index = 0; index < entries.size(); ++index) measure(index, {});
  std::set<std::string> previous;
  for (;;) {
    std::set<std::string> pool;
    for (auto const& e : entries)
      if (e.admitted) pool.insert(e.executorKey);
    if (pool.size() < 2) {
      for (auto& e : entries)
        if (e.admitted) {
          e.poolNativeSamples.clear();
          e.poolDispatchSamples.clear();
          e.poolCorrect = false;
        }
      break;
    }
    if (pool == previous) break;
    previous = pool;
    for (size_t index = 0; index < entries.size(); ++index)
      if (entries[index].admitted) measure(index, pool);
  }
  {
    int stable = 1;
    try {
      Internal scope;
      baseline.backend->prepare();
    } catch (...) {
      stable = 0;
    }
    agree(stable, "native ABI baseline configuration changed");
  }
  identityReady = 1;
  try {
    identityReady = identity() == preparedIdentity;
  } catch (...) {
    identityReady = 0;
  }
  agree(identityReady,
        "ABI topology/software/configuration changed during qualification");
  any = false;
  for (auto const& e : entries)
    if (e.admitted) {
      any = true;
      if (!executors.count(e.executorKey)) executors[e.executorKey] = create(e);
    }
  preparing = false;
  localNative = nativeAgreed = !any;
}

template <class F>
ncclResult_t collective(ncclComm_t comm, int op, void const* in, void* out,
                        size_t count, ncclDataType_t dtype, bool sum,
                        cudaStream_t stream, F fallback) {
  return boundary([&] {
    if (internalDepth) {
      return fallback();
    }
    auto state = lookup(comm);
    if (!state) {
      Internal scope;
      return fallback();
    }
    if (!count) {
      // Native zero-count calls may complete without peer participation.
      // Do not turn them into a CPU rendezvous or add a CUDA dependency.
      state->localNative = true;
      ++state->nativeCalls;
      Internal scope;
      return fallback();
    }
    return state->dispatch(op, in, out, count, dtype, sum, stream, fallback);
  });
}
template <class F>
ncclResult_t point(ncclComm_t comm, cudaStream_t stream, F fallback) {
  return boundary([&] {
    if (!internalDepth)
      if (auto state = lookup(comm)) {
        if (state->preparing) throw std::logic_error("ABI preparation active");
        state->beforeNative(stream);
        ++state->nativeCalls;
      }
    Internal scope;
    return fallback();
  });
}
void forget(ncclComm_t comm, bool aborting) {
  auto state = lookup(comm);
  if (!state) {
    std::lock_guard<std::mutex> lock(registryMutex);
    automaticNotes.erase(comm);
    return;
  }
  if (aborting) {
    state->aborted = true;
    bool active = state->preparing || !state->executors.empty();
    if (active) abortedPreparedProcess = true;
    // Submitted UCCL kernels cannot be cancelled by native NCCL. Do not wait
    // or free their storage during fault handling; the process owns cleanup.
    if (active || state->appWork) (void)new std::shared_ptr<State>(state);
  }
  std::lock_guard<std::mutex> lock(registryMutex);
  registry.erase(comm);
  automaticNotes.erase(comm);
}
void automaticPrepare(std::vector<integration::AutoInitRecord> const& records) {
  if (records.empty()) return;
  integration::AutoInitApi functions{
      api().get<decltype(&ncclCommCount)>("ncclCommCount"),
      api().get<decltype(&ncclCommUserRank)>("ncclCommUserRank"),
      api().get<decltype(&ncclCommCuDevice)>("ncclCommCuDevice"),
      api().get<decltype(&ncclAllGather)>("ncclAllGather"),
      api().get<decltype(&ncclGroupStart)>("ncclGroupStart"),
      api().get<decltype(&ncclGroupEnd)>("ncclGroupEnd"),
      &ucclLiteGetUniqueId,
      &ucclLiteNcclPrepare,
      [] {
        std::lock_guard<std::mutex> lock(registryMutex);
        if (abortedPreparedProcess) return false;
        for (auto const& s : registry)
          if (s.second->preparing || !s.second->executors.empty()) return false;
        return true;
      },
      [](std::vector<integration::AutoInitCompletion> const& participants) {
        std::vector<std::future<void>> pending;
        for (auto participant : participants)
          pending.push_back(std::async(std::launch::async, [participant] {
            bootstrap::HostControl completion(
                participant.root.c_str(), &participant.id,
                sizeof(participant.id), participant.rank, participant.ranks,
                120000);
            int ready = 1;
            std::vector<int> peers(participant.ranks);
            completion.allGather(&ready, peers.data(), sizeof(ready));
            for (auto peer : peers)
              if (peer != 1)
                throw std::runtime_error("invalid initialization completion");
          }));
        for (auto& result : pending) result.get();
      },
      {},
      {}};
  auto version = api().get<decltype(&ncclGetVersion)>("ncclGetVersion");
  Dl_info native{};
  if (!dladdr(reinterpret_cast<void*>(version), &native) || !native.dli_fname)
    throw std::runtime_error(
        "native library identity unavailable for automatic init");
  functions.library = native.dli_fname;
  functions.implementation =
      artifactIdentity(reinterpret_cast<uintptr_t>(&automaticPrepare));
  auto results = integration::prepareNativeInitializations(records, functions);
  ncclResult_t status = ncclSuccess;
  std::string reason;
  for (auto const& r : results) {
    std::lock_guard<std::mutex> lock(registryMutex);
    automaticNotes[r.comm] = r.reason;
    if (r.status != ncclSuccess) {
      status = r.status;
      reason = r.reason;
    }
  }
  if (status != ncclSuccess) throw NativeError(status, reason.c_str());
}
void initialized(ncclComm_t comm, ncclConfig_t const* config, bool grouped) {
  if (!integration::automaticInitEnabled() || !comm) return;
  auto record = integration::autoInitRecord(comm, config, grouped);
  if (grouped)
    pendingInitializations.push_back(std::move(record));
  else
    automaticPrepare({record});
}
}  // namespace

extern "C" {
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ranks, ncclUniqueId id,
                              int rank) {
  return boundary([&] {
    bool user = !internalDepth;
    ncclResult_t status;
    {
      Internal scope;
      static auto fn =
          api().get<decltype(&ncclCommInitRank)>("ncclCommInitRank");
      status = fn(comm, ranks, id, rank);
    }
    if (user && status == ncclSuccess && comm)
      initialized(*comm, nullptr, groupDepth != 0);
    return status;
  });
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int ranks,
                                    ncclUniqueId id, int rank,
                                    ncclConfig_t* config) {
  return boundary([&] {
    bool user = !internalDepth;
    ncclResult_t status;
    {
      Internal scope;
      static auto fn = api().get<decltype(&ncclCommInitRankConfig)>(
          "ncclCommInitRankConfig");
      status = fn(comm, ranks, id, rank, config);
    }
    if (user && status == ncclSuccess && comm)
      initialized(*comm, config, groupDepth != 0);
    return status;
  });
}
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int count, int const* devices) {
  return boundary([&] {
    Internal scope;
    static auto fn = api().get<decltype(&ncclCommInitAll)>("ncclCommInitAll");
    return fn(comms, count, devices);
  });
}
ncclResult_t ucclLiteNcclPrepare(ncclComm_t comm, ucclLiteUniqueId const* id,
                                 ucclLiteOptions const* o,
                                 ucclLiteRequest const* requests,
                                 size_t count) {
  return boundary([&] {
    if (groupDepth || !comm || !id || !o || o->version != 1 ||
        o->structSize != sizeof(*o) || !o->nativeLibrary ||
        !o->configurationIdentity || !count || count > 1024 || !requests ||
        (o->flags & ~63u))
      throw std::invalid_argument("invalid prepared NCCL ABI configuration");
    if (abortedPreparedProcess)
      throw std::logic_error(
          "prepared abort requires a new process before acceleration can be "
          "re-enabled");
    auto nativeVersion = api().get<decltype(&ncclGetVersion)>("ncclGetVersion");
    void* explicitLibrary = dlopen(o->nativeLibrary, RTLD_NOW | RTLD_LOCAL);
    if (!explicitLibrary) throw std::runtime_error(dlerror());
    auto expected = dlsym(explicitLibrary, "ncclGetVersion");
    bool same =
        expected &&
        artifactIdentity(reinterpret_cast<uintptr_t>(expected)) ==
            artifactIdentity(reinterpret_cast<uintptr_t>(nativeVersion));
    dlclose(explicitLibrary);
    if (!same)
      throw std::invalid_argument(
          "prepared baseline differs from forwarded native NCCL");
    ScalarType scalar = o->dtype == ncclFloat32    ? ScalarType::Float32
                        : o->dtype == ncclBfloat16 ? ScalarType::BFloat16
                        : o->dtype == ncclFloat16  ? ScalarType::Float16
                                                   : ScalarType(-1);
    if (!scalarBytes(scalar))
      throw std::invalid_argument("unsupported prepared dtype");
    NetworkId bootstrap;
    std::memcpy(bootstrap.data(), id, sizeof(*id));
    RuntimeContext context;
    {
      Internal scope;
      context = integration::nativeNcclContext(comm, o->nativeLibrary,
                                               bootstrap, o->capacityBytes);
    }
    if (context.ranks < 1 || context.ranks > 32)
      throw std::invalid_argument("prepared ABI supports 1..32 ranks");
    auto state = std::make_shared<State>(comm, context, o->dtype, scalar);
    {
      std::lock_guard<std::mutex> lock(registryMutex);
      if (registry.count(comm))
        throw std::invalid_argument(
            "communicator already prepared or native-owned");
      for (auto const& item : registry)
        if (!item.second->executors.empty() || item.second->preparing)
          throw std::invalid_argument(
              "one active prepared communicator per process");
      registry.emplace(comm, state);
    }
    try {
      state->control =
          std::make_shared<mscclpp::TcpBootstrap>(context.rank, context.ranks);
      state->control->initialize(bootstrap, 120);
      struct Description {
        size_t count, capacity;
        int dtype, flags;
      } description{count, o->capacityBytes, int(o->dtype), int(o->flags)};
      for (auto d : state->gather(description))
        if (d.count != count || d.capacity != description.capacity ||
            d.dtype != description.dtype || d.flags != description.flags)
          throw std::invalid_argument("ABI preparation policy mismatch");
      PreparationOptions options;
      options.dtype = scalar;
      options.fallbackOnly = o->flags & UCCL_LITE_FALLBACK_ONLY;
      options.allowPeer = o->flags & UCCL_LITE_ALLOW_PEER;
      options.allowHost = o->flags & UCCL_LITE_ALLOW_HOST;
      options.allowPeerPacket = o->flags & UCCL_LITE_ALLOW_PEER_PACKET;
      options.allowNetwork = o->flags & UCCL_LITE_ALLOW_NETWORK;
      options.probeResources = o->flags & UCCL_LITE_PROBE_RESOURCES;
      if (o->hardwareProfile) options.hardwareProfile = o->hardwareProfile;
      {
        Internal scope;
        options.fallback = integration::makeNativeNcclFallback(
            context, comm, o->nativeLibrary, o->configurationIdentity, scalar);
      }
      state->baseline = options.fallback;
      auto coreContext = context;
      coreContext.id = state->id();
      {
        Internal scope;
        PreparedRuntime core(coreContext, options);
        std::vector<size_t> corePlans;
        for (size_t i = 0; i < count; ++i) {
          auto const& r = requests[i];
          CollectiveRequest request;
          request.operation = NetworkOperation(r.operation);
          request.count = r.count;
          request.fallbackOnly = r.fallbackOnly != 0;
          if (r.layouts && r.ranks == size_t(context.ranks) &&
              r.fallbackOnly <= 1)
            for (size_t rank = 0; rank < r.ranks; ++rank)
              request.layouts.push_back({r.layouts[rank].inPlace != 0,
                                         r.layouts[rank].inPlace > 1
                                             ? 0
                                             : r.layouts[rank].alignment});
          bool duplicate = false;
          for (auto const& previous : state->entries) {
            bool same =
                previous.request.operation == request.operation &&
                previous.request.count == request.count &&
                previous.request.layouts.size() == request.layouts.size();
            for (size_t rank = 0; same && rank < request.layouts.size(); ++rank)
              same &= previous.request.layouts[rank].inPlace ==
                          request.layouts[rank].inPlace &&
                      previous.request.layouts[rank].alignment ==
                          request.layouts[rank].alignment;
            duplicate |= same;
          }
          state->agree(!duplicate,
                       "duplicate or conflicting ABI request signature");
          auto plan = core.prepare(request);
          corePlans.push_back(plan);
          Entry e;
          e.request = request;
          e.core = core.selection(plan);
          e.executorKey = std::to_string(int(e.core.backend));
          for (auto const& candidate : e.core.candidates)
            if (e.core.backend == RuntimeBackend::PeerTree &&
                candidate.backend == RuntimeBackend::PeerTree)
              for (auto p : candidate.peerTreeParents)
                e.executorKey += ':' + std::to_string(p);
          state->entries.push_back(std::move(e));
        }
        for (size_t i = 0; i < state->entries.size(); ++i) {
          auto& entry = state->entries[i];
          entry.core = core.selection(corePlans[i]);
          entry.executorKey = std::to_string(int(entry.core.backend));
          for (auto const& candidate : entry.core.candidates)
            if (entry.core.backend == RuntimeBackend::PeerTree &&
                candidate.backend == RuntimeBackend::PeerTree)
              for (auto p : candidate.peerTreeParents)
                entry.executorKey += ':' + std::to_string(p);
        }
        // Per-process CUDA ordinals need not match another rank's view.
        int visible;
        cudaCheck(cudaGetDeviceCount(&visible));
        std::map<std::string, int> byUuid;
        for (int device = 0; device < visible; ++device) {
          cudaDeviceProp property{};
          cudaCheck(cudaGetDeviceProperties(&property, device));
          std::ostringstream uuid;
          for (unsigned char byte : property.uuid.bytes)
            uuid << std::hex << std::setw(2) << std::setfill('0')
                 << unsigned(byte);
          byUuid.emplace(uuid.str(), device);
        }
        for (auto const& participant : core.placement().ranks)
          if (participant.host == core.placement().ranks[context.rank].host) {
            auto found = byUuid.find(participant.gpu);
            if (found != byUuid.end()) state->devices.push_back(found->second);
          }
        if (core.placement().domain == RuntimeDomain::Distributed)
          std::sort(state->devices.begin(), state->devices.end());
      }
      state->qualify();
      return ncclSuccess;
    } catch (...) {
      std::lock_guard<std::mutex> lock(registryMutex);
      registry.erase(comm);
      throw;
    }
  });
}
ncclResult_t ucclLiteNcclRelease(ncclComm_t comm) {
  return boundary([&] {
    if (groupDepth) throw std::logic_error("release inside group");
    auto s = lookup(comm);
    if (!s) throw std::invalid_argument("communicator not prepared");
    s->control->barrier();
    s->retire();
    return ncclSuccess;
  });
}
ncclResult_t ucclLiteNcclReport(ncclComm_t comm, char* buffer, size_t* bytes) {
  return boundary([&] {
    auto s = lookup(comm);
    std::string note;
    {
      std::lock_guard<std::mutex> lock(registryMutex);
      auto n = automaticNotes.find(comm);
      if (n != automaticNotes.end()) note = n->second;
    }
    if ((!s && note.empty()) || !bytes)
      throw std::invalid_argument("invalid ABI report arguments");
    std::ostringstream out;
    if (!s) {
      out << "{\"automatic_init\":true,\"native_owned\":true,\"accelerated_"
             "resources\":false,\"init_status\":";
      jsonString(out, note);
      out << ",\"plans\":[]}";
    } else {
      out << std::setprecision(17)
          << "{\"native_owned\":" << (s->nativeAgreed ? "true" : "false")
          << ",\"accelerated_resources\":"
          << (!s->executors.empty() ? "true" : "false")
          << ",\"accelerated_calls\":" << s->acceleratedCalls
          << ",\"native_calls\":" << s->nativeCalls
          << ",\"guard_calls\":" << s->guardCalls
          << ",\"automatic_init\":" << (!note.empty() ? "true" : "false")
          << ",\"init_status\":";
      jsonString(out, note);
      out << ",\"baseline_identity\":";
      jsonString(out, s->baseline.identity);
      out << ",\"abi_identity\":";
      jsonString(out, artifactIdentity(reinterpret_cast<uintptr_t>(&abiCall)));
      out << ",\"scope\":\"host-max-completion;raw-native-vs-guard-plus-"
             "executor;"
             "3-ABBA;2-warmups;3-timed\",\"plans\":[";
      for (size_t i = 0; i < s->entries.size(); ++i) {
        if (i) out << ',';
        auto const& e = s->entries[i];
        out << "{\"operation\":" << int(e.request.operation)
            << ",\"count\":" << e.request.count
            << ",\"core_backend\":" << int(e.core.backend) << ",\"attempted\":"
            << (!e.nativeSamples.empty() ? "true" : "false")
            << ",\"correct\":" << (e.correct ? "true" : "false")
            << ",\"admitted\":" << (e.admitted ? "true" : "false");
        out << ",\"pool_correct\":" << (e.poolCorrect ? "true" : "false");
        for (auto p : {std::make_pair("native_us", &e.nativeSamples),
                       {"dispatch_us", &e.dispatchSamples},
                       {"pool_native_us", &e.poolNativeSamples},
                       {"pool_dispatch_us", &e.poolDispatchSamples}}) {
          out << ",\"" << p.first << "\":[";
          for (size_t j = 0; j < p.second->size(); ++j) {
            if (j) out << ',';
            out << (*p.second)[j];
          }
          out << ']';
        }
        out << '}';
      }
      out << "]}";
    }
    auto text = out.str();
    size_t available = buffer ? *bytes : 0;
    *bytes = text.size() + 1;
    if (buffer) {
      if (available < *bytes)
        throw std::invalid_argument("ABI report buffer too small");
      std::memcpy(buffer, text.c_str(), *bytes);
    }
    return ncclSuccess;
  });
}
ncclResult_t ncclAllReduce(void const* in, void* out, size_t count,
                           ncclDataType_t dtype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  return collective(comm, 3, in, out, count, dtype, op == ncclSum, stream, [&] {
    static auto fn = api().get<decltype(&ncclAllReduce)>("ncclAllReduce");
    return fn(in, out, count, dtype, op, comm, stream);
  });
}
ncclResult_t ncclAllGather(void const* in, void* out, size_t count,
                           ncclDataType_t dtype, ncclComm_t comm,
                           cudaStream_t stream) {
  return collective(comm, 1, in, out, count, dtype, true, stream, [&] {
    static auto fn = api().get<decltype(&ncclAllGather)>("ncclAllGather");
    return fn(in, out, count, dtype, comm, stream);
  });
}
ncclResult_t ncclReduceScatter(void const* in, void* out, size_t count,
                               ncclDataType_t dtype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
  return collective(comm, 2, in, out, count, dtype, op == ncclSum, stream, [&] {
    static auto fn =
        api().get<decltype(&ncclReduceScatter)>("ncclReduceScatter");
    return fn(in, out, count, dtype, op, comm, stream);
  });
}
ncclResult_t ncclBroadcast(void const* in, void* out, size_t count,
                           ncclDataType_t dtype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  return collective(comm, 4, in, out, count, dtype, false, stream, [&] {
    static auto fn = api().get<decltype(&ncclBroadcast)>("ncclBroadcast");
    return fn(in, out, count, dtype, root, comm, stream);
  });
}
ncclResult_t ncclBcast(void* buffer, size_t count, ncclDataType_t dtype,
                       int root, ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buffer, buffer, count, dtype, root, comm, stream);
}
ncclResult_t ncclReduce(void const* in, void* out, size_t count,
                        ncclDataType_t dtype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  return collective(comm, 5, in, out, count, dtype, false, stream, [&] {
    static auto fn = api().get<decltype(&ncclReduce)>("ncclReduce");
    return fn(in, out, count, dtype, op, root, comm, stream);
  });
}
ncclResult_t ncclAlltoAll(void const* in, void* out, size_t count,
                          ncclDataType_t dtype, ncclComm_t comm,
                          cudaStream_t stream) {
  return collective(comm, 6, in, out, count, dtype, false, stream, [&] {
    static auto fn = api().get<decltype(&ncclAlltoAll)>("ncclAlltoAll");
    return fn(in, out, count, dtype, comm, stream);
  });
}
ncclResult_t ncclGather(void const* in, void* out, size_t count,
                        ncclDataType_t dtype, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  return collective(comm, 7, in, out, count, dtype, false, stream, [&] {
    static auto fn = api().get<decltype(&ncclGather)>("ncclGather");
    return fn(in, out, count, dtype, root, comm, stream);
  });
}
ncclResult_t ncclScatter(void const* in, void* out, size_t count,
                         ncclDataType_t dtype, int root, ncclComm_t comm,
                         cudaStream_t stream) {
  return collective(comm, 8, in, out, count, dtype, false, stream, [&] {
    static auto fn = api().get<decltype(&ncclScatter)>("ncclScatter");
    return fn(in, out, count, dtype, root, comm, stream);
  });
}
ncclResult_t ncclSend(void const* in, size_t count, ncclDataType_t dtype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  return point(comm, stream, [&] {
    static auto fn = api().get<decltype(&ncclSend)>("ncclSend");
    return fn(in, count, dtype, peer, comm, stream);
  });
}
ncclResult_t ncclRecv(void* out, size_t count, ncclDataType_t dtype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
  return point(comm, stream, [&] {
    static auto fn = api().get<decltype(&ncclRecv)>("ncclRecv");
    return fn(out, count, dtype, peer, comm, stream);
  });
}
ncclResult_t ncclGroupStart() {
  return boundary([&] {
    bool user = !internalDepth;
    Internal scope;
    static auto fn = api().get<decltype(&ncclGroupStart)>("ncclGroupStart");
    auto result = fn();
    if (user && result == ncclSuccess) ++groupDepth;
    return result;
  });
}
ncclResult_t ncclGroupEnd() {
  return boundary([&] {
    bool user = !internalDepth;
    static auto fn = api().get<decltype(&ncclGroupEnd)>("ncclGroupEnd");
    ncclResult_t result;
    {
      Internal scope;
      result = fn();
    }
    if (user && groupDepth) --groupDepth;
    if (user && !groupDepth && !pendingInitializations.empty()) {
      auto pending = std::move(pendingInitializations);
      pendingInitializations.clear();
      if (result == ncclSuccess) automaticPrepare(pending);
    }
    return result;
  });
}
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* info) {
  return boundary([&] {
    bool user = !internalDepth;
    Internal scope;
    static auto fn =
        api().get<decltype(&ncclGroupSimulateEnd)>("ncclGroupSimulateEnd");
    auto result = fn(info);
    if (user && groupDepth) --groupDepth;
    if (user && !groupDepth) pendingInitializations.clear();
    return result;
  });
}
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  return boundary([&] {
    bool user = !internalDepth;
    auto state = user ? lookup(comm) : nullptr;
    if (state) state->retire();
    Internal scope;
    static auto fn = api().get<decltype(&ncclCommDestroy)>("ncclCommDestroy");
    auto result = fn(comm);
    if (user && result == ncclSuccess) forget(comm, false);
    return result;
  });
}
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  return boundary([&] {
    if (!internalDepth) forget(comm, true);
    Internal scope;
    static auto fn = api().get<decltype(&ncclCommAbort)>("ncclCommAbort");
    return fn(comm);
  });
}
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  return boundary([&] {
    if (!internalDepth)
      if (auto state = lookup(comm)) state->retire();
    Internal scope;
    static auto fn = api().get<decltype(&ncclCommFinalize)>("ncclCommFinalize");
    return fn(comm);
  });
}
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t* child, ncclConfig_t* config) {
  return boundary([&] {
    if (!internalDepth)
      if (auto state = lookup(comm)) state->retire();
    Internal scope;
    static auto fn = api().get<decltype(&ncclCommSplit)>("ncclCommSplit");
    return fn(comm, color, key, child, config);
  });
}
ncclResult_t ncclCommRevoke(ncclComm_t comm, int flags) {
  return boundary([&] {
    if (!internalDepth)
      if (auto state = lookup(comm))
        if (!state->nativeAgreed || !state->executors.empty())
          throw std::logic_error(
              "release prepared resources before native revoke");
    Internal scope;
    static auto fn = api().get<decltype(&ncclCommRevoke)>("ncclCommRevoke");
    return fn(comm, flags);
  });
}
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excluded, int excludedCount,
                            ncclComm_t* child, ncclConfig_t* config,
                            int flags) {
  return boundary([&] {
    if (!internalDepth)
      if (auto state = lookup(comm))
        if (!state->nativeAgreed || !state->executors.empty())
          throw std::logic_error(
              "release prepared resources before native shrink");
    Internal scope;
    static auto fn = api().get<decltype(&ncclCommShrink)>("ncclCommShrink");
    return fn(comm, excluded, excludedCount, child, config, flags);
  });
}
char const* ncclGetLastError(ncclComm_t comm) {
  if (!lastError.empty()) return lastError.c_str();
  try {
    return api().get<decltype(&ncclGetLastError)>("ncclGetLastError")(comm);
  } catch (...) {
    return "prepared NCCL ABI error lookup failed";
  }
}
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* status) {
  return boundary([&] {
    bool user = !internalDepth;
    Internal scope;
    static auto fn =
        api().get<decltype(&ncclCommGetAsyncError)>("ncclCommGetAsyncError");
    auto result = fn(comm, status);
    if (user && result == ncclSuccess && status && *status == ncclSuccess)
      if (auto state = lookup(comm))
        if (!state->preparing && state->appWork && state->completion) {
          auto ready = cudaEventQuery(state->completion->event);
          if (ready == cudaErrorNotReady) {
            auto prior = cudaGetLastError();
            if (prior != cudaSuccess && prior != cudaErrorNotReady)
              *status = ncclUnhandledCudaError;
          }
          if (ready != cudaSuccess && ready != cudaErrorNotReady)
            *status = ncclUnhandledCudaError;
        }
    return result;
  });
}
#define UCCL_PROFILE_ALIAS(name)            \
  extern __typeof__(nccl##name) pnccl##name \
      __attribute__((alias("nccl" #name)));
UCCL_PROFILE_ALIAS(AllReduce)
UCCL_PROFILE_ALIAS(CommInitRank)
UCCL_PROFILE_ALIAS(CommInitRankConfig)
UCCL_PROFILE_ALIAS(CommInitAll)
UCCL_PROFILE_ALIAS(AllGather)
UCCL_PROFILE_ALIAS(ReduceScatter)
UCCL_PROFILE_ALIAS(Broadcast)
UCCL_PROFILE_ALIAS(Bcast)
UCCL_PROFILE_ALIAS(Reduce)
UCCL_PROFILE_ALIAS(AlltoAll)
UCCL_PROFILE_ALIAS(Gather)
UCCL_PROFILE_ALIAS(Scatter)
UCCL_PROFILE_ALIAS(Send)
UCCL_PROFILE_ALIAS(Recv)
UCCL_PROFILE_ALIAS(GroupStart)
UCCL_PROFILE_ALIAS(GroupEnd)
UCCL_PROFILE_ALIAS(GroupSimulateEnd)
UCCL_PROFILE_ALIAS(CommDestroy)
UCCL_PROFILE_ALIAS(CommAbort)
UCCL_PROFILE_ALIAS(CommFinalize)
UCCL_PROFILE_ALIAS(CommSplit)
UCCL_PROFILE_ALIAS(CommRevoke)
UCCL_PROFILE_ALIAS(CommShrink)
UCCL_PROFILE_ALIAS(GetLastError)
UCCL_PROFILE_ALIAS(CommGetAsyncError)
#undef UCCL_PROFILE_ALIAS
}

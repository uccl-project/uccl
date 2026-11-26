#include "uccl_lite/multiprocess_collectives.h"
#include "nccl_fusion.h"

#include <nccl.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#ifndef UCCL_LITE_REAL_NCCL
#define UCCL_LITE_REAL_NCCL "libnccl.so.2"
#endif

namespace {
void* realLibrary() {
  static void* library = [] {
    const char* path = std::getenv("UCCL_LITE_REAL_NCCL");
    if (!path || !*path) path = UCCL_LITE_REAL_NCCL;
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      std::fprintf(stderr, "UCCL Lite NCCL proxy: dlopen %s: %s\n", path,
                   dlerror());
      std::abort();
    }
    return handle;
  }();
  return library;
}
template <typename Fn> Fn real(const char* name) {
  auto* symbol = dlsym(realLibrary(), name);
  if (!symbol) {
    std::fprintf(stderr, "UCCL Lite NCCL proxy: missing %s\n", name);
    std::abort();
  }
  return reinterpret_cast<Fn>(symbol);
}

struct State {
  std::string id;
  int ranks;
  int rank;
  void* collective = nullptr;
  void* readyCollective = nullptr;
  cublasHandle_t gateBlas = nullptr;
  std::mutex lock;
  uint64_t agCalls = 0;
  uint64_t agGemmCalls = 0;
  uint64_t rsCalls = 0;
  uint64_t rsSum2Calls = 0;
  uint64_t rsReadyCalls = 0;
  uint64_t arCalls = 0;
  std::map<std::pair<int, size_t>, uint64_t> shapeCalls;
  ~State() {
    if (gateBlas) cublasDestroy(gateBlas);
    if (readyCollective) uccl_lite_mp_destroy(readyCollective);
    if (collective) {
      uccl_lite_mp_destroy(collective);
      std::fprintf(stderr,
                   "UCCL Lite NCCL proxy: rank=%d completed AG=%llu RS=%llu AR=%llu AG_GEMM=%llu RS_SUM2=%llu RS_READY=%llu\n",
                   rank, static_cast<unsigned long long>(agCalls),
                   static_cast<unsigned long long>(rsCalls),
                   static_cast<unsigned long long>(arCalls),
                   static_cast<unsigned long long>(agGemmCalls),
                   static_cast<unsigned long long>(rsSum2Calls),
                   static_cast<unsigned long long>(rsReadyCalls));
      for (const auto& [shape, calls] : shapeCalls)
        std::fprintf(stderr,
                     "UCCL Lite NCCL proxy SHAPE rank=%d op=%s count=%zu calls=%llu comm=%s\n",
                     rank,
                     shape.first == 1 ? "AG" : shape.first == 2 ? "RS" : "AR",
                     shape.second, static_cast<unsigned long long>(calls),
                     id.c_str());
    }
  }
};
std::mutex statesLock;
std::unordered_map<ncclComm_t, std::shared_ptr<State>> states;
thread_local int groupDepth = 0;
bool enabled();

std::string hashId(ncclUniqueId id) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : id.internal) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  char result[17];
  std::snprintf(result, sizeof(result), "%016llx",
                static_cast<unsigned long long>(hash));
  return result;
}
std::shared_ptr<State> stateFor(ncclComm_t comm) {
  std::lock_guard<std::mutex> guard(statesLock);
  auto it = states.find(comm);
  return it == states.end() ? nullptr : it->second;
}
void remember(ncclComm_t comm, int ranks, ncclUniqueId id, int rank) {
  if (!enabled()) return;
  auto state = std::make_shared<State>();
  state->id = hashId(id);
  state->ranks = ranks;
  state->rank = rank;
  std::lock_guard<std::mutex> guard(statesLock);
  states[comm] = state;
  std::fprintf(stderr, "UCCL Lite NCCL proxy: communicator rank=%d/%d id=%s\n",
               rank, ranks, state->id.c_str());
}
bool enabled() {
  const char* value = std::getenv("UCCL_LITE_NCCL_ENABLE");
  return value && std::strcmp(value, "1") == 0;
}
size_t maxBytes() {
  const char* value = std::getenv("UCCL_LITE_NCCL_MAX_BYTES");
  if (!value || !*value) return 32UL << 20;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (!end || *end || !parsed || parsed > (1ULL << 30)) return 0;
  return static_cast<size_t>(parsed);
}
const char* operationName(int operation) {
  return operation == 1 ? "AG" : operation == 2 ? "RS" : "AR";
}
ncclResult_t dispatch(int operation, const void* send, void* recv, size_t count,
                      ncclDataType_t dtype, ncclRedOp_t reduction,
                      ncclComm_t comm, cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 ||
      dtype != ncclBfloat16 ||
      (operation != 1 && operation != 2 && operation != 3) ||
      (operation != 1 && reduction != ncclSum) || !capacity ||
      count > capacity /
                  (sizeof(uint16_t) *
                   (operation == 3 ? 1 : size_t(s->ranks))) ||
      !send || !recv)
    return ncclInvalidUsage;
  std::lock_guard<std::mutex> guard(s->lock);
  if (!s->collective) {
    s->collective = uccl_lite_mp_create(s->id.c_str(), s->ranks, s->rank,
                                         capacity);
    if (!s->collective) {
      std::fprintf(stderr, "UCCL Lite NCCL proxy: create: %s\n",
                   uccl_lite_mp_last_error());
      return ncclSystemError;
    }
  }
  if (uccl_lite_mp_enqueue(s->collective, operation, send, recv, count,
                            stream) != 0) {
    std::fprintf(stderr, "UCCL Lite NCCL proxy: enqueue: %s\n",
                 uccl_lite_mp_last_error());
    return ncclSystemError;
  }
  uint64_t calls = operation == 1   ? ++s->agCalls
                   : operation == 2 ? ++s->rsCalls
                                    : ++s->arCalls;
  const char* traceShapes = std::getenv("UCCL_LITE_NCCL_TRACE_SHAPES");
  if (s->rank == 0 && traceShapes && std::strcmp(traceShapes, "1") == 0) {
    auto& shapeCalls = s->shapeCalls[{operation, count}];
    if (++shapeCalls == 1)
      std::fprintf(stderr,
                   "UCCL Lite NCCL proxy SHAPE_FIRST rank=0 op=%s count=%zu comm=%s\n",
                   operationName(operation), count, s->id.c_str());
  }
  if (calls == 1)
    std::fprintf(stderr, "UCCL Lite NCCL proxy: rank=%d op=%s BF16 count=%zu\n",
                 s->rank, operationName(operation), count);
  const char* trace = std::getenv("UCCL_LITE_NCCL_TRACE");
  if (s->rank == 0 && trace && std::strcmp(trace, "1") == 0)
    std::fprintf(stderr,
                 "UCCL Lite NCCL proxy TRACE rank=0 op=%s calls=%llu "
                 "ag_calls=%llu rs_calls=%llu ar_calls=%llu count=%zu comm=%s\n",
                 operationName(operation),
                 static_cast<unsigned long long>(calls),
                 static_cast<unsigned long long>(s->agCalls),
                 static_cast<unsigned long long>(s->rsCalls),
                 static_cast<unsigned long long>(s->arCalls), count,
                 s->id.c_str());
  return ncclSuccess;
}
}  // namespace

extern "C" {
ncclResult_t uccl_lite_nccl_prepare_rs_ready(ncclComm_t comm) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 ||
      capacity < 8 * 4096 * sizeof(uint16_t))
    return ncclInvalidUsage;
  std::lock_guard<std::mutex> guard(s->lock);
  if (!s->readyCollective) {
    std::string id = s->id + "-rs-ready";
    const char* gpuOnly = std::getenv("UCCL_LITE_READY_RS_GPU_ONLY");
    s->readyCollective = gpuOnly && std::strcmp(gpuOnly, "1") == 0
                             ? uccl_lite_mp_create_rs_ready(id.c_str(),
                                                             s->ranks, s->rank)
                             : uccl_lite_mp_create(id.c_str(), s->ranks,
                                                   s->rank, capacity);
    if (!s->readyCollective) {
      std::fprintf(stderr, "UCCL Lite NCCL proxy: ready RS prepare: %s\n",
                   uccl_lite_mp_last_error());
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

ncclResult_t uccl_lite_nccl_reduce_scatter_ready_sum2(
    const void* input, const void* auxiliary, void* output,
    const uint32_t* ready, ncclComm_t comm, cudaStream_t stream) {
  auto s = stateFor(comm);
  if (!enabled() || !s || groupDepth || s->ranks != 8 || !input || !auxiliary ||
      !output || !ready)
    return ncclInvalidUsage;
  std::lock_guard<std::mutex> guard(s->lock);
  if (!s->readyCollective) return ncclInvalidUsage;
  if (uccl_lite_mp_enqueue_rs_ready_sum2(
          s->readyCollective, input, auxiliary, output, ready, stream) != 0) {
    std::fprintf(stderr, "UCCL Lite NCCL proxy: ready RS enqueue: %s\n",
                 uccl_lite_mp_last_error());
    return ncclSystemError;
  }
  if (++s->rsReadyCalls == 1)
    std::fprintf(stderr,
                 "UCCL Lite NCCL proxy: rank=%d first RS_READY_SUM2 BF16 count=4096\n",
                 s->rank);
  return ncclSuccess;
}

ncclResult_t uccl_lite_nccl_all_gather_gemm_128x4096(
    const void* input, void* gathered, const void* weight, void* gemmOutput,
    ncclComm_t comm, cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 || !input || !gathered ||
      !weight || !gemmOutput || capacity < 8 * 4096 * sizeof(uint16_t))
    return ncclInvalidUsage;
  std::lock_guard<std::mutex> guard(s->lock);
  const char* backend = std::getenv("UCCL_LITE_AG_GEMM_BACKEND");
  bool useCublas = backend && std::strcmp(backend, "cublas") == 0;
  if (useCublas && !s->gateBlas &&
      cublasCreate(&s->gateBlas) != CUBLAS_STATUS_SUCCESS)
    return ncclSystemError;
  if (!s->collective) {
    s->collective = uccl_lite_mp_create(s->id.c_str(), s->ranks, s->rank,
                                         capacity);
    if (!s->collective) {
      std::fprintf(stderr, "UCCL Lite NCCL proxy: AG+GEMM create: %s\n",
                   uccl_lite_mp_last_error());
      return ncclSystemError;
    }
  }
  int enqueueStatus =
      useCublas
          ? uccl_lite_mp_enqueue(s->collective, 1, input, gathered, 4096,
                                 stream)
          : uccl_lite_mp_enqueue_ag_gemm_128x4096(
                s->collective, input, gathered, weight, gemmOutput, stream);
  if (enqueueStatus != 0) {
    std::fprintf(stderr, "UCCL Lite NCCL proxy: AG+GEMM enqueue: %s\n",
                 uccl_lite_mp_last_error());
    return ncclSystemError;
  }
  if (useCublas) {
    float alpha = 1.f, beta = 0.f;
    if (cublasSetStream(s->gateBlas, stream) != CUBLAS_STATUS_SUCCESS ||
        cublasGemmEx(s->gateBlas, CUBLAS_OP_T, CUBLAS_OP_N, 128, 8, 4096,
                     &alpha, weight, CUDA_R_16BF, 4096, gathered,
                     CUDA_R_16BF, 4096, &beta, gemmOutput, CUDA_R_16BF, 128,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "UCCL Lite NCCL proxy: AG+GEMM cuBLAS failed\n");
      return ncclSystemError;
    }
  }
  ++s->agCalls;
  if (++s->agGemmCalls == 1)
    std::fprintf(stderr,
                 "UCCL Lite NCCL proxy: rank=%d first AG_GEMM BF16 8x128x4096 backend=%s\n",
                 s->rank, useCublas ? "cublas" : "kernel");
  return ncclSuccess;
}

// SGLang may call this explicit boundary only after its two MoE producers
// finish on stream. It keeps NCCL's communicator identity and sequence order.
ncclResult_t uccl_lite_nccl_reduce_scatter_sum2(
    const void* input, const void* auxiliary, void* output, size_t shardCount,
    ncclComm_t comm, cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 || !input || !auxiliary ||
      !output || !capacity ||
      shardCount > capacity / (sizeof(uint16_t) * size_t(s->ranks)))
    return ncclInvalidUsage;
  std::lock_guard<std::mutex> guard(s->lock);
  if (!s->collective) {
    s->collective = uccl_lite_mp_create(s->id.c_str(), s->ranks, s->rank,
                                         capacity);
    if (!s->collective) {
      std::fprintf(stderr, "UCCL Lite NCCL proxy: sum2 create: %s\n",
                   uccl_lite_mp_last_error());
      return ncclSystemError;
    }
  }
  if (uccl_lite_mp_enqueue_rs_sum2(s->collective, input, auxiliary, output,
                                    shardCount, stream) != 0) {
    std::fprintf(stderr, "UCCL Lite NCCL proxy: sum2 enqueue: %s\n",
                 uccl_lite_mp_last_error());
    return ncclSystemError;
  }
  ++s->rsCalls;
  if (++s->rsSum2Calls == 1)
    std::fprintf(stderr,
                 "UCCL Lite NCCL proxy: rank=%d first RS_SUM2 BF16 count=%zu\n",
                 s->rank, shardCount);
  return ncclSuccess;
}

const char* ncclGetErrorString(ncclResult_t result) {
  return real<decltype(&ncclGetErrorString)>("ncclGetErrorString")(result);
}
ncclResult_t ncclGetVersion(int* version) {
  return real<decltype(&ncclGetVersion)>("ncclGetVersion")(version);
}
ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
  return real<decltype(&ncclGetUniqueId)>("ncclGetUniqueId")(id);
}
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ranks, ncclUniqueId id,
                              int rank) {
  auto result = real<decltype(&ncclCommInitRank)>("ncclCommInitRank")(
      comm, ranks, id, rank);
  if (result == ncclSuccess) remember(*comm, ranks, id, rank);
  return result;
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int ranks,
                                    ncclUniqueId id, int rank,
                                    ncclConfig_t* config) {
  auto result = real<decltype(&ncclCommInitRankConfig)>("ncclCommInitRankConfig")(
      comm, ranks, id, rank, config);
  if (result == ncclSuccess) remember(*comm, ranks, id, rank);
  return result;
}
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  {
    std::lock_guard<std::mutex> guard(statesLock);
    states.erase(comm);
  }
  return real<decltype(&ncclCommDestroy)>("ncclCommDestroy")(comm);
}
ncclResult_t ncclAllGather(const void* send, void* recv, size_t count,
                           ncclDataType_t dtype, ncclComm_t comm,
                           cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 ||
      dtype != ncclBfloat16 || !capacity ||
      count > capacity / (sizeof(uint16_t) * size_t(s->ranks)))
    return real<decltype(&ncclAllGather)>("ncclAllGather")(
        send, recv, count, dtype, comm, stream);
  return dispatch(1, send, recv, count, dtype, ncclSum, comm, stream);
}
ncclResult_t ncclReduceScatter(const void* send, void* recv, size_t count,
                               ncclDataType_t dtype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 ||
      dtype != ncclBfloat16 || op != ncclSum || !capacity ||
      count > capacity / (sizeof(uint16_t) * size_t(s->ranks)))
    return real<decltype(&ncclReduceScatter)>("ncclReduceScatter")(
        send, recv, count, dtype, op, comm, stream);
  return dispatch(2, send, recv, count, dtype, op, comm, stream);
}
ncclResult_t ncclAllReduce(const void* send, void* recv, size_t count,
                           ncclDataType_t dtype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 ||
      dtype != ncclBfloat16 || op != ncclSum || !capacity ||
      count > capacity / sizeof(uint16_t))
    return real<decltype(&ncclAllReduce)>("ncclAllReduce")(
        send, recv, count, dtype, op, comm, stream);
  return dispatch(3, send, recv, count, dtype, op, comm, stream);
}
ncclResult_t ncclReduce(const void* send, void* recv, size_t count,
                        ncclDataType_t dtype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclReduce)>("ncclReduce")(
      send, recv, count, dtype, op, root, comm, stream);
}
ncclResult_t ncclBroadcast(const void* send, void* recv, size_t count,
                           ncclDataType_t dtype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclBroadcast)>("ncclBroadcast")(
      send, recv, count, dtype, root, comm, stream);
}
ncclResult_t ncclSend(const void* send, size_t count, ncclDataType_t dtype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclSend)>("ncclSend")(
      send, count, dtype, peer, comm, stream);
}
ncclResult_t ncclRecv(void* recv, size_t count, ncclDataType_t dtype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclRecv)>("ncclRecv")(
      recv, count, dtype, peer, comm, stream);
}
ncclResult_t ncclGroupStart() {
  ++groupDepth;
  return real<decltype(&ncclGroupStart)>("ncclGroupStart")();
}
ncclResult_t ncclGroupEnd() {
  auto result = real<decltype(&ncclGroupEnd)>("ncclGroupEnd")();
  if (groupDepth) --groupDepth;
  return result;
}
}  // extern "C"

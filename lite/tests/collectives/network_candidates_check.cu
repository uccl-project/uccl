#include "native_fallback.hpp"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/network_cache.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include "uccl_lite/runtime/prepared_network.hpp"
#include <linux/mempolicy.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
using namespace uccl_lite;
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
template <class Bits>
Bits encode(float x) {
  if constexpr (sizeof(Bits) == 4) {
    Bits bits;
    std::memcpy(&bits, &x, 4);
    return bits;
  } else
    return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
template <class Bits>
float decode(Bits x) {
  if constexpr (sizeof(Bits) == 4) {
    float value;
    std::memcpy(&value, &x, 4);
    return value;
  } else
    return __bfloat162float(__ushort_as_bfloat16(x));
}
template <class Bits>
Bits value(size_t i, int rank, int generation, bool gather) {
  if (gather)
    return Bits(uint32_t(i) * 2246822519u + rank * 7919u + generation);
  // Every partial sum is integral and representable for all supported ranks,
  // allowing an independent bitwise oracle for both native and direct paths.
  float x = float(int((i + rank) % 9) - 4 + generation);
  return encode<Bits>(x);
}
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 2000000) {
  }
}
template <class Bits, class Backend>
void exercise(Backend& peer, int rank, int ranks, int op, size_t count,
              bool alias, std::array<cudaStream_t, 2> streams) {
  size_t in = count * (op == 2 ? ranks : 1),
         out = count * (op == 1 ? ranks : 1), extent = std::max(in, out) + 128;
  size_t aOffset = 1 + (alias && op == 1 ? size_t(rank) * count : 0),
         bOffset = 1 + (alias && op == 2 ? size_t(rank) * count : 0);
  bool inputB = alias && op == 1, outputA = alias && op != 1;
  Bits *a, *b;
  check(cudaMalloc(&a, extent * sizeof(Bits)));
  check(cudaMalloc(&b, extent * sizeof(Bits)));
  auto* input = (inputB ? b : a) + aOffset;
  auto* output = (outputA ? a : b) + bOffset;
  for (int generation = 0; generation < 3; ++generation) {
    Bits guard = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
    std::vector<Bits> ha(extent, guard), hb(extent, guard), actual(extent);
    for (size_t i = 0; i < in; ++i)
      (inputB ? hb : ha)[aOffset + i] =
          value<Bits>(i, rank, generation, op == 1);
    // Pageable synchronous H2D may return before DMA completion. Produce on
    // the consumer stream so nonblocking streams have an explicit dependency.
    auto stream = streams[generation % 2];
    check(cudaMemcpyAsync(a, ha.data(), extent * sizeof(Bits),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(b, hb.data(), extent * sizeof(Bits),
                          cudaMemcpyHostToDevice, stream));
    if (rank == ranks - 1) skew<<<1, 32, 0, stream>>>();
    check(cudaGetLastError());
    peer.enqueue(NetworkOperation(op), input, output, count, stream);
    peer.synchronize();
    for (size_t i = 0; i < out; ++i) {
      Bits wanted;
      if (op == 1)
        wanted = value<Bits>(i % count, int(i / count), generation, true);
      else {
        // Independent source-ordered FP32 reference. Cancellation inputs
        // distinguish this contract from a reordered partial-sum tree.
        float sum = 0;
        for (int source = 0; source < ranks; ++source)
          sum += decode(value<Bits>(op == 2 ? size_t(rank) * count + i : i,
                                    source, generation, false));
        wanted = encode<Bits>(sum);
      }
      (outputA ? ha : hb)[bOffset + i] = wanted;
    }
    for (auto pair : {std::make_pair(a, &ha), {b, &hb}}) {
      check(cudaMemcpy(actual.data(), pair.first, extent * sizeof(Bits),
                       cudaMemcpyDeviceToHost));
      if (actual != *pair.second) {
        for (size_t i = 0; i < extent; ++i)
          if (actual[i] != (*pair.second)[i]) {
            std::cerr << "direct exchange mismatch rank=" << rank
                      << " op=" << op << " count=" << count
                      << " alias=" << alias << " generation=" << generation
                      << " buffer=" << (pair.first == a ? "a" : "b")
                      << " index=" << i << " got=" << uint64_t(actual[i])
                      << " expected=" << uint64_t((*pair.second)[i]) << '\n';
            break;
          }
        throw std::runtime_error(
            "direct exchange full-buffer/source-order/input/guard oracle "
            "mismatch");
      }
    }
  }
  check(cudaFree(a));
  check(cudaFree(b));
}

using namespace uccl_lite::runtime;
void nativeCheck(ncclResult_t r) {
  if (r != ncclSuccess) throw std::runtime_error(ncclGetErrorString(r));
}
struct PlanView {
  PreparedNetworkRuntime& runtime;
  size_t plan;
  void enqueue(NetworkOperation, void const* input, void* output, size_t,
               cudaStream_t stream) {
    runtime.enqueue(plan, input, output, stream);
  }
  void synchronize() { runtime.synchronize(); }
};
template <class Bits>
void run(int rank, int ranks, int localRanks, int device, char const* library,
         std::string mode, std::string cache) {
  auto dtype = sizeof(Bits) == 4 ? ScalarType::Float32 : ScalarType::BFloat16;
  std::array<cudaStream_t, 2> streams;
  for (auto& stream : streams)
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (mode == "current") {
    if (!rank) {
      auto saved = loadNetworkPlanCache(cache);
      if (!saved)
        throw std::runtime_error("current-sample fixture needs a cache");
      bool changed = false;
      for (auto record : saved->records)
        if (record.candidate == NetworkCacheCandidate::DirectExchange &&
            record.workload.collective == Collective::AllReduce &&
            record.workload.bytes == 256) {
          record.candidateSamples.assign(
              18, 0.001);  // Synthetic history, never current evidence.
          updateNetworkPlanCache(cache, *saved, record);
          changed = true;
        }
      if (!changed)
        throw std::runtime_error("direct historical fixture missing");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    ncclUniqueId id;
    if (!rank) nativeCheck(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    nativeCheck(ncclCommInitRank(&comm, ranks, id, rank));
    {
      BackendContext context;
      if (!rank) context.id = PreparedNetworkRuntime::createUniqueId();
      MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
      context.rank = rank;
      context.ranks = ranks;
      context.device = device;
      context.capacity = 1UL << 20;
      int visible;
      check(cudaGetDeviceCount(&visible));
      if (visible == 1)
        context.localDevices = {0};
      else
        for (int i = 0; i < localRanks; ++i) context.localDevices.push_back(i);
      NetworkPreparationOptions options;
      options.dtype = dtype;
      options.probeResources = mode != "noprobes";
      options.planCache = cache;
      options.fallbackOnly = mode == "fallback";
      options.allowHierarchy = mode != "direct";
      options.fallback = uccl_lite::integration::makeNativeNcclFallback(
          context, comm, library,
          "network-portfolio-test;fixture-if-delayed-library", dtype);
      PreparedNetworkRuntime runtime(context, options);
      std::vector<size_t> plans;
      for (auto shape : std::vector<std::pair<int, size_t>>{
               {3, 64}, {3, 257}, {1, 257}, {2, 257}}) {
        CollectiveRequest request{
            NetworkOperation(shape.first), shape.second, {}};
        for (int r = 0; r < ranks; ++r)
          request.layouts.push_back({bool(r & 1), unsigned(sizeof(Bits))});
        plans.push_back(runtime.prepare(request));
        if (runtime.prepare(request) != plans.back())
          throw std::runtime_error("live plan cache missed");
        auto const& p = runtime.selection(plans.back());
        bool hierarchy =
            p.candidateEligible &&
            qualifiedLocalBackend(p.baselineSamples, p.candidateSamples,
                                  p.candidateCorrect) == LocalBackend::Host;
        bool direct =
            p.direct.eligible &&
            qualifiedLocalBackend(p.direct.baselineSamples, p.direct.samples,
                                  p.direct.correct) == LocalBackend::Host;
        auto expected = chooseNetworkCandidate(
            hierarchy,
            p.fromCache && !p.validationCandidateSamples.empty()
                ? p.validationCandidateSamples
                : p.candidateSamples,
            direct,
            p.direct.fromCache && !p.direct.validationSamples.empty()
                ? p.direct.validationSamples
                : p.direct.samples);
        if (!p.poolCandidateSamples.empty() &&
            qualifiedLocalBackend(p.poolBaselineSamples, p.poolCandidateSamples,
                                  true) != LocalBackend::Host)
          expected = NetworkBackend::Caller;
        if (p.backend != expected)
          throw std::runtime_error(
              "portfolio differs from paired/current evidence");
        if (p.direct.eligible &&
            (!p.direct.correct || p.direct.samples.size() != 18 ||
             p.direct.baselineSamples.size() != 18 ||
             p.direct.configurationIdentities.size() != size_t(ranks)))
          throw std::runtime_error("direct qualification evidence incomplete");
        if (mode == "noprobes" &&
            (!p.direct.modelIdentity.empty() || p.direct.model.estimated))
          throw std::runtime_error(
              "disabled direct model probing still produced a calibration");
        if (mode == "fallback" && (p.direct.eligible || p.candidateEligible ||
                                   runtime.networkInitialized()))
          throw std::runtime_error("fallback initialized candidates");
        if (mode != "fallback" && !p.direct.eligible)
          throw std::runtime_error("direct capability unexpectedly absent");
      }
      if (mode == "current") {
        auto const& selected = runtime.selection(plans[0]);
        if (!selected.fromCache || !selected.direct.fromCache ||
            selected.direct.samples.front() != 0.001 ||
            selected.backend != NetworkBackend::Hierarchy)
          throw std::runtime_error(
              "historical direct timing overrode current candidate "
              "measurements");
      }
      auto fallback = runtime.selection(plans[1]).request;
      fallback.fallbackOnly = true;
      plans.push_back(runtime.prepare(fallback));
      bool needResources = false, bothH = false, bothD = false;
      for (auto plan : plans) {
        auto const& p = runtime.selection(plan);
        needResources |= p.backend != NetworkBackend::Caller;
        bothH |= p.backend == NetworkBackend::Hierarchy;
        bothD |= p.backend == NetworkBackend::DirectExchange;
        if ((mode == "forced" || mode == "direct") && !p.request.fallbackOnly &&
            p.backend == NetworkBackend::Caller)
          throw std::runtime_error("delay fixture did not admit candidate");
        if (!rank)
          std::cout << "PLAN lifetime=" << lifetime
                    << " count=" << p.request.count
                    << " op=" << int(p.request.operation)
                    << " selected=" << int(p.backend)
                    << " hierarchy=" << p.candidateEligible
                    << " direct=" << p.direct.eligible
                    << " direct_samples=" << p.direct.samples.size()
                    << " direct_cache=" << p.direct.fromCache
                    << " direct_validation="
                    << p.direct.validationSamples.size()
                    << " hierarchy_cache=" << p.fromCache
                    << " pool_samples=" << p.poolCandidateSamples.size()
                    << '\n';
        PlanView view{runtime, plan};
        exercise<Bits>(view, rank, ranks, int(p.request.operation),
                       p.request.count, bool(rank & 1), streams);
      }
      if (runtime.networkInitialized() != needResources)
        throw std::runtime_error(
            "portfolio resources differ from retained plans");
      if (bothH && bothD)
        for (auto plan : plans) {
          auto const& p = runtime.selection(plan);
          if (p.backend != NetworkBackend::Caller &&
              (p.poolCandidateSamples.size() != 18 ||
               p.poolBaselineSamples.size() != 18))
            throw std::runtime_error("simultaneous pool not qualified");
        }
      if (lifetime && mode == "forced" && ranks == 8 &&
          dtype == ScalarType::Float32 && !cache.empty() &&
          !runtime.selection(plans[0]).fromCache)
        throw std::runtime_error(
            "hierarchy cache was not reused alongside fresh direct trials");
    }
    nativeCheck(ncclCommDestroy(comm));
  }
  for (auto stream : streams) check(cudaStreamDestroy(stream));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks, localRank, localRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local);
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localRanks);
  try {
    if (argc != 5)
      throw std::invalid_argument(
          "NATIVE_LIBRARY fp32|bf16 real|forced|direct|fallback CACHE_OR_DASH");
    int visible;
    check(cudaGetDeviceCount(&visible));
    int device = visible == 1 ? 0 : localRanks - 1 - localRank;
    check(cudaSetDevice(device));
    std::string cache = std::string(argv[4]) == "-" ? "" : argv[4];
    if (std::string(argv[2]) == "fp32")
      run<uint32_t>(rank, ranks, localRanks, device, argv[1], argv[3], cache);
    else if (std::string(argv[2]) == "bf16")
      run<uint16_t>(rank, ranks, localRanks, device, argv[1], argv[3], cache);
    else
      throw std::invalid_argument("bad dtype");
    if (!rank)
      std::cout << "PASS network portfolio ranks=" << ranks
                << " dtype=" << argv[2] << " mode=" << argv[3]
                << "; full buffers/guards, mixed aliases, cached hierarchy, "
                   "direct/native and pool evidence, two lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

#include "native_fallback.hpp"
#include "scalar_oracle.cuh"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/runtime/prepared_network.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <mpi.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
using scalar_test::check;
void nativeCheck(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
template <class Action>
void rejects(Action action) {
  bool failed = false;
  try {
    action();
  } catch (std::exception const&) {
    failed = true;
  }
  if (!failed) throw std::runtime_error("expected distributed rejection");
}
class Delayed : public CollectiveBackend {
  std::shared_ptr<CollectiveBackend> native_;

 public:
  explicit Delayed(std::shared_ptr<CollectiveBackend> n)
      : native_(std::move(n)) {}
  void prepare() override { native_->prepare(); }
  void enqueue(NetworkOperation op, void const* a, void* b, size_t count,
               cudaStream_t stream) override {
    native_->enqueue(op, a, b, count, stream);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  void synchronize() override { native_->synchronize(); }
  bool initialized() const override { return native_->initialized(); }
  bool idle() const override { return native_->idle(); }
};
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 50000000) {
  }
}
void transitions(PreparedNetworkRuntime& runtime, int rank, int ranks,
                 std::array<cudaStream_t, 2> streams) {
  constexpr size_t count = 64, guard = 64, extent = count + 2 * guard;
  CollectiveRequest request{NetworkOperation::AllReduce, count,
                            std::vector<RankBufferLayout>(ranks, {false, 256})};
  size_t network = runtime.prepare(request);
  request.fallbackOnly = true;
  size_t native = runtime.prepare(request);
  if (runtime.selection(network).backend != NetworkBackend::Hierarchy ||
      runtime.selection(native).backend != NetworkBackend::Caller)
    throw std::runtime_error(
        "test delay did not yield the intended mixed backends");
  float *a, *b, *c;
  check(cudaMalloc(&a, extent * 4));
  check(cudaMalloc(&b, extent * 4));
  check(cudaMalloc(&c, extent * 4));
  for (bool reverse : {false, true}) {
    std::vector<float> ha(extent, -99), hb(extent, -99), hc(extent, -99),
        actual(extent);
    for (size_t i = 0; i < count; ++i) ha[guard + i] = rank + 1;
    check(cudaMemcpy(a, ha.data(), extent * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, hb.data(), extent * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(c, hc.data(), extent * 4, cudaMemcpyHostToDevice));
    if (rank == ranks - 1) skew<<<1, 32, 0, streams[0]>>>();
    check(cudaGetLastError());
    runtime.enqueue(reverse ? native : network, a + guard, b + guard,
                    streams[0]);
    runtime.enqueue(reverse ? network : native, b + guard, c + guard,
                    streams[1]);
    runtime.synchronize();
    for (size_t i = 0; i < count; ++i) {
      hb[guard + i] = ranks * (ranks + 1) / 2;
      hc[guard + i] = ranks * hb[guard + i];
    }
    for (auto pair : {std::pair<float*, std::vector<float> const*>{a, &ha},
                      {b, &hb},
                      {c, &hc}}) {
      check(cudaMemcpy(actual.data(), pair.first, extent * 4,
                       cudaMemcpyDeviceToHost));
      if (actual != *pair.second)
        throw std::runtime_error(
            "cross-backend producer/consumer/guard failure");
    }
  }
  check(cudaFree(a));
  check(cudaFree(b));
  check(cudaFree(c));
}
template <class Bits>
void run(int rank, int ranks, int localRank, int localRanks, std::string mode,
         char const* library, ScalarType dtype) {
  int device = localRanks - 1 - localRank;
  check(cudaSetDevice(device));
  ncclUniqueId nativeId;
  if (!rank) nativeCheck(ncclGetUniqueId(&nativeId));
  MPI_Bcast(&nativeId, sizeof(nativeId), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t native;
  nativeCheck(ncclCommInitRank(&native, ranks, nativeId, rank));
  std::array<cudaStream_t, 2> streams;
  for (auto& stream : streams)
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    BackendContext c;
    c.rank = rank;
    c.device = device;
    c.ranks = ranks;
    c.capacity = 65536UL * ranks * sizeof(Bits);
    c.localDevices.resize(localRanks);
    std::iota(c.localDevices.begin(), c.localDevices.end(), 0);
    if (!rank) c.id = PreparedNetworkRuntime::createUniqueId();
    MPI_Bcast(c.id.data(), c.id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    NetworkPreparationOptions options;
    options.allowDirect =
        false;  // This suite isolates the retained hierarchy contract.
    options.dtype = dtype;
    options.fallbackOnly = mode == "fallback";
    options.fallback = integration::makeNativeNcclFallback(
        c, native, library,
        mode == "instrumented" ? "distributed-memcheck-correctness-only"
                               : "distributed-check",
        dtype);
    if (mode == "transitions") {
      options.fallback.backend =
          std::make_shared<Delayed>(options.fallback.backend);
      options.fallback.identity += ";test-only-3ms-delay";
    }
    PreparedNetworkRuntime runtime(c, options);
    CollectiveRequest bad{NetworkOperation::AllReduce,
                          size_t(64 + (rank == ranks - 1)),
                          std::vector<RankBufferLayout>(ranks, {false, 256})};
    rejects([&] { runtime.prepare(bad); });
    if (runtime.networkInitialized() || options.fallback.backend->initialized())
      throw std::runtime_error("metadata mismatch initialized a backend");
    if (ranks == 8 && dtype == ScalarType::Float32 && mode != "fallback") {
      auto file = std::getenv("UCCL_LITE_NETWORK_PROFILE");
      std::string saved = file ? file : "";
      bool present = file != nullptr;
      if (rank == ranks - 1)
        setenv("UCCL_LITE_NETWORK_PROFILE", "/nonexistent/uccl_lite_profile",
               1);
      auto request = bad;
      request.count = 64;
      rejects([&] { runtime.prepare(request); });
      if (rank == ranks - 1) {
        if (present)
          setenv("UCCL_LITE_NETWORK_PROFILE", saved.c_str(), 1);
        else
          unsetenv("UCCL_LITE_NETWORK_PROFILE");
      }
      if (runtime.networkInitialized() ||
          options.fallback.backend->initialized())
        throw std::runtime_error(
            "invalid network configuration initialized a backend");
    }
    std::vector<size_t> plans;
    bool needNetwork = false;
    for (size_t count : {64UL, 257UL, 4096UL, 65536UL})
      for (int layout = 0; layout < 2; ++layout)
        for (int op = 1; op <= 3; ++op) {
          CollectiveRequest request{NetworkOperation(op), count, {}};
          for (int r = 0; r < ranks; ++r)
            request.layouts.push_back(
                {bool(layout && (r & 1)), unsigned(sizeof(Bits))});
          size_t plan = runtime.prepare(request);
          auto selection = runtime.selection(plan);
          if (runtime.prepare(request) != plan ||
              runtime.selection(plan).candidateSamples !=
                  selection.candidateSamples)
            throw std::runtime_error("distributed live cache was not reused");
          auto expected =
              selection.candidateEligible &&
                      qualifiedLocalBackend(
                          selection.baselineSamples, selection.candidateSamples,
                          selection.candidateCorrect) == LocalBackend::Host
                  ? NetworkBackend::Hierarchy
                  : NetworkBackend::Caller;
          if (selection.backend != expected)
            throw std::runtime_error(
                "distributed choice differs from paired evidence");
          if (selection.candidateEligible &&
              (!selection.candidateCorrect ||
               selection.baselineSamples.size() != 18 ||
               selection.candidateSamples.size() != 18))
            throw std::runtime_error(
                "distributed qualification incomplete or incorrect");
          if ((mode == "fallback" || mode == "unavailable" || ranks != 8 ||
               dtype != ScalarType::Float32 || count == 257) &&
              selection.candidateEligible)
            throw std::runtime_error(
                "unsupported candidate entered qualification");
          if (mode != "fallback" && mode != "unavailable" && ranks == 8 &&
              dtype == ScalarType::Float32 && count != 257 &&
              !selection.candidateEligible)
            throw std::runtime_error(
                "supported network candidate unexpectedly declined preflight");
          if (mode == "unavailable" && count != 257 &&
              selection.reason.find("preflight unavailable") ==
                  std::string::npos)
            throw std::runtime_error(
                "unavailable test did not reach network preflight");
          needNetwork |= selection.backend == NetworkBackend::Hierarchy;
          if (runtime.networkInitialized() != needNetwork)
            throw std::runtime_error("distributed resource retention mismatch");
          if (!rank) {
            std::cout << "op=" << op << " count=" << count
                      << " layout=" << layout
                      << " selected=" << int(selection.backend)
                      << " eligible=" << selection.candidateEligible
                      << " candidate_us=";
            for (double t : selection.candidateSamples) std::cout << t << ',';
            std::cout << " baseline_us=";
            for (double t : selection.baselineSamples) std::cout << t << ',';
            std::cout << " reason=" << selection.reason << std::endl;
          }
          auto l = request.layouts[rank];
          scalar_test::exercise<Bits>(
              op, count, l.inPlace, l.alignment, rank, ranks, streams,
              [&](int, void const* input, void* output, size_t,
                  cudaStream_t stream) {
                runtime.enqueue(plan, input, output, stream);
              });
          runtime.synchronize();
          plans.push_back(plan);
        }
    for (size_t plan : plans) {
      auto const& request = runtime.selection(plan).request;
      auto l = request.layouts[rank];
      scalar_test::exercise<Bits>(int(request.operation), request.count,
                                  l.inPlace, l.alignment, rank, ranks, streams,
                                  [&](int, void const* input, void* output,
                                      size_t, cudaStream_t stream) {
                                    runtime.enqueue(plan, input, output,
                                                    stream);
                                  });
      runtime.synchronize();
    }
    auto first = runtime.selection(plans.front()).request;
    char const* old = std::getenv("NCCL_BUFFSIZE");
    std::string saved = old ? old : "";
    bool had = old != nullptr;
    if (rank == ranks - 1) setenv("NCCL_BUFFSIZE", "2097153", 1);
    rejects([&] { runtime.prepare(first); });
    rejects([&] {
      runtime.enqueue(plans.front(), reinterpret_cast<void*>(256),
                      reinterpret_cast<void*>(512), streams[0]);
    });
    if (rank == ranks - 1) {
      if (had)
        setenv("NCCL_BUFFSIZE", saved.c_str(), 1);
      else
        unsetenv("NCCL_BUFFSIZE");
    }
    if (runtime.prepare(first) != plans.front())
      throw std::runtime_error("restored configuration lost cached plan");
    cudaGraph_t graph;
    check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
    rejects([&] {
      runtime.enqueue(plans.front(), reinterpret_cast<void*>(sizeof(Bits)),
                      reinterpret_cast<void*>(4096), streams[0]);
    });
    check(cudaStreamEndCapture(streams[0], &graph));
    check(cudaGraphDestroy(graph));
    if (mode == "transitions") transitions(runtime, rank, ranks, streams);
    runtime.synchronize();
  }
  scalar_test::exercise<Bits>(
      3, 64, false, unsigned(sizeof(Bits)), rank, ranks, streams,
      [&](int, void const* input, void* output, size_t count,
          cudaStream_t stream) {
        nativeCheck(ncclAllReduce(
            input, output, count,
            dtype == ScalarType::Float32 ? ncclFloat32 : ncclFloat16, ncclSum,
            native, stream));
      });
  for (auto stream : streams) check(cudaStreamDestroy(stream));
  nativeCheck(ncclCommDestroy(native));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local);
  int localRank, localRanks;
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localRanks);
  try {
    if (argc != 4)
      throw std::invalid_argument(
          "NATIVE_LIBRARY "
          "automatic|fallback|unavailable|transitions|instrumented fp32|fp16");
    std::string mode = argv[2], type = argv[3];
    if (mode != "automatic" && mode != "fallback" && mode != "unavailable" &&
        mode != "transitions" && mode != "instrumented")
      throw std::invalid_argument("invalid mode");
    if (type == "fp32")
      run<uint32_t>(rank, ranks, localRank, localRanks, mode, argv[1],
                    ScalarType::Float32);
    else if (type == "fp16")
      run<uint16_t>(rank, ranks, localRank, localRanks, mode, argv[1],
                    ScalarType::Float16);
    else
      throw std::invalid_argument("invalid dtype");
    if (!rank)
      std::cout
          << "PASS: prepared distributed " << mode << ' ' << type
          << " ranks=" << ranks
          << "; paired admission, full guards, mixed layouts, cache/recovery, "
             "stream ordering and caller ownership, two lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

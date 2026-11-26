#include "native_fallback.hpp"
#include "scalar_oracle.cuh"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/runtime/prepared.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <mpi.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
using scalar_test::check;
void nativeCheck(ncclResult_t r) {
  if (r != ncclSuccess) throw std::runtime_error(ncclGetErrorString(r));
}
template <class F>
void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (std::exception const&) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("expected runtime rejection");
}
class Delayed : public CollectiveBackend {
  std::shared_ptr<CollectiveBackend> native_;

 public:
  explicit Delayed(std::shared_ptr<CollectiveBackend> native)
      : native_(std::move(native)) {}
  void prepare() override { native_->prepare(); }
  void enqueue(NetworkOperation op, void const* in, void* out, size_t count,
               cudaStream_t stream) override {
    native_->enqueue(op, in, out, count, stream);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  void synchronize() override { native_->synchronize(); }
  bool initialized() const override { return native_->initialized(); }
  bool idle() const override { return native_->idle(); }
  bool supportsGraphs() const override { return native_->supportsGraphs(); }
  size_t prepareGraph(NetworkOperation op, void const* in, void* out,
                      size_t count) override {
    return native_->prepareGraph(op, in, out, count);
  }
  void launchGraph(size_t graph, cudaStream_t stream) override {
    native_->launchGraph(graph, stream);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  void releaseGraph(size_t graph) override { native_->releaseGraph(graph); }
};
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 50000000) {
  }
}
void graphChecks(PreparedRuntime& runtime, int rank, int ranks,
                 std::array<cudaStream_t, 2> streams, bool forced) {
  constexpr size_t count = 64, guard = 64, extent = count + 2 * guard;
  float *a, *b, *c;
  check(cudaMalloc(&a, extent * 4));
  check(cudaMalloc(&b, extent * 4));
  check(cudaMalloc(&c, extent * 4));
  CollectiveRequest request{NetworkOperation::AllReduce, count,
                            std::vector<RankBufferLayout>(ranks, {false, 256})};
  auto ab = runtime.prepareGraph(request, a + guard, b + guard);
  auto bc = runtime.prepareGraph(request, b + guard, c + guard);
  auto eager = runtime.prepare(request);
  auto fallback = request;
  fallback.fallbackOnly = true;
  auto native = runtime.prepare(fallback);
  if (!runtime.graphSelection(ab).graph || runtime.selection(eager).graph)
    throw std::runtime_error("execution modes share admission");
  if (runtime.placement().domain != RuntimeDomain::Local &&
      runtime.graphSelection(ab).backend != RuntimeBackend::Caller)
    throw std::runtime_error(
        "distributed eager evidence admitted a graph candidate");
  if (forced && runtime.selection(eager).backend == RuntimeBackend::Caller)
    throw std::runtime_error("test delay did not qualify an eager alternative");
  if (ranks > 1) {
    rejects([&] {
      if (rank == ranks - 1)
        runtime.prepare(request);
      else
        runtime.prepareGraph(request, a + guard, b + guard);
    });
    rejects([&] { runtime.launchGraph(ab, streams[0]); });
  }
  rejects([&] {
    runtime.prepareGraph(request, rank == ranks - 1 ? nullptr : a + guard,
                         b + guard);
  });
  runtime.prepare(request);
  cudaGraph_t external;
  check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
  rejects([&] { runtime.launchGraph(ab, streams[0]); });
  check(cudaStreamEndCapture(streams[0], &external));
  check(cudaGraphDestroy(external));
  for (int mode = 0; mode < 4; ++mode) {
    std::vector<float> ha(extent, -99), hb(extent, -99), hc(extent, -99),
        actual(extent);
    for (size_t i = 0; i < count; ++i) ha[guard + i] = rank + 1 + mode;
    check(cudaMemcpy(a, ha.data(), extent * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, hb.data(), extent * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(c, hc.data(), extent * 4, cudaMemcpyHostToDevice));
    if (rank == ranks - 1) skew<<<1, 32, 0, streams[0]>>>();
    check(cudaGetLastError());
    if (mode == 0 || mode == 2)
      runtime.launchGraph(ab, streams[0]);
    else
      runtime.enqueue(mode == 1 ? native : eager, a + guard, b + guard,
                      streams[0]);
    if (mode == 1 || mode == 3)
      runtime.launchGraph(bc, streams[1]);
    else
      runtime.enqueue(mode == 0 ? native : eager, b + guard, c + guard,
                      streams[1]);
    runtime.synchronize();
    for (size_t i = 0; i < count; ++i) {
      hb[guard + i] = ranks * (ranks + 1) / 2 + ranks * mode;
      hc[guard + i] = ranks * hb[guard + i];
    }
    for (auto pair : {std::pair<float*, std::vector<float> const*>{a, &ha},
                      {b, &hb},
                      {c, &hc}}) {
      check(cudaMemcpy(actual.data(), pair.first, extent * 4,
                       cudaMemcpyDeviceToHost));
      if (actual != *pair.second)
        throw std::runtime_error(
            "unified graph/eager dependency or guard failure");
    }
  }
  if (ranks > 1)
    rejects([&] { runtime.releaseGraph(rank == ranks - 1 ? bc : ab); });
  runtime.releaseGraph(ab);
  rejects([&] { runtime.launchGraph(ab, streams[0]); });
  runtime.releaseGraph(bc);
  check(cudaFree(a));
  check(cudaFree(b));
  check(cudaFree(c));
}
template <class Bits>
void run(int rank, int ranks, int localRank, int localRanks, int hosts,
         char const* library, std::string mode, ScalarType dtype) {
  int visible;
  check(cudaGetDeviceCount(&visible));
  int device = visible == 1 ? 0 : localRanks - 1 - localRank;
  check(cudaSetDevice(device));
  ncclUniqueId id;
  if (!rank) nativeCheck(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t native;
  nativeCheck(ncclCommInitRank(&native, ranks, id, rank));
  std::array<cudaStream_t, 2> streams;
  for (auto& stream : streams)
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    RuntimeContext context;
    context.rank = rank;
    context.device = device;
    context.ranks = ranks;
    context.capacity = 4096UL * ranks * sizeof(Bits);
    if (!rank) context.id = PreparedRuntime::createUniqueId();
    MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    PreparationOptions options;
    options.dtype = dtype;
    options.probeResources = false;
    options.fallbackOnly = mode == "fallback";
    options.fallback = integration::makeNativeNcclFallback(
        context, native, library,
        mode == "instrumented" ? "unified-memcheck-correctness-only"
                               : "unified-check",
        dtype);
    if (mode == "transitions" || mode == "tree") {
      options.fallback.backend =
          std::make_shared<Delayed>(options.fallback.backend);
      options.fallback.identity += ";test-only-3ms-delay";
    }
    if (mode == "tree") {
      options.allowPeerDirect = false;
      options.allowPeerPacket = false;
      options.allowHost = false;
      options.allowNetwork = false;
    }
    PreparedRuntime runtime(context, options);
    CollectiveRequest invalid{
        NetworkOperation::AllReduce, 64,
        std::vector<RankBufferLayout>(ranks, {false, unsigned(sizeof(Bits))})};
    if (ranks > 1) {
      auto wrong = invalid;
      wrong.count += rank == ranks - 1;
      rejects([&] { runtime.prepare(wrong); });
    }
    invalid.layouts.back().alignment = 3;
    rejects([&] { runtime.prepare(invalid); });
    if (runtime.acceleratedResourcesInitialized() ||
        options.fallback.backend->initialized())
      throw std::runtime_error("invalid request initialized resources");
    std::vector<size_t> plans;
    for (size_t count : {64UL, 257UL, 4096UL})
      for (int layout = 0; layout < 2; ++layout)
        for (int op = 1; op <= 3; ++op) {
          CollectiveRequest r{NetworkOperation(op), count, {}};
          for (int i = 0; i < ranks; ++i)
            r.layouts.push_back(
                {bool(layout && (i & 1)), unsigned(sizeof(Bits))});
          auto plan = runtime.prepare(r);
          if (runtime.prepare(r) != plan)
            throw std::runtime_error("unified cache failed");
          auto expected = hosts > 1          ? RuntimeDomain::Distributed
                          : mode == "hidden" ? RuntimeDomain::CallerOnly
                                             : RuntimeDomain::Local;
          if (runtime.placement().domain != expected ||
              runtime.placement().ranks.size() != size_t(ranks))
            throw std::runtime_error(
                "host/view discovery chose wrong selector");
          auto const& choice = runtime.selection(plan);
          if (mode == "tree" && choice.backend != RuntimeBackend::PeerTree)
            throw std::runtime_error(
                "test delay did not qualify the sole tree alternative");
          if (choice.graph)
            throw std::runtime_error("eager request reused graph evidence");
          if (choice.backend != RuntimeBackend::Caller) {
            bool justified = false;
            for (auto const& c : choice.candidates)
              if (c.backend == choice.backend)
                justified = c.eligible && c.correct &&
                            qualifiedLocalBackend(c.baselineSamples, c.samples,
                                                  true) == LocalBackend::Host;
            if (!justified)
              throw std::runtime_error("unified choice lacks paired evidence");
          }
          if ((mode == "fallback" || mode == "hidden" ||
               dtype == ScalarType::Float16) &&
              runtime.acceleratedResourcesInitialized())
            throw std::runtime_error(
                "caller-only path initialized acceleration");
          if (!rank)
            std::cout << "domain=" << int(runtime.placement().domain)
                      << " op=" << op << " count=" << count
                      << " layout=" << layout
                      << " backend=" << int(choice.backend) << '\n';
          auto l = r.layouts[rank];
          scalar_test::exercise<Bits>(
              op, count, l.inPlace, l.alignment, rank, ranks, streams,
              [&](int, void const* in, void* out, size_t, cudaStream_t stream) {
                runtime.enqueue(plan, in, out, stream);
              });
          runtime.synchronize();
          plans.push_back(plan);
        }
    for (auto plan : plans) {
      auto const& r = runtime.selection(plan).request;
      auto l = r.layouts[rank];
      scalar_test::exercise<Bits>(
          int(r.operation), r.count, l.inPlace, l.alignment, rank, ranks,
          streams,
          [&](int, void const* in, void* out, size_t, cudaStream_t stream) {
            runtime.enqueue(plan, in, out, stream);
          });
      runtime.synchronize();
    }
    if (dtype == ScalarType::Float32)
      graphChecks(runtime, rank, ranks, streams,
                  mode == "transitions" || mode == "tree");
    runtime.synchronize();
  }
  scalar_test::exercise<Bits>(
      3, 64, false, unsigned(sizeof(Bits)), rank, ranks, streams,
      [&](int, void const* in, void* out, size_t count, cudaStream_t stream) {
        nativeCheck(ncclAllReduce(
            in, out, count,
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
  int leader = localRank == 0, hosts;
  MPI_Allreduce(&leader, &hosts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  try {
    if (argc != 4)
      throw std::invalid_argument(
          "NATIVE_LIBRARY "
          "automatic|fallback|hidden|transitions|instrumented|tree "
          "fp32|fp16");
    std::string mode = argv[2], type = argv[3];
    if (mode != "automatic" && mode != "fallback" && mode != "hidden" &&
        mode != "transitions" && mode != "instrumented" && mode != "tree")
      throw std::invalid_argument("invalid mode");
    if (mode == "tree" && (type != "fp32" || hosts != 1 || ranks < 2))
      throw std::invalid_argument("tree fixture requires local FP32 ranks");
    if (type == "fp32")
      run<uint32_t>(rank, ranks, localRank, localRanks, hosts, argv[1], mode,
                    ScalarType::Float32);
    else if (type == "fp16")
      run<uint16_t>(rank, ranks, localRank, localRanks, hosts, argv[1], mode,
                    ScalarType::Float16);
    else
      throw std::invalid_argument("invalid dtype");
    if (!rank)
      std::cout
          << "PASS: unified prepared runtime " << mode << ' ' << type
          << " hosts=" << hosts << " ranks=" << ranks
          << "; discovery, paired selection, guards, aliases, graph/eager "
             "chains, rejection recovery and caller ownership, two lifetimes\n";
  } catch (std::exception const& e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

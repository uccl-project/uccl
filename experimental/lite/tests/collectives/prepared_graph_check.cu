#include "native_fallback.hpp"
#include "scalar_oracle.cuh"
#include "uccl_lite/runtime/prepared_local.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <cuda_bf16.h>
#include <mpi.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
using scalar_test::check;
void nativeCheck(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
// Test-only delayed graph baseline forces a peer winner for ordering coverage.
// These samples must never be reported as gains against unmodified NCCL.
class DelayedGraph : public CollectiveBackend {
  std::shared_ptr<CollectiveBackend> native_;

 public:
  explicit DelayedGraph(std::shared_ptr<CollectiveBackend> native)
      : native_(std::move(native)) {}
  void prepare() override { native_->prepare(); }
  void enqueue(NetworkOperation op, void const* in, void* out, size_t count,
               cudaStream_t stream) override {
    native_->enqueue(op, in, out, count, stream);
  }
  void synchronize() override { native_->synchronize(); }
  bool initialized() const override { return native_->initialized(); }
  bool idle() const override { return native_->idle(); }
  bool supportsGraphs() const override { return true; }
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
template <class Bits>
Bits encode(float value) {
  if constexpr (sizeof(Bits) == 4)
    return scalar_test::encode<Bits>(value);
  else
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}
template <class Bits>
struct Buffers {
  LocalRequest request;
  int rank, ranks, op;
  size_t inputs, outputs, extent, inOffset, outOffset, graph, eager;
  bool inputB, outputA;
  Bits *a = nullptr, *b = nullptr;
  std::vector<Bits> expectedA, expectedB;
  Buffers(LocalRequest r, int rank, int ranks)
      : request(std::move(r)),
        rank(rank),
        ranks(ranks),
        op(int(request.operation)) {
    auto l = request.layouts[rank];
    inputs = request.count * (op == 2 ? ranks : 1);
    outputs = request.count * (op == 1 ? ranks : 1);
    extent = std::max(inputs, outputs) + 512 / sizeof(Bits);
    inOffset = l.alignment / sizeof(Bits) +
               (l.inPlace && op == 1 ? rank * request.count : 0);
    outOffset = l.alignment / sizeof(Bits) +
                (l.inPlace && op == 2 ? rank * request.count : 0);
    inputB = l.inPlace && op == 1;
    outputA = l.inPlace && op != 1;
    check(cudaMalloc(&a, extent * sizeof(Bits)));
    check(cudaMalloc(&b, extent * sizeof(Bits)));
  }
  ~Buffers() {
    cudaFree(a);
    cudaFree(b);
  }
  Bits const* input() const { return (inputB ? b : a) + inOffset; }
  Bits* output() const { return (outputA ? a : b) + outOffset; }
  Bits value(size_t index, int source, int generation) {
    if (op == 1)
      return scalar_test::input<Bits>(index, source, generation, true);
    return encode<Bits>(float(source + 1 + int(index % 7) - 3 + generation));
  }
  void initialize(int generation) {
    Bits guard = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
    expectedA.assign(extent, guard);
    expectedB.assign(extent, guard);
    auto& source = inputB ? expectedB : expectedA;
    for (size_t i = 0; i < inputs; ++i)
      source[inOffset + i] = value(i, rank, generation);
    check(cudaMemcpy(a, expectedA.data(), extent * sizeof(Bits),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, expectedB.data(), extent * sizeof(Bits),
                     cudaMemcpyHostToDevice));
    auto& target = outputA ? expectedA : expectedB;
    for (size_t i = 0; i < outputs; ++i) {
      if (op == 1)
        target[outOffset + i] =
            value(i % request.count, i / request.count, generation);
      else {
        size_t index = op == 2 ? rank * request.count + i : i;
        // Independent closed-form sum of integer-valued per-rank inputs.
        double sum = double(ranks * (ranks + 1)) / 2 +
                     ranks * double(int(index % 7) - 3 + generation);
        target[outOffset + i] = encode<Bits>(float(sum));
      }
    }
  }
  void verify() {
    std::vector<Bits> actual(extent);
    check(cudaMemcpy(actual.data(), a, extent * sizeof(Bits),
                     cudaMemcpyDeviceToHost));
    if (actual != expectedA)
      throw std::runtime_error("prepared graph input/alias/guard failure");
    check(cudaMemcpy(actual.data(), b, extent * sizeof(Bits),
                     cudaMemcpyDeviceToHost));
    if (actual != expectedB)
      throw std::runtime_error("prepared graph output/guard failure");
  }
};
template <class Action>
void rejected(Action action) {
  bool failed = false;
  try {
    action();
  } catch (std::exception const&) {
    failed = true;
  }
  if (!failed) throw std::runtime_error("expected collective rejection");
}
template <class Bits>
void run(int rank, int ranks, std::string mode, char const* nativeLibrary) {
  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    BackendContext context;
    context.rank = context.device = rank;
    context.ranks = ranks;
    context.capacity = 4096UL * ranks * sizeof(Bits);
    context.localDevices.resize(ranks);
    std::iota(context.localDevices.begin(), context.localDevices.end(), 0);
    if (!rank) {
      std::ifstream random("/dev/urandom", std::ios::binary);
      if (!random.read(reinterpret_cast<char*>(context.id.data()),
                       context.id.size()))
        throw std::runtime_error("random session ID failed");
    }
    MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    ncclComm_t native = nullptr;
    if (mode != "peer") {
      ncclUniqueId id;
      if (!rank) nativeCheck(ncclGetUniqueId(&id));
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
      nativeCheck(ncclCommInitRank(&native, ranks, id, rank));
    }
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<std::unique_ptr<Buffers<Bits>>> buffers;
    {
      LocalPreparationOptions options;
      options.dtype =
          sizeof(Bits) == 4 ? ScalarType::Float32 : ScalarType::BFloat16;
      options.probeResources = false;
      options.fallbackOnly = mode == "fallback";
      if (native)
        options.fallback = integration::makeNativeNcclFallback(
            context, native, nativeLibrary, "prepared-owned-graph-check",
            options.dtype);
      if (mode == "transitions") {
        options.fallback.backend =
            std::make_shared<DelayedGraph>(options.fallback.backend);
        options.fallback.identity += ";test-only-3ms-graph-delay";
      }
      PreparedLocalRuntime runtime(context, options);
      LocalRequest invalid{NetworkOperation::AllReduce, 64,
                           std::vector<RankBufferLayout>(
                               ranks, {false, unsigned(sizeof(Bits))})};
      rejected([&] { runtime.prepareGraph(invalid, nullptr, nullptr); });
      if (runtime.peerInitialized() || runtime.hostInitialized() ||
          (native && options.fallback.backend->initialized()))
        throw std::runtime_error(
            "invalid graph bindings initialized resources");
      for (size_t count : {64UL, 257UL, 4096UL})
        for (int inplace = 0; inplace < 2; ++inplace)
          for (int op = 1; op <= 3; ++op) {
            LocalRequest request{NetworkOperation(op), count, {}};
            for (int r = 0; r < ranks; ++r)
              request.layouts.push_back(
                  {bool(inplace && (r & 1)),
                   unsigned(count == 257 ? sizeof(Bits)
                                         : sizeof(Bits) * (1u << (r % 3)))});
            auto b = std::make_unique<Buffers<Bits>>(request, rank, ranks);
            b->graph = runtime.prepareGraph(request, b->input(), b->output());
            auto selection = runtime.graphSelection(b->graph);
            if (!selection.graph || selection.hostEligible ||
                selection.peerPacketEligible)
              throw std::runtime_error(
                  "graph execution capability admission failed");
            if (mode == "fallback" &&
                (runtime.hostInitialized() || runtime.peerInitialized() ||
                 runtime.peerPacketInitialized() ||
                 runtime.peerTreeInitialized()))
              throw std::runtime_error(
                  "fallback-only graph allocated alternative resources");
            if (native && mode != "fallback") {
              std::array<std::vector<double>, 4> samples{
                  {selection.samples[0], selection.samples[1],
                   selection.samples[3], selection.samples[4]}};
              if (selection.backend != qualifiedFallbackAlternative(
                                           selection.fallbackSamples, samples,
                                           selection.candidateCorrect))
                throw std::runtime_error(
                    "graph selection does not match graph evidence");
            }
            if (!rank) {
              std::cout << "graph op=" << op << " count=" << count
                        << " layout=" << inplace
                        << " selected=" << int(selection.backend)
                        << " candidate_us=";
              for (double t : selection.samples[0]) std::cout << t << ',';
              std::cout << " baseline_us=";
              for (double t : selection.fallbackSamples[0])
                std::cout << t << ',';
              std::cout << std::endl;
            }
            b->eager = runtime.prepare(request);
            if (runtime.selection(b->eager).graph)
              throw std::runtime_error("eager reused graph qualification");
            // Eager preparation suspends/rebuilds old graphs. Exercise both.
            for (int generation = 0; generation < 3; ++generation) {
              b->initialize(generation);
              runtime.launchGraph(b->graph, streams[generation % 2]);
              runtime.synchronize();
              b->verify();
              b->initialize(generation);
              runtime.enqueue(b->eager, b->input(), b->output(),
                              streams[1 - generation % 2]);
              runtime.synchronize();
              b->verify();
            }
            buffers.push_back(std::move(b));
          }
      auto& first = *buffers.front();
      rejected([&] {
        if (rank == ranks - 1)
          runtime.prepare(first.request);
        else
          runtime.prepareGraph(first.request, first.input(), first.output());
      });
      rejected([&] { runtime.launchGraph(first.graph, streams[0]); });
      rejected([&] {
        runtime.prepareGraph(first.request,
                             rank == ranks - 1 ? nullptr : first.input(),
                             first.output());
      });
      runtime.prepare(first.request);
      // Cache hit must restore retained graphs after failed preparation.
      auto duplicate =
          runtime.prepareGraph(first.request, first.input(), first.output());
      if (runtime.graphSelection(duplicate).samples !=
          runtime.graphSelection(first.graph).samples)
        throw std::runtime_error("graph qualification cache failed");
      runtime.releaseGraph(duplicate);
      rejected([&] { runtime.launchGraph(duplicate, streams[0]); });
      rejected([&] {
        runtime.releaseGraph(rank == ranks - 1 ? duplicate : first.graph);
      });
      cudaGraph_t external;
      check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      rejected([&] { runtime.launchGraph(first.graph, streams[0]); });
      check(cudaStreamEndCapture(streams[0], &external));
      check(cudaGraphDestroy(external));
      // Queue all retained programs across streams before one synchronization.
      for (auto& b : buffers) b->initialize(2);
      for (size_t i = 0; i < buffers.size(); ++i)
        runtime.launchGraph(buffers[i]->graph, streams[i % 2]);
      runtime.synchronize();
      for (auto& b : buffers) b->verify();
      if (mode == "transitions") {
        constexpr size_t count = 64, extent = count + 128;
        Bits *a, *b, *c;
        check(cudaMalloc(&a, extent * sizeof(Bits)));
        check(cudaMalloc(&b, extent * sizeof(Bits)));
        check(cudaMalloc(&c, extent * sizeof(Bits)));
        LocalRequest request{NetworkOperation::AllReduce, count,
                             std::vector<RankBufferLayout>(
                                 ranks, {false, unsigned(sizeof(Bits))})};
        for (bool reverse : {false, true}) {
          request.fallbackOnly = reverse;
          auto firstGraph = runtime.prepareGraph(request, a + 1, b + 1);
          request.fallbackOnly = !reverse;
          auto secondGraph = runtime.prepareGraph(request, b + 1, c + 1);
          if (runtime.graphSelection(firstGraph).backend !=
                  (reverse ? LocalBackend::Fallback : LocalBackend::Peer) ||
              runtime.graphSelection(secondGraph).backend !=
                  (reverse ? LocalBackend::Peer : LocalBackend::Fallback))
            throw std::runtime_error(
                "test delay did not produce mixed graph backends");
          Bits guard = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
          std::vector<Bits> ha(extent, guard), hb(extent, guard),
              hc(extent, guard);
          for (size_t i = 0; i < count; ++i)
            ha[1 + i] = encode<Bits>(float(rank + 1 + i % 7));
          check(cudaMemcpy(a, ha.data(), extent * sizeof(Bits),
                           cudaMemcpyHostToDevice));
          check(cudaMemcpy(b, hb.data(), extent * sizeof(Bits),
                           cudaMemcpyHostToDevice));
          check(cudaMemcpy(c, hc.data(), extent * sizeof(Bits),
                           cudaMemcpyHostToDevice));
          runtime.launchGraph(firstGraph, streams[0]);
          runtime.launchGraph(secondGraph, streams[1]);
          runtime.synchronize();
          for (size_t i = 0; i < count; ++i) {
            float sum = float(ranks * (ranks + 1) / 2 + ranks * (i % 7));
            hb[1 + i] = encode<Bits>(sum);
            hc[1 + i] = encode<Bits>(ranks * sum);
          }
          std::vector<Bits> actual(extent);
          for (auto pair : {std::pair<Bits*, std::vector<Bits> const*>{a, &ha},
                            {b, &hb},
                            {c, &hc}}) {
            check(cudaMemcpy(actual.data(), pair.first, extent * sizeof(Bits),
                             cudaMemcpyDeviceToHost));
            if (actual != *pair.second)
              throw std::runtime_error("mixed graph dependency/guard failure");
          }
          runtime.releaseGraph(firstGraph);
          runtime.releaseGraph(secondGraph);
        }
        check(cudaFree(a));
        check(cudaFree(b));
        check(cudaFree(c));
      }
      for (size_t i = 0; i < buffers.size(); i += 2)
        runtime.releaseGraph(buffers[i]->graph);
      // Remaining executables are owned by runtime teardown, before buffers.
    }
    if (native) {
      auto& b = *buffers.front();
      b.initialize(1);
      nativeCheck(ncclAllGather(b.input(), b.output(), b.request.count,
                                sizeof(Bits) == 4 ? ncclFloat32 : ncclBfloat16,
                                native, streams[0]));
      check(cudaStreamSynchronize(streams[0]));
      b.verify();
      nativeCheck(ncclCommDestroy(native));
    }
    buffers.clear();
    for (auto stream : streams) check(cudaStreamDestroy(stream));
  }
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 4)
      throw std::invalid_argument(
          "bf16|fp32 peer|native|fallback|transitions NATIVE_LIBRARY");
    std::string mode = argv[2];
    if (mode != "peer" && mode != "native" && mode != "fallback" &&
        mode != "transitions")
      throw std::invalid_argument("invalid mode");
    check(cudaSetDevice(rank));
    if (std::string(argv[1]) == "fp32")
      run<uint32_t>(rank, ranks, mode, argv[3]);
    else if (std::string(argv[1]) == "bf16")
      run<uint16_t>(rank, ranks, mode, argv[3]);
    else
      throw std::invalid_argument("invalid dtype");
    if (!rank)
      std::cout << "PASS: prepared graphs " << argv[1] << ' ' << mode
                << " ranks=" << ranks
                << "; independent graph admission/cache, full guards, mixed "
                   "layouts, eager/graph transitions, retained bindings and "
                   "rejection recovery, two lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

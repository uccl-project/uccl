#include "native_fallback.hpp"
#include "prepared_local_oracle.cuh"
#include <fstream>
#include <iostream>
#include <numeric>
#include <mpi.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
using namespace uccl_lite_test;
void nativeCheck(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
__global__ void delay() {
  auto start = clock64();
  while (clock64() - start < 2000000) {
  }
}
__global__ void fill(unsigned short* data, size_t n, int rank) {
  for (size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    data[i] =
        __bfloat16_as_ushort(__float2bfloat16_rn(float(rank + 1 + i % 7)));
}
// Test-only delayed baseline makes an actual peer/native transition
// deterministic. Its samples never claim a gain against unmodified NCCL.
class Delayed : public CollectiveBackend {
  std::shared_ptr<CollectiveBackend> native_;

 public:
  explicit Delayed(std::shared_ptr<CollectiveBackend> native)
      : native_(std::move(native)) {}
  void prepare() override { native_->prepare(); }
  void enqueue(NetworkOperation op, void const* in, void* out, size_t count,
               cudaStream_t stream) override {
    if (op == NetworkOperation::AllGather && count == 64) {
      delay<<<1, 32, 0, stream>>>();
      check(cudaGetLastError());
    }
    native_->enqueue(op, in, out, count, stream);
  }
  void synchronize() override { native_->synchronize(); }
  bool initialized() const override { return native_->initialized(); }
  bool idle() const override { return native_->idle(); }
};
void transition(PreparedLocalRuntime& runtime, int rank, int ranks,
                std::array<cudaStream_t, 2> streams) {
  LocalRequest gather{NetworkOperation::AllGather, 64,
                      std::vector<RankBufferLayout>(ranks, {false, 256})};
  auto ag = runtime.prepare(gather);
  if (runtime.selection(ag).backend != LocalBackend::Peer)
    throw std::runtime_error("test delay did not qualify peer transition");
  LocalRequest scatter{NetworkOperation::ReduceScatter, 64, gather.layouts,
                       true};
  auto rs = runtime.prepare(scatter);
  if (runtime.selection(rs).backend != LocalBackend::Fallback)
    throw std::runtime_error("explicit fallback handle not honored");
  size_t extent = 64 * ranks + 256;
  unsigned short *a, *b, *c;
  check(cudaMalloc(&a, extent * 2));
  check(cudaMalloc(&b, extent * 2));
  check(cudaMalloc(&c, extent * 2));
  for (int iteration = 0; iteration < 12; ++iteration) {
    check(cudaMemset(a, 0x5a, extent * 2));
    check(cudaMemset(b, 0x5a, extent * 2));
    check(cudaMemset(c, 0x5a, extent * 2));
    if (rank == ranks - 1) delay<<<1, 32, 0, streams[0]>>>();
    fill<<<1, 128, 0, streams[0]>>>(a + 128, 64, rank);
    check(cudaGetLastError());
    runtime.enqueue(ag, a + 128, b + 128, streams[0]);
    runtime.enqueue(rs, b + 128, c + 128, streams[1]);
    runtime.synchronize();
    std::vector<unsigned short> ha(extent), hb(extent), hc(extent);
    check(cudaMemcpy(ha.data(), a, extent * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hb.data(), b, extent * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hc.data(), c, extent * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < extent; ++i) {
      auto bits = [](float v) {
        return __bfloat16_as_ushort(__float2bfloat16_rn(v));
      };
      auto ea =
          i >= 128 && i < 192 ? bits(float(rank + 1 + (i - 128) % 7)) : 0x5a5a;
      auto eb = i >= 128 && i < 128 + 64 * ranks
                    ? bits(float((i - 128) / 64 + 1 + ((i - 128) % 64) % 7))
                    : 0x5a5a;
      auto ec = i >= 128 && i < 192
                    ? bits(float(ranks * (rank + 1 + (i - 128) % 7)))
                    : 0x5a5a;
      if (ha[i] != ea || hb[i] != eb || hc[i] != ec)
        throw std::runtime_error(
            "cross-backend producer/consumer/guard failure");
    }
  }
  check(cudaFree(a));
  check(cudaFree(b));
  check(cudaFree(c));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 4)
      throw std::invalid_argument(
          "NATIVE_LIBRARY STRICT_LIBRARY "
          "automatic|fallback|host-only|transitions");
    std::string mode = argv[3];
    if (mode != "automatic" && mode != "fallback" && mode != "host-only" &&
        mode != "transitions" && mode != "instrumented")
      throw std::invalid_argument("invalid test mode");
    check(cudaSetDevice(rank));
    ncclUniqueId id;
    if (!rank) nativeCheck(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    nativeCheck(ncclCommInitRank(&comm, ranks, id, rank));
    BackendContext context;
    context.rank = context.device = rank;
    context.ranks = ranks;
    context.capacity = 65536UL * ranks * 2;
    context.localDevices.resize(ranks);
    std::iota(context.localDevices.begin(), context.localDevices.end(), 0);
    if (!rank) {
      std::ifstream random("/dev/urandom", std::ios::binary);
      if (!random.read(reinterpret_cast<char*>(context.id.data()),
                       context.id.size()))
        throw std::runtime_error("session ID failed");
    }
    MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    bool rejected = false;
    try {
      integration::makeNativeNcclFallback(context, comm, argv[2], "default");
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("strict ABI accepted as native fallback");
    {
      auto fallback = integration::makeNativeNcclFallback(
          context, comm, argv[1],
          mode == "instrumented"
              ? "ncclCommInitRank/default;memcheck-correctness-only"
              : "ncclCommInitRank/default");
      if (fallback.backend->initialized())
        throw std::runtime_error(
            "fallback construction created execution resources");
      if (mode == "transitions") {
        fallback.backend = std::make_shared<Delayed>(fallback.backend);
        fallback.identity += ";test-only-ag64-delay";
      }
      LocalPreparationOptions options;
      options.fallback = fallback;
      options.probeResources = mode == "automatic" || mode == "fallback";
      options.fallbackOnly = mode == "fallback";
      options.allowPeer = mode != "host-only";
      options.allowHost = mode != "transitions";
      if (ranks > 1) {
        auto badContext = context;
        badContext.id[0] ^= 0x5a;
        auto badOptions = options;
        if (rank == ranks - 1) badOptions.fallback.identity += ";mismatch";
        PreparedLocalRuntime bad(badContext, badOptions);
        LocalRequest request{
            NetworkOperation::AllReduce, 64,
            std::vector<RankBufferLayout>(ranks, {false, 256})};
        bool rejected = false;
        try {
          bad.prepare(request);
        } catch (std::invalid_argument const&) {
          rejected = true;
        }
        if (!rejected || bad.peerInitialized() || bad.hostInitialized() ||
            fallback.backend->initialized())
          throw std::runtime_error(
              "fallback implementation disagreement initialized a backend");
      }
      PreparedLocalRuntime runtime(context, options);
      if (ranks > 1) {
        LocalRequest request{NetworkOperation::AllReduce, 64,
                             std::vector<RankBufferLayout>(ranks, {false, 256}),
                             rank == ranks - 1};
        bool rejected = false;
        try {
          runtime.prepare(request);
        } catch (std::invalid_argument const&) {
          rejected = true;
        }
        if (!rejected || runtime.peerInitialized() ||
            runtime.hostInitialized() || fallback.backend->initialized())
          throw std::runtime_error(
              "per-rank fallback policy disagreement accepted");
      }
      std::array<cudaStream_t, 2> streams;
      for (auto& stream : streams)
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      if (mode == "transitions")
        transition(runtime, rank, ranks, streams);
      else {
        bool needPeer = false, needHost = false, needTree = false;
        std::vector<size_t> plans;
        for (size_t count : {64UL, 257UL, 65536UL})
          for (int op = 1; op <= 3; ++op) {
            LocalRequest request{NetworkOperation(op), count, {}};
            for (int r = 0; r < ranks; ++r)
              request.layouts.push_back(
                  {count == 65536 || (count == 257 && (r & 1)),
                   count == 257 ? 2u : 2u << (r % 4)});
            auto plan = runtime.prepare(request);
            plans.push_back(plan);
            if (runtime.prepare(request) != plan)
              throw std::runtime_error("native baseline live cache missed");
            auto const& choice = runtime.selection(plan);
            std::array<std::vector<double>, 4> samples{
                {choice.samples[0], choice.samples[1], choice.samples[3],
                 choice.samples[4]}};
            if (choice.backend !=
                qualifiedFallbackAlternative(choice.fallbackSamples, samples,
                                             choice.candidateCorrect))
              throw std::runtime_error(
                  "native admission differs from evidence");
            needPeer |= choice.backend == LocalBackend::Peer;
            needHost |= choice.backend == LocalBackend::Host;
            needTree |= choice.backend == LocalBackend::PeerTree;
            if (runtime.peerInitialized() != needPeer ||
                runtime.hostInitialized() != needHost ||
                runtime.peerTreeInitialized() != needTree)
              throw std::runtime_error("unused alternative resources retained");
            if ((mode == "fallback" || ranks == 1) &&
                (needPeer || needHost || needTree ||
                 !choice.samples[2].empty()))
              throw std::runtime_error(
                  "fallback-only initialized or benchmarked alternatives");
            if (mode == "host-only" &&
                (runtime.peerInitialized() || runtime.peerTreeInitialized() ||
                 choice.peerEligible || choice.peerTreeEligible))
              throw std::runtime_error("excluded peer initialized");
            if (!rank) {
              std::cout << "op=" << op << " count=" << count
                        << " selected=" << int(choice.backend)
                        << " baseline=" << choice.fallbackIdentity;
              for (int c = 0; c < 2; ++c) {
                std::cout << " candidate" << c << "_us=";
                for (double us : choice.samples[c]) std::cout << us << ',';
                std::cout << " baseline" << c << "_us=";
                for (double us : choice.fallbackSamples[c])
                  std::cout << us << ',';
              }
              std::cout << std::endl;
            }
            execute(runtime, plan, rank, ranks, streams);
          }
        for (auto plan : plans) execute(runtime, plan, rank, ranks, streams);
      }
      runtime.synchronize();
      // Cache hits still validate the caller baseline's live configuration.
      auto request = runtime.selection(0).request;
      auto previous = std::getenv("NCCL_DEBUG");
      bool hadDebug = previous != nullptr;
      std::string debug = previous ? previous : "";
      if (rank == ranks - 1)
        setenv("NCCL_DEBUG", debug == "INFO" ? "VERSION" : "INFO", 1);
      bool refused = false;
      try {
        runtime.prepare(request);
      } catch (std::runtime_error const&) {
        refused = true;
      }
      int localRefused = refused, allRefused = 0;
      MPI_Allreduce(&localRefused, &allRefused, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      if (allRefused != ranks)
        throw std::runtime_error(
            "cached plan ignored changed native environment");
      refused = false;
      try {
        runtime.enqueue(0, nullptr, nullptr, streams[0]);
      } catch (std::logic_error const& error) {
        refused =
            std::string(error.what())
                .find("successful collective preparation") != std::string::npos;
      }
      if (!refused)
        throw std::runtime_error("failed prepare left old plans executable");
      if (rank == ranks - 1) {
        if (hadDebug)
          setenv("NCCL_DEBUG", debug.c_str(), 1);
        else
          unsetenv("NCCL_DEBUG");
      }
      if (runtime.prepare(request) != 0)
        throw std::runtime_error(
            "restored fallback configuration lost live cache");
      execute(runtime, 0, rank, ranks, streams);
      for (auto stream : streams) check(cudaStreamDestroy(stream));
    }
    // The adapter borrowed, rather than replaced or destroyed, this
    // communicator.
    int count = 0;
    nativeCheck(ncclCommCount(comm, &count));
    if (count != ranks)
      throw std::runtime_error("caller communicator ownership lost");
    __nv_bfloat16 *in, *out;
    check(cudaMalloc(&in, 128));
    check(cudaMalloc(&out, 128));
    std::vector<__nv_bfloat16> values(64, __float2bfloat16_rn(1));
    check(cudaMemcpy(in, values.data(), 128, cudaMemcpyHostToDevice));
    nativeCheck(
        ncclAllReduce(in, out, 64, ncclBfloat16, ncclSum, comm, nullptr));
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(values.data(), out, 128, cudaMemcpyDeviceToHost));
    for (auto v : values)
      if (__bfloat162float(v) != ranks)
        throw std::runtime_error(
            "caller communicator failed after adapter destruction");
    check(cudaFree(in));
    check(cudaFree(out));
    nativeCheck(ncclCommDestroy(comm));
    if (!rank)
      std::cout << "PASS: native caller fallback mode=" << mode
                << " ranks=" << ranks
                << "; guards, aliases, cache, ownership and stream ordering\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

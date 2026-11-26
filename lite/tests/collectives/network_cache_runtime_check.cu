#include "native_fallback.hpp"
#include "scalar_oracle.cuh"
#include "uccl_lite/network_cache.hpp"
#include "uccl_lite/runtime/prepared_network.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <mpi.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
using scalar_test::check;
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void native(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(status));
}
class SchedulingDelay : public CollectiveBackend {
  std::shared_ptr<CollectiveBackend> implementation;
  int milliseconds;
  bool corrupt = false;

 public:
  int calls = 0;
  SchedulingDelay(std::shared_ptr<CollectiveBackend> p, int ms, bool bad)
      : implementation(std::move(p)), milliseconds(ms), corrupt(bad) {}
  void prepare() override { implementation->prepare(); }
  void enqueue(NetworkOperation op, void const* in, void* out, size_t count,
               cudaStream_t stream) override {
    ++calls;
    implementation->enqueue(op, in, out, count, stream);
    if (corrupt) check(cudaMemsetAsync(out, 0, count * sizeof(float), stream));
    if (milliseconds)
      std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
  }
  void synchronize() override { implementation->synchronize(); }
  bool initialized() const override { return implementation->initialized(); }
  bool idle() const override { return implementation->idle(); }
};
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
    require(argc == 4 && ranks == 8 && localRanks == 4,
            "NATIVE_LIBRARY CACHE_FILE "
            "cold|warm|drift|native-cold|native-warm|stale|corrupt|fallback|"
            "mapping|bad-baseline|resource-mismatch; "
            "two four-GPU hosts");
    std::string mode = argv[3];
    require(mode == "cold" || mode == "warm" || mode == "drift" ||
                mode == "native-cold" || mode == "native-warm" ||
                mode == "stale" || mode == "corrupt" || mode == "fallback" ||
                mode == "mapping" || mode == "bad-baseline" ||
                mode == "resource-mismatch",
            "invalid mode");
    require(!std::getenv("UCCL_LITE_NETWORK_PROFILE"),
            "cache fixture requires absent explicit profile");
    int device = localRanks - 1 - localRank;
    check(cudaSetDevice(device));
    ncclUniqueId id;
    if (!rank) native(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    native(ncclCommInitRank(&comm, ranks, id, rank));
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::optional<NetworkPlanCache> original;
    if (!rank && (mode == "mapping" || mode == "resource-mismatch")) {
      original = loadNetworkPlanCache(argv[2]);
      require(original && !original->records.empty(),
              "fixture needs a populated cache");
      auto changed = *original;
      if (mode == "mapping")
        std::swap(changed.records[0].rankOrder[0],
                  changed.records[0].rankOrder[1]);
      else
        changed.calibration->hosts[0].transportIdentity += ";fixture-mismatch";
      updateNetworkPlanCache(argv[2], changed, changed.records[0]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    {
      BackendContext context;
      context.rank = rank;
      context.device = device;
      context.ranks = ranks;
      context.capacity = 2UL << 20;
      context.localDevices.resize(localRanks);
      std::iota(context.localDevices.begin(), context.localDevices.end(), 0);
      if (!rank) context.id = PreparedNetworkRuntime::createUniqueId();
      MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
      NetworkPreparationOptions options;
      options.allowDirect =
          false;  // This suite isolates the retained hierarchy contract.
      options.planCache = argv[2];
      options.probeResources =
          mode != "warm" && mode != "native-warm" && mode != "drift";
      options.fallbackOnly = mode == "fallback";
      options.fallback = integration::makeNativeNcclFallback(
          context, comm, argv[1],
          mode == "stale" ? "cache-test-new-configuration"
                          : "cache-test-variable-scheduling-correctness-only",
          ScalarType::Float32);
      int delay = (mode == "cold" || mode == "warm" || mode == "stale" ||
                   mode == "mapping")
                      ? 3
                      : 0;
      auto counter = std::make_shared<SchedulingDelay>(
          options.fallback.backend, delay, mode == "bad-baseline");
      options.fallback.backend = counter;
      options.fallback.identity += ";test-only-variable-scheduling-delay";
      PreparedNetworkRuntime runtime(context, options);
      bool callerCase = mode == "native-cold" || mode == "native-warm";
      auto operation = callerCase ? NetworkOperation::AllGather
                                  : NetworkOperation::AllReduce;
      size_t count = callerCase ? 64 : 4096;
      CollectiveRequest request{
          operation, count, std::vector<RankBufferLayout>(ranks, {false, 4})};
      if (mode == "corrupt" || mode == "resource-mismatch" ||
          mode == "bad-baseline") {
        bool rejected = false;
        try {
          runtime.prepare(request);
        } catch (std::exception const&) {
          rejected = true;
        }
        require(rejected && !runtime.networkInitialized() &&
                    counter->calls == (mode == "bad-baseline" ? 5 : 0),
                "invalid snapshot/baseline reached candidate execution");
      } else {
        auto plan = runtime.prepare(request);
        auto const& selection = runtime.selection(plan);
        if (mode == "warm") {
          require(selection.fromCache &&
                      selection.backend == NetworkBackend::Hierarchy &&
                      counter->calls == 10,
                  "positive cache checkpoint did not reuse plan");
          require(selection.validationCandidateSamples.size() == 6 &&
                      selection.validationBaselineSamples.size() == 6,
                  "cache ABBA evidence missing");
        } else if (mode == "native-warm") {
          require(selection.fromCache &&
                      selection.backend == NetworkBackend::Caller &&
                      counter->calls == 5 && !runtime.networkInitialized(),
                  "cached caller initialized candidate or repeated tuning");
          require(selection.validationCandidateSamples.empty() &&
                      selection.validationBaselineSamples.size() == 3,
                  "caller-only checkpoint scope wrong");
        } else if (mode == "fallback") {
          require(!selection.fromCache && counter->calls == 0 &&
                      !runtime.networkInitialized(),
                  "fallback accessed corrupt cache or initialized network");
        } else if (mode == "drift") {
          require(!selection.fromCache && counter->calls == 40 &&
                      selection.cacheStatus.find("performance drift") !=
                          std::string::npos,
                  "changed performance failed to refresh qualification");
          require(selection.backend == NetworkBackend::Caller,
                  "drift fixture did not return to native");
        } else {
          require(!selection.fromCache && counter->calls == 30,
                  "cold/context-miss path did not fully qualify");
          require(selection.backend == (callerCase ? NetworkBackend::Caller
                                                   : NetworkBackend::Hierarchy),
                  "fixture did not produce expected cached decision");
          if (mode == "stale")
            require(selection.cacheStatus == "identity miss",
                    "configuration change reused stale snapshot");
          if (mode == "mapping")
            require(selection.cacheStatus == "execution-map miss",
                    "different cached execution map reused");
        }
        if (mode != "fallback")
          require(selection.candidateSamples.size() == 18 &&
                      selection.baselineSamples.size() == 18 &&
                      selection.calibration,
                  "original evidence/profile not retained");
        int calls = counter->calls;
        require(runtime.prepare(request) == plan && counter->calls == calls,
                "live cache repeated persisted checkpoint");
        scalar_test::exercise<uint32_t>(
            int(operation), count, false, 4, rank, ranks, streams,
            [&](int, void const* in, void* out, size_t, cudaStream_t stream) {
              runtime.enqueue(plan, in, out, stream);
              runtime.synchronize();
            });
        if (!rank)
          std::cout << "mode=" << mode << " from_cache=" << selection.fromCache
                    << " backend=" << int(selection.backend)
                    << " baseline_calls=" << calls << " checkpoint="
                    << selection.validationCandidateSamples.size() << '/'
                    << selection.validationBaselineSamples.size()
                    << " cache_status=" << selection.cacheStatus << '\n';
      }
    }
    if (!rank && mode == "resource-mismatch")
      updateNetworkPlanCache(argv[2], *original, original->records[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    scalar_test::exercise<uint32_t>(
        3, 64, false, 4, rank, ranks, streams,
        [&](int, void const* in, void* out, size_t count, cudaStream_t stream) {
          native(ncclAllReduce(in, out, count, ncclFloat32, ncclSum, comm,
                               stream));
        });
    for (auto stream : streams) check(cudaStreamDestroy(stream));
    native(ncclCommDestroy(comm));
    if (!rank)
      std::cout << "PASS: persistent distributed plan " << mode
                << "; independent buffers/guards, exact checkpoint scope, live "
                   "reuse and caller ownership; scheduling fixture is not "
                   "performance evidence\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

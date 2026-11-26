#include "native_fallback.hpp"
#include "scalar_oracle.cuh"
#include "uccl_lite/runtime/prepared_network.hpp"
#include <iostream>
#include <numeric>
#include <mpi.h>
#include <sched.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
using scalar_test::check;
void native(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(status));
}
void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
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
    require(argc == 4 && ranks == 8 && localRanks == 4,
            "usage: automatic_network_check NATIVE_LIBRARY "
            "automatic|no-probe|fallback PROFILE_OUTPUT; two four-GPU hosts");
    std::string mode = argv[2];
    require(mode == "automatic" || mode == "no-probe" || mode == "fallback",
            "invalid mode");
    require(!std::getenv("UCCL_LITE_NETWORK_PROFILE"),
            "automatic check requires absent input profile");
    int device = localRanks - 1 - localRank;
    check(cudaSetDevice(device));
    ncclUniqueId id;
    if (!rank) native(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    native(ncclCommInitRank(&comm, ranks, id, rank));
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
    options.fallback = integration::makeNativeNcclFallback(
        context, comm, argv[1],
        "automatic-network-resource-check;blocking-defaults",
        ScalarType::Float32);
    options.probeResources = mode != "no-probe";
    options.fallbackOnly = mode == "fallback";
    std::array<cudaStream_t, 2> streams;
    for (auto& s : streams)
      check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    {
      PreparedNetworkRuntime runtime(context, options);
      std::shared_ptr<NetworkProfile const> profile;
      for (size_t count : {size_t(4096), size_t(65536)}) {
        CollectiveRequest request{
            NetworkOperation::AllReduce, count,
            std::vector<RankBufferLayout>(ranks, {false, 4})};
        cpu_set_t before{}, after{};
        require(!sched_getaffinity(0, sizeof(before), &before),
                "cannot read affinity");
        auto plan = runtime.prepare(request);
        require(!sched_getaffinity(0, sizeof(after), &after) &&
                    CPU_EQUAL(&before, &after),
                "prepare changed caller affinity");
        auto const& selected = runtime.selection(plan);
        if (mode == "automatic") {
          require(selected.candidateEligible && selected.candidateCorrect &&
                      selected.calibration,
                  "missing automatic calibration/correct candidate");
          if (profile)
            require(profile == selected.calibration,
                    "resource profile was reprobed/replaced");
          profile = selected.calibration;
          require(profile->groupScope == kNetworkBootstrapGroupProbeScope &&
                      profile->pathScope == kNetworkProbeScope &&
                      profile->paths.size() >= 8 && !profile->groups.empty(),
                  "missing scope or raw resource observations");
          for (auto const& path : profile->paths) {
            require(path.rejection.empty() && path.sizes.size() == 3,
                    "path calibration incomplete");
            for (auto const& size : path.sizes)
              require(size.microseconds.size() == 15 &&
                          bool(estimateNetworkPath(*profile, profile->hosts,
                                                   path.key, size.bytes)),
                      "directed path samples unavailable");
          }
          for (auto const& group : profile->groups) {
            require(group.rejection.empty() && group.flows.size() == 8 &&
                        group.sizes.size() == 2,
                    "concurrent group incomplete");
            for (auto const& size : group.sizes)
              require(networkGroupSamples(group, size.bytes).size() == 15 &&
                          bool(estimateNetworkGroup(*profile, profile->hosts,
                                                    group.flows, size.bytes)),
                      "complete cohort samples unavailable");
          }
        } else {
          require(!selected.calibration && !selected.candidateEligible &&
                      !runtime.networkInitialized(),
                  "disabled/missing-evidence fallback created accelerated "
                  "resources");
        }
        require(runtime.prepare(request) == plan &&
                    runtime.selection(plan).calibration == profile,
                "cached preparation replaced resource evidence");
        scalar_test::exercise<uint32_t>(
            3, count, false, 4, rank, ranks, streams,
            [&](int, void const* a, void* b, size_t, cudaStream_t s) {
              runtime.enqueue(plan, a, b, s);
              runtime.synchronize();
            });
        if (mode == "automatic" && count == 4096) {
          if (!rank) {
            cpu_set_t changed;
            CPU_ZERO(&changed);
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
              if (CPU_ISSET(cpu, &before)) {
                CPU_SET(cpu, &changed);
                break;
              }
            require(!sched_setaffinity(0, sizeof(changed), &changed),
                    "cannot alter test affinity");
          }
          bool rejected = false;
          try {
            runtime.prepare(request);
          } catch (std::exception const&) {
            rejected = true;
          }
          if (!rank)
            require(!sched_setaffinity(0, sizeof(before), &before),
                    "cannot restore test affinity");
          require(rejected,
                  "changed resource identity reused cached preparation");
          require(
              runtime.prepare(request) == plan &&
                  runtime.selection(plan).calibration == profile,
              "identity rejection recovery reprobed or lost the cached plan");
        }
        if (!rank)
          std::cout << "count=" << count << " backend=" << int(selected.backend)
                    << " calibrated_paths="
                    << (profile ? profile->paths.size() : 0)
                    << " cohorts=" << (profile ? profile->groups.size() : 0)
                    << '\n';
      }
      if (!rank && profile) {
        saveNetworkProfile(argv[3], *profile);
        auto loaded = loadNetworkProfile(argv[3]);
        require(loaded.groupScope == profile->groupScope &&
                    loaded.probeIdentity == profile->probeIdentity &&
                    loaded.paths.size() == profile->paths.size() &&
                    loaded.groups.size() == profile->groups.size(),
                "automatic profile persistence lost provenance");
      }
    }
    scalar_test::exercise<uint32_t>(
        3, 64, false, 4, rank, ranks, streams,
        [&](int, void const* a, void* b, size_t n, cudaStream_t s) {
          native(ncclAllReduce(a, b, n, ncclFloat32, ncclSum, comm, s));
        });
    for (auto s : streams) check(cudaStreamDestroy(s));
    native(ncclCommDestroy(comm));
    if (!rank)
      std::cout
          << "PASS: automatic network resources " << mode
          << "; profile policy, caller collective oracle, live profile reuse, "
             "affinity and caller ownership"
          << (mode == "automatic" ? "; full private path/cohort verification"
                                  : "")
          << '\n';
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

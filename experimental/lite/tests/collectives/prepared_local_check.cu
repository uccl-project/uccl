#include "prepared_local_oracle.cuh"
#include "uccl_lite/runtime/prepared_local.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
using namespace uccl_lite_test;
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    check(cudaSetDevice(rank));
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
    LocalPreparationOptions options;
    if (argc > 1 && std::string(argv[1]) == "no-probe")
      options.probeResources = false;
    PreparedLocalRuntime runtime(context, options);
    if (runtime.hostInitialized())
      throw std::runtime_error("construction initialized host transport");
    LocalRequest mismatch{NetworkOperation::AllReduce, rank ? 128UL : 64UL,
                          std::vector<RankBufferLayout>(ranks)};
    bool rejected = false;
    try {
      runtime.prepare(mismatch);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    if (!rejected || runtime.hostInitialized())
      throw std::runtime_error("rank mismatch admission mutated host state");
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<size_t> plans;
    bool hostNeeded = false;
    for (size_t count : {64UL, 257UL, 65536UL}) {
      for (int op = 1; op <= 3; ++op) {
        LocalRequest request{NetworkOperation(op), count, {}};
        for (int r = 0; r < ranks; ++r)
          request.layouts.push_back(
              {count == 65536 || (count == 257 && (r & 1)),
               count == 257 ? 2u : 2u << (r % 4)});
        size_t plan = runtime.prepare(request);
        if (runtime.prepare(request) != plan)
          throw std::runtime_error(
              "identical prepared workload missed the live cache");
        plans.push_back(plan);
        auto const& result = runtime.selection(plan);
        if (result.backend != qualifiedLocalBackend(result.samples[0],
                                                    result.samples[1],
                                                    result.hostCorrect))
          throw std::runtime_error(
              "common decision disagrees with raw evidence");
        hostNeeded |= result.backend == LocalBackend::Host;
        if (runtime.hostInitialized() != hostNeeded)
          throw std::runtime_error("unused host buffers/workers retained");
        if (!rank) {
          std::cout << "plan op=" << op << " count=" << count << " selected="
                    << (result.backend == LocalBackend::Peer ? "peer" : "host")
                    << " resource_us=" << result.peerResourceEstimateUs
                    << " topology=" << result.topologyIdentity
                    << " artifact=" << result.executorIdentity;
          for (int backend = 0; backend < 2; ++backend) {
            std::cout << (backend ? " host_us=" : " peer_us=");
            for (double sample : result.samples[backend])
              std::cout << sample << ',';
          }
          std::cout << std::endl;
        }
        execute(runtime, plan, rank, ranks, streams);
      }
    }
    // Earlier plans survive new prepare calls and host teardown/recreation.
    for (size_t plan : plans) execute(runtime, plan, rank, ranks, streams);
    for (auto stream : streams) check(cudaStreamDestroy(stream));
    if (!rank)
      std::cout << "PASS: collective prepared selection, mismatched-rank "
                   "rejection, mixed layouts, scalar tails, old-plan reuse and "
                   "independent guards\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

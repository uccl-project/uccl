#include "prepared_abi.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
void check(cudaError_t r) {
  if (r != cudaSuccess) throw std::runtime_error(cudaGetErrorString(r));
}
void nccl(ncclResult_t r) {
  if (r != ncclSuccess)
    throw std::runtime_error(std::string(ncclGetErrorString(r)) + ": " +
                             ncclGetLastError(nullptr));
}
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
std::string report(ncclComm_t comm) {
  size_t n = 0;
  nccl(ucclLiteNcclReport(comm, nullptr, &n));
  std::vector<char> data(n);
  nccl(ucclLiteNcclReport(comm, data.data(), &n));
  return data.data();
}
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 200000000) {
  }
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
    require(
        argc == 3 || (argc == 4 && std::string(argv[3]) == "auto"),
        "NATIVE_LIBRARY "
        "group|mixed-group|capture|p2p|layout|zero|reduction|simulate|eager|"
        "ordinary|fallback");
    std::string mode = argv[2];
    require(mode == "group" || mode == "capture" || mode == "p2p" ||
                mode == "layout" || mode == "ordinary" || mode == "fallback" ||
                mode == "mixed-group" || mode == "zero" ||
                (mode == "eager" || mode == "portfolio") ||
                mode == "reduction" || mode == "simulate",
            "bad mode");
    int visible;
    check(cudaGetDeviceCount(&visible));
    int device = visible == 1 ? 0 : localRanks - 1 - localRank;
    check(cudaSetDevice(device));
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr size_t count = 64, guard = 64;
    size_t extent = std::max(count * size_t(ranks), size_t(257)) + 2 * guard;
    float *a, *b, *c, *d;
    for (auto pointer : {&a, &b, &c, &d})
      check(cudaMalloc(pointer, extent * 4));
    auto initialize = [&](int generation) {
      std::vector<float> x(extent, -99), empty(extent, -99);
      for (size_t i = 0; i < count; ++i) x[guard + i] = rank + 1 + generation;
      check(cudaMemcpy(a, x.data(), extent * 4, cudaMemcpyHostToDevice));
      for (auto p : {b, c, d})
        check(cudaMemcpy(p, empty.data(), extent * 4, cudaMemcpyHostToDevice));
    };
    auto verify = [&](float* pointer, size_t offset, size_t elements,
                      float value) {
      std::vector<float> actual(extent);
      check(cudaMemcpy(actual.data(), pointer, extent * 4,
                       cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < extent; ++i)
        if (actual[i] != (i >= offset && i - offset < elements ? value : -99)) {
          std::cerr << "ABI mismatch rank=" << rank << " buffer="
                    << (pointer == a   ? 'a'
                        : pointer == b ? 'b'
                        : pointer == c ? 'c'
                                       : 'd')
                    << " index=" << i << " got=" << actual[i] << " expected="
                    << (i >= offset && i - offset < elements ? value : -99)
                    << '\n';
          throw std::runtime_error(
              "prepared ABI full-buffer/guard reference mismatch");
        }
    };
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      ncclUniqueId nativeId;
      if (!rank) nccl(ncclGetUniqueId(&nativeId));
      MPI_Bcast(&nativeId, sizeof(nativeId), MPI_BYTE, 0, MPI_COMM_WORLD);
      ncclComm_t comm;
      nccl(ncclCommInitRank(&comm, ranks, nativeId, rank));
      if (argc != 4) {
        ucclLiteUniqueId id;
        if (!rank) nccl(ucclLiteGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ucclLiteOptions options;
        nccl(ucclLiteOptionsInit(&options));
        options.nativeLibrary = argv[1];
        options.configurationIdentity =
            "prepared-ABI-check;test-fixture-if-native-shim";
        options.capacityBytes = 1UL << 20;
        options.dtype = ncclFloat32;
        if (mode == "fallback") options.flags |= UCCL_LITE_FALLBACK_ONLY;
        if (mode != "ordinary" && mode != "fallback")
          options.flags &=
              ~(UCCL_LITE_ALLOW_HOST | UCCL_LITE_ALLOW_PEER_PACKET);
        std::vector<ucclLiteRankLayout> layouts(ranks, {0, 256});
        ucclLiteRequest request{ucclLiteAllReduce, count, layouts.data(),
                                size_t(ranks), 0};
        std::vector<ucclLiteRequest> requests{request};
        if ((mode == "eager" || mode == "portfolio")) {
          auto ag = request;
          ag.operation = ucclLiteAllGather;
          requests.push_back(ag);
          auto rs = request;
          rs.operation = ucclLiteReduceScatter;
          requests.push_back(rs);
        }
        if (mode == "portfolio") {
          auto tail = request;
          tail.count = 257;
          requests.push_back(tail);
        }
        nccl(ucclLiteNcclPrepare(comm, &id, &options, requests.data(),
                                 requests.size()));
      }
      auto first = report(comm);
      if (argc == 4)
        require(first.find("\"automatic_init\":true") != std::string::npos,
                "initialization did not automatically prepare");
      if (!rank) std::cout << "PREPARED " << first << '\n';
      bool accelerated = first.find("\"admitted\":true") != std::string::npos;
      if (mode != "ordinary" && mode != "fallback")
        require(accelerated,
                "delayed-native fixture did not admit whole ABI path");
      initialize(0);
      if (rank == ranks - 1) skew<<<1, 32, 0, streams[0]>>>();
      check(cudaGetLastError());
      nccl(((mode == "eager" || mode == "portfolio") ? pncclAllReduce
                                                     : ncclAllReduce)(
          a + guard, b + guard, count, ncclFloat32, ncclSum, comm, streams[0]));
      float sum = float(ranks * (ranks + 1) / 2), expectedB = sum;
      ncclResult_t async = ncclSuccess;
      nccl(ncclCommGetAsyncError(comm, &async));
      require(async == ncclSuccess,
              "unexpected asynchronous communicator error");
      if (mode == "capture") {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(streams[1], cudaStreamCaptureModeGlobal));
        nccl(ncclAllReduce(b + guard, c + guard, count, ncclFloat32, ncclSum,
                           comm, streams[1]));
        check(cudaStreamEndCapture(streams[1], &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        nccl(ucclLiteNcclRelease(
            comm));  // event lease must survive resource retirement
        for (int replay = 0; replay < 3; ++replay) {
          if (replay) {
            std::vector<float> input(extent, -99);
            for (size_t i = 0; i < count; ++i) input[guard + i] = sum + replay;
            check(cudaMemcpy(b, input.data(), extent * 4,
                             cudaMemcpyHostToDevice));
          }
          check(cudaGraphLaunch(executable, streams[1]));
          check(cudaStreamSynchronize(streams[1]));
          verify(c, guard, count, ranks * (sum + replay));
        }
        expectedB = sum + 2;
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(graph));
      } else if (mode == "group" || mode == "mixed-group") {
        if (mode == "group" || rank == 0) {
          nccl(ncclGroupStart());
          nccl(ncclGroupStart());
        }
        auto firstGrouped =
            ncclAllReduce(b + guard, c + guard, count, ncclFloat32, ncclSum,
                          comm, streams[1]);
        if (mode == "mixed-group") {
          require(firstGrouped == ncclInvalidUsage,
                  "mixed group status was not rejected collectively");
          if (rank == 0) {
            nccl(ncclGroupEnd());
            nccl(ncclGroupEnd());
          }
          nccl(ncclAllReduce(b + guard, c + guard, count, ncclFloat32, ncclSum,
                             comm, streams[1]));
        } else
          nccl(firstGrouped);
        nccl(ncclAllGather(b + guard, d + guard, count, ncclFloat32, comm,
                           streams[1]));
        if (mode == "group") {
          nccl(ncclGroupEnd());
          nccl(ncclGroupEnd());
        }
        check(cudaStreamSynchronize(streams[1]));
        verify(c, guard, count, ranks * sum);
        verify(d, guard, count * ranks, sum);
      } else if (mode == "p2p") {
        require(ranks >= 2, "P2P fixture needs peers");
        if (rank < 2) {
          nccl(ncclGroupStart());
          if (!rank)
            nccl(ncclSend(b + guard, count, ncclFloat32, 1, comm, streams[1]));
          else
            nccl(ncclRecv(c + guard, count, ncclFloat32, 0, comm, streams[1]));
          nccl(ncclGroupEnd());
        }
        nccl(ncclAllReduce(b + guard, d + guard, count, ncclFloat32, ncclSum,
                           comm, streams[1]));
        check(cudaStreamSynchronize(streams[1]));
        verify(d, guard, count, ranks * sum);
        if (rank == 1)
          verify(c, guard, count, sum);
        else
          verify(c, guard, 0, 0);
      } else if (mode == "simulate") {
        ncclSimInfo_t info = NCCL_SIM_INFO_INITIALIZER;
        nccl(ncclGroupStart());
        nccl(ncclAllReduce(b + guard, c + guard, count, ncclFloat32, ncclSum,
                           comm, streams[1]));
        nccl(ncclGroupSimulateEnd(&info));
        check(cudaStreamSynchronize(streams[1]));
        verify(c, guard, 0, 0);
        nccl(ncclAllReduce(b + guard, c + guard, count, ncclFloat32, ncclSum,
                           comm, streams[1]));
        check(cudaStreamSynchronize(streams[1]));
        verify(c, guard, count, ranks * sum);
      } else if (mode == "reduction") {
        nccl(ncclAllReduce(b + guard, c + guard, count, ncclFloat32, ncclMax,
                           comm, streams[1]));
        check(cudaStreamSynchronize(streams[1]));
        verify(c, guard, count, sum);
      } else if (mode == "zero") {
        if (!rank)
          nccl(ncclAllReduce(b + guard, d + guard, 0, ncclFloat32, ncclSum,
                             comm, streams[1]));
        nccl(ncclAllReduce(b + guard, c + guard, count, ncclFloat32, ncclSum,
                           comm, streams[1]));
        check(cudaStreamSynchronize(streams[1]));
        verify(c, guard, count, ranks * sum);
        verify(d, guard, 0, 0);
      } else if ((mode == "eager" || mode == "portfolio")) {
        nccl(ncclAllGather(b + guard, d + guard, count, ncclFloat32, comm,
                           streams[1]));
        nccl(ncclReduceScatter(d + guard, c + guard, count, ncclFloat32,
                               ncclSum, comm, streams[0]));
        check(cudaStreamSynchronize(streams[0]));
        verify(d, guard, count * ranks, sum);
        verify(c, guard, count, ranks * sum);
        require(
            report(comm).find("\"accelerated_calls\":3") != std::string::npos,
            "all three standard collectives did not accelerate");
        if (mode == "portfolio") {
          std::vector<float> source(extent, -99), empty(extent, -99);
          for (size_t i = 0; i < 257; ++i) source[guard + i] = rank + 1;
          check(cudaMemcpyAsync(a, source.data(), extent * 4,
                                cudaMemcpyHostToDevice, streams[1]));
          check(cudaMemcpyAsync(d, empty.data(), extent * 4,
                                cudaMemcpyHostToDevice, streams[1]));
          nccl(ncclAllReduce(a + guard, d + guard, 257, ncclFloat32, ncclSum,
                             comm, streams[1]));
          check(cudaStreamSynchronize(streams[1]));
          verify(a, guard, 257, rank + 1);
          verify(d, guard, 257, sum);
          require(
              report(comm).find("\"accelerated_calls\":4") != std::string::npos,
              "portfolio tail did not accelerate");
        }
      } else if (mode == "layout") {
        check(cudaStreamSynchronize(streams[0]));
        std::vector<float> x(extent, -99);
        size_t offset = rank == ranks - 1 ? guard + 1 : guard;
        for (size_t i = 0; i < count; ++i) x[offset + i] = rank + 1;
        check(cudaMemcpy(a, x.data(), extent * 4, cudaMemcpyHostToDevice));
        nccl(ncclAllReduce(a + offset, c + offset, count, ncclFloat32, ncclSum,
                           comm, streams[1]));
        check(cudaStreamSynchronize(streams[1]));
        verify(c, offset, count, sum);
      } else {
        nccl(ncclAllReduce(b + guard, c + guard, count, ncclFloat32, ncclSum,
                           comm, streams[1]));
        check(cudaStreamSynchronize(streams[1]));
        verify(c, guard, count, ranks * sum);
      }
      check(cudaStreamSynchronize(streams[0]));
      verify(b, guard, count, expectedB);
      if (mode != "ordinary" && mode != "fallback" &&
          (mode != "eager" && mode != "portfolio"))
        require(report(comm).find("\"native_owned\":true") != std::string::npos,
                "native ownership not agreed");
      nccl(ucclLiteNcclRelease(comm));
      initialize(1);
      nccl(((mode == "eager" || mode == "portfolio") ? pncclAllReduce
                                                     : ncclAllReduce)(
          a + guard, b + guard, count, ncclFloat32, ncclSum, comm, streams[0]));
      check(cudaStreamSynchronize(streams[0]));
      verify(b, guard, count, sum + ranks);
      if (!rank) std::cout << "FINAL " << report(comm) << '\n';
      nccl(ncclCommDestroy(comm));
    }
    for (auto p : {a, b, c, d}) check(cudaFree(p));
    for (auto stream : streams) check(cudaStreamDestroy(stream));
    if (!rank)
      std::cout
          << "PASS: prepared NCCL ABI " << mode << " ranks=" << ranks
          << "; full buffers/guards, cross-stream handoff, native "
             "groups/capture/P2P, resource retirement and two lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

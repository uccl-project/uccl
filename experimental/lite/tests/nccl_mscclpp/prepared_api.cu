#include "../collectives/scalar_oracle.cuh"
#include "prepared_api.h"
#include <iostream>
#include <string>
#include <mpi.h>
using scalar_test::check;
void api(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(std::string("prepared API: ") +
                             ucclLiteLastError());
}
void native(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(status));
}
void rejected(ncclResult_t status) {
  if (status == ncclSuccess || !*ucclLiteLastError())
    throw std::runtime_error("expected API rejection and diagnostic");
}
std::string report(ucclLiteSession session, ucclLitePlan plan) {
  size_t size = 0;
  api(ucclLitePlanReport(session, plan, nullptr, &size));
  std::vector<char> buffer(size, '~');
  size_t shortSize = 1;
  rejected(ucclLitePlanReport(session, plan, buffer.data(), &shortSize));
  if (shortSize != size || buffer.front() != '~')
    throw std::runtime_error("report size/write contract");
  api(ucclLitePlanReport(session, plan, buffer.data(), &size));
  return buffer.data();
}
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 30000000) {
  }
}
void graphs(ucclLiteSession session, int rank, int ranks,
            std::array<cudaStream_t, 2> streams) {
  constexpr size_t count = 64, guard = 64, extent = count + 2 * guard;
  float *a, *b, *c;
  check(cudaMalloc(&a, extent * sizeof(float)));
  check(cudaMalloc(&b, extent * sizeof(float)));
  check(cudaMalloc(&c, extent * sizeof(float)));
  std::vector<ucclLiteRankLayout> layouts(ranks, {0, 256});
  ucclLiteRequest req{ucclLiteAllReduce, count, layouts.data(), size_t(ranks),
                      0};
  ucclLitePlan graph, eager;
  api(ucclLitePrepareGraph(session, &req, a + guard, b + guard, &graph));
  api(ucclLitePrepare(session, &req, &eager));
  if (report(session, graph).find("\"graph\":true") == std::string::npos)
    throw std::runtime_error("graph report mode");
  rejected(ucclLiteEnqueue(session, graph, a + guard, b + guard, streams[0]));
  rejected(ucclLiteLaunchGraph(session, eager, streams[0]));
  cudaGraph_t external;
  check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
  rejected(ucclLiteLaunchGraph(session, graph, streams[0]));
  check(cudaStreamEndCapture(streams[0], &external));
  check(cudaGraphDestroy(external));
  std::vector<float> ha(extent, -99), hb(extent, -99), hc(extent, -99);
  for (size_t i = 0; i < count; ++i) ha[guard + i] = rank + 1;
  check(cudaMemcpy(a, ha.data(), extent * 4, cudaMemcpyHostToDevice));
  check(cudaMemcpy(b, hb.data(), extent * 4, cudaMemcpyHostToDevice));
  check(cudaMemcpy(c, hc.data(), extent * 4, cudaMemcpyHostToDevice));
  if (rank == ranks - 1) skew<<<1, 32, 0, streams[0]>>>();
  check(cudaGetLastError());
  api(ucclLiteLaunchGraph(session, graph, streams[0]));
  api(ucclLiteEnqueue(session, eager, b + guard, c + guard, streams[1]));
  api(ucclLiteSynchronize(session));
  for (size_t i = 0; i < count; ++i) {
    hb[guard + i] = ranks * (ranks + 1) / 2;
    hc[guard + i] = ranks * hb[guard + i];
  }
  for (auto pair : {std::make_pair(a, &ha), {b, &hb}, {c, &hc}}) {
    std::vector<float> got(extent);
    check(
        cudaMemcpy(got.data(), pair.first, extent * 4, cudaMemcpyDeviceToHost));
    if (got != *pair.second)
      throw std::runtime_error("graph chain/guard oracle");
  }
  rejected(ucclLiteReleaseGraph(session, rank == ranks - 1 ? eager : graph));
  api(ucclLiteReleaseGraph(session, graph));
  rejected(ucclLiteLaunchGraph(session, graph, streams[0]));
  size_t size = 0;
  api(ucclLitePlanReport(session, graph, nullptr, &size));
  check(cudaFree(a));
  check(cudaFree(b));
  check(cudaFree(c));
}
template <class Bits>
void run(int rank, int ranks, int localRank, int localRanks,
         char const* library, bool fallback, bool hidden) {
  int visible;
  check(cudaGetDeviceCount(&visible));
  check(cudaSetDevice(visible == 1 ? 0 : localRanks - 1 - localRank));
  ncclUniqueId nativeId;
  if (!rank) native(ncclGetUniqueId(&nativeId));
  MPI_Bcast(&nativeId, sizeof(nativeId), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  native(ncclCommInitRank(&comm, ranks, nativeId, rank));
  std::array<cudaStream_t, 2> streams;
  for (auto& s : streams)
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  ucclLiteSession prior = 0;
  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    ucclLiteUniqueId id;
    if (!rank) api(ucclLiteGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ucclLiteOptions options;
    api(ucclLiteOptionsInit(&options));
    options.capacityBytes = 1u << 20;
    options.nativeLibrary = library;
    options.configurationIdentity = "prepared-C-ABI-check;default-blocking";
    options.dtype = sizeof(Bits) == 4 ? ncclFloat32 : ncclFloat16;
    if (fallback) options.flags |= UCCL_LITE_FALLBACK_ONLY;
    ucclLiteSession session = 99;
    ++options.version;
    rejected(ucclLiteAttach(comm, &id, &options, &session));
    if (session) throw std::runtime_error("failed attach published a session");
    --options.version;
    api(ucclLiteAttach(comm, &id, &options, &session));
    if (!session || session == prior)
      throw std::runtime_error("reused session");
    ucclLiteSession duplicate;
    rejected(ucclLiteAttach(comm, &id, &options, &duplicate));
    rejected(ucclLiteSynchronize(prior));
    for (size_t count : {size_t(64), size_t(257), size_t(4096)})
      for (int op = 1; op <= 3; ++op)
        for (int layout = 0; layout < 2; ++layout) {
          std::vector<ucclLiteRankLayout> layouts;
          for (int r = 0; r < ranks; ++r)
            layouts.push_back({unsigned(layout && r % 2 == 0), sizeof(Bits)});
          ucclLiteRequest req{ucclLiteOperation(op), count, layouts.data(),
                              size_t(ranks), 0};
          ucclLitePlan plan, cached;
          api(ucclLitePrepare(session, &req, &plan));
          api(ucclLitePrepare(session, &req, &cached));
          if (plan != cached) throw std::runtime_error("cache mismatch");
          auto evidence = report(session, plan);
          if ((fallback || (hidden && localRanks == ranks) ||
               sizeof(Bits) == 2) &&
              (evidence.find("\"backend\":0") == std::string::npos ||
               evidence.find("\"accelerated_resources\":false") ==
                   std::string::npos))
            throw std::runtime_error(
                "native-only path initialized acceleration");
          if (!rank) std::cout << "REPORT " << evidence << '\n';
          if (count == 64 && op == 1 && layout == 0) {
            rejected(ucclLitePrepare(
                session, rank == ranks - 1 ? nullptr : &req, &cached));
            if (cached)
              throw std::runtime_error("failed prepare published handle");
            api(ucclLitePrepare(session, &req, &cached));
            if (cached != plan)
              throw std::runtime_error("recovery lost old plan");
          }
          scalar_test::exercise<Bits>(
              op, count, layouts[rank].inPlace, sizeof(Bits), rank, ranks,
              streams,
              [&](int, void const* a, void* b, size_t, cudaStream_t s) {
                api(ucclLiteEnqueue(session, plan, a, b, s));
                api(ucclLiteSynchronize(session));
              });
        }
    if constexpr (sizeof(Bits) == 4) graphs(session, rank, ranks, streams);
    api(ucclLiteDetach(session));
    rejected(ucclLiteDetach(session));
    prior = session;
    scalar_test::exercise<Bits>(
        3, 64, false, sizeof(Bits), rank, ranks, streams,
        [&](int, void const* a, void* b, size_t n, cudaStream_t s) {
          native(ncclAllReduce(a, b, n, options.dtype, ncclSum, comm, s));
        });
  }
  for (auto s : streams) check(cudaStreamDestroy(s));
  native(ncclCommDestroy(comm));
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
    if (argc != 4)
      throw std::runtime_error(
          "usage: check NATIVE_LIBRARY automatic|fallback|hidden fp32|fp16");
    std::string mode = argv[2], type = argv[3];
    if (mode != "automatic" && mode != "fallback" && mode != "hidden")
      throw std::runtime_error("invalid mode");
    if (type == "fp32")
      run<uint32_t>(rank, ranks, localRank, localRanks, argv[1],
                    mode == "fallback", mode == "hidden");
    else if (type == "fp16")
      run<uint16_t>(rank, ranks, localRank, localRanks, argv[1],
                    mode == "fallback", mode == "hidden");
    else
      throw std::runtime_error("invalid dtype");
    if (!rank)
      std::cout << "PASS: native communicator C API " << mode << ' ' << type
                << " ranks=" << ranks << "; full guards, aliases"
                << (type == "fp32" ? ", graph/eager ordering" : "")
                << ", rejection recovery, two lifetimes, caller ownership\n";
  } catch (std::exception const& e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

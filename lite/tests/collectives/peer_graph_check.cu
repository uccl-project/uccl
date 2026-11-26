#include "uccl_lite/collectives/peer_staged.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
using namespace uccl_lite;
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
template <class Bits>
Bits bits(float value) {
  if constexpr (sizeof(Bits) == 4) {
    Bits result;
    std::memcpy(&result, &value, 4);
    return result;
  } else
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 1000000) {
  }
}
template <class Bits>
void run(int rank, int ranks, int iterations, std::vector<int> const& parents) {
  constexpr size_t count = 257;
  size_t offset = 17 + rank, extent = count * ranks + 128;
  Bits *a, *b, *c, *d;
  check(cudaMalloc(&a, extent * sizeof(Bits)));
  check(cudaMalloc(&b, extent * sizeof(Bits)));
  check(cudaMalloc(&c, extent * sizeof(Bits)));
  check(cudaMalloc(&d, extent * sizeof(Bits)));
  std::array<cudaStream_t, 2> streams;
  for (auto& stream : streams)
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    NetworkId id;
    if (!rank) id = PeerStagedCollectives::createUniqueId();
    MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    PeerStagedCollectives peer(
        id, rank, rank, ranks, count * ranks * sizeof(Bits),
        sizeof(Bits) == 4 ? ScalarType::Float32 : ScalarType::BFloat16,
        parents);
    peer.prepare();
    auto chain = peer.prepareGraph(
        {{NetworkOperation::AllGather, b + offset + rank * count, b + offset,
          count},
         {NetworkOperation::ReduceScatter, b + offset, c + offset, count},
         {NetworkOperation::AllReduce, c + offset, c + offset, count}});
    auto ar = peer.prepareGraph(
        {{NetworkOperation::AllReduce, a + offset, d + offset, count}});
    auto rs = peer.prepareGraph(
        {{NetworkOperation::ReduceScatter, b + offset, c + offset, count}});
    auto alias = peer.prepareGraph(
        {{NetworkOperation::AllGather, a + offset, b + offset, count},
         {NetworkOperation::ReduceScatter, b + offset,
          b + offset + rank * count, count}});
    bool rejected = false;
    try {
      peer.prepareGraph({{NetworkOperation::AllReduce, a + offset, d + offset,
                          count + (rank == ranks - 1)}});
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("different rank graph programs accepted");
    auto released = peer.prepareGraph(
        {{NetworkOperation::AllReduce, a + offset, d + offset, count}});
    peer.releaseGraph(released);
    rejected = false;
    try {
      peer.launchGraph(released, streams[0]);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("released graph remained executable");
    cudaGraph_t external;
    check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
    rejected = false;
    try {
      peer.launchGraph(ar, streams[0]);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    check(cudaStreamEndCapture(streams[0], &external));
    check(cudaGraphDestroy(external));
    if (!rejected)
      throw std::runtime_error("external capture modified owned graph state");
    for (int iteration = 0; iteration < iterations; ++iteration) {
      int generation = iteration % 3, mode = iteration % 4;
      Bits guard = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
      std::vector<Bits> ha(extent, guard), hb(extent, guard), hc(extent, guard),
          hd(extent, guard);
      for (size_t i = 0; i < count; ++i) {
        ha[offset + i] = bits<Bits>(float(rank + 1 + i % 7 + generation));
        if (mode == 0) hb[offset + rank * count + i] = ha[offset + i];
      }
      check(cudaMemcpy(a, ha.data(), extent * sizeof(Bits),
                       cudaMemcpyHostToDevice));
      check(cudaMemcpy(b, hb.data(), extent * sizeof(Bits),
                       cudaMemcpyHostToDevice));
      check(cudaMemcpy(c, hc.data(), extent * sizeof(Bits),
                       cudaMemcpyHostToDevice));
      check(cudaMemcpy(d, hd.data(), extent * sizeof(Bits),
                       cudaMemcpyHostToDevice));
      if (rank == ranks - 1) skew<<<1, 32, 0, streams[0]>>>();
      check(cudaGetLastError());
      if (mode == 0) peer.launchGraph(chain, streams[0]);
      if (mode == 1) {
        peer.launchGraph(ar, streams[0]);
        // The same executable must remain ordered when replayed on another
        // stream before its previous launch completes.
        peer.launchGraph(ar, streams[1]);
        peer.launchGraph(ar, streams[0]);
        peer.enqueue(NetworkOperation::AllReduce, d + offset, c + offset, count,
                     streams[1]);
      }
      if (mode == 2) {
        peer.enqueue(NetworkOperation::AllGather, a + offset, b + offset, count,
                     streams[0]);
        peer.launchGraph(rs, streams[1]);
      }
      if (mode == 3) peer.launchGraph(alias, streams[0]);
      peer.synchronize();
      for (size_t i = 0; i < count * ranks; ++i)
        if (mode != 1)
          hb[offset + i] =
              bits<Bits>(float(i / count + 1 + (i % count) % 7 + generation));
      for (size_t i = 0; i < count; ++i) {
        float sum =
            float(ranks * (ranks + 1) / 2 + ranks * (i % 7 + generation));
        if (mode == 0 || mode == 1) hc[offset + i] = bits<Bits>(ranks * sum);
        if (mode == 1) hd[offset + i] = bits<Bits>(sum);
        if (mode == 2)
          hc[offset + i] =
              bits<Bits>(float(ranks * (rank + 1 + i % 7 + generation)));
        if (mode == 3)
          hb[offset + rank * count + i] =
              bits<Bits>(float(ranks * (rank + 1 + i % 7 + generation)));
      }
      std::vector<Bits> actual(extent);
      for (auto pair : {std::pair<Bits*, std::vector<Bits> const*>{a, &ha},
                        {b, &hb},
                        {c, &hc},
                        {d, &hd}}) {
        check(cudaMemcpy(actual.data(), pair.first, extent * sizeof(Bits),
                         cudaMemcpyDeviceToHost));
        if (actual != *pair.second)
          throw std::runtime_error(
              "graph replay/alias/producer-consumer/full-guard failure");
      }
    }
    for (auto graph : {chain, ar, rs, alias}) peer.releaseGraph(graph);
  }
  for (auto stream : streams) check(cudaStreamDestroy(stream));
  check(cudaFree(a));
  check(cudaFree(b));
  check(cudaFree(c));
  check(cudaFree(d));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 3 && argc != 4)
      throw std::invalid_argument("bf16|fp32 ITERATIONS [path|star]");
    std::vector<int> parents;
    if (argc == 4) {
      std::string mode = argv[3];
      if (mode != "path" && mode != "star")
        throw std::invalid_argument("invalid tree mode");
      parents.resize(ranks, 0);
      parents[0] = -1;
      if (mode == "path")
        for (int r = 1; r < ranks; ++r) parents[r] = r - 1;
    }
    int iterations = std::stoi(argv[2]);
    if (iterations < 4)
      throw std::invalid_argument("at least four iterations required");
    check(cudaSetDevice(rank));
    if (std::string(argv[1]) == "fp32")
      run<uint32_t>(rank, ranks, iterations, parents);
    else if (std::string(argv[1]) == "bf16")
      run<uint16_t>(rank, ranks, iterations, parents);
    else
      throw std::invalid_argument("unsupported dtype");
    if (!rank)
      std::cout << "PASS: owned peer graphs " << argv[1] << " ranks=" << ranks
                << " iterations=" << iterations
                << " route=" << (argc == 4 ? argv[3] : "direct")
                << " x two lifetimes; graph/eager ordering, aliases, metadata "
                   "rejection and full guards\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

#include "uccl_lite/collectives/peer_staged.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
using namespace uccl_lite;
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
template <class Bits>
Bits encode(float x) {
  if constexpr (sizeof(Bits) == 4) {
    Bits bits;
    std::memcpy(&bits, &x, 4);
    return bits;
  } else
    return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
template <class Bits>
float decode(Bits x) {
  if constexpr (sizeof(Bits) == 4) {
    float value;
    std::memcpy(&value, &x, 4);
    return value;
  } else
    return __bfloat162float(__ushort_as_bfloat16(x));
}
template <class Bits>
Bits value(size_t i, int rank, int generation, bool gather) {
  if (gather)
    return Bits(uint32_t(i) * 2246822519u + rank * 7919u + generation);
  float x;
  if (i % 11 == 0 && rank < 4) {
    float cancellation[4] = {16777216.0f, float(1 + generation), -16777216.0f,
                             3.0f};
    x = cancellation[rank];
  } else
    x = float((int(i % 17) - 8) * (rank + 1)) / float(rank + 3) +
        generation * .125f;
  return encode<Bits>(x);
}
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 2000000) {
  }
}
template <class Bits>
void exercise(PeerStagedCollectives& peer, int rank, int ranks, int op,
              size_t count, bool alias, std::array<cudaStream_t, 2> streams) {
  size_t in = count * (op == 2 ? ranks : 1),
         out = count * (op == 1 ? ranks : 1), extent = std::max(in, out) + 128;
  size_t aOffset = 1 + (alias && op == 1 ? size_t(rank) * count : 0),
         bOffset = 1 + (alias && op == 2 ? size_t(rank) * count : 0);
  bool inputB = alias && op == 1, outputA = alias && op != 1;
  Bits *a, *b;
  check(cudaMalloc(&a, extent * sizeof(Bits)));
  check(cudaMalloc(&b, extent * sizeof(Bits)));
  auto* input = (inputB ? b : a) + aOffset;
  auto* output = (outputA ? a : b) + bOffset;
  auto graph =
      peer.prepareGraph({{NetworkOperation(op), input, output, count}});
  for (int generation = 0; generation < 3; ++generation) {
    Bits guard = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
    std::vector<Bits> ha(extent, guard), hb(extent, guard), actual(extent);
    for (size_t i = 0; i < in; ++i)
      (inputB ? hb : ha)[aOffset + i] =
          value<Bits>(i, rank, generation, op == 1);
    check(cudaMemcpy(a, ha.data(), extent * sizeof(Bits),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, hb.data(), extent * sizeof(Bits),
                     cudaMemcpyHostToDevice));
    auto stream = streams[generation % 2];
    if (rank == ranks - 1) skew<<<1, 32, 0, stream>>>();
    check(cudaGetLastError());
    if (generation == 1)
      peer.launchGraph(graph, stream);
    else
      peer.enqueue(NetworkOperation(op), input, output, count, stream);
    peer.synchronize();
    for (size_t i = 0; i < out; ++i) {
      Bits wanted;
      if (op == 1)
        wanted = value<Bits>(i % count, int(i / count), generation, true);
      else {
        // Independent source-ordered FP32 reference. Cancellation inputs
        // distinguish this contract from a reordered partial-sum tree.
        float sum = 0;
        for (int source = 0; source < ranks; ++source)
          sum += decode(value<Bits>(op == 2 ? size_t(rank) * count + i : i,
                                    source, generation, false));
        wanted = encode<Bits>(sum);
      }
      (outputA ? ha : hb)[bOffset + i] = wanted;
    }
    for (auto pair : {std::make_pair(a, &ha), {b, &hb}}) {
      check(cudaMemcpy(actual.data(), pair.first, extent * sizeof(Bits),
                       cudaMemcpyDeviceToHost));
      if (actual != *pair.second) {
        for (size_t i = 0; i < extent; ++i)
          if (actual[i] != (*pair.second)[i]) {
            std::cerr << "tree mismatch rank=" << rank << " op=" << op
                      << " count=" << count << " alias=" << alias
                      << " generation=" << generation
                      << " buffer=" << (pair.first == a ? "a" : "b")
                      << " index=" << i << " got=" << uint64_t(actual[i])
                      << " expected=" << uint64_t((*pair.second)[i]) << '\n';
            break;
          }
        throw std::runtime_error(
            "tree full-buffer/source-order/input/guard oracle mismatch");
      }
    }
  }
  peer.releaseGraph(graph);
  check(cudaFree(a));
  check(cudaFree(b));
}
template <class Bits>
void run(int rank, int ranks, size_t capacity, std::string route) {
  std::array<cudaStream_t, 2> streams;
  for (auto& stream : streams)
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::vector<int> parents(ranks, 0);
  parents[0] = -1;
  if (route == "path")
    for (int r = 1; r < ranks; ++r) parents[r] = r - 1;
  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    NetworkId id;
    if (!rank) id = PeerStagedCollectives::createUniqueId();
    MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    {
      PeerStagedCollectives bad(
          id, rank, rank, ranks,
          capacity + (rank == ranks - 1 ? sizeof(Bits) : 0),
          sizeof(Bits) == 4 ? ScalarType::Float32 : ScalarType::BFloat16,
          parents);
      bool rejected = false;
      try {
        bad.prepare();
      } catch (std::exception const&) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("inconsistent tree capacity accepted");
    }
    if (!rank) id = PeerStagedCollectives::createUniqueId();
    MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    PeerStagedCollectives peer(
        id, rank, rank, ranks, capacity,
        sizeof(Bits) == 4 ? ScalarType::Float32 : ScalarType::BFloat16,
        parents);
    peer.prepare();
    for (size_t count : {size_t(1), size_t(257), size_t(4096),
                         capacity / (ranks * sizeof(Bits))})
      for (int op = 1; op <= 3; ++op)
        for (bool alias : {false, true})
          exercise<Bits>(peer, rank, ranks, op, count, alias, streams);
    exercise<Bits>(peer, rank, ranks, 3, capacity / sizeof(Bits), true,
                   streams);
  }
  for (auto stream : streams) check(cudaStreamDestroy(stream));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 4 || ranks < 2 || ranks > 32)
      throw std::invalid_argument(
          "bf16|fp32 path|star MAX_BYTES; 2..32 local ranks");
    std::string type = argv[1], route = argv[2];
    size_t capacity = std::stoull(argv[3]);
    if ((route != "path" && route != "star") || capacity < (65536UL * ranks) ||
        capacity > (1UL << 30) || capacity % (4 * ranks))
      throw std::invalid_argument("invalid route/capacity");
    check(cudaSetDevice(rank));
    if (type == "fp32")
      run<uint32_t>(rank, ranks, capacity, route);
    else if (type == "bf16")
      run<uint16_t>(rank, ranks, capacity, route);
    else
      throw std::invalid_argument("invalid dtype");
    if (!rank)
      std::cout << "PASS: tree " << route << ' ' << type << " ranks=" << ranks
                << " capacity=" << capacity
                << "; exact source order, bitwise gather, full buffers/guards, "
                   "aliases, scalar tails, graph/eager, two lifetimes\n";
  } catch (std::exception const& e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

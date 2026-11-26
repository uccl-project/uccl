#include "uccl_lite/collectives.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
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
  // Every partial sum is integral and representable for all supported ranks,
  // allowing an independent bitwise oracle for both native and direct paths.
  float x = float(int((i + rank) % 9) - 4 + generation);
  return encode<Bits>(x);
}
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 2000000) {
  }
}
template <class Bits, class Backend>
void exercise(Backend& peer, int rank, int ranks, int op, size_t count,
              bool alias, std::array<cudaStream_t, 2> streams) {
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
  for (int generation = 0; generation < 3; ++generation) {
    Bits guard = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
    std::vector<Bits> ha(extent, guard), hb(extent, guard), actual(extent);
    for (size_t i = 0; i < in; ++i)
      (inputB ? hb : ha)[aOffset + i] =
          value<Bits>(i, rank, generation, op == 1);
    // Pageable synchronous H2D may return before DMA completion. Produce on
    // the consumer stream so nonblocking streams have an explicit dependency.
    auto stream = streams[generation % 2];
    check(cudaMemcpyAsync(a, ha.data(), extent * sizeof(Bits),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(b, hb.data(), extent * sizeof(Bits),
                          cudaMemcpyHostToDevice, stream));
    if (rank == ranks - 1) skew<<<1, 32, 0, stream>>>();
    check(cudaGetLastError());
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
            std::cerr << "direct exchange mismatch rank=" << rank
                      << " op=" << op << " count=" << count
                      << " alias=" << alias << " generation=" << generation
                      << " buffer=" << (pair.first == a ? "a" : "b")
                      << " index=" << i << " got=" << uint64_t(actual[i])
                      << " expected=" << uint64_t((*pair.second)[i]) << '\n';
            break;
          }
        throw std::runtime_error(
            "direct exchange full-buffer/source-order/input/guard oracle "
            "mismatch");
      }
    }
  }
  check(cudaFree(a));
  check(cudaFree(b));
}

void ncclCheck(ncclResult_t r) {
  if (r != ncclSuccess) throw std::runtime_error(ncclGetLastError(nullptr));
}
struct Api {
  ncclComm_t comm;
  ncclDataType_t dtype;
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) {
    if (op == NetworkOperation::AllGather)
      ncclCheck(ncclAllGather(input, output, count, dtype, comm, stream));
    else if (op == NetworkOperation::ReduceScatter)
      ncclCheck(ncclReduceScatter(input, output, count, dtype, ncclSum, comm,
                                  stream));
    else
      ncclCheck(
          ncclAllReduce(input, output, count, dtype, ncclSum, comm, stream));
  }
  void synchronize() { check(cudaDeviceSynchronize()); }
};
void noNative() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    auto slash = line.rfind('/');
    if (slash != std::string::npos &&
        line.substr(slash + 1).rfind("libnccl.so", 0) == 0)
      throw std::runtime_error("native NCCL image loaded");
  }
}
template <class Bits>
void run(int rank, int ranks, std::array<cudaStream_t, 2> streams, bool reject,
         bool tiny) {
  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    ncclUniqueId id;
    ncclComm_t comm = nullptr;
    if (!rank) ncclCheck(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    auto status = ncclCommInitRank(&comm, ranks, id, rank);
    if (reject) {
      if (status == ncclSuccess || comm)
        throw std::runtime_error("bad strict profile was accepted");
      continue;
    }
    ncclCheck(status);
    noNative();
    Api api{comm, sizeof(Bits) == 4 ? ncclFloat32 : ncclBfloat16};
    for (size_t count : (tiny ? std::vector<size_t>{1}
                              : std::vector<size_t>{1, 64, 257, 4096}))
      for (int op = 1; op <= 3; ++op)
        exercise<Bits>(api, rank, ranks, op, count, bool(rank & 1), streams);
    Bits *a, *b;
    check(cudaMalloc(&a, 128 * sizeof(Bits)));
    check(cudaMalloc(&b, 128 * sizeof(Bits)));
    if (ncclAllReduce(a, b, 13, api.dtype, ncclSum, comm, streams[0]) !=
        ncclInvalidUsage)
      throw std::runtime_error("undeclared strict shape accepted");
    cudaGraph_t graph;
    check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
    if (ncclAllReduce(a, b, 1, api.dtype, ncclSum, comm, streams[0]) !=
        ncclInvalidUsage)
      throw std::runtime_error("unqualified strict capture accepted");
    check(cudaStreamEndCapture(streams[0], &graph));
    check(cudaGraphDestroy(graph));
    check(cudaFree(a));
    check(cudaFree(b));
    ncclCheck(ncclCommDestroy(comm));
    noNative();
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
    if (argc != 3) throw std::invalid_argument("fp32|bf16 run|reject|tiny");
    auto configured = std::getenv("UCCL_LITE_TEST_DEVICE");
    check(cudaSetDevice(configured ? std::stoi(configured)
                                   : localRanks - 1 - localRank));
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (std::string(argv[1]) == "fp32")
      run<uint32_t>(rank, ranks, streams, std::string(argv[2]) == "reject",
                    std::string(argv[2]) == "tiny");
    else
      run<uint16_t>(rank, ranks, streams, std::string(argv[2]) == "reject",
                    std::string(argv[2]) == "tiny");
    for (auto stream : streams) check(cudaStreamDestroy(stream));
    if (!rank)
      std::cout << "PASS strict distributed " << argv[1] << ' ' << argv[2]
                << " ranks=" << ranks
                << "; full buffers/guards, mixed aliases, tails, two "
                   "lifetimes, no native image\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

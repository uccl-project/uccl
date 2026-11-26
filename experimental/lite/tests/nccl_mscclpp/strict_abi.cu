// This client links only NCCL/MPI/CUDA. Exercise the interposer without its
// API.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

#ifdef UCCL_LITE_ABI_LOCAL8
using Element = __nv_bfloat16;
constexpr auto kType = ncclBfloat16;
constexpr int kLocalRanks = 8;
#else
using Element = float;
constexpr auto kType = ncclFloat;
constexpr int kLocalRanks = 4;
#endif

#ifdef UCCL_LITE_ABI_LOCAL
constexpr int kRanks = 4;
#else
constexpr int kRanks = 8;
#endif

void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
float value(size_t index, int rank, int generation) {
  return float(int(index % 17) - 8) * (rank + 1) + generation;
}
void call(int op, Element const* input, Element* output, size_t count,
          ncclComm_t comm, cudaStream_t stream) {
  if (op == 1)
    check(ncclAllGather(input, output, count, kType, comm, stream));
  else if (op == 2)
    check(
        ncclReduceScatter(input, output, count, kType, ncclSum, comm, stream));
  else
    check(ncclAllReduce(input, output, count, kType, ncclSum, comm, stream));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local);
  int localRank, localRanks;
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localRanks);
  try {
    if (world != kRanks || localRanks != kLocalRanks)
      throw std::runtime_error("requires the configured number of ranks");
    int device = std::getenv("UCCL_LITE_ABI_REVERSE_DEVICES")
                     ? kLocalRanks - 1 - localRank
                     : localRank;
    check(cudaSetDevice(device));
    cudaStream_t streams[2];
    for (auto& s : streams)
      check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
#ifdef UCCL_LITE_ABI_MEDIUM128
    size_t maxElements = (128 << 20) / sizeof(Element);
#elif defined(UCCL_LITE_ABI_MEDIUM64)
    size_t maxElements = (64 << 20) / sizeof(Element);
#elif defined(UCCL_LITE_ABI_EXTENDED)
    size_t maxElements = (32 << 20) / sizeof(Element);
#else
    size_t maxElements = (16 << 20) / sizeof(Element);
#endif
    size_t guard = 33 + rank;
    Element *a, *b;
    check(cudaMalloc(&a, (maxElements + 128) * sizeof(Element)));
    check(cudaMalloc(&b, (maxElements + 128) * sizeof(Element)));
    int calls = 0;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      ncclUniqueId id;
      if (!rank) check(ncclGetUniqueId(&id));
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
      ncclComm_t comm;
      check(ncclGroupStart());
      check(ncclCommInitRank(&comm, world, id, rank));
      check(ncclGroupEnd());
      // Saved empty graphs must not disable later eager routing.
      cudaGraph_t empty;
      cudaGraphExec_t emptyExec;
      check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      for (int op = 1; op <= 3; ++op) call(op, a, b, 0, comm, streams[0]);
      check(cudaStreamEndCapture(streams[0], &empty));
      check(cudaGraphInstantiate(&emptyExec, empty, nullptr, nullptr, 0));
      for (int generation = 0; generation < 3; ++generation) {
        // Alternate small double-buffered calls with large single-bank rings.
        for (size_t bytes : {1024UL, 16777216UL, 16384UL, 1048576UL, 262144UL,
#ifdef UCCL_LITE_ABI_EXTENDED
                             33554432UL,
#endif
#ifdef UCCL_LITE_ABI_MEDIUM64
                             67108864UL,
#endif
#ifdef UCCL_LITE_ABI_MEDIUM128
                             134217728UL,
#endif
#ifdef UCCL_LITE_ABI_BOUNDARY512
                             524288UL,
#endif
#ifdef UCCL_LITE_ABI_SERIAL_PACKET
                             32768UL, 65536UL, 131072UL,
#endif
                             8388608UL}) {
          for (int op = 1; op <= 3; ++op) {
            size_t count = bytes / (sizeof(Element) * (op == 3 ? 1 : kRanks));
            size_t ni = count * (op == 2 ? kRanks : 1);
            size_t no = count * (op == 1 ? kRanks : 1);
            bool inplace = (rank + generation) % 2;
            size_t sourceOffset = inplace && op == 1 ? rank * count : 0;
            size_t targetOffset = inplace && op == 2 ? rank * count : 0;
            std::vector<Element> input(maxElements + 128, -12345.f);
            std::vector<Element> output(input.size(), -12345.f);
            for (size_t i = 0; i < ni; ++i)
              input[guard + sourceOffset + i] = value(i, rank, generation);
            auto stream = streams[calls % 2];
            Element* inputAllocation = a;
            Element* otherAllocation = b;
#ifdef UCCL_LITE_ABI_EXTENDED
            // Graphs cached by extent must not keep the first client address.
            if (generation & 1) {
              inputAllocation = b;
              otherAllocation = a;
            }
#endif
            check(cudaMemcpyAsync(inputAllocation, input.data(),
                                  input.size() * sizeof(Element),
                                  cudaMemcpyHostToDevice, stream));
            if (!inplace)
              check(cudaMemcpyAsync(otherAllocation, output.data(),
                                    output.size() * sizeof(Element),
                                    cudaMemcpyHostToDevice, stream));
            Element* outputBase = inplace ? inputAllocation : otherAllocation;
            if (rank == 0)
              std::fprintf(stderr,
                           "strict ABI begin lifetime=%d generation=%d "
                           "bytes=%zu op=%d\n",
                           lifetime, generation, bytes, op);
#ifdef UCCL_LITE_ABI_EXTENDED
            if (rank == 3 && generation == 1 &&
                (bytes == 16777216UL || bytes == 33554432UL) && op == 2)
              usleep(10000);
#endif
#ifdef UCCL_LITE_ABI_SERIAL_PACKET
            if (rank == 3 && generation == 1 && bytes == 131072UL && op == 1)
              usleep(10000);
#endif
#ifdef UCCL_LITE_ABI_MEDIUM64
            if (rank == 3 && generation == 1 && bytes == 67108864UL && op == 2)
              usleep(10000);
#endif
#ifdef UCCL_LITE_ABI_MEDIUM128
            if (rank == 3 && generation == 1 && bytes == 134217728UL && op == 2)
              usleep(10000);
#endif
            call(op, inputAllocation + guard + sourceOffset,
                 outputBase + guard + targetOffset, count, comm, stream);
            check(cudaMemcpyAsync(output.data(), outputBase,
                                  output.size() * sizeof(Element),
                                  cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            for (size_t i = 0; i < output.size(); ++i) {
              float expected =
                  inplace ? float(input[i]) : float(Element(-12345.f));
              if (i >= guard + targetOffset && i < guard + targetOffset + no) {
                size_t index = i - guard - targetOffset;
                if (op == 1)
                  expected = value(index % count, index / count, generation);
                else {
                  expected = 0;
                  for (int peer = 0; peer < kRanks; ++peer)
                    expected += value(op == 2 ? rank * count + index : index,
                                      peer, generation);
                }
              }
              if (float(output[i]) != float(Element(expected))) {
                std::fprintf(
                    stderr,
                    "mismatch rank=%d lifetime=%d generation=%d bytes=%zu "
                    "op=%d index=%zu guard=%zu sourceOffset=%zu "
                    "targetOffset=%zu got=%g expected=%g\n",
                    rank, lifetime, generation, bytes, op, i, guard,
                    sourceOffset, targetOffset, double(float(output[i])),
                    double(float(Element(expected))));
                throw std::runtime_error("full-buffer/guard mismatch");
              }
            }
            ++calls;
          }
        }
#ifdef UCCL_LITE_ABI_EXTENDED
        // The graph cache must tolerate allocation recycling, not just two
        // long-lived addresses. Every preceding call is complete above.
        Element** recycled = (generation & 1) ? &b : &a;
        check(cudaFree(*recycled));
        check(cudaMalloc(recycled, (maxElements + 128) * sizeof(Element)));
#endif
      }
      check(cudaGraphLaunch(emptyExec, streams[1]));
      check(cudaStreamSynchronize(streams[1]));
      check(cudaGraphExecDestroy(emptyExec));
      check(cudaGraphDestroy(empty));
      // Unsupported requests must return errors, never execute NCCL.
      check(ncclGroupStart());
      if (ncclAllReduce(a + guard, b + guard, 1024 / sizeof(Element), kType,
                        ncclSum, comm, streams[0]) != ncclInvalidUsage)
        throw std::runtime_error("grouped call did not reject");
      check(ncclGroupEnd());
      if (ncclAllReduce(a + guard, b + guard, 1024 / sizeof(Element), kType,
                        ncclProd, comm, streams[0]) != ncclInvalidUsage)
        throw std::runtime_error("unsupported reduction did not reject");
      if (ncclAllReduce(a + guard, b + guard, 1, kType, ncclSum, comm,
                        streams[0]) != ncclInvalidUsage)
        throw std::runtime_error("unsupported extent did not reject");
      cudaGraph_t graph;
      check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      if (ncclAllReduce(a + guard, b + guard, 1024 / sizeof(Element), kType,
                        ncclSum, comm, streams[0]) != ncclInvalidUsage)
        throw std::runtime_error("capture did not reject");
      check(cudaStreamEndCapture(streams[0], &graph));
      check(cudaGraphDestroy(graph));
      check(ncclCommFinalize(comm));
      if (ncclAllReduce(a + guard, b + guard, 1024 / sizeof(Element), kType,
                        ncclSum, comm, streams[0]) != ncclInvalidArgument)
        throw std::runtime_error(
            "finalized handle did not return ncclInvalidArgument");
      check(ncclCommDestroy(comm));
    }
    check(cudaFree(a));
    check(cudaFree(b));
    for (auto s : streams) check(cudaStreamDestroy(s));
    if (!rank)
      std::printf(
          "Strict ABI full-buffer checks passed: %d calls/rank, two "
          "lifetimes\n",
          calls);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

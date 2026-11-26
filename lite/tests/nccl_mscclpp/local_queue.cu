// NCCL ABI client: queue different collectives on alternating and recycled
// streams without synchronizing each call. Every output and guard is checked.
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#ifdef UCCL_LITE_LOCAL_MIXED_ALLOC_QUEUE
#include <cuda.h>
#endif
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

void checked(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void checked(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
float value(size_t i, int rank, int lifetime) {
  return float(int(i % 17) - 8) * (rank + 1) + lifetime;
}
int queuedOperation(int i) {
#ifdef UCCL_LITE_LOCAL_AG_GRAPH_QUEUE
  return i < 24 ? i % 3 : 0;
#elif defined(UCCL_LITE_LOCAL_HIER_QUEUE)
  return i < 6 ? i % 3 : 1;
#elif defined(UCCL_LITE_LOCAL_SM_QUEUE)
  return i < 12 ? i % 3 : 2;
#elif defined(UCCL_LITE_LOCAL_RING_QUEUE)
  return i % 3;
#else
  return i < 50 ? 0 : i < 100 ? 1 : i >= 200 ? 2 : i % 3;
#endif
}
size_t queuedCount(int i) {
#ifdef UCCL_LITE_LOCAL_AG_GRAPH_QUEUE
#ifdef UCCL_LITE_LOCAL_AG_GRAPH_RANGE_QUEUE
  constexpr size_t counts[] = {1024, 2048, 4096, 8192, 16384};
  return counts[(i / 6) % 5];
#else
  constexpr size_t counts[] = {1024, 4096, 8192, 16384};
  return counts[(i / 6) % 4];
#endif
#elif defined(UCCL_LITE_LOCAL_HIER_QUEUE)
  constexpr size_t counts[] = {256, 8UL << 20, 16UL << 20, 16UL << 20};
  return counts[i / 3];
#elif defined(UCCL_LITE_LOCAL_SM_QUEUE)
  constexpr size_t counts[] = {256, 8192, 32768, 524288};
  return counts[(i / 6) % 4];
#elif defined(UCCL_LITE_LOCAL_RING_QUEUE)
#ifdef UCCL_LITE_LOCAL_RING4_QUEUE
  return (i / 3) % 3 == 2 ? (4UL << 20) : (i / 3) % 3 == 1 ? (1UL << 20) : 256;
#else
  return (i / 3) % 3 == 2 ? (4UL << 20) : (i / 3) % 3 == 1 ? (2UL << 20) : 256;
#endif
#else
  return (i / 3) % 3 == 2 ? 65536 : (i / 3) % 3 == 1 ? 4096 : 256;
#endif
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (world != 4) throw std::runtime_error("requires four ranks");
    checked(cudaSetDevice(rank));
#ifdef UCCL_LITE_LOCAL_AG_GRAPH_QUEUE
#ifdef UCCL_LITE_LOCAL_AG_GRAPH_RANGE_QUEUE
    constexpr int calls = 120;
#else
    constexpr int calls = 96;
#endif
    constexpr size_t stride = 4 * 16384 + 64;
#elif defined(UCCL_LITE_LOCAL_HIER_QUEUE)
    constexpr int calls = 12;
    constexpr size_t stride = 4 * (16UL << 20) + 64;
#elif defined(UCCL_LITE_LOCAL_SM_QUEUE)
    constexpr int calls = 48;
    constexpr size_t stride = 4 * 524288 + 64;
#elif defined(UCCL_LITE_LOCAL_RING_QUEUE)
    constexpr int calls = 24;
    constexpr size_t stride = 4 * (4UL << 20) + 64;
#else
    constexpr int calls = 300;
    constexpr size_t stride = 4 * 65536 + 64;
#endif
    float *input, *output;
#ifdef UCCL_LITE_LOCAL_MIXED_ALLOC_QUEUE
    cudaStream_t allocationStream = nullptr;
    if (rank == 3) {
      checked(
          cudaStreamCreateWithFlags(&allocationStream, cudaStreamNonBlocking));
      checked(cudaMallocAsync(&input, calls * stride * 4, allocationStream));
      checked(cudaMallocAsync(&output, calls * stride * 4, allocationStream));
      checked(cudaStreamSynchronize(allocationStream));
    } else {
#endif
      checked(cudaMalloc(&input, calls * stride * 4));
      checked(cudaMalloc(&output, calls * stride * 4));
#ifdef UCCL_LITE_LOCAL_MIXED_ALLOC_QUEUE
    }
    unsigned legacy = 0;
    if (cuPointerGetAttribute(
            &legacy, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
            reinterpret_cast<CUdeviceptr>(input)) != CUDA_SUCCESS ||
        legacy != unsigned(rank != 3))
      throw std::runtime_error("mixed allocation property mismatch");
    std::printf("UCCL Lite mixed allocation: rank=%d legacy=%u\n", rank,
                legacy);
    std::fflush(stdout);
#endif
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      cudaStream_t streams[2];
      for (auto& stream : streams)
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      std::vector<float> hostInput(calls * stride),
          hostOutput(calls * stride, -12345.f);
      for (size_t i = 0; i < calls * stride; ++i)
        hostInput[i] = value(i % stride, rank, lifetime + i / stride);
      checked(cudaMemcpy(input, hostInput.data(), calls * stride * 4,
                         cudaMemcpyHostToDevice));
      checked(cudaMemcpy(output, hostOutput.data(), calls * stride * 4,
                         cudaMemcpyHostToDevice));
      ncclUniqueId id;
      if (!rank) checked(ncclGetUniqueId(&id));
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
      ncclComm_t comm;
      checked(ncclCommInitRank(&comm, world, id, rank));
      for (int i = 0; i < calls; ++i) {
        if (rank == 3 && i % 7 == 0) usleep(1000);
        int which = i % 2, op = queuedOperation(i);
        size_t count = queuedCount(i);
        float* target = output + size_t(i) * stride + 1 + rank;
        if (op == 0)
          checked(ncclAllGather(input + size_t(i) * stride, target, count,
                                ncclFloat, comm, streams[which]));
        else if (op == 1)
          checked(ncclReduceScatter(input + size_t(i) * stride, target, count,
                                    ncclFloat, ncclSum, comm, streams[which]));
        else
          checked(ncclAllReduce(input + size_t(i) * stride, target, count,
                                ncclFloat, ncclSum, comm, streams[which]));
        if (i == 10 || i == 21) {
          // Destruction may return while old GPU work is pending. A new
          // stream can reuse the handle address but must get a distinct ID.
          checked(cudaStreamDestroy(streams[which]));
          checked(cudaStreamCreateWithFlags(&streams[which],
                                            cudaStreamNonBlocking));
        }
      }
      for (auto stream : streams) checked(cudaStreamSynchronize(stream));
      checked(cudaMemcpy(hostOutput.data(), output, calls * stride * 4,
                         cudaMemcpyDeviceToHost));
      for (int i = 0; i < calls; ++i) {
        int op = queuedOperation(i);
        size_t count = queuedCount(i);
        size_t extent = op == 0 ? 4 * count : count;
        for (size_t j = 0; j < stride; ++j) {
          float expected = -12345.f;
          if (j >= 1 + rank && j < 1 + rank + extent) {
            size_t k = j - 1 - rank;
            if (op == 0)
              expected = value(k % count, k / count, lifetime + i);
            else {
              expected = 0;
              for (int r = 0; r < 4; ++r)
                expected +=
                    value(k + (op == 1 ? rank * count : 0), r, lifetime + i);
            }
          }
          if (hostOutput[size_t(i) * stride + j] != expected)
            throw std::runtime_error("queued output/guard mismatch");
        }
      }
      checked(ncclCommFinalize(comm));
      checked(ncclCommDestroy(comm));
      for (auto stream : streams) checked(cudaStreamDestroy(stream));
    }
#ifdef UCCL_LITE_LOCAL_MIXED_ALLOC_QUEUE
    if (rank == 3) {
      checked(cudaFreeAsync(input, allocationStream));
      checked(cudaFreeAsync(output, allocationStream));
      checked(cudaStreamSynchronize(allocationStream));
      checked(cudaStreamDestroy(allocationStream));
    } else {
#endif
      checked(cudaFree(input));
      checked(cudaFree(output));
#ifdef UCCL_LITE_LOCAL_MIXED_ALLOC_QUEUE
    }
#endif
    if (!rank)
      std::printf(
          "Strict ABI full-buffer checks passed: %d queued calls/rank, two "
          "lifetimes\n",
          2 * calls);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

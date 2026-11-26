// Queue changing operations, sizes and layouts across two caller streams.
// Every call owns its buffers; no host synchronization occurs between calls.
#include <cstdio>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
__device__ float value(size_t i, int rank, int generation) {
  return float(int((i + 7 * rank + generation) % 31) - 15) / 32.f;
}
__global__ void fill(__nv_bfloat16* data, size_t count, int rank,
                     int generation, bool sentinel) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    data[i] = __float2bfloat16(sentinel ? -123.f : value(i, rank, generation));
}
__global__ void verify(__nv_bfloat16 const* data, size_t count, size_t shard,
                       int rank, int op, int generation, bool inplace,
                       unsigned* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count + 16;
       i += gridDim.x * blockDim.x) {
    float expected = inplace && op == 2 && rank < 7 && i >= count
                         ? value(rank * shard + i, rank, generation)
                         : -123.f;
    if (i < count) {
      if (op == 1)
        expected = value(i % shard, int(i / shard), generation);
      else {
        expected = 0;
        for (int r = 0; r < 8; ++r)
          expected += value(op == 2 ? rank * shard + i : i, r, generation);
      }
    }
    if (__bfloat162float(data[i]) != expected) atomicAdd(errors, 1u);
  }
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (ranks != 8) throw std::runtime_error("requires eight local ranks");
    check(cudaSetDevice(rank));
    constexpr int calls = 128;
    constexpr size_t stride = (1 << 19) + 32;
    __nv_bfloat16 *a, *b;
    unsigned* errors;
    check(cudaMalloc(&a, calls * stride * 2));
    check(cudaMalloc(&b, calls * stride * 2));
    check(cudaMalloc(&errors, sizeof(unsigned)));
    cudaStream_t streams[2];
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ncclUniqueId id;
    if (!rank) check(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    check(ncclCommInitRank(&comm, ranks, id, rank));
    // Prepare all producers first, keeping input initialization out of the
    // cross-stream serialization being exercised by the ABI itself.
    for (int c = 0; c < calls; ++c) {
      int op = c % 3 + 1;
      size_t bytes = size_t(1024) << (6 * ((c / 3) % 3));
      // 1 KiB, 64 KiB, 1 MiB: include CPU/GPU route transitions.
      if (bytes > (1UL << 20)) bytes = 1UL << 20;
      size_t count = bytes / 2, shard = count / 8;
      bool inplace = (c + rank) % 2;
      auto input = a + c * stride + (inplace && op == 1 ? rank * shard : 0);
      auto output = inplace ? a + c * stride + (op == 2 ? rank * shard : 0)
                            : b + c * stride;
      fill<<<64, 256, 0, streams[0]>>>(output, (op == 2 ? shard : count) + 16,
                                       rank, c, true);
      fill<<<64, 256, 0, streams[0]>>>(input, op == 1 ? shard : count, rank, c,
                                       false);
    }
    check(cudaGetLastError());
    check(cudaStreamSynchronize(streams[0]));
    MPI_Barrier(MPI_COMM_WORLD);
    for (int c = 0; c < calls; ++c) {
      int op = c % 3 + 1;
      size_t bytes = size_t(1024) << (6 * ((c / 3) % 3));
      if (bytes > (1UL << 20)) bytes = 1UL << 20;
      size_t count = bytes / 2, shard = count / 8;
      bool inplace = (c + rank) % 2;
      auto input = a + c * stride + (inplace && op == 1 ? rank * shard : 0);
      auto output = inplace ? a + c * stride + (op == 2 ? rank * shard : 0)
                            : b + c * stride;
      auto stream = streams[c % 2];
      if (rank == 3 && c == 31) usleep(10000);
      if (op == 1)
        check(ncclAllGather(input, output, shard, ncclBfloat16, comm, stream));
      else if (op == 2)
        check(ncclReduceScatter(input, output, shard, ncclBfloat16, ncclSum,
                                comm, stream));
      else
        check(ncclAllReduce(input, output, count, ncclBfloat16, ncclSum, comm,
                            stream));
    }
    for (auto stream : streams) check(cudaStreamSynchronize(stream));
    check(cudaMemsetAsync(errors, 0, sizeof(unsigned), streams[0]));
    for (int c = 0; c < calls; ++c) {
      int op = c % 3 + 1;
      size_t bytes = size_t(1024) << (6 * ((c / 3) % 3));
      if (bytes > (1UL << 20)) bytes = 1UL << 20;
      size_t count = bytes / 2, shard = count / 8;
      bool inplace = (c + rank) % 2;
      auto output = inplace ? a + c * stride + (op == 2 ? rank * shard : 0)
                            : b + c * stride;
      verify<<<64, 256, 0, streams[0]>>>(output, op == 2 ? shard : count, shard,
                                         rank, op, c, inplace, errors);
    }
    check(cudaGetLastError());
    unsigned bad, total;
    check(cudaMemcpyAsync(&bad, errors, sizeof(bad), cudaMemcpyDeviceToHost,
                          streams[0]));
    check(cudaStreamSynchronize(streams[0]));
    MPI_Allreduce(&bad, &total, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    if (total) throw std::runtime_error("queued full-output/guard mismatch");
    check(ncclCommFinalize(comm));
    check(ncclCommDestroy(comm));
    for (auto stream : streams) check(cudaStreamDestroy(stream));
    check(cudaFree(errors));
    check(cudaFree(b));
    check(cudaFree(a));
    if (!rank)
      std::puts(
          "PASS local8 queued: 128 mixed calls/rank, alternating streams");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

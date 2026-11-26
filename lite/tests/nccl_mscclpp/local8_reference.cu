// Local port of the per-call reference timing client. Input generation and
// complete output/guard checks match compact_collectives.cu's BF16 protocol.
// Every timed interval includes the ABI call, staging and stream completion.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
__device__ float value(size_t i, int rank, unsigned generation) {
  uint32_t x = uint32_t(i) ^ (uint32_t(rank + 1) * 0x9e3779b9u) ^
               (generation * 0x85ebca6bu);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  return (int(x % 63) - 31) / 32.0f;
}
__global__ void fill(__nv_bfloat16* data, size_t count, int rank,
                     unsigned generation, bool sentinel) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    data[i] = __float2bfloat16(sentinel ? -123.f : value(i, rank, generation));
}
__global__ void verify(__nv_bfloat16 const* data, size_t count, size_t shard,
                       int rank, int op, bool inplace, unsigned generation,
                       unsigned* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count + 16;
       i += gridDim.x * blockDim.x) {
    float expected = (inplace && op == 2 && rank < 7 && i >= count)
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
    if (argc != 8 || ranks != 8)
      throw std::invalid_argument(
          "op(ag/rs/ar) layout(out/in) begin end warmups iterations repeats; "
          "eight ranks");
    std::string name = argv[1], layout = argv[2];
    int op = name == "ag" ? 1 : name == "rs" ? 2 : name == "ar" ? 3 : 0;
    bool inplace = layout == "in";
    size_t begin = std::stoull(argv[3]), end = std::stoull(argv[4]);
    int warmup = std::stoi(argv[5]), iterations = std::stoi(argv[6]);
    int repeats = std::stoi(argv[7]);
    if (!op || (layout != "in" && layout != "out") || begin < 1024 ||
        end > (1UL << 30) || begin > end || (begin & (begin - 1)) ||
        (end & (end - 1)) || warmup < 1 || iterations < 1 || repeats < 1)
      throw std::invalid_argument("invalid timing protocol");
    check(cudaSetDevice(rank));
    cudaStream_t stream;
    cudaEvent_t start, stop;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    __nv_bfloat16 *a, *b;
    unsigned* errors;
    check(cudaMalloc(&a, end + 32));
    check(cudaMalloc(&b, end + 32));
    check(cudaMalloc(&errors, sizeof(unsigned)));
    ncclUniqueId id;
    if (!rank) check(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    check(ncclCommInitRank(&comm, ranks, id, rank));
    if (!rank)
      std::printf(
          "# BF16, maximum-rank per-call CUDA events; warmup=%d iterations=%d "
          "repeats=%d; layout=%s\n",
          warmup, iterations, repeats, layout.c_str());
    unsigned generation = 0;
    for (size_t bytes = begin; bytes <= end; bytes *= 2) {
      size_t count = bytes / 2, shard = count / 8;
      size_t ni = op == 1 ? shard : count, no = op == 2 ? shard : count;
      auto input = a + (inplace && op == 1 ? rank * shard : 0);
      auto output = inplace ? a + (op == 2 ? rank * shard : 0) : b;
      for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int iteration = -warmup; iteration < iterations; ++iteration) {
          ++generation;
          fill<<<256, 256, 0, stream>>>(output, no + 16, rank, generation,
                                        true);
          fill<<<256, 256, 0, stream>>>(input, ni, rank, generation, false);
          check(cudaGetLastError());
          check(cudaStreamSynchronize(stream));
          MPI_Barrier(MPI_COMM_WORLD);
          check(cudaEventRecord(start, stream));
          if (op == 1)
            check(ncclAllGather(input, output, shard, ncclBfloat16, comm,
                                stream));
          else if (op == 2)
            check(ncclReduceScatter(input, output, shard, ncclBfloat16, ncclSum,
                                    comm, stream));
          else
            check(ncclAllReduce(input, output, count, ncclBfloat16, ncclSum,
                                comm, stream));
          check(cudaEventRecord(stop, stream));
          check(cudaStreamSynchronize(stream));
          float ms;
          check(cudaEventElapsedTime(&ms, start, stop));
          double us = double(ms) * 1000, maximum;
          MPI_Allreduce(&us, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
          check(cudaMemsetAsync(errors, 0, sizeof(unsigned), stream));
          verify<<<256, 256, 0, stream>>>(output, no, shard, rank, op, inplace,
                                          generation, errors);
          check(cudaGetLastError());
          unsigned bad = 0, total = 0;
          check(cudaMemcpyAsync(&bad, errors, sizeof(bad),
                                cudaMemcpyDeviceToHost, stream));
          check(cudaStreamSynchronize(stream));
          MPI_Allreduce(&bad, &total, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
          if (total)
            throw std::runtime_error("full-output verification failed");
          if (!rank && iteration >= 0)
            std::printf("sample,%zu,candidate,%d,%d,%.6f\n", bytes, repeat,
                        iteration, maximum);
        }
      }
      if (!rank) std::fflush(stdout);
    }
    check(ncclCommFinalize(comm));
    check(ncclCommDestroy(comm));
    check(cudaFree(errors));
    check(cudaFree(b));
    check(cudaFree(a));
    check(cudaEventDestroy(stop));
    check(cudaEventDestroy(start));
    check(cudaStreamDestroy(stream));
    if (!rank) std::puts("PASS local8 reference full-output checks");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

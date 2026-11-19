// NCCL-only client: queued 128/256-MiB layout transitions and allocation reuse.
#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

void checked(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void checked(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(status));
}
__global__ void prepare(float* input, float* output, size_t count,
                        size_t allocation, size_t guard, int rank,
                        int generation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < allocation + 128;
       i += gridDim.x * blockDim.x) {
    input[i] =
        i >= guard && i < guard + count
            ? float(rank + 1 + generation * 8) * float(1 + (i - guard) % 31)
            : -12345.f;
    output[i] = -12345.f;
  }
}
__global__ void verify(float const* output, size_t count, size_t allocation,
                       size_t guard, int rank, int generation, bool scatter,
                       unsigned* errors) {
  size_t resultCount = scatter ? count / 8 : count;
  size_t origin = scatter ? rank * resultCount : 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < allocation + 128;
       i += gridDim.x * blockDim.x) {
    float expected =
        i >= guard && i < guard + resultCount
            ? float(36 + generation * 64) * float(1 + (origin + i - guard) % 31)
            : -12345.f;
    if (output[i] != expected) atomicAdd(errors, 1);
  }
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (world != 8) throw std::runtime_error("requires eight ranks");
    checked(cudaSetDevice(rank % 4));
    constexpr size_t allocation = (256UL << 20) / 4;
    constexpr size_t sizes[] = {128UL << 20, 256UL << 20, 128UL << 20,
                                128UL << 20, 256UL << 20, 128UL << 20};
    size_t guard = 33 + rank;
    float *input, *output;
    unsigned* errors;
    checked(cudaMalloc(&input, (allocation + 128) * 4));
    checked(cudaMalloc(&output, (allocation + 128) * 4));
    checked(cudaMalloc(&errors, sizeof(unsigned)));
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ncclUniqueId id;
    if (!rank) checked(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    checked(ncclCommInitRank(&comm, world, id, rank));
    int recycled = 0;
    for (int stage = 0; stage < 6; ++stage) {
      bool scatter = stage < 3;
      size_t count = sizes[stage] / 4;
      checked(cudaMemsetAsync(errors, 0, sizeof(unsigned), stream));
      MPI_Barrier(MPI_COMM_WORLD);
      for (int i = 0; i < 8; ++i) {
        int generation = stage * 8 + i + 1;
        // Change offsets once after graph warmup, then queue repeated calls
        // with stable addresses while earlier GPU work is still in flight.
        size_t callGuard = guard + ((stage == 1 && i == 0) ? 16 : 0);
        prepare<<<256, 256, 0, stream>>>(input, output, count, allocation,
                                         callGuard, rank, generation);
        checked(cudaGetLastError());
        if (scatter)
          checked(ncclReduceScatter(input + callGuard, output + callGuard,
                                    count / 8, ncclFloat, ncclSum, comm,
                                    stream));
        else
          checked(ncclAllReduce(input + callGuard, output + callGuard, count,
                                ncclFloat, ncclSum, comm, stream));
        verify<<<256, 256, 0, stream>>>(output, count, allocation, callGuard,
                                        rank, generation, scatter, errors);
        checked(cudaGetLastError());
      }
      checked(cudaStreamSynchronize(stream));
      unsigned wrong;
      checked(
          cudaMemcpy(&wrong, errors, sizeof(wrong), cudaMemcpyDeviceToHost));
      if (wrong) throw std::runtime_error("queued full-output/guard mismatch");
      if (stage == 1 || stage == 4) {
        auto* old = input;
        checked(cudaFree(input));
        checked(cudaMalloc(&input, (allocation + 128) * 4));
        recycled += input == old;
      }
    }
    checked(ncclCommFinalize(comm));
    checked(ncclCommDestroy(comm));
    checked(cudaFree(input));
    checked(cudaFree(output));
    checked(cudaFree(errors));
    checked(cudaStreamDestroy(stream));
    std::printf("UCCL layout queue: rank=%d recycled_va=%d calls=48 passed\n",
                rank, recycled);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

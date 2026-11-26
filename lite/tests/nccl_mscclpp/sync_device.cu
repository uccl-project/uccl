// Completion must retain the primary-context and device-switching behavior.
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
extern "C" ncclResult_t uccl_lite_nccl_synchronize(ncclComm_t);
void checked(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void checked(ncclResult_t s) {
  if (s != ncclSuccess) throw std::runtime_error(ncclGetErrorString(s));
}
void checked(CUresult s) {
  if (s != CUDA_SUCCESS) {
    char const* text = nullptr;
    cuGetErrorString(s, &text);
    throw std::runtime_error(text ? text : "CUDA driver error");
  }
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (world != 8) throw std::runtime_error("requires eight ranks");
    int device = rank % 4, other = (device + 1) % 4;
    checked(cudaSetDevice(device));
    CUcontext primary;
    checked(cuCtxGetCurrent(&primary));
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    float *input, *output;
    checked(cudaMalloc(&input, 1024));
    checked(cudaMalloc(&output, 1024));
    ncclUniqueId id;
    if (!rank) checked(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    checked(ncclCommInitRank(&comm, world, id, rank));
    for (int stage = 0; stage < 4; ++stage) {
      checked(cudaSetDevice(device));
      std::vector<float> values(256, float(rank + 1 + stage));
      checked(cudaMemcpyAsync(input, values.data(), 1024,
                              cudaMemcpyHostToDevice, stream));
      checked(
          ncclAllReduce(input, output, 256, ncclFloat, ncclSum, comm, stream));
      CUcontext expected = primary, foreign = nullptr;
      if (stage == 1) {
        checked(cudaSetDevice(other));
        checked(cuCtxGetCurrent(&expected));
      } else if (stage == 2) {
        CUdevice dev;
        checked(cuDeviceGet(&dev, device));
        checked(cuCtxCreate(&foreign, 0, dev));
      }
      checked(uccl_lite_nccl_synchronize(comm));
      CUcontext actual;
      checked(cuCtxGetCurrent(&actual));
      if (actual != expected)
        throw std::runtime_error("completion context mismatch");
      int current;
      checked(cudaGetDevice(&current));
      if (current != (stage == 1 ? other : device))
        throw std::runtime_error("completion device mismatch");
      if (foreign) checked(cuCtxDestroy(foreign));
      checked(cudaSetDevice(device));
      checked(cudaMemcpy(values.data(), output, 1024, cudaMemcpyDeviceToHost));
      for (float value : values)
        if (value != float(36 + stage * 8))
          throw std::runtime_error("completion output mismatch");
    }
    checked(ncclCommFinalize(comm));
    checked(ncclCommDestroy(comm));
    checked(cudaFree(input));
    checked(cudaFree(output));
    checked(cudaStreamDestroy(stream));
    std::printf("UCCL sync context: rank=%d cases=4 passed\n", rank);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

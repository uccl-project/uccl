// Diagnostic only: observe whether dependent operations in one native group
// consume the earlier operation's result. It is not a collective correctness
// oracle.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  cudaSetDevice(rank);
  ncclUniqueId id;
  if (!rank) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  ncclCommInitRank(&comm, ranks, id, rank);
  float *a, *b, *c;
  cudaMalloc(&a, 64 * 4);
  cudaMalloc(&b, 64 * 4);
  cudaMalloc(&c, 64 * ranks * 4);
  std::vector<float> in(64, 10), out(64, -99), gather(64 * ranks, -99);
  cudaMemcpy(a, in.data(), 256, cudaMemcpyHostToDevice);
  cudaMemcpy(b, out.data(), 256, cudaMemcpyHostToDevice);
  cudaMemcpy(c, gather.data(), gather.size() * 4, cudaMemcpyHostToDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  bool mixed = argc > 1;
  if (!mixed || rank == 0) ncclGroupStart();
  ncclAllReduce(a, b, 64, ncclFloat32, ncclSum, comm, stream);
  ncclAllGather(mixed ? a : b, c, 64, ncclFloat32, comm, stream);
  auto status = !mixed || rank == 0 ? ncclGroupEnd() : ncclSuccess;
  auto cudaStatus = cudaStreamSynchronize(stream);
  cudaMemcpy(out.data(), b, 256, cudaMemcpyDeviceToHost);
  cudaMemcpy(gather.data(), c, gather.size() * 4, cudaMemcpyDeviceToHost);
  if (!rank)
    std::cout << "Native-only group observation: AR=" << out[0]
              << " AG=" << gather[0] << " statuses=" << int(status) << ','
              << int(cudaStatus) << '\n';
  cudaStreamDestroy(stream);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  ncclCommDestroy(comm);
  MPI_Finalize();
  return status == ncclSuccess && cudaStatus == cudaSuccess ? 0 : 1;
}

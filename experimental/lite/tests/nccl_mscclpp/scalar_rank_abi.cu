#include "../collectives/scalar_oracle.cuh"
#include <iostream>
#include <mpi.h>
#include <nccl.h>
void check(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(status));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    int device =
        std::getenv("UCCL_LITE_ABI_REVERSE_DEVICES") ? ranks - 1 - rank : rank;
    scalar_test::check(cudaSetDevice(device));
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      scalar_test::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      ncclUniqueId id;
      if (!rank) check(ncclGetUniqueId(&id));
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
      ncclComm_t comm;
      check(ncclCommInitRank(&comm, ranks, id, rank));
      void *a, *b;
      scalar_test::check(cudaMalloc(&a, 256));
      scalar_test::check(cudaMalloc(&b, 256));
      cudaGraph_t graph;
      scalar_test::check(
          cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      if (ncclAllReduce(a, b, 64, ncclFloat32, ncclSum, comm, streams[0]) !=
          ncclInvalidUsage)
        throw std::runtime_error("unqualified FP32 capture accepted");
      scalar_test::check(cudaStreamEndCapture(streams[0], &graph));
      scalar_test::check(cudaGraphDestroy(graph));
      scalar_test::check(cudaFree(a));
      scalar_test::check(cudaFree(b));
      for (size_t count : {64UL, 256UL, 4096UL, 65536UL})
        for (int op = 1; op <= 3; ++op) {
          bool inplace = count == 64 || (count == 4096 && (rank & 1));
          unsigned align = count == 256 ? 4u << (rank % 3) : 256u;
          scalar_test::exercise<uint32_t>(
              op, count, inplace, align, rank, ranks, streams,
              [&](int operation, void const* in, void* out, size_t n,
                  cudaStream_t stream) {
                if (operation == 1)
                  check(ncclAllGather(in, out, n, ncclFloat32, comm, stream));
                else if (operation == 2)
                  check(ncclReduceScatter(in, out, n, ncclFloat32, ncclSum,
                                          comm, stream));
                else
                  check(ncclAllReduce(in, out, n, ncclFloat32, ncclSum, comm,
                                      stream));
              });
        }
      check(ncclCommDestroy(comm));
    }
    for (auto stream : streams) scalar_test::check(cudaStreamDestroy(stream));
    if (!rank)
      std::cout << "PASS: public FP32 NCCL ABI, " << ranks
                << " ranks, two lifetimes, full guards, mixed "
                   "aliases/alignment and bitwise AG\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

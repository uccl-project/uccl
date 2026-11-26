#include "uccl_lite/collectives.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>

void checked(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
float value(size_t index, int rank, int generation) {
  return float(int(index % 17) - 8) * (rank + 1) + generation;
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (world != 8) throw std::runtime_error("requires two four-GPU hosts");
    checked(cudaSetDevice(rank % 4));
    cudaStream_t streams[2];
    for (auto& stream : streams)
      checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    size_t total = ((32UL << 20) + 128) / 4 + 256;
    size_t guard = 33 + rank;
    float *a, *b;
    checked(cudaMalloc(&a, total * 4));
    checked(cudaMalloc(&b, total * 4));
    int calls = 0;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      uccl_lite::NetworkId id{};
      if (!rank) id = uccl_lite::NetworkCollectives::createUniqueId();
      MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      // On a failed check MPI_Abort terminates the job without waiting on a
      // possibly incomplete peer. The normal path destroys both lifetimes.
      auto* comm = new uccl_lite::NetworkCollectives(id, rank, rank % 4,
                                                     (rank % 4) / 2, 1UL << 30);
      comm->prepare();
      for (int generation = 0; generation < 2; ++generation) {
        for (size_t bytes :
             {128UL, 768UL, 1024UL, (16UL << 10) + 128, (32UL << 10) + 128,
              (64UL << 10) + 128, (512UL << 10) + 128, (1UL << 20) + 128,
              (4UL << 20) + 128, (7UL << 20) + 128, (8UL << 20) + 128,
              12UL << 20, (16UL << 20) + 128, (32UL << 20) + 128}) {
          for (int op = 1; op <= 3; ++op) {
            size_t count = bytes / (op == 3 ? 4 : 32);
            size_t ni = count * (op == 2 ? 8 : 1);
            size_t no = count * (op == 1 ? 8 : 1);
            bool inplace = (rank + generation + op) % 2;
            size_t sourceOffset = inplace && op == 1 ? rank * count : 0;
            size_t targetOffset = inplace && op == 2 ? rank * count : 0;
            std::vector<float> input(total, -12345.f), output(total, -12345.f);
            for (size_t i = 0; i < ni; ++i)
              input[guard + sourceOffset + i] = value(i, rank, generation);
            float* source = generation & 1 ? b : a;
            float* target = inplace ? source : (generation & 1 ? a : b);
            cudaStream_t stream = streams[calls % 2];
            checked(cudaMemcpyAsync(source, input.data(), total * 4,
                                    cudaMemcpyHostToDevice, stream));
            if (!inplace)
              checked(cudaMemcpyAsync(target, output.data(), total * 4,
                                      cudaMemcpyHostToDevice, stream));
            comm->enqueue(static_cast<uccl_lite::NetworkOperation>(op),
                          source + guard + sourceOffset,
                          target + guard + targetOffset, count, stream);
            checked(cudaMemcpyAsync(output.data(), target, total * 4,
                                    cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            for (size_t i = 0; i < total; ++i) {
              float expected = inplace ? input[i] : -12345.f;
              if (i >= guard + targetOffset && i < guard + targetOffset + no) {
                size_t index = i - guard - targetOffset;
                if (op == 1)
                  expected = value(index % count, index / count, generation);
                else {
                  expected = 0;
                  for (int peer = 0; peer < 8; ++peer)
                    expected += value(op == 2 ? rank * count + index : index,
                                      peer, generation);
                }
              }
              if (output[i] != expected) {
                std::fprintf(
                    stderr,
                    "rank=%d bytes=%zu op=%d index=%zu got=%g expected=%g\n",
                    rank, bytes, op, i, output[i], expected);
                throw std::runtime_error("full-buffer/guard mismatch");
              }
            }
            ++calls;
          }
        }
        checked(cudaFree(a));
        checked(cudaMalloc(&a, total * 4));
      }
      delete comm;
      MPI_Barrier(MPI_COMM_WORLD);
    }
    checked(cudaFree(a));
    checked(cudaFree(b));
    for (auto stream : streams) checked(cudaStreamDestroy(stream));
    if (!rank)
      std::printf("Network size checks passed: %d calls/rank, two lifetimes\n",
                  calls);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

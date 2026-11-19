#include "uccl_lite/collectives.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>

void checked(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
struct Slot {
  float *input, *output, *hostInput, *hostOutput;
  size_t guard;
  int epoch;
  bool inplace;
};
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (world != 8) throw std::runtime_error("requires two four-GPU hosts");
    checked(cudaSetDevice(rank % 4));
    constexpr size_t elements = (8UL << 20) / 4;
    constexpr size_t storage = elements + 256;
    constexpr int batch = 8;
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int calls = 0;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      std::vector<Slot> slots(batch);
      for (auto& s : slots) {
        checked(cudaMalloc(&s.input, storage * 4));
        checked(cudaMalloc(&s.output, storage * 4));
        checked(cudaMallocHost(&s.hostInput, storage * 4));
        checked(cudaMallocHost(&s.hostOutput, storage * 4));
      }
      uccl_lite::NetworkId id{};
      if (!rank) id = uccl_lite::NetworkCollectives::createUniqueId();
      MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      auto* comm = new uccl_lite::NetworkCollectives(id, rank, rank % 4,
                                                     (rank % 4) / 2, 1UL << 30);
      comm->prepare();
      auto runBatch = [&](int first, int count) {
        // Prepare pinned inputs before enqueueing any work; each outstanding
        // call has distinct caller buffers and a different signed oracle.
        for (int j = 0; j < count; ++j) {
          auto& s = slots[j];
          s.epoch = lifetime * 400 + first + j + 1;
          s.guard = 32 + 4 * (rank % 4) + ((rank + s.epoch) & 1);
          s.inplace = (rank + s.epoch / 2) & 1;
          std::fill(s.hostInput, s.hostInput + storage, -12345.f);
          std::fill(s.hostOutput, s.hostOutput + storage, -12345.f);
          for (size_t i = 0; i < elements; ++i)
            s.hostInput[s.guard + i] =
                float(int(i % 17) - 8) * (rank + 1) + s.epoch;
        }
        if (rank == 3 && first == batch) usleep(10000);
        for (int j = 0; j < count; ++j) {
          auto& s = slots[j];
          checked(cudaMemcpyAsync(s.input, s.hostInput, storage * 4,
                                  cudaMemcpyHostToDevice, stream));
          float* target = s.inplace ? s.input : s.output;
          if (!s.inplace)
            checked(cudaMemcpyAsync(target, s.hostOutput, storage * 4,
                                    cudaMemcpyHostToDevice, stream));
          comm->enqueue(uccl_lite::NetworkOperation::AllReduce,
                        s.input + s.guard, target + s.guard, elements, stream);
          checked(cudaMemcpyAsync(s.hostOutput, target, storage * 4,
                                  cudaMemcpyDeviceToHost, stream));
        }
        checked(cudaStreamSynchronize(stream));
        for (int j = 0; j < count; ++j) {
          auto& s = slots[j];
          for (size_t i = 0; i < storage; ++i) {
            float expected = -12345.f;
            if (i >= s.guard && i < s.guard + elements)
              expected = float(int((i - s.guard) % 17) - 8) * 36 + 8 * s.epoch;
            if (s.hostOutput[i] != expected)
              throw std::runtime_error("queue stress output/guard mismatch");
          }
          ++calls;
        }
      };
      for (int first = 0; first < 301; first += batch)
        runBatch(first, std::min(batch, 301 - first));
      if (comm->idle())
        throw std::runtime_error("unretired transport was reported idle");
      // Releasing caller allocations must not invalidate deferred sends.
      for (auto& s : slots) {
        checked(cudaFree(s.input));
        checked(cudaFree(s.output));
        s.input = s.output = nullptr;
      }
      comm->synchronize();
      if (!comm->idle())
        throw std::runtime_error("synchronize did not retire the transport");
      checked(cudaMalloc(&slots[0].input, storage * 4));
      checked(cudaMalloc(&slots[0].output, storage * 4));
      runBatch(301, 1);
      if (comm->idle())
        throw std::runtime_error("last deferred send was not tracked");
      delete comm;
      for (auto& s : slots) {
        if (s.input) checked(cudaFree(s.input));
        if (s.output) checked(cudaFree(s.output));
        checked(cudaFreeHost(s.hostInput));
        checked(cudaFreeHost(s.hostOutput));
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    checked(cudaStreamDestroy(stream));
    if (!rank)
      std::printf("Network queue checks passed: %d calls/rank, two lifetimes\n",
                  calls);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

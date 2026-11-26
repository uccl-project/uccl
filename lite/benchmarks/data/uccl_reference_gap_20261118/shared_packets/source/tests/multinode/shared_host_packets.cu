#include "../../src/multinode/shared_host_packets.h"
#include <cstdio>
#include <vector>
#include <mpi.h>

void checked(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
__global__ void uccl_lite_copy_shared_words(uint32_t const* source,
                                            uint32_t* target, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    target[i] = source[i];
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (world != 8) throw std::runtime_error("requires two four-GPU hosts");
    checked(cudaSetDevice(rank % 4));
    constexpr size_t capacity = 256UL << 10;
    uint32_t *input, *output;
    checked(cudaMalloc(&input, capacity));
    checked(cudaMalloc(&output, 4 * capacity));
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int cases = 0;
    for (int life = 0; life < 2; ++life) {
      auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, 8);
      mscclpp::UniqueId id{};
      if (!rank) id = mscclpp::TcpBootstrap::createUniqueId();
      MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      bootstrap->initialize(id, 120);
      auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
      auto* window =
          new uccl_lite::multinode::SharedHostPackets(*comm, rank, capacity);
      for (int mode = 0; mode < 2; ++mode)
        for (size_t bytes : {128UL, 4096UL, capacity - 16, capacity})
          for (int slot = 0; slot < 3; ++slot) {
            size_t words = bytes / 4;
            std::vector<uint32_t> source(words), result(4 * words);
            for (size_t i = 0; i < words; ++i)
              source[i] = rank * 1000000 + cases * 10000 + i;
            checked(cudaMemcpyAsync(input, source.data(), bytes,
                                    cudaMemcpyHostToDevice, stream));
            checked(cudaMemcpyAsync(window->row(rank, slot, bytes), input,
                                    bytes, cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            __atomic_store_n(window->flag((rank % 4) / 2, slot, rank % 2),
                             uint64_t(cases * 16 + rank + 1), __ATOMIC_RELEASE);
            bootstrap->barrier();
            for (int local = 0; local < 4; ++local) {
              int owner = rank / 4 * 4 + local;
              if (__atomic_load_n(window->flag(local / 2, slot, local % 2),
                                  __ATOMIC_ACQUIRE) !=
                  uint64_t(cases * 16 + owner + 1))
                throw std::runtime_error("shared flag mismatch");
              if (!mode)
                checked(cudaMemcpyAsync(output + local * words,
                                        window->row(owner, slot, bytes), bytes,
                                        cudaMemcpyHostToDevice, stream));
              else {
                uccl_lite_copy_shared_words<<<32, 256, 0, stream>>>(
                    reinterpret_cast<uint32_t const*>(
                        window->row(owner, slot, bytes, true)),
                    output + local * words, words);
                checked(cudaGetLastError());
              }
            }
            checked(cudaMemcpyAsync(result.data(), output, 4 * bytes,
                                    cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            for (int local = 0; local < 4; ++local)
              for (size_t i = 0; i < words; ++i)
                if (result[local * words + i] !=
                    uint32_t((rank / 4 * 4 + local) * 1000000 + cases * 10000 +
                             i))
                  throw std::runtime_error("shared CUDA packet mismatch");
            bootstrap->barrier();
            ++cases;
          }
      delete window;
      bootstrap->barrier();
    }
    checked(cudaFree(input));
    checked(cudaFree(output));
    checked(cudaStreamDestroy(stream));
    if (!rank)
      std::printf("Shared host packets passed: %d cases/rank, two lifetimes\n",
                  cases);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

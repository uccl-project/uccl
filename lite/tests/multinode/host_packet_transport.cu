#include "../../src/multinode/host_packet_transport.h"
#include <chrono>
#include <cstdio>
#include <thread>
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
    checked(cudaMalloc(&output, 8 * capacity));
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
      auto* transport =
          new uccl_lite::multinode::HostPacketTransport(*comm, *window, rank);
      std::array<uint64_t, 3> epochs{};
      auto wait = [](auto predicate) {
        auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!predicate()) {
          if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("host packet progress timeout");
          std::this_thread::yield();
        }
      };
      for (int mode = 0; mode < 2; ++mode)
        for (size_t bytes : {128UL, 4096UL, capacity - 16, capacity})
          for (int slot = 0; slot < 3; ++slot) {
            size_t words = bytes / 4;
            std::vector<uint32_t> source(words), result(8 * words);
            uint64_t epoch = ++epochs[slot];
            wait([&] { return transport->canReuse(slot, epoch); });
            if (!(rank & 1)) transport->grant(slot, epoch);
            for (size_t i = 0; i < words; ++i)
              source[i] = rank * 1000000 + cases * 10000 + i;
            checked(cudaMemcpyAsync(input, source.data(), bytes,
                                    cudaMemcpyHostToDevice, stream));
            checked(cudaMemcpyAsync(window->row(rank, slot, bytes), input,
                                    bytes, cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            transport->uploaded(slot, epoch);
            if (!(rank & 1)) {
              wait([&] { return transport->canSend(slot, epoch); });
              transport->send(slot, bytes, epoch);
              transport->flush();
            }
            wait([&] {
              return transport->localReady(slot, epoch) &&
                     transport->received(slot, epoch);
            });
            if (rank == 3 && cases % 5 == 0) usleep(10000);
            for (int local = 0; local < 8; ++local) {
              int owner = local;
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
            checked(cudaMemcpyAsync(result.data(), output, 8 * bytes,
                                    cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            for (int local = 0; local < 8; ++local)
              for (size_t i = 0; i < words; ++i)
                if (result[local * words + i] !=
                    uint32_t(local * 1000000 + cases * 10000 + i))
                  throw std::runtime_error("shared CUDA packet mismatch");
            if (transport->canReuse(slot, epoch + 1))
              throw std::runtime_error("slot reused before reader ACK");
            if (!(rank & 1)) {
              bool rejected = false;
              try {
                transport->grant(slot, epoch + 1);
              } catch (std::logic_error const&) {
                rejected = true;
              }
              if (!rejected)
                throw std::runtime_error("receive credit bypassed reader ACK");
            }
            transport->consumed(slot, epoch);
            ++cases;
          }
      bootstrap->barrier();
      delete transport;
      delete window;
      bootstrap->barrier();
    }
    checked(cudaFree(input));
    checked(cudaFree(output));
    checked(cudaStreamDestroy(stream));
    if (!rank)
      std::printf(
          "Host packet transport passed: %d cases/rank, two lifetimes\n",
          cases);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

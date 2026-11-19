#include "../../src/multinode/host_packet_transport.h"
#include <chrono>
#include <cstdio>
#include <exception>
#include <functional>
#include <thread>
#include <vector>
#include <mpi.h>

void checked(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct Callback {
  std::function<void()> action;
  std::exception_ptr error;
  std::atomic<bool> failed{false};
  static void CUDART_CB invoke(void* pointer) {
    auto& self = *static_cast<Callback*>(pointer);
    try {
      self.action();
    } catch (...) {
      self.error = std::current_exception();
      self.failed.store(true, std::memory_order_release);
    }
  }
  void check() {
    if (failed.load(std::memory_order_acquire)) std::rethrow_exception(error);
  }
};
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
            for (size_t i = 0; i < words; ++i)
              source[i] = rank * 1000000 + cases * 10000 + i;
            checked(cudaMemcpyAsync(input, source.data(), bytes,
                                    cudaMemcpyHostToDevice, stream));
            checked(cudaStreamSynchronize(stream));
            Callback acquire, exchange, acknowledge;
            acquire.action = [&] {
              wait([&] { return transport->canReuse(slot, epoch); });
              if (!(rank & 1)) transport->grant(slot, epoch);
            };
            exchange.action = [&] {
              if (acquire.failed.load(std::memory_order_acquire)) return;
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
            };
            acknowledge.action = [&] {
              if (acquire.failed.load(std::memory_order_acquire) ||
                  exchange.failed.load(std::memory_order_acquire))
                return;
              transport->consumed(slot, epoch);
            };
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            if (mode)
              checked(cudaStreamBeginCapture(stream,
                                             cudaStreamCaptureModeThreadLocal));
            checked(cudaLaunchHostFunc(stream, Callback::invoke, &acquire));
            checked(cudaMemcpyAsync(window->row(rank, slot, bytes), input,
                                    bytes, cudaMemcpyDeviceToHost, stream));
            checked(cudaLaunchHostFunc(stream, Callback::invoke, &exchange));
            for (int owner = 0; owner < 8; ++owner)
              checked(cudaMemcpyAsync(output + owner * words,
                                      window->row(owner, slot, bytes), bytes,
                                      cudaMemcpyHostToDevice, stream));
            checked(cudaLaunchHostFunc(stream, Callback::invoke, &acknowledge));
            if (mode) {
              checked(cudaStreamEndCapture(stream, &graph));
              checked(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                           0));
              checked(cudaGraphLaunch(executable, stream));
            }
            // Deliberately pageable: callbacks must progress while the caller
            // is inside a CUDA runtime copy/synchronization operation.
            checked(cudaMemcpyAsync(result.data(), output, 8 * bytes,
                                    cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            acquire.check();
            exchange.check();
            acknowledge.check();
            if (mode) {
              checked(cudaGraphExecDestroy(executable));
              checked(cudaGraphDestroy(graph));
            }
            for (int local = 0; local < 8; ++local)
              for (size_t i = 0; i < words; ++i)
                if (result[local * words + i] !=
                    uint32_t(local * 1000000 + cases * 10000 + i))
                  throw std::runtime_error("shared CUDA packet mismatch");
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
          "Host packet callbacks passed: %d cases/rank, two lifetimes\n",
          cases);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

#include "uccl_lite/cpu_router/channel_device.cuh"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using uccl_lite::cpu_router::DeviceSignalRef;
using uccl_lite::cpu_router::GenerationSignal;

struct alignas(4096) SharedPacket {
  GenerationSignal uploaded[8];
  int payload[8];
};

__global__ void upload(SharedPacket* shared, int rank) {
  if (threadIdx.x == 0) {
    shared->payload[rank] = rank + 1;
    DeviceSignalRef{&shared->uploaded[rank]}.publish(1);
  }
}

__global__ void consume(SharedPacket const* shared, int* result) {
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int rank = 0; rank < 8; ++rank) sum += shared->payload[rank];
    *result = sum;
  }
}

void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
    std::exit(2);
  }
}

void child(int rank, SharedPacket* shared, size_t bytes) {
  check(cudaSetDevice(rank));
  check(cudaHostRegister(shared, bytes,
                         cudaHostRegisterPortable | cudaHostRegisterMapped));
  SharedPacket* mapped = nullptr;
  check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), shared, 0));
  upload<<<1, 32>>>(mapped, rank);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  for (int peer = 0; peer < 8; ++peer)
    while (__atomic_load_n(&shared->uploaded[peer].value, __ATOMIC_ACQUIRE) <
           1) {
      if (std::chrono::steady_clock::now() >= deadline) {
        std::fprintf(stderr, "rank %d: shared GPU signal timed out\n", rank);
        std::exit(3);
      }
      std::this_thread::yield();
    }

  int* result = nullptr;
  check(cudaMalloc(&result, sizeof(int)));
  consume<<<1, 32>>>(mapped, result);
  check(cudaGetLastError());
  int host = 0;
  check(cudaMemcpy(&host, result, sizeof(int), cudaMemcpyDeviceToHost));
  if (host != 36) {
    std::fprintf(stderr, "rank %d: shared GPU read got %d\n", rank, host);
    std::exit(4);
  }
  check(cudaFree(result));
  check(cudaHostUnregister(shared));
}
}  // namespace

int main() {
  char name[64];
  std::snprintf(name, sizeof(name), "/uccl-lite-packet-test-%d", int(getpid()));
  int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) return 1;
  size_t bytes = 4096;
  if (ftruncate(fd, bytes) != 0) return 1;
  void* mapping =
      mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapping == MAP_FAILED) return 1;
  auto* shared = new (mapping) SharedPacket{};
  bool passed = true;
  for (int rank = 0; rank < 8; ++rank) {
    pid_t pid = fork();
    if (pid == 0) {
      child(rank, shared, bytes);
      std::_Exit(0);
    }
    if (pid < 0) passed = false;
  }
  for (int rank = 0; rank < 8; ++rank) {
    int status = 0;
    if (wait(&status) < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
      passed = false;
  }
  munmap(mapping, bytes);
  close(fd);
  shm_unlink(name);
  if (passed) std::puts("8-process mapped packet and GPU signals passed");
  return passed ? 0 : 1;
}

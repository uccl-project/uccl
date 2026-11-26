#pragma once
// Shared allocation and harness helpers for explicit two-node host collectives.
#include <linux/mempolicy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace uccl_lite {
namespace host_collective {
constexpr int kNodes = 2;
constexpr int kGroups = 2;
constexpr int kGpusPerGroup = 2;
constexpr int kRanks = kNodes * kGroups;
constexpr int kGpus = kRanks * kGpusPerGroup;
constexpr int kSlots = 3;

inline void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
inline void check(int result) {
  if (result != MPI_SUCCESS) {
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(result, message, &length);
    throw std::runtime_error(std::string(message, length));
  }
}
inline void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}

inline void bindGroupBeforeMpi() {
  char const* text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int group = text ? std::atoi(text) : 0;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  for (int cpu = group; cpu < 48; cpu += 2) CPU_SET(cpu, &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity))
    throw std::runtime_error("CPU affinity failed");
  if (!std::getenv("UCX_NET_DEVICES"))
    setenv("UCX_NET_DEVICES", group == 0 ? "mlx5_0:1" : "mlx5_1:1", 1);
  if (!std::getenv("UCX_TLS")) setenv("UCX_TLS", "rc_mlx5,ud_mlx5,sm,self", 1);
}
struct HostBlock {
  char* host = nullptr;
  bool ownsMapping = true;
  size_t bytes = 0;
  std::array<char*, kGpusPerGroup> mapped{};
  void initialize(size_t capacity, int node,
                  std::array<int, kGpusPerGroup> const& devices,
                  void* external = nullptr) {
    bytes = capacity;
    ownsMapping = external == nullptr;
    void* address = external ? external
                             : mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED) throw std::runtime_error("host mmap failed");
    unsigned long mask = 1UL << node;
    if (ownsMapping && syscall(SYS_mbind, address, bytes, MPOL_BIND, &mask,
                               8 * sizeof(mask), 0)) {
      munmap(address, bytes);
      throw std::runtime_error("host mbind failed");
    }
    if (ownsMapping) std::memset(address, 0, bytes);
    check(cudaSetDevice(devices[0]));
    auto result = cudaHostRegister(
        address, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (result != cudaSuccess) {
      if (ownsMapping) munmap(address, bytes);
      check(result);
    }
    host = static_cast<char*>(address);
    for (int i = 0; i < kGpusPerGroup; ++i) {
      check(cudaSetDevice(devices[i]));
      check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped[i]), host,
                                     0));
    }
  }
  ~HostBlock() {
    if (host) {
      cudaHostUnregister(host);
      if (ownsMapping) munmap(host, bytes);
    }
  }
  HostBlock() = default;
  HostBlock(HostBlock const&) = delete;
  HostBlock& operator=(HostBlock const&) = delete;
};
// Each owner first-touches its own shared pages before the peer registers
// them for GPU reads. Windows are explicitly closed after CUDA unregistration.
struct SharedPackets {
  std::array<MPI_Win, kSlots> windows{MPI_WIN_NULL, MPI_WIN_NULL, MPI_WIN_NULL};
  std::array<char*, kSlots> own{}, peer{};
  void initialize(size_t capacity, int group, MPI_Comm localComm) {
    size_t page = size_t(sysconf(_SC_PAGESIZE));
    size_t rounded = (capacity + page - 1) / page * page;
    MPI_Info info;
    check(MPI_Info_create(&info));
    check(MPI_Info_set(info, "alloc_shared_noncontig", "true"));
    for (int slot = 0; slot < kSlots; ++slot) {
      void* raw = nullptr;
      check(MPI_Win_allocate_shared(rounded + page, 1, info, localComm, &raw,
                                    &windows[slot]));
      MPI_Aint peerSize = 0;
      int disp = 0;
      void* other = nullptr;
      check(MPI_Win_shared_query(windows[slot], group ^ 1, &peerSize, &disp,
                                 &other));
      own[slot] = reinterpret_cast<char*>(
          (reinterpret_cast<uintptr_t>(raw) + page - 1) / page * page);
      peer[slot] = reinterpret_cast<char*>(
          (reinterpret_cast<uintptr_t>(other) + page - 1) / page * page);
      int* model = nullptr;
      int present = 0;
      check(MPI_Win_get_attr(windows[slot], MPI_WIN_MODEL, &model, &present));
      if (!present || *model != MPI_WIN_UNIFIED)
        throw std::runtime_error("shared packets require unified windows");
      unsigned long mask = 1UL << group;
      if (syscall(SYS_mbind, own[slot], rounded, MPOL_BIND, &mask,
                  8 * sizeof(mask), 0))
        throw std::runtime_error("shared packet mbind failed");
      std::memset(own[slot], 0, rounded);
    }
    check(MPI_Info_free(&info));
    check(MPI_Barrier(localComm));
  }
  void close() {
    for (auto& window : windows)
      if (window != MPI_WIN_NULL) check(MPI_Win_free(&window));
  }
};

struct Device {
  int physical;
  int globalRank;
  float *input = nullptr, *output = nullptr;
  unsigned int* wrong = nullptr;
  cudaStream_t upload = nullptr, consume = nullptr;
  std::array<cudaEvent_t, kSlots> uploadDone{};
  std::array<cudaEvent_t, kSlots> consumeDone{};
  Device(int device, int global, size_t maxShardBytes, size_t maxOutputBytes)
      : physical(device), globalRank(global) {
    check(cudaSetDevice(physical));
    check(cudaMalloc(&input, maxShardBytes));
    check(cudaMalloc(&output, maxOutputBytes));
    check(cudaMalloc(&wrong, sizeof(unsigned int)));
    check(cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&consume, cudaStreamNonBlocking));
    for (int slot = 0; slot < kSlots; ++slot) {
      check(
          cudaEventCreateWithFlags(&uploadDone[slot], cudaEventDisableTiming));
      check(
          cudaEventCreateWithFlags(&consumeDone[slot], cudaEventDisableTiming));
    }
  }
  ~Device() {
    cudaSetDevice(physical);
    for (int slot = 0; slot < kSlots; ++slot) {
      if (uploadDone[slot]) cudaEventDestroy(uploadDone[slot]);
      if (consumeDone[slot]) cudaEventDestroy(consumeDone[slot]);
    }
    if (upload) cudaStreamDestroy(upload);
    if (consume) cudaStreamDestroy(consume);
    if (wrong) cudaFree(wrong);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
  }
};
static __global__ void fillInput(float* input, size_t count, int rank,
                                 int generation) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  float factor = float(rank + 1 + generation * kGpus);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride)
    input[i] = factor * float(1 + i % 97);
}

static __global__ void delayConsumerKernel() {
  unsigned long long start = clock64();
  while (clock64() - start < 200000) __nanosleep(64);
}

inline double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace host_collective
}  // namespace uccl_lite

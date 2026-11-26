// Hierarchical two-node AllReduce without GPU P2P or GPUDirect RDMA.
// GPU kernels perform all arithmetic while MPI routes mapped pinned-host slots.

#include <mscclpp/concurrency_device.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <mpi.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CUDACHECK(cmd)                                           \
  do {                                                           \
    cudaError_t e = (cmd);                                       \
    if (e != cudaSuccess) {                                      \
      std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); \
      MPI_Abort(MPI_COMM_WORLD, 1);                              \
    }                                                            \
  } while (0)
#define MPICHECK(cmd)                            \
  do {                                           \
    int e = (cmd);                               \
    if (e != MPI_SUCCESS) {                      \
      std::fprintf(stderr, "MPI error %d\n", e); \
      MPI_Abort(MPI_COMM_WORLD, e);              \
    }                                            \
  } while (0)

constexpr int kLocalRanks = 4;
constexpr int kWorld = 8;
constexpr int kMaxChannels = 8;
constexpr int kLocalStages = 6;
__device__ mscclpp::DeviceSyncer gChannelSyncers[kMaxChannels];

__device__ __forceinline__ void channelSync(int channel) {
  gChannelSyncers[channel].sync(gridDim.x);
}

__device__ __forceinline__ uint64_t loadHostFlag(uint64_t const* flag) {
  uint64_t value;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(flag)
               : "memory");
  return value;
}

__device__ __forceinline__ void storeHostFlag(uint64_t* flag, uint64_t value) {
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(flag), "l"(value)
               : "memory");
  __threadfence_system();
}

__device__ void copyGpuHostDevice(float* dst, float const* src, size_t count,
                                  int tid, int nth) {
  if ((((uintptr_t)dst | (uintptr_t)src) & 15) == 0) {
    size_t n4 = count / 4;
    float4* d4 = reinterpret_cast<float4*>(dst);
    float4 const* s4 = reinterpret_cast<float4 const*>(src);
    for (size_t i = tid; i < n4; i += nth) d4[i] = s4[i];
    for (size_t i = n4 * 4 + tid; i < count; i += nth) dst[i] = src[i];
  } else {
    for (size_t i = tid; i < count; i += nth) dst[i] = src[i];
  }
}

__device__ __forceinline__ float4 loadVolatile4(float4 const* ptr) {
  float4 value;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(ptr)
               : "memory");
  return value;
}

__device__ __forceinline__ float loadVolatile1(float const* ptr) {
  float value;
  asm volatile("ld.volatile.global.f32 %0, [%1];"
               : "=f"(value)
               : "l"(ptr)
               : "memory");
  return value;
}

__device__ void copyHostGpuDevice(float* dst, float const* src, size_t count,
                                  int tid, int nth) {
  if ((((uintptr_t)dst | (uintptr_t)src) & 15) == 0) {
    size_t n4 = count / 4;
    float4* d4 = reinterpret_cast<float4*>(dst);
    float4 const* s4 = reinterpret_cast<float4 const*>(src);
    for (size_t i = tid; i < n4; i += nth) d4[i] = loadVolatile4(s4 + i);
    for (size_t i = n4 * 4 + tid; i < count; i += nth)
      dst[i] = loadVolatile1(src + i);
  } else {
    for (size_t i = tid; i < count; i += nth) dst[i] = loadVolatile1(src + i);
  }
}

__device__ void addHostDevice(float* dst, float const* src, size_t count,
                              int tid, int nth) {
  if ((((uintptr_t)dst | (uintptr_t)src) & 15) == 0) {
    size_t n4 = count / 4;
    float4* d4 = reinterpret_cast<float4*>(dst);
    float4 const* s4 = reinterpret_cast<float4 const*>(src);
    for (size_t i = tid; i < n4; i += nth) {
      float4 x = d4[i], y = loadVolatile4(s4 + i);
      x.x += y.x;
      x.y += y.y;
      x.z += y.z;
      x.w += y.w;
      d4[i] = x;
    }
    for (size_t i = n4 * 4 + tid; i < count; i += nth)
      dst[i] += loadVolatile1(src + i);
  } else {
    for (size_t i = tid; i < count; i += nth) dst[i] += loadVolatile1(src + i);
  }
}

__global__ void persistentHostChannel(float* data, size_t count,
                                      float* localInbox, float* successorInbox,
                                      size_t channelStride,
                                      size_t localFlagOffset, float* crossSend,
                                      float* crossRecv, size_t crossFlagOffset,
                                      int position, int channel, int channels,
                                      uint64_t sequence) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  size_t chunk = ((count + kLocalRanks - 1) / kLocalRanks + 31) & ~(size_t)31;
  size_t channelElems = ((chunk + channels - 1) / channels + 31) & ~(size_t)31;
  size_t subOffset = (size_t)channel * channelElems;
  float* sendSlot = successorInbox + (size_t)channel * channelStride;
  float* recvSlot = localInbox + (size_t)channel * channelStride;
  uint64_t* successorFlags = reinterpret_cast<uint64_t*>(
      reinterpret_cast<char*>(successorInbox) + localFlagOffset);
  uint64_t* localFlags = reinterpret_cast<uint64_t*>(
      reinterpret_cast<char*>(localInbox) + localFlagOffset);
  uint64_t* successorAcks = successorFlags + channels * kLocalStages;
  uint64_t* localAcks = localFlags + channels * kLocalStages;
  uint64_t* crossSendFlags = reinterpret_cast<uint64_t*>(
      reinterpret_cast<char*>(crossSend) + crossFlagOffset);
  uint64_t* crossRecvFlags = reinterpret_cast<uint64_t*>(
      reinterpret_cast<char*>(crossRecv) + crossFlagOffset);
  auto partLen = [&](int chunkIndex) {
    size_t base = (size_t)chunkIndex * chunk + subOffset;
    if (base >= count || subOffset >= chunk) return (size_t)0;
    return min(channelElems, min(chunk - subOffset, count - base));
  };

  for (int step = 0; step < kLocalRanks - 1; step++) {
    int sendChunk = (position - step + kLocalRanks) % kLocalRanks;
    int recvChunk = (sendChunk - 1 + kLocalRanks) % kLocalRanks;
    size_t sendLen = partLen(sendChunk), recvLen = partLen(recvChunk);
    copyGpuHostDevice(sendSlot, data + (size_t)sendChunk * chunk + subOffset,
                      sendLen, tid, nth);
    __threadfence_system();
    channelSync(channel);
    if (tid == 0) {
      storeHostFlag(successorFlags + channel * kLocalStages + step, sequence);
      while (loadHostFlag(localFlags + channel * kLocalStages + step) !=
             sequence) {
      }
    }
    channelSync(channel);
    addHostDevice(data + (size_t)recvChunk * chunk + subOffset, recvSlot,
                  recvLen, tid, nth);
    channelSync(channel);
    if (tid == 0) {
      __threadfence_system();
      storeHostFlag(localAcks + channel * kLocalStages + step, sequence);
      while (loadHostFlag(successorAcks + channel * kLocalStages + step) !=
             sequence) {
      }
    }
    channelSync(channel);
  }

  int ownChunk = (position + 1) % kLocalRanks;
  size_t ownLen = partLen(ownChunk);
  float* crossSendSlot = crossSend + (size_t)channel * channelStride;
  float* crossRecvSlot = crossRecv + (size_t)channel * channelStride;
  copyGpuHostDevice(crossSendSlot, data + (size_t)ownChunk * chunk + subOffset,
                    ownLen, tid, nth);
  __threadfence_system();
  channelSync(channel);
  if (tid == 0) {
    storeHostFlag(crossSendFlags + channel, sequence);
    while (loadHostFlag(crossRecvFlags + channel) != sequence) {
    }
  }
  channelSync(channel);
  addHostDevice(data + (size_t)ownChunk * chunk + subOffset, crossRecvSlot,
                ownLen, tid, nth);
  channelSync(channel);

  for (int step = 0; step < kLocalRanks - 1; step++) {
    int sendChunk = (position + 1 - step + kLocalRanks) % kLocalRanks;
    int recvChunk = (sendChunk - 1 + kLocalRanks) % kLocalRanks;
    size_t sendLen = partLen(sendChunk), recvLen = partLen(recvChunk);
    copyGpuHostDevice(sendSlot, data + (size_t)sendChunk * chunk + subOffset,
                      sendLen, tid, nth);
    __threadfence_system();
    channelSync(channel);
    if (tid == 0) {
      storeHostFlag(successorFlags + channel * kLocalStages + 3 + step,
                    sequence);
      while (loadHostFlag(localFlags + channel * kLocalStages + 3 + step) !=
             sequence) {
      }
    }
    channelSync(channel);
    copyHostGpuDevice(data + (size_t)recvChunk * chunk + subOffset, recvSlot,
                      recvLen, tid, nth);
    channelSync(channel);
    if (tid == 0) {
      __threadfence_system();
      storeHostFlag(localAcks + channel * kLocalStages + 3 + step, sequence);
      while (loadHostFlag(successorAcks + channel * kLocalStages + 3 + step) !=
             sequence) {
      }
    }
    channelSync(channel);
  }
}

__global__ void gpuToHost(float* host, float const* gpu, size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  if (((uintptr_t)host & 15) || ((uintptr_t)gpu & 15)) {
    for (size_t i = tid; i < count; i += stride) host[i] = gpu[i];
    __threadfence_system();
    return;
  }
  size_t n4 = count / 4;
  float4* host4 = reinterpret_cast<float4*>(host);
  float4 const* gpu4 = reinterpret_cast<float4 const*>(gpu);
  for (size_t i = tid; i < n4; i += stride) host4[i] = gpu4[i];
  for (size_t i = n4 * 4 + tid; i < count; i += stride) host[i] = gpu[i];
  __threadfence_system();
}

__global__ void hostToGpu(float* gpu, float const* host, size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  float const volatile* source = host;
  for (size_t i = tid; i < count; i += stride) gpu[i] = source[i];
}

__global__ void hostAddToGpu(float* gpu, float const* host, size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  float const volatile* source = host;
  for (size_t i = tid; i < count; i += stride) gpu[i] += source[i];
}

__global__ void gpuAddGpu(float* dst, float const* src, size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = tid; i < count; i += stride) dst[i] += src[i];
}

struct Config {
  size_t minBytes = 1UL << 10;
  size_t maxBytes = 1UL << 30;
  int warmup = 5;
  int iters = 10;
  int blocks = 16;
  int threads = 256;
  bool check = true;
  bool persistent = false;
  int channels = 2;
};

static int ringPosition(int rank) {
  constexpr int positions[4] = {0, 1, 3, 2};
  return positions[rank];
}
static int ringRank(int position) {
  constexpr int ranks[4] = {0, 1, 3, 2};
  return ranks[(position + 4) % 4];
}

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  int rank = 0, world = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
  if (world != kWorld) MPI_Abort(MPI_COMM_WORLD, 2);
  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                               MPI_INFO_NULL, &localComm));
  int localRank = 0, localSize = 0;
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  MPICHECK(MPI_Comm_size(localComm, &localSize));
  if (localSize != kLocalRanks) MPI_Abort(MPI_COMM_WORLD, 2);
  CUDACHECK(cudaSetDevice(localRank));

  Config cfg;
  try {
    for (int i = 1; i < argc; i++) {
      std::string key = argv[i];
      auto next = [&]() {
        if (++i >= argc) throw std::runtime_error("missing " + key);
        return argv[i];
      };
      if (key == "-b")
        cfg.minBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-e")
        cfg.maxBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-w")
        cfg.warmup = std::atoi(next());
      else if (key == "-n")
        cfg.iters = std::atoi(next());
      else if (key == "-B")
        cfg.blocks = std::atoi(next());
      else if (key == "-T")
        cfg.threads = std::atoi(next());
      else if (key == "-c")
        cfg.check = std::atoi(next()) != 0;
      else if (key == "--persistent")
        cfg.persistent = std::atoi(next()) != 0;
      else if (key == "-C")
        cfg.channels = std::atoi(next());
      else
        throw std::runtime_error("unknown option " + key);
    }
  } catch (std::exception const& error) {
    if (rank == 0) std::fprintf(stderr, "%s\n", error.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  char* device = nullptr;
  CUDACHECK(cudaMalloc(&device, cfg.maxBytes));
  if (cfg.channels < 1 || cfg.channels > kMaxChannels)
    MPI_Abort(MPI_COMM_WORLD, 2);
  size_t maxChunkBytes = (cfg.maxBytes + kLocalRanks - 1) / kLocalRanks + 512;
  size_t maxChunkElems = maxChunkBytes / sizeof(float);
  size_t channelStride =
      ((maxChunkElems + cfg.channels - 1) / cfg.channels + 31) & ~(size_t)31;
  size_t payloadBytes = channelStride * cfg.channels * sizeof(float);
  size_t hostSlotBytes = payloadBytes + 4096;
  float* gpuScratch = nullptr;
  CUDACHECK(cudaMalloc(&gpuScratch, maxChunkBytes));
  float *hostSend = nullptr, *hostRecv = nullptr, *hostSendDev = nullptr,
        *hostRecvDev = nullptr;
  CUDACHECK(cudaHostAlloc(&hostSend, hostSlotBytes,
                          cudaHostAllocPortable | cudaHostAllocMapped));
  CUDACHECK(cudaHostAlloc(&hostRecv, hostSlotBytes,
                          cudaHostAllocPortable | cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&hostSendDev, hostSend, 0));
  CUDACHECK(cudaHostGetDevicePointer(&hostRecvDev, hostRecv, 0));
  size_t sharedSlotBytes = (hostSlotBytes + 4095) & ~(size_t)4095;
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaStream_t channelStreams[kMaxChannels]{};
  for (int channel = 0; channel < cfg.channels; channel++)
    CUDACHECK(cudaStreamCreateWithFlags(&channelStreams[channel],
                                        cudaStreamNonBlocking));
  std::vector<float> input, output;

  int position = ringPosition(localRank);
  int successor = ringRank(position + 1);
  int predecessor = ringRank(position - 1);
  int crossRank = rank ^ kLocalRanks;
  uint64_t persistentSequence = 1;
  size_t localFlagOffset = sharedSlotBytes - 4096;
  size_t crossFlagOffset = hostSlotBytes - 4096;
  char ownPath[128], predecessorPath[128];
  std::snprintf(ownPath, sizeof(ownPath), "/uccl_lite-hier-%d", localRank);
  std::snprintf(predecessorPath, sizeof(predecessorPath), "/uccl_lite-hier-%d",
                predecessor);
  shm_unlink(ownPath);
  int ownFd = shm_open(ownPath, O_CREAT | O_RDWR, 0600);
  if (ownFd < 0 || ftruncate(ownFd, sharedSlotBytes) != 0)
    MPI_Abort(MPI_COMM_WORLD, 2);
  float* ownSendSlot = reinterpret_cast<float*>(mmap(
      nullptr, sharedSlotBytes, PROT_READ | PROT_WRITE, MAP_SHARED, ownFd, 0));
  if (ownSendSlot == MAP_FAILED) MPI_Abort(MPI_COMM_WORLD, 2);
  MPICHECK(MPI_Barrier(localComm));
  int predecessorFd = shm_open(predecessorPath, O_RDWR, 0600);
  if (predecessorFd < 0) MPI_Abort(MPI_COMM_WORLD, 2);
  float* predecessorSendSlot = reinterpret_cast<float*>(
      mmap(nullptr, sharedSlotBytes, PROT_READ | PROT_WRITE, MAP_SHARED,
           predecessorFd, 0));
  if (predecessorSendSlot == MAP_FAILED) MPI_Abort(MPI_COMM_WORLD, 2);
  CUDACHECK(
      cudaHostRegister(ownSendSlot, sharedSlotBytes,
                       cudaHostRegisterPortable | cudaHostRegisterMapped));
  CUDACHECK(
      cudaHostRegister(predecessorSendSlot, sharedSlotBytes,
                       cudaHostRegisterPortable | cudaHostRegisterMapped));
  float *localInboxDev = nullptr, *successorInboxDev = nullptr;
  CUDACHECK(cudaHostGetDevicePointer(&localInboxDev, predecessorSendSlot, 0));
  CUDACHECK(cudaHostGetDevicePointer(&successorInboxDev, ownSendSlot, 0));
  std::memset(ownSendSlot, 0, sharedSlotBytes);
  std::memset(hostSend, 0, hostSlotBytes);
  std::memset(hostRecv, 0, hostSlotBytes);
  MPICHECK(MPI_Barrier(localComm));

  auto fill = [&](size_t count) {
    input.resize(count);
    for (size_t i = 0; i < count; i++)
      input[i] = (float)(rank + 1) * (float)(1 + i % 97);
    CUDACHECK(cudaMemcpy(device, input.data(), count * sizeof(float),
                         cudaMemcpyHostToDevice));
  };

  auto launch = [&](size_t count) {
    size_t chunk = ((count + kLocalRanks - 1) / kLocalRanks + 31) & ~(size_t)31;
    auto length = [&](int chunkIndex) {
      size_t offset = (size_t)chunkIndex * chunk;
      return offset >= count ? (size_t)0 : std::min(chunk, count - offset);
    };
    if (cfg.persistent) {
      uint64_t sequence = persistentSequence++;
      for (int channel = 0; channel < cfg.channels; channel++) {
        persistentHostChannel<<<cfg.blocks, cfg.threads, 0,
                                channelStreams[channel]>>>(
            reinterpret_cast<float*>(device), count, localInboxDev,
            successorInboxDev, channelStride, localFlagOffset, hostSendDev,
            hostRecvDev, crossFlagOffset, position, channel, cfg.channels,
            sequence);
      }
      uint64_t* sendFlags = reinterpret_cast<uint64_t*>(
          reinterpret_cast<char*>(hostSend) + crossFlagOffset);
      uint64_t* recvFlags = reinterpret_cast<uint64_t*>(
          reinterpret_cast<char*>(hostRecv) + crossFlagOffset);
      std::vector<MPI_Request> requests(cfg.channels * 2);
      int ownChunk = (position + 1) % kLocalRanks;
      size_t channelElems =
          ((chunk + cfg.channels - 1) / cfg.channels + 31) & ~(size_t)31;
      for (int channel = 0; channel < cfg.channels; channel++) {
        while (__atomic_load_n(sendFlags + channel, __ATOMIC_ACQUIRE) !=
               sequence) {
        }
        size_t subOffset = (size_t)channel * channelElems;
        size_t ownBase = (size_t)ownChunk * chunk + subOffset;
        size_t ownLen = ownBase >= count || subOffset >= chunk
                            ? 0
                            : std::min(channelElems, std::min(chunk - subOffset,
                                                              count - ownBase));
        float* sendPtr = hostSend + (size_t)channel * channelStride;
        float* recvPtr = hostRecv + (size_t)channel * channelStride;
        MPICHECK(MPI_Irecv(recvPtr, (int)ownLen, MPI_FLOAT, crossRank,
                           900 + channel, MPI_COMM_WORLD,
                           &requests[channel * 2]));
        MPICHECK(MPI_Isend(sendPtr, (int)ownLen, MPI_FLOAT, crossRank,
                           900 + channel, MPI_COMM_WORLD,
                           &requests[channel * 2 + 1]));
      }
      MPICHECK(
          MPI_Waitall(cfg.channels * 2, requests.data(), MPI_STATUSES_IGNORE));
      for (int channel = 0; channel < cfg.channels; channel++)
        __atomic_store_n(recvFlags + channel, sequence, __ATOMIC_RELEASE);
      for (int channel = 0; channel < cfg.channels; channel++)
        CUDACHECK(cudaStreamSynchronize(channelStreams[channel]));
      return;
    }
    for (int step = 0; step < kLocalRanks - 1; step++) {
      int sendChunk = (position - step + kLocalRanks) % kLocalRanks;
      int recvChunk = (sendChunk - 1 + kLocalRanks) % kLocalRanks;
      size_t sendLen = length(sendChunk), recvLen = length(recvChunk);
      CUDACHECK(cudaMemcpy(
          hostSend,
          reinterpret_cast<float*>(device) + (size_t)sendChunk * chunk,
          sendLen * sizeof(float), cudaMemcpyDeviceToHost));
      MPICHECK(MPI_Sendrecv(hostSend, (int)sendLen, MPI_FLOAT, successor,
                            600 + step, hostRecv, (int)recvLen, MPI_FLOAT,
                            predecessor, 600 + step, localComm,
                            MPI_STATUS_IGNORE));
      CUDACHECK(cudaMemcpy(gpuScratch, hostRecv, recvLen * sizeof(float),
                           cudaMemcpyHostToDevice));
      gpuAddGpu<<<cfg.blocks, cfg.threads, 0, stream>>>(
          reinterpret_cast<float*>(device) + (size_t)recvChunk * chunk,
          gpuScratch, recvLen);
      CUDACHECK(cudaStreamSynchronize(stream));
    }

    int ownChunk = (position + 1) % kLocalRanks;
    size_t ownLen = length(ownChunk);
    CUDACHECK(cudaMemcpy(
        hostSend, reinterpret_cast<float*>(device) + (size_t)ownChunk * chunk,
        ownLen * sizeof(float), cudaMemcpyDeviceToHost));
    MPICHECK(MPI_Sendrecv(hostSend, (int)ownLen, MPI_FLOAT, crossRank, 700,
                          hostRecv, (int)ownLen, MPI_FLOAT, crossRank, 700,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    CUDACHECK(cudaMemcpy(gpuScratch, hostRecv, ownLen * sizeof(float),
                         cudaMemcpyHostToDevice));
    gpuAddGpu<<<cfg.blocks, cfg.threads, 0, stream>>>(
        reinterpret_cast<float*>(device) + (size_t)ownChunk * chunk, gpuScratch,
        ownLen);
    CUDACHECK(cudaStreamSynchronize(stream));

    for (int step = 0; step < kLocalRanks - 1; step++) {
      int sendChunk = (position + 1 - step + kLocalRanks) % kLocalRanks;
      int recvChunk = (sendChunk - 1 + kLocalRanks) % kLocalRanks;
      size_t sendLen = length(sendChunk), recvLen = length(recvChunk);
      CUDACHECK(cudaMemcpy(
          hostSend,
          reinterpret_cast<float*>(device) + (size_t)sendChunk * chunk,
          sendLen * sizeof(float), cudaMemcpyDeviceToHost));
      MPICHECK(MPI_Sendrecv(hostSend, (int)sendLen, MPI_FLOAT, successor,
                            800 + step, hostRecv, (int)recvLen, MPI_FLOAT,
                            predecessor, 800 + step, localComm,
                            MPI_STATUS_IGNORE));
      CUDACHECK(cudaMemcpy(
          reinterpret_cast<float*>(device) + (size_t)recvChunk * chunk,
          hostRecv, recvLen * sizeof(float), cudaMemcpyHostToDevice));
    }
  };

  auto verify = [&](size_t count) {
    output.resize(count);
    CUDACHECK(cudaMemcpy(output.data(), device, count * sizeof(float),
                         cudaMemcpyDeviceToHost));
    double rankSum = (double)world * (world + 1) / 2.0, worst = 0.0;
    size_t firstBad = count;
    for (size_t i = 0; i < count; i++) {
      double expected = rankSum * (double)(1 + i % 97);
      double error = std::fabs((double)output[i] - expected) / expected;
      if (error > 1e-5 && firstBad == count) firstBad = i;
      worst = std::max(worst, error);
    }
    if (firstBad != count)
      std::fprintf(stderr, "rank %d first bad %zu got %.1f expected %.1f\n",
                   rank, firstBad, (double)output[firstBad],
                   rankSum * (double)(1 + firstBad % 97));
    return worst;
  };

  if (rank == 0) {
    std::printf(
        "# UCCL Lite host-staged hierarchical AllReduce; no GPU P2P/GDR\n");
    std::printf(
        "# local ring 0->1->3->2; GPU reduction over mapped host slots\n");
    std::printf("# %12s %12s %10s %10s %10s %10s\n", "size(B)", "count",
                "time(us)", "algbw", "busbw", "maxrelerr");
  }
  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    bytes = (bytes + 511) & ~(size_t)511;
    size_t count = bytes / sizeof(float);
    double error = -1.0;
    if (cfg.check) {
      fill(count);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      launch(count);
      error = verify(count);
    }
    fill(count);
    for (int i = 0; i < cfg.warmup; i++) launch(count);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; i++) launch(count);
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - begin).count() /
                cfg.iters;
    double worstUs = 0.0, worstError = 0.0;
    MPICHECK(
        MPI_Reduce(&us, &worstUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&error, &worstError, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    if (rank == 0) {
      double algbw = (double)bytes / worstUs / 1000.0;
      std::printf("  %12zu %12zu %10.2f %10.2f %10.2f %10.1e\n", bytes, count,
                  worstUs, algbw, algbw * 1.75, worstError);
    }
    if (bytes > cfg.maxBytes / 2) break;
  }

  CUDACHECK(cudaStreamDestroy(stream));
  for (int channel = 0; channel < cfg.channels; channel++)
    CUDACHECK(cudaStreamDestroy(channelStreams[channel]));
  CUDACHECK(cudaHostUnregister(predecessorSendSlot));
  CUDACHECK(cudaHostUnregister(ownSendSlot));
  munmap(predecessorSendSlot, sharedSlotBytes);
  munmap(ownSendSlot, sharedSlotBytes);
  close(predecessorFd);
  close(ownFd);
  MPICHECK(MPI_Barrier(localComm));
  shm_unlink(ownPath);
  CUDACHECK(cudaFreeHost(hostSend));
  CUDACHECK(cudaFreeHost(hostRecv));
  CUDACHECK(cudaFree(gpuScratch));
  CUDACHECK(cudaFree(device));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}

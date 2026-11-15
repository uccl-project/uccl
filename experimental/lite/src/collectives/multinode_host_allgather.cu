// Experimental two-node host-RDMA AllGather.
// One MPI process owns the two GPUs in each NUMA group. Each source group
// uploads two shards into NUMA-local pinned host memory and sends the same
// packet to both remote NUMA groups. Each destination group also receives the
// other local group's packet, then its two GPUs read the shared host packets.
// Three slots overlap upload(j), network(j-1), and multicast(j-2).
#include <cuda_runtime.h>
#ifdef UCCL_LITE_HOST_RDMA
#include <mscclpp/core.hpp>
#endif
#include <linux/mempolicy.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <mpi.h>
#include <nccl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
constexpr int kNodes = 2;
constexpr int kGroups = 2;
constexpr int kGpusPerGroup = 2;
constexpr int kRanks = kNodes * kGroups;
constexpr int kGpus = kRanks * kGpusPerGroup;
constexpr int kSlots = 3;
constexpr int kTagRemote = 510;
constexpr int kTagLocal = 511;
constexpr int kTagRelay = 512;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void check(int result) {
  if (result != MPI_SUCCESS) {
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(result, message, &length);
    throw std::runtime_error(std::string(message, length));
  }
}
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
struct Config {
  size_t begin = 1UL << 10;
  size_t end = 1UL << 30;
  size_t chunkBytes = 1UL << 20;
  int warmups = 20;
  int iterations = 10;
  int repeats = 3;
  int consumeThreads = 256;
  int delayConsumer = -1;
  bool dmaMulticast = false;
  bool smUpload = false;
  bool gpuWorkers = false;
  bool earlyConsume = false;
  bool readyProgress = false;
  bool leanContext = false;
  bool sharedLocal = false;
  bool pairedNetwork = false;
  bool directRdma = false;
  int localInputs = 0;  // 0: all host, 1: own GPU, 2: own NUMA GPU pair.
  bool compareNccl = false;
};
Config parse(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    if (option == "--help") {
      std::puts(
          "Usage: uccl_lite_multinode_host_allgather "
          "[-b BYTES] [-e BYTES] [-k CHUNK_BYTES] "
          "[-w WARMUPS] [-n ITERATIONS] [-r REPEATS] "
          "[--download sm|dma] [--upload dma|sm] [--consume-threads "
          "128|256|512] [--delay-consumer-rank RANK] [--gpu-workers 0|1] "
          "[--early-consume 0|1] [--progress steps|ready] [--lean-context 0|1] "
          "[--local-exchange copy|shared] [--network multicast|paired|rdma] "
          "[--local-input "
          "host|self|pair] [--nccl 0|1]");
      std::exit(0);
    }
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    char const* value = argv[++i];
    if (option == "-b") cfg.begin = std::strtoull(value, nullptr, 0);
    else if (option == "-e") cfg.end = std::strtoull(value, nullptr, 0);
    else if (option == "-k") cfg.chunkBytes = std::strtoull(value, nullptr, 0);
    else if (option == "-w") cfg.warmups = std::atoi(value);
    else if (option == "-n") cfg.iterations = std::atoi(value);
    else if (option == "-r") cfg.repeats = std::atoi(value);
    else if (option == "--download") {
      if (std::strcmp(value, "sm") == 0) cfg.dmaMulticast = false;
      else if (std::strcmp(value, "dma") == 0) cfg.dmaMulticast = true;
      else throw std::invalid_argument("invalid download method");
    } else if (option == "--upload") {
      if (std::strcmp(value, "sm") == 0)
        cfg.smUpload = true;
      else if (std::strcmp(value, "dma") == 0)
        cfg.smUpload = false;
      else
        throw std::invalid_argument("invalid upload method");
    } else if (option == "--network") {
      cfg.directRdma = cfg.pairedNetwork = false;
      if (std::strcmp(value, "paired") == 0)
        cfg.pairedNetwork = true;
      else if (std::strcmp(value, "multicast") == 0)
        cfg.pairedNetwork = false;
      else if (std::strcmp(value, "rdma") == 0) {
        cfg.directRdma = true;
        cfg.pairedNetwork = false;
      } else
        throw std::invalid_argument("invalid host network schedule");
    } else if (option == "--local-input") {
      if (std::strcmp(value, "host") == 0)
        cfg.localInputs = 0;
      else if (std::strcmp(value, "self") == 0)
        cfg.localInputs = 1;
      else if (std::strcmp(value, "pair") == 0)
        cfg.localInputs = 2;
      else
        throw std::invalid_argument("invalid local input source");
    } else if (option == "--early-consume")
      cfg.earlyConsume = std::atoi(value) != 0;
    else if (option == "--progress") {
      if (std::strcmp(value, "ready") == 0)
        cfg.readyProgress = true;
      else if (std::strcmp(value, "steps") == 0)
        cfg.readyProgress = false;
      else
        throw std::invalid_argument("invalid progress schedule");
    } else if (option == "--local-exchange") {
      if (std::strcmp(value, "shared") == 0)
        cfg.sharedLocal = true;
      else if (std::strcmp(value, "copy") == 0)
        cfg.sharedLocal = false;
      else
        throw std::invalid_argument("invalid local exchange");
    } else if (option == "--delay-consumer-rank")
      cfg.delayConsumer = std::atoi(value);
    else if (option == "--consume-threads")
      cfg.consumeThreads = std::atoi(value);
    else if (option == "--lean-context")
      cfg.leanContext = std::atoi(value) != 0;
    else if (option == "--gpu-workers")
      cfg.gpuWorkers = std::atoi(value) != 0;
    else if (option == "--nccl")
      cfg.compareNccl = std::atoi(value) != 0;
    else throw std::invalid_argument("unknown option " + option);
  }
  if (cfg.begin < 1024 || cfg.begin > cfg.end || cfg.end > (1UL << 30) ||
      cfg.begin % 128 || cfg.end % 128 || cfg.chunkBytes < 128 ||
      cfg.chunkBytes % 128 || cfg.chunkBytes > (16UL << 20) ||
      cfg.warmups < 5 || cfg.iterations < 10 || cfg.repeats < 1 ||
      cfg.delayConsumer < -1 || cfg.delayConsumer >= kGpus ||
      (cfg.consumeThreads != 128 && cfg.consumeThreads != 256 &&
       cfg.consumeThreads != 512))
    throw std::invalid_argument("invalid size or iteration settings");
#ifndef UCCL_LITE_HOST_RDMA
  if (cfg.directRdma)
    throw std::invalid_argument(
        "RDMA mode requires make multinode_host_rdma_allgather");
#endif
  return cfg;
}
void bindGroupBeforeMpi() {
  char const* text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int group = text ? std::atoi(text) : 0;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  for (int cpu = group; cpu < 48; cpu += 2) CPU_SET(cpu, &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity))
    throw std::runtime_error("CPU affinity failed");
  if (!std::getenv("UCX_NET_DEVICES"))
    setenv("UCX_NET_DEVICES", group == 0 ? "mlx5_0:1" : "mlx5_1:1", 1);
  if (!std::getenv("UCX_TLS"))
    setenv("UCX_TLS", "rc_mlx5,ud_mlx5,sm,self", 1);
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
    auto result = cudaHostRegister(address, bytes,
                                   cudaHostRegisterPortable |
                                       cudaHostRegisterMapped);
    if (result != cudaSuccess) {
      if (ownsMapping) munmap(address, bytes);
      check(result);
    }
    host = static_cast<char*>(address);
    for (int i = 0; i < kGpusPerGroup; ++i) {
      check(cudaSetDevice(devices[i]));
      check(cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&mapped[i]), host, 0));
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
struct Slot {
  HostBlock send;
  HostBlock local;
  std::array<HostBlock, kGroups> remote;
  std::array<MPI_Request, 3> receive{MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                     MPI_REQUEST_NULL};
  std::array<MPI_Request, 3> outgoing{MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                      MPI_REQUEST_NULL};
  int piece = -1;
  size_t length = 0;
  size_t shardBytes = 0;
  uint64_t generation = 0;
  bool active = false;
  bool sent = false;
  bool multicast = false;
  bool localConsumed = false;
  bool relaySent = false;
  bool ackSent = false;
  std::array<MPI_Request, 2> sharedRequests{MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  void initialize(size_t capacity, int node,
                  std::array<int, kGpusPerGroup> const& devices,
                  void* sharedSend = nullptr, void* sharedPeer = nullptr) {
    send.initialize(capacity, node, devices, sharedSend);
    local.initialize(capacity, node, devices, sharedPeer);
    for (auto& block : remote) block.initialize(capacity, node, devices);
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
      check(cudaEventCreateWithFlags(&uploadDone[slot], cudaEventDisableTiming));
      check(cudaEventCreateWithFlags(&consumeDone[slot], cudaEventDisableTiming));
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
__global__ void fillInput(float* input, size_t count, int rank, int generation) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  float factor = float(rank + 1 + generation * kGpus);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride)
    input[i] = factor * float(1 + i % 97);
}
// SM producer can publish the host packet and the GPU's own output shard in
// one pass. All CTA writers rendezvous before system visibility is established.
__global__ void uploadPacket(uint4 const* input, uint4* host, uint4* own,
                             size_t vectors) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors;
       i += size_t(gridDim.x) * blockDim.x) {
    uint4 value = input[i];
    host[i] = value;
    if (own) own[i] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
}
__global__ void delayConsumerKernel() {
  unsigned long long start = clock64();
  while (clock64() - start < 200000) __nanosleep(64);
}
struct Sources {
  uint4 const* pointer[kGpus];
};
__global__ void multicastPacket(Sources sources, uint4* output,
                                size_t shardVectors, size_t pieceOffset,
                                size_t pieceVectors, int firstSource,
                                int skipSource) {
  int source = blockIdx.y + firstSource;
  if (skipSource >= firstSource && source >= skipSource) ++source;
  auto const* input = sources.pointer[source];
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < pieceVectors; i += size_t(gridDim.x) * blockDim.x)
    output[size_t(source) * shardVectors + pieceOffset + i] = input[i];
}
__global__ void checkOutput(float const* output, unsigned int* wrong,
                            size_t count, size_t shard, int generation) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    int source = int(i / shard);
    size_t index = i % shard;
    float expected = float(source + 1 + generation * kGpus) *
                     float(1 + index % 97);
    if (!(fabsf(output[i] - expected) <= 1e-5f)) atomicAdd(wrong, 1);
  }
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}
}  // namespace

#ifdef UCCL_LITE_HOST_RDMA
// Host-only RDMA: slot credits precede upload, data precedes its done flag on
// the same RC connection. Retiring a slot drains both send queues before its
// token or payload can change. Receiver credits guard GPU-reader lifetimes.
class HostRdma {
 public:
  HostRdma(std::array<std::unique_ptr<Slot>, kSlots>& slots, int rank,
           int group, std::array<int, kGpusPerGroup> const& physical)
      : slots_(slots), group_(group) {
    check(cudaSetDevice(physical[0]));
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, kRanks);
    mscclpp::UniqueId id{};
    if (rank == 0) id = mscclpp::TcpBootstrap::createUniqueId();
    check(MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD));
    bootstrap->initialize(id, 120);
    comm_ = std::make_unique<mscclpp::Communicator>(bootstrap);
    auto transport =
        static_cast<mscclpp::Transport>(int(mscclpp::Transport::IB0) + group);
    flags_.initialize(4096, group, physical);
    localFlags_ = comm_->registerMemory(flags_.host, flags_.bytes, transport);
    for (int slot = 0; slot < kSlots; ++slot) {
      send_[slot] = comm_->registerMemory(slots_[slot]->send.host,
                                          slots_[slot]->send.bytes, transport);
      for (int peer = 0; peer < kGroups; ++peer)
        receive_[peer][slot] =
            comm_->registerMemory(slots_[slot]->remote[peer].host,
                                  slots_[slot]->remote[peer].bytes, transport);
    }
    int remoteBase = (1 - rank / kGroups) * kGroups;
    std::array<std::shared_future<mscclpp::Connection>, kGroups> connections;
    std::array<std::shared_future<mscclpp::RegisteredMemory>, kGroups>
        flagFutures;
    std::array<
        std::array<std::shared_future<mscclpp::RegisteredMemory>, kSlots>,
        kGroups>
        dataFutures;
    for (int peer = 0; peer < kGroups; ++peer) {
      int other = remoteBase + peer;
      mscclpp::EndpointConfig endpoint(transport, mscclpp::DeviceType::CPU);
      endpoint.ib.mode = mscclpp::EndpointConfig::Ib::Mode::Host;
      connections[peer] = comm_->connect(endpoint, other, 610);
      comm_->sendMemory(localFlags_, other, 611);
      flagFutures[peer] = comm_->recvMemory(other, 611);
      for (int slot = 0; slot < kSlots; ++slot) {
        comm_->sendMemory(receive_[peer][slot], other, 612 + slot);
        dataFutures[peer][slot] = comm_->recvMemory(other, 612 + slot);
      }
    }
    for (int peer = 0; peer < kGroups; ++peer) {
      connections_[peer] = connections[peer].get();
      remoteFlags_[peer] = flagFutures[peer].get();
      for (int slot = 0; slot < kSlots; ++slot)
        remoteData_[peer][slot] = dataFutures[peer][slot].get();
    }
    check(MPI_Barrier(MPI_COMM_WORLD));
  }
  void grant(int slot, uint64_t epoch) {
    for (int peer = 0; peer < kGroups; ++peer)
      connections_[peer].updateAndSync(remoteFlags_[peer],
                                       flagOffset(false, group_, slot),
                                       &readySent_[peer][slot], epoch);
  }
  bool credits(int slot, uint64_t epoch) const {
    return flagsReady(false, slot, epoch);
  }
  bool received(int slot, uint64_t epoch) const {
    return flagsReady(true, slot, epoch);
  }
  void send(int slot, size_t bytes) {
    for (int peer = 0; peer < kGroups; ++peer) {
      connections_[peer].write(remoteData_[peer][slot], 0, send_[slot], 0,
                               bytes);
      // Native host-memory atomics preserve notification ordering even when
      // the data MR enables relaxed write ordering. No GPU atomics are used.
      connections_[peer].updateAndSync(
          remoteFlags_[peer], flagOffset(true, group_, slot),
          &doneSent_[peer][slot], slots_[slot]->generation);
    }
  }
  void flush() {
    for (auto& connection : connections_) connection.flush();
  }

 private:
  static size_t flagOffset(bool done, int peer, int slot) {
    return size_t((int(done) * kGroups + peer) * kSlots + slot) * 64;
  }
  bool flagsReady(bool done, int slot, uint64_t epoch) const {
    for (int peer = 0; peer < kGroups; ++peer) {
      auto const* flag = reinterpret_cast<uint64_t const*>(
          flags_.host + flagOffset(done, peer, slot));
      if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != epoch) return false;
    }
    return true;
  }
  std::array<std::unique_ptr<Slot>, kSlots>& slots_;
  int group_;
  HostBlock flags_;
  std::unique_ptr<mscclpp::Communicator> comm_;
  mscclpp::RegisteredMemory localFlags_;
  std::array<mscclpp::RegisteredMemory, kSlots> send_;
  std::array<std::array<mscclpp::RegisteredMemory, kSlots>, kGroups> receive_,
      remoteData_;
  std::array<mscclpp::RegisteredMemory, kGroups> remoteFlags_;
  std::array<mscclpp::Connection, kGroups> connections_;
  std::array<std::array<uint64_t, kSlots>, kGroups> readySent_{}, doneSent_{};
};
#else
class HostRdma {
 public:
  template <class... Args>
  HostRdma(Args&&...) {
    throw std::runtime_error("build multinode_host_rdma_allgather");
  }
  void grant(int, uint64_t) {}
  bool credits(int, uint64_t) const { return false; }
  bool received(int, uint64_t) const { return false; }
  void send(int, size_t) {}
  void flush() {}
};
#endif

// Per-GPU CUDA progress stays on a fixed context/NUMA CPU. MPI remains on
// the main thread. Epoch counters publish slot metadata and completion, so
// slot reuse cannot race with event polling or DMA on either device.
class GpuWorker {
 public:
  GpuWorker(Device& device, int cpu, std::function<void(int)> upload,
            std::function<void(int, int)> consume, bool split)
      : consumeSteps_(split ? 2 : 1) {
    for (int i = 0; i < kSlots; ++i) {
      uploadRequest_[i] = consumeRequest_[i] = 0;
      uploaded_[i] = consumed_[i] = 0;
    }
    thread_ = std::thread([&, cpu, upload, consume] {
      try {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set))
          throw std::runtime_error("GPU worker CPU affinity failed");
        check(cudaSetDevice(device.physical));
        std::array<uint64_t, kSlots> up{}, down{};
        ready_.store(true, std::memory_order_release);
        while (!stop_.load(std::memory_order_acquire)) {
          for (int i = 0; i < kSlots; ++i) {
            uint64_t u = uploadRequest_[i].load(std::memory_order_acquire);
            if (u != up[i]) {
              upload(i);
              up[i] = u;
            }
            if (up[i] != uploaded_[i].load(std::memory_order_relaxed)) {
              auto status = cudaEventQuery(device.uploadDone[i]);
              if (status == cudaSuccess)
                uploaded_[i].store(up[i], std::memory_order_release);
              else if (status != cudaErrorNotReady)
                check(status);
            }
            uint64_t d = consumeRequest_[i].load(std::memory_order_acquire);
            if (d != down[i]) {
              // Preserve both phase submissions if the producer posts them
              // before this worker wakes up. No phase may be overwritten.
              for (uint64_t next = down[i] + 1; next <= d; ++next)
                consume(i, consumeSteps_ == 2 ? int((next - 1) % 2) : -1);
              down[i] = d;
            }
            if (down[i] != consumed_[i].load(std::memory_order_relaxed)) {
              auto status = cudaEventQuery(device.consumeDone[i]);
              if (status == cudaSuccess)
                consumed_[i].store(down[i], std::memory_order_release);
              else if (status != cudaErrorNotReady)
                check(status);
            }
          }
          std::this_thread::yield();
        }
      } catch (...) {
        failure_ = std::current_exception();
        failed_.store(true, std::memory_order_release);
      }
    });
    while (!ready_.load(std::memory_order_acquire) &&
           !failed_.load(std::memory_order_acquire))
      std::this_thread::yield();
    if (failed_.load(std::memory_order_acquire)) {
      thread_.join();
      std::rethrow_exception(failure_);
    }
  }
  ~GpuWorker() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
  }
  void requestUpload(int slot, uint64_t epoch) {
    uploadRequest_[slot].store(epoch, std::memory_order_release);
  }
  void requestConsume(int slot, uint64_t epoch, int phase) {
    uint64_t step =
        (epoch - 1) * consumeSteps_ + (phase < 0 ? consumeSteps_ : phase + 1);
    consumeRequest_[slot].store(step, std::memory_order_release);
  }
  bool completed(int slot, uint64_t epoch, bool consume) const {
    return (consume ? consumed_[slot] : uploaded_[slot])
               .load(std::memory_order_acquire) ==
           epoch * (consume ? consumeSteps_ : 1);
  }
  void checkFailure() const {
    if (failed_.load(std::memory_order_acquire))
      std::rethrow_exception(failure_);
  }

 private:
  std::array<std::atomic<uint64_t>, kSlots> uploadRequest_, consumeRequest_,
      uploaded_, consumed_;
  std::atomic<bool> ready_{false}, stop_{false}, failed_{false};
  int consumeSteps_;
  std::exception_ptr failure_;
  std::thread thread_;
};

int main(int argc, char** argv) {
  bool mpiReady = false;
  try {
    Config cfg = parse(argc, argv);
    bindGroupBeforeMpi();
    check(MPI_Init(&argc, &argv));
    mpiReady = true;
    int rank = -1, world = 0;
    check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    check(MPI_Comm_size(MPI_COMM_WORLD, &world));
    MPI_Comm localComm;
    check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                              MPI_INFO_NULL, &localComm));
    int localRank = -1, localSize = 0;
    check(MPI_Comm_rank(localComm, &localRank));
    check(MPI_Comm_size(localComm, &localSize));
    if (world != kRanks || localSize != kGroups ||
        localRank != rank % kGroups)
      throw std::runtime_error("requires two nodes, two NUMA groups per node");
    int node = rank / kGroups;
    int remoteBase = (1 - node) * kGroups;
    int localPeer = rank ^ 1;
    std::array<int, kGpusPerGroup> physical{localRank * 2, localRank * 2 + 1};
    int available = 0;
    check(cudaGetDeviceCount(&available));
    if (available < 4) throw std::runtime_error("requires four local GPUs");
    if (cfg.localInputs == 2) {
      for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
        int access = 0;
        check(
            cudaDeviceCanAccessPeer(&access, physical[gpu], physical[gpu ^ 1]));
        if (!access)
          throw std::runtime_error("local GPU pair requires peer access");
        check(cudaSetDevice(physical[gpu]));
        auto status = cudaDeviceEnablePeerAccess(physical[gpu ^ 1], 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled)
          cudaGetLastError();
        else
          check(status);
      }
    }
    std::array<std::unique_ptr<Device>, kGpusPerGroup> devices;
    for (int gpu = 0; gpu < kGpusPerGroup; ++gpu)
      devices[gpu] = std::make_unique<Device>(
          physical[gpu], node * 4 + localRank * 2 + gpu,
          cfg.end / kGpus, cfg.end);
    size_t capacity = std::min(cfg.chunkBytes, cfg.end / kGpus);
    SharedPackets sharedPackets;
    if (cfg.sharedLocal)
      sharedPackets.initialize(2 * capacity, localRank, localComm);
    std::array<std::unique_ptr<Slot>, kSlots> slots;
    for (int i = 0; i < kSlots; ++i) {
      slots[i] = std::make_unique<Slot>();
      slots[i]->initialize(2 * capacity, localRank, physical,
                           cfg.sharedLocal ? sharedPackets.own[i] : nullptr,
                           cfg.sharedLocal ? sharedPackets.peer[i] : nullptr);
    }
    std::unique_ptr<HostRdma> rdma;
    if (cfg.directRdma)
      rdma = std::make_unique<HostRdma>(slots, rank, localRank, physical);
    std::array<ncclComm_t, kGpusPerGroup> ncclComms{};
    int ncclVersion = 0;
    if (cfg.compareNccl) {
      ncclUniqueId id{};
      if (rank == 0) check(ncclGetUniqueId(&id));
      check(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
      check(ncclGroupStart());
      for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
        check(cudaSetDevice(physical[gpu]));
        check(ncclCommInitRank(&ncclComms[gpu], kGpus, id,
                               devices[gpu]->globalRank));
      }
      check(ncclGroupEnd());
      check(ncclGetVersion(&ncclVersion));
    }
    // Keep MPI/NCCL progress workers on the NUMA node while the routing
    // thread stays on one CPU, avoiding scheduler migration in the hot loop.
    cpu_set_t mainAffinity;
    CPU_ZERO(&mainAffinity);
    CPU_SET(localRank == 0 ? 0 : 1, &mainAffinity);
    if (sched_setaffinity(0, sizeof(mainAffinity), &mainAffinity))
      throw std::runtime_error("routing CPU affinity failed");
    if (rank == 0) {
      std::printf("# two-node host-RDMA AllGather; FP32; 4 GPUs/node; "
                  "2 MPI groups/node; 2 IB rails; chunk=%zu B; slots=%d; "
                  "download=%s\n", cfg.chunkBytes, kSlots,
                  cfg.dmaMulticast ? "dma" : "sm");
      std::printf(
          "# host network=%s local_input=%s gpu_workers=%d early_consume=%d "
          "progress=%s upload=%s lean_context=%d consume_threads=%d "
          "local_exchange=%s delay_rank=%d\n",
          cfg.directRdma      ? "precredited-rdma"
          : cfg.pairedNetwork ? "paired-plus-local-relay"
                              : "remote-multicast",
          cfg.localInputs == 2   ? "pair"
          : cfg.localInputs == 1 ? "self"
                                 : "host",
          int(cfg.gpuWorkers), int(cfg.earlyConsume),
          cfg.readyProgress ? "ready" : "steps", cfg.smUpload ? "sm" : "dma",
          int(cfg.leanContext), cfg.consumeThreads,
          cfg.sharedLocal ? "shared" : "copy", cfg.delayConsumer);
      std::printf("# timing=host enqueue+completion, slowest of 4 groups/8 GPUs; "
                  "warmups=%d iterations=%d repeats=%d; every output checked\n",
                  cfg.warmups, cfg.iterations, cfg.repeats);
      if (cfg.compareNccl)
        std::printf("# paired NCCL version=%d; same two GPU streams and "
                    "complete output check\n", ncclVersion);
      std::puts("collective,bytes,backend,median_us,min_us,max_us");
    }
    auto fill = [&](size_t fullBytes, int generation) {
      size_t shardElems = fullBytes / sizeof(float) / kGpus;
      for (auto const& device : devices) {
        check(cudaSetDevice(device->physical));
        fillInput<<<256, 256, 0, device->upload>>>(
            device->input, shardElems, device->globalRank, generation);
        check(cudaGetLastError());
        check(cudaMemsetAsync(device->output, 0xff, fullBytes,
                              device->consume));
        check(cudaDeviceSynchronize());
      }
    };
    auto verify = [&](size_t fullBytes, int generation) {
      size_t outputElems = fullBytes / sizeof(float);
      size_t shardElems = outputElems / kGpus;
      unsigned int localWrong = 0;
      for (auto const& device : devices) {
        check(cudaSetDevice(device->physical));
        check(cudaMemsetAsync(device->wrong, 0, sizeof(unsigned int),
                              device->consume));
        checkOutput<<<256, 256, 0, device->consume>>>(
            device->output, device->wrong, outputElems, shardElems,
            generation);
        check(cudaGetLastError());
        unsigned int wrong = 0;
        check(cudaMemcpyAsync(&wrong, device->wrong, sizeof(wrong),
                              cudaMemcpyDeviceToHost, device->consume));
        check(cudaStreamSynchronize(device->consume));
        localWrong += wrong;
      }
      unsigned int totalWrong = 0;
      check(MPI_Allreduce(&localWrong, &totalWrong, 1, MPI_UNSIGNED,
                           MPI_SUM, MPI_COMM_WORLD));
      if (totalWrong)
        throw std::runtime_error("AllGather complete output mismatch: " +
                                 std::to_string(totalWrong));
    };
    auto uploadGpu = [&](int slotIndex, int gpu) {
      Slot& slot = *slots[slotIndex];
      Device& device = *devices[gpu];
      size_t offset = size_t(slot.piece) * cfg.chunkBytes;
      if (cfg.smUpload) {
        size_t vectors = slot.length / sizeof(uint4);
        int blocks = int(std::min<size_t>(128, (vectors + 255) / 256));
        auto* own =
            cfg.localInputs
                ? reinterpret_cast<uint4*>(
                      reinterpret_cast<char*>(device.output) +
                      size_t(device.globalRank) * slot.shardBytes + offset)
                : nullptr;
        uploadPacket<<<blocks, 256, 0, device.upload>>>(
            reinterpret_cast<uint4 const*>(
                reinterpret_cast<char*>(device.input) + offset),
            reinterpret_cast<uint4*>(slot.send.mapped[gpu] + gpu * slot.length),
            own, vectors);
        check(cudaGetLastError());
      } else {
        check(cudaMemcpyAsync(slot.send.host + gpu * slot.length,
                              reinterpret_cast<char*>(device.input) + offset,
                              slot.length, cudaMemcpyDeviceToHost,
                              device.upload));
      }
      check(cudaEventRecord(device.uploadDone[slotIndex], device.upload));
    };
    auto consumeGpu = [&](int slotIndex, int gpu, int phase) {
      Slot& slot = *slots[slotIndex];
      size_t piece = size_t(slot.piece);
      size_t shardBytes = slot.shardBytes;
      Device& device = *devices[gpu];
      if (cfg.delayConsumer == device.globalRank && slot.generation % 3 == 1) {
        check(cudaSetDevice(device.physical));
        delayConsumerKernel<<<1, 1, 0, device.consume>>>();
        check(cudaGetLastError());
      }
      Sources source{};
      int firstSource = phase < 0 ? 0 : (phase == 0 ? node : 1 - node) * 4;
      int sourceCount = phase < 0 ? kGpus : 4;
      int skipSource = cfg.smUpload && cfg.localInputs ? device.globalRank : -1;
      bool skipInRange =
          skipSource >= firstSource && skipSource < firstSource + sourceCount;
      for (int rankGpu = firstSource; rankGpu < firstSource + sourceCount;
           ++rankGpu) {
        if (rankGpu == skipSource) continue;
        int process = rankGpu / kGpusPerGroup;
        HostBlock* block = process == rank ? &slot.send
                           : process == localPeer
                               ? &slot.local
                               : &slot.remote[process % kGroups];
        char const* hostSource =
            block->host + size_t(rankGpu % kGpusPerGroup) * slot.length;
        bool gpuSource =
            process == rank &&
            (cfg.localInputs == 2 ||
             (cfg.localInputs == 1 && rankGpu == device.globalRank));
        if (gpuSource)
          hostSource = reinterpret_cast<char const*>(
                           devices[rankGpu % kGpusPerGroup]->input) +
                       piece * cfg.chunkBytes;
        if (cfg.dmaMulticast)
          check(cudaMemcpyAsync(
              reinterpret_cast<char*>(device.output) +
                  size_t(rankGpu) * shardBytes + piece * cfg.chunkBytes,
              hostSource, slot.length,
              gpuSource ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
              device.consume));
        else
          source.pointer[rankGpu] =
              gpuSource ? reinterpret_cast<uint4 const*>(hostSource)
                        : reinterpret_cast<uint4 const*>(
                              block->mapped[gpu] +
                              size_t(rankGpu % kGpusPerGroup) * slot.length);
      }
      if (!cfg.dmaMulticast) {
        size_t vectors = slot.length / sizeof(uint4);
        int blocks = int(std::min<size_t>(
            128, (vectors + cfg.consumeThreads - 1) / cfg.consumeThreads));
        multicastPacket<<<dim3(blocks, sourceCount - int(skipInRange)),
                          cfg.consumeThreads, 0, device.consume>>>(
            source, reinterpret_cast<uint4*>(device.output),
            shardBytes / sizeof(uint4), piece * cfg.chunkBytes / sizeof(uint4),
            vectors, firstSource, skipInRange ? skipSource : -1);
        check(cudaGetLastError());
      }
      check(cudaEventRecord(device.consumeDone[slotIndex], device.consume));
    };

    std::array<std::unique_ptr<GpuWorker>, kGpusPerGroup> workers;
    if (cfg.gpuWorkers)
      for (int gpu = 0; gpu < kGpusPerGroup; ++gpu)
        workers[gpu] = std::make_unique<GpuWorker>(
            *devices[gpu], localRank + 2 * (gpu + 1),
            [&, gpu](int slot) { uploadGpu(slot, gpu); },
            [&, gpu](int slot, int phase) { consumeGpu(slot, gpu, phase); },
            cfg.earlyConsume);
    auto waitGpu = [&](int slot, uint64_t generation, bool consume) {
      for (;;) {
        bool done = true;
        for (auto const& worker : workers) {
          worker->checkFailure();
          done &= worker->completed(slot, generation, consume);
        }
        if (done) return;
        // Drive UCX while device workers poll CUDA events on fixed contexts.
        int flag = 0;
        check(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag,
                         MPI_STATUS_IGNORE));
      }
    };
    auto launch = [&](size_t fullBytes) {
      size_t shardBytes = fullBytes / kGpus;
      size_t pieces = (shardBytes + cfg.chunkBytes - 1) / cfg.chunkBytes;
      auto acknowledge = [&](int slotIndex) {
        Slot& slot = *slots[slotIndex];
        if (cfg.sharedLocal && !slot.ackSent) {
          check(MPI_Isend(nullptr, 0, MPI_BYTE, localPeer, 530 + slotIndex,
                          MPI_COMM_WORLD, &slot.sharedRequests[1]));
          slot.ackSent = true;
        }
      };
      auto retire = [&](int slotIndex) {
        Slot& slot = *slots[slotIndex];
        if (!slot.active) return;
        if (!slot.sent || !slot.multicast)
          throw std::runtime_error("incomplete host packet slot");
        if (cfg.gpuWorkers)
          waitGpu(slotIndex, slot.generation, true);
        else
          for (auto const& device : devices) {
            if (!cfg.leanContext) check(cudaSetDevice(device->physical));
            check(cudaEventSynchronize(device->consumeDone[slotIndex]));
          }
        acknowledge(slotIndex);
        check(MPI_Waitall(3, slot.outgoing.data(), MPI_STATUSES_IGNORE));
        if (cfg.sharedLocal)
          check(
              MPI_Waitall(2, slot.sharedRequests.data(), MPI_STATUSES_IGNORE));
        if (rdma) rdma->flush();
        slot.active = false;
      };
      auto startPiece = [&](size_t piece) {
        int slotIndex = int(piece % kSlots);
        retire(slotIndex);
        Slot& slot = *slots[slotIndex];
        slot.active = true;
        slot.sent = slot.multicast = slot.localConsumed = slot.relaySent =
            slot.ackSent = false;
        slot.piece = int(piece);
        slot.shardBytes = shardBytes;
        ++slot.generation;
        size_t offset = piece * cfg.chunkBytes;
        slot.length = std::min(cfg.chunkBytes, shardBytes - offset);
        int count = int(2 * slot.length);
        if (rdma) rdma->grant(slotIndex, slot.generation);
        for (int group = 0; group < kGroups && !rdma; ++group) {
          bool relay = cfg.pairedNetwork && group != localRank;
          check(MPI_Irecv(slot.remote[group].host, count, MPI_BYTE,
                          relay ? localPeer : remoteBase + group,
                          relay ? kTagRelay : kTagRemote, MPI_COMM_WORLD,
                          &slot.receive[group]));
        }
        check(MPI_Irecv(cfg.sharedLocal ? nullptr : slot.local.host,
                        cfg.sharedLocal ? 0 : count, MPI_BYTE, localPeer,
                        cfg.sharedLocal ? 520 + slotIndex : kTagLocal,
                        MPI_COMM_WORLD, &slot.receive[2]));
        for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
          if (cfg.gpuWorkers)
            workers[gpu]->requestUpload(slotIndex, slot.generation);
          else {
            if (!cfg.leanContext || cfg.smUpload)
              check(cudaSetDevice(devices[gpu]->physical));
            uploadGpu(slotIndex, gpu);
          }
        }
      };
      auto sendPiece = [&](size_t piece, bool uploadReady = false) {
        int slotIndex = int(piece % kSlots);
        Slot& slot = *slots[slotIndex];
        if (!uploadReady && cfg.gpuWorkers)
          waitGpu(slotIndex, slot.generation, false);
        else if (!uploadReady)
          for (auto const& device : devices) {
            if (!cfg.leanContext) check(cudaSetDevice(device->physical));
            check(cudaEventSynchronize(device->uploadDone[slotIndex]));
          }
        int count = int(2 * slot.length);
        if (rdma) {
          while (!rdma->credits(slotIndex, slot.generation)) {
            int pending = 0;
            check(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                             &pending, MPI_STATUS_IGNORE));
          }
          rdma->send(slotIndex, count);
        } else if (cfg.pairedNetwork) {
          // One network copy per source group; the destination group relays
          // its immutable receive packet to the other local NUMA group.
          check(MPI_Isend(slot.send.host, count, MPI_BYTE,
                          remoteBase + localRank, kTagRemote, MPI_COMM_WORLD,
                          &slot.outgoing[0]));
        } else {
          check(MPI_Isend(slot.send.host, count, MPI_BYTE, remoteBase,
                          kTagRemote, MPI_COMM_WORLD, &slot.outgoing[0]));
          check(MPI_Isend(slot.send.host, count, MPI_BYTE, remoteBase + 1,
                          kTagRemote, MPI_COMM_WORLD, &slot.outgoing[1]));
        }
        if (cfg.sharedLocal) {
          check(MPI_Irecv(nullptr, 0, MPI_BYTE, localPeer, 530 + slotIndex,
                          MPI_COMM_WORLD, &slot.outgoing[2]));
          check(MPI_Isend(nullptr, 0, MPI_BYTE, localPeer, 520 + slotIndex,
                          MPI_COMM_WORLD, &slot.sharedRequests[0]));
        } else {
          check(MPI_Isend(slot.send.host, count, MPI_BYTE, localPeer, kTagLocal,
                          MPI_COMM_WORLD, &slot.outgoing[2]));
        }
        slot.sent = true;
      };
      auto dispatchConsume = [&](int slotIndex, int phase) {
        Slot& slot = *slots[slotIndex];
        for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
          if (cfg.gpuWorkers)
            workers[gpu]->requestConsume(slotIndex, slot.generation, phase);
          else {
            if (!cfg.leanContext || !cfg.dmaMulticast)
              check(cudaSetDevice(devices[gpu]->physical));
            consumeGpu(slotIndex, gpu, phase);
          }
        }
      };
      auto multicastPiece = [&](size_t piece) {
        int slotIndex = int(piece % kSlots);
        Slot& slot = *slots[slotIndex];
        if (cfg.earlyConsume && !slot.localConsumed) {
          check(MPI_Wait(&slot.receive[2], MPI_STATUS_IGNORE));
          dispatchConsume(slotIndex, 0);
          slot.localConsumed = true;
        }
        if (cfg.pairedNetwork && !slot.relaySent) {
          check(MPI_Wait(&slot.receive[localRank], MPI_STATUS_IGNORE));
          check(MPI_Isend(slot.remote[localRank].host, int(2 * slot.length),
                          MPI_BYTE, localPeer, kTagRelay, MPI_COMM_WORLD,
                          &slot.outgoing[1]));
          slot.relaySent = true;
        }
        while (rdma && !rdma->received(slotIndex, slot.generation)) {
          int pending = 0;
          check(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                           &pending, MPI_STATUS_IGNORE));
        }
        check(MPI_Waitall(3, slot.receive.data(), MPI_STATUSES_IGNORE));
        dispatchConsume(slotIndex, cfg.earlyConsume ? 1 : -1);
        slot.multicast = true;
      };
      if (cfg.readyProgress) {
        auto gpuReady = [&](int index, bool consume) {
          Slot& slot = *slots[index];
          bool ready = true;
          for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
            if (cfg.gpuWorkers) {
              workers[gpu]->checkFailure();
              ready &= workers[gpu]->completed(index, slot.generation, consume);
            } else {
              if (!cfg.leanContext)
                check(cudaSetDevice(devices[gpu]->physical));
              auto status =
                  cudaEventQuery(consume ? devices[gpu]->consumeDone[index]
                                         : devices[gpu]->uploadDone[index]);
              if (status == cudaErrorNotReady)
                ready = false;
              else
                check(status);
            }
          }
          return ready;
        };
        size_t nextProduce = 0, nextSend = 0, nextRelay = 0, retired = 0;
        while (retired < pieces) {
          for (int index = 0; index < kSlots; ++index) {
            Slot& slot = *slots[index];
            if (!slot.active) continue;
            if (cfg.gpuWorkers)
              for (auto const& worker : workers) worker->checkFailure();
            // Keep identical-tag MPI sends in piece order, even if device
            // completions or physical slot scans are out of order.
            if (!slot.sent && size_t(slot.piece) == nextSend &&
                gpuReady(index, false) &&
                (!rdma || rdma->credits(index, slot.generation))) {
              sendPiece(nextSend++, true);
            }
            if (slot.sent && !slot.multicast) {
              int ready = 0;
              if (cfg.earlyConsume && !slot.localConsumed) {
                check(MPI_Test(&slot.receive[2], &ready, MPI_STATUS_IGNORE));
                if (ready) {
                  dispatchConsume(index, 0);
                  slot.localConsumed = true;
                }
              }
              if (cfg.pairedNetwork && !slot.relaySent &&
                  size_t(slot.piece) == nextRelay) {
                check(MPI_Test(&slot.receive[localRank], &ready,
                               MPI_STATUS_IGNORE));
                if (ready) {
                  check(MPI_Isend(slot.remote[localRank].host,
                                  int(2 * slot.length), MPI_BYTE, localPeer,
                                  kTagRelay, MPI_COMM_WORLD,
                                  &slot.outgoing[1]));
                  slot.relaySent = true;
                  ++nextRelay;
                }
              }
              if (!cfg.pairedNetwork || slot.relaySent) {
                check(MPI_Testall(3, slot.receive.data(), &ready,
                                  MPI_STATUSES_IGNORE));
                if (ready && (!rdma || rdma->received(index, slot.generation)))
                  multicastPiece(size_t(slot.piece));
              }
            }
            if (slot.multicast && gpuReady(index, true)) {
              acknowledge(index);
              int done = 0;
              check(MPI_Testall(3, slot.outgoing.data(), &done,
                                MPI_STATUSES_IGNORE));
              if (done && cfg.sharedLocal)
                check(MPI_Testall(2, slot.sharedRequests.data(), &done,
                                  MPI_STATUSES_IGNORE));
              if (done) {
                if (rdma) rdma->flush();
                slot.active = false;
                ++retired;
              }
            }
          }
          if (nextProduce < pieces && !slots[nextProduce % kSlots]->active)
            startPiece(nextProduce++);
          int pending = 0;
          check(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                           &pending, MPI_STATUS_IGNORE));
        }
      } else {
        for (size_t step = 0; step < pieces + 2; ++step) {
          if (step < pieces) startPiece(step);
          if (step >= 1 && step - 1 < pieces) sendPiece(step - 1);
          if (step >= 2 && step - 2 < pieces) multicastPiece(step - 2);
        }
        for (int slot = 0; slot < kSlots; ++slot) retire(slot);
      }
    };
    auto launchNccl = [&](size_t fullBytes) {
      size_t shardElements = fullBytes / sizeof(float) / kGpus;
      check(ncclGroupStart());
      for (int gpu = 0; gpu < kGpusPerGroup; ++gpu) {
        Device& device = *devices[gpu];
        check(cudaSetDevice(device.physical));
        check(ncclAllGather(device.input, device.output, shardElements,
                            ncclFloat, ncclComms[gpu], device.consume));
      }
      check(ncclGroupEnd());
      for (auto const& device : devices) {
        check(cudaSetDevice(device->physical));
        check(cudaStreamSynchronize(device->consume));
      }
    };
    // Prime the comparison backend before measuring the first host case.
    // These checked warmups are outside every reported timing interval.
    if (cfg.compareNccl) {
      for (int i = 0; i < cfg.warmups; ++i) {
        fill(cfg.begin, i);
        check(MPI_Barrier(MPI_COMM_WORLD));
        launchNccl(cfg.begin);
        verify(cfg.begin, i);
      }
      if (rank == 0)
        std::printf("# primed_nccl_calls=%d,checked=1\n", cfg.warmups);
    }
    for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
      std::array<std::vector<double>, 2> runs;
      for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
        std::vector<int> order{0};
        if (cfg.compareNccl) {
          if (repeat % 2 == 0) order.push_back(1);
          else order = {1, 0};
        }
        for (int backend : order) {
          double sum = 0;
          for (int call = 0; call < cfg.warmups + cfg.iterations; ++call) {
            int generation = 10 + repeat * (cfg.warmups + cfg.iterations) + call;
            fill(bytes, generation);
            check(MPI_Barrier(MPI_COMM_WORLD));
            auto start = std::chrono::steady_clock::now();
            if (backend == 0) launch(bytes);
            else launchNccl(bytes);
            auto end = std::chrono::steady_clock::now();
            double localUs = std::chrono::duration<double, std::micro>(
                                 end - start).count();
            double maxUs = 0;
            check(MPI_Allreduce(&localUs, &maxUs, 1, MPI_DOUBLE, MPI_MAX,
                                 MPI_COMM_WORLD));
            verify(bytes, generation);
            if (call >= cfg.warmups) {
              sum += maxUs;
              if (rank == 0)
                std::printf("# sample,allgather,%zu,%s,%d,%d,%.3f,0\n", bytes,
                            backend
                                ? "nccl"
                                : (cfg.directRdma      ? "host-rdma-direct"
                                   : cfg.pairedNetwork ? "host-rdma-paired"
                                                       : "host-rdma-multicast"),
                            repeat + 1, call - cfg.warmups + 1, maxUs);
            }
          }
          runs[backend].push_back(sum / cfg.iterations);
        }
      }
      if (rank == 0) {
        for (int backend = 0; backend < (cfg.compareNccl ? 2 : 1); ++backend) {
          auto bounds = std::minmax_element(runs[backend].begin(),
                                             runs[backend].end());
          std::printf("allgather,%zu,%s,%.3f,%.3f,%.3f\n", bytes,
                      backend ? "nccl"
                              : (cfg.directRdma      ? "host-rdma-direct"
                                 : cfg.pairedNetwork ? "host-rdma-paired"
                                                     : "host-rdma-multicast"),
                      median(runs[backend]), *bounds.first, *bounds.second);
        }
        std::fflush(stdout);
      }
      if (bytes > cfg.end / 2) break;
    }
    if (cfg.compareNccl)
      for (auto comm : ncclComms) check(ncclCommDestroy(comm));
    workers = {};
    rdma.reset();
    slots = {};
    if (cfg.sharedLocal) sharedPackets.close();
    devices = {};
    check(MPI_Comm_free(&localComm));
    check(MPI_Finalize());
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "host RDMA AllGather: %s\n", error.what());
    if (mpiReady) MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

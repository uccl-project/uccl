#include "uccl_lite/multiprocess_collectives.h"
#include "uccl_lite/cpu_router/cpu_executor.hpp"
#include "uccl_lite/multiprocess_packet_ag.h"
#include <linux/mempolicy.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime_api.h>
#include <dirent.h>
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
constexpr int kMaxRanks = 32;
constexpr uint64_t kMagic = 0x5543434c4d503031ULL;
constexpr auto kTimeout = std::chrono::seconds(60);
thread_local std::string lastError;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

// Shared POD is initialized once by rank 0. Aligned 64-bit atomics are
// lock-free on the supported x86-64 hosts and operate on the shared mapping.
struct alignas(64) Shared {
  std::atomic<uint64_t> magic;
  int ranks;
  size_t maxBytes;
  size_t smallArGpuMaxBytes;
  size_t smallRsGpuMaxBytes;
  std::atomic<int> failed;
  std::atomic<int> openedCount;
  std::atomic<uint64_t> packetTouched[kMaxRanks];
  int packetMode;
  std::atomic<uint64_t> arrived[kMaxRanks];
  std::atomic<uint64_t> hostPacketArrived[kMaxRanks];
  std::atomic<uint64_t> directReadDone[kMaxRanks];
  std::atomic<uint64_t> reduced[kMaxRanks];
  std::atomic<uint64_t> finished[kMaxRanks];
  int operation[kMaxRanks];
  size_t count[kMaxRanks];
};
static_assert(std::atomic<uint64_t>::is_always_lock_free);

struct RsPacketJob {
  int operation;
  size_t count;
  uint64_t baseGeneration;
  uint64_t previousResult[2];
  bool previousResultAllReaders[2];
  uint64_t previousPartial[2];
};

struct PacketLaneTask {
  std::array<uccl_lite::cpu_router::MemorySpan, 8> sources;
  uccl_lite::cpu_router::MemorySpan output;
  int sourceCount = 0;
};

size_t alignPage(size_t n) { return (n + 4095) & ~size_t(4095); }

int deviceNumaNode(int device) {
  char pci[32];
  check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
  std::ifstream input(std::string("/sys/bus/pci/devices/") + pci +
                      "/numa_node");
  int node = -1;
  input >> node;
  if (!input || node < 0 || node >= 64)
    throw std::runtime_error("GPU NUMA node unavailable");
  return node;
}

void bindPacketMemory(void* address, size_t bytes, int node) {
  unsigned long mask = 1UL << node;
  if (syscall(SYS_mbind, address, bytes, MPOL_BIND, &mask, 64, 0) != 0)
    throw std::runtime_error(std::string("packet mbind: ") +
                             std::strerror(errno));
}

struct Handle {
  std::string name;
  int fd = -1;
  void* mapping = nullptr;
  size_t mappedBytes = 0;
  Shared* shared = nullptr;
  char* ownInput = nullptr;
  char* ownOutput = nullptr;
  int rank = -1;
  int ranks = 0;
  int device = -1;
  size_t maxBytes = 0;
  size_t smallArGpuMaxBytes = 0;
  size_t smallRsGpuMaxBytes = 0;
  uint64_t nextSequence = 1;
  uint64_t lastDirectSequence = 0;
  cudaStream_t lastStream = nullptr;
  bool readyOnly = false;
  bool directAllGather = false;
  bool directAllReduce = false;
  bool packetAllGather = false;
  bool pipelinedAllGather = false;
  bool packetReduceScatter = false;
  bool pipelinedReduceScatter = false;
  bool cpuReduceScatter = false;
  bool retainOwnReduceScatter = false;
  bool destReadyReduceScatter = false;
  bool hostGateReduceScatter = false;
  bool packetAllReduce = false;
  bool replicateAllReduce = false;
  bool hierarchicalPacket = false;
  bool hierarchicalReduceScatter = false;
  bool fastPacketSum = false;
  bool splitPacketCredits = false;
  bool packetScratch = false;
  bool parallelAllReduce = false;
  bool isolatePacketWorker = false;
  uccl_lite::multiprocess::PacketShared* packetHost = nullptr;
  uccl_lite::multiprocess::PacketShared* packetDevice = nullptr;
  uint16_t* agDevice = nullptr;
  uccl_lite::multiprocess::CrewState* crews = nullptr;
  uint32_t packetBarrierPhase = 0;
  uint64_t lastPacketGeneration[2]{};
  bool lastPacketWasCpu[2]{};
  uint64_t lastCpuResultGeneration[2]{};
  bool lastCpuResultAllReaders[2]{};
  uint64_t lastPartialGeneration[2]{};
  std::vector<char*> registeredPeerInputs;
  std::vector<char*> registeredPeerOutputs;
  std::mutex rsMutex;
  std::condition_variable rsAvailable;
  std::deque<RsPacketJob> rsJobs;
  bool rsStop = false;
  std::thread rsThread;
  std::atomic<pid_t> rsWorkerTid{0};
  std::atomic<pid_t> arLaneTid{0};
  std::unique_ptr<uccl_lite::cpu_router::CpuExecutor> rsHelper;
  std::thread arLaneThread;
  std::mutex arLaneMutex;
  std::condition_variable arLaneAvailable;
  bool arLaneStop = false;
  uint64_t arLaneJobSerial = 0;
  uint64_t arLaneJobBase = 0;
  size_t arLaneJobTiles = 0;
  PacketLaneTask arLaneTask;
  std::atomic<uint64_t> arLaneTaskGeneration{0};
  std::atomic<uint64_t> arLaneDoneGeneration{0};

  char* inputs() const {
    return static_cast<char*>(mapping) + alignPage(sizeof(Shared));
  }
  char* outputs() const { return inputs() + size_t(ranks) * maxBytes; }
  char* packetRegion() const { return outputs() + size_t(ranks) * maxBytes; }
  char* input(int r) const { return inputs() + size_t(r) * maxBytes; }
  char* output(int r) const { return outputs() + size_t(r) * maxBytes; }

  ~Handle() {
    int previousDevice = -1;
    cudaGetDevice(&previousDevice);
    if (device >= 0) cudaSetDevice(device);
    // The CUDA stream may already have been destroyed by the caller. Drain
    // outstanding staging copies before unregistering their host buffers.
    cudaDeviceSynchronize();
    if (rsThread.joinable()) {
      {
        std::lock_guard<std::mutex> guard(rsMutex);
        rsStop = true;
      }
      rsAvailable.notify_one();
      rsThread.join();
    }
    rsHelper.reset();
    if (arLaneThread.joinable()) {
      {
        std::lock_guard<std::mutex> guard(arLaneMutex);
        arLaneStop = true;
      }
      arLaneAvailable.notify_one();
      arLaneThread.join();
    }
    if (crews) cudaFree(crews);
    if (packetHost) cudaHostUnregister(packetHost);
    for (char* peer : registeredPeerOutputs) cudaHostUnregister(peer);
    for (char* peer : registeredPeerInputs) cudaHostUnregister(peer);
    if (ownOutput) cudaHostUnregister(ownOutput);
    if (ownInput) cudaHostUnregister(ownInput);
    if (mapping) munmap(mapping, mappedBytes);
    if (fd >= 0) close(fd);
    if (previousDevice >= 0 && previousDevice != device)
      cudaSetDevice(previousDevice);
    // The final successful opener unlinks the shared-memory name. Existing
    // mappings remain valid until every rank exits, including abnormal exit.
  }
};

void waitFor(std::atomic<uint64_t>& value, uint64_t sequence, Shared* shared) {
  auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (value.load(std::memory_order_acquire) < sequence) {
    if (shared->failed.load(std::memory_order_acquire) ||
        std::chrono::steady_clock::now() >= deadline) {
      shared->failed.store(1, std::memory_order_release);
      throw std::runtime_error("collective rank timeout or failure");
    }
    std::this_thread::yield();
  }
}

void reduceBf16(Handle* h, size_t sourceOffset, char* destination,
                size_t elements) {
  using uccl_lite::cpu_router::CpuExecutor;
  using uccl_lite::cpu_router::MemoryKind;
  using uccl_lite::cpu_router::MemorySpan;
  std::array<MemorySpan, kMaxRanks> sources;
  size_t bytes = elements * sizeof(uint16_t);
  for (int r = 0; r < h->ranks; ++r)
    sources[r] = {h->input(r) + sourceOffset * sizeof(uint16_t),
                  bytes,
                  MemoryKind::NumaPinned,
                  -1,
                  -1,
                  {}};
  MemorySpan outputSpan{destination, bytes, MemoryKind::NumaPinned, -1, -1, {}};
  CpuExecutor::sumBf16(sources.data(), h->ranks, outputSpan);
}

__attribute__((target("avx512f,avx512bw"))) void sumEightBf16(
    uint16_t const* const input[8], uint16_t* output, size_t elements) {
  size_t i = 0;
  for (; i + 16 <= elements; i += 16) {
    __m512 sum = _mm512_setzero_ps();
    for (int source = 0; source < 8; ++source) {
      auto packed = _mm256_loadu_si256(
          reinterpret_cast<__m256i const*>(input[source] + i));
      auto values = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
      sum = _mm512_add_ps(sum, values);
    }
    auto bits = _mm512_castps_si512(sum);
    auto bias = _mm512_add_epi32(
        _mm512_set1_epi32(0x7fff),
        _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1)));
    auto rounded = _mm512_add_epi32(bits, bias);
    auto nan = _mm512_cmp_epu32_mask(
        _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
        _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
    rounded = _mm512_mask_mov_epi32(
        rounded, nan, _mm512_or_si512(bits, _mm512_set1_epi32(0x00400000)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i),
                        _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16)));
  }
  for (; i < elements; ++i) {
    float sum = 0;
    for (int source = 0; source < 8; ++source) {
      uint32_t bits = uint32_t(input[source][i]) << 16;
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      sum += value;
    }
    uint32_t bits;
    std::memcpy(&bits, &sum, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U)
      bits |= 0x00400000U;
    else
      bits += 0x7fffU + ((bits >> 16) & 1U);
    output[i] = uint16_t(bits >> 16);
  }
}

int mappedCpu(char const* environment, int device) {
  char const* mapping = std::getenv(environment);
  if (!mapping || !*mapping) return -1;
  char const* cursor = mapping;
  long cpu = -1;
  for (int index = 0; index <= device; ++index) {
    char* end = nullptr;
    cpu = std::strtol(cursor, &end, 10);
    if (end == cursor || cpu < 0 || cpu >= CPU_SETSIZE ||
        (index < device && *end != ',') ||
        (index == device && *end != ',' && *end != '\0'))
      throw std::invalid_argument(std::string("invalid CPU map: ") +
                                  environment);
    cursor = end + 1;
  }
  return int(cpu);
}

void isolatePacketWorker(Handle* h, int workerCpu) {
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    throw std::runtime_error("scheduler affinity query failed");
  std::ifstream siblings("/sys/devices/system/cpu/cpu" +
                         std::to_string(workerCpu) +
                         "/topology/thread_siblings_list");
  std::string list;
  if (!std::getline(siblings, list))
    throw std::runtime_error("worker CPU sibling list unavailable");
  size_t cursor = 0;
  while (cursor < list.size()) {
    size_t comma = list.find(',', cursor);
    std::string group = list.substr(cursor, comma - cursor);
    size_t dash = group.find('-');
    int first = std::stoi(group.substr(0, dash));
    int last =
        dash == std::string::npos ? first : std::stoi(group.substr(dash + 1));
    if (first < 0 || last < first || last >= CPU_SETSIZE)
      throw std::invalid_argument("invalid worker CPU sibling list");
    for (int cpu = first; cpu <= last; ++cpu) CPU_CLR(cpu, &allowed);
    if (comma == std::string::npos) break;
    cursor = comma + 1;
  }
  bool any = false;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) any |= CPU_ISSET(cpu, &allowed);
  if (!any) throw std::runtime_error("no CPUs remain for scheduler");

  DIR* tasks = opendir("/proc/self/task");
  if (!tasks) throw std::runtime_error("scheduler thread list unavailable");
  int changed = 0;
  while (dirent* entry = readdir(tasks)) {
    char* end = nullptr;
    long tid = std::strtol(entry->d_name, &end, 10);
    if (end == entry->d_name || *end ||
        tid == h->rsWorkerTid.load(std::memory_order_acquire) ||
        tid == h->arLaneTid.load(std::memory_order_acquire))
      continue;
    cpu_set_t current;
    if (sched_getaffinity(pid_t(tid), sizeof(current), &current) != 0) {
      if (errno == ESRCH) continue;
      closedir(tasks);
      throw std::runtime_error("scheduler thread affinity query failed");
    }
    cpu_set_t target;
    CPU_ZERO(&target);
    bool nonempty = false;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      if (CPU_ISSET(cpu, &allowed) && CPU_ISSET(cpu, &current)) {
        CPU_SET(cpu, &target);
        nonempty = true;
      }
    if (!nonempty) target = allowed;
    if (sched_setaffinity(pid_t(tid), sizeof(target), &target) != 0 &&
        errno != ESRCH) {
      closedir(tasks);
      throw std::runtime_error("scheduler thread affinity update failed");
    }
    ++changed;
  }
  closedir(tasks);
  std::fprintf(stderr,
               "UCCL Lite packet worker device=%d isolated CPU=%d "
               "scheduler_threads=%d\n",
               h->device, workerCpu, changed);
}

void runParallelArLane(Handle* h, int cpu) {
  try {
    pthread_setname_np(pthread_self(), "uccl_ar_lane");
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
      throw std::runtime_error("AR lane CPU affinity failed");
    h->arLaneTid.store(pid_t(syscall(SYS_gettid)), std::memory_order_release);
    uint64_t serial = 0;
    while (true) {
      uint64_t base;
      size_t tiles;
      {
        std::unique_lock<std::mutex> lock(h->arLaneMutex);
        h->arLaneAvailable.wait(
            lock, [&] { return h->arLaneStop || h->arLaneJobSerial > serial; });
        if (h->arLaneStop) return;
        serial = h->arLaneJobSerial;
        base = h->arLaneJobBase;
        tiles = h->arLaneJobTiles;
      }
      for (size_t tile = 0; tile < tiles; ++tile) {
        uint64_t generation = base + tile + 1;
        auto deadline = std::chrono::steady_clock::now() + kTimeout;
        while (h->arLaneTaskGeneration.load(std::memory_order_acquire) <
               generation) {
          if (h->shared->failed.load(std::memory_order_acquire) ||
              std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("AR lane task timed out");
          _mm_pause();
        }
        auto& task = h->arLaneTask;
        uccl_lite::cpu_router::CpuExecutor::sumBf16(
            task.sources.data(), task.sourceCount, task.output);
        h->arLaneDoneGeneration.store(generation, std::memory_order_release);
      }
    }
  } catch (std::exception const& e) {
    h->shared->failed.store(1, std::memory_order_release);
    std::fprintf(stderr, "UCCL Lite parallel AR lane failed: %s\n", e.what());
    std::abort();
  }
}

void runRsPacketWorker(Handle* h) {
  using uccl_lite::cpu_router::CpuExecutor;
  using uccl_lite::cpu_router::MemoryKind;
  using uccl_lite::cpu_router::MemorySpan;
  try {
    pthread_setname_np(pthread_self(), "uccl_rs_worker");
    char const* spinSetting = std::getenv("UCCL_LITE_MP_PACKET_SPIN");
    bool spin = spinSetting && std::strcmp(spinSetting, "1") == 0;
    int cpu = mappedCpu("UCCL_LITE_MP_PACKET_WORKER_CPUS", h->device);
    if (cpu >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu, &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0)
        throw std::runtime_error("packet worker CPU affinity failed");
      std::fprintf(stderr, "UCCL Lite packet worker device=%d CPU=%ld\n",
                   h->device, long(cpu));
    }
    h->rsWorkerTid.store(pid_t(syscall(SYS_gettid)), std::memory_order_release);
    std::vector<uint16_t> scratch(
        h->packetScratch ? uccl_lite::multiprocess::kRsStripeElements : 0);
    while (true) {
      RsPacketJob job;
      {
        std::unique_lock<std::mutex> lock(h->rsMutex);
        h->rsAvailable.wait(lock,
                            [&] { return h->rsStop || !h->rsJobs.empty(); });
        if (h->rsJobs.empty() && h->rsStop) return;
        job = h->rsJobs.front();
        h->rsJobs.pop_front();
      }
      size_t maximumShard =
          job.operation == 2
              ? job.count
              : (job.count + size_t(h->ranks) - 1) / size_t(h->ranks);
      size_t stripeElements =
          job.operation == 3 && job.count <= (1UL << 19)
              ? uccl_lite::multiprocess::kRsStripeElements / 2
              : uccl_lite::multiprocess::kRsStripeElements;
      size_t tiles = (maximumShard + stripeElements - 1) / stripeElements;
      bool hierarchicalJob = (h->hierarchicalPacket && job.operation == 3 &&
                              job.count >= (8UL << 20)) ||
                             (h->hierarchicalReduceScatter &&
                              job.operation == 2 && job.count >= (1UL << 20));
      bool parallelJob =
          h->parallelAllReduce &&
          ((job.operation == 3 &&
            job.count >= (hierarchicalJob ? (8UL << 20) : (1UL << 19))) ||
           (h->hierarchicalReduceScatter && job.operation == 2 &&
            job.count >= (1UL << 20))) &&
          job.count % (size_t(h->ranks) *
                       uccl_lite::multiprocess::kRsStripeElements) ==
              0;
      if (parallelJob) {
        {
          std::lock_guard<std::mutex> guard(h->arLaneMutex);
          h->arLaneJobBase = job.baseGeneration;
          h->arLaneJobTiles = tiles;
          ++h->arLaneJobSerial;
        }
        h->arLaneAvailable.notify_one();
      }
      bool profile =
          h->rank == 0 && std::getenv("UCCL_LITE_MP_PACKET_PROFILE") != nullptr;
      auto jobStart = std::chrono::steady_clock::now();
      uint64_t waitMicros = 0, mathMicros = 0;
      uint64_t localMicros = 0, remoteMicros = 0;
      uint64_t mergeMicros = 0, copyMicros = 0;
      for (size_t tile = 0; tile < tiles; ++tile) {
        int slot = int(tile & 1);
        uint64_t generation = job.baseGeneration + tile + 1;
        auto deadline = std::chrono::steady_clock::now() + kTimeout;
        for (int peer = 0; peer < h->ranks; ++peer) {
          uint64_t* ready =
              job.operation == 2 && h->destReadyReduceScatter
                  ? &h->packetHost->uploadedDest[slot][peer][h->rank].value
                  : &h->packetHost->uploaded[slot][peer].value;
          while (__atomic_load_n(ready, __ATOMIC_ACQUIRE) < generation) {
            if (h->shared->failed.load(std::memory_order_acquire) ||
                std::chrono::steady_clock::now() >= deadline)
              throw std::runtime_error("RS packet upload timed out");
            if (spin)
              _mm_pause();
            else
              std::this_thread::yield();
          }
        }
        bool splitThisJob = h->splitPacketCredits && job.operation == 3;
        auto waitResultCredit = [&] {
          uint64_t priorResult = tile >= 2 ? job.baseGeneration + tile - 1
                                           : job.previousResult[slot];
          if (!priorResult) return;
          bool allReaders = tile >= 2 ? job.operation == 3
                                      : job.previousResultAllReaders[slot];
          int firstReader = allReaders ? 0 : h->rank;
          int lastReader = allReaders ? h->ranks : h->rank + 1;
          for (int peer = firstReader; peer < lastReader; ++peer)
            while (__atomic_load_n(&h->packetHost->consumed[slot][peer].value,
                                   __ATOMIC_ACQUIRE) < priorResult) {
              if (h->shared->failed.load(std::memory_order_acquire) ||
                  std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("RS packet result credit timed out");
              if (spin)
                _mm_pause();
              else
                std::this_thread::yield();
            }
        };
        if (splitThisJob && !h->packetScratch) waitResultCredit();
        if (hierarchicalJob) {
          uint64_t previousPartial = tile >= 2 ? job.baseGeneration + tile - 1
                                               : job.previousPartial[slot];
          if (previousPartial)
            while (__atomic_load_n(
                       &h->packetHost->partialReadDone[slot][h->rank].value,
                       __ATOMIC_ACQUIRE) < previousPartial) {
              if (h->shared->failed.load(std::memory_order_acquire) ||
                  std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("NUMA partial credit timed out");
              if (spin)
                _mm_pause();
              else
                std::this_thread::yield();
            }
        }
        auto mathStart = std::chrono::steady_clock::now();
        if (profile)
          waitMicros += std::chrono::duration_cast<std::chrono::microseconds>(
                            mathStart - jobStart)
                            .count();
        size_t begin = tile * stripeElements;
        size_t shardBegin =
            job.operation == 2 ? size_t(h->rank) * job.count
                               : job.count * size_t(h->rank) / size_t(h->ranks);
        size_t shardEnd = job.operation == 2 ? shardBegin + job.count
                                             : job.count * size_t(h->rank + 1) /
                                                   size_t(h->ranks);
        size_t shardElements = shardEnd - shardBegin;
        size_t elements = begin < shardElements
                              ? std::min(stripeElements, shardElements - begin)
                              : 0;
        size_t bytes = elements * sizeof(uint16_t);
        if (hierarchicalJob) {
          int pair = h->rank ^ 4;
          int firstSource = h->rank < 4 ? 0 : 4;
          auto computeLocalPart = [&](int part, bool helper) {
            int destination = part == 0 ? h->rank : pair;
            size_t destinationBegin =
                job.operation == 2
                    ? size_t(destination) * job.count
                    : job.count * size_t(destination) / size_t(h->ranks);
            size_t destinationEnd =
                job.operation == 2
                    ? destinationBegin + job.count
                    : job.count * size_t(destination + 1) / size_t(h->ranks);
            size_t destinationLength = destinationEnd - destinationBegin;
            size_t partElements =
                begin < destinationLength
                    ? std::min(stripeElements, destinationLength - begin)
                    : 0;
            if (!partElements) return;
            size_t partBytes = partElements * sizeof(uint16_t);
            if (job.operation == 3) {
              std::array<MemorySpan, 4> localSources;
              for (int source = 0; source < 4; ++source)
                localSources[source] = {
                    h->packetHost->input[slot][firstSource + source] +
                        size_t(destination) *
                            uccl_lite::multiprocess::kRsStripeElements,
                    partBytes,
                    MemoryKind::NumaPinned,
                    -1,
                    -1,
                    {}};
              MemorySpan partial{h->packetHost->partial[slot][h->rank][part],
                                 partBytes,
                                 MemoryKind::NumaPinned,
                                 -1,
                                 -1,
                                 {}};
              if (helper) {
                h->arLaneTask.sourceCount = 4;
                for (int source = 0; source < 4; ++source)
                  h->arLaneTask.sources[source] = localSources[source];
                h->arLaneTask.output = partial;
                h->arLaneTaskGeneration.store(generation,
                                              std::memory_order_release);
              } else {
                CpuExecutor::sumBf16(localSources.data(), 4, partial);
              }
              return;
            }
            std::array<MemorySpan, 4> localSources;
            int sourceCount = 0;
            for (int source = 0; source < 4; ++source) {
              if (h->retainOwnReduceScatter &&
                  firstSource + source == destination)
                continue;
              localSources[sourceCount++] = {
                  h->packetHost->input[slot][firstSource + source] +
                      size_t(destination) *
                          uccl_lite::multiprocess::kRsStripeElements,
                  partBytes,
                  MemoryKind::NumaPinned,
                  -1,
                  -1,
                  {}};
            }
            MemorySpan partial{h->packetHost->partial[slot][h->rank][part],
                               partBytes,
                               MemoryKind::NumaPinned,
                               -1,
                               -1,
                               {}};
            if (helper) {
              h->arLaneTask.sourceCount = sourceCount;
              for (int source = 0; source < sourceCount; ++source)
                h->arLaneTask.sources[source] = localSources[source];
              h->arLaneTask.output = partial;
              h->arLaneTaskGeneration.store(generation,
                                            std::memory_order_release);
            } else {
              CpuExecutor::sumBf16(localSources.data(), sourceCount, partial);
            }
          };
          if (parallelJob) {
            computeLocalPart(1, true);
            computeLocalPart(0, false);
            while (h->arLaneDoneGeneration.load(std::memory_order_acquire) <
                   generation) {
              if (h->shared->failed.load(std::memory_order_acquire) ||
                  std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("NUMA AR lane timed out");
              _mm_pause();
            }
          } else {
            computeLocalPart(0, false);
            computeLocalPart(1, false);
          }
          auto localEnd =
              profile ? std::chrono::steady_clock::now() : mathStart;
          if (profile)
            localMicros +=
                std::chrono::duration_cast<std::chrono::microseconds>(localEnd -
                                                                      mathStart)
                    .count();
          if (splitThisJob)
            __atomic_store_n(&h->packetHost->inputReadDone[slot][h->rank].value,
                             generation, __ATOMIC_RELEASE);
          __atomic_store_n(&h->packetHost->localReady[slot][h->rank].value,
                           generation, __ATOMIC_RELEASE);
          while (__atomic_load_n(&h->packetHost->localReady[slot][pair].value,
                                 __ATOMIC_ACQUIRE) < generation) {
            if (h->shared->failed.load(std::memory_order_acquire) ||
                std::chrono::steady_clock::now() >= deadline)
              throw std::runtime_error("remote NUMA partial timed out");
            if (spin)
              _mm_pause();
            else
              std::this_thread::yield();
          }
          auto pairReady =
              profile ? std::chrono::steady_clock::now() : mathStart;
          if (profile)
            remoteMicros +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    pairReady - localEnd)
                    .count();
          if (elements) {
            MemorySpan partials[2] = {{h->packetHost->partial[slot][h->rank][0],
                                       bytes,
                                       MemoryKind::NumaPinned,
                                       -1,
                                       -1,
                                       {}},
                                      {h->packetHost->partial[slot][pair][1],
                                       bytes,
                                       MemoryKind::NumaPinned,
                                       -1,
                                       -1,
                                       {}}};
            MemorySpan result{
                splitThisJob && h->packetScratch
                    ? static_cast<void*>(scratch.data())
                    : static_cast<void*>(
                          h->packetHost->result[slot][h->rank / 4][h->rank]),
                bytes,
                splitThisJob && h->packetScratch ? MemoryKind::NumaHost
                                                 : MemoryKind::NumaPinned,
                -1,
                -1,
                {}};
            CpuExecutor::sumBf16(partials, 2, result);
          }
          __atomic_store_n(&h->packetHost->partialReadDone[slot][pair].value,
                           generation, __ATOMIC_RELEASE);
          if (profile)
            mergeMicros +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - pairReady)
                    .count();
        } else if (elements) {
          std::array<MemorySpan, 8> sources;
          int sourceCount = 0;
          for (int peer = 0; peer < h->ranks; ++peer) {
            if (job.operation == 2 && h->retainOwnReduceScatter &&
                peer == h->rank)
              continue;
            sources[sourceCount++] = {
                h->packetHost->input[slot][peer] +
                    size_t(h->rank) *
                        uccl_lite::multiprocess::kRsStripeElements,
                bytes,
                MemoryKind::NumaPinned,
                -1,
                -1,
                {}};
          }
          MemorySpan output{
              splitThisJob && h->packetScratch
                  ? static_cast<void*>(scratch.data())
                  : static_cast<void*>(
                        h->packetHost->result[slot][h->rank / 4][h->rank]),
              bytes,
              splitThisJob && h->packetScratch ? MemoryKind::NumaHost
                                               : MemoryKind::NumaPinned,
              -1,
              -1,
              {}};
          if (parallelJob) {
            size_t firstBytes = (elements / 2) * sizeof(uint16_t);
            h->arLaneTask.sourceCount = sourceCount;
            for (int peer = 0; peer < sourceCount; ++peer) {
              h->arLaneTask.sources[peer] = sources[peer];
              h->arLaneTask.sources[peer].address =
                  static_cast<char*>(sources[peer].address) + firstBytes;
              h->arLaneTask.sources[peer].bytes = bytes - firstBytes;
              sources[peer].bytes = firstBytes;
            }
            h->arLaneTask.output = output;
            h->arLaneTask.output.address =
                static_cast<char*>(output.address) + firstBytes;
            h->arLaneTask.output.bytes = bytes - firstBytes;
            output.bytes = firstBytes;
            h->arLaneTaskGeneration.store(generation,
                                          std::memory_order_release);
            CpuExecutor::sumBf16(sources.data(), sourceCount, output);
            auto deadline = std::chrono::steady_clock::now() + kTimeout;
            while (h->arLaneDoneGeneration.load(std::memory_order_acquire) <
                   generation) {
              if (h->shared->failed.load(std::memory_order_acquire) ||
                  std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("AR lane completion timed out");
              _mm_pause();
            }
          } else if (h->rsHelper &&
                     (job.operation == 2 ||
                      (job.operation == 3 && job.count >= (8UL << 20))) &&
                     elements >= 2) {
            size_t firstBytes = (elements / 2) * sizeof(uint16_t);
            std::array<MemorySpan, 8> secondSources;
            for (int peer = 0; peer < sourceCount; ++peer) {
              secondSources[peer] = sources[peer];
              secondSources[peer].address =
                  static_cast<char*>(secondSources[peer].address) + firstBytes;
              secondSources[peer].bytes = bytes - firstBytes;
              sources[peer].bytes = firstBytes;
            }
            MemorySpan secondOutput = output;
            secondOutput.address =
                static_cast<char*>(output.address) + firstBytes;
            secondOutput.bytes = bytes - firstBytes;
            output.bytes = firstBytes;
            auto done = h->rsHelper->enqueueSum(0, secondSources.data(),
                                                sourceCount, secondOutput);
            CpuExecutor::sumBf16(sources.data(), sourceCount, output);
            done.wait();
          } else if (h->fastPacketSum && sourceCount == 8) {
            uint16_t const* input[8];
            for (int peer = 0; peer < h->ranks; ++peer)
              input[peer] = static_cast<uint16_t const*>(sources[peer].address);
            sumEightBf16(input, static_cast<uint16_t*>(output.address),
                         elements);
          } else {
            CpuExecutor::sumBf16(sources.data(), sourceCount, output);
          }
        }
        if (splitThisJob) {
          auto copyStart =
              profile ? std::chrono::steady_clock::now() : mathStart;
          if (!hierarchicalJob)
            __atomic_store_n(&h->packetHost->inputReadDone[slot][h->rank].value,
                             generation, __ATOMIC_RELEASE);
          if (h->packetScratch) {
            waitResultCredit();
            if (elements)
              std::memcpy(h->packetHost->result[slot][h->rank / 4][h->rank],
                          scratch.data(), bytes);
          }
          if (profile)
            copyMicros += std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - copyStart)
                              .count();
        }
        if (job.operation == 3 && h->replicateAllReduce && elements)
          std::memcpy(h->packetHost->result[slot][1 - h->rank / 4][h->rank],
                      h->packetHost->result[slot][h->rank / 4][h->rank], bytes);
        auto mathEnd = std::chrono::steady_clock::now();
        if (profile)
          mathMicros += std::chrono::duration_cast<std::chrono::microseconds>(
                            mathEnd - mathStart)
                            .count();
        __atomic_store_n(&h->packetHost->routed[slot][h->rank].value,
                         generation, __ATOMIC_RELEASE);
        jobStart = mathEnd;
      }
      if (profile)
        std::fprintf(stderr,
                     "UCCL packet CPU profile op=%d count=%zu tiles=%zu "
                     "wait_us=%llu math_us=%llu local_us=%llu "
                     "remote_us=%llu merge_us=%llu copy_us=%llu\n",
                     job.operation, job.count, tiles,
                     static_cast<unsigned long long>(waitMicros),
                     static_cast<unsigned long long>(mathMicros),
                     static_cast<unsigned long long>(localMicros),
                     static_cast<unsigned long long>(remoteMicros),
                     static_cast<unsigned long long>(mergeMicros),
                     static_cast<unsigned long long>(copyMicros));
    }
  } catch (std::exception const& e) {
    h->shared->failed.store(1, std::memory_order_release);
    std::fprintf(stderr, "UCCL Lite RS packet worker failed: %s\n", e.what());
    std::abort();
  }
}

void bindCallbackCpu(int device) {
  char const* mapping = std::getenv("UCCL_LITE_MP_CALLBACK_CPUS");
  if (!mapping || !*mapping) return;
  thread_local int boundDevice = -1;
  thread_local int boundCpu = -1;
  if (boundDevice == device && sched_getcpu() == boundCpu) return;

  char const* cursor = mapping;
  long cpu = -1;
  for (int i = 0; i <= device; ++i) {
    char* end = nullptr;
    cpu = std::strtol(cursor, &end, 10);
    if (end == cursor || cpu < 0 || cpu >= CPU_SETSIZE ||
        (i < device && *end != ',') ||
        (i == device && *end != '\0' && *end != ','))
      throw std::invalid_argument("invalid UCCL_LITE_MP_CALLBACK_CPUS");
    cursor = end + 1;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0)
    throw std::runtime_error(std::string("callback CPU affinity: ") +
                             std::strerror(errno));
  boundDevice = device;
  boundCpu = static_cast<int>(cpu);
  std::fprintf(stderr, "UCCL Lite callback device=%d CPU=%ld\n", device, cpu);
}

struct Call {
  Handle* handle;
  uint64_t sequence;
  int operation;
  size_t count;
};

struct DirectReadCall {
  Handle* handle;
  uint64_t sequence;
};

void CUDART_CB publishDirectReadDone(void* opaque) {
  std::unique_ptr<DirectReadCall> call(static_cast<DirectReadCall*>(opaque));
  call->handle->shared->directReadDone[call->handle->rank].store(
      call->sequence, std::memory_order_release);
}

void CUDART_CB waitDirectReaders(void* opaque) {
  std::unique_ptr<DirectReadCall> call(static_cast<DirectReadCall*>(opaque));
  auto* shared = call->handle->shared;
  try {
    for (int rank = 0; rank < call->handle->ranks; ++rank)
      waitFor(shared->directReadDone[rank], call->sequence, shared);
  } catch (std::exception const& e) {
    shared->failed.store(1, std::memory_order_release);
    std::fprintf(stderr, "UCCL Lite direct read failed: %s\n", e.what());
    std::abort();
  }
}

void CUDART_CB route(void* opaque) {
  std::unique_ptr<Call> call(static_cast<Call*>(opaque));
  auto* h = call->handle;
  try {
    bindCallbackCpu(h->device);
    Shared* s = h->shared;
    s->operation[h->rank] = call->operation;
    s->count[h->rank] = call->count;
    s->arrived[h->rank].store(call->sequence, std::memory_order_release);
    for (int r = 0; r < h->ranks; ++r)
      waitFor(s->arrived[r], call->sequence, s);
    for (int r = 0; r < h->ranks; ++r)
      if (s->operation[r] != call->operation || s->count[r] != call->count)
        throw std::runtime_error("collective metadata mismatch");

    if (call->operation == 1) {
      if (!h->directAllGather) {
        size_t bytes = call->count * sizeof(uint16_t);
        for (int r = 0; r < h->ranks; ++r)
          std::memcpy(h->ownOutput + size_t(r) * bytes, h->input(r), bytes);
      }
    } else if (call->operation == 2) {
      reduceBf16(h, size_t(h->rank) * call->count, h->ownOutput, call->count);
    } else {
      // Each rank reduces a different range. Its source range remains in its
      // own output buffer while peers copy it, so no extra staging is needed.
      size_t begin = call->count * size_t(h->rank) / size_t(h->ranks);
      size_t end = call->count * size_t(h->rank + 1) / size_t(h->ranks);
      reduceBf16(h, begin, h->ownOutput + begin * sizeof(uint16_t),
                 end - begin);
      s->reduced[h->rank].store(call->sequence, std::memory_order_release);
      for (int r = 0; r < h->ranks; ++r)
        waitFor(s->reduced[r], call->sequence, s);
      if (!h->directAllReduce)
        for (int r = 0; r < h->ranks; ++r) {
          if (r == h->rank) continue;
          size_t sourceBegin = call->count * size_t(r) / size_t(h->ranks);
          size_t sourceEnd = call->count * size_t(r + 1) / size_t(h->ranks);
          size_t offset = sourceBegin * sizeof(uint16_t);
          size_t bytes = (sourceEnd - sourceBegin) * sizeof(uint16_t);
          std::memcpy(h->ownOutput + offset, h->output(r) + offset, bytes);
        }
    }
    s->finished[h->rank].store(call->sequence, std::memory_order_release);
    for (int r = 0; r < h->ranks; ++r)
      waitFor(s->finished[r], call->sequence, s);
  } catch (std::exception const& e) {
    h->shared->failed.store(1, std::memory_order_release);
    std::fprintf(stderr, "UCCL Lite multiprocess collective failed: %s\n", e.what());
    // CUDA host callbacks have no error return. Never permit a following H2D
    // copy to publish stale output after a failed collective.
    std::abort();
  }
}

std::unique_ptr<Handle> create(char const* identifier, int ranks, int rank,
                               size_t maxBytes, bool readyOnly) {
  if (!identifier || !*identifier || ranks < 2 || ranks > kMaxRanks ||
      rank < 0 || rank >= ranks || maxBytes == 0)
    throw std::invalid_argument("invalid collective configuration");
  auto h = std::make_unique<Handle>();
  h->name = std::string("/uccl-lite-mp-") + identifier;
  h->rank = rank;
  h->ranks = ranks;
  h->readyOnly = readyOnly;
  check(cudaGetDevice(&h->device));
  h->maxBytes = alignPage(maxBytes);
  char const* packetAg = std::getenv("UCCL_LITE_MP_PACKET_AG");
  char const* pipelineAg = std::getenv("UCCL_LITE_MP_PACKET_AG_PIPELINE");
  char const* packetRs = std::getenv("UCCL_LITE_MP_PACKET_RS");
  char const* rsPipeline = std::getenv("UCCL_LITE_MP_PACKET_RS_PIPELINED");
  char const* cpuRs = std::getenv("UCCL_LITE_MP_PACKET_RS_CPU");
  char const* retainRs = std::getenv("UCCL_LITE_MP_PACKET_RS_RETAIN");
  char const* parallelAr = std::getenv("UCCL_LITE_MP_PACKET_PARALLEL_AR");
  char const* isolateWorker = std::getenv("UCCL_LITE_MP_ISOLATE_PACKET_WORKER");
  char const* destReady = std::getenv("UCCL_LITE_MP_PACKET_RS_DEST_READY");
  char const* hostGate = std::getenv("UCCL_LITE_MP_PACKET_RS_HOST_GATE");
  char const* packetAr = std::getenv("UCCL_LITE_MP_PACKET_AR");
  char const* replicate = std::getenv("UCCL_LITE_MP_PACKET_REPLICATE_AR");
  char const* hierarchy = std::getenv("UCCL_LITE_MP_PACKET_HIERARCHICAL");
  char const* hierarchyRs = std::getenv("UCCL_LITE_MP_PACKET_HIERARCHICAL_RS");
  char const* fastSum = std::getenv("UCCL_LITE_MP_FAST_SUM");
  char const* splitCredits = std::getenv("UCCL_LITE_MP_PACKET_SPLIT_CREDITS");
  char const* scratch = std::getenv("UCCL_LITE_MP_PACKET_SCRATCH");
  h->packetAllGather =
      ranks == 8 && packetAg && std::strcmp(packetAg, "1") == 0;
  h->pipelinedAllGather =
      h->packetAllGather && pipelineAg && std::strcmp(pipelineAg, "1") == 0;
  h->packetReduceScatter =
      ranks == 8 && packetRs && std::strcmp(packetRs, "1") == 0;
  h->pipelinedReduceScatter =
      h->packetReduceScatter && rsPipeline && std::strcmp(rsPipeline, "1") == 0;
  h->cpuReduceScatter =
      h->packetReduceScatter && cpuRs && std::strcmp(cpuRs, "1") == 0;
  h->retainOwnReduceScatter =
      h->cpuReduceScatter && retainRs && std::strcmp(retainRs, "1") == 0;
  h->destReadyReduceScatter =
      h->cpuReduceScatter && destReady && std::strcmp(destReady, "1") == 0;
  h->hostGateReduceScatter =
      h->packetReduceScatter && hostGate && std::strcmp(hostGate, "1") == 0;
  h->packetAllReduce =
      ranks == 8 && packetAr && std::strcmp(packetAr, "1") == 0;
  char const* smallAr = std::getenv("UCCL_LITE_MP_PACKET_AR_GPU_MAX_BYTES");
  if (h->packetAllReduce && smallAr && *smallAr) {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(smallAr, &end, 10);
    if (end == smallAr || *end || parsed % sizeof(uint16_t) ||
        parsed > uccl_lite::multiprocess::kTransportPacketBytes)
      throw std::invalid_argument("invalid small AR GPU size");
    h->smallArGpuMaxBytes = size_t(parsed);
  }
  char const* smallRs = std::getenv("UCCL_LITE_MP_PACKET_RS_GPU_MAX_BYTES");
  if (h->cpuReduceScatter && smallRs && *smallRs) {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(smallRs, &end, 10);
    if (end == smallRs || *end || parsed % (size_t(ranks) * sizeof(uint16_t)) ||
        parsed > uccl_lite::multiprocess::kTransportPacketBytes)
      throw std::invalid_argument("invalid small RS GPU size");
    h->smallRsGpuMaxBytes = size_t(parsed);
  }
  if (readyOnly) {
    if (ranks != 8 || maxBytes != 65536)
      throw std::invalid_argument("ready RS requires eight ranks and 64 KiB");
    h->packetAllGather = false;
    h->pipelinedAllGather = false;
    h->packetReduceScatter = true;
    h->pipelinedReduceScatter = false;
    h->cpuReduceScatter = false;
    h->retainOwnReduceScatter = false;
    h->destReadyReduceScatter = false;
    h->hostGateReduceScatter = false;
    h->packetAllReduce = false;
    h->smallArGpuMaxBytes = 0;
    h->smallRsGpuMaxBytes = 65536;
  }
  h->parallelAllReduce =
      h->packetAllReduce && parallelAr && std::strcmp(parallelAr, "1") == 0;
  h->isolatePacketWorker = (h->cpuReduceScatter || h->packetAllReduce) &&
                           isolateWorker &&
                           std::strcmp(isolateWorker, "1") == 0;
  h->replicateAllReduce =
      h->packetAllReduce && replicate && std::strcmp(replicate, "1") == 0;
  if (h->replicateAllReduce) {
    int firstNode = deviceNumaNode(0);
    int secondNode = deviceNumaNode(4);
    if (firstNode == secondNode ||
        deviceNumaNode(h->device) != (rank < 4 ? firstNode : secondNode))
      throw std::invalid_argument("replicated AR requires 4+4 NUMA rank order");
  }
  h->hierarchicalPacket = (h->cpuReduceScatter || h->packetAllReduce) &&
                          hierarchy && std::strcmp(hierarchy, "1") == 0;
  h->hierarchicalReduceScatter = h->cpuReduceScatter && h->hierarchicalPacket &&
                                 hierarchyRs &&
                                 std::strcmp(hierarchyRs, "1") == 0;
  h->fastPacketSum = (h->cpuReduceScatter || h->packetAllReduce) && fastSum &&
                     std::strcmp(fastSum, "1") == 0 &&
                     __builtin_cpu_supports("avx512f") &&
                     __builtin_cpu_supports("avx512bw");
  h->splitPacketCredits = (h->cpuReduceScatter || h->packetAllReduce) &&
                          splitCredits && std::strcmp(splitCredits, "1") == 0;
  h->packetScratch =
      h->splitPacketCredits && scratch && std::strcmp(scratch, "1") == 0;
  if (h->splitPacketCredits && h->destReadyReduceScatter)
    throw std::invalid_argument(
        "split packet credits require flat CPU routing");
  if (h->hierarchicalPacket && h->splitPacketCredits && !h->packetScratch)
    throw std::invalid_argument("hierarchical split credits require scratch");
  if (h->retainOwnReduceScatter && h->destReadyReduceScatter)
    throw std::invalid_argument("retained RS requires flat packet routing");
  if (rank == 0 && readyOnly)
    std::fprintf(stderr, "UCCL Lite packet mode: GPU-only ready RS\n");
  else if (rank == 0)
    std::fprintf(stderr,
                 "UCCL Lite packet mode AG=%d RS=%d AR=%d retain_rs=%d "
                 "split_credits=%d scratch=%d pipeline_ag=%d hier_rs=%d\n",
                 int(h->packetAllGather), int(h->cpuReduceScatter),
                 int(h->packetAllReduce), int(h->retainOwnReduceScatter),
                 int(h->splitPacketCredits), int(h->packetScratch),
                 int(h->pipelinedAllGather), int(h->hierarchicalReduceScatter));
  if (h->cpuReduceScatter) h->pipelinedReduceScatter = false;
  h->mappedBytes = alignPage(sizeof(Shared)) + 2 * size_t(ranks) * h->maxBytes +
                   alignPage(sizeof(uccl_lite::multiprocess::PacketShared));
  if (rank == 0) {
    h->fd = shm_open(h->name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (h->fd < 0) throw std::runtime_error("shm_open creator failed");
    if (ftruncate(h->fd, h->mappedBytes) != 0)
      throw std::runtime_error("ftruncate shared collective failed");
  } else {
    auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while ((h->fd = shm_open(h->name.c_str(), O_RDWR, 0600)) < 0) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("waiting for shared collective timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    struct stat status {};
    while (fstat(h->fd, &status) != 0 ||
           size_t(status.st_size) < h->mappedBytes) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("shared collective size timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  h->mapping = mmap(nullptr, h->mappedBytes, PROT_READ | PROT_WRITE,
                    MAP_SHARED, h->fd, 0);
  if (h->mapping == MAP_FAILED) {
    h->mapping = nullptr;
    throw std::runtime_error("mmap collective failed");
  }
  h->shared = static_cast<Shared*>(h->mapping);
  if (rank == 0) {
    new (h->mapping) Shared{};
    h->shared->ranks = ranks;
    h->shared->maxBytes = h->maxBytes;
    h->shared->smallArGpuMaxBytes = h->smallArGpuMaxBytes;
    h->shared->smallRsGpuMaxBytes = h->smallRsGpuMaxBytes;
    h->shared->packetMode =
        int(h->packetAllGather) | (int(h->packetReduceScatter) << 1) |
        (int(h->pipelinedReduceScatter) << 2) |
        (int(h->cpuReduceScatter) << 3) | (int(h->packetAllReduce) << 4) |
        (int(h->hierarchicalPacket) << 5) | (int(h->replicateAllReduce) << 6) |
        (int(h->destReadyReduceScatter) << 7) |
        (int(h->hostGateReduceScatter) << 8) | (int(h->fastPacketSum) << 9) |
        (int(h->splitPacketCredits) << 10) | (int(h->packetScratch) << 11) |
        (int(h->retainOwnReduceScatter) << 12) |
        (int(h->parallelAllReduce) << 13) |
        (int(h->hierarchicalReduceScatter) << 14) |
        (int(h->pipelinedAllGather) << 15);
    h->shared->magic.store(kMagic, std::memory_order_release);
  } else {
    auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (h->shared->magic.load(std::memory_order_acquire) != kMagic) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("shared collective initialization timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (h->shared->ranks != ranks || h->shared->maxBytes != h->maxBytes ||
        h->shared->smallArGpuMaxBytes != h->smallArGpuMaxBytes ||
        h->shared->smallRsGpuMaxBytes != h->smallRsGpuMaxBytes ||
        h->shared->packetMode !=
            (int(h->packetAllGather) | (int(h->packetReduceScatter) << 1) |
             (int(h->pipelinedReduceScatter) << 2) |
             (int(h->cpuReduceScatter) << 3) | (int(h->packetAllReduce) << 4) |
             (int(h->hierarchicalPacket) << 5) |
             (int(h->replicateAllReduce) << 6) |
             (int(h->destReadyReduceScatter) << 7) |
             (int(h->hostGateReduceScatter) << 8) |
             (int(h->fastPacketSum) << 9) | (int(h->splitPacketCredits) << 10) |
             (int(h->packetScratch) << 11) |
             (int(h->retainOwnReduceScatter) << 12) |
             (int(h->parallelAllReduce) << 13) |
             (int(h->hierarchicalReduceScatter) << 14) |
             (int(h->pipelinedAllGather) << 15)))
      throw std::runtime_error("shared collective layout mismatch");
  }
  if (h->packetAllGather || h->packetReduceScatter || h->packetAllReduce) {
    h->packetHost = reinterpret_cast<uccl_lite::multiprocess::PacketShared*>(
        h->packetRegion());
    char const* bindNuma = std::getenv("UCCL_LITE_MP_BIND_PACKET_NUMA");
    int packetNode = bindNuma && std::strcmp(bindNuma, "1") == 0
                         ? deviceNumaNode(h->device)
                         : -1;
    for (int slot = 0; slot < 2; ++slot) {
      if (packetNode >= 0)
        bindPacketMemory(h->packetHost->input[slot][rank],
                         uccl_lite::multiprocess::kTransportPacketBytes,
                         packetNode);
      std::memset(h->packetHost->input[slot][rank], 0,
                  uccl_lite::multiprocess::kTransportPacketBytes);
      if (h->cpuReduceScatter || h->packetAllReduce) {
        int localBank = rank / 4;
        for (int bank = 0; bank < (h->replicateAllReduce ? 2 : 1); ++bank) {
          int selectedBank = h->replicateAllReduce ? bank : localBank;
          int bankNode = h->replicateAllReduce
                             ? deviceNumaNode(selectedBank == 0 ? 0 : 4)
                             : packetNode;
          if (bankNode >= 0)
            bindPacketMemory(
                h->packetHost->result[slot][selectedBank][rank],
                uccl_lite::multiprocess::kRsStripeElements * sizeof(uint16_t),
                bankNode);
          std::memset(
              h->packetHost->result[slot][selectedBank][rank], 0,
              uccl_lite::multiprocess::kRsStripeElements * sizeof(uint16_t));
        }
        if (h->hierarchicalPacket) {
          char const* remotePart =
              std::getenv("UCCL_LITE_MP_BIND_REMOTE_PARTIAL");
          if (packetNode >= 0 && remotePart &&
              std::strcmp(remotePart, "1") == 0) {
            size_t partBytes =
                uccl_lite::multiprocess::kRsStripeElements * sizeof(uint16_t);
            bindPacketMemory(h->packetHost->partial[slot][rank][0], partBytes,
                             packetNode);
            bindPacketMemory(h->packetHost->partial[slot][rank][1], partBytes,
                             deviceNumaNode(rank ^ 4));
          }
          std::memset(h->packetHost->partial[slot][rank], 0,
                      2 * uccl_lite::multiprocess::kRsStripeElements *
                          sizeof(uint16_t));
        }
      }
    }
    h->shared->packetTouched[rank].store(1, std::memory_order_release);
    for (int peer = 0; peer < ranks; ++peer)
      waitFor(h->shared->packetTouched[peer], 1, h->shared);
    check(cudaHostRegister(h->packetHost,
                           sizeof(uccl_lite::multiprocess::PacketShared),
                           cudaHostRegisterPortable | cudaHostRegisterMapped));
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&h->packetDevice),
                                   h->packetHost, 0));
    h->agDevice = reinterpret_cast<uint16_t*>(
        reinterpret_cast<char*>(h->packetDevice) +
        offsetof(uccl_lite::multiprocess::PacketShared, input));
    if (h->pipelinedAllGather || h->pipelinedReduceScatter ||
        h->cpuReduceScatter || h->packetAllReduce) {
      check(cudaMalloc(reinterpret_cast<void**>(&h->crews),
                       2 * sizeof(uccl_lite::multiprocess::CrewState)));
      check(cudaMemset(h->crews, 0,
                       2 * sizeof(uccl_lite::multiprocess::CrewState)));
    }
  }
  h->ownInput = h->input(rank);
  check(cudaHostRegister(h->ownInput, h->maxBytes, cudaHostRegisterPortable));
  h->ownOutput = h->output(rank);
  check(cudaHostRegister(h->ownOutput, h->maxBytes, cudaHostRegisterPortable));
  if (h->cpuReduceScatter || h->packetAllReduce) {
    int helper = mappedCpu("UCCL_LITE_MP_PACKET_HELPER_CPUS", h->device);
    if (h->parallelAllReduce && helper < 0)
      throw std::invalid_argument("parallel AR requires helper CPU map");
    if (h->parallelAllReduce)
      h->arLaneThread = std::thread(
          [raw = h.get(), helper] { runParallelArLane(raw, helper); });
    else if (helper >= 0)
      h->rsHelper = std::make_unique<uccl_lite::cpu_router::CpuExecutor>(
          std::vector<int>{helper});
  }
  if (h->cpuReduceScatter || h->packetAllReduce)
    h->rsThread = std::thread([raw = h.get()] { runRsPacketWorker(raw); });
  if (h->isolatePacketWorker) {
    int worker = mappedCpu("UCCL_LITE_MP_PACKET_WORKER_CPUS", h->device);
    if (worker < 0)
      throw std::invalid_argument("worker isolation requires CPU map");
    auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (!h->rsWorkerTid.load(std::memory_order_acquire) ||
           (h->parallelAllReduce &&
            !h->arLaneTid.load(std::memory_order_acquire))) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("packet worker did not start");
      std::this_thread::yield();
    }
    isolatePacketWorker(h.get(), worker);
  }
  char const* direct = std::getenv("UCCL_LITE_MP_DIRECT_AG");
  h->directAllGather = !readyOnly && direct && std::strcmp(direct, "1") == 0;
  if (h->directAllGather) {
    for (int peer = 0; peer < ranks; ++peer) {
      if (peer == rank) continue;
      char* address = h->input(peer);
      check(cudaHostRegister(address, h->maxBytes, cudaHostRegisterPortable));
      h->registeredPeerInputs.push_back(address);
    }
  }
  char const* directAr = std::getenv("UCCL_LITE_MP_DIRECT_AR");
  if (!readyOnly && directAr && std::strcmp(directAr, "1") == 0) {
    for (int peer = 0; peer < ranks; ++peer) {
      if (peer == rank) continue;
      char* address = h->output(peer);
      check(cudaHostRegister(address, h->maxBytes, cudaHostRegisterPortable));
      h->registeredPeerOutputs.push_back(address);
    }
    h->directAllReduce = true;
  }
  if (h->shared->openedCount.fetch_add(1, std::memory_order_acq_rel) + 1 ==
      ranks)
    shm_unlink(h->name.c_str());
  return h;
}
}  // namespace

extern "C" void* uccl_lite_mp_create(char const* identifier, int ranks,
                                     int rank, size_t max_bytes) {
  try {
    return create(identifier, ranks, rank, max_bytes, false).release();
  } catch (std::exception const& e) {
    lastError = e.what();
    return nullptr;
  }
}
extern "C" void* uccl_lite_mp_create_rs_ready(char const* identifier,
                                                int ranks, int rank) {
  try {
    return create(identifier, ranks, rank, 65536, true).release();
  } catch (std::exception const& e) {
    lastError = e.what();
    return nullptr;
  }
}
static int enqueueImpl(void* handle, int operation, void const* input,
                       void const* auxiliary, void* output, void* gemmOutput,
                       uint32_t const* ready, size_t count, void* stream,
                       bool ordered = false) {
  try {
    auto* h = static_cast<Handle*>(handle);
    if (!h || !input || !output ||
        (operation != 1 && operation != 2 && operation != 3) ||
        count > SIZE_MAX / (sizeof(uint16_t) * size_t(h->ranks)))
      throw std::invalid_argument("invalid collective call");
    size_t inputBytes = count * sizeof(uint16_t) *
                        (operation == 2 ? size_t(h->ranks) : 1);
    size_t outputBytes = count * sizeof(uint16_t) *
                         (operation == 1 ? size_t(h->ranks) : 1);
    if (inputBytes > h->maxBytes || outputBytes > h->maxBytes)
      throw std::invalid_argument("collective exceeds prepared capacity");
    bool sum2GpuRs = operation == 2 && h->packetReduceScatter &&
                     (h->readyOnly || h->cpuReduceScatter) &&
                     !h->pipelinedReduceScatter &&
                     h->smallRsGpuMaxBytes && inputBytes >= 2048 &&
                     inputBytes <= h->smallRsGpuMaxBytes;
    bool agGemm = operation == 1 && gemmOutput;
    if (agGemm && (count != 4096 || !auxiliary || !h->packetAllGather))
      throw std::invalid_argument("AG+GEMM requires 4096-element packet AG");
    if (auxiliary && !sum2GpuRs && !agGemm)
      throw std::invalid_argument("unsupported fused collective input");
    if (ready && (!sum2GpuRs || count != 4096 || !auxiliary ||
                  h->hostGateReduceScatter))
      throw std::invalid_argument("ready RS requires 4096-element GPU packet");
    auto cudaStream = static_cast<cudaStream_t>(stream);
    // The ordinary API requires one stream to protect shared staging. The
    // ordered API's caller supplies CUDA dependencies when switching streams.
    if (!ordered && h->lastStream && h->lastStream != cudaStream)
      throw std::invalid_argument("communicator stream changed");
    h->lastStream = cudaStream;
    if ((operation == 1 && h->packetAllGather) ||
        (operation == 2 && h->packetReduceScatter) ||
        (operation == 3 && h->packetAllReduce)) {
      if (operation == 2 && h->hostGateReduceScatter) {
        h->shared->hostPacketArrived[h->rank].store(h->nextSequence,
                                                    std::memory_order_release);
        for (int peer = 0; peer < h->ranks; ++peer)
          waitFor(h->shared->hostPacketArrived[peer], h->nextSequence,
                  h->shared);
      }
      if (count) {
        auto* mapped = h->packetDevice;
        uint64_t base = h->nextSequence * 65536ULL;
        if (operation == 3 && h->smallArGpuMaxBytes &&
            inputBytes <= h->smallArGpuMaxBytes) {
          check(uccl_lite::multiprocess::launchArPacketSmall(
              mapped, input, output, count, h->rank, base + 1,
              h->lastPacketGeneration[0], cudaStream));
          h->lastPacketGeneration[0] = base + 1;
          h->lastPacketWasCpu[0] = false;
          ++h->nextSequence;
          return 0;
        }
        uint32_t previousCpuMask = uint32_t(h->lastPacketWasCpu[0]) |
                                   (uint32_t(h->lastPacketWasCpu[1]) << 1);
        bool smallGpuRs = operation == 2 && h->cpuReduceScatter &&
                          h->smallRsGpuMaxBytes && inputBytes >= 2048 &&
                          inputBytes <= h->smallRsGpuMaxBytes;
        bool cpuPacket = operation == 3 ||
                         (operation == 2 && h->cpuReduceScatter && !smallGpuRs);
        size_t packetElements =
            operation == 1 ? uccl_lite::multiprocess::kAgPacketElements
            : operation == 3 && count <= (1UL << 19)
                ? uccl_lite::multiprocess::kRsStripeElements / 2
                : uccl_lite::multiprocess::kRsStripeElements;
        bool pipelineAg = operation == 1 && h->pipelinedAllGather &&
                          outputBytes >= (512UL << 10);
        if (operation == 1) {
          if (agGemm)
            check(uccl_lite::multiprocess::launchAgGemm128x4096Packet(
                mapped, h->agDevice, input, output, auxiliary, gemmOutput,
                h->rank, base, h->lastPacketGeneration[0], cudaStream));
          else if (pipelineAg)
            check(uccl_lite::multiprocess::launchAgPacketPipelined(
                mapped, h->agDevice, h->crews, input, output, count, h->rank,
                base, h->lastPacketGeneration[0], h->lastPacketGeneration[1],
                h->packetBarrierPhase, cudaStream));
          else
            check(uccl_lite::multiprocess::launchAgPacket(
                mapped, h->agDevice, input, output, count, h->rank, base,
                h->lastPacketGeneration[0], h->lastPacketGeneration[1],
                cudaStream));
        } else if (operation == 2 && h->cpuReduceScatter && !smallGpuRs) {
          {
            std::lock_guard<std::mutex> guard(h->rsMutex);
            h->rsJobs.push_back(
                {operation,
                 count,
                 base,
                 {h->lastCpuResultGeneration[0], h->lastCpuResultGeneration[1]},
                 {h->lastCpuResultAllReaders[0], h->lastCpuResultAllReaders[1]},
                 {h->lastPartialGeneration[0], h->lastPartialGeneration[1]}});
          }
          h->rsAvailable.notify_one();
          if (h->destReadyReduceScatter)
            check(uccl_lite::multiprocess::launchRsPacketDestReady(
                mapped, h->crews, input, output, count, h->rank, base,
                h->lastPacketGeneration[0], h->lastPacketGeneration[1],
                h->packetBarrierPhase, cudaStream));
          else
            check(uccl_lite::multiprocess::launchRsPacketCpu(
                mapped, h->crews, input, output, count, h->rank, base,
                h->lastPacketGeneration[0], h->lastPacketGeneration[1],
                h->packetBarrierPhase, previousCpuMask, false,
                h->retainOwnReduceScatter, cudaStream));
        } else if (operation == 3) {
          {
            std::lock_guard<std::mutex> guard(h->rsMutex);
            h->rsJobs.push_back(
                {operation,
                 count,
                 base,
                 {h->lastCpuResultGeneration[0], h->lastCpuResultGeneration[1]},
                 {h->lastCpuResultAllReaders[0], h->lastCpuResultAllReaders[1]},
                 {h->lastPartialGeneration[0], h->lastPartialGeneration[1]}});
          }
          h->rsAvailable.notify_one();
          check(uccl_lite::multiprocess::launchArPacketCpu(
              mapped, h->crews, input, output, count, h->rank, base,
              h->lastPacketGeneration[0], h->lastPacketGeneration[1],
              h->packetBarrierPhase, h->replicateAllReduce, previousCpuMask,
              h->splitPacketCredits, cudaStream));
        } else if (h->pipelinedReduceScatter)
          check(uccl_lite::multiprocess::launchRsPacketPipelined(
              mapped, h->crews, input, output, count, h->rank, base,
              h->lastPacketGeneration[0], h->lastPacketGeneration[1],
              h->packetBarrierPhase, cudaStream));
        else if (ready)
          check(uccl_lite::multiprocess::launchRsReadySum2Packet(
              mapped, input, auxiliary, output, ready, h->rank, base,
              h->lastPacketGeneration[0], cudaStream));
        else if (auxiliary)
          check(uccl_lite::multiprocess::launchRsSum2Packet(
              mapped, input, auxiliary, output, count, h->rank, base,
              h->lastPacketGeneration[0], h->lastPacketGeneration[1],
              cudaStream));
        else
          check(uccl_lite::multiprocess::launchRsPacket(
              mapped, input, output, count, h->rank, base,
              h->lastPacketGeneration[0], h->lastPacketGeneration[1],
              cudaStream));
        size_t routedElements =
            operation == 3 ? (count + size_t(h->ranks) - 1) / size_t(h->ranks)
                           : count;
        size_t tiles = (routedElements + packetElements - 1) / packetElements;
        if (pipelineAg) h->packetBarrierPhase += uint32_t(tiles);
        if (operation == 2 &&
            (h->pipelinedReduceScatter || (h->cpuReduceScatter && !smallGpuRs)))
          h->packetBarrierPhase += uint32_t(tiles);
        if (operation == 3) h->packetBarrierPhase += uint32_t(tiles);
        for (size_t tile = 0; tile < tiles; ++tile) {
          h->lastPacketGeneration[tile & 1] = base + tile + 1;
          h->lastPacketWasCpu[tile & 1] =
              operation == 3 && h->splitPacketCredits;
          if (cpuPacket) h->lastCpuResultGeneration[tile & 1] = base + tile + 1;
          if (cpuPacket) h->lastCpuResultAllReaders[tile & 1] = operation == 3;
          if ((operation == 3 && h->hierarchicalPacket &&
               count >= (8UL << 20)) ||
              (operation == 2 && h->hierarchicalReduceScatter &&
               count >= (1UL << 20)))
            h->lastPartialGeneration[tile & 1] = base + tile + 1;
        }
      }
      ++h->nextSequence;
      return 0;
    }
    if (h->lastDirectSequence) {
      auto wait = std::make_unique<DirectReadCall>(
          DirectReadCall{h, h->lastDirectSequence});
      check(cudaLaunchHostFunc(cudaStream, waitDirectReaders, wait.get()));
      wait.release();
      h->lastDirectSequence = 0;
    }
    check(cudaMemcpyAsync(h->ownInput, input, inputBytes,
                          cudaMemcpyDeviceToHost, cudaStream));
    auto call = std::make_unique<Call>(Call{h, h->nextSequence, operation, count});
    check(cudaLaunchHostFunc(cudaStream, route, call.get()));
    call.release();
    if (operation == 1 && h->directAllGather) {
      size_t pieceBytes = count * sizeof(uint16_t);
      for (int r = 0; r < h->ranks; ++r)
        check(cudaMemcpyAsync(
            static_cast<char*>(output) + size_t(r) * pieceBytes, h->input(r),
            pieceBytes, cudaMemcpyHostToDevice, cudaStream));
    } else if (operation == 3 && h->directAllReduce) {
      for (int r = 0; r < h->ranks; ++r) {
        size_t begin = count * size_t(r) / size_t(h->ranks);
        size_t end = count * size_t(r + 1) / size_t(h->ranks);
        size_t offset = begin * sizeof(uint16_t);
        size_t bytes = (end - begin) * sizeof(uint16_t);
        check(cudaMemcpyAsync(static_cast<char*>(output) + offset,
                              h->output(r) + offset, bytes,
                              cudaMemcpyHostToDevice, cudaStream));
      }
    } else {
      check(cudaMemcpyAsync(output, h->ownOutput, outputBytes,
                            cudaMemcpyHostToDevice, cudaStream));
    }
    if ((operation == 1 && h->directAllGather) ||
        (operation == 3 && h->directAllReduce)) {
      auto done =
          std::make_unique<DirectReadCall>(DirectReadCall{h, h->nextSequence});
      check(cudaLaunchHostFunc(cudaStream, publishDirectReadDone, done.get()));
      done.release();
      h->lastDirectSequence = h->nextSequence;
    }
    ++h->nextSequence;
    return 0;
  } catch (std::exception const& e) {
    lastError = e.what();
    return -1;
  }
}
extern "C" int uccl_lite_mp_enqueue(void* handle, int operation,
                                    void const* input, void* output,
                                    size_t count, void* stream) {
  return enqueueImpl(handle, operation, input, nullptr, output, nullptr,
                     nullptr, count, stream);
}
extern "C" int uccl_lite_mp_enqueue_ordered(void* handle, int operation,
                                            void const* input, void* output,
                                            size_t count, void* stream) {
  return enqueueImpl(handle, operation, input, nullptr, output, nullptr,
                     nullptr, count, stream, true);
}
extern "C" int uccl_lite_mp_enqueue_ag_gemm_128x4096(
    void* handle, void const* input, void* gathered, void const* weight,
    void* gemm_output, void* stream) {
  if (!weight || !gemm_output) {
    lastError = "AG+GEMM weight or output is null";
    return -1;
  }
  return enqueueImpl(handle, 1, input, weight, gathered, gemm_output, nullptr,
                     4096, stream);
}
extern "C" int uccl_lite_mp_enqueue_rs_sum2(void* handle, void const* input,
                                              void const* auxiliary,
                                              void* output, size_t shard_count,
                                              void* stream) {
  if (!auxiliary) {
    lastError = "sum2 auxiliary input is null";
    return -1;
  }
  return enqueueImpl(handle, 2, input, auxiliary, output, nullptr, nullptr,
                     shard_count, stream);
}
extern "C" int uccl_lite_mp_enqueue_rs_ready_sum2(
    void* handle, void const* input, void const* auxiliary, void* output,
    uint32_t const* ready, void* stream) {
  if (!auxiliary || !ready) {
    lastError = "ready RS input is null";
    return -1;
  }
  return enqueueImpl(handle, 2, input, auxiliary, output, nullptr, ready, 4096,
                     stream);
}
extern "C" int uccl_lite_mp_destroy(void* handle) {
  delete static_cast<Handle*>(handle);
  return 0;
}
extern "C" char const* uccl_lite_mp_last_error(void) {
  return lastError.c_str();
}

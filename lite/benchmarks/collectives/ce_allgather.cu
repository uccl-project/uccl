// Experimental single-process, four-GPU CE schedules. No dispatch changes.
#include "../../tests/collectives/ce_allgather_check.cuh"
#include "readme_allgather_reference.cuh"
#include "uccl_lite/cpu_router/packet_pipeline.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <nccl.h>
#include <pthread.h>
#include <sched.h>
using namespace uccl_lite::cpu_router;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void check(ncclResult_t s) {
  if (s != ncclSuccess) throw std::runtime_error(ncclGetErrorString(s));
}
void check(CUresult status) {
  if (status != CUDA_SUCCESS) {
    char const* message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(message ? message : "CUDA driver error");
  }
}
cpu_set_t const& initialAffinity() {
  static cpu_set_t allowed = [] {
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask))
      throw std::runtime_error("cannot query initial CPU affinity");
    return mask;
  }();
  return allowed;
}
__global__ void smCopy(uint32_t* dst, uint32_t const* src, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  if ((reinterpret_cast<uintptr_t>(dst) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(src) % 16 == 0)) {
    for (size_t i = tid; i < n / 4; i += stride)
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<uint4 const*>(src)[i];
    for (size_t i = n / 4 * 4 + tid; i < n; i += stride) dst[i] = src[i];
  } else {
    for (size_t i = tid; i < n; i += stride) dst[i] = src[i];
  }
}
__global__ void delayMemopProducer() {
  for (int i = 0; i < 10000; ++i) __nanosleep(100);
}
struct DownloadPacket {
  uint32_t const* source[4];
};
__global__ void downloadPacket(DownloadPacket packet, uint32_t* output,
                               size_t count, size_t offset, size_t n) {
  int source = blockIdx.x % 4;
  size_t tid = (blockIdx.x / 4) * blockDim.x + threadIdx.x;
  size_t stride = (gridDim.x / 4) * blockDim.x;
  uint32_t* dst = output + size_t(source) * count + offset;
  uint32_t const* src = packet.source[source];
  if ((reinterpret_cast<uintptr_t>(dst) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(src) % 16 == 0)) {
    for (size_t i = tid; i < n / 4; i += stride)
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<uint4 const*>(src)[i];
    for (size_t i = n / 4 * 4 + tid; i < n; i += stride) dst[i] = src[i];
  } else {
    for (size_t i = tid; i < n; i += stride) dst[i] = src[i];
  }
}
// One CTA: four 64-thread source crews, preserving vector/tail handling.
__global__ void downloadSmall(DownloadPacket packet, uint32_t* output,
                              size_t count) {
  int source = threadIdx.x / 64;
  size_t lane = threadIdx.x % 64;
  auto* dst = output + size_t(source) * count;
  auto* src = packet.source[source];
  if ((reinterpret_cast<uintptr_t>(dst) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(src) % 16 == 0)) {
    for (size_t i = lane; i < count / 4; i += 64)
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<uint4 const*>(src)[i];
    for (size_t i = count / 4 * 4 + lane; i < count; i += 64) dst[i] = src[i];
  } else {
    for (size_t i = lane; i < count; i += 64) dst[i] = src[i];
  }
}
// Benchmark-only persistent CPU submitters. Construction and teardown are
// outside timing, but both phase handoffs are inside every measured call.
struct SubmitPair {
  std::array<std::thread, 2> threads;
  std::array<std::atomic<unsigned>, 2> done{};
  std::array<std::exception_ptr, 2> errors{};
  std::atomic<unsigned> request{0};
  std::atomic<bool> stop{false};
  int phase = 0;
  unsigned generation = 0;
  explicit SubmitPair(std::function<void(int, int)> fn) {
    for (int node = 0; node < 2; ++node) {
      done[node].store(0);
      threads[node] = std::thread([&, node, fn] {
        try {
          cpu_set_t allowed = initialAffinity();
          std::ifstream in("/sys/devices/system/node/node" +
                           std::to_string(node) + "/cpulist");
          std::string list, range;
          in >> list;
          std::istringstream parser(list);
          int chosen = -1;
          while (std::getline(parser, range, ',') && chosen < 0) {
            size_t dash = range.find('-');
            int first = std::stoi(range),
                last = dash == std::string::npos
                           ? first
                           : std::stoi(range.substr(dash + 1));
            for (int cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu)
              if (CPU_ISSET(cpu, &allowed)) {
                chosen = cpu;
                break;
              }
          }
          if (chosen < 0)
            throw std::runtime_error("no permitted NUMA-local worker CPU");
          cpu_set_t mask;
          CPU_ZERO(&mask);
          CPU_SET(chosen, &mask);
          if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
            throw std::runtime_error("cannot pin submit worker");
          std::fprintf(stderr, "# submit worker node=%d cpu=%d GPUs=%d,%d\n",
                       node, chosen, node * 2, node * 2 + 1);
        } catch (...) {
          errors[node] = std::current_exception();
        }
        unsigned seen = 0;
        while (!stop.load(std::memory_order_acquire)) {
          unsigned next = request.load(std::memory_order_acquire);
          if (next == seen) {
            asm volatile("pause");
            continue;
          }
          if (!errors[node]) {
            try {
              fn(node, phase);
            } catch (...) {
              errors[node] = std::current_exception();
            }
          }
          seen = next;
          done[node].store(next, std::memory_order_release);
        }
      });
    }
  }
  void run(int nextPhase) {
    phase = nextPhase;
    request.store(++generation, std::memory_order_release);
    for (int n = 0; n < 2; ++n)
      while (done[n].load(std::memory_order_acquire) != generation)
        asm volatile("pause");
    for (auto& error : errors)
      if (error) std::rethrow_exception(error);
  }
  ~SubmitPair() {
    stop.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
  }
};
// Four persistent host workers, each bound to one GPU for its lifetime.
struct FixedFour {
  struct alignas(64) Progress {
    std::atomic<unsigned> done{0}, uploaded{0};
    std::exception_ptr error;
  };
  std::array<Progress, 4> progress;
  std::array<std::thread, 4> threads;
  std::atomic<unsigned> request{0}, initialized{0};
  std::atomic<bool> stop{false}, failed{false};
  int phase = 0;
  unsigned generation = 0;
  explicit FixedFour(std::function<void(int, int, FixedFour&)> fn) {
    try {
      for (int rank = 0; rank < 4; ++rank) {
        threads[rank] = std::thread([&, rank, fn] {
          try {
            cpu_set_t allowed = initialAffinity();
            int node = rank / 2;
            std::ifstream in("/sys/devices/system/node/node" +
                             std::to_string(node) + "/cpulist");
            std::string list, range;
            in >> list;
            std::istringstream parser(list);
            int chosen = -1, ordinal = rank % 2;
            while (std::getline(parser, range, ',') && chosen < 0) {
              size_t dash = range.find('-');
              int first = std::stoi(range),
                  last = dash == std::string::npos
                             ? first
                             : std::stoi(range.substr(dash + 1));
              for (int cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu)
                if (CPU_ISSET(cpu, &allowed) && ordinal-- == 0) {
                  chosen = cpu;
                  break;
                }
            }
            if (chosen < 0)
              throw std::runtime_error("need two permitted CPUs per NUMA node");
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(chosen, &mask);
            if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
              throw std::runtime_error("cannot pin fixed GPU worker");
            check(
                cudaSetDevice(rank));  // Only here, outside collective timing.
            std::fprintf(stderr, "# fixed worker GPU=%d node=%d cpu=%d\n", rank,
                         node, chosen);
          } catch (...) {
            progress[rank].error = std::current_exception();
            failed.store(true, std::memory_order_release);
          }
          initialized.fetch_add(1, std::memory_order_acq_rel);
          unsigned seen = 0;
          while (!stop.load(std::memory_order_acquire)) {
            unsigned next = request.load(std::memory_order_acquire);
            if (next == seen) {
              asm volatile("pause");
              continue;
            }
            if (!progress[rank].error) {
              try {
                fn(rank, phase, *this);
              } catch (...) {
                progress[rank].error = std::current_exception();
                failed.store(true, std::memory_order_release);
              }
            }
            seen = next;
            progress[rank].done.store(next, std::memory_order_release);
          }
        });
      }
      while (initialized.load(std::memory_order_acquire) != 4)
        asm volatile("pause");
      for (auto& p : progress)
        if (p.error) std::rethrow_exception(p.error);
    } catch (...) {
      stop.store(true, std::memory_order_release);
      for (auto& thread : threads)
        if (thread.joinable()) thread.join();
      throw;
    }
  }
  void uploadBarrier(int rank) {
    unsigned tag = request.load(std::memory_order_acquire);
    progress[rank].uploaded.store(tag, std::memory_order_release);
    for (auto& p : progress)
      while (p.uploaded.load(std::memory_order_acquire) != tag) {
        if (failed.load(std::memory_order_acquire))
          throw std::runtime_error("fixed GPU worker failed during upload");
        asm volatile("pause");
      }
  }
  void run(int nextPhase) {
    phase = nextPhase;
    request.store(++generation, std::memory_order_release);
    for (auto& p : progress)
      while (p.done.load(std::memory_order_acquire) != generation)
        asm volatile("pause");
    for (auto& p : progress)
      if (p.error) std::rethrow_exception(p.error);
  }
  ~FixedFour() {
    stop.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
  }
};
struct Experiment {
  std::array<uint32_t*, 4> input{}, output{};
  std::array<unsigned*, 4> errors{};
  std::array<cudaStream_t, 4> stream{};
  std::array<cudaEvent_t, 4> ready{};
  std::array<ncclComm_t, 4> nccl{};
  std::array<cudaStream_t, 4> upload{};
  std::array<std::array<RegisteredMemory, 2>, 4> host;
  std::array<std::array<cudaEvent_t, 2>, 4> uploaded{}, consumed{};
  std::array<RegisteredMemory, 4> memopFlags;
  std::array<std::array<CUdeviceptr, 4>, 4> memopAddress{};
  std::array<bool, 4> memopFlush{};
  unsigned memopEpoch = 0;
  std::array<cudaGraphExec_t, 4> peerGraphs{};
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  bool fullStaging = false;
  std::array<RegisteredMemory, 4> referenceHost;
  uint32_t* referenceFlags = nullptr;
  uint32_t referenceGeneration = 0;
  size_t chunkElements = 0;
  size_t count;
  Experiment(size_t bytes) : count(bytes / 16) {
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      check(cudaMalloc(&input[r], count * 4));
      check(cudaMalloc(&output[r], bytes));
      check(cudaMalloc(&errors[r], sizeof(unsigned)));
      check(cudaStreamCreateWithFlags(&stream[r], cudaStreamNonBlocking));
      check(cudaStreamCreateWithFlags(&upload[r], cudaStreamNonBlocking));
      for (int slot = 0; slot < 2; ++slot) {
        check(cudaEventCreateWithFlags(&uploaded[r][slot],
                                       cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&consumed[r][slot],
                                       cudaEventDisableTiming));
      }
      check(cudaEventCreateWithFlags(&ready[r], cudaEventDisableTiming));
    }
    int devices[] = {0, 1, 2, 3};
    check(ncclCommInitAll(nccl.data(), 4, devices));
  }
  void sync() {
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      check(cudaStreamSynchronize(stream[r]));
    }
  }
  void prepare(size_t chunk, bool full = false, bool hostPackets = true) {
    fullStaging = full;
    sync();
    for (int r = 0; r < 4; ++r) {
      if (peerGraphs[r]) {
        check(cudaSetDevice(r));
        check(cudaGraphExecDestroy(peerGraphs[r]));
        peerGraphs[r] = nullptr;
      }
    }
    if (executable) check(cudaGraphExecDestroy(executable));
    if (graph) check(cudaGraphDestroy(graph));
    executable = nullptr;
    graph = nullptr;
    chunk = std::min(chunk, count);
    chunkElements = chunk;
    for (int r = 0; r < 4; ++r)
      for (int slot = 0; slot < 2; ++slot)
        host[r][slot] =
            hostPackets ? RegisteredMemory::pinned(
                              (full && slot == 0 ? count : chunk) * 4, r / 2)
                        : RegisteredMemory{};
  }

  void prepareReference() {
    if (count * 16 <= (512UL << 10)) {
      for (int r = 0; r < 4; ++r) {
        check(cudaSetDevice(r));
        for (int d = 0; d < 4; ++d) {
          if (d == r) continue;
          int access = 0;
          check(cudaDeviceCanAccessPeer(&access, r, d));
          if (!access)
            throw std::runtime_error("FP32 push reference requires P2P");
          auto status = cudaDeviceEnablePeerAccess(d, 0);
          if (status == cudaErrorPeerAccessAlreadyEnabled)
            cudaGetLastError();
          else
            check(status);
        }
      }
    } else if (count * 16 <= (16UL << 20)) {
      for (int r = 0; r < 4; ++r)
        referenceHost[r] = RegisteredMemory::pinned(count * 4, r / 2);
      check(cudaHostAlloc(&referenceFlags, 8 * 128,
                          cudaHostAllocPortable | cudaHostAllocMapped));
      std::memset(referenceFlags, 0, 8 * 128);
    }
  }
  void referenceSmallMedium() {
    if (count * 16 <= (512UL << 10)) {
      int threads = count * 16 == (64UL << 10) ? 128 : 256;
      for (int r = 0; r < 4; ++r) {
        check(cudaSetDevice(r));
        uccl_lite_reference::pushSerialGather<<<1, threads, 0, stream[r]>>>(
            reinterpret_cast<float*>(input[r]),
            reinterpret_cast<float*>(output[0]),
            reinterpret_cast<float*>(output[1]),
            reinterpret_cast<float*>(output[2]),
            reinterpret_cast<float*>(output[3]), count, r);
        check(cudaGetLastError());
      }
    } else {
      ++referenceGeneration;
      std::array<float*, 4> hosts;
      for (int r = 0; r < 4; ++r)
        hosts[r] = static_cast<float*>(referenceHost[r].data());
      for (int r = 0; r < 4; ++r) {
        check(cudaSetDevice(r));
        void* args[] = {&input[r],           &output[r], &hosts[0],
                        &hosts[1],           &hosts[2],  &hosts[3],
                        &referenceFlags,     &count,     &r,
                        &referenceGeneration};
        check(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(uccl_lite_reference::miniHostGather),
            dim3(16), dim3(256), args, 0, stream[r]));
      }
    }
  }
  void initialize(int epoch) {
    auto* delayed = std::getenv("UCCL_CE_DELAY_INPUT_RANK");
    int delayedRank = delayed ? std::stoi(delayed) : -1;
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      if (r == delayedRank) {
        delayMemopProducer<<<1, 1, 0, stream[r]>>>();
        check(cudaGetLastError());
      }
      check(cudaMemsetAsync(output[r], 0, count * 16, stream[r]));
      ceFill<<<256, 256, 0, stream[r]>>>(input[r], count, r, epoch);
      check(cudaGetLastError());
      check(cudaEventRecord(ready[r], stream[r]));
    }
    if (!delayed) sync();
  }
  void validate(int epoch) {
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      check(cudaMemsetAsync(errors[r], 0, sizeof(unsigned), stream[r]));
      ceCheck<<<256, 256, 0, stream[r]>>>(output[r], count, epoch, errors[r]);
      check(cudaGetLastError());
      unsigned wrong = 0;
      check(cudaMemcpyAsync(&wrong, errors[r], sizeof(wrong),
                            cudaMemcpyDeviceToHost, stream[r]));
      check(cudaStreamSynchronize(stream[r]));
      if (wrong) throw std::runtime_error("full-output bitwise check failed");
    }
  }
  void copies(bool dma, bool batch = false, bool waitAll = false,
              bool smDownload = false, bool groupedDownload = false) {
    // Same host-packet dataflow for SM and CE: one upload per source,
    // per-source readiness, local input retained, two slots, all-reader credit.
    auto move = [&](void* dst, void const* src, size_t n, cudaMemcpyKind kind,
                    cudaStream_t work) {
      if (dma && !(smDownload && kind != cudaMemcpyDeviceToHost))
        check(cudaMemcpyAsync(dst, src, n * 4, kind, work));
      else {
        smCopy<<<248, 256, 0, work>>>(static_cast<uint32_t*>(dst),
                                      static_cast<uint32_t const*>(src), n);
        check(cudaGetLastError());
      }
    };
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      check(cudaStreamWaitEvent(upload[r], ready[r], 0));
    }
    for (size_t offset = 0, piece = 0; offset < count;
         offset += chunkElements, ++piece) {
      int slot = fullStaging ? 0 : piece % 2;
      auto packet = [&](int r) {
        return static_cast<uint32_t*>(host[r][slot].data()) +
               (fullStaging ? offset : 0);
      };
      size_t n = std::min(chunkElements, count - offset);
      for (int r = 0; r < 4; ++r) {
        check(cudaSetDevice(r));
        // Binds to the previous slot generation before publishing new credits.
        if (!fullStaging)
          for (int d = 0; d < 4; ++d)
            check(cudaStreamWaitEvent(upload[r], consumed[d][slot], 0));
        move(packet(r), input[r] + offset, n, cudaMemcpyDeviceToHost,
             upload[r]);
        check(cudaEventRecord(uploaded[r][slot], upload[r]));
      }
      for (int d = 0; d < 4; ++d) {
        check(cudaSetDevice(d));
        if (groupedDownload) {
          DownloadPacket packetDesc{};
          for (int r = 0; r < 4; ++r) {
            packetDesc.source[r] = r == d ? input[d] + offset : packet(r);
            check(cudaStreamWaitEvent(stream[d], uploaded[r][slot], 0));
          }
          // Grouped SM downloads start after all source uploads complete;
          // no GPU polling on commands that could be queued behind this grid.
          downloadPacket<<<32, 256, 0, stream[d]>>>(packetDesc, output[d],
                                                    count, offset, n);
          check(cudaGetLastError());
          check(cudaEventRecord(consumed[d][slot], stream[d]));
          continue;
        }
        if (batch || waitAll)
          for (int r = 0; r < 4; ++r)
            check(cudaStreamWaitEvent(stream[d], uploaded[r][slot], 0));
        if (batch) {
          void* dsts[4];
          void* srcs[4];
          size_t sizes[4];
          for (int r = 0; r < 4; ++r) {
            dsts[r] = output[d] + r * count + offset;
            srcs[r] = r == d ? input[d] + offset : host[r][slot].data();
            sizes[r] = n * 4;
          }
          cudaMemcpyAttributes attributes{};
          // Host slots are produced asynchronously: source access MUST follow
          // the upload-ready waits. The four destination ranges are disjoint.
          attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
          size_t first = 0, failed = SIZE_MAX;
          check(cudaMemcpyBatchAsync(dsts, srcs, sizes, 4, &attributes, &first,
                                     1, &failed, stream[d]));
        } else
          for (int r = 0; r < 4; ++r) {
            if (!waitAll)
              check(cudaStreamWaitEvent(stream[d], uploaded[r][slot], 0));
            move(output[d] + r * count + offset,
                 r == d ? input[d] + offset : packet(r), n,
                 r == d ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                 stream[d]);
          }
        if (!fullStaging) check(cudaEventRecord(consumed[d][slot], stream[d]));
      }
    }
  }
  // These single-packet candidates require all prior readers complete,
  // established by sync() in this serialized harness. They are NOT async API
  // implementations: reuse credits cannot be omitted for in-flight calls.
  void singleUpload(int first, int last) {
    if (chunkElements < count)
      throw std::runtime_error("single packet exceeds capacity");
    for (int r = first; r < last; ++r) {
      check(cudaSetDevice(r));
      check(cudaStreamWaitEvent(upload[r], ready[r], 0));
      check(cudaMemcpyAsync(host[r][0].data(), input[r], count * 4,
                            cudaMemcpyDeviceToHost, upload[r]));
      check(cudaEventRecord(uploaded[r][0], upload[r]));
    }
  }
  void waitUpload(int first, int last) {
    for (int r = first; r < last; ++r) {
      check(cudaSetDevice(r));
      check(cudaEventSynchronize(uploaded[r][0]));
    }
  }
  void singleDownload(int first, int last, int blocks, bool gpuWait,
                      bool selectDevice = true) {
    for (int d = first; d < last; ++d) {
      if (selectDevice) check(cudaSetDevice(d));
      DownloadPacket packet{};
      for (int r = 0; r < 4; ++r) {
        if (gpuWait) check(cudaStreamWaitEvent(stream[d], uploaded[r][0], 0));
        packet.source[r] =
            r == d ? input[d] : static_cast<uint32_t*>(host[r][0].data());
      }
      if (blocks == 1)
        downloadSmall<<<1, 256, 0, stream[d]>>>(packet, output[d], count);
      else
        downloadPacket<<<blocks, 256, 0, stream[d]>>>(packet, output[d], count,
                                                      0, count);
      check(cudaGetLastError());
    }
  }
  void singlePacket(int blocks, bool cpuWait) {
    singleUpload(0, 4);
    if (cpuWait) waitUpload(0, 4);
    singleDownload(0, 4, blocks, !cpuWait);
  }
  void directStreamPacket(int blocks) {
    // Same serialized single-bank contract as singlePacket(). Upload is
    // ordered directly after the caller's producer on its own stream.
    if (chunkElements < count)
      throw std::runtime_error("single packet exceeds capacity");
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      check(cudaMemcpyAsync(host[r][0].data(), input[r], count * 4,
                            cudaMemcpyDeviceToHost, stream[r]));
    }
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      check(cudaStreamSynchronize(stream[r]));
    }
    singleDownload(0, 4, blocks, false);
  }
  void fixedWorkerPhase(int rank, int phase, int blocks, bool eventUpload,
                        FixedFour& team) {
    // The calling worker already owns this device; no device changes here.
    if (chunkElements < count)
      throw std::runtime_error("single packet exceeds capacity");
    if (phase == 0 || phase == 2) {
      if (eventUpload) {
        check(cudaStreamWaitEvent(upload[rank], ready[rank], 0));
        check(cudaMemcpyAsync(host[rank][0].data(), input[rank], count * 4,
                              cudaMemcpyDeviceToHost, upload[rank]));
        check(cudaEventRecord(uploaded[rank][0], upload[rank]));
        check(cudaEventSynchronize(uploaded[rank][0]));
      } else {
        check(cudaMemcpyAsync(host[rank][0].data(), input[rank], count * 4,
                              cudaMemcpyDeviceToHost, stream[rank]));
        check(cudaStreamSynchronize(stream[rank]));
      }
    }
    if (phase == 2) team.uploadBarrier(rank);
    if (phase == 1 || phase == 2) {
      singleDownload(rank, rank + 1, blocks, false, false);
      check(cudaStreamSynchronize(stream[rank]));
    }
  }
  void preparePeerAccess() {
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      for (int d = 0; d < 4; ++d) {
        if (d == r) continue;
        int access = 0;
        check(cudaDeviceCanAccessPeer(&access, r, d));
        if (!access)
          throw std::runtime_error("direct CE push requires all-pairs P2P");
        auto status = cudaDeviceEnablePeerAccess(d, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled)
          cudaGetLastError();
        else
          check(status);
      }
    }
  }
  // Caller selects the source device (or is its persistent fixed worker).
  // Exactly the SM push-serial source/destination order, with immutable input
  // and a disjoint source-owned slice in every output. No host payload buffer.
  void peerPush(int rank, bool useGraph, bool sm = false, bool batch = false,
                bool xorOrder = false) {
    if (useGraph) {
      check(cudaGraphLaunch(peerGraphs[rank], stream[rank]));
    } else if (sm) {
      int threads = count * 16 == (64UL << 10) ? 128 : 256;
      uccl_lite_reference::pushSerialGather<<<1, threads, 0, stream[rank]>>>(
          reinterpret_cast<float*>(input[rank]),
          reinterpret_cast<float*>(output[0]),
          reinterpret_cast<float*>(output[1]),
          reinterpret_cast<float*>(output[2]),
          reinterpret_cast<float*>(output[3]), count, rank);
      check(cudaGetLastError());
    } else if (batch) {
      void* dsts[4];
      void* srcs[4];
      size_t sizes[4];
      for (int phase = 0; phase < 4; ++phase) {
        int dest = xorOrder ? rank ^ ((phase + 1) % 4) : (rank + phase) % 4;
        dsts[phase] = output[dest] + size_t(rank) * count;
        srcs[phase] = input[rank];
        sizes[phase] = count * 4;
      }
      cudaMemcpyAttributes attributes{};
      attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      size_t first = 0, failedIndex = SIZE_MAX;
      check(cudaMemcpyBatchAsync(dsts, srcs, sizes, 4, &attributes, &first, 1,
                                 &failedIndex, stream[rank]));
    } else {
      for (int phase = 0; phase < 4; ++phase) {
        int dest = xorOrder ? rank ^ ((phase + 1) % 4) : (rank + phase) % 4;
        auto* target = output[dest] + size_t(rank) * count;
        if (dest == rank)
          check(cudaMemcpyAsync(target, input[rank], count * 4,
                                cudaMemcpyDeviceToDevice, stream[rank]));
        else
          check(cudaMemcpyPeerAsync(target, dest, input[rank], rank, count * 4,
                                    stream[rank]));
      }
    }
  }
  static CUstreamBatchMemOpParams flagWrite(CUdeviceptr address,
                                            unsigned epoch) {
    CUstreamBatchMemOpParams op{};
    op.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    op.writeValue.address = address;
    op.writeValue.value = epoch;
    op.writeValue.flags =
        CU_STREAM_WRITE_VALUE_DEFAULT;  // Stream system fence.
    return op;
  }
  static CUstreamBatchMemOpParams flagWait(CUdeviceptr address, unsigned epoch,
                                           bool flush) {
    CUstreamBatchMemOpParams op{};
    op.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    op.waitValue.address = address;
    op.waitValue.value = epoch;
    op.waitValue.flags =
        CU_STREAM_WAIT_VALUE_GEQ | (flush ? CU_STREAM_WAIT_VALUE_FLUSH : 0);
    return op;
  }
  void prepareMemops() {
    check(cuInit(0));
    memopEpoch = 0;
    for (int r = 0; r < 4; ++r) {
      memopFlags[r] = RegisteredMemory::pinned(4096, r / 2);
      std::memset(memopFlags[r].data(), 0, 4096);
    }
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      CUdevice device;
      check(cuDeviceGet(&device, r));
      int flush = 0, wide = 0;
      check(cuDeviceGetAttribute(
          &flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, device));
      check(cuDeviceGetAttribute(
          &wide, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device));
      memopFlush[r] = flush != 0;
      for (int source = 0; source < 4; ++source)
        check(cuMemHostGetDevicePointer(&memopAddress[r][source],
                                        memopFlags[source].data(), 0));
      // Probe actual v2 32-bit write/wait support, not a deprecated attribute.
      CUstreamBatchMemOpParams probe[] = {
          flagWrite(memopAddress[r][r], 0),
          flagWait(memopAddress[r][r], 0, false)};
      check(cuStreamBatchMemOp(reinterpret_cast<CUstream>(stream[r]), 2, probe,
                               0));
      check(cudaStreamSynchronize(stream[r]));
      std::fprintf(stderr,
                   "# memop GPU=%d write32/wait32=PASS flush=%d wide=%d "
                   "mapped_host_flags NUMA=%d\n",
                   r, flush, wide, r / 2);
    }
  }
  void memopBarrier(int rank, bool complete, bool publish, bool wait) {
    CUstreamBatchMemOpParams ops[4]{};
    unsigned n = 0;
    CUdeviceptr offset = complete ? 128 : 0;
    if (publish)
      ops[n++] = flagWrite(memopAddress[rank][rank] + offset, memopEpoch);
    if (wait)
      for (int other = 0; other < 4; ++other)
        if (other != rank)
          ops[n++] = flagWait(memopAddress[rank][other] + offset, memopEpoch,
                              complete && memopFlush[rank]);
    check(cuStreamBatchMemOp(reinterpret_cast<CUstream>(stream[rank]), n, ops,
                             0));
  }
  void memopPeer(int rank) {
    // One ordinary stream per device: its publication precedes its waits.
    // Standalone eager experiment; hidden cross-device memop ordering is not
    // a general graph/concurrent scheduler contract. Event-visible control
    // below.
    memopBarrier(rank, false, true, true);
    peerPush(rank, false, false, true, true);
    memopBarrier(rank, true, true, true);
  }
  void memopPeerVisible() {
    // Post all publications before cross-device waits, with CUDA-visible events
    // carrying the same ready/completion order as the memory operations.
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      memopBarrier(r, false, true, false);
      check(cudaEventRecord(ready[r], stream[r]));
    }
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      for (int other = 0; other < 4; ++other)
        if (other != r) check(cudaStreamWaitEvent(stream[r], ready[other], 0));
      memopBarrier(r, false, false, true);
      peerPush(r, false, false, true, true);
      memopBarrier(r, true, true, false);
      check(cudaEventRecord(uploaded[r][0], stream[r]));
    }
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      for (int other = 0; other < 4; ++other)
        if (other != r)
          check(cudaStreamWaitEvent(stream[r], uploaded[other][0], 0));
      memopBarrier(r, true, false, true);
    }
  }
  void armMemopPeer(bool sm) {
    // Prepared-trigger experiment: enqueue cost is recorded separately, never
    // counted as ordinary enqueue-to-completion latency. CPU is the ready
    // publisher, so no CUDA task is hidden behind another GPU's semaphore.
    ++memopEpoch;
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(r));
      auto wait = flagWait(memopAddress[r][0], memopEpoch, false);
      check(cuStreamBatchMemOp(reinterpret_cast<CUstream>(stream[r]), 1, &wait,
                               0));
      peerPush(r, false, sm, !sm, true);
      auto done = flagWrite(memopAddress[r][r] + 128, memopEpoch);
      check(cuStreamBatchMemOp(reinterpret_cast<CUstream>(stream[r]), 1, &done,
                               0));
    }
  }
  void fireMemopPeer() {
    __atomic_store_n(static_cast<unsigned*>(memopFlags[0].data()), memopEpoch,
                     __ATOMIC_RELEASE);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (int r = 0; r < 4; ++r) {
      auto* done = reinterpret_cast<unsigned*>(
          static_cast<char*>(memopFlags[r].data()) + 128);
      unsigned spins = 0;
      while (__atomic_load_n(done, __ATOMIC_ACQUIRE) != memopEpoch) {
        asm volatile("pause");
        if (++spins % 65536 == 0 && std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error("prepared memop completion timed out");
      }
    }
  }
  void preparePeerGraphs(bool xorOrder = false) {
    for (int rank = 0; rank < 4; ++rank) {
      check(cudaSetDevice(rank));
      cudaGraph_t captured;
      check(cudaGraphCreate(&captured, 0));
      cudaGraphNode_t previous = nullptr;
      for (int phase = 0; phase < 4; ++phase) {
        int dest = xorOrder ? rank ^ ((phase + 1) % 4) : (rank + phase) % 4;
        cudaGraphNode_t node;
        check(cudaGraphAddMemcpyNode1D(
            &node, captured, previous ? &previous : nullptr, previous ? 1 : 0,
            output[dest] + size_t(rank) * count, input[rank], count * 4,
            cudaMemcpyDeviceToDevice));
        previous = node;
      }
      check(cudaGraphInstantiate(&peerGraphs[rank], captured, 0));
      check(cudaGraphDestroy(captured));
      check(cudaGraphUpload(peerGraphs[rank], stream[rank]));
    }
    sync();
  }
  void prepareOneShotPeer(bool xorOrder, bool sourceMajor) {
    // One graph, four independent source chains. No inter-GPU phase barriers.
    // Round-major node insertion exposes one ready copy per source early.
    check(cudaGraphCreate(&graph, 0));
    std::array<cudaGraphNode_t, 4> previous{};
    for (int outer = 0; outer < 4; ++outer)
      for (int inner = 0; inner < 4; ++inner) {
        int rank = sourceMajor ? outer : inner;
        int phase = sourceMajor ? inner : outer;
        int dest = xorOrder ? rank ^ ((phase + 1) % 4) : (rank + phase) % 4;
        check(cudaSetDevice(rank));
        cudaGraphNode_t node;
        check(cudaGraphAddMemcpyNode1D(
            &node, graph, previous[rank] ? &previous[rank] : nullptr,
            previous[rank] ? 1 : 0, output[dest] + size_t(rank) * count,
            input[rank], count * 4, cudaMemcpyDeviceToDevice));
        previous[rank] = node;
      }
    check(cudaSetDevice(0));
    check(cudaGraphInstantiate(&executable, graph, 0));
    check(cudaGraphUpload(executable, stream[0]));
    sync();
  }
  void prepareGraph() {
    // Encode the same two-slot lifetime and serial stream dependencies directly
    // in the DAG. Host buffers and device pointers stay fixed through replay.
    check(cudaGraphCreate(&graph, 0));
    std::array<cudaGraphNode_t, 4> lastUpload{}, lastDownload{};
    std::array<std::array<cudaGraphNode_t, 4>, 2> consumedNodes{};
    auto node = [&](int device, void* dst, void const* src, size_t bytes,
                    cudaMemcpyKind kind, std::vector<cudaGraphNode_t> deps) {
      deps.erase(std::remove(deps.begin(), deps.end(), nullptr), deps.end());
      std::sort(deps.begin(), deps.end());
      deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
      check(cudaSetDevice(device));
      cudaGraphNode_t result;
      check(cudaGraphAddMemcpyNode1D(&result, graph, deps.data(), deps.size(),
                                     dst, src, bytes, kind));
      return result;
    };
    for (size_t offset = 0, piece = 0; offset < count;
         offset += chunkElements, ++piece) {
      int slot = piece % 2;
      size_t bytes = std::min(chunkElements, count - offset) * 4;
      for (int r = 0; r < 4; ++r) {
        std::vector<cudaGraphNode_t> deps(consumedNodes[slot].begin(),
                                          consumedNodes[slot].end());
        deps.push_back(lastUpload[r]);
        lastUpload[r] = node(r, host[r][slot].data(), input[r] + offset, bytes,
                             cudaMemcpyDeviceToHost, deps);
      }
      for (int d = 0; d < 4; ++d) {
        for (int r = 0; r < 4; ++r)
          lastDownload[d] =
              node(d, output[d] + r * count + offset,
                   r == d ? input[d] + offset : host[r][slot].data(), bytes,
                   r == d ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                   {lastDownload[d], lastUpload[r]});
        consumedNodes[slot][d] = lastDownload[d];
      }
    }
    check(cudaSetDevice(0));
    check(cudaGraphInstantiate(&executable, graph, 0));
    check(cudaGraphUpload(executable, stream[0]));
    sync();
  }
  void replay() {
    check(cudaSetDevice(0));
    check(cudaGraphLaunch(executable, stream[0]));
  }
  ~Experiment() {
    sync();
    if (referenceFlags) cudaFreeHost(referenceFlags);
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    for (int r = 0; r < 4; ++r) {
      cudaSetDevice(r);
      if (peerGraphs[r]) cudaGraphExecDestroy(peerGraphs[r]);
      cudaStreamSynchronize(upload[r]);
      cudaStreamDestroy(upload[r]);
      for (int slot = 0; slot < 2; ++slot) {
        cudaEventDestroy(uploaded[r][slot]);
        cudaEventDestroy(consumed[r][slot]);
      }
      cudaEventDestroy(ready[r]);
      ncclCommDestroy(nccl[r]);
      cudaFree(input[r]);
      cudaFree(output[r]);
      cudaFree(errors[r]);
      cudaStreamDestroy(stream[r]);
    }
  }
};
int main(int argc, char** argv) {
  try {
    size_t bytes = argc > 1 ? std::stoull(argv[1]) : 1UL << 24;
    int iterations = argc > 2 ? std::stoi(argv[2]) : 5;
    if (!bytes || bytes % 16 || iterations < 1)
      throw std::runtime_error(
          "usage: ce_allgather output_bytes_multiple_of_16 iterations");
    auto allowedCpus = initialAffinity();
    if (auto* value = std::getenv("UCCL_CE_COORDINATOR_CPU")) {
      int cpu = std::stoi(value);
      if (cpu < 0 || cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowedCpus))
        throw std::runtime_error("coordinator CPU not permitted");
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(cpu, &mask);
      if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
        throw std::runtime_error("cannot pin coordinator");
      std::fprintf(stderr, "# coordinator cpu=%d\n", cpu);
    }
    Experiment e(bytes);
    PacketPipelineOptions options;
    options.maxTensorBytes = bytes;
    options.packetBytes = 4UL << 20;
    options.transportBlocks = 248;
    options.workersPerNode = 8;
    options.reduction = ReductionPlacement::Gpu;
    options.gpuProgress = true;
    options.agPerSourceProgress = true;
    PacketPipeline baseline(options);
    std::vector<PacketPort> ports;
    for (int r = 0; r < 4; ++r)
      ports.push_back({e.input[r], e.output[r], e.stream[r], r});
    baseline.prepare(ports);
    e.prepareReference();
    struct Variant {
      char const* name;
      size_t chunk;
    };
    std::vector<Variant> variants = {{"uccl_fp32_reference", 1UL << 20},
                                     {"packet_sm", 1UL << 20},
                                     {"nccl", 1UL << 20},
                                     {"host_sm_4m", 1UL << 20},
                                     {"host_ce_4m", 1UL << 20},
                                     {"host_ce_1m", 256UL << 10},
                                     {"host_ce_256k", 64UL << 10},
                                     {"host_ce_8m", 2UL << 20},
                                     {"host_ce_16m", 4UL << 20},
                                     {"host_ce_32m", 8UL << 20},
                                     {"host_ce_64m", 16UL << 20},
                                     {"batch_ce_4m", 1UL << 20},
                                     {"batch_ce_16m", 4UL << 20},
                                     {"waitall_ce_4m", 1UL << 20},
                                     {"waitall_ce_16m", 4UL << 20},
                                     {"graph_ce_4m", 1UL << 20},
                                     {"graph_ce_16m", 4UL << 20},
                                     {"full_ce_4m", 1UL << 20},
                                     {"hybrid_1m", 256UL << 10},
                                     {"hybrid_4m", 1UL << 20},
                                     {"hybrid_16m", 4UL << 20},
                                     {"hybrid_grouped_1m", 256UL << 10},
                                     {"hybrid_grouped_2m", 512UL << 10},
                                     {"hybrid_grouped_4m", 1UL << 20},
                                     {"hybrid_grouped_8m", 2UL << 20},
                                     {"hybrid_grouped_16m", 4UL << 20},
                                     {"grouped_sm_1m", 256UL << 10},
                                     {"grouped_sm_4m", 1UL << 20},
                                     {"sp_gpu_32", 0},
                                     {"sp_cpu_32", 0},
                                     {"sp_cpu_16", 0},
                                     {"sp_cpu_8", 0},
                                     {"sp_cpu_4", 0},
                                     {"sp_cpu_1", 0},
                                     {"sp_threads_32", 0},
                                     {"sp_threads_16", 0},
                                     {"sp_threads_8", 0},
                                     {"sp_threads_4", 0},
                                     {"sp_threads_1", 0},
                                     {"sp_direct_8", 0},
                                     {"sp_direct_1", 0},
                                     {"sp_fixed4_8", 0},
                                     {"sp_fixed4_1", 0},
                                     {"sp_fixed4_events_8", 0},
                                     {"sp_fixed4_barrier_8", 0},
                                     {"sp_fixed4_barrier_1", 0},
                                     {"p2p_ce", 0},
                                     {"p2p_ce_fixed4", 0},
                                     {"p2p_sm_fixed4", 0},
                                     {"p2p_graph_ce", 0},
                                     {"p2p_graph_ce_fixed4", 0},
                                     {"p2p_batch_ce", 0},
                                     {"p2p_batch_ce_fixed4", 0},
                                     {"p2p_xor_ce_fixed4", 0},
                                     {"p2p_xor_graph_ce_fixed4", 0},
                                     {"p2p_oneshot_source", 0},
                                     {"p2p_oneshot_round", 0},
                                     {"p2p_oneshot_xor", 0},
                                     {"p2p_memop", 0},
                                     {"p2p_memop_fixed4", 0},
                                     {"p2p_memop_events", 0},
                                     {"p2p_memop_onewait", 0},
                                     {"p2p_memop_onewait_fixed4", 0},
                                     {"p2p_memop_armed_ce", 0},
                                     {"p2p_memop_armed_sm", 0},
                                     {"p2p_xor_batch_ce", 0},
                                     {"p2p_xor_batch_ce_fixed4", 0}};
    int epoch = 0;
    printf("bytes,variant,round,iteration,host_us,enqueue_us,arm_us\n");
    for (int round = 0; round < 3; ++round) {
      if (round) std::reverse(variants.begin(), variants.end());
      for (auto v : variants) {
        if (argc > 3 &&
            ("," + std::string(argv[3]) + ",")
                    .find("," + std::string(v.name) + ",") == std::string::npos)
          continue;
        if (std::string(v.name).find("sp_") == 0 && bytes > (16UL << 20))
          continue;
        e.prepare(v.chunk ? v.chunk : e.count,
                  std::string(v.name).find("full") == 0,
                  std::string(v.name).find("p2p_") != 0);
        std::string name(v.name);
        if (name.find("graph") == 0) e.prepareGraph();
        bool peer = name.find("p2p_") == 0;
        bool peerGraph = peer && name.find("graph") != std::string::npos;
        bool oneShotPeer = name.find("p2p_oneshot") == 0;
        bool xorOrder = peer && name.find("xor") != std::string::npos;
        bool memops = name.find("p2p_memop") == 0;
        bool armedMemop = name.find("p2p_memop_armed") == 0;
        if (std::getenv("UCCL_CE_DELAY_INPUT_RANK") && (!memops || armedMemop))
          throw std::runtime_error(
              "delayed-input mode requires memop all-rank barriers");
        bool oneMemopWait = memops && name.find("onewait") != std::string::npos;
        if (peer) {
          e.preparePeerAccess();
          if (memops) e.prepareMemops();
          if (peerGraph) e.preparePeerGraphs(xorOrder);
          if (oneShotPeer)
            e.prepareOneShotPeer(xorOrder, name == "p2p_oneshot_source");
        }
        int singleBlocks = name.find("sp_") == 0
                               ? std::stoi(name.substr(name.rfind('_') + 1))
                               : 0;
        std::unique_ptr<SubmitPair> submitters;
        if (name.find("sp_threads") == 0)
          submitters = std::make_unique<SubmitPair>([&](int node, int phase) {
            if (phase == 0) {
              e.singleUpload(node * 2, node * 2 + 2);
              e.waitUpload(node * 2, node * 2 + 2);
            } else
              e.singleDownload(node * 2, node * 2 + 2, singleBlocks, false);
          });
        std::unique_ptr<FixedFour> fixedWorkers;
        if (name.find("sp_fixed4") == 0 ||
            (peer && name.find("fixed4") != std::string::npos))
          fixedWorkers = std::make_unique<FixedFour>([&](int rank, int phase,
                                                         FixedFour& team) {
            if (peer) {
              if (memops)
                e.memopPeer(rank);
              else
                e.peerPush(rank, peerGraph, name == "p2p_sm_fixed4",
                           name.find("batch") != std::string::npos, xorOrder);
              check(cudaStreamSynchronize(e.stream[rank]));
            } else
              e.fixedWorkerPhase(rank, phase, singleBlocks,
                                 name.find("events") != std::string::npos,
                                 team);
          });
        for (int i = -3; i < iterations; ++i) {
          e.initialize(++epoch);
          bool profileWarm = std::getenv("UCCL_CE_PROFILE_WARM");
          bool capture = (std::getenv("UCCL_CE_PROFILE") || profileWarm) &&
                         i == 0 && round == 0;
          bool captureStart = (std::getenv("UCCL_CE_PROFILE") || profileWarm) &&
                              i == (profileWarm ? -3 : 0) && round == 0;
          if (captureStart) check(cudaProfilerStart());
          double armUs = 0;
          if (armedMemop) {
            auto armStart = std::chrono::steady_clock::now();
            e.armMemopPeer(name == "p2p_memop_armed_sm");
            armUs = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - armStart)
                        .count();
          }
          auto start = std::chrono::steady_clock::now();
          if (memops && !armedMemop) ++e.memopEpoch;
          if (armedMemop)
            e.fireMemopPeer();
          else if (name == "uccl_fp32_reference" && bytes <= (16UL << 20))
            e.referenceSmallMedium();
          else if (name == "packet_sm" || name == "uccl_fp32_reference")
            baseline.enqueue(PacketProgram::allGather(e.count * 2, 4), ports);
          else if (std::string(v.name) == "nccl") {
            check(ncclGroupStart());
            for (int r = 0; r < 4; ++r) {
              check(cudaSetDevice(r));
              check(ncclAllGather(e.input[r], e.output[r], e.count, ncclFloat,
                                  e.nccl[r], e.stream[r]));
            }
            check(ncclGroupEnd());
          } else if (fixedWorkers) {
            if (peer)
              fixedWorkers->run(3);
            else if (name.find("barrier") != std::string::npos)
              fixedWorkers->run(2);
            else {
              fixedWorkers->run(0);
              fixedWorkers->run(1);
            }
          } else if (name == "p2p_memop_events")
            e.memopPeerVisible();
          else if (oneShotPeer)
            e.replay();
          else if (peer) {
            for (int r = 0; r < 4; ++r) {
              check(cudaSetDevice(r));
              if (memops)
                e.memopPeer(r);
              else
                e.peerPush(r, peerGraph, false,
                           name.find("batch") != std::string::npos, xorOrder);
            }
          } else if (submitters) {
            submitters->run(0);
            submitters->run(1);
          } else if (name.find("sp_direct") == 0)
            e.directStreamPacket(singleBlocks);
          else if (singleBlocks)
            e.singlePacket(singleBlocks, name.find("sp_cpu") == 0);
          else if (name.find("graph") == 0)
            e.replay();
          else
            e.copies(name != "host_sm_4m" && name.find("grouped_sm") != 0,
                     name.find("batch") == 0, name.find("waitall") == 0,
                     name.find("hybrid") == 0,
                     name.find("hybrid_grouped") == 0 ||
                         name.find("grouped_sm") == 0);
          double enqueue = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - start)
                               .count();
          if (armedMemop) {
          }  // All sender completion writes were observed above.
          else if (oneShotPeer)
            check(cudaStreamSynchronize(e.stream[0]));
          else if (oneMemopWait && !fixedWorkers) {
            check(cudaSetDevice(0));
            check(cudaStreamSynchronize(e.stream[0]));
          } else if (!fixedWorkers)
            e.sync();  // Fixed workers already joined GPU completion.
          double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count();
          if (capture) check(cudaProfilerStop());
          e.validate(epoch);
          if (i >= 0)
            printf("%zu,%s,%d,%d,%.3f,%.3f,%.3f\n", bytes, v.name, round, i, us,
                   enqueue, armUs);
          fflush(stdout);
        }
      }
    }
  } catch (std::exception const& e) {
    fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}

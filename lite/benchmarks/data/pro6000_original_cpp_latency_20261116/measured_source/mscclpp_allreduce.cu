// Topology-aware AllReduce for this host's 8 PCIe-only Blackwell GPUs.
//
// Rank routing follows the physical hierarchy reported by `nvidia-smi topo -m`:
//   xor 1: the other GPU behind the same PCIe switch (PIX)
//   xor 2: the other switch attached to the same CPU socket (NODE)
//   xor 4: the matching GPU on the other CPU socket (SYS)
//
// Recursive halving performs ReduceScatter in that order; recursive doubling
// performs AllGather in reverse.  Each rank moves
//   2 * (N/2 + N/4 + N/8) = 1.75N bytes,
// the bandwidth-optimal volume for eight ranks.  Only N/8 bytes per rank cross
// the inter-socket link in either direction.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>

#include <cuda_runtime.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

constexpr int kRanks = 8;
constexpr int kLevels = 3;
constexpr double kBusFactor = 2.0 * (kRanks - 1) / kRanks;

#define CUDACHECK(cmd)                                                                            \
  do {                                                                                            \
    cudaError_t e = (cmd);                                                                        \
    if (e != cudaSuccess) {                                                                       \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

using Chan = mscclpp::MemoryChannelDeviceHandle;
using PortChan = mscclpp::PortChannelDeviceHandle;
using Pkt = mscclpp::LL16Packet;

__device__ mscclpp::DeviceSyncer gSyncer;

// CE uses one immutable send stage and one receive stage per ReduceScatter
// level.  Old send stages are deliberately not reused: proxy copies drain on a
// different CUDA stream and do not acknowledge completion to their sender.
// Receive stages are safely reused as AllGather assembly stages after all three
// matching waits have completed.
struct Layout {
  size_t data;
  size_t send[kLevels + 1];
  size_t recv[kLevels];
  size_t pkt[2 * kLevels];
};

__device__ __forceinline__ uint2 f2u(float2 v) {
  return make_uint2(__float_as_uint(v.x), __float_as_uint(v.y));
}

__device__ __forceinline__ float2 u2f(uint2 v) {
  return make_float2(__uint_as_float(v.x), __uint_as_float(v.y));
}

__device__ __forceinline__ void addOut(float* dst, const float* a, const float* b, size_t n,
                                       int tid, int nth) {
  size_t n4 = n / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  const float4* a4 = reinterpret_cast<const float4*>(a);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  for (size_t i = tid; i < n4; i += nth) {
    float4 x = a4[i], y = b4[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    d4[i] = x;
  }
  for (size_t i = n4 * 4 + tid; i < n; i += nth) dst[i] = a[i] + b[i];
}

__device__ __forceinline__ void addInto(float* dst, const float* src, size_t n, int tid,
                                        int nth) {
  size_t n4 = n / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  const float4* s4 = reinterpret_cast<const float4*>(src);
  for (size_t i = tid; i < n4; i += nth) {
    float4 x = d4[i], y = s4[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    d4[i] = x;
  }
  for (size_t i = n4 * 4 + tid; i < n; i += nth) dst[i] += src[i];
}

__device__ __forceinline__ void copyOut(float* dst, const float* src, size_t n, int tid,
                                        int nth) {
  size_t n4 = n / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  const float4* s4 = reinterpret_cast<const float4*>(src);
  for (size_t i = tid; i < n4; i += nth) d4[i] = s4[i];
  for (size_t i = n4 * 4 + tid; i < n; i += nth) dst[i] = src[i];
}

__device__ __forceinline__ Chan levelChan(int level, Chan sw, Chan socket, Chan system) {
  return level == 0 ? sw : (level == 1 ? socket : system);
}

// All threads publish their own writes before thread 0 hands the source region
// to the proxy.  A fence on thread 0 alone does not publish stores made by the
// rest of the grid.
__device__ __forceinline__ void publishToProxy() {
  __threadfence_system();
  gSyncer.sync(gridDim.x);
}

__global__ void __launch_bounds__(1024)
ar8Put(Chan chSwitch, Chan chSocket, Chan chSystem, int rank, size_t nelem, Layout L) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  char* base = reinterpret_cast<char*>(chSwitch.src_);
  float* data = reinterpret_cast<float*>(base + L.data);

  size_t segBase = 0, segN = nelem;
  for (int level = 0; level < kLevels; ++level) {
    Chan ch = levelChan(level, chSwitch, chSocket, chSystem);
    size_t half = segN / 2;
    size_t keepBase = segBase + ((rank >> level) & 1) * half;
    size_t sendBase = segBase + (((rank >> level) & 1) ^ 1) * half;
    size_t bytes = half * sizeof(float);

    ch.put<16, true>(L.recv[level], L.data + sendBase * sizeof(float), bytes, tid, nth);
    gSyncer.sync(gridDim.x);
    if (tid == 0) {
      ch.signal();
      ch.wait();
    }
    gSyncer.sync(gridDim.x);
    float* scratch = reinterpret_cast<float*>(base + L.recv[level]);
    addInto(data + keepBase, scratch, half, tid, nth);
    gSyncer.sync(gridDim.x);
    segBase = keepBase;
    segN = half;
  }

  for (int level = kLevels - 1; level >= 0; --level) {
    Chan ch = levelChan(level, chSwitch, chSocket, chSystem);
    size_t bytes = segN * sizeof(float);
    ch.put<16, true>(L.data + segBase * sizeof(float), L.data + segBase * sizeof(float), bytes,
                     tid, nth);
    gSyncer.sync(gridDim.x);
    if (tid == 0) {
      ch.signal();
      ch.wait();
    }
    gSyncer.sync(gridDim.x);
    size_t otherBase = segBase ^ segN;
    segBase = min(segBase, otherBase);
    segN *= 2;
  }
}

// LL is used only for small messages and launches one CTA.  Packet flags make
// peer handovers self-synchronising; __syncthreads preserves dependencies among
// the six local phases without paying a grid-wide barrier.
__global__ void __launch_bounds__(1024)
ar8LL(Chan chSwitch, Chan chSocket, Chan chSystem, int rank, size_t nelem, Layout L,
      uint32_t flag) {
  int tid = threadIdx.x;
  int nth = blockDim.x;
  char* base = reinterpret_cast<char*>(chSwitch.src_);
  float2* data = reinterpret_cast<float2*>(base + L.data);

  size_t segBase = 0, segN = nelem;
  for (int level = 0; level < kLevels; ++level) {
    Chan ch = levelChan(level, chSwitch, chSocket, chSystem);
    size_t half = segN / 2;
    size_t keepBase = segBase + ((rank >> level) & 1) * half;
    size_t sendBase = segBase + (((rank >> level) & 1) ^ 1) * half;
    Pkt* tx = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(ch.dst_) + L.pkt[level]);
    Pkt* rx = reinterpret_cast<Pkt*>(base + L.pkt[level]);
    for (size_t i = tid; i < half / 2; i += nth) {
      tx[i].write(f2u(data[sendBase / 2 + i]), flag);
      float2 a = data[keepBase / 2 + i];
      float2 b = u2f(rx[i].read(flag));
      a.x += b.x;
      a.y += b.y;
      data[keepBase / 2 + i] = a;
    }
    __syncthreads();
    segBase = keepBase;
    segN = half;
  }

  for (int level = kLevels - 1; level >= 0; --level) {
    Chan ch = levelChan(level, chSwitch, chSocket, chSystem);
    size_t otherBase = segBase ^ segN;
    Pkt* tx = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(ch.dst_) + L.pkt[2 * kLevels - 1 - level]);
    Pkt* rx = reinterpret_cast<Pkt*>(base + L.pkt[2 * kLevels - 1 - level]);
    for (size_t i = tid; i < segN / 2; i += nth) {
      tx[i].write(f2u(data[segBase / 2 + i]), flag);
      data[otherBase / 2 + i] = u2f(rx[i].read(flag));
    }
    __syncthreads();
    segBase = min(segBase, otherBase);
    segN *= 2;
  }
}

__device__ __forceinline__ PortChan levelPort(int level, PortChan p0, PortChan p1,
                                              PortChan p2) {
  return level == 0 ? p0 : (level == 1 ? p1 : p2);
}

__global__ void __launch_bounds__(1024)
ar8CE(PortChan rs0, PortChan rs1, PortChan rs2, PortChan ag0, PortChan ag1, PortChan ag2,
      float* rawBase, int rank, size_t nelem, Layout L) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  char* base = reinterpret_cast<char*>(rawBase);

  size_t segBase = 0, segN = nelem;
  for (int level = 0; level < kLevels; ++level) {
    PortChan p = levelPort(level, rs0, rs1, rs2);
    size_t half = segN / 2;
    size_t keepBase = segBase + ((rank >> level) & 1) * half;
    size_t sendBase = segBase + (((rank >> level) & 1) ^ 1) * half;
    if (level != 0) publishToProxy();
    else gSyncer.sync(gridDim.x);
    if (tid == 0) {
      p.putWithSignal(sendBase * sizeof(float), sendBase * sizeof(float), half * sizeof(float));
      p.wait();
    }
    gSyncer.sync(gridDim.x);

    float* src = reinterpret_cast<float*>(base + L.send[level]);
    float* recv = reinterpret_cast<float*>(base + L.recv[level]);
    float* dst = reinterpret_cast<float*>(base + L.send[level + 1]);
    addOut(dst + keepBase, src + keepBase, recv + keepBase, half, tid, nth);
    gSyncer.sync(gridDim.x);
    segBase = keepBase;
    segN = half;
  }

  // Reuse recv[2], recv[1], then data as successively larger assembly buffers.
  for (int level = kLevels - 1; level >= 0; --level) {
    PortChan p = levelPort(level, ag0, ag1, ag2);
    size_t dstAbs = level == 2 ? L.recv[2] : (level == 1 ? L.recv[1] : L.data);
    size_t srcAbs = level == 2 ? L.send[3] : (level == 1 ? L.recv[2] : L.recv[1]);
    float* src = reinterpret_cast<float*>(base + srcAbs);
    float* dst = reinterpret_cast<float*>(base + dstAbs);
    copyOut(dst + segBase, src + segBase, segN, tid, nth);
    publishToProxy();
    if (tid == 0) {
      p.putWithSignal(segBase * sizeof(float), segBase * sizeof(float), segN * sizeof(float));
      p.wait();
    }
    gSyncer.sync(gridDim.x);
    segBase = min(segBase, segBase ^ segN);
    segN *= 2;
  }
}

enum Mode { MODE_SM, MODE_LL, MODE_CE, MODE_AUTO, MODE_ALL };

struct Cfg {
  size_t minBytes = 128;
  size_t maxBytes = 256UL << 20;
  int warmup = 10;
  int iters = 20;
  int blocks = 16;
  int threads = 256;
  int ceBlocks = 16;
  int ceThreads = 512;
  int llThreads = 1024;
  size_t llCutoff = 32UL << 10;
  size_t ceCutoff = 4UL << 20;
  bool check = true;
  Mode mode = MODE_ALL;
};

static size_t roundUp(size_t v, size_t a) { return ((v + a - 1) / a) * a; }

static void pinRankToLocalCpu(int rank) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
  int lo = rank < 4 ? 0 : 84;
  int hi = rank < 4 ? 84 : 168;
  int preferred = lo + (rank & 3);
  for (int offset = 0; offset < hi - lo; ++offset) {
    int cpu = lo + ((preferred - lo + offset) % (hi - lo));
    if (CPU_ISSET(cpu, &allowed)) {
      cpu_set_t one;
      CPU_ZERO(&one);
      CPU_SET(cpu, &one);
      sched_setaffinity(0, sizeof(one), &one);
      return;
    }
  }
}

static void runRank(int rank, mscclpp::UniqueId id, const Cfg& cfg) {
  pinRankToLocalCpu(rank);
  CUDACHECK(cudaSetDevice(rank));

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, kRanks);
  bootstrap->initialize(id);
  mscclpp::Communicator comm(bootstrap);

  size_t N = cfg.maxBytes;
  if (N >= (1ULL << 32)) {
    std::fprintf(stderr, "-e must be below 4 GiB because proxy offsets are 32-bit\n");
    std::exit(1);
  }
  size_t pktSlot = roundUp(cfg.llCutoff, 4096);
  size_t pktBase = 7 * N;
  size_t bufBytes = pktBase + 2 * (2 * kLevels) * pktSlot;

  char* buf = nullptr;
  CUDACHECK(cudaMalloc(&buf, bufBytes));
  CUDACHECK(cudaMemset(buf, 0, bufBytes));

  Layout fixed{};
  fixed.data = 0;
  fixed.send[0] = 0;
  fixed.send[1] = N;
  fixed.send[2] = 2 * N;
  fixed.send[3] = 3 * N;
  fixed.recv[0] = 4 * N;
  fixed.recv[1] = 5 * N;
  fixed.recv[2] = 6 * N;

  auto fullReg = comm.registerMemory(buf, bufBytes, mscclpp::Transport::CudaIpc);
  std::array<mscclpp::RegisteredMemory, 7> region = {
      comm.registerMemory(buf + 0 * N, N, mscclpp::Transport::CudaIpc),
      comm.registerMemory(buf + 1 * N, N, mscclpp::Transport::CudaIpc),
      comm.registerMemory(buf + 2 * N, N, mscclpp::Transport::CudaIpc),
      comm.registerMemory(buf + 3 * N, N, mscclpp::Transport::CudaIpc),
      comm.registerMemory(buf + 4 * N, N, mscclpp::Transport::CudaIpc),
      comm.registerMemory(buf + 5 * N, N, mscclpp::Transport::CudaIpc),
      comm.registerMemory(buf + 6 * N, N, mscclpp::Transport::CudaIpc)};

  std::array<int, kLevels> peers = {rank ^ 1, rank ^ 2, rank ^ 4};
  std::array<std::shared_future<mscclpp::Connection>, kLevels> connF;
  std::array<std::shared_future<mscclpp::RegisteredMemory>, kLevels> fullF;
  std::array<std::array<std::shared_future<mscclpp::RegisteredMemory>, 7>, kLevels> remoteF;
  for (int level = 0; level < kLevels; ++level) {
    int peer = peers[level];
    comm.sendMemory(fullReg, peer, 10);
    fullF[level] = comm.recvMemory(peer, 10);
    for (int r = 0; r < 7; ++r) {
      comm.sendMemory(region[r], peer, r);
      remoteF[level][r] = comm.recvMemory(peer, r);
    }
    connF[level] = comm.connect(mscclpp::EndpointConfig(mscclpp::Transport::CudaIpc), peer, 0);
  }

  std::array<mscclpp::Connection, kLevels> conn = {
      connF[0].get(), connF[1].get(), connF[2].get()};
  std::array<mscclpp::Semaphore, kLevels> sem;
  for (int level = 0; level < kLevels; ++level)
    sem[level] = comm.buildSemaphore(conn[level], peers[level]).get();

  std::array<std::unique_ptr<mscclpp::MemoryChannel>, kLevels> memObj;
  std::array<Chan, kLevels> mem;
  for (int level = 0; level < kLevels; ++level) {
    memObj[level] =
        std::make_unique<mscclpp::MemoryChannel>(sem[level], fullF[level].get(), fullReg, buf);
  }

  auto proxy = std::make_shared<mscclpp::ProxyService>();
  std::array<mscclpp::MemoryId, 7> localId;
  for (int r = 0; r < 7; ++r) localId[r] = proxy->addMemory(region[r]);
  std::array<std::array<mscclpp::MemoryId, 7>, kLevels> remoteId;
  std::array<mscclpp::SemaphoreId, kLevels> sid;
  for (int level = 0; level < kLevels; ++level) {
    sid[level] = proxy->buildAndAddSemaphore(comm, conn[level]);
    for (int r = 0; r < 7; ++r) remoteId[level][r] = proxy->addMemory(remoteF[level][r].get());
  }
  for (int level = 0; level < kLevels; ++level) mem[level] = memObj[level]->deviceHandle();

  // ReduceScatter: data->R0, S1->R1, S2->R2.
  std::array<std::unique_ptr<mscclpp::PortChannel>, kLevels> rsObj;
  rsObj[0] = std::make_unique<mscclpp::PortChannel>(proxy->portChannel(sid[0], remoteId[0][4], localId[0]));
  rsObj[1] = std::make_unique<mscclpp::PortChannel>(proxy->portChannel(sid[1], remoteId[1][5], localId[1]));
  rsObj[2] = std::make_unique<mscclpp::PortChannel>(proxy->portChannel(sid[2], remoteId[2][6], localId[2]));
  // AllGather: S3->R2, R2->R1, R1->data.
  std::array<std::unique_ptr<mscclpp::PortChannel>, kLevels> agObj;
  agObj[0] = std::make_unique<mscclpp::PortChannel>(proxy->portChannel(sid[0], remoteId[0][0], localId[5]));
  agObj[1] = std::make_unique<mscclpp::PortChannel>(proxy->portChannel(sid[1], remoteId[1][5], localId[6]));
  agObj[2] = std::make_unique<mscclpp::PortChannel>(proxy->portChannel(sid[2], remoteId[2][6], localId[3]));
  std::array<PortChan, kLevels> rs = {rsObj[0]->deviceHandle(), rsObj[1]->deviceHandle(),
                                      rsObj[2]->deviceHandle()};
  std::array<PortChan, kLevels> ag = {agObj[0]->deviceHandle(), agObj[1]->deviceHandle(),
                                      agObj[2]->deviceHandle()};
  proxy->startProxy();

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::vector<float> hostIn, hostOut;
  uint32_t flag = 1;

  auto layoutFor = [&](int parity) {
    Layout L = fixed;
    for (int p = 0; p < 2 * kLevels; ++p)
      L.pkt[p] = pktBase + (2 * p + parity) * pktSlot;
    return L;
  };

  auto concreteMode = [&](Mode requested, size_t bytes) {
    if (requested != MODE_AUTO) return requested;
    if (bytes <= cfg.llCutoff) return MODE_LL;
    if (bytes >= cfg.ceCutoff) return MODE_CE;
    return MODE_SM;
  };

  auto launch = [&](Mode requested, size_t nelem, int iter) {
    Layout L = layoutFor(iter & 1);
    Mode mode = concreteMode(requested, nelem * sizeof(float));
    if (mode == MODE_LL) {
      ar8LL<<<1, cfg.llThreads, 0, stream>>>(mem[0], mem[1], mem[2], rank, nelem, L, flag++);
    } else if (mode == MODE_SM) {
      ar8Put<<<cfg.blocks, cfg.threads, 0, stream>>>(mem[0], mem[1], mem[2], rank, nelem, L);
    } else {
      ar8CE<<<cfg.ceBlocks, cfg.ceThreads, 0, stream>>>(
          rs[0], rs[1], rs[2], ag[0], ag[1], ag[2], reinterpret_cast<float*>(buf), rank, nelem, L);
    }
    CUDACHECK(cudaGetLastError());
  };

  auto fill = [&](size_t nelem) {
    hostIn.resize(nelem);
    for (size_t j = 0; j < nelem; ++j)
      hostIn[j] = static_cast<float>(rank + 1) * static_cast<float>(1 + (j % 97));
    CUDACHECK(cudaMemcpyAsync(buf, hostIn.data(), nelem * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  };

  auto verify = [&](size_t nelem) {
    hostOut.resize(nelem);
    CUDACHECK(cudaMemcpy(hostOut.data(), buf, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    double worst = 0.0;
    long long bad = -1;
    for (size_t j = 0; j < nelem; ++j) {
      double want = 36.0 * static_cast<double>(1 + (j % 97));
      double err = std::fabs(static_cast<double>(hostOut[j]) - want) / want;
      if (err > 1e-5 && bad < 0) bad = static_cast<long long>(j);
      worst = std::max(worst, err);
    }
    if (bad >= 0)
      std::fprintf(stderr, "rank %d first mismatch at %lld: got %.1f\n", rank, bad,
                   static_cast<double>(hostOut[bad]));
    return worst;
  };

  if (rank == 0) {
    std::printf("# PCIeCCL 8x RTX PRO 5000 Blackwell (sm_120), float32\n");
    std::printf("# route: xor1=PIX, xor2=NODE, xor4=SYS; put=%dx%d CE=%dx%d LL=1x%d; auto LL<=%zu CE>=%zu\n",
                cfg.blocks, cfg.threads, cfg.ceBlocks, cfg.ceThreads, cfg.llThreads, cfg.llCutoff,
                cfg.ceCutoff);
    std::printf("# %12s %12s %10s %10s %10s %10s %10s\n", "size(B)", "count", "mode",
                "time(us)", "algbw", "busbw", "maxrelerr");
  }

  std::vector<Mode> modes;
  if (cfg.mode == MODE_ALL)
    modes = {MODE_LL, MODE_SM, MODE_CE, MODE_AUTO};
  else
    modes = {cfg.mode};
  const char* names[] = {"put", "ll", "ce", "auto", "all"};

  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    if (bytes < 32 || bytes % 32 != 0) continue;
    size_t nelem = bytes / sizeof(float);
    for (Mode requested : modes) {
      if (requested == MODE_LL && bytes > cfg.llCutoff) continue;
      double relerr = -1.0;
      if (cfg.check) {
        fill(nelem);
        bootstrap->barrier();
        launch(requested, nelem, 0);
        CUDACHECK(cudaStreamSynchronize(stream));
        relerr = verify(nelem);
        bootstrap->barrier();
      }

      fill(nelem);
      for (int i = 0; i < cfg.warmup; ++i) launch(requested, nelem, i);
      CUDACHECK(cudaStreamSynchronize(stream));
      bootstrap->barrier();
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < cfg.iters; ++i) launch(requested, nelem, i);
      CUDACHECK(cudaStreamSynchronize(stream));
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / cfg.iters;

      std::array<double, kRanks> times{}, errs{};
      times[rank] = us;
      errs[rank] = relerr;
      bootstrap->allGather(times.data(), sizeof(double));
      bootstrap->allGather(errs.data(), sizeof(double));
      double worst = *std::max_element(times.begin(), times.end());
      double worstErr = *std::max_element(errs.begin(), errs.end());
      if (rank == 0) {
        double algbw = static_cast<double>(bytes) / (worst * 1e-6) / 1e9;
        std::printf("  %12zu %12zu %10s %10.2f %10.2f %10.2f %10.1e\n", bytes, nelem,
                    names[requested], worst, algbw, algbw * kBusFactor, worstErr);
        std::fflush(stdout);
      }
      bootstrap->barrier();
    }
    if (bytes > cfg.maxBytes / 2) break;
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  bootstrap->barrier();
  proxy->stopProxy();
  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(buf));
}

static Mode parseMode(const char* s) {
  if (!std::strcmp(s, "put") || !std::strcmp(s, "sm")) return MODE_SM;
  if (!std::strcmp(s, "ll")) return MODE_LL;
  if (!std::strcmp(s, "ce")) return MODE_CE;
  if (!std::strcmp(s, "auto")) return MODE_AUTO;
  if (!std::strcmp(s, "all")) return MODE_ALL;
  std::fprintf(stderr, "invalid mode '%s' (use all, auto, ll, put, or ce)\n", s);
  std::exit(2);
}

int main(int argc, char** argv) {
  Cfg cfg;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value after %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "-b") cfg.minBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-e") cfg.maxBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-n") cfg.iters = std::atoi(next());
    else if (k == "-w") cfg.warmup = std::atoi(next());
    else if (k == "-B") cfg.blocks = std::atoi(next());
    else if (k == "-T") cfg.threads = std::atoi(next());
    else if (k == "-P") cfg.ceBlocks = std::atoi(next());
    else if (k == "-Q") cfg.ceThreads = std::atoi(next());
    else if (k == "-t") cfg.llThreads = std::atoi(next());
    else if (k == "-l") cfg.llCutoff = std::strtoull(next(), nullptr, 0);
    else if (k == "-C") cfg.ceCutoff = std::strtoull(next(), nullptr, 0);
    else if (k == "-c") cfg.check = std::atoi(next()) != 0;
    else if (k == "-m") cfg.mode = parseMode(next());
    else {
      std::fprintf(stderr, "unknown argument %s\n", k.c_str());
      return 2;
    }
  }
  if (cfg.minBytes == 0 || cfg.maxBytes < cfg.minBytes || cfg.iters < 1 || cfg.warmup < 0 ||
      cfg.blocks < 1 || cfg.threads < 32 || cfg.threads > 1024 || cfg.ceBlocks < 1 ||
      cfg.ceThreads < 32 || cfg.ceThreads > 1024 || cfg.llThreads < 32 || cfg.llThreads > 1024 ||
      cfg.llCutoff < 32 || cfg.ceCutoff <= cfg.llCutoff) {
    std::fprintf(stderr, "invalid benchmark configuration\n");
    return 2;
  }

  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();
  std::vector<pid_t> children;
  for (int rank = 1; rank < kRanks; ++rank) {
    pid_t pid = fork();
    if (pid == 0) {
      runRank(rank, id, cfg);
      _exit(0);
    }
    if (pid < 0) {
      std::perror("fork");
      return 1;
    }
    children.push_back(pid);
  }
  runRank(0, id, cfg);
  int rc = 0;
  for (pid_t pid : children) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) rc = 1;
  }
  return rc;
}

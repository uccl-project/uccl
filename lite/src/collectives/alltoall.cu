// All-to-All for a PCIe-only 4-GPU node, built on MSCCL++.
//
// Topology assumed: 4x NVIDIA L4 (sm_89), PCIe Gen4 x16 each, no NVLink, dual
// Xeon Silver 4410Y.
//     NUMA0: GPU0 GPU1     NUMA1: GPU2 GPU3
// 8 of the 12 flows cross the inter-socket link. Unlike AllGather, All-to-All
// has no freedom in *which* flows exist -- only in *when* each one runs. This
// file is therefore about scheduling and nothing else: every variant moves
// exactly the same 12 chunks over exactly the same links.
//
// Two measured properties of this fabric drive the design:
//
//   fan-out     A GPU can absorb any number of concurrent inbound remote writes
//               for free, but a GPU that *initiates* more than one concurrent
//               remote relationship collapses: one kernel writing 2 peer
//               destinations drops 19.2 -> 3.1 GB/s, reading 2 sources
//               20.7 -> 3.3. A naive SM-issued All-to-All fans out to 3 peers
//               at once and lands at ~5 GB/s aggregate against an 82 GB/s node
//               ceiling. Copy engines are exempt (flat 22.6 GB/s).
//
//   collision   Even on the copy engines, the *order* in which each GPU walks
//   its
//               destination list decides whether destinations collide. Same
//               bytes, same links, same mechanism: 39.9 GB/s colliding vs 74.5
//               staggered.
//
// So the scheduling axis is orthogonal to the mechanism axis, and this
// benchmark crosses them:
//
//   mechanism   sm    SM-issued put over a MemoryChannel
//               ll    LL16 packets (flag-carrying, no barriers at all)
//               ce    copy engine via ProxyService / PortChannel
//
//   schedule    fanout  all 3 peers addressed at once            (the DeepEP
//   shape)
//               naive   peers walked in ascending rank order     (destinations
//               collide) shift   step k: rank r -> (r+k) % N (collision-free
//               rotation) xor     step k: rank r <-> r ^ k (collision-free
//               pairing)
//
// `shift` and `xor` are both collision-free permutations, and they differ only
// in how they spread the 8 cross-socket flows over the 3 steps: shift gives
// (2,4,2), xor gives (0,4,4). That difference is the whole point of keeping
// both.
//
// `host` is not an MSCCL++ variant at all: it is the D2H/H2D staging path, kept
// here because a naive P2P All-to-All is genuinely slower than bouncing every
// byte through system memory -- and because staging is what NCCL itself falls
// back to on this topology, so it is the baseline that actually matters. See
// docs/alltoall.md.
//
// Build: see Makefile

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define NRANKS 4
#define NPEERS (NRANKS - 1)

#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                               \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

using Chan = mscclpp::MemoryChannelDeviceHandle;
using PortChan = mscclpp::PortChannelDeviceHandle;
using Pkt = mscclpp::LL16Packet;

__device__ mscclpp::DeviceSyncer gSyncer;

// Channel handles are indexed by peer *rank*; the self slot is never touched.
struct Chans {
  Chan c[NRANKS];
};
struct PChans {
  PortChan c[NRANKS];
};

// Byte offsets inside the symmetric per-rank buffer. Chunk offsets scale with
// the current message size; the region bases are fixed at the maximum so that a
// single registration serves the whole sweep.
//   [send  NRANKS*NMAX][recv  NRANKS*NMAX][pkt parity0][pkt parity1]
// send chunk d is what this rank sends to rank d; recv chunk s is what it
// receives from rank s. Packet slot s holds the LL packets rank s writes into
// this rank.
struct Layout {
  size_t send;
  size_t recv;
  size_t pkt;
  size_t pktSlot;
};

enum Sched { SC_SHIFT = 0, SC_XOR = 1, SC_NAIVE = 2 };

// Destination of `rank` at step k (k = 1..NPEERS), and the rank that sends to
// `rank` at the same step. For a permutation schedule the two are consistent by
// construction, which is what makes a step collision-free: every GPU has
// exactly one outbound and one inbound flow. SC_NAIVE is not a permutation --
// at step 1 three of the four ranks all address rank 0 -- so it is only ever
// used without step barriers, where nothing has to pair up.
__device__ __host__ __forceinline__ int destOf(int rank, int k, int sched) {
  if (sched == SC_XOR) return rank ^ k;
  if (sched == SC_NAIVE) return (k - 1 < rank) ? (k - 1) : k;
  return (rank + k) % NRANKS;
}
__device__ __host__ __forceinline__ int srcOf(int rank, int k, int sched) {
  if (sched == SC_XOR) return rank ^ k;
  if (sched == SC_NAIVE) return (k - 1 < rank) ? (k - 1) : k;
  return (rank - k + NRANKS) % NRANKS;
}
// i-th peer in ascending rank order, i = 0..NPEERS-1.
__device__ __host__ __forceinline__ int ascPeer(int rank, int i) {
  return (i < rank) ? i : i + 1;
}

// ---------------------------------------------------------------------------
// SM-issued put
// ---------------------------------------------------------------------------

// The DeepEP shape: every peer addressed at once, blocks partitioned across
// destinations so the three writes are genuinely concurrent rather than merely
// consecutive in program order. This is the configuration F2 says collapses.
__global__ void __launch_bounds__(1024)
    a2aPutFanout(Chans ch, int rank, size_t n, Layout L) {
  int const g = blockIdx.x % NPEERS;
  int const nbInG = gridDim.x / NPEERS;
  int const tidG = (blockIdx.x / NPEERS) * blockDim.x + threadIdx.x;
  int const nthG = nbInG * blockDim.x;
  int const d = ascPeer(rank, g);

  ch.c[d].put<16, true>(L.recv + (size_t)rank * n, L.send + (size_t)d * n, n,
                        tidG, nthG);
  gSyncer.sync(gridDim.x);

  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < NPEERS) {
    int p = ascPeer(rank, tid);
    ch.c[p].signal();
    ch.c[p].wait();
  }
  gSyncer.sync(gridDim.x);
}

// One destination at a time, in schedule order.
//
// `bmode` decides how much lockstep is bought, and the two levels turn out to
// be answering different questions:
//   0  none. The blocks free-run, so at any instant different blocks of the
//   same
//      grid are writing different peers -- which is fan-out again, just
//      assembled accidentally instead of on purpose.
//   1  grid sync between steps. Restores the *intra-GPU* invariant: the whole
//   grid
//      is on one destination at a time. Costs nothing off-chip.
//   2  grid sync plus a handshake with the step partner, which additionally
//   holds
//      the four GPUs in the same step of the permutation.
__global__ void __launch_bounds__(1024)
    a2aPutSched(Chans ch, int rank, size_t n, Layout L, int sched, int bmode) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;

  for (int k = 1; k <= NPEERS; k++) {
    int const d = destOf(rank, k, sched);
    ch.c[d].put<16, true>(L.recv + (size_t)rank * n, L.send + (size_t)d * n, n,
                          tid, nth);
    if (bmode == 2) {
      int const s = srcOf(rank, k, sched);
      gSyncer.sync(gridDim.x);
      if (tid == 0) {
        ch.c[d].signal();
        ch.c[s].wait();
      }
      gSyncer.sync(gridDim.x);
    } else if (bmode == 1 && k < NPEERS) {
      gSyncer.sync(gridDim.x);
    }
  }

  // With bmode 2 every peer has already been waited on exactly once, so the
  // collective is complete; otherwise the arrivals still have to be collected.
  if (bmode != 2) {
    gSyncer.sync(gridDim.x);
    if (tid < NPEERS) {
      int p = ascPeer(rank, tid);
      ch.c[p].signal();
      ch.c[p].wait();
    }
    gSyncer.sync(gridDim.x);
  }
}

// ---------------------------------------------------------------------------
// LL packets. The flag carried in each 16-byte packet replaces every barrier,
// so this kernel has no grid sync and no semaphore traffic at all -- which is
// the entire reason it wins below ~64 KiB. Fan-out is unavoidable here (a rank
// writes all three peers) but at these sizes the collective is latency-bound,
// not bandwidth-bound, so the F2 cliff has nothing to bite on.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(1024)
    a2aLL(Chans ch, char* base, int rank, size_t n, Layout L, uint32_t flag,
          int sched) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t nUnit = n / sizeof(uint2);

  for (int k = 1; k <= NPEERS; k++) {
    int const d = destOf(rank, k, sched);
    Pkt* tx = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(ch.c[d].dst_) +
                                     L.pkt + (size_t)rank * L.pktSlot);
    uint2 const* s =
        reinterpret_cast<uint2 const*>(base + L.send + (size_t)d * n);
    for (size_t i = tid; i < nUnit; i += nth) tx[i].write(s[i], flag);
  }
  for (int k = 1; k <= NPEERS; k++) {
    int const q = srcOf(rank, k, sched);
    Pkt* rx = reinterpret_cast<Pkt*>(base + L.pkt + (size_t)q * L.pktSlot);
    uint2* r = reinterpret_cast<uint2*>(base + L.recv + (size_t)q * n);
    for (size_t i = tid; i < nUnit; i += nth) r[i] = rx[i].read(flag);
  }
}

// ---------------------------------------------------------------------------
// Copy engine, via the proxy. Every trigger from one rank lands in the same
// FIFO and MSCCL++ drains it onto a single CudaIpcStream, so the issue order
// below is the execution order -- the schedule is exact, not a hint.
// ---------------------------------------------------------------------------
__global__ void a2aPortSched(PChans pc, int rank, size_t n, Layout L, int sched,
                             int barriered) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __threadfence_system();

  for (int k = 1; k <= NPEERS; k++) {
    int const d = destOf(rank, k, sched);
    pc.c[d].putWithSignal(L.recv + (size_t)rank * n, L.send + (size_t)d * n, n);
    if (barriered) pc.c[srcOf(rank, k, sched)].wait();
  }
  if (!barriered) {
    for (int i = 0; i < NPEERS; i++) pc.c[ascPeer(rank, i)].wait();
  }
}

// ---------------------------------------------------------------------------
// Host-side scaffolding
// ---------------------------------------------------------------------------
// The variants form a factorial over mechanism x destination order x lockstep,
// so the three axes can be read off independently.
enum Variant {
  V_SM_FANOUT = 0,
  V_SM_SHIFT_NB,
  V_SM_NAIVE_GS,
  V_SM_SHIFT_GS,
  V_SM_SHIFT,
  V_LL_NAIVE,
  V_LL_SHIFT,
  V_CE_NAIVE,
  V_CE_SHIFT_NB,
  V_CE_SHIFT,
  V_CE_XOR,
  V_HOST,
  NVAR
};
static char const* kName[NVAR] = {"sm-fanout",   "sm-shift-nb", "sm-naive-gs",
                                  "sm-shift-gs", "sm-shift",    "ll-naive",
                                  "ll-shift",    "ce-naive",    "ce-shift-nb",
                                  "ce-shift",    "ce-xor",      "host"};

// Cross-process barrier living in the shared mmap, so the host-staged variant
// is not charged for a TCP bootstrap round trip between its two phases.
struct HostSync {
  std::atomic<unsigned> count;
  std::atomic<unsigned> sense;
};

// Pipeline depth of the host-staged path: enough to overlap the two directions,
// few enough that the per-chunk spin barrier stays in the noise.
static int const kMaxHostChunks = 16;

struct Cfg {
  size_t minBytes = 1UL << 10;  // per-rank buffer, matching nccl-tests' `size`
  size_t maxBytes = 256UL << 20;
  size_t factor = 2;
  int warmup = 10;
  int iters = 20;
  int blocks = 24;  // divisible by NPEERS for the fan-out kernel
  int threads = 256;
  int llBlocks = 24;
  int llThreads = 512;
  size_t llMaxBytes =
      32UL << 10;  // per-rank; LL doubles wire bytes, so it has a ceiling
  size_t hostChunkBytes = 256UL
                          << 10;  // host-staging pipeline granularity, per flow
  bool check = true;
  int only = -1;
  bool doHost = true;
};

static size_t roundUp(size_t v, size_t a) { return ((v + a - 1) / a) * a; }

static void hostBarrier(HostSync* hs, unsigned& localSense) {
  localSense ^= 1u;
  if (hs->count.fetch_add(1, std::memory_order_acq_rel) + 1 ==
      (unsigned)NRANKS) {
    hs->count.store(0, std::memory_order_relaxed);
    hs->sense.store(localSense, std::memory_order_release);
  } else {
    while (hs->sense.load(std::memory_order_acquire) != localSense)
      __builtin_ia32_pause();
  }
}

static void runRank(int rank, mscclpp::UniqueId id, Cfg const& cfg, void* shmem,
                    size_t stageBytes) {
  CUDACHECK(cudaSetDevice(rank));

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, NRANKS);
  bootstrap->initialize(id);
  mscclpp::Communicator comm(bootstrap);

  const size_t NMAX = cfg.maxBytes / NRANKS;  // bytes per flow, maximum
  const size_t pktSlot =
      roundUp(2 * (cfg.llMaxBytes / NRANKS), 4096);  // LL16 doubles payload
  const size_t sendBase = 0, recvBase = NRANKS * NMAX,
               pktBase = 2 * NRANKS * NMAX;
  const size_t bufBytes = pktBase + 2 * NRANKS * pktSlot;

  // The proxy FIFO packs each offset into 32 bits (mscclpp fifo_device.hpp:
  // TriggerBitsOffset = 32) and the device-side assert that would catch an
  // overflow is compiled out in release builds, so it would corrupt silently.
  if (bufBytes >= (1UL << 32)) {
    std::fprintf(stderr, "-e %zu too large: proxy offsets are 32-bit\n",
                 cfg.maxBytes);
    std::exit(1);
  }

  char* buf = nullptr;
  CUDACHECK(cudaMalloc(&buf, bufBytes));
  CUDACHECK(cudaMemset(buf, 0, bufBytes));

  auto regMem = comm.registerMemory(buf, bufBytes, mscclpp::Transport::CudaIpc);

  std::vector<std::shared_future<mscclpp::Connection>> connF(NRANKS);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> memF(NRANKS);
  for (int i = 0; i < NPEERS; i++) {
    int p = ascPeer(rank, i);
    comm.sendMemory(regMem, p, 0);
    memF[p] = comm.recvMemory(p, 0);
    connF[p] = comm.connect(mscclpp::Transport::CudaIpc, p);
  }
  std::vector<mscclpp::Connection> conns(NRANKS);
  for (int i = 0; i < NPEERS; i++) {
    int p = ascPeer(rank, i);
    conns[p] = connF[p].get();
  }

  // Kept alive for the whole run: the device handles below alias their state.
  std::vector<mscclpp::MemoryChannel> memChans(NRANKS);
  std::vector<mscclpp::PortChannel> portChans(NRANKS);
  auto proxy = std::make_shared<mscclpp::ProxyService>();
  mscclpp::MemoryId midLocal = proxy->addMemory(regMem);

  std::vector<std::shared_future<mscclpp::Semaphore>> semF(NRANKS);
  std::vector<mscclpp::SemaphoreId> sid(NRANKS);
  std::vector<mscclpp::MemoryId> midRemote(NRANKS);
  for (int i = 0; i < NPEERS; i++) {
    int p = ascPeer(rank, i);
    semF[p] = comm.buildSemaphore(conns[p], p);
    sid[p] = proxy->buildAndAddSemaphore(comm, conns[p]);
    midRemote[p] = proxy->addMemory(memF[p].get());
  }

  Chans hChans{};
  PChans hPChans{};
  for (int i = 0; i < NPEERS; i++) {
    int p = ascPeer(rank, i);
    // Both channel types declare operator=(T&), so the temporaries need a name.
    mscclpp::MemoryChannel mc(semF[p].get(), memF[p].get(), regMem, buf);
    mscclpp::PortChannel pc =
        proxy->portChannel(sid[p], midRemote[p], midLocal);
    memChans[p] = mc;
    portChans[p] = pc;
    hChans.c[p] = memChans[p].deviceHandle();
    hPChans.c[p] = portChans[p].deviceHandle();
  }
  proxy->startProxy();

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // The host path needs two: PCIe is full duplex, so the push and the pull of
  // different chunks should be in flight at the same time.
  cudaStream_t streamD, streamH;
  CUDACHECK(cudaStreamCreateWithFlags(&streamD, cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&streamH, cudaStreamNonBlocking));
  std::vector<cudaEvent_t> evD(kMaxHostChunks);
  for (cudaEvent_t& ev : evD)
    CUDACHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));

  // Host staging: one pinned slice per rank in the shared mapping. Rank r parks
  // the chunk bound for d at stage[r][d]; rank d then pulls stage[r][d] back
  // down.
  HostSync* hs = reinterpret_cast<HostSync*>(shmem);
  char* stage = reinterpret_cast<char*>(shmem) + 4096;
  bool hostOk = cfg.doHost;
  if (hostOk) {
    cudaError_t e =
        cudaHostRegister(stage, stageBytes, cudaHostRegisterPortable);
    if (e != cudaSuccess) {
      if (rank == 0)
        std::fprintf(stderr,
                     "# host staging disabled: cudaHostRegister failed (%s)\n",
                     cudaGetErrorString(e));
      hostOk = false;
      cudaGetLastError();
    }
  }
  unsigned localSense = 0;

  std::vector<uint32_t> hostIn, hostOut;
  uint32_t flag = 1;  // strictly increasing, so a stale packet can never match

  auto layoutFor = [&](int parity) {
    Layout L;
    L.send = sendBase;
    L.recv = recvBase;
    L.pkt = pktBase + (size_t)parity * NRANKS * pktSlot;
    L.pktSlot = pktSlot;
    return L;
  };

  // One collective. `iter` only selects the packet parity: a rank that runs
  // ahead must not overwrite packets its peer has not consumed, and two slots
  // is enough because reaching iteration i+2 already implies every peer
  // finished i.
  auto launch = [&](int v, size_t n, int iter) {
    Layout L = layoutFor(iter & 1);
    switch (v) {
      case V_SM_FANOUT:
        a2aPutFanout<<<cfg.blocks, cfg.threads, 0, stream>>>(hChans, rank, n,
                                                             L);
        break;
      case V_SM_SHIFT_NB:
        a2aPutSched<<<cfg.blocks, cfg.threads, 0, stream>>>(hChans, rank, n, L,
                                                            SC_SHIFT, 0);
        break;
      case V_SM_NAIVE_GS:
        a2aPutSched<<<cfg.blocks, cfg.threads, 0, stream>>>(hChans, rank, n, L,
                                                            SC_NAIVE, 1);
        break;
      case V_SM_SHIFT_GS:
        a2aPutSched<<<cfg.blocks, cfg.threads, 0, stream>>>(hChans, rank, n, L,
                                                            SC_SHIFT, 1);
        break;
      case V_SM_SHIFT:
        a2aPutSched<<<cfg.blocks, cfg.threads, 0, stream>>>(hChans, rank, n, L,
                                                            SC_SHIFT, 2);
        break;
      case V_LL_NAIVE:
        a2aLL<<<cfg.llBlocks, cfg.llThreads, 0, stream>>>(hChans, buf, rank, n,
                                                          L, flag++, SC_NAIVE);
        break;
      case V_LL_SHIFT:
        a2aLL<<<cfg.llBlocks, cfg.llThreads, 0, stream>>>(hChans, buf, rank, n,
                                                          L, flag++, SC_SHIFT);
        break;
      case V_CE_NAIVE:
        a2aPortSched<<<1, 32, 0, stream>>>(hPChans, rank, n, L, SC_NAIVE, 0);
        break;
      case V_CE_SHIFT_NB:
        a2aPortSched<<<1, 32, 0, stream>>>(hPChans, rank, n, L, SC_SHIFT, 0);
        break;
      case V_CE_SHIFT:
        a2aPortSched<<<1, 32, 0, stream>>>(hPChans, rank, n, L, SC_SHIFT, 1);
        break;
      case V_CE_XOR:
        a2aPortSched<<<1, 32, 0, stream>>>(hPChans, rank, n, L, SC_XOR, 1);
        break;
      default:
        break;
    }
  };

  // The host-staged path cannot be enqueued and left alone: a rank may not pull
  // a chunk until every rank has pushed it. Chunking is what keeps that from
  // costing half the link -- pushes of later chunks overlap pulls of earlier
  // ones, and PCIe is full duplex, so the two directions cost nothing to run
  // together. Without it the collective degenerates into a D2H phase followed
  // by an H2D phase. The barrier is a spin on the shared mapping rather than a
  // bootstrap round trip, for the same reason: this baseline should be given
  // its best shot.
  auto launchHost = [&](size_t n) {
    // One chunk below the point where overlap can pay for a barrier: each extra
    // chunk costs an event sync plus a four-way spin, which is pure loss on a
    // transfer too short to hide it.
    const int ncWant = (int)std::max<size_t>(
        1, std::min<size_t>(kMaxHostChunks, n / cfg.hostChunkBytes));
    const size_t cl = roundUp((n + ncWant - 1) / ncWant, 4096);
    const int nc =
        (int)((n + cl - 1) / cl);  // <= ncWant after the alignment round-up
    for (int c = 0; c < nc; c++) {
      const size_t off = (size_t)c * cl, len = std::min(cl, n - off);
      for (int i = 0; i < NPEERS; i++) {
        int d = ascPeer(rank, i);
        CUDACHECK(
            cudaMemcpyAsync(stage + ((size_t)rank * NRANKS + d) * NMAX + off,
                            buf + sendBase + (size_t)d * n + off, len,
                            cudaMemcpyDeviceToHost, streamD));
      }
      CUDACHECK(cudaEventRecord(evD[c], streamD));
    }
    for (int c = 0; c < nc; c++) {
      const size_t off = (size_t)c * cl, len = std::min(cl, n - off);
      CUDACHECK(cudaEventSynchronize(evD[c]));
      hostBarrier(hs, localSense);
      for (int i = 0; i < NPEERS; i++) {
        int s = ascPeer(rank, i);
        CUDACHECK(
            cudaMemcpyAsync(buf + recvBase + (size_t)s * n + off,
                            stage + ((size_t)s * NRANKS + rank) * NMAX + off,
                            len, cudaMemcpyHostToDevice, streamH));
      }
    }
    CUDACHECK(cudaStreamSynchronize(streamH));
  };

  // Position-sensitive fill: the word at index j of the chunk rank s sends to
  // rank d is a function of (s, d, j), so a chunk landing in the wrong slot,
  // from the wrong sender, or at the wrong offset is all caught.
  auto word = [](int s, int d, size_t j) -> uint32_t {
    return (uint32_t)((s * NRANKS + d) * 2654435761u + (uint32_t)j * 40503u +
                      12345u);
  };
  auto fill = [&](size_t n) {
    const size_t ne = n / sizeof(uint32_t);
    hostIn.resize(ne * NRANKS);
    for (int d = 0; d < NRANKS; d++)
      for (size_t j = 0; j < ne; j++)
        hostIn[(size_t)d * ne + j] = word(rank, d, j);
    CUDACHECK(cudaMemcpy(buf + sendBase, hostIn.data(), n * NRANKS,
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(buf + recvBase, 0, n * NRANKS));
  };
  auto verify = [&](size_t n) -> long long {
    const size_t ne = n / sizeof(uint32_t);
    hostOut.resize(ne * NRANKS);
    CUDACHECK(cudaMemcpy(hostOut.data(), buf + recvBase, n * NRANKS,
                         cudaMemcpyDeviceToHost));
    long long bad = 0;
    for (int s = 0; s < NRANKS; s++) {
      if (s == rank) continue;  // self chunk is not exchanged
      for (size_t j = 0; j < ne; j++)
        if (hostOut[(size_t)s * ne + j] != word(s, rank, j)) bad++;
    }
    return bad;
  };

  if (rank == 0) {
    std::printf("# MSCCL++ All-to-All, %d x L4, PCIe only\n", NRANKS);
    std::printf(
        "# put: %d x %d   LL: %d x %d   LL cutoff %zu B/rank   size = per-rank "
        "buffer (nccl-tests convention)\n",
        cfg.blocks, cfg.threads, cfg.llBlocks, cfg.llThreads, cfg.llMaxBytes);
    std::printf(
        "# aggregate = total bytes on the wire / time = 4 x busbw\n#\n");
    std::printf("# %12s %11s %13s %10s %9s %9s %10s %8s\n", "size(B)",
                "perflow(B)", "variant", "time(us)", "algbw", "busbw",
                "aggregate", "wrong");
  }

  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes;
       bytes *= cfg.factor) {
    const size_t n = bytes / NRANKS;
    if (n < 64)
      continue;  // one LL unit per thread-group, and 16 B put alignment

    for (int v = 0; v < NVAR; v++) {
      if (cfg.only >= 0 && v != cfg.only) continue;
      if ((v == V_LL_NAIVE || v == V_LL_SHIFT) && bytes > cfg.llMaxBytes)
        continue;
      if (v == V_HOST && !hostOk) continue;

      long long wrong = -1;
      if (cfg.check) {
        fill(n);
        bootstrap->barrier();
        if (v == V_HOST) {
          launchHost(n);
        } else {
          launch(v, n, 0);
          CUDACHECK(cudaStreamSynchronize(stream));
        }
        wrong = verify(n);
        bootstrap->barrier();
      }

      fill(n);
      for (int i = 0; i < cfg.warmup; i++) {
        if (v == V_HOST)
          launchHost(n);
        else
          launch(v, n, i);
      }
      CUDACHECK(cudaStreamSynchronize(stream));
      bootstrap->barrier();

      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < cfg.iters; i++) {
        if (v == V_HOST)
          launchHost(n);
        else
          launch(v, n, i);
      }
      CUDACHECK(cudaStreamSynchronize(stream));
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                  cfg.iters;

      // The collective is only as fast as its slowest participant.
      std::vector<double> all(NRANKS, 0.0);
      all[rank] = us;
      bootstrap->allGather(all.data(), sizeof(double));
      double worst = *std::max_element(all.begin(), all.end());

      std::vector<long long> errs(NRANKS, 0);
      errs[rank] = wrong;
      bootstrap->allGather(errs.data(), sizeof(long long));
      long long worstErr = *std::max_element(errs.begin(), errs.end());

      if (rank == 0) {
        double algbw = (double)bytes / (worst * 1e-6) / 1e9;
        double busbw = algbw * (NRANKS - 1) / NRANKS;
        std::printf("  %12zu %11zu %13s %10.2f %9.2f %9.2f %10.2f %8lld\n",
                    bytes, n, kName[v], worst, algbw, busbw, busbw * NRANKS,
                    worstErr);
        std::fflush(stdout);
      }
      bootstrap->barrier();
    }
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  bootstrap->barrier();
  if (hostOk) cudaHostUnregister(stage);
  proxy->stopProxy();
  for (cudaEvent_t& ev : evD) CUDACHECK(cudaEventDestroy(ev));
  CUDACHECK(cudaStreamDestroy(streamH));
  CUDACHECK(cudaStreamDestroy(streamD));
  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(buf));
}

int main(int argc, char** argv) {
  Cfg cfg;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto next = [&]() -> char const* {
      return (i + 1 < argc) ? argv[++i] : "0";
    };
    if (k == "-b")
      cfg.minBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-e")
      cfg.maxBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-f")
      cfg.factor = std::strtoull(next(), nullptr, 0);
    else if (k == "-n")
      cfg.iters = std::atoi(next());
    else if (k == "-w")
      cfg.warmup = std::atoi(next());
    else if (k == "-B")
      cfg.blocks = std::atoi(next());
    else if (k == "-T")
      cfg.threads = std::atoi(next());
    else if (k == "-L")
      cfg.llBlocks = std::atoi(next());
    else if (k == "-t")
      cfg.llThreads = std::atoi(next());
    else if (k == "-l")
      cfg.llMaxBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-k")
      cfg.hostChunkBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-c")
      cfg.check = std::atoi(next()) != 0;
    else if (k == "-H")
      cfg.doHost = std::atoi(next()) != 0;
    else if (k == "-a") {
      std::string want = next();
      cfg.only = -1;
      for (int j = 0; j < NVAR; j++)
        if (want == kName[j]) cfg.only = j;
      if (cfg.only < 0) {
        std::fprintf(stderr, "unknown variant %s\n", want.c_str());
        return 1;
      }
    } else {
      std::fprintf(stderr, "unknown arg %s\n", k.c_str());
      return 1;
    }
  }
  if (cfg.blocks % NPEERS != 0) {
    std::fprintf(stderr,
                 "-B must be a multiple of %d (fan-out kernel partitions "
                 "blocks by peer)\n",
                 NPEERS);
    return 1;
  }

  // Shared, page-aligned staging for the host baseline, mapped before the fork
  // so every rank sees the same physical pages; each rank pins its own mapping
  // of it.
  const size_t stageBytes = (size_t)NRANKS * cfg.maxBytes;
  void* shmem = mmap(nullptr, stageBytes + 4096, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (shmem == MAP_FAILED) {
    std::fprintf(
        stderr,
        "mmap of %zu B failed; run with -H 0 to skip the host baseline\n",
        stageBytes);
    return 1;
  }
  std::memset(shmem, 0, 4096);

  // Fork before touching CUDA so every rank gets a clean context. The unique id
  // is created in the parent and inherited, which removes the need for MPI.
  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();

  std::vector<pid_t> kids;
  for (int r = 1; r < NRANKS; r++) {
    pid_t p = fork();
    if (p == 0) {
      runRank(r, id, cfg, shmem, stageBytes);
      _exit(0);
    }
    kids.push_back(p);
  }
  runRank(0, id, cfg, shmem, stageBytes);

  int rc = 0;
  for (pid_t p : kids) {
    int st = 0;
    waitpid(p, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1;
  }
  munmap(shmem, stageBytes + 4096);
  return rc;
}

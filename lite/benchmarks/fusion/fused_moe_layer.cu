// The whole MoE layer on a PCIe-only 4-GPU node: dispatch -> GEMM -> SwiGLU ->
// GEMM -> combine, with both collectives overlapped into the CUTLASS pipeline.
//
// Topology: 4x NVIDIA L4 (sm_89), PCIe Gen4 x16 each, no NVLink, dual Xeon
// Silver 4410Y.  NUMA0: GPU0 GPU1   NUMA1: GPU2 GPU3
//
// Shape is DeepSeek-V2-Lite at EP=4, bf16 in/out with fp32 accumulate:
//   H 2048, 64 routed experts (16 per rank), top-6 softmax without renorm,
//   gate/up [2048 -> 1408] each (fused to N=2816), down [1408 -> 2048].
// The 2 shared experts are out of scope; they are a dense path that depends on
// no routing and no collective, so including them would add ~25% compute that
// dilutes the communication share this file is about.
//
// benchmarks/fusion/fused_moe.cu measures the first half of this -- dispatch +
// GEMM1 -- and gets 1.48-1.61x over NCCL + grouped cuBLAS. This file is the
// rest, and the reason it is a separate binary is that the half-layer result
// should stay reproducible exactly as published.
//
// ---------------------------------------------------------------------------
// The one thing worth knowing before reading the code
// ---------------------------------------------------------------------------
// Dispatch and combine look symmetric and are not, because their dependency
// arrows point opposite ways:
//
//   dispatch   comm -> compute.  A GEMM tile has to *wait*. That needs an
//              in-kernel gate, the collective living in block 0 of the GEMM's
//              own grid, and gather-A so no second kernel needs an SM. All of
//              that is built (cutlass_gated_grouped.h) and hard-won: every
//              separate-kernel version of it deadlocked.
//
//   combine    compute -> comm.  *Nothing waits.* putWithSignal posts a
//              descriptor to the proxy FIFO and returns; the copy engine moves
//              the bytes while the next kernel owns every SM. Stream order
//              alone gives the overlap.
//
// So the combine needs no fused kernel at all. It needs GEMM2 cut into chunks
// of arrival groups, so a chunk's rows can be reduced and put on their way
// while the next chunk computes. `-Q` is that chunk count and `-Q 1` is the
// control that prices it.
//
// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------
//   stage        gather this rank's own rows into the send slots
//   K1           block 0 runs the ce-shift dispatch and publishes arrived[k];
//                blocks 1..N run a gated CUTLASS grouped GEMM that gathers its
//                A operand straight out of the receive buffer.
//                [M,2048] x [2048,2816] -> h
//   K2, per chunk q:
//       swiglu   h[.,2816] -> a[.,1408]
//       GEMM2    [Mq,1408] x [1408,2048] -> y      (plain grouped, no gather)
//       reduce   weighted segmented sum over the reduce CSR -> combine send
//       put      1-block kernel; returns at once, transfer overlaps chunk q+1
//   K3           wait for the three incoming combine signals
//   K4           accumulate the four partials into out[T, 2048]
//
// Build: see Makefile

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/numeric_types.h"
#include "cutlass_gated_grouped.h"
#include "megamoe_core.cuh"
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/wait.h>
#include <unistd.h>

#define CUBLASCHECK(cmd)                                                  \
  do {                                                                    \
    cublasStatus_t s_ = (cmd);                                            \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "%s:%d cuBLAS error %d\n", __FILE__, __LINE__, \
                   (int)s_);                                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#define NCCLCHECK(cmd)                                                   \
  do {                                                                   \
    ncclResult_t r_ = (cmd);                                             \
    if (r_ != ncclSuccess) {                                             \
      std::fprintf(stderr, "%s:%d NCCL error: %s\n", __FILE__, __LINE__, \
                   ncclGetErrorString(r_));                              \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

// Routing is a pure function of (rank, token), so every rank can recompute
// every other rank's assignment and the movement checks are exact rather than
// statistical. The expert *selection* here is identical to the one in
// fused_moe.cu -- same hash, same accept/reject sequence -- so the two binaries
// agree on plans; only the weights are new. They are a softmax over the top-k
// logits, matching the model's scoring_func=softmax with norm_topk_prob=false.
__host__ __forceinline__ void routeOf(int s, int t, int* experts,
                                      float* weights) {
  uint32_t h =
      (uint32_t)(s * 2654435761u) ^ (uint32_t)(t * 40503u) ^ 0x9e3779b9u;
  float logit[kTopK];
  int n = 0;
  while (n < kTopK) {
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    int const e = (int)(h % kE);
    bool dup = false;
    for (int i = 0; i < n; i++)
      if (experts[i] == e) dup = true;
    if (dup) continue;
    experts[n] = e;
    logit[n] = (float)((h >> 8) % 1000) * 0.002f - 1.0f;
    n++;
  }
  if (!weights) return;
  float mx = logit[0];
  for (int i = 1; i < kTopK; i++) mx = logit[i] > mx ? logit[i] : mx;
  float sum = 0.0f;
  for (int i = 0; i < kTopK; i++) {
    weights[i] = std::exp(logit[i] - mx);
    sum += weights[i];
  }
  for (int i = 0; i < kTopK; i++) weights[i] /= sum;
}

// Deterministic activations, so any rank can reconstruct any other rank's
// tokens and the dispatch check can compare exact values.
__host__ __device__ __forceinline__ float xvalf(int s, int t, int i) {
  const uint32_t h =
      (uint32_t)(s * 7919 + t) * 2654435761u + (uint32_t)i * 40503u;
  return (float)(h % 13) * 0.125f - 0.75f;
}

// ---------------------------------------------------------------------------
// Host-side plan
// ---------------------------------------------------------------------------
enum Variant {
  V_GEMM_ONLY = 0,  // full compute, no comm -- cuBLAS. The compute floor.
  V_CUTLASS_ONLY,   // same, CUTLASS grouped GEMMs
  V_DISP_ONLY,      // the dispatch alone
  V_COMM_ONLY,      // both collectives, no compute. The comm floor.
  V_NCCL_CUBLAS,    // the baseline
  V_OURS_CUBLAS,    // + ce-shift instead of NCCL
  V_OURS_CUTLASS,   // + CUTLASS grouped GEMMs, still strictly sequential
  V_FUSED,          // + dispatch in block 0, gated tiles, chunked combine
  NVAR
};
// disp-only and comm-only are both here so the combine's cost is a difference
// of two measured rows rather than half of one -- the two collectives carry
// different byte counts on a given rank, since a token crosses out once per
// destination and back once per source.
static char const* kName[NVAR] = {"gemm-only",    "cutlass-only", "disp-only",
                                  "comm-only",    "nccl+cublas",  "ours+cublas",
                                  "ours+cutlass", "fused"};

struct Cfg {
  int minT = 8, maxT = 16384, factorT = 4;
  // Few iterations per visit and many rounds, because the variants are
  // interleaved: short visits keep a slow variant from throttling itself
  // inside its own measurement. See the timing block in runRank.
  int warmup = 2, iters = 4, rounds = 12;
  double settleMs = 2500.0;
  // 0 none, 1 against the fp32 GPU reference, 2 also against the fp64 host
  // reference (small T only).
  int check = 1;
  int only = -1;
  bool skip[NVAR] = {false};
  // SMs held back from the gated GEMM's grid, so block 0's dispatcher is
  // guaranteed a slot.
  int reserveSM = 1;
  // Where the four arrival groups are cut into two gate groups for GEMM1.
  // 2 is a measured optimum, not an unquestioned default: an even split beats
  // an early start because 380 rows is exactly 3 tiles of BM=128 at 99% fill
  // while 190 needs 2 at 74% (see fused_moe.cu's -P).
  int gateSplit = 2;
  // How many chunks the combine is cut into. More chunks put more of the
  // combine behind compute, because a chunk's rows cannot be sent before they
  // are computed and the *last* chunk is always exposed -- making it smaller is
  // the whole lever. They cost tile fill in return: at -Q 4 an expert's chunk
  // is ~190 rows into a 128-row tile, 74% full.
  //
  //   measured, hidden share of comm-only, T=2048/4096/8192
  //     -Q 2   43% / 47% / 44%      1.39x / 1.38x / 1.38x
  //     -Q 3   65% / 70% / 67%      1.40x / 1.51x / 1.53x
  //
  // 3 wins, and the boundaries are {0,1},{2},{3} -- the two cheap groups last,
  // where the remaining compute is smallest.
  //
  // -Q 1 is meant to be the no-overlap control and it deadlocks; see the
  // Limitations section of docs/fusion.md. The Q=2/Q=3 pair prices the
  // overlap instead.
  int chunks = 3;
};

struct Plan {
  int T = 0, cap = 0;
  std::vector<int> sendTok[NRANKS];  // rows this rank owes each peer
  int nSend[NRANKS] = {0, 0, 0, 0};
  std::vector<int> recvTok[NRANKS];  // rows arriving from each source
  int nRecv[NRANKS] = {0, 0, 0, 0};
  std::vector<int> rowIdx;  // expert-major, group-minor
  int expOff[kELoc] = {0};
  int grpOff[kELoc][NRANKS] = {{0}};
  int grpCnt[kELoc][NRANKS] = {{0}};
  int totalRows = 0;
  std::vector<int> redOff, redRow;  // CSR over (source, token) slots
  std::vector<float> redW;
};

static void buildPlan(Plan& p, int rank, int T, int cap) {
  p.T = T;
  p.cap = cap;
  int ex[kTopK];
  float w[kTopK];

  // What this rank sends: a token crosses to a peer once if any of its experts
  // lives there. That dedup is what makes the flows ragged and unequal.
  for (int d = 0; d < NRANKS; d++) p.sendTok[d].clear();
  for (int t = 0; t < T; t++) {
    routeOf(rank, t, ex, nullptr);
    bool want[NRANKS] = {false, false, false, false};
    for (int i = 0; i < kTopK; i++) want[ex[i] / kELoc] = true;
    for (int d = 0; d < NRANKS; d++)
      if (want[d]) p.sendTok[d].push_back(t);
  }
  for (int d = 0; d < NRANKS; d++) p.nSend[d] = (int)p.sendTok[d].size();

  // What arrives: recomputed from the same function, so it matches by
  // construction and the check below is exact rather than statistical.
  for (int s = 0; s < NRANKS; s++) {
    p.recvTok[s].clear();
    for (int t = 0; t < T; t++) {
      routeOf(s, t, ex, nullptr);
      for (int i = 0; i < kTopK; i++)
        if (ex[i] / kELoc == rank) {
          p.recvTok[s].push_back(t);
          break;
        }
    }
    p.nRecv[s] = (int)p.recvTok[s].size();
  }

  // Row order: expert-major so each expert's rows are contiguous for a grouped
  // GEMM, group-minor inside that so a (chunk, expert) span is contiguous too.
  // One layout serves the grouped calls, the arrival-gated GEMM1 problems and
  // the combine chunks. The (source, token, expert) each row came from is
  // recorded on the way past, which makes the reduce CSR a single pass.
  p.rowIdx.clear();
  std::vector<int> rowSrc, rowTokIdx, rowExp;
  int pos = 0;
  for (int e = 0; e < kELoc; e++) {
    p.expOff[e] = pos;
    for (int g = 0; g < NRANKS; g++) {
      int const s = srcOfGroup(rank, g);
      p.grpOff[e][g] = pos - p.expOff[e];
      for (int j = 0; j < p.nRecv[s]; j++) {
        routeOf(s, p.recvTok[s][j], ex, nullptr);
        for (int i = 0; i < kTopK; i++)
          if (ex[i] == rank * kELoc + e) {
            // Slots are indexed at the sweep-wide capacity, not at this T, so
            // one registration and one row-index space serve every size.
            p.rowIdx.push_back(s * cap + j);
            rowSrc.push_back(s);
            rowTokIdx.push_back(j);
            rowExp.push_back(e);
            pos++;
            break;
          }
      }
      p.grpCnt[e][g] = pos - p.expOff[e] - p.grpOff[e][g];
    }
  }
  p.totalRows = pos;

  p.redOff.assign((size_t)NRANKS * cap + 1, 0);
  {
    std::vector<int> cnt((size_t)NRANKS * cap, 0);
    for (int r = 0; r < pos; r++) cnt[(size_t)rowSrc[r] * cap + rowTokIdx[r]]++;
    for (size_t i = 0; i < (size_t)NRANKS * cap; i++)
      p.redOff[i + 1] = p.redOff[i] + cnt[i];
    std::vector<int> cur(p.redOff.begin(), p.redOff.end() - 1);
    p.redRow.assign(pos, 0);
    p.redW.assign(pos, 0.0f);
    for (int r = 0; r < pos; r++) {
      int const s = rowSrc[r], j = rowTokIdx[r], e = rowExp[r];
      routeOf(s, p.recvTok[s][j], ex, w);
      float wt = 0.0f;
      for (int i = 0; i < kTopK; i++)
        if (ex[i] == rank * kELoc + e) wt = w[i];
      int const slot = cur[(size_t)s * cap + j]++;
      p.redRow[slot] = r;
      p.redW[slot] = wt;
    }
  }
}

// Independent host reference, fp64, for small T.
//
// The GPU reference below shares every index structure with the paths it
// checks -- the same rowIdx, the same reduce CSR, the same sendTok -- so a bug
// in buildPlan would be invisible to it: both sides would be wrong the same
// way. This one uses nothing but routeOf, xvalf and the weight generator, so it
// validates the dispatch slots, the local reduce and the combine mapping rather
// than assuming them.
//
// The weights are the same numbers on every rank (wval is a pure function of
// the flat index), so an expert is identified by its local index alone and this
// needs no cross-rank exchange. O(T x topk x H x (2I + I)) in double, which is
// a second or two at T=32 and hopeless above ~64.
static __nv_bfloat16 wvalOf(size_t i) {
  return __float2bfloat16((float)((i * 2654435761u) % 17) * 0.0625f - 0.5f);
}

static void hostReference(int rank, int T, std::vector<double>& outH) {
  outH.assign((size_t)T * kH, 0.0);
  int ex[kTopK];
  float w[kTopK];

  std::vector<float> wgu((size_t)kGU * kH), wdn((size_t)kH * kI);
  std::vector<double> hv(kGU), av(kI);
  for (int e = 0; e < kELoc; e++) {
    for (size_t j = 0; j < (size_t)kGU * kH; j++)
      wgu[j] = __bfloat162float(wvalOf((size_t)e * kGU * kH + j));
    for (size_t j = 0; j < (size_t)kH * kI; j++)
      wdn[j] = __bfloat162float(wvalOf((size_t)e * kH * kI + j + 7717));

    for (int t = 0; t < T; t++) {
      routeOf(rank, t, ex, w);
      for (int i = 0; i < kTopK; i++) {
        if (ex[i] % kELoc != e) continue;
        // gate/up
        for (int j = 0; j < kGU; j++) {
          double acc = 0.0;
          float const* wr = wgu.data() + (size_t)j * kH;
          for (int k = 0; k < kH; k++)
            acc +=
                (double)__bfloat162float(__float2bfloat16(xvalf(rank, t, k))) *
                wr[k];
          hv[j] = acc;
        }
        for (int j = 0; j < kI; j++)
          hv[j] = hv[j] / (1.0 + std::exp(-hv[j])) * hv[kI + j];
        for (int j = 0; j < kI; j++) av[j] = hv[j];
        // down, weighted into the token's output
        for (int o = 0; o < kH; o++) {
          double acc = 0.0;
          float const* wr = wdn.data() + (size_t)o * kI;
          for (int j = 0; j < kI; j++) acc += av[j] * wr[j];
          outH[(size_t)t * kH + o] += (double)w[i] * acc;
        }
      }
    }
  }
}

static void runRank(int rank, mscclpp::UniqueId id, Cfg const& cfg) {
  // FML_TRACE=1 prints per-phase progress to stderr, unbuffered. The rows are
  // collective, so a hang shows up as one rank missing a line the others have.
  bool const traceOn = getenv("FML_TRACE") != nullptr;
#define TRACE(...)                          \
  do {                                      \
    if (traceOn) {                          \
      std::fprintf(stderr, "[r%d] ", rank); \
      std::fprintf(stderr, __VA_ARGS__);    \
      std::fprintf(stderr, "\n");           \
    }                                       \
  } while (0)
  CUDACHECK(cudaSetDevice(rank));
  // Host-mapped progress word, FML_TRACE=1 only. A collective kernel that is
  // stuck is stuck *spinning*, and a spinning kernel never flushes device
  // printf, so the only way to see where it stopped is a word the GPU writes
  // into mapped host memory and a host thread reads while the kernel still
  // runs. Codes: 1x issuing dispatch put k, 2x waiting dispatch k, 29 dispatch
  // done, 3x issuing combine put k, 39 puts done, 4x waiting combine k, 49
  // done. Costs nothing when FML_TRACE is unset: the pointer is null and the
  // stores are predicated away.
  int volatile* hProg = nullptr;
  int* dProg = nullptr;
  if (traceOn) {
    CUDACHECK(cudaHostAlloc((void**)&hProg, sizeof(int), cudaHostAllocMapped));
    *hProg = 0;
    CUDACHECK(cudaHostGetDevicePointer((void**)&dProg, (void*)hProg, 0));
    std::thread([hProg, rank]() {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::fprintf(stderr, "[r%d] prog=%d\n", rank, *hProg);
      }
    }).detach();
  }
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, NRANKS);
  bootstrap->initialize(id);
  mscclpp::Communicator comm(bootstrap);

  const size_t CAP = (size_t)cfg.maxT;
  const size_t slot = CAP * kH * sizeof(__nv_bfloat16);
  // [dispatch send][dispatch recv][combine send][combine recv]. The combine
  // gets its own regions rather than reusing the dispatch ones: they are the
  // same size and would alias perfectly, but the dispatch receive buffer is
  // still the GEMM's A operand, and keeping them apart leaves room to start the
  // combine before GEMM1 has drained.
  const size_t dSend = 0, dRecv = NRANKS * slot, cSend = 2 * NRANKS * slot,
               cRecv = 3 * NRANKS * slot;
  const size_t bufBytes = 4 * NRANKS * slot;
  // The proxy FIFO packs each offset into 32 bits (mscclpp fifo_device.hpp,
  // TriggerBitsOffset = 32) and the device-side assert that would catch an
  // overflow is compiled out in release builds, so it would corrupt silently.
  // Four regions rather than two halves the reachable T: 32768 is the cap.
  if (bufBytes >= (1UL << 32)) {
    std::fprintf(stderr,
                 "-e %d too large: four %zu MiB regions overflow the 32-bit "
                 "proxy offset\n",
                 cfg.maxT, slot >> 20);
    std::exit(1);
  }

  char* buf = nullptr;
  CUDACHECK(cudaMalloc(&buf, bufBytes));
  CUDACHECK(cudaMemset(buf, 0, bufBytes));
  auto regMem = comm.registerMemory(buf, bufBytes, mscclpp::Transport::CudaIpc);

  std::vector<std::shared_future<mscclpp::Connection>> connF(NRANKS);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> memF(NRANKS);
  for (int i = 0; i < NPEERS; i++) {
    int const q = ascPeer(rank, i);
    comm.sendMemory(regMem, q, 0);
    memF[q] = comm.recvMemory(q, 0);
    connF[q] = comm.connect(mscclpp::Transport::CudaIpc, q);
  }
  std::vector<mscclpp::Connection> conns(NRANKS);
  for (int i = 0; i < NPEERS; i++)
    conns[ascPeer(rank, i)] = connF[ascPeer(rank, i)].get();

  std::vector<mscclpp::PortChannel> portChans(NRANKS);
  auto proxy = std::make_shared<mscclpp::ProxyService>();
  mscclpp::MemoryId midLocal = proxy->addMemory(regMem);
  std::vector<mscclpp::SemaphoreId> sid(NRANKS);
  std::vector<mscclpp::MemoryId> midRemote(NRANKS);
  for (int i = 0; i < NPEERS; i++) {
    int const q = ascPeer(rank, i);
    sid[q] = proxy->buildAndAddSemaphore(comm, conns[q]);
    midRemote[q] = proxy->addMemory(memF[q].get());
  }
  PChans hPc{};
  for (int i = 0; i < NPEERS; i++) {
    int const q = ascPeer(rank, i);
    // Through a named local: PortChannel's assignment takes a non-const
    // lvalue reference, so the temporary cannot bind directly.
    mscclpp::PortChannel pcx =
        proxy->portChannel(sid[q], midRemote[q], midLocal);
    portChans[q] = pcx;
    hPc.c[q] = portChans[q].deviceHandle();
  }
  proxy->startProxy();

  // NCCL over the same four processes; the unique id rides the bootstrap that
  // is already there, so no MPI and no second rendezvous.
  ncclUniqueId nid;
  std::vector<ncclUniqueId> nids(NRANKS);
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&nid));
  nids[rank] = nid;
  bootstrap->allGather(nids.data(), sizeof(ncclUniqueId));
  ncclComm_t ncclComm;
  NCCLCHECK(ncclCommInitRank(&ncclComm, NRANKS, nids[0], rank));

  cublasHandle_t cub;
  CUBLASCHECK(cublasCreate(&cub));
  cudaStream_t sMain;
  CUDACHECK(cudaStreamCreateWithFlags(&sMain, cudaStreamNonBlocking));
  CUBLASCHECK(cublasSetStream(cub, sMain));

  Plan maxPlan;
  buildPlan(maxPlan, rank, cfg.maxT, (int)CAP);
  const size_t maxRows = (size_t)maxPlan.totalRows + FM_BM;
  size_t maxMe = 1;
  for (int e = 0; e < kELoc; e++) {
    size_t m = 0;
    for (int g = 0; g < NRANKS; g++) m += maxPlan.grpCnt[e][g];
    maxMe = std::max(maxMe, m);
  }
  maxMe += FM_BM;

  __nv_bfloat16 *x, *Wgu, *Wdn, *h, *a, *y, *Ap, *out, *outRef;
  float* h32;
  CUDACHECK(cudaMalloc(&x, CAP * kH * 2));
  CUDACHECK(cudaMalloc(&Wgu, (size_t)kELoc * kGU * kH * 2));
  CUDACHECK(cudaMalloc(&Wdn, (size_t)kELoc * kH * kI * 2));
  CUDACHECK(cudaMalloc(&h, maxRows * kGU * 2));
  CUDACHECK(cudaMalloc(&a, maxRows * kI * 2));
  CUDACHECK(cudaMalloc(&y, maxRows * kH * 2));
  CUDACHECK(cudaMalloc(&Ap, maxRows * kH * 2));
  CUDACHECK(cudaMalloc(&out, CAP * kH * 2));
  CUDACHECK(cudaMalloc(&outRef, CAP * kH * 2));
  // fp32 h for the reference only, and only one expert wide -- the reference
  // walks experts one at a time, so sizing this for all rows would cost 2 GB
  // to hold data that is dead one launch later.
  CUDACHECK(cudaMalloc(&h32, maxMe * kGU * sizeof(float)));

  int *dRowIdx, *dSendTok, *dRedOff, *dRedRow;
  float* dRedW;
  uint32_t* dArrived;
  size_t *dSendOff, *dRecvOff, *dSendBytes, *dCSendOff, *dCRecvOff,
      *dCSendBytes;
  CUDACHECK(cudaMalloc(&dRowIdx, maxRows * sizeof(int)));
  CUDACHECK(cudaMalloc(&dSendTok, (size_t)NRANKS * CAP * sizeof(int)));
  CUDACHECK(cudaMalloc(&dRedOff, ((size_t)NRANKS * CAP + 1) * sizeof(int)));
  CUDACHECK(cudaMalloc(&dRedRow, maxRows * sizeof(int)));
  CUDACHECK(cudaMalloc(&dRedW, maxRows * sizeof(float)));
  CUDACHECK(cudaMalloc(&dArrived, NRANKS * sizeof(uint32_t)));
  CUDACHECK(cudaMalloc(&dSendOff, NRANKS * sizeof(size_t)));
  CUDACHECK(cudaMalloc(&dRecvOff, NRANKS * sizeof(size_t)));
  CUDACHECK(cudaMalloc(&dSendBytes, NRANKS * sizeof(size_t)));
  CUDACHECK(cudaMalloc(&dCSendOff, NRANKS * sizeof(size_t)));
  CUDACHECK(cudaMalloc(&dCRecvOff, NRANKS * sizeof(size_t)));
  CUDACHECK(cudaMalloc(&dCSendBytes, NRANKS * sizeof(size_t)));

  {
    std::vector<__nv_bfloat16> hx(CAP * kH);
    for (size_t t = 0; t < CAP; t++)
      for (int i = 0; i < kH; i++)
        hx[t * kH + i] = __float2bfloat16(xvalf(rank, (int)t, i));
    CUDACHECK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));

    std::vector<__nv_bfloat16> wg((size_t)kELoc * kGU * kH);
    for (size_t i = 0; i < wg.size(); i++) wg[i] = wvalOf(i);
    CUDACHECK(
        cudaMemcpy(Wgu, wg.data(), wg.size() * 2, cudaMemcpyHostToDevice));
    std::vector<__nv_bfloat16> wd((size_t)kELoc * kH * kI);
    for (size_t i = 0; i < wd.size(); i++) wd[i] = wvalOf(i + 7717);
    CUDACHECK(
        cudaMemcpy(Wdn, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice));
  }

  // Each gatedGroupedKernel<Op> is its own template instantiation and needs its
  // own >48 KB opt-in. Missing one makes the launch fail silently and the row
  // hang, which is a genuinely unpleasant afternoon.
  int const guSmem = int(sizeof(typename CtGuFused::SharedStorage));
  int const dnSmem = int(sizeof(typename CtDnPlain::SharedStorage));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtGuFused>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 guSmem));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtGuPlain>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 guSmem));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtDnPlain>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dnSmem));

  int nSM = 0;
  CUDACHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0));
  float const alpha = 1.0f, beta = 0.0f;

  // Descriptors for cublasGemmGroupedBatchedEx. Grouped, not a loop over
  // experts: without it the baseline is artificially weak exactly where fusion
  // claims its win, since at decode M_e drops to single digits and 16
  // sequential launches each underfill the GPU.
  cublasOperation_t bTransA[kELoc], bTransB[kELoc];
  int b1M[kELoc], b1N[kELoc], b1K[kELoc], b1LdA[kELoc], b1LdB[kELoc],
      b1LdC[kELoc];
  int b2M[kELoc], b2N[kELoc], b2K[kELoc], b2LdA[kELoc], b2LdB[kELoc],
      b2LdC[kELoc];
  int bGroupSize[kELoc];
  float bAlpha[kELoc], bBeta[kELoc];
  int b1Groups = 0, b2Groups = 0;
  __nv_bfloat16 const **d1A, **d1B, **d2A, **d2B;
  __nv_bfloat16 **d1C, **d2C;
  CUDACHECK(cudaMalloc(&d1A, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&d1B, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&d1C, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&d2A, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&d2B, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&d2C, kELoc * sizeof(void*)));

  ProbSet psGu;          // GEMM1, split by gate group
  ProbSet psDn[NRANKS];  // GEMM2, one list per combine chunk
  psGu.alloc(kGateGroups * kELoc);
  for (int q = 0; q < NRANKS; q++) psDn[q].alloc(kELoc);

  // The fused row runs through MegaMoe rather than through this file's own
  // launches, so the thing measured here and the thing that runs inside vLLM
  // are the same code and cannot drift. The other ladder rows keep their local
  // lambdas: they are baselines, not the shipped path.
  MegaMoe mm;
  mm.rank = rank;
  mm.buf = buf;
  mm.pc = hPc;
  mm.h = h;
  mm.a = a;
  mm.y = y;
  mm.gateSplit = cfg.gateSplit;
  mm.chunks = cfg.chunks;
  mm.reserveSM = cfg.reserveSM;
  mm.dProg = dProg;
  mm.allocOnce(CAP, maxRows, nSM);
  mm.dSendOff = dSendOff;
  mm.dRecvOff = dRecvOff;
  mm.dSendBytes = dSendBytes;
  mm.dCSendOff = dCSendOff;
  mm.dCRecvOff = dCRecvOff;
  mm.dCSendBytes = dCSendBytes;
  PlanView pv;

  if (rank == 0) {
    std::printf(
        "# MoE layer: dispatch + GEMM + SwiGLU + GEMM + combine, %d x L4, PCIe "
        "only, no NVLink\n",
        NRANKS);
    std::printf(
        "# DeepSeek-V2-Lite EP=4: H=%d I=%d (N=%d), %d experts, %d local, "
        "top-%d, bf16\n",
        kH, kI, kGU, kE, kELoc, kTopK);
    std::printf(
        "# CUTLASS tile %dx%dx32, 4 stages; %d SM; gate split -P %d; combine "
        "chunks -Q %d\n",
        FM_BM, FM_BN, nSM, cfg.gateSplit, cfg.chunks);
    std::printf(
        "# time is the worst rank; the whole layer, both collectives\n");
    std::printf(
        "# relerr is rms(delta)/rms(reference) against the fp32 reference "
        "layer, not pointwise\n");
    std::printf(
        "# %d rounds x %d iters, variants interleaved, min per variant, after "
        "%.0f ms settling:\n",
        cfg.rounds, cfg.iters, cfg.settleMs);
    std::printf(
        "# this L4 power-throttles from 2040 to 885 MHz under sustained load "
        "(72 W cap), so\n");
    std::printf(
        "# timing the rows one after another does not give comparable "
        "numbers\n#\n");
    std::printf("# %7s %8s %8s %14s %11s %10s %10s %9s %9s\n", "T/rank", "rows",
                "commMiB", "variant", "time(us)", "TFLOP/s", "GB/s", "badrow",
                "relerr");
  }

  Plan p;
  for (int T = cfg.minT; T <= cfg.maxT; T *= cfg.factorT) {
    buildPlan(p, rank, T, (int)CAP);
    // Useful FLOPs: both GEMMs over the rows that actually arrived. Not
    // T x top-k, which over-counts because the dispatch de-duplicates, and not
    // the padded tile count, which would flatter whichever variant splits more.
    double const flops =
        2.0 * (double)p.totalRows * ((double)kH * kGU + (double)kI * kH);

    size_t hSendOff[NRANKS], hRecvOff[NRANKS], hSendBytes[NRANKS];
    size_t hCSendOff[NRANKS], hCRecvOff[NRANKS], hCSendBytes[NRANKS];
    size_t dispEgress = 0, combEgress = 0;
    for (int d = 0; d < NRANKS; d++) {
      hSendOff[d] = dSend + (size_t)d * slot;
      hRecvOff[d] = dRecv + (size_t)d * slot;
      hSendBytes[d] = (size_t)p.nSend[d] * kH * 2;
      hCSendOff[d] = cSend + (size_t)d * slot;
      hCRecvOff[d] = cRecv + (size_t)d * slot;
      hCSendBytes[d] = (size_t)p.nRecv[d] * kH * 2;
      if (d != rank) {
        dispEgress += hSendBytes[d];
        combEgress += hCSendBytes[d];
      }
    }
    const size_t egress = dispEgress + combEgress;
    // Bytes each row actually moves, so the GB/s column means something on the
    // compute-only and dispatch-only rows too.
    auto egressOf = [&](int v) -> size_t {
      if (v == V_GEMM_ONLY || v == V_CUTLASS_ONLY) return 0;
      if (v == V_DISP_ONLY) return dispEgress;
      return egress;
    };
    CUDACHECK(cudaMemcpy(dSendOff, hSendOff, sizeof(hSendOff),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dRecvOff, hRecvOff, sizeof(hRecvOff),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dSendBytes, hSendBytes, sizeof(hSendBytes),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCSendOff, hCSendOff, sizeof(hCSendOff),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCRecvOff, hCRecvOff, sizeof(hCRecvOff),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCSendBytes, hCSendBytes, sizeof(hCSendBytes),
                         cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(dRowIdx, p.rowIdx.data(),
                         p.rowIdx.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dRedOff, p.redOff.data(),
                         p.redOff.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
    if (!p.redRow.empty()) {
      CUDACHECK(cudaMemcpy(dRedRow, p.redRow.data(),
                           p.redRow.size() * sizeof(int),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(dRedW, p.redW.data(), p.redW.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    }
    for (int d = 0; d < NRANKS; d++)
      if (p.nSend[d])
        CUDACHECK(cudaMemcpy(dSendTok + (size_t)d * CAP, p.sendTok[d].data(),
                             p.nSend[d] * sizeof(int), cudaMemcpyHostToDevice));

    // ---- cuBLAS descriptors -------------------------------------------------
    // Column-major convention: C[kGU, m] = W^T[kGU, kH] * A[kH, m].
    {
      __nv_bfloat16 const* pa[kELoc];
      __nv_bfloat16 const* pb[kELoc];
      __nv_bfloat16* pc[kELoc];
      b1Groups = 0;
      for (int e = 0; e < kELoc; e++) {
        int m = 0;
        for (int g = 0; g < NRANKS; g++) m += p.grpCnt[e][g];
        if (!m) continue;
        int const off = p.expOff[e], i = b1Groups++;
        bTransA[i] = CUBLAS_OP_T;
        bTransB[i] = CUBLAS_OP_N;
        b1M[i] = kGU;
        b1N[i] = m;
        b1K[i] = kH;
        b1LdA[i] = kH;
        b1LdB[i] = kH;
        b1LdC[i] = kGU;
        bGroupSize[i] = 1;
        bAlpha[i] = 1.0f;
        bBeta[i] = 0.0f;
        pa[i] = Wgu + (size_t)e * kGU * kH;
        pb[i] = Ap + (size_t)off * kH;
        pc[i] = h + (size_t)off * kGU;
      }
      if (b1Groups) {
        CUDACHECK(cudaMemcpy(d1A, pa, b1Groups * sizeof(void*),
                             cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d1B, pb, b1Groups * sizeof(void*),
                             cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d1C, pc, b1Groups * sizeof(void*),
                             cudaMemcpyHostToDevice));
      }
      b2Groups = 0;
      for (int e = 0; e < kELoc; e++) {
        int m = 0;
        for (int g = 0; g < NRANKS; g++) m += p.grpCnt[e][g];
        if (!m) continue;
        int const off = p.expOff[e], i = b2Groups++;
        b2M[i] = kH;
        b2N[i] = m;
        b2K[i] = kI;
        b2LdA[i] = kI;
        b2LdB[i] = kI;
        b2LdC[i] = kH;
        pa[i] = Wdn + (size_t)e * kH * kI;
        pb[i] = a + (size_t)off * kI;
        pc[i] = y + (size_t)off * kH;
      }
      if (b2Groups) {
        CUDACHECK(cudaMemcpy(d2A, pa, b2Groups * sizeof(void*),
                             cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d2B, pb, b2Groups * sizeof(void*),
                             cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d2C, pc, b2Groups * sizeof(void*),
                             cudaMemcpyHostToDevice));
      }
    }

    // ---- GEMM1: one problem per (gate group, expert), group-major -----------
    // Merging the four arrival groups into two is not a simplification, it is
    // the result: at four groups the group-major order costs 2.6x in lost
    // weight reuse, at two it costs 4%.
    int const gateLo[kGateGroups + 1] = {0, cfg.gateSplit, NRANKS};
    psGu.clear();
    for (int gg = 0; gg < kGateGroups; gg++)
      for (int e = 0; e < kELoc; e++) {
        int mr = 0;
        for (int g = gateLo[gg]; g < gateLo[gg + 1]; g++) mr += p.grpCnt[e][g];
        if (mr <= 0) continue;
        int const off = p.expOff[e] + p.grpOff[e][gateLo[gg]];
        psGu.add(
            mr, kGU, kH,
            reinterpret_cast<CtElem*>(
                buf + dRecv),  // tensor base; rows come from the index list
            reinterpret_cast<CtElem*>(Wgu) + (size_t)e * kGU * kH,
            reinterpret_cast<CtElem*>(h) + (size_t)off * kGU, kH, kH, kGU,
            dRowIdx + off,
            gateLo[gg + 1] - 1);  // the last arrival group this problem needs
      }
    psGu.upload();
    // Two grids for the same GEMM. The fused row holds an SM back so block 0's
    // dispatcher is guaranteed to be scheduled; the unfused rows have nothing
    // to reserve for and get the whole GPU. Reserving is a *cost of fusing* and
    // belongs on the fused row, not on the rung it is measured against.
    int const guGrid = psGu.grid(nSM, cfg.reserveSM);
    int const guGridFull = psGu.grid(nSM, 0);
    auto guArgs = psGu.args<CtGatherKernel>(guGrid);
    auto guArgsFull = psGu.args<CtGatherKernel>(guGridFull);
    CUTLASSCHECK(CtDev::can_implement(
        *reinterpret_cast<typename CtKernel::Arguments*>(&guArgs)));
    MoeDispatch disp{hPc,        rank,     dSendOff, dRecvOff,
                     dSendBytes, dArrived, dProg};
    typename CtGuFused::Params guFusedParams(
        typename CtGatherKernel::Params(guArgs, nullptr, 0), dArrived,
        psGu.dGroupOf, psGu.dIdx, disp);
    // Same problems and pointers, arrived = nullptr, which compiles the gate
    // out entirely. This is what separates "our GEMM is faster" from "the
    // communication is hidden".
    typename CtGuPlain::Params guPlainParams(
        typename CtGatherKernel::Params(guArgsFull, nullptr, 0), nullptr,
        nullptr, psGu.dIdx);

    // ---- GEMM2: one problem list per combine chunk --------------------------
    // Chunk q covers arrival groups [chunkLo[q], chunkLo[q+1]). Rows are
    // group-minor within an expert, so that span is contiguous and GEMM2 needs
    // neither a gather nor a permute. The uneven boundary for Q=3 is
    // deliberate: it puts the two cheap groups last, where the remaining
    // compute is smallest.
    int const Q = cfg.chunks;
    int chunkLo[NRANKS + 1];
    for (int q = 0; q <= Q; q++) chunkLo[q] = (q * NRANKS + Q - 1) / Q;
    for (int q = 0; q < Q; q++) {
      psDn[q].clear();
      for (int e = 0; e < kELoc; e++) {
        int mr = 0;
        for (int g = chunkLo[q]; g < chunkLo[q + 1]; g++) mr += p.grpCnt[e][g];
        if (mr <= 0) continue;
        int const off = p.expOff[e] + p.grpOff[e][chunkLo[q]];
        psDn[q].add(mr, kH, kI, reinterpret_cast<CtElem*>(a) + (size_t)off * kI,
                    reinterpret_cast<CtElem*>(Wdn) + (size_t)e * kH * kI,
                    reinterpret_cast<CtElem*>(y) + (size_t)off * kH, kI, kI, kH,
                    nullptr, 0);
      }
      psDn[q].upload();
    }
    int dnGrid[NRANKS];
    std::vector<typename CtDnPlain::Params> dnParams;
    dnParams.reserve(Q);
    for (int q = 0; q < Q; q++) {
      dnGrid[q] = psDn[q].grid(nSM, 0);
      auto dnArgs = psDn[q].args<CtKernel>(dnGrid[q]);
      dnParams.emplace_back(typename CtKernel::Params(dnArgs, nullptr, 0),
                            nullptr, nullptr, nullptr);
    }

    // Hand the same plan to MegaMoe. It rebuilds its own problem lists rather
    // than borrowing the ones above -- outside the timed region, and it keeps
    // the object self-contained for the extension, which has no ladder.
    pv.T = T;
    pv.cap = (int)CAP;
    pv.totalRows = p.totalRows;
    for (int d = 0; d < NRANKS; d++) {
      pv.nSend[d] = p.nSend[d];
      pv.nRecv[d] = p.nRecv[d];
    }
    for (int e = 0; e < kELoc; e++) {
      pv.expOff[e] = p.expOff[e];
      for (int g = 0; g < NRANKS; g++) {
        pv.grpOff[e][g] = p.grpOff[e][g];
        pv.grpCnt[e][g] = p.grpCnt[e][g];
      }
    }
    pv.dRowIdx = dRowIdx;
    pv.dSendTok = dSendTok;
    pv.dRedOff = dRedOff;
    pv.dRedRow = dRedRow;
    pv.dRedW = dRedW;
    mm.configure(pv, x, Wgu, Wdn, out);

    // ---- the pieces every row is assembled from -----------------------------
    auto stageAll = [&](cudaStream_t s) {
      for (int d = 0; d < NRANKS; d++) {
        if (!p.nSend[d]) continue;
        // Own rows go straight into the receive slot, so the GEMM sees one
        // uniform [4][CAP][H] view whether a row crossed PCIe or not.
        __nv_bfloat16* dst =
            (d == rank) ? reinterpret_cast<__nv_bfloat16*>(buf + hRecvOff[rank])
                        : reinterpret_cast<__nv_bfloat16*>(buf + hSendOff[d]);
        stageRows<<<p.nSend[d], 256, 0, s>>>(dst, x, dSendTok + (size_t)d * CAP,
                                             p.nSend[d]);
      }
    };
    auto ncclA2A = [&](size_t sOff, size_t rOff, int const* cs, int const* cr,
                       cudaStream_t s) {
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < NPEERS; i++) {
        const int d = ascPeer(rank, i);
        if (cs[d])
          NCCLCHECK(ncclSend(buf + sOff + (size_t)d * slot, (size_t)cs[d] * kH,
                             ncclBfloat16, d, ncclComm, s));
        if (cr[d])
          NCCLCHECK(ncclRecv(buf + rOff + (size_t)d * slot, (size_t)cr[d] * kH,
                             ncclBfloat16, d, ncclComm, s));
      }
      NCCLCHECK(ncclGroupEnd());
    };
    auto permuteAll = [&](cudaStream_t s) {
      for (int e = 0; e < kELoc; e++) {
        int m = 0;
        for (int g = 0; g < NRANKS; g++) m += p.grpCnt[e][g];
        if (!m) continue;
        const int off = p.expOff[e];
        permuteRows<<<m, 256, 0, s>>>(
            Ap + (size_t)off * kH,
            reinterpret_cast<__nv_bfloat16*>(buf + dRecv), dRowIdx + off, m);
      }
    };
    // SwiGLU over one chunk: per (chunk, expert), because rows are expert-major
    // and only contiguous within one expert.
    auto swigluChunk = [&](int q, cudaStream_t s) {
      for (int e = 0; e < kELoc; e++) {
        int mr = 0;
        for (int g = chunkLo[q]; g < chunkLo[q + 1]; g++) mr += p.grpCnt[e][g];
        if (!mr) continue;
        const int off = p.expOff[e] + p.grpOff[e][chunkLo[q]];
        swigluKernel<<<mr, 256, 0, s>>>(a + (size_t)off * kI,
                                        h + (size_t)off * kGU, mr, kI);
      }
    };
    auto reduceChunk = [&](int q, cudaStream_t s) {
      for (int g = chunkLo[q]; g < chunkLo[q + 1]; g++) {
        const int sr = srcOfGroup(rank, g);
        if (p.nRecv[sr])
          weightedReduce<<<p.nRecv[sr], 256, 0, s>>>(
              reinterpret_cast<__nv_bfloat16*>(buf + cSend + (size_t)sr * slot),
              y, dRedOff, dRedRow, dRedW, sr * (int)CAP, p.nRecv[sr]);
      }
    };
    // Rotation steps this chunk releases. Group 0 is this rank's own rows: its
    // reduce writes cSend[rank] and `finish` reads it locally, so there is no
    // step 0 to send.
    auto putChunk = [&](int q, cudaStream_t s) {
      const int kLo = std::max(1, chunkLo[q]), kHi = chunkLo[q + 1];
      if (kLo < kHi)
        putCombine<<<1, 32, 0, s>>>(hPc, rank, dCSendOff, dCRecvOff,
                                    dCSendBytes, kLo, kHi, dProg);
    };
    auto finish = [&](cudaStream_t s) {
      zeroRows<<<T, 256, 0, s>>>(out, T);
      for (int d = 0; d < NRANKS; d++) {
        if (!p.nSend[d]) continue;
        const __nv_bfloat16* src = (d == rank)
                                       ? reinterpret_cast<__nv_bfloat16*>(
                                             buf + cSend + (size_t)rank * slot)
                                       : reinterpret_cast<__nv_bfloat16*>(
                                             buf + cRecv + (size_t)d * slot);
        accumulatePeer<<<p.nSend[d], 256, 0, s>>>(
            out, src, dSendTok + (size_t)d * CAP, p.nSend[d]);
      }
    };

    // The compute half in cuBLAS: permute, grouped GEMM1, SwiGLU, grouped
    // GEMM2, weighted local reduce.
    auto cublasCompute = [&](cudaStream_t s) {
      permuteAll(s);
      if (b1Groups)
        CUBLASCHECK(cublasGemmGroupedBatchedEx(
            cub, bTransA, bTransB, b1M, b1N, b1K, bAlpha,
            (const void* const*)d1A, CUDA_R_16BF, b1LdA,
            (const void* const*)d1B, CUDA_R_16BF, b1LdB, bBeta,
            (void* const*)d1C, CUDA_R_16BF, b1LdC, b1Groups, bGroupSize,
            CUBLAS_COMPUTE_32F));
      for (int q = 0; q < Q; q++) swigluChunk(q, s);
      if (b2Groups)
        CUBLASCHECK(cublasGemmGroupedBatchedEx(
            cub, bTransA, bTransB, b2M, b2N, b2K, bAlpha,
            (const void* const*)d2A, CUDA_R_16BF, b2LdA,
            (const void* const*)d2B, CUDA_R_16BF, b2LdB, bBeta,
            (void* const*)d2C, CUDA_R_16BF, b2LdC, b2Groups, bGroupSize,
            CUBLAS_COMPUTE_32F));
      for (int q = 0; q < Q; q++) reduceChunk(q, s);
    };
    // The same, with CUTLASS. Same gather-A GEMM1 the fused row runs, minus the
    // gate and the dispatcher; same chunked GEMM2.
    auto cutlassCompute = [&](cudaStream_t s) {
      if (psGu.n())
        uccl_lite::gatedGroupedKernel<CtGuPlain>
            <<<guGridFull, CtGuPlain::kThreadCount, guSmem, s>>>(guPlainParams);
      for (int q = 0; q < Q; q++) {
        swigluChunk(q, s);
        if (psDn[q].n())
          uccl_lite::gatedGroupedKernel<CtDnPlain>
              <<<dnGrid[q], CtDnPlain::kThreadCount, dnSmem, s>>>(dnParams[q]);
        reduceChunk(q, s);
      }
    };

    auto runVariant = [&](int v) {
      switch (v) {
        case V_GEMM_ONLY:
          cublasCompute(sMain);
          break;
        case V_CUTLASS_ONLY:
          cutlassCompute(sMain);
          break;
        case V_DISP_ONLY:
          stageAll(sMain);
          dispatchCE<<<1, 32, 0, sMain>>>(hPc, rank, dSendOff, dRecvOff,
                                          dSendBytes, dProg);
          break;
        case V_COMM_ONLY:
          stageAll(sMain);
          dispatchCE<<<1, 32, 0, sMain>>>(hPc, rank, dSendOff, dRecvOff,
                                          dSendBytes, dProg);
          putCombine<<<1, 32, 0, sMain>>>(hPc, rank, dCSendOff, dCRecvOff,
                                          dCSendBytes, 1, NPEERS + 1, dProg);
          waitCombine<<<1, 32, 0, sMain>>>(hPc, rank, dProg);
          break;
        case V_NCCL_CUBLAS:
          stageAll(sMain);
          ncclA2A(dSend, dRecv, p.nSend, p.nRecv, sMain);
          cublasCompute(sMain);
          ncclA2A(cSend, cRecv, p.nRecv, p.nSend, sMain);
          finish(sMain);
          break;
        case V_OURS_CUBLAS:
          stageAll(sMain);
          dispatchCE<<<1, 32, 0, sMain>>>(hPc, rank, dSendOff, dRecvOff,
                                          dSendBytes, dProg);
          cublasCompute(sMain);
          putCombine<<<1, 32, 0, sMain>>>(hPc, rank, dCSendOff, dCRecvOff,
                                          dCSendBytes, 1, NPEERS + 1, dProg);
          waitCombine<<<1, 32, 0, sMain>>>(hPc, rank, dProg);
          finish(sMain);
          break;
        case V_OURS_CUTLASS:
          stageAll(sMain);
          dispatchCE<<<1, 32, 0, sMain>>>(hPc, rank, dSendOff, dRecvOff,
                                          dSendBytes, dProg);
          cutlassCompute(sMain);
          putCombine<<<1, 32, 0, sMain>>>(hPc, rank, dCSendOff, dCRecvOff,
                                          dCSendBytes, 1, NPEERS + 1, dProg);
          waitCombine<<<1, 32, 0, sMain>>>(hPc, rank, dProg);
          finish(sMain);
          break;
        // The fused row. One stream throughout: the dispatch is block 0 of
        // GEMM1's own grid (a block cannot be scheduled behind its own grid,
        // which is what made every separate-kernel version deadlock), and the
        // combine needs nothing but the put being issued a chunk early.
        case V_FUSED:
          mm.run(sMain);
          break;
        default:
          break;
      }
    };

    // FML_PHASE=1: where the compute half actually goes. Not a ladder row --
    // these are the same kernels the `cutlass-only` row runs, timed one phase
    // at a time with events, after a warm-up, so the phases sum to that row.
    if (getenv("FML_PHASE")) {
      cudaEvent_t e0, e1;
      CUDACHECK(cudaEventCreate(&e0));
      CUDACHECK(cudaEventCreate(&e1));
      auto tphase = [&](char const* name, auto&& fn) {
        for (int i = 0; i < 3; i++) fn();
        CUDACHECK(cudaStreamSynchronize(sMain));
        CUDACHECK(cudaEventRecord(e0, sMain));
        for (int i = 0; i < 5; i++) fn();
        CUDACHECK(cudaEventRecord(e1, sMain));
        CUDACHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDACHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (rank == 0)
          std::fprintf(stderr, "[phase] T=%d %-10s %8.0f us\n", T, name,
                       ms * 1000.0f / 5);
      };
      tphase("gemm1", [&] {
        if (psGu.n())
          uccl_lite::gatedGroupedKernel<CtGuPlain>
              <<<guGridFull, CtGuPlain::kThreadCount, guSmem, sMain>>>(
                  guPlainParams);
      });
      tphase("swiglu", [&] {
        for (int q = 0; q < Q; q++) swigluChunk(q, sMain);
      });
      tphase("gemm2", [&] {
        for (int q = 0; q < Q; q++)
          if (psDn[q].n())
            uccl_lite::gatedGroupedKernel<CtDnPlain>
                <<<dnGrid[q], CtDnPlain::kThreadCount, dnSmem, sMain>>>(
                    dnParams[q]);
      });
      tphase("reduce", [&] {
        for (int q = 0; q < Q; q++) reduceChunk(q, sMain);
      });
      tphase("finish", [&] { finish(sMain); });
      tphase("permute", [&] { permuteAll(sMain); });
      CUDACHECK(cudaEventDestroy(e0));
      CUDACHECK(cudaEventDestroy(e1));
    }

    // ---- the reference ------------------------------------------------------
    // The complete NCCL + cuBLAS layer, both collectives, with h kept in fp32
    // through SwiGLU. It has to be the whole layer: a token's output is a sum
    // over four ranks' partials, so a locally rebuilt reference is missing
    // three quarters of it.
    auto refCompute = [&](cudaStream_t s) {
      permuteAll(s);
      for (int e = 0; e < kELoc; e++) {
        int m = 0;
        for (int g = 0; g < NRANKS; g++) m += p.grpCnt[e][g];
        if (!m) continue;
        const int off = p.expOff[e];
        CUBLASCHECK(cublasGemmEx(
            cub, CUBLAS_OP_T, CUBLAS_OP_N, kGU, m, kH, &alpha,
            Wgu + (size_t)e * kGU * kH, CUDA_R_16BF, kH, Ap + (size_t)off * kH,
            CUDA_R_16BF, kH, &beta, h32, CUDA_R_32F, kGU, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        swigluKernel32<<<m, 256, 0, s>>>(a + (size_t)off * kI, h32, m, kI);
        CUBLASCHECK(cublasGemmEx(
            cub, CUBLAS_OP_T, CUBLAS_OP_N, kH, m, kI, &alpha,
            Wdn + (size_t)e * kH * kI, CUDA_R_16BF, kI, a + (size_t)off * kI,
            CUDA_R_16BF, kI, &beta, y + (size_t)off * kH, CUDA_R_16BF, kH,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      for (int q = 0; q < Q; q++) reduceChunk(q, s);
    };

    if (cfg.check) {
      CUDACHECK(cudaMemsetAsync(buf + dRecv, 0, NRANKS * slot, sMain));
      CUDACHECK(cudaMemsetAsync(buf + cRecv, 0, NRANKS * slot, sMain));
      CUDACHECK(cudaStreamSynchronize(sMain));
      bootstrap->barrier();
      stageAll(sMain);
      ncclA2A(dSend, dRecv, p.nSend, p.nRecv, sMain);
      refCompute(sMain);
      ncclA2A(cSend, cRecv, p.nRecv, p.nSend, sMain);
      finish(sMain);
      CUDACHECK(cudaStreamSynchronize(sMain));
      CUDACHECK(cudaMemcpy(outRef, out, (size_t)T * kH * 2,
                           cudaMemcpyDeviceToDevice));
      bootstrap->barrier();
    }

    // Correctness first for every variant, then timing for every variant. The
    // two passes are separate because the timing pass interleaves the variants
    // and a check in the middle would perturb it.
    std::vector<double> hostOut;
    if (cfg.check > 1 && T <= 64) hostReference(rank, T, hostOut);
    std::vector<long long> badRowOf(NVAR, -1);
    std::vector<double> relErrOf(NVAR, -1.0);
    for (int v = 0; v < NVAR; v++) {
      if (cfg.only >= 0 && v != cfg.only) continue;
      if (cfg.skip[v]) continue;
      bool const full =
          (v >= V_NCCL_CUBLAS);  // rows that produce a real layer output
      if (!cfg.check || !full) continue;

      CUDACHECK(cudaMemsetAsync(buf + dRecv, 0, NRANKS * slot, sMain));
      CUDACHECK(cudaMemsetAsync(buf + cRecv, 0, NRANKS * slot, sMain));
      CUDACHECK(cudaStreamSynchronize(sMain));
      bootstrap->barrier();
      TRACE("check %s: run", kName[v]);
      runVariant(v);
      CUDACHECK(cudaStreamSynchronize(sMain));
      TRACE("check %s: synced", kName[v]);
      bootstrap->barrier();

      // Movement, exactly: every dispatched row must be bit-identical to the
      // source rank's token, in the slot the plan says, from the sender the
      // plan says -- so a row in the wrong place is caught even though the
      // arithmetic below can only be checked to a tolerance. Strided, because a
      // full compare is 100x the runtime.
      std::vector<__nv_bfloat16> hr((size_t)NRANKS * CAP * kH);
      CUDACHECK(cudaMemcpy(hr.data(), buf + dRecv, hr.size() * 2,
                           cudaMemcpyDeviceToHost));
      long long badRow = 0;
      for (int s = 0; s < NRANKS; s++)
        for (int j = 0; j < p.nRecv[s]; j++) {
          int const t = p.recvTok[s][j];
          for (int i = 0; i < kH; i += 97)
            if (__bfloat162float(hr[((size_t)s * CAP + j) * kH + i]) !=
                xvalf(s, t, i)) {
              badRow++;
              break;
            }
        }

      std::vector<__nv_bfloat16> ha((size_t)T * kH), hb((size_t)T * kH);
      CUDACHECK(
          cudaMemcpy(ha.data(), out, ha.size() * 2, cudaMemcpyDeviceToHost));
      CUDACHECK(
          cudaMemcpy(hb.data(), outRef, hb.size() * 2, cudaMemcpyDeviceToHost));
      // Error relative to the *scale* of the output, not pointwise. A layer
      // output is a weighted sum over four ranks' partials and it cancels hard,
      // so plenty of elements land near zero; a pointwise ratio there reports a
      // factor of three for one bf16 ulp. rms(delta)/rms(ref) does not have
      // that failure mode and still catches structure: one misrouted row out of
      // 12145 moves this to 3%, an order of magnitude above the bf16 floor.
      double sum2 = 0.0, err2 = 0.0;
      for (size_t i = 0; i < ha.size(); i++) {
        double const gv = __bfloat162float(ha[i]), rv = __bfloat162float(hb[i]);
        sum2 += rv * rv;
        err2 += (gv - rv) * (gv - rv);
      }
      double relErr = std::sqrt(err2 / (sum2 + 1e-30));

      // -c 2: additionally check against the fp64 host reference, which shares
      // no index structure with any of this.
      if (!hostOut.empty()) {
        double s2 = 0.0, e2 = 0.0;
        for (size_t i = 0; i < ha.size(); i++) {
          double const gv = __bfloat162float(ha[i]);
          s2 += hostOut[i] * hostOut[i];
          e2 += (gv - hostOut[i]) * (gv - hostOut[i]);
        }
        double const hostErr = std::sqrt(e2 / (s2 + 1e-30));
        if (rank == 0)
          std::fprintf(stderr, "[host fp64] T=%d %-14s rms-rel err %.3e\n", T,
                       kName[v], hostErr);
        relErr = std::max(relErr, hostErr);
      }
      badRowOf[v] = badRow;
      relErrOf[v] = relErr;
      bootstrap->barrier();
    }

    // ---- timing -------------------------------------------------------------
    // Settle, then interleave the variants round-robin and keep each one's
    // minimum. Timing each variant to completion in turn is not sound on this
    // machine: an L4 is a 72 W card, sustained bf16 work saturates that budget,
    // and the SM clock swings between 2040 and 885 MHz during a run, so
    // whichever variant is measured first gets a colder GPU.
    std::vector<double> bestUs(NVAR, 1e30);
    {
      auto once = [&](int v) -> double {
        bootstrap->barrier();
        TRACE("time %s", kName[v]);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < cfg.iters; i++) runVariant(v);
        CUDACHECK(cudaStreamSynchronize(sMain));
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() /
               cfg.iters;
      };
      // The barrier inside the settle loop is not decoration: without it the
      // ranks drift, and for a cheap row 2500 ms of settling is tens of
      // thousands of un-barriered collectives, which deadlocks reproducibly.
      // The exit from this loop has to be *unanimous*, and getting that wrong
      // is what produced every unexplained deadlock in this file's history.
      // Each rank used to read its own clock and break on it; when one left
      // after N passes and another after N+1, the extra pass issued a round of
      // puts and waits with no partner and the collective counts were
      // permanently off. It is timing-dependent, which is why it looked like a
      // bug in whatever was being changed at the time -- a single cheap row
      // selected with -a, or one particular -Q. So: every rank votes, and
      // nobody leaves until all four have.
      constexpr int kSettleMaxPasses = 300;
      auto s0 = std::chrono::steady_clock::now();
      for (int pass = 0; pass < kSettleMaxPasses; pass++) {
        TRACE("settle pass %d", pass);
        for (int i = 0; i < cfg.warmup; i++)
          for (int v = 0; v < NVAR; v++) {
            if (cfg.only >= 0 && v != cfg.only) continue;
            if (cfg.skip[v]) continue;
            runVariant(v);
          }
        CUDACHECK(cudaStreamSynchronize(sMain));
        std::vector<int> vote(NRANKS, 0);
        vote[rank] = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - s0)
                                 .count() >= cfg.settleMs
                         ? 1
                         : 0;
        bootstrap->allGather(vote.data(), sizeof(int));
        if (*std::min_element(vote.begin(), vote.end()) == 1) break;
      }
      bootstrap->barrier();
      for (int r = 0; r < cfg.rounds; r++)
        for (int v = 0; v < NVAR; v++) {
          if (cfg.only >= 0 && v != cfg.only) continue;
          if (cfg.skip[v]) continue;
          double const us = once(v);
          std::vector<double> all(NRANKS, 0.0);
          all[rank] = us;
          bootstrap->allGather(all.data(), sizeof(double));
          double const worst = *std::max_element(all.begin(), all.end());
          if (rank == 0) {
            std::printf("# sample,%d,%s,%d,%.3f\n", T, kName[v], r + 1,
                        worst);
            std::fflush(stdout);
          }
          if (worst < bestUs[v]) bestUs[v] = worst;
        }
    }

    for (int v = 0; v < NVAR; v++) {
      if (cfg.only >= 0 && v != cfg.only) continue;
      if (cfg.skip[v]) continue;

      // The step is only as fast as its slowest participant.
      std::vector<double> all(NRANKS, 0.0);
      all[rank] = bestUs[v];
      bootstrap->allGather(all.data(), sizeof(double));
      double const worst = *std::max_element(all.begin(), all.end());
      std::vector<long long> bad(NRANKS, 0);
      bad[rank] = badRowOf[v];
      bootstrap->allGather(bad.data(), sizeof(long long));
      long long const worstBad = *std::max_element(bad.begin(), bad.end());
      std::vector<double> errs(NRANKS, 0.0);
      errs[rank] = relErrOf[v];
      bootstrap->allGather(errs.data(), sizeof(double));
      double const worstErr = *std::max_element(errs.begin(), errs.end());

      if (rank == 0) {
        double const eg = (double)egressOf(v);
        std::printf("  %7d %8d %8.2f %14s %11.2f %10.2f %10.2f %9lld %9.2e\n",
                    T, p.totalRows, eg / (1 << 20), kName[v], worst,
                    flops / (worst * 1e-6) / 1e12,
                    eg * NRANKS / (worst * 1e-6) / 1e9, worstBad, worstErr);
        std::fflush(stdout);
      }
      bootstrap->barrier();
    }
  }

  CUDACHECK(cudaStreamSynchronize(sMain));
  bootstrap->barrier();
  ncclCommDestroy(ncclComm);
  proxy->stopProxy();
  cublasDestroy(cub);
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
      cfg.minT = std::atoi(next());
    else if (k == "-e")
      cfg.maxT = std::atoi(next());
    else if (k == "-f")
      cfg.factorT = std::atoi(next());
    else if (k == "-n")
      cfg.iters = std::atoi(next());
    else if (k == "-w")
      cfg.warmup = std::atoi(next());
    else if (k == "-r")
      cfg.rounds = std::atoi(next());
    else if (k == "-S")
      cfg.settleMs = std::atof(next());
    else if (k == "-R")
      cfg.reserveSM = std::atoi(next());
    else if (k == "-P")
      cfg.gateSplit = std::atoi(next());
    else if (k == "-Q")
      cfg.chunks = std::atoi(next());
    else if (k == "-c")
      cfg.check = std::atoi(next());
    else if (k == "-X") {
      const std::string want = next();
      for (int j = 0; j < NVAR; j++)
        if (want == kName[j]) cfg.skip[j] = true;
    } else if (k == "-a") {
      const std::string want = next();
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
  if (cfg.chunks < 1 || cfg.chunks > NRANKS) {
    std::fprintf(stderr, "-Q must be 1..%d\n", NRANKS);
    return 1;
  }
  if (cfg.gateSplit < 1 || cfg.gateSplit >= NRANKS) {
    std::fprintf(stderr, "-P must be 1..%d\n", NRANKS - 1);
    return 1;
  }

  // Fork before touching CUDA so every rank gets a clean context; the unique id
  // is made in the parent and inherited, which removes the need for MPI.
  //
  // FML_SPAWN=1 replaces that with fork+exec, so each rank is a fresh process
  // image rather than a copy of this one -- which is what torchrun and vLLM
  // do. It exists for exactly one experiment: the fused layer deadlocks under
  // the torch extension and not here, and the only differences left are the
  // process model and torch's allocator. This isolates the first.
  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();

  // Child of a spawn: rank and id come in through the environment.
  if (char const* rk = getenv("FML_RANK")) {
    char const* uid = getenv("FML_UID");
    if (!uid || std::strlen(uid) != id.size() * 2) {
      std::fprintf(stderr, "FML_RANK set but FML_UID missing or malformed\n");
      return 1;
    }
    for (size_t i = 0; i < id.size(); i++) {
      auto hex = [](char c) {
        return c <= '9' ? c - '0' : (c | 32) - 'a' + 10;
      };
      id.data()[i] = (char)((hex(uid[2 * i]) << 4) | hex(uid[2 * i + 1]));
    }
    runRank(atoi(rk), id, cfg);
    return 0;
  }

  if (getenv("FML_SPAWN")) {
    std::string uid(id.size() * 2, '0');
    for (size_t i = 0; i < id.size(); i++)
      std::snprintf(&uid[2 * i], 3, "%02x", (unsigned char)id.data()[i]);
    std::vector<pid_t> spawned;
    for (int r = 0; r < NRANKS; r++) {
      pid_t pid = fork();
      if (pid == 0) {
        setenv("FML_RANK", std::to_string(r).c_str(), 1);
        setenv("FML_UID", uid.c_str(), 1);
        execv(argv[0], argv);
        _exit(127);  // only reached if exec failed
      }
      spawned.push_back(pid);
    }
    int rc = 0;
    for (pid_t pid : spawned) {
      int st = 0;
      waitpid(pid, &st, 0);
      if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1;
    }
    return rc;
  }

  std::vector<pid_t> kids;
  for (int r = 1; r < NRANKS; r++) {
    pid_t pid = fork();
    if (pid == 0) {
      runRank(r, id, cfg);
      _exit(0);
    }
    kids.push_back(pid);
  }
  runRank(0, id, cfg);
  int rc = 0;
  for (pid_t pid : kids) {
    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1;
  }
  return rc;
}

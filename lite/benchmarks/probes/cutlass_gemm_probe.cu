// CUTLASS grouped GEMM on the MoE shapes, against cublasGemmGroupedBatchedEx.
//
// This is the go/no-go gate for using CUTLASS as the fused kernel's compute
// engine. Four attempts to close the gap by hand are recorded in
// mma_gemm_probe.cu and benchmarks/data/mma_gemm_occupancy.txt; all failed, the
// last one (warp specialisation) because mbarrier.try_wait is sm_90 only. So
// the question is not "can we write a faster GEMM" but "does an already-tuned
// one clear the bar", and then whether it still clears it once the problem list
// is cut the way arrival-gating needs.
//
// Three arms, one binary, identical data:
//
//   E16  one problem per local expert, M_e rows each. The drop-in comparison.
//   Eg   one problem per (arrival group, expert), ordered group-major. This is
//        what gating requires -- a tile can only wait on a group if the group
//        is a property of its problem -- and group-major is the order that
//        reaches all of group 0, this rank's own rows, before anything that
//        waits.
//   Ee   the same problems ordered expert-major. Same tiles, same arithmetic,
//        only the visitor's order differs. Eg vs Ee is therefore a clean read
//        on weight reuse in L2 and nothing else.
//
// What the sweep found, none of which was the expected answer:
//
//   1. Ordering dominates, not tile fill. At 4 groups Eg is 0.25x and Ee 0.65x
//      at decode -- byte-identical work, 2.6x apart. At decode this GEMM is
//      weight-bandwidth-bound (184 MB of weights, almost no math), and
//      group-major puts concurrent blocks on different experts' 11.5 MB weight
//      matrices, so it pays 4x the HBM traffic.
//   2. Two arrival groups instead of four fixes it. At G=2 a window of blocks
//      spans few enough experts to stay in the 48 MB L2, and group-major costs
//      ~4% instead of 2.6x. Coarser gating, but gating that works.
//   3. Tile fill is the second-order term, and it is real: M_e/G must be a
//   whole
//      number of BM. At G=2, M_e/2 = 384 is exactly 3 tiles of BM=128.
//
// Best shape found: 128x256x32, 4 stages, G=2 -- 1.24x grouped cuBLAS at
// T=2048 with E16 and 1.18x with the group-major split that gating needs.
// Below T=1024 the split costs 2x and gating is not viable; there is also
// almost no communication to hide there, so that is the honest crossover.
//
// Timing protocol matters as much as the kernel here; see the timing block in
// main(). This L4 power-throttles from 2040 to 885 MHz under sustained load, so
// arms are interleaved and settled rather than timed one after another.
//
// Layout matches mma_gemm_probe.cu so the numbers are comparable: weights are
// [E][N][H] "n-major", read here as a column-major B with ld=K and by cuBLAS as
// OP_T with ld=K. One copy of the weights, no repacking. A is row-major [M][K],
// C row-major [M][N].
//
// The A operand is permuted, not gathered. CUTLASS has a native gather but we
// measured that at 21% against the 4.3% a separate permute kernel costs, so the
// permute stays separate and is priced as its own column. In the fused kernel
// it disappears entirely: the receive buffer is already (group,
// expert)-blocked, which is exactly the layout Eg wants.
//
//   nvcc -O3 -std=c++17 -arch=sm_89 -I$CUTLASS/include -o cutlass_gemm_probe \
//        cutlass_gemm_probe.cu -lcublas

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/numeric_types.h"
#include "cutlass/version.h"
#include "cutlass_gated_grouped.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                               \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

#define CUBLASCHECK(cmd)                                                  \
  do {                                                                    \
    cublasStatus_t s = (cmd);                                             \
    if (s != CUBLAS_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "%s:%d cuBLAS error %d\n", __FILE__, __LINE__, \
                   (int)s);                                               \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#define CUTLASSCHECK(cmd)                                             \
  do {                                                                \
    cutlass::Status s = (cmd);                                        \
    if (s != cutlass::Status::kSuccess) {                             \
      std::fprintf(stderr, "%s:%d CUTLASS error %d (%s)\n", __FILE__, \
                   __LINE__, (int)s, cutlassGetStatusString(s));      \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

constexpr int kH = 2048;   // hidden, the GEMM's K
constexpr int kN = 2816;   // moe_intermediate 1408, gate+up fused
constexpr int kELoc = 16;  // 64 routed experts at EP=4
constexpr int kTopK = 6;
// Arrival groups. 4 is one per rank, the finest gating the ce-shift rotation
// can express, and it is not the right choice: at G=4 the group-major ordering
// that gating needs costs 2.6x at decode and 1.35x at prefill, because a window
// of concurrent blocks then spans four different experts' weight matrices. G=2
// gives up half the gating resolution and gets the L2 reuse back.
#ifndef CT_GROUPS
#define CT_GROUPS 2
#endif
constexpr int kGroups = CT_GROUPS;
constexpr int kE = 64;  // routed experts across all 4 ranks
constexpr int kRanks = 4;

// The same routing hash fused_moe.cu uses, so the row counts this probe sees
// are the ones the 4-rank harness sees. It matters more than it looks: with an
// even M_e per expert CUTLASS reads 1.21-1.24x grouped cuBLAS, and on the
// harness's ragged counts the same comparison came out 1.008x. -g 1 switches
// this on.
static void routeOf(int s, int t, int* experts) {
  uint32_t h =
      (uint32_t)(s * 2654435761u) ^ (uint32_t)(t * 40503u) ^ 0x9e3779b9u;
  int n = 0;
  while (n < kTopK) {
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    int const e = (int)(h % kE);
    bool dup = false;
    for (int i = 0; i < n; i++)
      if (experts[i] == e) dup = true;
    if (!dup) experts[n++] = e;
  }
}

// Rows landing on rank 0's local experts, per (source rank, local expert), for
// T tokens per rank. This reproduces the harness's receive-side shape exactly.
static void raggedCounts(int T, int cnt[kELoc][kRanks]) {
  for (int e = 0; e < kELoc; e++)
    for (int g = 0; g < kRanks; g++) cnt[e][g] = 0;
  int ex[kTopK];
  for (int s = 0; s < kRanks; s++)
    for (int t = 0; t < T; t++) {
      routeOf(s, t, ex);
      for (int i = 0; i < kTopK; i++)
        if (ex[i] / kELoc == 0) cnt[ex[i] % kELoc][s]++;
    }
}

// Tile shape, overridable from the build line so the sweep is a shell loop;
// cutlass_gemm_probe.sh drives it. The defaults are the sweep's winner, not
// CUTLASS's own sm_80 bf16 default of 128x128x32 -- the wider 128x256 tile is
// what lets group-major ordering keep its weight reuse.
#ifndef CT_BM
#define CT_BM 128
#endif
#ifndef CT_BN
#define CT_BN 256
#endif
#ifndef CT_BK
#define CT_BK 32
#endif
#ifndef CT_WM
#define CT_WM 64
#endif
#ifndef CT_WN
#define CT_WN 64
#endif
#ifndef CT_STAGES
#define CT_STAGES 4
#endif
// Who computes the tile -> (problem, tile) map. kDeviceOnly makes every
// next_tile() walk the problem list on the GPU; kHostPrecompute builds the
// whole schedule on the host and ships it. With 64 problems instead of 16 that
// walk is 4x longer, so it is a candidate for the E64 deficit and gets a knob.
#ifndef CT_HOST_SCHED
#define CT_HOST_SCHED 0
#endif
#if CT_HOST_SCHED
#define CT_SCHED cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute
#else
#define CT_SCHED cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
#endif

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAcc = float;

// A row-major [M][K], B column-major [K][N] (== the n-major weight buffer), C
// row-major [M][N]. Both operands are k-major, which is the layout the tensor
// core path wants and the one the weights are already stored in.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int kAlign = 8;  // 16 B; K=2048 and N=2816 are both multiples of 8

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, kAlign, ElementB,
    LayoutB, cutlass::ComplexTransform::kNone, kAlign, ElementC, LayoutC,
    ElementAcc, cutlass::arch::OpClassTensorOp,
    // Sm80, not Sm89: Ada's tensor cores are Ampere's for bf16 (no fp8 here),
    // and CUTLASS 2.x has no separate sm_89 grouped tuning. This is the tag the
    // sm_89 build is meant to use.
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<CT_BM, CT_BN, CT_BK>,
    cutlass::gemm::GemmShape<CT_WM, CT_WN, CT_BK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc,
        ElementAcc>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, CT_STAGES,
    CT_SCHED>::GemmKernel;

using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// The same kernel with the arrival gate. Here the flags are all pre-set, so
// this arm measures what the wrapper costs when nothing actually waits -- the
// spin, the extra __syncthreads and the group_of load -- separately from any
// overlap it might buy. That number belongs on its own: a fused result that
// quotes only the end-to-end win cannot tell a good gate from a lucky one.
using Gated = uccl_lite::GatedGrouped<GemmKernel>;

// The same grouped kernel built with gather-A, so the GEMM reads scattered rows
// through a row-index list instead of a permute kernel materialising them.
//
// DefaultGemmGrouped hardcodes GatherA=false, so this reassembles what it does:
// build the Mma and Epilogue with DefaultGemm (which does expose the flag),
// then hand them to the same kernel::GemmGrouped. kInternalTranspose is false
// here because LayoutC is RowMajor, so gather really does apply to A.
using GatherBase = typename cutlass::gemm::kernel::DefaultGemm<
    ElementA, LayoutA, kAlign, ElementB, LayoutB, kAlign, ElementC, LayoutC,
    ElementAcc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<CT_BM, CT_BN, CT_BK>,
    cutlass::gemm::GemmShape<CT_WM, CT_WN, CT_BK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc,
        ElementAcc>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, CT_STAGES,
    true,  // SplitKSerial, as DefaultGemmGrouped passes
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ElementA, ElementB,
        ElementC, ElementAcc>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    true,   // GatherA
    false,  // GatherB
    false   // ScatterD
    >::GemmKernel;
using GatherKernel = cutlass::gemm::kernel::GemmGrouped<
    typename GatherBase::Mma, typename GatherBase::Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, CT_SCHED,
    false>;
using GatedGather = uccl_lite::GatedGrouped<GatherKernel>;

// Gather rows of `src` named by rowIdx into a contiguous `dst`. This is what a
// non-fused path must run before a grouped GEMM, and what Stage 2's group-major
// ordering will produce. Priced as its own row rather than folded into the
// GEMM.
__global__ void permuteRows(__nv_bfloat16 const* __restrict__ src,
                            int const* __restrict__ rowIdx,
                            __nv_bfloat16* __restrict__ dst, int rows, int K) {
  int const r = blockIdx.x;
  if (r >= rows) return;
  int const s = rowIdx[r];
  int4 const* in = reinterpret_cast<int4 const*>(src + (size_t)s * K);
  int4* out = reinterpret_cast<int4*>(dst + (size_t)r * K);
  for (int i = threadIdx.x; i < K / 8; i += blockDim.x) out[i] = in[i];
}

int main(int argc, char** argv) {
  // -n 20 -w 8 by default, not the 30/5 the older probes use. At -n 10 our
  // kernels read 8-12% faster while cuBLAS barely moves, which misreported a
  // prefill number once already; 20/8 reproduces to under 1.5%.
  // Few iterations per visit, many rounds: the arms are interleaved so they all
  // see the same power state, and short visits keep a slow arm from throttling
  // itself inside its own measurement. See the timing block below.
  int iters = 4, warmup = 2, rounds = 12;
  float settleMs =
      2500.0f;  // drive the GPU to its throttled steady state first
  bool check = true;
  int tLo = 0, tHi = 1 << 30;
  bool freshHandle = false;
  // -g 1: use the harness's real top-6 routing instead of an even M_e per
  // expert. The default is even, so the old numbers stay reproducible.
  bool ragged = false;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    if (k == "-n" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (k == "-w" && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
    else if (k == "-c" && i + 1 < argc)
      check = std::atoi(argv[++i]) != 0;
    else if (k == "-b" && i + 1 < argc)
      tLo = std::atoi(argv[++i]);
    else if (k == "-e" && i + 1 < argc)
      tHi = std::atoi(argv[++i]);
    else if (k == "-r" && i + 1 < argc)
      rounds = std::atoi(argv[++i]);
    else if (k == "-S" && i + 1 < argc)
      settleMs = (float)std::atof(argv[++i]);
    else if (k == "-g" && i + 1 < argc)
      ragged = std::atoi(argv[++i]) != 0;
    else if (k == "-F")
      freshHandle = true;
  }

  CUDACHECK(cudaSetDevice(0));
  cublasHandle_t h;
  CUBLASCHECK(cublasCreate(&h));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLASCHECK(cublasSetStream(h, stream));

  int const kTs[] = {64, 128, 256, 512, 1024, 2048, 4096};
  int const nT = sizeof(kTs) / sizeof(kTs[0]);
  int const maxT = kTs[nT - 1];
  const size_t maxRows = (size_t)maxT * kTopK;

  __nv_bfloat16 *W, *A, *Ap, *Y, *Yref;
  int* rowIdx;
  CUDACHECK(cudaMalloc(&W, (size_t)kELoc * kN * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&A, maxRows * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&Ap, maxRows * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&Y, maxRows * kN * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&Yref, maxRows * kN * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&rowIdx, maxRows * sizeof(int)));

  // Small values: bf16 has 8 mantissa bits, and a K=2048 reduction of large
  // random values would be dominated by accumulation order rather than by
  // whether the kernel is correct.
  {
    std::vector<__nv_bfloat16> hw((size_t)kELoc * kN * kH), ha(maxRows * kH);
    for (size_t i = 0; i < hw.size(); i++)
      hw[i] =
          __float2bfloat16((float)((i * 2654435761u) % 17) * 0.0625f - 0.5f);
    for (size_t i = 0; i < ha.size(); i++)
      ha[i] = __float2bfloat16((float)((i * 40503u) % 13) * 0.125f - 0.75f);
    CUDACHECK(cudaMemcpy(W, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(A, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
    std::vector<int> hr(maxRows);
    for (size_t i = 0; i < maxRows; i++) hr[i] = (int)i;
    CUDACHECK(cudaMemcpy(rowIdx, hr.data(), maxRows * sizeof(int),
                         cudaMemcpyHostToDevice));
  }

  // One self-contained set of device argument arrays per variant. Sharing them
  // and rewriting the contents between runs is how this file started, and it is
  // a trap: initialize() captures the *addresses*, so the last writer silently
  // decides what every operator runs.
  constexpr int kMaxProb = kELoc * kGroups;
  struct Variant {
    char const* name = nullptr;
    int groups = 1;
    bool groupMajor = false;
    cutlass::gemm::GemmCoord* prob = nullptr;
    ElementA** pa = nullptr;
    ElementB** pb = nullptr;
    ElementC** pc = nullptr;
    ElementC** pd = nullptr;
    int64_t* lda = nullptr;
    int64_t* ldb = nullptr;
    int64_t* ldc = nullptr;
    int64_t* ldd = nullptr;
    Gemm op;
    void* ws = nullptr;
    int np = 0;
    typename Gemm::Arguments args{};
    int tb = 0;
  };
  Variant vars[3];
  vars[0].name = "E16";
  vars[0].groups = 1;
  vars[0].groupMajor = false;
  vars[1].name = "Eg";
  vars[1].groups = kGroups;
  vars[1].groupMajor = true;
  vars[2].name = "Ee";
  vars[2].groups = kGroups;
  vars[2].groupMajor = false;

  // Arrival flags and the problem -> group map for the gated arm. All flags are
  // set: on one GPU there is nothing to wait for.
  uint32_t* dArrived = nullptr;
  int* dGroupOf = nullptr;
  CUDACHECK(cudaMalloc(&dArrived, kGroups * sizeof(uint32_t)));
  CUDACHECK(cudaMalloc(&dGroupOf, kMaxProb * sizeof(int)));
  {
    std::vector<uint32_t> ones(kGroups, 1u);
    CUDACHECK(cudaMemcpy(dArrived, ones.data(), ones.size() * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
  }
  // The gated kernel launches its own grid, so it needs the smem size and the
  // opt-in for tiles above 48 KB that device::GemmGrouped does internally.
  int const gatedSmem = int(sizeof(typename Gated::SharedStorage));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<Gated>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 gatedSmem));
  int const gatherSmem = int(sizeof(typename GatedGather::SharedStorage));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<GatedGather>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 gatherSmem));
  ElementA** dGatherA = nullptr;
  int const** dGatherIdx = nullptr;
  CUDACHECK(cudaMalloc(&dGatherA, kMaxProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dGatherIdx, kMaxProb * sizeof(void*)));
  for (auto& v : vars) {
    CUDACHECK(cudaMalloc(&v.prob, kMaxProb * sizeof(cutlass::gemm::GemmCoord)));
    CUDACHECK(cudaMalloc(&v.pa, kMaxProb * sizeof(void*)));
    CUDACHECK(cudaMalloc(&v.pb, kMaxProb * sizeof(void*)));
    CUDACHECK(cudaMalloc(&v.pc, kMaxProb * sizeof(void*)));
    CUDACHECK(cudaMalloc(&v.pd, kMaxProb * sizeof(void*)));
    CUDACHECK(cudaMalloc(&v.lda, kMaxProb * sizeof(int64_t)));
    CUDACHECK(cudaMalloc(&v.ldb, kMaxProb * sizeof(int64_t)));
    CUDACHECK(cudaMalloc(&v.ldc, kMaxProb * sizeof(int64_t)));
    CUDACHECK(cudaMalloc(&v.ldd, kMaxProb * sizeof(int64_t)));
  }

  // cuBLAS grouped: 16 problems, always. Splitting the baseline the way E64
  // splits ours would be scoring against a handicap it does not have to accept.
  cublasOperation_t gTA[kELoc], gTB[kELoc];
  int gM[kELoc], gN[kELoc], gK[kELoc], gLdA[kELoc], gLdB[kELoc], gLdC[kELoc],
      gGS[kELoc];
  float gAl[kELoc], gBe[kELoc];
  __nv_bfloat16 const **dGA = nullptr, **dGB = nullptr;
  __nv_bfloat16** dGC = nullptr;
  CUDACHECK(cudaMalloc(&dGA, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dGB, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dGC, kELoc * sizeof(void*)));

  int nSM = 0;
  CUDACHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0));
  int const bpsm = Gemm::maximum_active_blocks();
  int const tbCount = Gemm::sufficient();

  std::printf(
      "# CUTLASS %d.%d.%d grouped bf16 GEMM vs cublasGemmGroupedBatchedEx, 1x "
      "L4 sm_89\n",
      CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH);
  std::printf(
      "# DeepSeek-V2-Lite at EP=4: K=H=%d N=%d, %d local experts, top-%d\n", kH,
      kN, kELoc, kTopK);
  std::printf(
      "# tile %dx%dx%d, warp %dx%dx%d, mma 16x8x16, %d stages, Sm80 tag, %d "
      "arrival groups\n",
      CT_BM, CT_BN, CT_BK, CT_WM, CT_WN, CT_BK, CT_STAGES, kGroups);
  std::printf("# %d SMs, %d blocks/SM resident, %d threadblocks launched\n",
              nSM, bpsm, tbCount);
  std::printf("# E16 = one problem per expert. This is the gate.\n");
  std::printf(
      "# Eg  = %d problems, one per (arrival group, expert), ordered "
      "group-major -- the order\n",
      kELoc * kGroups);
  std::printf(
      "#       gating wants, since it reaches all of group 0 before anything "
      "that waits.\n");
  std::printf(
      "# Ee  = the same %d problems ordered expert-major. Same tiles, same "
      "arithmetic; only\n",
      kELoc * kGroups);
  std::printf(
      "#       the visitor's order differs, so Eg vs Ee isolates weight reuse "
      "in L2.\n");
  std::printf(
      "# perm = the A-operand permute a non-fused path must run first.\n");
  std::printf(
      "# -g 1 uses the harness's real top-6 routing; the default is an even "
      "M_e per expert.\n");
  std::printf(
      "# %d rounds x %d iters, arms interleaved, min per arm, after %.0f ms of "
      "settling: this\n",
      rounds, iters, settleMs);
  std::printf(
      "# L4 power-throttles from 2040 to 885 MHz under sustained load (72 W "
      "cap), so neither\n");
  std::printf(
      "# sequential timing nor a cold GPU gives comparable numbers.\n#\n");
  std::printf(
      "# gated = Eg through the arrival-gated kernel with every flag pre-set, "
      "on %d blocks\n",
      nSM - 1);
  std::printf(
      "#         instead of %d. gated vs Eg is what the gate costs when "
      "nothing waits.\n#\n",
      nSM);
  std::printf(
      "# gather = the gated kernel built with CUTLASS gather-A: reads "
      "scattered rows through\n");
  std::printf(
      "#          the row list, so it needs no permute and nothing "
      "co-scheduled with it.\n");
  std::printf(
      "# Eg/gated/gather ratios are against cuBLAS+perm, since those paths "
      "replace the permute.\n#\n");
  std::printf("# %7s %7s %10s %9s %9s %9s %9s %8s %8s %9s %9s %9s %9s\n",
              "T/rank", "M_e", "cubGrp(us)", "E16(us)", "Eg(us)", "gated(us)",
              "gathr(us)", "perm(us)", "E16/cub", "Eg/cub+p", "gtd/cub+p",
              "gth/cub+p", "maxrelerr");

  cudaEvent_t e0, e1;
  CUDACHECK(cudaEventCreate(&e0));
  CUDACHECK(cudaEventCreate(&e1));

  for (int ti = 0; ti < nT; ti++) {
    int const T = kTs[ti];
    if (T < tLo || T > tHi) continue;
    int const m = (T * kTopK) / kELoc;  // nominal rows per expert
    if (m < 1) continue;

    // Row layout, expert-major and source-minor, matching the harness's receive
    // side. cnt[e][g] is how many of expert e's rows came from source rank g;
    // off[e][g] is where they start. In even mode every cnt is m/4 and the row
    // list is the identity, which is what this probe measured before -- kept as
    // a control, because the whole point is that ragged and even disagree.
    int cnt[kELoc][kRanks], off[kELoc][kRanks];
    std::vector<int> hRowIdx;
    if (ragged)
      raggedCounts(T, cnt);
    else
      for (int e = 0; e < kELoc; e++)
        for (int g = 0; g < kRanks; g++) cnt[e][g] = m / kRanks;

    // The row indices themselves. The harness's gather is scattered: rows from
    // one source are in increasing order but not contiguous, because a received
    // row feeds whichever of our experts routed to it. Reproduce that, or the
    // gather arm measures a contiguous read it will never get.
    std::vector<std::vector<int>> recvTok(kRanks);
    if (ragged) {
      int ex[kTopK];
      for (int s = 0; s < kRanks; s++)
        for (int t = 0; t < T; t++) {
          routeOf(s, t, ex);
          for (int i = 0; i < kTopK; i++)
            if (ex[i] / kELoc == 0) {
              recvTok[s].push_back(t);
              break;
            }
        }
    }
    {
      int pos = 0;
      int ex[kTopK];
      for (int e = 0; e < kELoc; e++)
        for (int g = 0; g < kRanks; g++) {
          off[e][g] = pos;
          if (ragged) {
            for (size_t j = 0; j < recvTok[g].size(); j++) {
              routeOf(g, recvTok[g][j], ex);
              for (int i = 0; i < kTopK; i++)
                if (ex[i] == e) {
                  hRowIdx.push_back(g * maxT + (int)j);
                  pos++;
                  break;
                }
            }
          } else {
            for (int j = 0; j < cnt[e][g]; j++) hRowIdx.push_back(pos++);
          }
        }
      if (ragged)
        for (int e = 0; e < kELoc; e++)
          for (int g = 0; g < kRanks; g++)
            cnt[e][g] =
                (g + 1 < kRanks ? off[e][g + 1]
                                : (e + 1 < kELoc ? off[e + 1][0] : pos)) -
                off[e][g];
    }
    const size_t totalRows = hRowIdx.size();
    if (!totalRows) continue;
    CUDACHECK(cudaMemcpy(rowIdx, hRowIdx.data(), totalRows * sizeof(int),
                         cudaMemcpyHostToDevice));
    // Rows per expert, for the report and for the cuBLAS problem list.
    int mOf[kELoc], offOf[kELoc];
    for (int e = 0; e < kELoc; e++) {
      offOf[e] = off[e][0];
      mOf[e] = 0;
      for (int g = 0; g < kRanks; g++) mOf[e] += cnt[e][g];
    }

    // --- CUTLASS problem lists
    // ------------------------------------------------ E16 : problem e covers
    // expert e's m rows, contiguous at Ap + e*m*K. E64*: problem (g,e) covers
    // the g-th quarter of expert e's rows.
    //
    // `groupMajor` decides which index is the outer one, and it is not
    // cosmetic. The visitor hands tiles out in problem order, so the ordering
    // decides both what a window of concurrent blocks holds in L2 --
    // group-major puts adjacent blocks on *different* experts' 11.5 MB weight
    // matrices, expert-major keeps them on one -- and how early a gated kernel
    // can start, since group-major reaches all of group 0 (this rank's own
    // rows, which never wait) before anything that does. The two pull opposite
    // ways, so both are measured rather than assumed.
    auto build = [&](Variant& v) {
      std::vector<cutlass::gemm::GemmCoord> hp;
      std::vector<ElementA*> ha;
      std::vector<ElementB*> hb;
      std::vector<ElementC*> hc;
      std::vector<int64_t> la, lb, lc;
      // Gate group g of v.groups covers source ranks [g*4/groups,
      // (g+1)*4/groups). Rows are source-minor within an expert, so that span
      // is contiguous.
      auto emit = [&](int g, int e) {
        const int gLo = g * kRanks / v.groups,
                  gHi = (g + 1) * kRanks / v.groups;
        const int r0 = off[e][gLo];
        int mr = 0;
        for (int gg = gLo; gg < gHi; gg++) mr += cnt[e][gg];
        if (mr <= 0) return;
        hp.push_back(cutlass::gemm::GemmCoord(mr, kN, kH));
        ha.push_back(reinterpret_cast<ElementA*>(Ap) + (size_t)r0 * kH);
        hb.push_back(reinterpret_cast<ElementB*>(W) + (size_t)e * kN * kH);
        hc.push_back(reinterpret_cast<ElementC*>(Y) + (size_t)r0 * kN);
        la.push_back(kH);  // row-major A: stride between rows
        lb.push_back(kH);  // column-major B: stride between columns
        lc.push_back(kN);  // row-major C
      };
      if (v.groupMajor)
        for (int g = 0; g < v.groups; g++)
          for (int e = 0; e < kELoc; e++) emit(g, e);
      else
        for (int e = 0; e < kELoc; e++)
          for (int g = 0; g < v.groups; g++) emit(g, e);
      v.np = (int)hp.size();
      CUDACHECK(cudaMemcpy(v.prob, hp.data(), v.np * sizeof(hp[0]),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(v.pa, ha.data(), v.np * sizeof(void*),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(v.pb, hb.data(), v.np * sizeof(void*),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(v.pc, hc.data(), v.np * sizeof(void*),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(v.pd, hc.data(), v.np * sizeof(void*),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(v.lda, la.data(), v.np * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(v.ldb, lb.data(), v.np * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(v.ldc, lc.data(), v.np * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(v.ldd, lc.data(), v.np * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
      // sufficient() caps the grid at the total tile count, so a short problem
      // list does not launch a wave of blocks that immediately exit.
      const int tb = Gemm::sufficient(hp.data(), v.np);
      typename Gemm::Arguments args(
          v.prob, v.np, tb, typename Gemm::EpilogueOutputOp::Params(1.0f, 0.0f),
          v.pa, v.pb, v.pc, v.pd, v.lda, v.ldb, v.ldc, v.ldd, hp.data());
      const size_t wsb = Gemm::get_workspace_size(args);
      if (wsb) CUDACHECK(cudaMalloc(&v.ws, wsb));
      // initialize() outside the timing loop: it does host-side work and, under
      // kHostPrecompute, a host-to-device copy of the whole schedule.
      CUTLASSCHECK(v.op.initialize(args, v.ws, stream));
      v.args = args;
      v.tb = tb;
    };
    for (auto& v : vars) build(v);

    // The gated arm reuses Eg's problem list -- same pointers, same group-major
    // order -- so the only difference between the two rows is the gate itself.
    typename Gated::Params gatedParams;
    {
      Variant& v = vars[1];
      std::vector<int> hGroupOf(v.np);
      for (int i = 0; i < v.np; i++)
        hGroupOf[i] = i / kELoc;  // group-major: group = idx / experts
      CUDACHECK(cudaMemcpy(dGroupOf, hGroupOf.data(), v.np * sizeof(int),
                           cudaMemcpyHostToDevice));
      gatedParams = typename Gated::Params(
          typename GemmKernel::Params(v.args, nullptr, 0), dArrived, dGroupOf);
    }
    // One block fewer than the GPU has SMs. On the 4-rank path that slot is
    // what the dispatcher runs in; here it only keeps the two rows comparable.
    int const gatedGrid = std::min(vars[1].tb, nSM - 1);
    auto runGated = [&]() {
      uccl_lite::gatedGroupedKernel<Gated>
          <<<gatedGrid, Gated::kThreadCount, gatedSmem, stream>>>(gatedParams);
    };

    // Gather arm: the same gated kernel built with gather-A. A comes straight
    // from the unpermuted buffer through the row list, so `perm` is not part of
    // its cost and no second kernel has to be co-scheduled with it. Its problem
    // list is Eg's, except ptr_A is the buffer base and the per-problem index
    // list starts at that problem's first row.
    typename GatedGather::Params gatherParams;
    {
      Variant& v = vars[1];
      std::vector<ElementA*> ga(v.np);
      std::vector<int const*> gi(v.np);
      int idx = 0;
      for (int g = 0; g < v.groups; g++)
        for (int e = 0; e < kELoc; e++) {
          int const gLo = g * kRanks / v.groups,
                    gHi = (g + 1) * kRanks / v.groups;
          int mr = 0;
          for (int gg = gLo; gg < gHi; gg++) mr += cnt[e][gg];
          if (mr <= 0) continue;
          ga[idx] = reinterpret_cast<ElementA*>(
              A);  // tensor base; rows come from the index list
          gi[idx] = rowIdx + off[e][gLo];
          idx++;
        }
      CUDACHECK(cudaMemcpy(dGatherA, ga.data(), v.np * sizeof(void*),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(dGatherIdx, gi.data(), v.np * sizeof(void*),
                           cudaMemcpyHostToDevice));
      typename GatherKernel::Arguments ga2(
          v.prob, v.np, std::min(v.tb, nSM - 1),
          typename GatherKernel::EpilogueOutputOp::Params(1.0f, 0.0f), dGatherA,
          v.pb, v.pc, v.pd, v.lda, v.ldb, v.ldc, v.ldd, nullptr);
      gatherParams = typename GatedGather::Params(
          typename GatherKernel::Params(ga2, nullptr, 0), dArrived, dGroupOf,
          dGatherIdx);
    }
    auto runGather = [&]() {
      uccl_lite::gatedGroupedKernel<GatedGather>
          <<<gatedGrid, GatedGather::kThreadCount, gatherSmem, stream>>>(
              gatherParams);
    };

    // --- cuBLAS grouped, 16 problems -----------------------------------------
    // The baseline stays at 16 whatever we do. Splitting it the way Eg/Ee split
    // ours would be scoring against a handicap it never has to accept.
    //
    // -F recreates the handle for each size. Sweeping sizes on one handle makes
    // cuBLAS 12% slower at T=2048 than measuring T=2048 alone, while CUTLASS
    // does not move at all -- and a baseline that is slow for a reason of our
    // own making is not a baseline.
    if (freshHandle) {
      CUBLASCHECK(cublasDestroy(h));
      CUBLASCHECK(cublasCreate(&h));
      CUBLASCHECK(cublasSetStream(h, stream));
    }
    {
      __nv_bfloat16 const *hA[kELoc], *hB[kELoc];
      __nv_bfloat16* hC[kELoc];
      for (int e = 0; e < kELoc; e++) {
        gTA[e] = CUBLAS_OP_T;
        gTB[e] = CUBLAS_OP_N;
        gM[e] = kN;
        gN[e] = mOf[e];
        gK[e] = kH;
        gLdA[e] = kH;
        gLdB[e] = kH;
        gLdC[e] = kN;
        gGS[e] = 1;
        gAl[e] = 1.0f;
        gBe[e] = 0.0f;
        hA[e] = W + (size_t)e * kN * kH;
        hB[e] = Ap + (size_t)offOf[e] * kH;
        hC[e] = Yref + (size_t)offOf[e] * kN;
      }
      CUDACHECK(cudaMemcpy(dGA, hA, sizeof(hA), cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(dGB, hB, sizeof(hB), cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(dGC, hC, sizeof(hC), cudaMemcpyHostToDevice));
    }
    auto cubGrp = [&]() {
      CUBLASCHECK(cublasGemmGroupedBatchedEx(
          h, gTA, gTB, gM, gN, gK, gAl, (const void* const*)dGA, CUDA_R_16BF,
          gLdA, (const void* const*)dGB, CUDA_R_16BF, gLdB, gBe,
          (void* const*)dGC, CUDA_R_16BF, gLdC, kELoc, gGS,
          CUBLAS_COMPUTE_32F));
    };
    auto perm = [&]() {
      permuteRows<<<(int)totalRows, 128, 0, stream>>>(A, rowIdx, Ap,
                                                      (int)totalRows, kH);
    };

    // The permute feeds both paths, so run it once up front and leave Ap alone.
    perm();
    CUDACHECK(cudaStreamSynchronize(stream));

    // --- correctness ---------------------------------------------------------
    // Every variant against the same cuBLAS reference on the same permuted A.
    // Both accumulate in fp32 and round the output to bf16, so agreement should
    // be at the bf16 noise floor -- and splitting the problem list must not
    // move it at all, since that changes the schedule and not the arithmetic.
    double maxRel = -1.0;
    if (check) {
      cubGrp();
      CUDACHECK(cudaStreamSynchronize(stream));
      std::vector<__nv_bfloat16> a(totalRows * kN), b(totalRows * kN);
      CUDACHECK(
          cudaMemcpy(b.data(), Yref, b.size() * 2, cudaMemcpyDeviceToHost));
      maxRel = 0.0;
      auto compare = [&]() {
        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(
            cudaMemcpy(a.data(), Y, a.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < a.size(); i++) {
          const double x = (double)__bfloat162float(a[i]),
                       y = (double)__bfloat162float(b[i]);
          const double den = (y < 0 ? -y : y) + 1e-3;
          const double r = ((x - y) < 0 ? (y - x) : (x - y)) / den;
          if (r > maxRel) maxRel = r;
        }
      };
      for (auto& v : vars) {
        CUDACHECK(cudaMemsetAsync(Y, 0, totalRows * kN * 2, stream));
        CUTLASSCHECK(v.op.run(stream));
        compare();
      }
      // The gated kernel is a hand-copied CUTLASS body; it gets checked like
      // any other arm, against the same cuBLAS reference.
      CUDACHECK(cudaMemsetAsync(Y, 0, totalRows * kN * 2, stream));
      runGated();
      compare();
      CUDACHECK(cudaMemsetAsync(Y, 0, totalRows * kN * 2, stream));
      runGather();
      compare();
    }

    // --- timing
    // ---------------------------------------------------------------
    // Interleaved rounds, minimum per arm. This is not fussiness: an L4 is a
    // 72 W card and sustained bf16 tensor-core work saturates that budget, so
    // the SM clock swings between 2040 and 885 MHz *within a single run*.
    // Timing one arm to completion and then the next hands whichever ran first
    // a colder GPU -- which is how measuring cuBLAS at T=2048 alone gave 2480
    // us and measuring it after T=1024 in the same process gave 2785, a 12%
    // swing that has nothing to do with either kernel.
    //
    // Round-robin at a few iterations per visit puts every arm at the same
    // average power state; the minimum over rounds is each arm's
    // least-throttled observation. Both are needed -- interleaving alone still
    // lets a long arm throttle itself more inside its own measurement.
    auto timeOnce = [&](auto&& fn) -> double {
      CUDACHECK(cudaStreamSynchronize(stream));
      CUDACHECK(cudaEventRecord(e0, stream));
      for (int i = 0; i < iters; i++) fn();
      CUDACHECK(cudaEventRecord(e1, stream));
      CUDACHECK(cudaEventSynchronize(e1));
      float ms = 0;
      CUDACHECK(cudaEventElapsedTime(&ms, e0, e1));
      return (double)ms * 1000.0 / iters;
    };
    auto runArm = [&](int a) {
      if (a == 0)
        cubGrp();
      else if (a <= 3)
        CUTLASSCHECK(vars[a - 1].op.run(stream));
      else if (a == 4)
        runGated();
      else if (a == 5)
        runGather();
      else
        perm();
    };
    constexpr int kArms = 7;  // cub, E16, Eg, Ee, gated, gather, perm
    double best[kArms];
    for (int a = 0; a < kArms; a++) best[a] = 1e30;
    // Settle to the throttled steady state before timing. A cold GPU runs this
    // GEMM at 70 TF/s and a settled one at 61, and which of those a benchmark
    // reports should not depend on how many sizes happened to run before it. A
    // server runs these back to back, so the settled number is the honest one.
    {
      cudaEvent_t s0, s1;
      CUDACHECK(cudaEventCreate(&s0));
      CUDACHECK(cudaEventCreate(&s1));
      CUDACHECK(cudaEventRecord(s0, stream));
      float el = 0;
      do {
        for (int i = 0; i < warmup; i++)
          for (int a = 0; a < kArms; a++) runArm(a);
        CUDACHECK(cudaEventRecord(s1, stream));
        CUDACHECK(cudaEventSynchronize(s1));
        CUDACHECK(cudaEventElapsedTime(&el, s0, s1));
      } while (el < settleMs);
      CUDACHECK(cudaEventDestroy(s0));
      CUDACHECK(cudaEventDestroy(s1));
    }
    for (int r = 0; r < rounds; r++)
      for (int a = 0; a < kArms; a++) {
        double const t = timeOnce([&] { runArm(a); });
        if (t < best[a]) best[a] = t;
      }
    double const tCub = best[0], tGated = best[4], tGather = best[5],
                 tPerm = best[6];
    double const* t = best + 1;

    double const flops = 2.0 * (double)totalRows * kN * kH;
    // gather does no permute, so the honest comparison for it is against
    // cuBLAS *plus* the permute cuBLAS needs -- and likewise for Eg.
    std::printf(
        "  %7d %7d %10.1f %9.1f %9.1f %9.1f %9.1f %8.1f %7.2fx %7.2fx %7.2fx "
        "%7.2fx %9.2e\n",
        T, (int)(totalRows / kELoc), tCub, t[0], t[1], tGated, tGather, tPerm,
        tCub / t[0], (tCub + tPerm) / (t[1] + tPerm),
        (tCub + tPerm) / (tGated + tPerm), (tCub + tPerm) / tGather, maxRel);
    std::fflush(stdout);

    for (auto& v : vars) {
      if (v.ws) CUDACHECK(cudaFree(v.ws));
      v.ws = nullptr;
    }
  }
  return 0;
}

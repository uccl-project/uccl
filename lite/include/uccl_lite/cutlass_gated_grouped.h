// Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
// reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Adapted for UCCL Lite: arrival gating, dispatcher, and output-readiness
// hooks. A CUTLASS grouped GEMM whose tiles wait for their rows to arrive.
//
// This is the piece CUTLASS does not have. Everything else the fused MoE step
// needs it already ships: a tuned sm_80/89 bf16 grouped GEMM that beats
// cublasGemmGroupedBatchedEx by 1.2x on these shapes
// (benchmarks/data/cutlass_gemm_probe.txt). What it cannot do is start a tile
// before the rows that tile reads have crossed PCIe, which is the whole point
// of fusing dispatch into the GEMM.
//
// So this file wraps cutlass::gemm::kernel::GemmGrouped rather than replacing
// it. `operator()` is a copy of that kernel's, from CUTLASS 4.5.0, with exactly
// one thing added: after the problem visitor names a tile, the block spins
// until the arrival group that tile's rows belong to has been published. Every
// type, the mainloop, the epilogue, the problem visitor and the tile scheduler
// are the upstream ones, used unmodified.
//
// Two conventions make the gate a single integer compare:
//
//   Problems are one per (arrival group, expert), so a tile's arrival group is
//   a property of its *problem* and needs no arithmetic on row indices. The
//   receive buffer is already blocked that way -- the dispatch writes each
//   peer's rows into its own slot -- so this costs no permute. See the header
//   of cutlass_gemm_probe.cu for why the group count is 2 and not 4.
//
//   Problems are ordered group-major, so the visitor reaches every tile of
//   group 0 (this rank's own rows, which never wait) before any tile that can
//   block.
//
// The optional Dispatcher_ reserves blocks inside this grid so communication
// can make progress while GEMM workers wait for arrival flags. Standalone
// compute probes use NoDispatch. See docs/fusion.md for the separate host-proxy
// progress constraints: reserving GPU occupancy alone does not address blocking
// CUDA calls issued by the host.
//
// Pinned to CUTLASS 4.5.0. The copied body is ~120 lines and the diff against
// upstream is the block marked ARRIVAL GATE; if CUTLASS changes that kernel
// this file has to be re-synced by hand.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

namespace uccl_lite {

// Default: no dispatcher, every block is a worker. The standalone probe uses
// this, so the wrapper costs nothing when there is no collective to run.
struct NoDispatch {
  static constexpr bool kEnabled = false;
  static constexpr int kBlocks = 0;
  CUTLASS_DEVICE void operator()() {}
};

struct NoOutputHook {
  static constexpr bool kEnabled = false;
};

// Wraps a CUTLASS grouped GEMM kernel type with an arrival gate.
//
// `Kernel_` is whatever DefaultGemmGrouped produced. Nothing about it is
// redefined here; the wrapper only adds two pointers to the parameter block and
// the spin to the tile loop.
// `Dispatcher_` lets block 0 run a collective instead of GEMM tiles. It is a
// template parameter, and holds its own state by value, so this header stays
// free of any dependency on the communication library.
//
// Running the dispatcher as a *separate kernel* is the obvious design and it is
// the wrong one. A separate kernel can be scheduled behind the grid that is
// waiting for it -- the GEMM's blocks take every SM, spin on flags the
// dispatcher was never scheduled to publish, and nothing moves. Reserving SMs
// only lowers the probability; measured, it still deadlocked about half the
// time at prefill. A block of the same grid cannot be scheduled behind its own
// grid, so putting the dispatcher here removes the cycle rather than making it
// rarer. This is also what `fused-ce` does, and that kernel has never
// deadlocked.
template <typename Kernel_, typename Dispatcher_ = NoDispatch,
          typename OutputHook_ = NoOutputHook>
struct GatedGrouped {
  using Kernel = Kernel_;
  using Dispatcher = Dispatcher_;
  using OutputHook = OutputHook_;
  using Mma = typename Kernel::Mma;
  using Epilogue = typename Kernel::Epilogue;
  using EpilogueOutputOp = typename Kernel::EpilogueOutputOp;
  using ProblemVisitor = typename Kernel::ProblemVisitor;
  using SharedStorage = typename Kernel::SharedStorage;
  using Arguments = typename Kernel::Arguments;
  static bool const kTransposed = Kernel::kTransposed;
  static int const kThreadCount = Kernel::kThreadCount;

  struct Params {
    typename Kernel::Params base{};

    // One flag per arrival group, published by the dispatcher with a release
    // fence once that group's rows are in the receive buffer. `volatile`
    // because the compiler must not hoist the load out of the spin. nullptr
    // disables the gate entirely and compiles it out of the hot path, which is
    // how the standalone probe prices the wrapper against ungated CUTLASS.
    uint32_t const volatile* arrived{nullptr};

    // Problem index -> arrival group. Gates at *problem* granularity, which
    // requires the problem list to be split by arrival group. Kept for the
    // measured comparison; tile_group below is the better one.
    int const* group_of{nullptr};

    // Tile-level gating, and the reason it is better: the problem list is not
    // split at all. `tile_group[tile_group_off[problem_idx] + m_tile]` is the
    // arrival group of the *last row* that m-tile covers, which is the latest
    // arrival it depends on, because rows are ordered by source within an
    // expert.
    //
    // Splitting the problem list to carry the group is what forces the two
    // costs that pin the design down: an M_e/G-row problem underfills a BM-row
    // tile, and the group-major order it needs makes consecutive problems
    // different experts, so a block striding by gridDim.x lands on a fresh
    // 11.5 MB weight matrix every tile. Deriving the group from the tile index
    // instead pays neither -- one problem per expert, expert-major order, full
    // tiles -- and gates *finer*, per tile rather than per group.
    //
    // Non-null takes precedence over group_of.
    int const* tile_group{nullptr};
    int const* tile_group_off{nullptr};

    // Optional cyclic M-tile offset.  Dense AG uses this to start each rank at
    // the rows already local to that rank, mirroring Flux's rank-offset
    // threadblock swizzle.  Zero preserves CUTLASS's original traversal.
    int tile_m_offset{0};

    // Per-problem row-index lists for a gather-A build. nullptr means the A
    // operand is already contiguous and a separate permute kernel put it there.
    //
    // Gathering inside the GEMM is not just a saved kernel. The permute has to
    // run *while* the gated GEMM spins, and a persistent GEMM whose blocks each
    // hold most of an SM's shared memory cannot share an SM with anything --
    // which is what deadlocks the separate-permute design at prefill. Gather-A
    // removes the second kernel, so nothing needs to be co-scheduled.
    int const* const* gather_indices{nullptr};

    // Block 0's collective, by value. Empty when Dispatcher is NoDispatch.
    Dispatcher dispatcher{};
    int dispatcher_blocks = Dispatcher::kBlocks;

    // Optional per-output-tile completion flags.  The resident collective can
    // consume a (M,N) tile immediately after its CUTLASS epilogue reaches
    // global memory, without splitting the GEMM into host launches.
    uint32_t* completed{nullptr};

    // Optional worker-side continuation, used by TP-MoE to publish GEMM
    // readiness and execute token-combination tasks between GEMM tiles.
    OutputHook output_hook{};

    Params() = default;
    Params(typename Kernel::Params const& b, uint32_t const volatile* a,
           int const* g, int const* const* gi = nullptr,
           Dispatcher const& d = Dispatcher{})
        : base(b), arrived(a), group_of(g), gather_indices(gi), dispatcher(d) {}
  };

  CUTLASS_DEVICE
  GatedGrouped() {}

  CUTLASS_DEVICE
  void operator()(Params const& params_gated, SharedStorage& shared_storage) {
    typename Kernel::Params const& params = params_gated.base;

    // These shadow the type-level definitions to support the transposed
    // variant, exactly as upstream does.
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    // The first dispatcher blocks run the collective, and the workers renumber
    // around them so the visitor still sees a dense 0..gridDim-1 range.
    if constexpr (Dispatcher::kEnabled) {
      if (blockIdx.x < params_gated.dispatcher_blocks) {
        // By value: the channel handles' device methods are non-const, and
        // Params is passed by const reference.
        Dispatcher d = params_gated.dispatcher;
        d();
        return;
      }
    }
    int const kShift =
        Dispatcher::kEnabled ? params_gated.dispatcher_blocks : 0;

    ProblemVisitor problem_visitor(params.problem_visitor,
                                   shared_storage.problem_visitor,
                                   blockIdx.x - kShift);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {
      cutlass::gemm::GemmCoord problem_size = problem_visitor.problem_size();
      int32_t problem_idx = problem_visitor.problem_index();
      int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

      cutlass::gemm::GemmCoord grid_shape =
          problem_visitor.grid_shape(problem_size);

      // Keep the visitor's unique linear tile assignment, but rotate its M
      // coordinate.  N stays unchanged, so every logical output tile is still
      // visited exactly once.
      if (params_gated.tile_m_offset != 0 && grid_shape.m() > 1) {
        int const n_tile = threadblock_idx % grid_shape.n();
        int const m_tile =
            (threadblock_idx / grid_shape.n() + params_gated.tile_m_offset) %
            grid_shape.m();
        threadblock_idx = m_tile * grid_shape.n() + n_tile;
      }

      // ---- ARRIVAL GATE -- the main addition to the upstream kernel
      // ---------- One thread polls and the rest wait on the barrier, so a
      // blocked tile costs one warp spinning rather than eight. The dispatcher
      // fences before it sets the flag, so seeing the flag implies the rows are
      // visible.
      //
      // Inside the tile loop, not before it: a block works through many tiles
      // and what each one waits for changes. Tiles covering only this rank's
      // own rows never block at all.
      if (params_gated.arrived != nullptr) {
        int g;
        if (params_gated.tile_group != nullptr) {
          int const m_tile = int(threadblock_idx) / grid_shape.n();
          g = params_gated.tile_group[params_gated.tile_group_off[problem_idx] +
                                      m_tile];
        } else {
          g = params_gated.group_of[problem_idx];
        }
        if (threadIdx.x == 0)
          while (params_gated.arrived[g] == 0) __threadfence_block();
        __syncthreads();
      }
      // ---- end ARRIVAL GATE
      // --------------------------------------------------

      cutlass::gemm::GemmCoord threadblock_offset(
          int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
          int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN, 0);

      // Load element pointers. Exchange pointers and strides if working on the
      // transpose
      ElementA* ptr_A = reinterpret_cast<ElementA*>((
          kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
      typename LayoutA::LongIndex ldm_A =
          (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

      ElementB* ptr_B = reinterpret_cast<ElementB*>((
          kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
      typename LayoutB::LongIndex ldm_B =
          (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{threadblock_offset.m(), 0};
      cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n()};

      int thread_idx = threadIdx.x;

      // The indices argument is present on PredicatedTileIterator whether or
      // not it was built with Gather, and ignored when it was not, so one body
      // serves both builds.
      typename Mma::IteratorA iterator_A(
          LayoutA(ldm_A), ptr_A, {problem_size.m(), problem_size.k()},
          thread_idx, tb_offset_A,
          params_gated.gather_indices ? params_gated.gather_indices[problem_idx]
                                      : nullptr);
      typename Mma::IteratorB iterator_B(LayoutB(ldm_B), ptr_B,
                                         {problem_size.k(), problem_size.n()},
                                         thread_idx, tb_offset_B);

      typename Mma::FragmentC accumulators;
      accumulators.clear();

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = cutlass::canonical_warp_idx_sync();
      int lane_idx = threadIdx.x % 32;

      Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

      int gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous
      // tile.
      __syncthreads();

      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
          accumulators);

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      ElementC* ptr_C = params.ptr_C[problem_idx];
      ElementC* ptr_D = params.ptr_D[problem_idx];

      LayoutC layout_C(params.ldc[problem_idx]);
      LayoutC layout_D(params.ldd[problem_idx]);

      typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      typename Epilogue::OutputTileIterator::Params params_D(layout_D);

      typename Epilogue::OutputTileIterator iterator_C(
          params_C, ptr_C, problem_size.mn(), thread_idx,
          threadblock_offset.mn());
      typename Epilogue::OutputTileIterator iterator_D(
          params_D, ptr_D, problem_size.mn(), thread_idx,
          threadblock_offset.mn());

      Epilogue epilogue(shared_storage.kernel.epilogue, thread_idx, warp_idx,
                        lane_idx);

      epilogue(output_op, iterator_D, accumulators, iterator_C);

      if constexpr (OutputHook::kEnabled) {
        params_gated.output_hook.complete_tile(problem_idx);
      }

      // Every epilogue thread releases its stores at device scope before thread
      // 0 publishes the tile with a system-scope atomic.  The consumer first
      // acquires that flag and MemoryChannel::signal later performs the system
      // publication of the PCIe payload; forcing every epilogue thread to flush
      // L2 at system scope here is both redundant and very expensive.
      if (params_gated.completed != nullptr) {
        __syncthreads();
        __threadfence();
        __syncthreads();
        if (thread_idx == 0) {
          int const m_tile = threadblock_offset.m() / Mma::Shape::kM;
          int const n_tile = threadblock_offset.n() / Mma::Shape::kN;
          atomicMax_system(
              params_gated.completed + m_tile * grid_shape.n() + n_tile, 1u);
        }
      }

      // Next tile
      problem_visitor.advance(gridDim.x - kShift);
    }
    if constexpr (OutputHook::kEnabled) params_gated.output_hook.finish();
  }
};

// Launcher. cutlass::Kernel2 would do, but it takes Operator::Params by value
// and we want the gated Params, so this is the same three lines with our type.
template <typename Op>
__global__ void gatedGroupedKernel(typename Op::Params params) {
  extern __shared__ int gatedSharedStorage[];
  typename Op::SharedStorage* shared_storage =
      reinterpret_cast<typename Op::SharedStorage*>(gatedSharedStorage);
  Op op;
  op(params, *shared_storage);
}

}  // namespace uccl_lite

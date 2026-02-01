#pragma once

#include "kittens.cuh"
#include "../task.h"

namespace UKernel {
namespace Compute {

using namespace kittens;

// TK level_08 constants
// [TODO: Yihan]basic tk level 8 gemm integration, future perf opt can be done here
constexpr int TK_BLOCK_SIZE = 64;
constexpr int TK_M_BLOCK = 2;  // output tiles per row
constexpr int TK_N_BLOCK = 4;  // output tiles per col
constexpr int TK_NUM_PRODUCER_WARPS = 4;
constexpr int TK_NUM_CONSUMER_WARPS = TK_M_BLOCK * 4;
constexpr int TK_NUM_THREADS = (TK_NUM_PRODUCER_WARPS + TK_NUM_CONSUMER_WARPS) * kittens::WARP_THREADS;  // 384

// TMA-enabled global layout for GEMM tiles
using TkSubTile = st_bf<TK_BLOCK_SIZE, TK_BLOCK_SIZE>;
using TkTileGL = gl<bf16, 1, 1, -1, -1, TkSubTile>;

// Pre-constructed TMA descriptors for A, B, C 
struct TkMatmulGlobals {
    TkTileGL A, B, C;

    // Host-only constructor
    __host__ TkMatmulGlobals(bf16* a_ptr, bf16* b_ptr, bf16* c_ptr,
                              int M, int N, int K)
        : A(a_ptr, nullptr, nullptr, M, K),
          B(b_ptr, nullptr, nullptr, K, N),
          C(c_ptr, nullptr, nullptr, M, N) {}


    __host__ __device__ TkMatmulGlobals(const TkMatmulGlobals& other)
        : A(other.A), B(other.B), C(other.C) {}
};

// Level_08 GEMM device kernel
__device__ void run_tk_gemm(TkMatmulGlobals const& g, int tile_row, int tile_col, char* smem) {
    shared_allocator al((int*)smem);
    st_bf<TK_BLOCK_SIZE, TK_BLOCK_SIZE> (&As)[2][TK_M_BLOCK] =
        al.allocate<st_bf<TK_BLOCK_SIZE, TK_BLOCK_SIZE>, 2, TK_M_BLOCK>();
    st_bf<TK_BLOCK_SIZE, TK_BLOCK_SIZE> (&Bs)[2][TK_N_BLOCK] =
        al.allocate<st_bf<TK_BLOCK_SIZE, TK_BLOCK_SIZE>, 2, TK_N_BLOCK>();
    st_bf<TK_BLOCK_SIZE, TK_BLOCK_SIZE> (&C_tiles)[TK_M_BLOCK][TK_N_BLOCK] =
        al.allocate<st_bf<TK_BLOCK_SIZE, TK_BLOCK_SIZE>, TK_M_BLOCK, TK_N_BLOCK>();

    int tic = 0, toc = 1;

    using wide_tile = st_bf<TK_BLOCK_SIZE, TK_BLOCK_SIZE * TK_N_BLOCK>;
    rt_fl<16, TK_BLOCK_SIZE * TK_N_BLOCK> C_accum;

    int row = tile_row * TK_M_BLOCK;
    int col = tile_col * TK_N_BLOCK;

    // Warpgroup roles: 0=producer, 1..M_BLOCK=consumers
    const int warpid = kittens::warpid();
    const int warpgroupid = warpid / 4;
    bool is_producer = (warpgroupid == 0);
    bool is_consumer = (warpgroupid > 0 && warpgroupid <= TK_M_BLOCK);
    int consumer_idx = is_consumer ? (warpgroupid - 1) : 0;

    // Initial TMA load
    __shared__ semaphore bar;
    if (threadIdx.x == 0) {
        init_semaphore(bar, 0, 1);
        tma::expect_bytes(bar,
            TK_M_BLOCK * size_bytes<typeof(As[0][0])> +
            TK_N_BLOCK * size_bytes<typeof(Bs[0][0])>);
        for (int m = 0; m < TK_M_BLOCK; m++)
            tma::load_async(As[tic][m], g.A, {0, 0, row + m, 0}, bar);
        for (int n = 0; n < TK_N_BLOCK; n++)
            tma::load_async(Bs[tic][n], g.B, {0, 0, 0, col + n}, bar);
    }
    __syncthreads();

    if (is_consumer) kittens::warp::zero(C_accum);

    // Main loop: producer prefetches, consumers compute
    int K = g.A.cols();
    int num_tiles = (K + TK_BLOCK_SIZE - 1) / TK_BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile, tic ^= 1, toc ^= 1) {
        wait(bar, tic);
        __syncthreads();

        if (is_producer) {
            warpgroup::decrease_registers<40>();
            if (threadIdx.x == 0 && tile + 1 < num_tiles) {
                tma::expect_bytes(bar,
                    TK_M_BLOCK * size_bytes<typeof(As[0][0])> +
                    TK_N_BLOCK * size_bytes<typeof(Bs[0][0])>);
                for (int m = 0; m < TK_M_BLOCK; m++)
                    tma::load_async(As[toc][m], g.A, {0, 0, row + m, tile + 1}, bar);
                for (int n = 0; n < TK_N_BLOCK; n++)
                    tma::load_async(Bs[toc][n], g.B, {0, 0, tile + 1, col + n}, bar);
            }
        } else if (is_consumer) {
            warpgroup::increase_registers<232>();
            warpgroup::mma_AB(C_accum, As[tic][consumer_idx],
                reinterpret_cast<wide_tile&>(Bs[tic][0]));
            warpgroup::mma_async_wait();
        }
        __syncthreads();
    }

    // Store result using TMA
    if (is_consumer) {
        wide_tile& wide_C_temp = reinterpret_cast<wide_tile&>(C_tiles[consumer_idx][0]);
        warpgroup::store(wide_C_temp, C_accum);
        warpgroup::sync(warpgroupid + 4);

        if (warpid % 4 == 0) {
            for (int n = 0; n < TK_N_BLOCK; n++) {
                tma::store_async(g.C, C_tiles[consumer_idx][n], {0, 0, row + consumer_idx, col + n});
                tma::store_async_read_wait();
            }
        }
    }
}

}  // namespace Compute
}  // namespace UKernel

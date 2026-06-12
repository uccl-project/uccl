// ===========================================================================
// UCCL-GIN put + quiet smoke (the AMD/RoCE validation vehicle)
// ===========================================================================
//
// Exercises ONLY the two primitives that work on the non-EFA RC path:
//   gin.put<Rail>()  -> plain RDMA WRITE
//   gin.quiet()      -> drain the lane's WRITE CQEs (source reusable)
//
// No atomics: red_add_rel / put_tail_add / put_value are EFA-shaped and unusable
// on non-EFA. Completion is proven structurally, not by a counter:
//   each rank fills its send window with a rank-tagged pattern, poisons recv,
//   put()s to its paired-remote peer, quiet()s (CQE drained => data landed at
//   the remote NIC for reliable RC), then an MPI barrier orders both ranks'
//   writes before either reads its recv window and checks it equals the PEER's
//   pattern. A lost/torn/early write leaves poison -> FAIL.
//
// Raw cuda* calls here are intentional: nvcc compiles them natively (NVIDIA
// Makefile path) and torch's hipify translates them to hip* (AMD setup.py path).

#include "../context.hpp"
#include "../uccl_gin/uccl_gin.cuh"

#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace uccl_gin {

namespace {

__global__ void smoke_fill_pattern(int* p, size_t n, int rank) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = rank * 1000003 + static_cast<int>(i);
}

// Single-thread: one put of `bytes` to the paired-remote peer, then quiet.
__global__ void smoke_put_quiet(UCCLGinResources res, int peer, void* send_ptr,
                                void* recv_ptr, uint32_t bytes, int lane) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  UCCLGin gin(res);
  gin.put<ncclTeamTagRail>(recv_ptr, send_ptr, static_cast<int>(bytes), peer,
                           lane);
  gin.quiet(lane);
}

// Bandwidth: stream `iters` puts to the peer, fanned round-robin across all D2H
// lanes, then quiet every lane so the timed region ends when all WRITE CQEs have
// drained. per-rank BW = bytes*iters / elapsed.
__global__ void smoke_put_bench(UCCLGinResources res, int peer, void* send_ptr,
                                void* recv_ptr, uint32_t bytes, int iters,
                                int num_lanes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  UCCLGin gin(res);
  for (int it = 0; it < iters; ++it) {
    gin.put<ncclTeamTagRail>(recv_ptr, send_ptr, static_cast<int>(bytes), peer,
                             it % num_lanes);
  }
  for (int l = 0; l < num_lanes; ++l) gin.quiet(l);
}

}  // namespace

// Returns true if recv == peer's pattern after put/quiet/barrier. `bytes` must
// be a multiple of 4 and <= max_message_bytes. Symmetric: every rank both sends
// to and receives from its peer, so all ranks call this in lockstep.
bool run_put_quiet_smoke(Context& ctx, int peer, int bytes) {
  if (bytes <= 0 || (bytes & 3) ||
      static_cast<size_t>(bytes) > ctx.max_message_bytes()) {
    std::fprintf(stderr, "[put_quiet_smoke] bad bytes=%d (max=%zu)\n", bytes,
                 ctx.max_message_bytes());
    return false;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int* send = static_cast<int*>(ctx.send_ptr());
  int* recv = static_cast<int*>(ctx.recv_ptr());
  const size_t n = static_cast<size_t>(bytes) / sizeof(int);

  smoke_fill_pattern<<<static_cast<unsigned>((n + 255) / 256), 256>>>(send, n,
                                                                      rank);
  cudaMemset(recv, 0xFF, bytes);  // poison
  if (cudaDeviceSynchronize() != cudaSuccess) return false;

  MPI_Barrier(MPI_COMM_WORLD);
  smoke_put_quiet<<<1, 32>>>(ctx.resources(), peer, send, recv,
                             static_cast<uint32_t>(bytes), /*lane=*/0);
  if (cudaDeviceSynchronize() != cudaSuccess) return false;  // quiet drained

  // Order both ranks' writes before anyone reads. After this, our recv holds
  // the peer's payload (RC delivery + quiet CQE happened-before the barrier).
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<int> h(n);
  if (cudaMemcpy(h.data(), recv, n * sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    const int expect = peer * 1000003 + static_cast<int>(i);
    if (h[i] != expect) {
      std::fprintf(stderr,
                   "[put_quiet_smoke] rank %d word %zu: got %d want %d\n", rank,
                   i, h[i], expect);
      return false;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

// Timed put bandwidth to the paired-remote peer. Returns per-rank GB/s
// (bytes*iters / elapsed), or -1.0 on error. `warmup` puts are run untimed.
double run_put_bench(Context& ctx, int peer, int bytes, int iters, int warmup) {
  if (bytes <= 0 || (bytes & 3) ||
      static_cast<size_t>(bytes) > ctx.max_message_bytes() || iters <= 0) {
    return -1.0;
  }
  void* send = ctx.send_ptr();
  void* recv = ctx.recv_ptr();
  const int num_lanes = ctx.num_queues();

  smoke_put_bench<<<1, 32>>>(ctx.resources(), peer, send, recv,
                             static_cast<uint32_t>(bytes), warmup, num_lanes);
  if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  smoke_put_bench<<<1, 32>>>(ctx.resources(), peer, send, recv,
                             static_cast<uint32_t>(bytes), iters, num_lanes);
  if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
  const double t1 = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  const double secs = t1 - t0;
  if (secs <= 0.0) return -1.0;
  return (static_cast<double>(bytes) * iters) / secs / 1e9;
}

}  // namespace uccl_gin

// ===========================================================================
// UCCL-GIN vs NCCL-GIN standalone microbench (Rail / inter-node path)
// ===========================================================================
//
// Goal: exercise the few UCCL-GIN Rail ops (put + red_add_rel) in isolation,
// WITHOUT DeepEP V2, and compare against the same workload run over native
// NCCL-GIN. Both paths live in one MPI program for apples-to-apples timing.
//
// Workload (paired-remote, mirrors AGENTS.md gin_proxy_bench "same-local-remote"):
//   - 2 nodes x local_world ranks (EP16 default: 8/node).
//   - Each rank r streams `iters` messages of `msg_bytes` to its remote pair
//     peer = (r + local_world) % world (same local rank on the other node),
//     then issues a red_add to a per-(peer) counter.
//   - Per-rank achievable BW = iters*msg_bytes / elapsed; correctness checks the
//     receiver's recv buffer + the red_add counter.
//
// Two paths:
//   [NCCL-GIN]  ncclMemAlloc + window + ncclDevComm(GIN) + ncclGin.put(SignalInc)
//               (reference, modeled on nccl/docs/examples/06_device_api/02_alltoall_gin)
//   [UCCL-GIN]  UcclProxy (EFA) + D2H ring + handle::UCCLGin (gin.put<Rail> / red_add_rel<Rail>)
//
// Build: see Makefile. Launch with mpirun across 2 nodes (paired-remote workload).
// ===========================================================================

#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>
#if UCCL_GIN_WITH_NCCL_GIN
#include <nccl_device/comm.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <nccl_device/gin_barrier.h>
#include <nccl_device/impl/gin__funcs.h>
#include <nccl_device/impl/gin_barrier__funcs.h>
#endif
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../transport/ring_buffer.cuh"
#include "../transport/d2h_queue_device.cuh"
#include "../transport/uccl_proxy.hpp"
#include "../context.hpp"
#include "../uccl_gin/uccl_gin.cuh"

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"[CUDA] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define NCCL_OK(x) do { ncclResult_t r=(x); if(r!=ncclSuccess){ \
  fprintf(stderr,"[NCCL] %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct Args {
  int   iters       = 50;
  int   warmup      = 5;
  bool  run_nccl    = UCCL_GIN_WITH_NCCL_GIN;
  bool  run_uccl    = true;
  bool  correctness_only = false;
  const char* ifname = "enp71s0";       // NCCL_SOCKET_IFNAME on p5en
  std::string only = "all";
  std::vector<size_t> sizes = {4096, 16384, 65536, 262144, 1048576, 16777216};
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](){ return (i+1<argc) ? argv[++i] : nullptr; };
    if (s == "--iters") a.iters = atoi(next());
    else if (s == "--warmup") a.warmup = atoi(next());
    else if (s == "--ifname") a.ifname = next();
    else if (s == "--no-nccl") a.run_nccl = false;
    else if (s == "--no-uccl") a.run_uccl = false;
    else if (s == "--correctness-only") a.correctness_only = true;
    else if (s == "--only") a.only = next();
    else if (s == "--sizes") { a.sizes.clear(); char* t=strtok(next(),","); while(t){a.sizes.push_back(strtoull(t,0,10)); t=strtok(0,",");} }
  }
  return a;
}

static bool selected(Args const& args, const char* primitive) {
  return args.only == "all" || args.only == primitive;
}

// ===========================================================================
// NCCL-GIN reference path
// ===========================================================================
//
// Paired put: rank r writes `count` floats into peer's recv buffer, signaling
// the peer's signal index. Mirrors the alltoall_gin example but 1 peer (the
// remote pair) instead of all ranks, so it matches the UCCL paired workload.

#if UCCL_GIN_WITH_NCCL_GIN
#define NCCL_CTAS 16
#define NCCL_TPB  512

__global__ void nccl_gin_paired_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                       size_t bytes, int peer, int iters,
                                       struct ncclDevComm devComm) {
  int ginCtx = 0;
  unsigned sig = blockIdx.x;
  ncclGin gin{devComm, ginCtx};
  uint64_t base = gin.readSignal(sig);
  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  // One put per iter from this CTA (CTA 0 only, to keep it a single stream that
  // is comparable to one UCCL D2H lane; multi-CTA fan-out is a later sweep).
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      for (int it = 0; it < iters; ++it) {
        gin.put(ncclTeamWorld(devComm), peer,
                recvwin, /*recvoff=*/0, sendwin, /*sendoff=*/0,
                bytes, ncclGin_SignalInc{sig});
      }
    }
    __syncthreads();
    gin.waitSignal(ncclCoopCta(), sig, base + iters);  // wait inbound from our pair
    gin.flush(ncclCoopCta());
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

static double run_nccl_gin(ncclComm_t comm, struct ncclDevComm devComm,
                           ncclWindow_t sendwin, ncclWindow_t recvwin,
                           void* d_send, void* d_recv,
                           size_t bytes, int peer, int iters, int warmup,
                           cudaStream_t stream) {
  CUDA_OK(cudaMemset(d_recv, 0, bytes));
  // warmup
  nccl_gin_paired_kernel<<<NCCL_CTAS, NCCL_TPB, 0, stream>>>(sendwin, recvwin, bytes, peer, warmup, devComm);
  CUDA_OK(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  // Timed region: identical wall-clock + MPI-barrier methodology as the UCCL
  // path (apples-to-apples). The kernel streams `iters` puts then waitSignal()s
  // for `iters` inbound from the pair, so stream-sync == full-round completion
  // on this rank (matches UCCL's wait-for-peer-red_add completion).
  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = std::chrono::steady_clock::now();
  nccl_gin_paired_kernel<<<NCCL_CTAS, NCCL_TPB, 0, stream>>>(sendwin, recvwin, bytes, peer, iters, devComm);
  CUDA_OK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::steady_clock::now();
  MPI_Barrier(MPI_COMM_WORLD);
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
#endif  // UCCL_GIN_WITH_NCCL_GIN

// ===========================================================================
// UCCL-GIN path
// ===========================================================================
//
// Kernel: a single warp streams `iters` gin.put<Rail>()s of `bytes` to the remote
// pair through handle::UCCLGin, then gin.red_add_rel<Rail>() to the peer counter.
// Layout of the registered window buffer:
//   [ send region (max_bytes) | recv region (max_bytes) ]  (offset 0 / max_bytes)
// red_add counter lives in the proxy atomic buffer at a fixed offset.

// Mirrors the NCCL-GIN kernel's call site, but through handle::UCCLGin:
// same gin.put<Rail>(...) / gin.red_add_rel<Rail>(...) shape, UCCL backend.
__global__ void uccl_gin_paired_kernel(uccl_gin::UCCLGinResources res, int peer,
                                       void* send_ptr, void* recv_ptr, uint32_t bytes,
                                       int iters, void* counter_ptr, int num_lanes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uccl_gin::UCCLGin gin(res);  // the exact handle DeepEP kernels will use
  // Fan payload puts across ALL D2H lanes (round-robin) so the stream drives
  // every bound NIC, not just lane 0's single NIC.
  for (int it = 0; it < iters; ++it)
    gin.put<ncclTeamTagRail>(recv_ptr, send_ptr, (int)bytes, peer, /*lane_hint=*/it);
  // One ordered red_add per lane into a distinct counter slot. The proxy holds
  // each lane's atomic until that lane's payload WRITEs complete (payload-before-
  // tail), so the peer observing all `num_lanes` slots == every lane landed.
  for (int L = 0; L < num_lanes; ++L)
    gin.red_add_rel<ncclTeamTagRail>((char*)counter_ptr + (size_t)L * 8,
                                     /*delta=*/1, peer, /*lane_hint=*/L);
}

// Exercises the other two rail primitives on a single lane: put_tail_add (one
// WRITE that also piggybacks a +1 onto a receiver tail slot) repeated `iters`
// times, then quiet() to drain the lane. The piggyback count rides with its
// payload WRITE, so the peer's tail == iters implies all payloads landed.
__global__ void uccl_gin_tailadd_kernel(uccl_gin::UCCLGinResources res, int peer,
                                        void* send_ptr, void* recv_ptr, uint32_t bytes,
                                        int iters, uint32_t tail_off, int lane) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uccl_gin::UCCLGin gin(res);
  for (int it = 0; it < iters; ++it)
    gin.put_tail_add<ncclTeamTagRail>(recv_ptr, send_ptr, (int)bytes, peer,
                                      /*count_delta=*/1, tail_off, /*lane_hint=*/lane);
  gin.quiet(lane);  // must return once the proxy has consumed all of lane's cmds
}

static double run_uccl_gin(uccl_gin::Context& c, size_t bytes, int peer, int iters, int warmup,
                           cudaStream_t stream, size_t max_bytes) {
  // Symmetric pointers into the registered window: send region @0, recv region
  // @max_bytes; counter @ atomic buffer base. The handle converts to offsets.
  void* send_ptr = c.send_ptr();
  void* recv_ptr = c.recv_ptr();
  void* counter_ptr = c.counter_ptr();
  const int num_lanes = c.num_queues();          // one completion slot per lane
  auto* slots = reinterpret_cast<std::atomic<int64_t>*>(counter_ptr);
  auto reset_slots = [&]() {
    for (int L = 0; L < num_lanes; ++L) slots[L].store(0, std::memory_order_release);
  };
  auto wait_all = [&]() {
    using clock = std::chrono::steady_clock;
    auto start = clock::now();
    for (int L = 0; L < num_lanes; ++L) {
      int spins = 0;
      while (slots[L].load(std::memory_order_acquire) < 1) {
        if ((++spins & 0xFFFF) == 0) std::this_thread::yield();
        if (clock::now() - start > std::chrono::seconds(30)) {
          fprintf(stderr, "[UCCL-GIN] timeout waiting slot %d/%d (now=%ld)\n",
                  L, num_lanes, (long)slots[L].load(std::memory_order_relaxed));
          std::abort();
        }
      }
    }
  };

  // warmup
  reset_slots();
  MPI_Barrier(MPI_COMM_WORLD);
  uccl_gin_paired_kernel<<<1, 32, 0, stream>>>(c.resources(), peer, send_ptr, recv_ptr,
                                               (uint32_t)bytes, warmup, counter_ptr, num_lanes);
  CUDA_OK(cudaStreamSynchronize(stream));
  wait_all();
  MPI_Barrier(MPI_COMM_WORLD);

  reset_slots();
  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = std::chrono::steady_clock::now();
  uccl_gin_paired_kernel<<<1, 32, 0, stream>>>(c.resources(), peer, send_ptr, recv_ptr,
                                               (uint32_t)bytes, iters, counter_ptr, num_lanes);
  CUDA_OK(cudaStreamSynchronize(stream));
  wait_all();
  auto t1 = std::chrono::steady_clock::now();
  MPI_Barrier(MPI_COMM_WORLD);
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ===========================================================================
// Correctness + ordering (payload-before-tail) verification
// ===========================================================================
// Each rank fills its send region with a rank-tagged pattern word[i]=rank*P+i,
// poisons its recv region, transfers, waits for the completion signal, then
// checks recv == the PEER's pattern. Because recv is read only AFTER the
// completion signal is observed, a payload-before-tail violation (tail seen
// before data landed) would leave poison -> FAIL. So this validates both data
// integrity and the ordering guarantee.
__global__ void fill_pattern_kernel(int* p, size_t n, int rank) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = rank * 1000003 + (int)i;
}

static bool verify_recv(int* d_recv, size_t bytes, int peer) {
  size_t n = bytes / sizeof(int);
  std::vector<int> h(n);
  CUDA_OK(cudaMemcpy(h.data(), d_recv, n * sizeof(int), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; ++i)
    if (h[i] != peer * 1000003 + (int)i) return false;
  return true;
}

static bool verify_nccl(ncclComm_t comm, struct ncclDevComm devComm,
                        ncclWindow_t sendwin, ncclWindow_t recvwin,
                        void* d_send, void* d_recv, size_t bytes, int peer,
                        int rank, cudaStream_t stream) {
#if UCCL_GIN_WITH_NCCL_GIN
  (void)comm;
  size_t n = bytes / sizeof(int);
  fill_pattern_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>((int*)d_send, n, rank);
  CUDA_OK(cudaMemset(d_recv, 0xFF, bytes));
  CUDA_OK(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  const int iters = 4;
  nccl_gin_paired_kernel<<<NCCL_CTAS, NCCL_TPB, 0, stream>>>(sendwin, recvwin, bytes, peer, iters, devComm);
  CUDA_OK(cudaStreamSynchronize(stream));  // kernel waitSignal => peer's iters landed
  bool ok = verify_recv((int*)d_recv, bytes, peer);
  MPI_Barrier(MPI_COMM_WORLD);
  return ok;
#else
  (void)comm; (void)devComm; (void)sendwin; (void)recvwin; (void)d_send;
  (void)d_recv; (void)bytes; (void)peer; (void)rank; (void)stream;
  return true;
#endif
}

static bool verify_uccl(uccl_gin::Context& c, size_t bytes, int peer, int rank,
                        cudaStream_t stream, size_t max_bytes) {
  (void)max_bytes;
  int* send = (int*)c.send_ptr();
  int* recv = (int*)c.recv_ptr();
  size_t n = bytes / sizeof(int);
  fill_pattern_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(send, n, rank);
  CUDA_OK(cudaMemset(recv, 0xFF, bytes));
  CUDA_OK(cudaStreamSynchronize(stream));
  const int num_lanes = c.num_queues();
  void* counter_ptr = c.counter_ptr();
  auto* slots = reinterpret_cast<std::atomic<int64_t>*>(counter_ptr);
  for (int L = 0; L < num_lanes; ++L) slots[L].store(0, std::memory_order_release);
  MPI_Barrier(MPI_COMM_WORLD);
  const int iters = num_lanes > 4 ? num_lanes : 4;  // exercise every lane
  uccl_gin_paired_kernel<<<1, 32, 0, stream>>>(c.resources(), peer, send, recv,
                                               (uint32_t)bytes, iters, counter_ptr, num_lanes);
  CUDA_OK(cudaStreamSynchronize(stream));
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  for (int L = 0; L < num_lanes; ++L)
    while (slots[L].load(std::memory_order_acquire) < 1) {
      if (clock::now() - t0 > std::chrono::seconds(30)) {
        fprintf(stderr, "[verify] UCCL slot %d/%d timeout\n", L, num_lanes);
        return false;
      }
    }
  bool ok = verify_recv(recv, bytes, peer);  // checked AFTER completion => ordering
  MPI_Barrier(MPI_COMM_WORLD);
  return ok;
}

// Correctness for the remaining two rail primitives: put_tail_add (piggyback
// WRITE+count) and quiet (drain). Single lane, single tail slot. After the
// kernel returns (proves quiet() did not deadlock), wait for the peer tail slot
// to reach `iters`, then check recv == peer pattern (data read only after the
// tail count => payload-before-tail ordering) and tail == iters exactly.
static bool verify_uccl_tailadd(uccl_gin::Context& c, size_t bytes, int peer, int rank,
                                cudaStream_t stream, size_t max_bytes) {
  (void)max_bytes;
  int* send = (int*)c.send_ptr();
  int* recv = (int*)c.recv_ptr();
  size_t n = bytes / sizeof(int);
  fill_pattern_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(send, n, rank);
  CUDA_OK(cudaMemset(recv, 0xFF, bytes));
  CUDA_OK(cudaStreamSynchronize(stream));
  // Tail slot 1 (1-based; slot 0 is reserved so the piggyback fires under the
  // V1 atomic_offset>0 trigger).
  const uint32_t tail_off = 8;
  auto* counter = reinterpret_cast<std::atomic<int64_t>*>(
      (char*)c.counter_ptr() + tail_off);
  counter->store(0, std::memory_order_release);
  const int iters = 8;
  MPI_Barrier(MPI_COMM_WORLD);
  uccl_gin_tailadd_kernel<<<1, 32, 0, stream>>>(c.resources(), peer, send, recv,
                                                (uint32_t)bytes, iters, tail_off, /*lane=*/0);
  CUDA_OK(cudaStreamSynchronize(stream));  // returns only if quiet() drained, no deadlock
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  while (counter->load(std::memory_order_acquire) < iters) {
    if (clock::now() - t0 > std::chrono::seconds(30)) {
      fprintf(stderr, "[verify] put_tail_add tail timeout (got %ld want %d)\n",
              (long)counter->load(std::memory_order_relaxed), iters);
      return false;
    }
  }
  bool ok = verify_recv(recv, bytes, peer);  // data after tail => ordering
  long got = (long)counter->load(std::memory_order_acquire);
  if (got != iters) {
    fprintf(stderr, "[verify] put_tail_add tail count %ld != %d\n", got, iters);
    ok = false;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return ok;
}

// Tests NCCL-GIN-compatible quiet semantics: after quiet returns, the source
// buffer is safe to reuse. A piggyback tail attached to the original payload is
// used only to observe eventual remote completion; quiet itself does not
// promise remote visibility.
__global__ void uccl_gin_put_quiet_kernel(uccl_gin::UCCLGinResources res, int peer,
                                          void* send_ptr, void* recv_ptr, uint32_t bytes,
                                          int iters, uint32_t tail_off, int lane) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uccl_gin::UCCLGin gin(res);
  for (int it = 0; it < iters; ++it)
    gin.put_tail_add<ncclTeamTagRail>(recv_ptr, send_ptr, (int)bytes, peer,
                                      /*count_delta=*/1, tail_off,
                                      /*lane_hint=*/lane);
  gin.quiet(lane);
}

static bool verify_uccl_put_quiet(uccl_gin::Context& c, size_t bytes, int peer, int rank,
                                  cudaStream_t stream, size_t max_bytes) {
  (void)max_bytes;
  int* send = (int*)c.send_ptr();
  int* recv = (int*)c.recv_ptr();
  size_t n = bytes / sizeof(int);
  fill_pattern_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(send, n, rank);
  CUDA_OK(cudaMemset(recv, 0xFF, bytes));
  CUDA_OK(cudaStreamSynchronize(stream));
  const int lane = 0;
  const int iters = 4;
  const uint32_t tail_off = 64;
  auto* completion = reinterpret_cast<std::atomic<int64_t>*>(
      static_cast<char*>(c.counter_ptr()) + tail_off);
  completion->store(0, std::memory_order_release);
  MPI_Barrier(MPI_COMM_WORLD);
  uccl_gin_put_quiet_kernel<<<1, 32, 0, stream>>>(c.resources(), peer, send, recv,
                                                   (uint32_t)bytes, iters,
                                                   tail_off, lane);
  CUDA_OK(cudaStreamSynchronize(stream));

  // Reuse the source immediately after quiet. If quiet acknowledged before the
  // transport consumed the source, the remote payload can be corrupted.
  CUDA_OK(cudaMemsetAsync(send, 0xA5, bytes, stream));
  CUDA_OK(cudaStreamSynchronize(stream));

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  while (completion->load(std::memory_order_acquire) < iters) {
    if (clock::now() - t0 > std::chrono::seconds(30)) {
      fprintf(stderr, "[verify] put+quiet completion timeout (bytes=%zu)\n",
              bytes);
      return false;
    }
  }

  bool ok = verify_recv(recv, bytes, peer);
  MPI_Barrier(MPI_COMM_WORLD);
  if (!ok && rank == 0)
    fprintf(stderr,
            "[verify] put+quiet source-reuse FAIL: recv data mismatch "
            "(bytes=%zu)\n",
            bytes);
  return ok;
}

// Tests red_add_rel counter correctness: each lane posts `iters` ordered atomic
// adds of delta=1 to its own counter slot. The peer waits for all slots to reach
// exactly `iters`. This validates the counter add itself (no payload data involved)
// and the multi-lane dispatch of ordered atomics.
__global__ void uccl_gin_red_add_kernel(uccl_gin::UCCLGinResources res, int peer,
                                        void* counter_base, int iters, int num_lanes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uccl_gin::UCCLGin gin(res);
  for (int L = 0; L < num_lanes; ++L) {
    for (int it = 0; it < iters; ++it)
      gin.red_add_rel<ncclTeamTagRail>(
          (char*)counter_base + (size_t)L * 8, 1, peer, L);
  }
}

static bool verify_uccl_red_add(uccl_gin::Context& c, int peer, int rank,
                                int iters, cudaStream_t stream) {
  (void)rank; (void)stream;
  const int num_lanes = c.num_queues();
  void* counter_ptr = c.counter_ptr();
  auto* slots = reinterpret_cast<std::atomic<int64_t>*>(counter_ptr);
  for (int L = 0; L < num_lanes; ++L)
    slots[L].store(0, std::memory_order_release);
  MPI_Barrier(MPI_COMM_WORLD);

  uccl_gin_red_add_kernel<<<1, 32, 0, stream>>>(
      c.resources(), peer, counter_ptr, iters, num_lanes);
  CUDA_OK(cudaStreamSynchronize(stream));

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  for (int L = 0; L < num_lanes; ++L) {
    while (slots[L].load(std::memory_order_acquire) < iters) {
      if (clock::now() - t0 > std::chrono::seconds(30)) {
        fprintf(stderr, "[verify] red_add slot %d/%d timeout (got %ld want %d)\n",
                L, num_lanes, (long)slots[L].load(std::memory_order_relaxed), iters);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // After the barrier the counters are stable; verify exact values.
  for (int L = 0; L < num_lanes; ++L) {
    long v = (long)slots[L].load(std::memory_order_acquire);
    if (v != (long)iters) {
      fprintf(stderr, "[verify] red_add counter[%d] %ld != %d\n", L, v, iters);
      return false;
    }
  }
  return true;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  Args args = parse_args(argc, argv);

  // local_world: ranks per node (from env, else assume 8). MPI launch must place
  // contiguous ranks per node (mpirun -npernode 8).
  int local_world = atoi(getenv("LOCAL_WORLD_SIZE") ? getenv("LOCAL_WORLD_SIZE") : "8");
  int local_rank  = rank % local_world;
  int peer        = (rank + local_world) % world;   // remote pair
  CUDA_OK(cudaSetDevice(local_rank));
  cudaStream_t stream; CUDA_OK(cudaStreamCreate(&stream));
  size_t max_bytes = 0; for (auto s : args.sizes) max_bytes = std::max(max_bytes, s);

  if (rank == 0) {
    printf("uccl_gin_microbench world=%d local_world=%d iters=%d (paired-remote)\n",
           world, local_world, args.iters);
    printf("%-12s %-18s %-18s\n", "bytes", "NCCL-GIN GB/s", "UCCL-GIN GB/s");
  }

  // -------- NCCL-GIN setup (once, optional) --------
  ncclComm_t comm = nullptr; struct ncclDevComm devComm{};
  ncclWindow_t sendwin = nullptr, recvwin = nullptr; void* d_send = nullptr; void* d_recv = nullptr;
  if (args.run_nccl) {
#if UCCL_GIN_WITH_NCCL_GIN
    ncclUniqueId id;
    if (rank == 0) NCCL_OK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCL_OK(ncclCommInitRank(&comm, world, id, rank));
    NCCL_OK(ncclMemAlloc(&d_send, max_bytes));
    NCCL_OK(ncclMemAlloc(&d_recv, max_bytes));
    NCCL_OK(ncclCommWindowRegister(comm, d_send, max_bytes, &sendwin, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_OK(ncclCommWindowRegister(comm, d_recv, max_bytes, &recvwin, NCCL_WIN_COLL_SYMMETRIC));
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.worldGinBarrierCount = NCCL_CTAS;
    reqs.ginSignalCount = NCCL_CTAS;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCL_OK(ncclDevCommCreate(comm, &reqs, &devComm));
#else
    if (rank == 0) {
      fprintf(stderr, "NCCL-GIN path was requested but this binary was built with "
                      "UCCL_GIN_WITH_NCCL_GIN=0\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
#endif
  }

  // -------- UCCL-GIN setup (once) --------
  std::unique_ptr<uccl_gin::Context> uctx;
  if (args.run_uccl) {
    uccl_gin::ContextConfig cfg;
    cfg.rank = rank;
    cfg.world_size = world;
    cfg.local_world_size = local_world;
    cfg.max_message_bytes = max_bytes;
    cfg.ifname = args.ifname;
    uctx = std::make_unique<uccl_gin::Context>(cfg);
  }

  // -------- correctness + ordering pass (must pass before any BW number) -----
  {
    bool all_ok = true;
    // red_add counter-only test (no payload, size-independent)
    {
      int a_ok = 1;
      if (args.run_uccl && selected(args, "red-add"))
        a_ok = verify_uccl_red_add(*uctx, peer, rank, 16, stream) ? 1 : 0;
      int ga = 1;
      MPI_Allreduce(&a_ok, &ga, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_ok = all_ok && ga;
      if (rank == 0) {
        printf("UCCL-red_add counter: %s\n",
               selected(args, "red-add") ? (ga ? "PASS" : "FAIL") : "-");
      }
    }
    if (rank == 0) {
      printf("%-12s %-8s %-14s %-14s %-14s\n",
             "bytes", "NCCL", "UCCL-put/add", "UCCL-tail/q", "UCCL-put+q");
    }
    for (size_t bytes : args.sizes) {
      int n_ok = 1, u_ok = 1, t_ok = 1, q_ok = 1;
      if (args.run_nccl && selected(args, "put-add"))
        n_ok = verify_nccl(comm, devComm, sendwin, recvwin, d_send, d_recv,
                           bytes, peer, rank, stream) ? 1 : 0;
      if (args.run_uccl && selected(args, "put-add"))
        u_ok = verify_uccl(*uctx, bytes, peer, rank, stream, max_bytes) ? 1 : 0;
      if (args.run_uccl && selected(args, "tail-add"))
        t_ok = verify_uccl_tailadd(*uctx, bytes, peer, rank, stream, max_bytes) ? 1 : 0;
      if (args.run_uccl && selected(args, "quiet"))
        q_ok = verify_uccl_put_quiet(*uctx, bytes, peer, rank, stream, max_bytes) ? 1 : 0;
      int gn = 1, gu = 1, gt = 1, gq = 1;
      MPI_Allreduce(&n_ok, &gn, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&u_ok, &gu, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&t_ok, &gt, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&q_ok, &gq, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (rank == 0) {
        printf("%-12zu %-8s %-14s %-14s %-14s\n", bytes,
               args.run_nccl && selected(args, "put-add")
                   ? (gn ? "PASS" : "FAIL") : "-",
               args.run_uccl && selected(args, "put-add")
                   ? (gu ? "PASS" : "FAIL") : "-",
               args.run_uccl && selected(args, "tail-add")
                   ? (gt ? "PASS" : "FAIL") : "-",
               args.run_uccl && selected(args, "quiet")
                   ? (gq ? "PASS" : "FAIL") : "-");
        if (args.run_uccl && selected(args, "put-add"))
          printf("UCCL-put/add bytes=%zu: %s\n", bytes,
                 gu ? "PASS" : "FAIL");
        if (args.run_uccl && selected(args, "tail-add"))
          printf("UCCL-tail/q bytes=%zu: %s\n", bytes,
                 gt ? "PASS" : "FAIL");
        if (args.run_uccl && selected(args, "quiet"))
          printf("UCCL-put+q source-reuse bytes=%zu: %s\n", bytes,
                 gq ? "PASS" : "FAIL");
      }
      all_ok = all_ok && gn && gu && gt && gq;
    }
    if (!all_ok) {
      if (rank == 0) printf("CORRECTNESS FAILED -- not reporting BW\n");
      uctx.reset();
      MPI_Finalize();
      return 2;
    }
    if (rank == 0) printf("all correctness PASS\n\n");
  }

  if (args.correctness_only) {
    uctx.reset();
    if (args.run_nccl) {
#if UCCL_GIN_WITH_NCCL_GIN
      NCCL_OK(ncclDevCommDestroy(comm, &devComm));
      NCCL_OK(ncclCommWindowDeregister(comm, sendwin));
      NCCL_OK(ncclCommWindowDeregister(comm, recvwin));
      ncclMemFree(d_send); ncclMemFree(d_recv);
      NCCL_OK(ncclCommFinalize(comm)); NCCL_OK(ncclCommDestroy(comm));
#endif
    }
    CUDA_OK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return 0;
  }

  // -------- sweep --------
  for (size_t bytes : args.sizes) {
    double nccl_gbps = 0, uccl_gbps = 0;
    if (args.run_nccl) {
#if UCCL_GIN_WITH_NCCL_GIN
      double ms = run_nccl_gin(comm, devComm, sendwin, recvwin, d_send, d_recv,
                               bytes, peer, args.iters, args.warmup, stream);
      double per_rank = (double)bytes * args.iters / (ms * 1e-3) / 1e9;
      MPI_Reduce(rank==0?MPI_IN_PLACE:&per_rank, &per_rank, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      nccl_gbps = per_rank / world;  // avg per-rank
#endif
    }
    if (args.run_uccl) {
      double ms = run_uccl_gin(*uctx, bytes, peer, args.iters, args.warmup, stream, max_bytes);
      double per_rank = (double)bytes * args.iters / (ms * 1e-3) / 1e9;
      MPI_Reduce(rank==0?MPI_IN_PLACE:&per_rank, &per_rank, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      uccl_gbps = per_rank / world;
    }
    if (rank == 0)
      printf("%-12zu %-18.2f %-18.2f\n", bytes, nccl_gbps, uccl_gbps);
  }

  uctx.reset();
  if (args.run_nccl) {
#if UCCL_GIN_WITH_NCCL_GIN
    NCCL_OK(ncclDevCommDestroy(comm, &devComm));
    NCCL_OK(ncclCommWindowDeregister(comm, sendwin));
    NCCL_OK(ncclCommWindowDeregister(comm, recvwin));
    ncclMemFree(d_send); ncclMemFree(d_recv);
    NCCL_OK(ncclCommFinalize(comm)); NCCL_OK(ncclCommDestroy(comm));
#endif
  }
  CUDA_OK(cudaStreamDestroy(stream));
  MPI_Finalize();
  return 0;
}

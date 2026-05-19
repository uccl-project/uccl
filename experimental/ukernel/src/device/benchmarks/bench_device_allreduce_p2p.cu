#include "gpu_rt.h"
#include "c2d_fifo.h"
#include "fifo/fifo_util.hpp"
#include "persistent_kernel_ops.h"
#include "task.h"
#include "worker.h"
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

using namespace UKernel::Device;

// ── Config ─────────────────────────────────────────────────────────────────
struct BenchConfig {
  size_t data_bytes = 1024;
  int workers_per_rank = 1;
  int num_blocks = 1;
  int num_warmup = 5;
  int num_iters = 20;
  char bench[32] = "all";  // "allreduce", "alltoall", "all"
};

bool sm_measure_enabled = false;

// ── Helpers ────────────────────────────────────────────────────────────────
static void fill_buf(std::vector<float>& buf, float val) {
  for (auto& x : buf) x = val;
}
static uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── Task batch ──────────────────────────────────────────────────────────────
struct TaskBatch {
  uint32_t fifoId;
  std::vector<uint64_t> task_ids;
  std::vector<uint32_t> args_indices;
};

static void submit_reduce(TaskManager& tmgr, WorkerPool& pool,
                          void* src, void* dst, size_t bytes,
                          int src_rank, ReduceType rt,
                          uint32_t fifoId, TaskBatch& batch) {
  TaskArgs args{};
  args.src = src;
  args.dst = dst;
  args.bytes = bytes;
  args.src_rank = src_rank;
  args.dst_rank = 0;
  args.set_red_type(rt);
  Task t = tmgr.create_task(args, (rt == ReduceType::None) ? TaskType::CollCopy
                                                             : TaskType::CollReduce,
                            DataType::Fp32, 0);
  batch.args_indices.push_back(t.args_index());
  batch.task_ids.push_back(pool.enqueue(t, fifoId));
  batch.fifoId = fifoId;
}

static void submit_copy(TaskManager& tmgr, WorkerPool& pool,
                        void* src, void* dst, size_t bytes,
                        int src_rank, int dst_rank,
                        uint32_t fifoId, TaskBatch& batch) {
  TaskArgs args{};
  args.src = src;
  args.dst = dst;
  args.bytes = bytes;
  args.src_rank = src_rank;
  args.dst_rank = dst_rank;
  Task t = tmgr.create_task(args, TaskType::CollCopy, DataType::Fp32, 0);
  batch.args_indices.push_back(t.args_index());
  batch.task_ids.push_back(pool.enqueue(t, fifoId));
  batch.fifoId = fifoId;
}

static void sync_all_batches(TaskManager& tmgr, WorkerPool& pool,
                             std::vector<TaskBatch>& batches) {
  for (auto& b : batches) {
    for (auto tid : b.task_ids) pool.sync(tid, b.fifoId);
    for (auto idx : b.args_indices) tmgr.free_task_args(idx);
    b.task_ids.clear();
    b.args_indices.clear();
  }
}

// ── SM measurement ─────────────────────────────────────────────────────────
struct SmBuf {
  SmTimestamp* d_ts = nullptr;
  uint32_t* d_count = nullptr;
};

static SmBuf sm_init_buffer(uint32_t max_tasks) {
  SmBuf buf;
  GPU_RT_CHECK(gpuMalloc(&buf.d_ts, sizeof(SmTimestamp) * max_tasks));
  GPU_RT_CHECK(gpuMalloc(&buf.d_count, sizeof(uint32_t)));
  GPU_RT_CHECK(gpuMemset(buf.d_count, 0, sizeof(uint32_t)));
  return buf;
}

static std::vector<SmTimestamp> sm_read(SmBuf& buf) {
  std::vector<SmTimestamp> results;
  uint32_t count = 0;
  GPU_RT_CHECK(gpuMemcpy(&count, buf.d_count, sizeof(count),
                          gpuMemcpyDeviceToHost));
  results.resize(count);
  if (count > 0) {
    GPU_RT_CHECK(gpuMemcpy(results.data(), buf.d_ts,
                            sizeof(SmTimestamp) * count, gpuMemcpyDeviceToHost));
  }
  return results;
}

static void sm_free(SmBuf& buf) {
  if (buf.d_ts) gpuFree(buf.d_ts);
  if (buf.d_count) gpuFree(buf.d_count);
  buf.d_ts = nullptr;
  buf.d_count = nullptr;
}

static void sm_print(int rank, std::vector<SmTimestamp> const& ts) {
  uint64_t poll = 0, sync = 0, compute = 0, tail = 0;
  for (auto& s : ts) {
    poll    += s.t[1] - s.t[0];
    sync    += s.t[2] - s.t[1];
    compute += s.t[3] - s.t[2];
    tail    += s.t[4] - s.t[3];
  }
  uint64_t total = poll + sync + compute + tail;
  if (total == 0) return;
  auto pct = [&](uint64_t v) { return 100.0 * v / total; };
  printf("[rank %d] SM (%zu tasks): idle+poll=%.1f%% sync=%.1f%% compute=%.1f%% tail=%.1f%%  eff=%.1f%%\n",
         rank, ts.size(),
         pct(poll), pct(sync), pct(compute), pct(tail), pct(compute));
}

// ── P2P Allreduce ─────────────────────────────────────────────────────────
// Each rank r: copy(peer[0]→dst) + reduce(peer[1..K-1]→dst)
static double run_allreduce(TaskManager& tmgr, WorkerPool& pool,
                            int rank, int num_ranks,
                            std::vector<float*>& peer_bufs, float* d_dst,
                            size_t bytes, int num_workers,
                            BenchConfig const& cfg,
                            uint64_t& rr_counter) {
  // Warmup
  for (int iter = 0; iter < cfg.num_warmup; ++iter) {
    GPU_RT_CHECK(gpuMemset(d_dst, 0, bytes));
    std::vector<TaskBatch> batches(num_workers);
    for (int k = 0; k < num_ranks; ++k) {
      int wi = rr_counter++ % num_workers;
      ReduceType rt = (k == 0) ? ReduceType::None : ReduceType::Sum;
      submit_reduce(tmgr, pool, peer_bufs[k], d_dst, bytes, k, rt,
                    (uint32_t)wi, batches[wi]);
    }
    sync_all_batches(tmgr, pool, batches);
  }

  // Timed
  std::vector<double> lats;
  for (int iter = 0; iter < cfg.num_iters; ++iter) {
    GPU_RT_CHECK(gpuMemset(d_dst, 0, bytes));
    uint64_t t0 = now_ns();
    std::vector<TaskBatch> batches(num_workers);
    for (int k = 0; k < num_ranks; ++k) {
      int wi = rr_counter++ % num_workers;
      ReduceType rt = (k == 0) ? ReduceType::None : ReduceType::Sum;
      submit_reduce(tmgr, pool, peer_bufs[k], d_dst, bytes, k, rt,
                    (uint32_t)wi, batches[wi]);
    }
    sync_all_batches(tmgr, pool, batches);
    uint64_t t1 = now_ns();
    lats.push_back((t1 - t0) / 1000.0);
  }

  std::sort(lats.begin(), lats.end());
  double avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
  printf("[rank %d] Latency (us): min=%.1f avg=%.1f p50=%.1f p99=%.1f max=%.1f\n",
         rank, lats.front(), avg, lats[lats.size()/2],
         lats[lats.size()*99/100], lats.back());
  double bw = (bytes * num_ranks / 1e9) / (avg / 1e6);
  printf("[rank %d] BW: %.2f GB/s\n", rank, bw);
  return avg;
}

// ── P2P Alltoall ──────────────────────────────────────────────────────────
// Each rank r: for s in 0..K-1, copy(peer[s] + r*chunk → local_ws + s*chunk)
static double run_alltoall(TaskManager& tmgr, WorkerPool& pool,
                           int rank, int num_ranks,
                           std::vector<float*>& peer_bufs, float* d_local_ws,
                           size_t data_bytes, int num_workers,
                           BenchConfig const& cfg,
                           uint64_t& rr_counter) {
  size_t chunk = data_bytes;

  // Warmup
  for (int iter = 0; iter < cfg.num_warmup; ++iter) {
    std::vector<TaskBatch> batches(num_workers);
    for (int s = 0; s < num_ranks; ++s) {
      int wi = rr_counter++ % num_workers;
      char* src = reinterpret_cast<char*>(peer_bufs[s]) + rank * chunk;
      char* dst = reinterpret_cast<char*>(d_local_ws) + s * chunk;
      submit_copy(tmgr, pool, src, dst, chunk, s, rank,
                  (uint32_t)wi, batches[wi]);
    }
    sync_all_batches(tmgr, pool, batches);
  }

  // Timed
  std::vector<double> lats;
  for (int iter = 0; iter < cfg.num_iters; ++iter) {
    uint64_t t0 = now_ns();
    std::vector<TaskBatch> batches(num_workers);
    for (int s = 0; s < num_ranks; ++s) {
      int wi = rr_counter++ % num_workers;
      char* src = reinterpret_cast<char*>(peer_bufs[s]) + rank * chunk;
      char* dst = reinterpret_cast<char*>(d_local_ws) + s * chunk;
      submit_copy(tmgr, pool, src, dst, chunk, s, rank,
                  (uint32_t)wi, batches[wi]);
    }
    sync_all_batches(tmgr, pool, batches);
    uint64_t t1 = now_ns();
    lats.push_back((t1 - t0) / 1000.0);
  }

  std::sort(lats.begin(), lats.end());
  double avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
  printf("[rank %d] Latency (us): min=%.1f avg=%.1f p50=%.1f p99=%.1f max=%.1f\n",
         rank, lats.front(), avg, lats[lats.size()/2],
         lats[lats.size()*99/100], lats.back());
  double bw = (data_bytes * num_ranks / 1e9) / (avg / 1e6);
  printf("[rank %d] BW: %.2f GB/s\n", rank, bw);
  return avg;
}

// ── main ───────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  BenchConfig cfg;

  for (int i = 1; i < argc; ++i) {
    char const* arg = argv[i];
    auto match = [&](char const* flag) -> bool {
      size_t len = strlen(flag);
      if (strncmp(arg, flag, len) != 0) return false;
      return arg[len] == '\0' || arg[len] == '=';
    };
    auto val = [&](int next_i) -> char const* {
      char const* eq = strchr(arg, '=');
      if (eq) return eq + 1;
      if (next_i < argc) return argv[next_i];
      return nullptr;
    };

    if (match("--bench")) {
      char const* v = val(i + 1);
      if (v) { strncpy(cfg.bench, v, sizeof(cfg.bench) - 1); if (!strchr(arg, '=')) ++i; }
    } else if (match("--workers")) {
      char const* v = val(i + 1);
      if (v) { cfg.workers_per_rank = atoi(v); if (!strchr(arg, '=')) ++i; }
    } else if (match("--bytes")) {
      char const* v = val(i + 1);
      if (v) { cfg.data_bytes = atol(v); if (!strchr(arg, '=')) ++i; }
    } else if (match("--blocks")) {
      char const* v = val(i + 1);
      if (v) { cfg.num_blocks = atoi(v); if (!strchr(arg, '=')) ++i; }
    } else if (match("--warmup")) {
      char const* v = val(i + 1);
      if (v) { cfg.num_warmup = atoi(v); if (!strchr(arg, '=')) ++i; }
    } else if (match("--iters")) {
      char const* v = val(i + 1);
      if (v) { cfg.num_iters = atoi(v); if (!strchr(arg, '=')) ++i; }
    } else if (match("--sm")) {
      sm_measure_enabled = true;
    }
  }

  int K = num_ranks;
  int W = cfg.workers_per_rank;

  if (rank == 0) {
    printf("=== P2P %s Bench ===\n", cfg.bench);
    printf("  Ranks: %d | Workers/rank: %d | Data/rank: %zu B | Blocks: %d | Warmup: %d | Iters: %d | SM: %s\n",
           K, W, cfg.data_bytes, cfg.num_blocks,
           cfg.num_warmup, cfg.num_iters, sm_measure_enabled ? "on" : "off");
  }

  // ── GPU & buffers ──────────────────────────────────────────────────────
  int num_gpus = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&num_gpus));
  int dev = rank % num_gpus;
  GPU_RT_CHECK(gpuSetDevice(dev));
  if (rank == 0) printf("[init] %d GPUs detected\n", num_gpus);

  // Enable P2P access between all GPUs
  for (int i = 0; i < num_gpus; ++i) {
    if (i == dev) continue;
    int can_access = 0;
    gpuDeviceCanAccessPeer(&can_access, dev, i);
    if (can_access) {
      gpuDeviceEnablePeerAccess(i, 0);
    } else {
      fprintf(stderr, "[rank %d] P2P access to GPU %d not available\n", rank, i);
    }
  }

  size_t buf_bytes = cfg.data_bytes * K;  // alltoall needs K * data_bytes
  float* d_local_buf = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d_local_buf, buf_bytes));

  // Fill with rank-specific data
  {
    std::vector<float> host(buf_bytes / sizeof(float));
    fill_buf(host, static_cast<float>((rank + 1) * 100));
    GPU_RT_CHECK(gpuMemcpy(d_local_buf, host.data(), buf_bytes, gpuMemcpyHostToDevice));
  }

  // ── IPC handle exchange via MPI ─────────────────────────────────────────
  cudaIpcMemHandle_t local_handle;
  GPU_RT_CHECK(cudaIpcGetMemHandle(&local_handle, d_local_buf));

  std::vector<cudaIpcMemHandle_t> all_handles(K);
  MPI_Allgather(&local_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                all_handles.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                MPI_COMM_WORLD);

  std::vector<float*> peer_bufs(K);
  peer_bufs[rank] = d_local_buf;
  for (int i = 0; i < K; ++i) {
    if (i == rank) continue;
    GPU_RT_CHECK(cudaIpcOpenMemHandle((void**)&peer_bufs[i], all_handles[i],
                                      cudaIpcMemLazyEnablePeerAccess));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) printf("[init] IPC handles exchanged, P2P ready\n");

  // ── Worker pool ─────────────────────────────────────────────────────────
  TaskManager::instance().init(4096 * W);
  WorkerPool::Config wcfg;
  wcfg.numMaxWorkers = std::max(8, W);
  wcfg.threadsPerBlock = 256;
  wcfg.fifoCapacity = 64;
  WorkerPool pool(wcfg);

  uint32_t max_ts = (cfg.num_warmup + cfg.num_iters) * K * 2;
  std::vector<SmBuf> sm_bufs;
  if (sm_measure_enabled) sm_bufs.resize(W);

  for (int w = 0; w < W; ++w) {
    SmTimestamp* sm_ts = nullptr;
    uint32_t* sm_cnt = nullptr;
    if (sm_measure_enabled) {
      sm_bufs[w] = sm_init_buffer(max_ts);
      sm_ts = sm_bufs[w].d_ts;
      sm_cnt = sm_bufs[w].d_count;
    }
    if (!pool.createWorker((uint32_t)w, cfg.num_blocks, sm_ts, sm_cnt)) {
      fprintf(stderr, "[rank %d] FAIL: createWorker for fifo=%d\n", rank, w);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  float* d_allreduce_dst = nullptr;
  float* d_alltoall_ws = nullptr;
  if (strcmp(cfg.bench, "allreduce") == 0 || strcmp(cfg.bench, "all") == 0) {
    GPU_RT_CHECK(gpuMalloc(&d_allreduce_dst, cfg.data_bytes));
  }
  if (strcmp(cfg.bench, "alltoall") == 0 || strcmp(cfg.bench, "all") == 0) {
    GPU_RT_CHECK(gpuMalloc(&d_alltoall_ws, buf_bytes));
  }

  uint64_t rr_counter = 0;

  // ── Allreduce ──────────────────────────────────────────────────────────
  if (strcmp(cfg.bench, "allreduce") == 0 || strcmp(cfg.bench, "all") == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("\n=== P2P Allreduce ===\n");
    double lat = run_allreduce(TaskManager::instance(), pool,
                               rank, K, peer_bufs, d_allreduce_dst,
                               cfg.data_bytes, W, cfg, rr_counter);
    (void)lat;
  }

  // ── Alltoall ───────────────────────────────────────────────────────────
  if (strcmp(cfg.bench, "alltoall") == 0 || strcmp(cfg.bench, "all") == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("\n=== P2P Alltoall ===\n");
    double lat = run_alltoall(TaskManager::instance(), pool,
                              rank, K, peer_bufs, d_alltoall_ws,
                              cfg.data_bytes, W, cfg, rr_counter);
    (void)lat;
  }

  // ── Final barrier + cleanup ────────────────────────────────────────────
  MPI_Barrier(MPI_COMM_WORLD);
  pool.shutdown_all();

  if (sm_measure_enabled) {
    for (int w = 0; w < W; ++w) {
      auto ts = sm_read(sm_bufs[w]);
      if (!ts.empty()) sm_print(rank, ts);
    }
  }

  for (int i = 0; i < K; ++i) {
    if (i != rank) cudaIpcCloseMemHandle(peer_bufs[i]);
  }
  if (d_allreduce_dst) gpuFree(d_allreduce_dst);
  if (d_alltoall_ws) gpuFree(d_alltoall_ws);
  gpuFree(d_local_buf);
  for (int w = 0; w < W; ++w) sm_free(sm_bufs[w]);
  TaskManager::instance().release();

  MPI_Finalize();
  return 0;
}

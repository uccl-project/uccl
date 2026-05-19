#include "gpu_rt.h"
#include "c2d_fifo.h"
#include "fifo/fifo_util.hpp"
#include "persistent_kernel_ops.h"
#include "task.h"
#include "worker.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

using namespace UKernel::Device;

// ── GPU SM timestamp buffer (written by persistent kernel thread 0) ───────
SmTimestamp* d_sm_ts_host = nullptr;
uint32_t* d_sm_count_host = nullptr;
bool sm_measure_enabled = false;

// ── Config ─────────────────────────────────────────────────────────────────
struct BenchConfig {
  int num_nodes = 2;
  size_t data_bytes = 1024;
  int num_blocks = 1;
  int num_warmup = 5;
  int num_iters = 20;
};

// ── Helpers ────────────────────────────────────────────────────────────────
static void fill_buf(std::vector<float>& buf, float val) {
  for (auto& x : buf) x = val;
}
static uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── Submit + sync helper, tracks args_idx ──────────────────────────────────
struct TaskBatch {
  std::vector<uint64_t> task_ids;
  std::vector<uint32_t> args_indices;
};

static void submit_reduce(TaskManager& tmgr, WorkerPool& pool,
                          void* src, void* dst, size_t bytes,
                          int src_rank, ReduceType rt, TaskBatch& batch) {
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
  batch.task_ids.push_back(pool.enqueue(t, 0));
}

static void submit_copy(TaskManager& tmgr, WorkerPool& pool,
                        void* src, void* dst, size_t bytes,
                        int src_rank, int dst_rank, TaskBatch& batch) {
  TaskArgs args{};
  args.src = src;
  args.dst = dst;
  args.bytes = bytes;
  args.src_rank = src_rank;
  args.dst_rank = dst_rank;

  Task t = tmgr.create_task(args, TaskType::CollCopy, DataType::Fp32, 0);
  batch.args_indices.push_back(t.args_index());
  batch.task_ids.push_back(pool.enqueue(t, 0));
}

static void sync_and_free(TaskManager& tmgr, WorkerPool& pool, TaskBatch& batch) {
  for (auto tid : batch.task_ids) pool.sync(tid, 0);
  for (auto idx : batch.args_indices) tmgr.free_task_args(idx);
  batch.task_ids.clear();
  batch.args_indices.clear();
}

// ── SM measurement ─────────────────────────────────────────────────────────
static void init_sm_measure(uint32_t max_tasks) {
  sm_measure_enabled = true;
  GPU_RT_CHECK(gpuMalloc(&d_sm_ts_host, sizeof(SmTimestamp) * max_tasks));
  GPU_RT_CHECK(gpuMalloc(&d_sm_count_host, sizeof(uint32_t)));

  UKernel::Device::set_sm_device_pointers(d_sm_ts_host, d_sm_count_host);
  GPU_RT_CHECK(gpuMemset(d_sm_count_host, 0, sizeof(uint32_t)));
}

static void read_sm_results(std::vector<SmTimestamp>& results) {
  if (!sm_measure_enabled) return;
  uint32_t count = 0;
  GPU_RT_CHECK(gpuMemcpy(&count, d_sm_count_host, sizeof(count),
                          gpuMemcpyDeviceToHost));
  fprintf(stderr, "  [SM] raw count: %u\n", count);
  results.resize(count);
  if (count > 0) {
    GPU_RT_CHECK(gpuMemcpy(results.data(), d_sm_ts_host,
                            sizeof(SmTimestamp) * count, gpuMemcpyDeviceToHost));
  }
}

static void print_sm_stats(std::vector<SmTimestamp> const& ts) {
  if (ts.empty()) return;
  uint64_t total_poll = 0, total_sync = 0, total_disp = 0, total_tail = 0;
  for (auto& s : ts) {
    total_poll += s.t[1] - s.t[0];   // T1-T0: poll wait
    total_sync += s.t[2] - s.t[1];   // T2-T1: __syncthreads
    total_disp += s.t[3] - s.t[2];   // T3-T2: dispatch (compute)
    total_tail += s.t[4] - s.t[3];   // T4-T3: tail/fence
  }
  uint64_t total = total_poll + total_sync + total_disp + total_tail;
  if (total == 0) return;

  auto pct = [&](uint64_t v) { return 100.0 * v / total; };
  printf("\n  SM Occupancy (clock64 cycles):\n");
  printf("    Poll wait:  %10lu  (%.1f%%)\n", (unsigned long)total_poll, pct(total_poll));
  printf("    Sync:       %10lu  (%.1f%%)\n", (unsigned long)total_sync, pct(total_sync));
  printf("    Compute:    %10lu  (%.1f%%)\n", (unsigned long)total_disp, pct(total_disp));
  printf("    Tail/fence: %10lu  (%.1f%%)\n", (unsigned long)total_tail, pct(total_tail));
  printf("    SM efficiency: %.1f%%\n", pct(total_disp));
}

// ── Allreduce: seq CollReduce into dst ─────────────────────────────────────
static double run_allreduce(TaskManager& tmgr, WorkerPool& pool,
                            std::vector<float*>& d_bufs, void* d_dst,
                            size_t bytes, int num_nodes, BenchConfig const& cfg) {
  for (int iter = 0; iter < cfg.num_warmup; ++iter) {
    fprintf(stderr, "  [warmup %d] memset...\n", iter);
    GPU_RT_CHECK(gpuMemset(d_dst, 0, bytes));
    TaskBatch batch;
    fprintf(stderr, "  [warmup %d] submit...\n", iter);
    submit_reduce(tmgr, pool, d_bufs[0], d_dst, bytes, 0, ReduceType::None, batch);
    for (int n = 1; n < num_nodes; ++n)
      submit_reduce(tmgr, pool, d_bufs[n], d_dst, bytes, n, ReduceType::Sum, batch);
    fprintf(stderr, "  [warmup %d] sync (tid=%llu, %llu)...\n",
            iter, (unsigned long long)batch.task_ids[0],
            (unsigned long long)batch.task_ids[1]);
    sync_and_free(tmgr, pool, batch);
    fprintf(stderr, "  [warmup %d] done\n", iter);
  }

  std::vector<double> lats;
  for (int iter = 0; iter < cfg.num_iters; ++iter) {
    GPU_RT_CHECK(gpuMemset(d_dst, 0, bytes));
    uint64_t t0 = now_ns();
    TaskBatch batch;
    submit_reduce(tmgr, pool, d_bufs[0], d_dst, bytes, 0, ReduceType::None, batch);
    for (int n = 1; n < num_nodes; ++n)
      submit_reduce(tmgr, pool, d_bufs[n], d_dst, bytes, n, ReduceType::Sum, batch);
    sync_and_free(tmgr, pool, batch);
    uint64_t t1 = now_ns();
    lats.push_back((t1 - t0) / 1000.0);
  }
  std::sort(lats.begin(), lats.end());
  double avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
  printf("  Latency (us): min=%.1f avg=%.1f p50=%.1f p99=%.1f max=%.1f\n",
         lats.front(), avg, lats[lats.size()/2],
         lats[lats.size()*99/100], lats.back());
  double bw = (bytes * num_nodes / 1e9) / (avg / 1e6);
  printf("  BW: %.2f GB/s\n", bw);
  return avg;
}

// ── Alltoall: N×N CollCopy ─────────────────────────────────────────────────
static double run_alltoall(TaskManager& tmgr, WorkerPool& pool,
                           std::vector<float*>& d_bufs,
                           size_t bytes_per_node, int num_nodes,
                           BenchConfig const& cfg) {
  size_t chunk = bytes_per_node / num_nodes;

  for (int iter = 0; iter < cfg.num_warmup; ++iter) {
    TaskBatch batch;
    for (int s = 0; s < num_nodes; ++s) {
      for (int d = 0; d < num_nodes; ++d) {
        char* src = reinterpret_cast<char*>(d_bufs[s]) + d * chunk;
        char* dst = reinterpret_cast<char*>(d_bufs[d]) + s * chunk;
        submit_copy(tmgr, pool, src, dst, chunk, s, d, batch);
      }
    }
    sync_and_free(tmgr, pool, batch);
  }

  std::vector<double> lats;
  for (int iter = 0; iter < cfg.num_iters; ++iter) {
    uint64_t t0 = now_ns();
    TaskBatch batch;
    for (int s = 0; s < num_nodes; ++s) {
      for (int d = 0; d < num_nodes; ++d) {
        char* src = reinterpret_cast<char*>(d_bufs[s]) + d * chunk;
        char* dst = reinterpret_cast<char*>(d_bufs[d]) + s * chunk;
        submit_copy(tmgr, pool, src, dst, chunk, s, d, batch);
      }
    }
    sync_and_free(tmgr, pool, batch);
    uint64_t t1 = now_ns();
    lats.push_back((t1 - t0) / 1000.0);
  }
  std::sort(lats.begin(), lats.end());
  double avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
  size_t total_bytes = bytes_per_node * num_nodes;  // each node's full data
  printf("  Latency (us): min=%.1f avg=%.1f p50=%.1f p99=%.1f max=%.1f\n",
         lats.front(), avg, lats[lats.size()/2],
         lats[lats.size()*99/100], lats.back());
  double bw = (total_bytes / 1e9) / (avg / 1e6);
  printf("  BW: %.2f GB/s\n", bw);
  return avg;
}

// ── main ───────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  BenchConfig cfg;

  // Parse: support both --key value and --key=value
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

    if (match("--nodes")) {
      char const* v = val(i + 1);
      if (v) { cfg.num_nodes = atoi(v); if (!strchr(arg, '=')) ++i; }
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

  printf("=== Device Allreduce/Alltoall Bench ===\n");
  printf("  Nodes: %d | Data/node: %zu B | Blocks: %d | Warmup: %d | Iters: %d | SM: %s\n",
         cfg.num_nodes, cfg.data_bytes, cfg.num_blocks,
         cfg.num_warmup, cfg.num_iters, sm_measure_enabled ? "on" : "off");

  GPU_RT_CHECK(gpuSetDevice(0));
  TaskManager::instance().init(4096);
  printf("[init] TaskManager ready\n");

  WorkerPool::Config wcfg;
  wcfg.numMaxWorkers = 8;
  wcfg.threadsPerBlock = 256;
  wcfg.fifoCapacity = 64;
  WorkerPool pool(wcfg);
  if (!pool.createWorker(0, cfg.num_blocks)) {
    fprintf(stderr, "FAIL: createWorker\n"); return 1;
  }
  printf("[init] Worker: fifo=0 blocks=%d\n", cfg.num_blocks);

  // SM measurement
  if (sm_measure_enabled) {
    uint32_t max_ts = (cfg.num_warmup + cfg.num_iters) * cfg.num_nodes * cfg.num_nodes * 2;
    init_sm_measure(max_ts);
    printf("[init] SM timestamp buffer: %u slots\n", max_ts);
  }

  // Buffers
  std::vector<float*> d_src_bufs(cfg.num_nodes);
  void* d_allreduce_dst = nullptr;
  size_t alltoall_buf_size = cfg.data_bytes * cfg.num_nodes;

  for (int n = 0; n < cfg.num_nodes; ++n) {
    GPU_RT_CHECK(gpuMalloc(&d_src_bufs[n], alltoall_buf_size));
    std::vector<float> host(alltoall_buf_size / sizeof(float));
    fill_buf(host, static_cast<float>((n + 1) * 100));
    GPU_RT_CHECK(gpuMemcpy(d_src_bufs[n], host.data(), alltoall_buf_size, gpuMemcpyHostToDevice));
  }
  GPU_RT_CHECK(gpuMalloc(&d_allreduce_dst, cfg.data_bytes));

  // ── Allreduce ──────────────────────────────────────────────────────────
  printf("\n=== Allreduce (nodes=%d) ===\n", cfg.num_nodes);
  double ar_lat = run_allreduce(TaskManager::instance(), pool,
                                d_src_bufs, d_allreduce_dst,
                                cfg.data_bytes, cfg.num_nodes, cfg);

  // ── Alltoall ───────────────────────────────────────────────────────────
  printf("\n=== Alltoall (nodes=%d) ===\n", cfg.num_nodes);
  double aa_lat = run_alltoall(TaskManager::instance(), pool,
                               d_src_bufs, cfg.data_bytes * cfg.num_nodes,
                               cfg.num_nodes, cfg);

  // ── Cleanup ──────────────────────────────────────────────────────────────
  fprintf(stderr, "[cleanup] shutdown pool...\n");
  pool.shutdown_all();

  // ── SM Results (must be AFTER shutdown — kernel must stop before D2H memcpy) ─
  if (sm_measure_enabled) {
    fprintf(stderr, "[cleanup] reading SM timestamps...\n");
    std::vector<SmTimestamp> ts;
    read_sm_results(ts);
    print_sm_stats(ts);
  }

  fprintf(stderr, "[cleanup] freeing buffers...\n");
  for (auto p : d_src_bufs) gpuFree(p);
  gpuFree(d_allreduce_dst);
  if (d_sm_ts_host) gpuFree(d_sm_ts_host);
  if (d_sm_count_host) gpuFree(d_sm_count_host);
  TaskManager::instance().release();
  printf("\nDone.\n");
  return 0;
}

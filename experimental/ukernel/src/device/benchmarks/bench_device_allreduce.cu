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

// ── Config ─────────────────────────────────────────────────────────────────
struct BenchConfig {
  int num_nodes = 2;
  int workers_per_node = 1;
  size_t data_bytes = 1024;
  int num_blocks = 1;
  int num_warmup = 5;
  int num_iters = 20;
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

// ── Task batch (per-worker, per-collective iteration) ──────────────────────
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
  uint32_t max_slots = 0;
};

static SmBuf sm_init_buffer(uint32_t max_tasks) {
  SmBuf buf;
  buf.max_slots = max_tasks;
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

struct SmAccum {
  uint64_t poll = 0, sync = 0, compute = 0, tail = 0;
  int tasks = 0;
  void add(std::vector<SmTimestamp> const& ts) {
    for (auto& s : ts) {
      poll    += s.t[1] - s.t[0];
      sync    += s.t[2] - s.t[1];
      compute += s.t[3] - s.t[2];
      tail    += s.t[4] - s.t[3];
    }
    tasks += (int)ts.size();
  }
  uint64_t total() const { return poll + sync + compute + tail; }
  bool empty() const { return tasks == 0; }
};

static void sm_print_aggregate(std::vector<SmBuf>& sm_bufs,
                               std::vector<std::vector<int>> const& node_workers,
                               BenchConfig const& cfg) {
  SmAccum agg;
  int active_workers = 0;
  for (int n = 0; n < cfg.num_nodes; ++n) {
    for (int wi : node_workers[n]) {
      auto ts = sm_read(sm_bufs[wi]);
      if (ts.empty()) continue;
      ++active_workers;
      SmAccum wa;
      wa.add(ts);
      uint64_t t = wa.total();
      auto pct = [&](uint64_t v) { return 100.0 * v / t; };
      printf("  Worker %2d [node=%d, fifo=%d, blocks=%d]: poll=%.1f%% sync=%.1f%% compute=%.1f%% tail=%.1f%%  eff=%.1f%%\n",
             wi, n, wi, cfg.num_blocks,
             pct(wa.poll), pct(wa.sync), pct(wa.compute), pct(wa.tail),
             pct(wa.compute));
      agg.poll += wa.poll; agg.sync += wa.sync;
      agg.compute += wa.compute; agg.tail += wa.tail;
      agg.tasks += wa.tasks;
    }
  }
  if (agg.empty()) return;
  uint64_t t = agg.total();
  auto pct = [&](uint64_t v) { return 100.0 * v / t; };
  printf("  Aggregate (%d workers x %d blocks, %d tasks): poll=%.1f%% sync=%.1f%% compute=%.1f%% tail=%.1f%%  eff=%.1f%%\n",
         active_workers, cfg.num_blocks, agg.tasks,
         pct(agg.poll), pct(agg.sync), pct(agg.compute), pct(agg.tail),
         pct(agg.compute));
}

// ── Allreduce (multi-worker, per-node task distribution) ──────────────────
// Node n: K tasks (1 copy + K-1 reduces) → round-robin to node's workers
static double run_allreduce(TaskManager& tmgr, WorkerPool& pool,
                            std::vector<float*>& d_src_bufs,
                            std::vector<std::vector<int>> const& node_workers,
                            std::vector<void*>& d_node_dst,
                            size_t bytes, int num_nodes, int num_workers,
                            std::vector<uint32_t> const& fifo_ids,
                            BenchConfig const& cfg,
                            uint64_t& rr_counter) {
  // Warmup
  for (int iter = 0; iter < cfg.num_warmup; ++iter) {
    for (int n = 0; n < num_nodes; ++n) {
      GPU_RT_CHECK(gpuMemset(d_node_dst[n], 0, bytes));
    }
    std::vector<TaskBatch> batches(num_workers);
    for (int n = 0; n < num_nodes; ++n) {
      auto& workers = node_workers[n];
      if (workers.empty()) continue;
      int wi = rr_counter++ % (int)workers.size();
      int w = workers[wi];
      submit_reduce(tmgr, pool, d_src_bufs[0], d_node_dst[n], bytes, 0,
                    ReduceType::None, fifo_ids[w], batches[w]);
      for (int k = 1; k < num_nodes; ++k) {
        wi = rr_counter++ % (int)workers.size();
        w = workers[wi];
        submit_reduce(tmgr, pool, d_src_bufs[k], d_node_dst[n], bytes, k,
                      ReduceType::Sum, fifo_ids[w], batches[w]);
      }
    }
    sync_all_batches(tmgr, pool, batches);
  }

  // Timed
  std::vector<double> lats;
  for (int iter = 0; iter < cfg.num_iters; ++iter) {
    for (int n = 0; n < num_nodes; ++n) {
      GPU_RT_CHECK(gpuMemset(d_node_dst[n], 0, bytes));
    }
    uint64_t t0 = now_ns();
    std::vector<TaskBatch> batches(num_workers);
    for (int n = 0; n < num_nodes; ++n) {
      auto& workers = node_workers[n];
      if (workers.empty()) continue;
      int wi = rr_counter++ % (int)workers.size();
      int w = workers[wi];
      submit_reduce(tmgr, pool, d_src_bufs[0], d_node_dst[n], bytes, 0,
                    ReduceType::None, fifo_ids[w], batches[w]);
      for (int k = 1; k < num_nodes; ++k) {
        wi = rr_counter++ % (int)workers.size();
        w = workers[wi];
        submit_reduce(tmgr, pool, d_src_bufs[k], d_node_dst[n], bytes, k,
                      ReduceType::Sum, fifo_ids[w], batches[w]);
      }
    }
    sync_all_batches(tmgr, pool, batches);
    uint64_t t1 = now_ns();
    lats.push_back((t1 - t0) / 1000.0);
  }

  std::sort(lats.begin(), lats.end());
  double avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
  printf("  Latency (us): min=%.1f avg=%.1f p50=%.1f p99=%.1f max=%.1f\n",
         lats.front(), avg, lats[lats.size()/2],
         lats[lats.size()*99/100], lats.back());
  double total_bytes = bytes * num_nodes;
  double bw = (total_bytes / 1e9) / (avg / 1e6);
  printf("  BW: %.2f GB/s  (%d nodes)\n", bw, num_nodes);
  for (int n = 0; n < num_nodes; ++n)
    printf("  Node %d: %zu workers\n", n, node_workers[n].size());
  return avg;
}

// ── Alltoall (multi-worker, per-node task distribution) ───────────────────
// Node n: K×K copies → round-robin to node's workers
// All workers of node n share d_node_ws[n] (write to disjoint offsets)
static double run_alltoall(TaskManager& tmgr, WorkerPool& pool,
                           std::vector<float*>& d_src_bufs,
                           std::vector<std::vector<int>> const& node_workers,
                           std::vector<void*>& d_node_ws,
                           size_t bytes_per_node, int num_nodes, int num_workers,
                           std::vector<uint32_t> const& fifo_ids,
                           BenchConfig const& cfg,
                           uint64_t& rr_counter) {
  size_t chunk = bytes_per_node / num_nodes;
  size_t row_bytes = chunk * num_nodes;

  // Warmup
  for (int iter = 0; iter < cfg.num_warmup; ++iter) {
    std::vector<TaskBatch> batches(num_workers);
    for (int n = 0; n < num_nodes; ++n) {
      auto& workers = node_workers[n];
      if (workers.empty()) continue;
      for (int s = 0; s < num_nodes; ++s) {
        for (int d = 0; d < num_nodes; ++d) {
          int wi = rr_counter++ % (int)workers.size();
          int w = workers[wi];
          char* src = reinterpret_cast<char*>(d_src_bufs[s]) + d * chunk;
          char* dst = reinterpret_cast<char*>(d_node_ws[n]) + d * row_bytes + s * chunk;
          submit_copy(tmgr, pool, src, dst, chunk, s, d, fifo_ids[w], batches[w]);
        }
      }
    }
    sync_all_batches(tmgr, pool, batches);
  }

  // Timed
  std::vector<double> lats;
  for (int iter = 0; iter < cfg.num_iters; ++iter) {
    uint64_t t0 = now_ns();
    std::vector<TaskBatch> batches(num_workers);
    for (int n = 0; n < num_nodes; ++n) {
      auto& workers = node_workers[n];
      if (workers.empty()) continue;
      for (int s = 0; s < num_nodes; ++s) {
        for (int d = 0; d < num_nodes; ++d) {
          int wi = rr_counter++ % (int)workers.size();
          int w = workers[wi];
          char* src = reinterpret_cast<char*>(d_src_bufs[s]) + d * chunk;
          char* dst = reinterpret_cast<char*>(d_node_ws[n]) + d * row_bytes + s * chunk;
          submit_copy(tmgr, pool, src, dst, chunk, s, d, fifo_ids[w], batches[w]);
        }
      }
    }
    sync_all_batches(tmgr, pool, batches);
    uint64_t t1 = now_ns();
    lats.push_back((t1 - t0) / 1000.0);
  }

  std::sort(lats.begin(), lats.end());
  double avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
  size_t total_bytes = bytes_per_node * num_nodes;
  printf("  Latency (us): min=%.1f avg=%.1f p50=%.1f p99=%.1f max=%.1f\n",
         lats.front(), avg, lats[lats.size()/2],
         lats[lats.size()*99/100], lats.back());
  double bw = (total_bytes / 1e9) / (avg / 1e6);
  printf("  BW: %.2f GB/s  (%d nodes)\n", bw, num_nodes);
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
    } else if (match("--workers")) {
      char const* v = val(i + 1);
      if (v) { cfg.workers_per_node = atoi(v); if (!strchr(arg, '=')) ++i; }
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

  int K = cfg.num_nodes;
  int WPN = cfg.workers_per_node;
  int M = K * WPN;

  printf("=== Device Allreduce/Alltoall Bench ===\n");
  printf("  Nodes: %d | Workers/node: %d | Total workers: %d | Data/node: %zu B | Blocks: %d | Warmup: %d | Iters: %d | SM: %s\n",
         K, WPN, M, cfg.data_bytes, cfg.num_blocks,
         cfg.num_warmup, cfg.num_iters, sm_measure_enabled ? "on" : "off");

  // Assign workers to nodes: worker w belongs to node w / WPN
  std::vector<std::vector<int>> node_workers(K);
  for (int w = 0; w < M; ++w) {
    node_workers[w / WPN].push_back(w);
  }

  GPU_RT_CHECK(gpuSetDevice(0));
  TaskManager::instance().init(4096 * M);
  printf("[init] TaskManager ready\n");

  WorkerPool::Config wcfg;
  wcfg.numMaxWorkers = std::max(8, M);
  wcfg.threadsPerBlock = 256;
  wcfg.fifoCapacity = 64;
  WorkerPool pool(wcfg);

  // Create M workers + FIFOs
  std::vector<uint32_t> fifo_ids(M);
  for (int w = 0; w < M; ++w) fifo_ids[w] = static_cast<uint32_t>(w);

  std::vector<SmBuf> sm_bufs;
  uint32_t max_ts = (cfg.num_warmup + cfg.num_iters) * K * K * 2;
  if (sm_measure_enabled) {
    sm_bufs.resize(M);
  }

  for (int w = 0; w < M; ++w) {
    int node_id = w / WPN;
    SmTimestamp* sm_ts = nullptr;
    uint32_t* sm_cnt = nullptr;
    if (sm_measure_enabled) {
      sm_bufs[w] = sm_init_buffer(max_ts);
      sm_ts = sm_bufs[w].d_ts;
      sm_cnt = sm_bufs[w].d_count;
    }
    if (!pool.createWorker(fifo_ids[w], cfg.num_blocks, sm_ts, sm_cnt)) {
      fprintf(stderr, "FAIL: createWorker for fifo=%u\n", fifo_ids[w]);
      return 1;
    }
    printf("[init] Worker %d: node=%d fifo=%u blocks=%d\n",
           w, node_id, fifo_ids[w], cfg.num_blocks);
  }
  if (sm_measure_enabled) {
    printf("[init] SM timestamp buffer: %u slots x %d workers\n", max_ts, M);
  }

  // Per-node buffers
  // src bufs: K bufs shared by all workers (read-only)
  // dst bufs: per-node workspace for allreduce result
  // ws bufs:  per-node workspace for alltoall output
  std::vector<float*> d_src_bufs(K);
  std::vector<void*> d_node_dst(K);
  std::vector<void*> d_node_ws(K);
  size_t src_buf_bytes = cfg.data_bytes * K;
  size_t ws_bytes = cfg.data_bytes * K * K;

  for (int n = 0; n < K; ++n) {
    GPU_RT_CHECK(gpuMalloc(&d_src_bufs[n], src_buf_bytes));
    std::vector<float> host(src_buf_bytes / sizeof(float));
    fill_buf(host, static_cast<float>((n + 1) * 100));
    GPU_RT_CHECK(gpuMemcpy(d_src_bufs[n], host.data(), src_buf_bytes, gpuMemcpyHostToDevice));

    GPU_RT_CHECK(gpuMalloc(&d_node_dst[n], cfg.data_bytes));
    GPU_RT_CHECK(gpuMalloc(&d_node_ws[n], ws_bytes));
  }

  // ── Allreduce ──────────────────────────────────────────────────────────
  printf("\n=== Allreduce (nodes=%d) ===\n", K);
  uint64_t rr_counter = 0;
  double ar_lat = run_allreduce(TaskManager::instance(), pool,
                                d_src_bufs, node_workers, d_node_dst,
                                cfg.data_bytes, K, M, fifo_ids, cfg, rr_counter);

  // ── Alltoall ───────────────────────────────────────────────────────────
  printf("\n=== Alltoall (nodes=%d) ===\n", K);
  double aa_lat = run_alltoall(TaskManager::instance(), pool,
                                d_src_bufs, node_workers, d_node_ws,
                                cfg.data_bytes * K, K, M, fifo_ids, cfg, rr_counter);

  // ── Cleanup ──────────────────────────────────────────────────────────────
  fprintf(stderr, "[cleanup] shutdown pool...\n");
  pool.shutdown_all();

  if (sm_measure_enabled) {
    fprintf(stderr, "[cleanup] reading SM timestamps...\n");
    sm_print_aggregate(sm_bufs, node_workers, cfg);
    for (int w = 0; w < M; ++w) sm_free(sm_bufs[w]);
  }

  fprintf(stderr, "[cleanup] freeing buffers...\n");
  for (auto p : d_src_bufs) gpuFree(p);
  for (int n = 0; n < K; ++n) {
    gpuFree(d_node_dst[n]);
    gpuFree(d_node_ws[n]);
  }
  TaskManager::instance().release();
  printf("\nDone.\n");
  return 0;
}

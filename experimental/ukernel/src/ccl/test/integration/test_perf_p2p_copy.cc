#include "backend/async_backend.h"
#include "backend/backend.h"
#include "backend/device_backend.h"
#include "coll_config.h"
#include "coll_types.h"
#include "gpu_rt.h"
#include "transport.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace UKernel;
using namespace UKernel::CCL;
using namespace UKernel::Transport;

namespace {

static constexpr int kWorldSize = 2;
static constexpr int kServerRank = 0;
static constexpr int kClientRank = 1;

std::string get_arg(int argc, char** argv, std::string const& name,
                    std::string const& def) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == name && i + 1 < argc) return argv[i + 1];
    std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
  }
  return def;
}

int get_int_arg(int argc, char** argv, std::string const& name, int def) {
  std::string s = get_arg(argc, argv, name, std::to_string(def));
  try { return std::stoi(s); } catch (...) { return def; }
}

std::shared_ptr<Communicator> make_communicator(
    int gpu, int rank, std::string const& exchanger_ip, int exchanger_port,
    PreferredTransport preferred) {
  auto cfg = std::make_shared<CommunicatorConfig>();
  cfg->exchanger_ip = exchanger_ip;
  cfg->exchanger_port = exchanger_port;
  cfg->local_id = rank;
  cfg->preferred_transport = preferred;
  return std::make_shared<Communicator>(gpu, rank, kWorldSize, cfg);
}

bool setup_bidirectional_peer(std::shared_ptr<Communicator> const& comm,
                              int rank, int peer_rank) {
  if (rank < peer_rank)
    return comm->connect(peer_rank) && comm->accept(peer_rank);
  return comm->accept(peer_rank) && comm->connect(peer_rank);
}

std::vector<size_t> make_size_scan() {
  std::vector<size_t> sizes;
  for (size_t b = 1024; b <= 1024ULL * 1024 * 1024; b *= 2)
    sizes.push_back(b);
  return sizes;
}

double elapsed_us(std::chrono::steady_clock::time_point const& start,
                  std::chrono::steady_clock::time_point const& end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

struct Result {
  std::string method;
  uint32_t threads_per_block;
  size_t bytes;
  double latency_us;
  double throughput_gbps;
};

}  // namespace

int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  std::string role = get_arg(argc, argv, "--role", "");
  if (role.empty()) {
    std::fprintf(stderr, "Usage: test_perf_p2p_copy --role=server|client "
                         "[--gpu GPU] [--exchanger-ip IP] [--exchanger-port PORT]\n"
                         "  Both processes must have the same CUDA_VISIBLE_DEVICES.\n");
    return 1;
  }

  int rank = (role == "server") ? kServerRank : kClientRank;
  int peer = (rank == 0) ? 1 : 0;
  int default_gpu = (role == "server") ? 0 : 1;
  int gpu = get_int_arg(argc, argv, "--gpu", default_gpu);
  std::string exchanger_ip =
      (role == "server") ? "0.0.0.0" : get_arg(argc, argv, "--exchanger-ip", "127.0.0.1");
  int exchanger_port = get_int_arg(argc, argv, "--exchanger-port", 6979);
  std::string transport_str = get_arg(argc, argv, "--transport", "auto");

  PreferredTransport preferred = PreferredTransport::Auto;
  if (transport_str == "ipc")        preferred = PreferredTransport::Ipc;
  else if (transport_str == "tcp")   preferred = PreferredTransport::Tcp;
  else if (transport_str == "rdma")  preferred = PreferredTransport::Rdma;

  std::printf("[p2p-perf] role=%s rank=%d gpu=%d preferred=%s\n",
              role.c_str(), rank, gpu, transport_str.c_str());

  // ── 1. Create Communicator ──────────────────────────────────────────
  auto comm = make_communicator(gpu, rank, exchanger_ip, exchanger_port, preferred);
  if (!setup_bidirectional_peer(comm, rank, peer)) {
    std::fprintf(stderr, "[p2p-perf] peer setup failed\n");
    return 1;
  }
  std::printf("[p2p-perf] peer setup done\n");
  fflush(stdout);

  // ── 2. Allocate local GPU memory ────────────────────────────────────
  GPU_RT_CHECK(gpuSetDevice(gpu));
  constexpr size_t kMaxBytes = 1024ULL * 1024 * 1024;
  void* d_local = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d_local, kMaxBytes));
  GPU_RT_CHECK(gpuMemset(d_local, 0xAB, kMaxBytes));

  // ── 3. Exchange buffers via IPC ─────────────────────────────────────
  uint32_t local_buf_id = (rank == 0) ? 0x1000 : 0x2000;
  uint32_t remote_buf_id = (rank == 0) ? 0x2000 : 0x1000;

  comm->reg_ipc(local_buf_id, d_local, kMaxBytes, true);
  if (!comm->wait_ipc(peer, remote_buf_id, 30000)) {
    std::fprintf(stderr, "[p2p-perf] wait_ipc failed for rank %d\n", rank);
    return 1;
  }

  void* d_remote = nullptr;
  int remote_dev = -1;
  if (!comm->try_resolve_remote_ipc_pointer(peer, remote_buf_id, 0, kMaxBytes,
                                            &d_remote, &remote_dev)) {
    std::fprintf(stderr,
                 "[p2p-perf] ERROR: Cannot resolve remote IPC.\n"
                 "  Peer access may not be supported between GPU %d and %d.\n",
                 gpu, (gpu == 0) ? 1 : 0);
    GPU_RT_CHECK(gpuFree(d_local));
    comm->dereg_ipc(local_buf_id);
    return 1;
  }
  std::printf("[p2p-perf] remote_dev=%d resolved\n", remote_dev);

  // ── 4. Enable peer access ───────────────────────────────────────────
  int can_access = 0;
  GPU_RT_CHECK(gpuDeviceCanAccessPeer(&can_access, gpu, remote_dev));
  if (!can_access) {
    std::fprintf(stderr, "[p2p-perf] Peer access NOT supported between GPU %d and %d\n",
                 gpu, remote_dev);
    GPU_RT_CHECK(gpuFree(d_local));
    comm->dereg_ipc(local_buf_id);
    return 1;
  }
  gpuError_t err = gpuDeviceEnablePeerAccess(remote_dev, 0);
  if (err != gpuSuccess && err != gpuErrorPeerAccessAlreadyEnabled)
    GPU_RT_CHECK(err);
  std::printf("[p2p-perf] peer access enabled: GPU %d -> %d\n", gpu, remote_dev);

  // ── 5. Warm-up ──────────────────────────────────────────────────────
  gpuStream_t warmup_stream;
  GPU_RT_CHECK(gpuStreamCreate(&warmup_stream));
  for (int i = 0; i < 10; ++i)
    GPU_RT_CHECK(gpuMemcpyPeerAsync(d_remote, remote_dev, d_local, gpu, 1024, warmup_stream));
  GPU_RT_CHECK(gpuStreamSynchronize(warmup_stream));
  gpuStreamDestroy(warmup_stream);

  // ── 6. Prepare size scan ────────────────────────────────────────────
  auto sizes = make_size_scan();
  std::vector<Result> results;

  // ── 7. Benchmark DeviceBackend (various blocks_per_worker) ──────────
  std::vector<uint32_t> all_blocks = {1, 2, 4, 8, 16, 32, 64, 128};
  int sm_count = 0;
  GPU_RT_CHECK(gpuDeviceGetAttribute(&sm_count, gpuDevAttrMultiProcessorCount, gpu));
  std::vector<uint32_t> blocks_configs;
  for (uint32_t b : all_blocks)
    if (b <= (uint32_t)sm_count) blocks_configs.push_back(b);
    else break;
  std::printf("[p2p-perf] SM count=%d, blocks_per_worker capped at %u\n",
              sm_count, blocks_configs.back());
  std::printf("[p2p-perf] Starting DeviceBackend benchmarks (%zu sizes x %zu blocks, ~1 min)...\n",
              sizes.size(), blocks_configs.size());

  for (uint32_t blocks_per_worker : blocks_configs) {
    std::printf("[p2p-perf]   DeviceBackend blocks_per_worker=%u ...\n", blocks_per_worker);
    DeviceBackendConfig dev_cfg;
    dev_cfg.task_capacity = 4096;
    dev_cfg.max_fifos = 1;
    dev_cfg.threads_per_block = 256;
    dev_cfg.blocks_per_worker = blocks_per_worker;
    dev_cfg.fifo_capacity = 64;  // enough for throughput batch (16)

    DeviceBackend dev_be(dev_cfg);
    BufSpec bufs[3] = {{d_local, kMaxBytes}, {d_remote, kMaxBytes}, {nullptr, 0}};
    dev_be.init(bufs);

    // Wrap with AsyncBackend (same pattern as SprayExecutor)
    AsyncBackend async(&dev_be, 2048, 2048);
    async.start();

    for (size_t bytes : sizes) {
      // ── Latency ──
      constexpr int kLatencyIters = 5;
      std::vector<double> latencies;
      latencies.reserve(kLatencyIters);

      for (int iter = 0; iter < kLatencyIters; ++iter) {
        CmdWithId cwi;
        cwi.cmd.kind = OpKind::Copy;
        cwi.cmd.bytes = static_cast<uint32_t>(bytes);
        cwi.cmd.src_buf = 1; cwi.cmd.dst_buf = 2;
        cwi.cmd.src_off = 0; cwi.cmd.dst_off = 0;
        cwi.cmd.src_peer = ~0u; cwi.cmd.dst_peer = ~0u;
        cwi.caller_id = static_cast<uint32_t>(iter);

        auto t0 = std::chrono::steady_clock::now();
        while (async.try_enqueue(&cwi, 1) == 0)
          std::this_thread::yield();
        uint32_t done = ~0u;
        while (async.try_drain(&done, 1) == 0)
          std::this_thread::yield();
        auto t1 = std::chrono::steady_clock::now();
        latencies.push_back(elapsed_us(t0, t1));
      }

      double avg_lat = 0;
      for (double l : latencies) avg_lat += l;
      avg_lat /= latencies.size();

      // ── Throughput ──
      constexpr int kBatchSize = 16;
      constexpr int kThroughputIters = 3;
      std::vector<double> throughputs;

      for (int iter = 0; iter < kThroughputIters; ++iter) {
        CmdWithId cwis[kBatchSize];
        for (int b = 0; b < kBatchSize; ++b) {
          cwis[b].cmd.kind = OpKind::Copy;
          cwis[b].cmd.bytes = static_cast<uint32_t>(bytes);
          cwis[b].cmd.src_buf = 1; cwis[b].cmd.dst_buf = 2;
          cwis[b].cmd.src_off = 0; cwis[b].cmd.dst_off = 0;
          cwis[b].cmd.src_peer = ~0u; cwis[b].cmd.dst_peer = ~0u;
          cwis[b].caller_id = static_cast<uint32_t>(b);
        }

        auto t0 = std::chrono::steady_clock::now();
        size_t enqueued = 0;
        while (enqueued < kBatchSize) {
          size_t n = async.try_enqueue(cwis + enqueued, kBatchSize - enqueued);
          enqueued += n;
          if (enqueued < kBatchSize) std::this_thread::yield();
        }

        size_t drained = 0;
        uint32_t done_buf[kBatchSize];
        while (drained < kBatchSize) {
          size_t n = async.try_drain(done_buf + drained, kBatchSize - drained);
          drained += n;
          if (drained < kBatchSize) std::this_thread::yield();
        }
        auto t1 = std::chrono::steady_clock::now();

        double total_bytes = static_cast<double>(bytes) * kBatchSize;
        double time_s = elapsed_us(t0, t1) / 1e6;
        double gbps = (total_bytes / time_s) / 1e9;
        throughputs.push_back(gbps);
      }

      double avg_tp = 0;
      for (double t : throughputs) avg_tp += t;
      avg_tp /= throughputs.size();

      results.push_back({"device_backend", blocks_per_worker, bytes, avg_lat, avg_tp});
    }
    async.stop();
    std::printf("[p2p-perf]   DeviceBackend blocks_per_worker=%u done.\n", blocks_per_worker);
  }

  // ── 8. Benchmark gpuMemcpyPeerAsync ─────────────────────────────────
  std::printf("[p2p-perf] Starting gpuMemcpyPeerAsync benchmarks...\n");
  gpuStream_t stream;
  GPU_RT_CHECK(gpuStreamCreate(&stream));

  for (size_t bytes : sizes) {
    constexpr int kLatencyIters = 5;
    std::vector<double> latencies;

    for (int iter = 0; iter < kLatencyIters; ++iter) {
      auto t0 = std::chrono::steady_clock::now();
      GPU_RT_CHECK(gpuMemcpyPeerAsync(d_remote, remote_dev, d_local, gpu, bytes, stream));
      GPU_RT_CHECK(gpuStreamSynchronize(stream));
      auto t1 = std::chrono::steady_clock::now();
      latencies.push_back(elapsed_us(t0, t1));
    }
    double avg_lat = 0;
    for (double l : latencies) avg_lat += l;
    avg_lat /= latencies.size();

    constexpr int kBatchSize = 16;
    constexpr int kThroughputIters = 3;
    std::vector<double> throughputs;

    for (int iter = 0; iter < kThroughputIters; ++iter) {
      auto t0 = std::chrono::steady_clock::now();
      for (int b = 0; b < kBatchSize; ++b) {
        GPU_RT_CHECK(gpuMemcpyPeerAsync(
            d_remote, remote_dev,
            d_local, gpu, bytes, stream));
      }
      GPU_RT_CHECK(gpuStreamSynchronize(stream));
      auto t1 = std::chrono::steady_clock::now();

      double total_bytes = static_cast<double>(bytes) * kBatchSize;
      double time_s = elapsed_us(t0, t1) / 1e6;
      double gbps = (total_bytes / time_s) / 1e9;
      throughputs.push_back(gbps);
    }
    double avg_tp = 0;
    for (double t : throughputs) avg_tp += t;
    avg_tp /= throughputs.size();

    results.push_back({"gpuMemcpyPeerAsync", 0, bytes, avg_lat, avg_tp});
  }
  std::printf("[p2p-perf] gpuMemcpyPeerAsync benchmarks done.\n");

  gpuStreamDestroy(stream);

  // ── 9. Benchmark comm->put (IPC adapter direct path) ──────────────────
  comm->reg_mr(local_buf_id, d_local, kMaxBytes, true);
  comm->reg_mr(remote_buf_id, d_local, kMaxBytes, true);
  // Re-resolve remote buffer to ensure MR is fresh
  comm->resolve_remote_buffer(peer, remote_buf_id);

  std::printf("[p2p-perf] Starting comm->put benchmarks...\n");

  for (size_t bytes : sizes) {
    // Latency: blocking put
    constexpr int kLatencyIters = 5;
    std::vector<double> latencies;
    for (int iter = 0; iter < kLatencyIters; ++iter) {
      auto t0 = std::chrono::steady_clock::now();
      comm->put(peer, local_buf_id, 0, remote_buf_id, 0, bytes);
      auto t1 = std::chrono::steady_clock::now();
      latencies.push_back(elapsed_us(t0, t1));
    }
    double avg_lat = 0;
    for (double l : latencies) avg_lat += l;
    avg_lat /= latencies.size();

    // Throughput: async put_async + try_complete
    constexpr int kBatchSize = 16;
    constexpr int kThroughputIters = 3;
    std::vector<double> throughputs;
    for (int iter = 0; iter < kThroughputIters; ++iter) {
      auto t0 = std::chrono::steady_clock::now();
      unsigned rids[kBatchSize];
      for (int b = 0; b < kBatchSize; ++b)
        rids[b] = comm->put_async(peer, local_buf_id, 0, remote_buf_id, 0, bytes);
      size_t drained = 0;
      while (drained < kBatchSize) {
        unsigned done[16];
        size_t n = comm->try_complete(done, 16);
        drained += n;
        if (drained < kBatchSize) std::this_thread::yield();
      }
      auto t1 = std::chrono::steady_clock::now();
      double total_bytes = static_cast<double>(bytes) * kBatchSize;
      double time_s = elapsed_us(t0, t1) / 1e6;
      double gbps = (total_bytes / time_s) / 1e9;
      throughputs.push_back(gbps);
    }
    double avg_tp = 0;
    for (double t : throughputs) avg_tp += t;
    avg_tp /= throughputs.size();

    results.push_back({"comm_put", 0, bytes, avg_lat, avg_tp});
  }
  std::printf("[p2p-perf] comm->put benchmarks done.\n");

  // ── 10. Output tables ─────────────────────────────────────────────────
  comm->barrier("results_barrier", 30000);
  if (rank != 0) {
    // Cleanup and exit, only rank 0 prints
    GPU_RT_CHECK(gpuFree(d_local));
    comm->dereg_ipc(local_buf_id);
    return 0;
  }

  auto fmt_size = [](size_t b) -> std::string {
    if (b < 1024) return std::to_string(b) + "B";
    if (b < 1024 * 1024) return std::to_string(b / 1024) + "KB";
    if (b < 1024 * 1024 * 1024) return std::to_string(b / (1024 * 1024)) + "MB";
    return std::to_string(b / (1024 * 1024 * 1024)) + "GB";
  };

  // Build column names from results
  std::vector<uint32_t> blocks_seen;
  for (auto& r : results) {
    if (r.method == "device_backend" && r.threads_per_block > 0) {
      bool found = false;
      for (auto b : blocks_seen) if (b == r.threads_per_block) { found = true; break; }
      if (!found) blocks_seen.push_back(r.threads_per_block);
    }
  }
  std::sort(blocks_seen.begin(), blocks_seen.end());
  std::vector<std::string> names;
  for (auto b : blocks_seen) names.push_back("b" + std::to_string(b));
  names.push_back("gpuMemcpyPeer");
  names.push_back("comm_put");
  std::vector<uint32_t> tb_values = blocks_seen;
  tb_values.push_back(0);
  tb_values.push_back(0);

  // Build lookup: (size, method_idx) -> latency, throughput
  struct M {
    double lat = 0;
    double tp = 0;
  };
  std::map<std::pair<size_t, size_t>, M> map;
  for (auto& r : results) {
    size_t mi = 0;
    if (r.method == "device_backend") {
      for (size_t i = 0; i < blocks_seen.size(); ++i)
        if (r.threads_per_block == blocks_seen[i]) { mi = i; break; }
    } else if (r.method == "gpuMemcpyPeerAsync") {
      mi = blocks_seen.size();
    } else {
      mi = blocks_seen.size() + 1;  // comm_put
    }
    map[{r.bytes, mi}] = {r.latency_us, r.throughput_gbps};
  }

  int w_name = 10;
  int w_num = 9;

  // ── Latency table ──
  std::printf("\n========== P2P Copy Latency (us) ==========\n");
  std::printf("  %-*s", w_name, "size");
  for (auto& n : names) std::printf("  %*s", w_num, n.c_str());
  std::printf("\n");

  for (auto& sz : sizes) {
    std::string label = fmt_size(sz);
    std::printf("  %-*s", w_name, label.c_str());
    for (size_t mi = 0; mi < names.size(); ++mi) {
      auto it = map.find({sz, mi});
      if (it != map.end())
        std::printf("  %*.2f", w_num, it->second.lat);
      else
        std::printf("  %*s", w_num, "N/A");
    }
    std::printf("\n");
  }

  // ── Throughput table ──
  std::printf("\n========== P2P Copy Throughput (GB/s) ==========\n");
  std::printf("  %-*s", w_name, "size");
  for (auto& n : names) std::printf("  %*s", w_num, n.c_str());
  std::printf("\n");

  for (auto& sz : sizes) {
    std::string label = fmt_size(sz);
    std::printf("  %-*s", w_name, label.c_str());
    for (size_t mi = 0; mi < names.size(); ++mi) {
      auto it = map.find({sz, mi});
      if (it != map.end())
        std::printf("  %*.2f", w_num, it->second.tp);
      else
        std::printf("  %*s", w_num, "N/A");
    }
    std::printf("\n");
  }
  std::printf("\n");
  fflush(stdout);

  // ── Cleanup ─────────────────────────────────────────────────────────
  GPU_RT_CHECK(gpuFree(d_local));
  comm->dereg_ipc(local_buf_id);

  return 0;
}

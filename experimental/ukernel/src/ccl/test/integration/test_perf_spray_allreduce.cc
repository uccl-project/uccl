// SprayExecutor collective throughput benchmark (AllReduce / AllToAll).

#include "coll_config.h"
#include "executor.h"
#include "gpu_rt.h"
#include "transport.h"
#include "util/uk_debug.h"
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

using namespace UKernel::CCL;

static std::string get_arg(int argc, char** argv, std::string const& name,
                           std::string const& def) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == name && i + 1 < argc) return argv[i + 1];
    std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
  }
  return def;
}
static int get_int_arg(int argc, char** argv, std::string const& name,
                       int def) {
  return std::stoi(get_arg(argc, argv, name, std::to_string(def)));
}

int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  std::string role = get_arg(argc, argv, "--role", "");
  if (role != "server" && role != "client") {
    std::fprintf(stderr,
                 "Usage: --role=server|client [--gpu GPU] "
                 "[--kind=allreduce|alltoall]\n");
    return 1;
  }
  int rank = (role == "server") ? 0 : 1;
  int gpu = get_int_arg(argc, argv, "--gpu", rank);
  std::string kind_str = get_arg(argc, argv, "--kind", "allreduce");
  // Signal aggregation factor: one Signal/WaitSignal per this many tiles.
  int sig_group = get_int_arg(argc, argv, "--sig-group", 1);
  // DeviceBackend parallelism knobs.
  int dev_fifos = get_int_arg(argc, argv, "--dev-fifos", 1);
  int dev_blocks = get_int_arg(argc, argv, "--dev-blocks", 1);
  CollKind coll_kind = (kind_str == "alltoall") ? CollKind::AllToAllPairwise
                                                : CollKind::AllReduceRing;
  bool inplace = (coll_kind == CollKind::AllToAllPairwise);
  std::string exchanger_ip = get_arg(argc, argv, "--exchanger-ip",
                                     (rank == 0) ? "0.0.0.0" : "127.0.0.1");
  int exchanger_port = get_int_arg(argc, argv, "--exchanger-port", 16998);

  GPU_RT_CHECK(gpuSetDevice(gpu));

  SprayExecutorConfig cfg;
  cfg.gpu_id = gpu;
  cfg.rank = rank;
  cfg.world_size = 2;
  cfg.exchanger_ip = exchanger_ip;
  cfg.exchanger_port = exchanger_port;
  cfg.local_id = gpu;
  cfg.max_device_fifos = dev_fifos;
  cfg.blocks_per_worker = (size_t)dev_blocks;
  auto ex = SprayExecutor::create(cfg);

  // Synchronous memset helper — avoids async gpuMemset racing with RDMA writes.
  auto sync_memset = [](void* ptr, int val, size_t bytes) {
    gpuStream_t s;
    GPU_RT_CHECK(gpuStreamCreate(&s));
    GPU_RT_CHECK(gpuMemsetAsync(ptr, val, bytes, s));
    GPU_RT_CHECK(gpuStreamSynchronize(s));
    GPU_RT_CHECK(gpuStreamDestroy(s));
  };

  size_t sizes[] = {262144,   1048576,   4194304,  16777216,
                    67108864, 268435456, 536870912};
  constexpr int kSizes = sizeof(sizes) / sizeof(sizes[0]);
  constexpr int kWarmup = 5;
  constexpr int kIters = 20;
  size_t max_bytes = sizes[kSizes - 1];

  void *d_in = nullptr, *d_out = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d_in, max_bytes));
  if (inplace) {
    d_out = d_in;
  } else {
    GPU_RT_CHECK(gpuMalloc(&d_out, max_bytes));
  }

  // Prepare connections and buffer resources.
  {
    CollectiveConfig prep;
    prep.nranks = 2;
    prep.rank = rank;
    prep.input_bytes = max_bytes;
    prep.output_bytes = max_bytes;
    prep.tile_bytes = 65536;
    prep.kind = coll_kind;
    ex->prepare(prep, d_in, d_out);
  }

  // Handshake: one warm AllReduce to establish peer paths on both sides.
  {
    CollectiveConfig hs;
    hs.nranks = 2;
    hs.rank = rank;
    hs.input_bytes = 65536;
    hs.output_bytes = 65536;
    hs.tile_bytes = 65536;
    hs.kind = CollKind::AllReduceRing;
    sync_memset(d_in, 0, 65536);
    sync_memset(d_out, 0, 65536);
    UK_DBG(UK_DBG_LVL_EXEC, "[handshake r%d] before submit", rank);
    auto h = ex->submit(hs, d_in, d_out);
    UK_DBG(UK_DBG_LVL_EXEC, "[handshake r%d] after submit", rank);
    int spin = 0;
    while (ex->status(h) != CollectiveOpStatus::Completed) {
      if (uk_dbg_lvl() >= UK_DBG_LVL_EXEC && ++spin % 100000 == 0)
        UK_DBG(UK_DBG_LVL_EXEC, "[handshake r%d] waiting... spin=%d", rank,
               spin);
      std::this_thread::yield();
    }
    ex->release(h);
  }

  if (rank == 1) {
    bool show_counters = uk_dbg_lvl() >= UK_DBG_LVL_EXEC;
    std::printf("%9s %10s %10s\n", "Size", "Lat(us)", "BW(GB/s)");
    for (int si = 0; si < kSizes; ++si) {
      size_t bytes = sizes[si];
      size_t tile_bytes = adaptive_tile_bytes(bytes);
      CollectiveConfig ar;
      ar.nranks = 2;
      ar.rank = rank;
      ar.input_bytes = bytes;
      ar.output_bytes = bytes;
      ar.tile_bytes = tile_bytes;
      ar.kind = coll_kind;
      ar.signal_group_tiles = (uint32_t)sig_group;

      for (int w = 0; w < kWarmup; ++w) {
        sync_memset(d_in, 0, bytes);
        sync_memset(d_out, 0, bytes);
        auto h = ex->submit(ar, d_in, d_out);
        while (ex->status(h) != CollectiveOpStatus::Completed)
          std::this_thread::yield();
        ex->release(h);
      }
      PathCounters before;
      if (show_counters) before = ex->get_path_counters();
      double total_us = 0;
      for (int iter = 0; iter < kIters; ++iter) {
        sync_memset(d_in, 0, bytes);
        sync_memset(d_out, 0, bytes);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto h = ex->submit(ar, d_in, d_out);
        while (ex->status(h) != CollectiveOpStatus::Completed)
          std::this_thread::yield();
        auto t1 = std::chrono::high_resolution_clock::now();
        ex->release(h);
        total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
      }
      PathCounters after;
      if (show_counters) after = ex->get_path_counters();
      double avg_us = total_us / kIters;
      double bw_gbs = (bytes * 2.0) / (avg_us * 1e3);
      if (coll_kind == CollKind::AllToAllPairwise)
        bw_gbs = bytes / (avg_us * 1e3);
      char const* unit;
      double sz;
      if (bytes >= 1ul << 30) {
        unit = "GB";
        sz = bytes / (double)(1ul << 30);
      } else if (bytes >= 1ul << 20) {
        unit = "MB";
        sz = bytes / (double)(1ul << 20);
      } else {
        unit = "KB";
        sz = bytes / (double)(1ul << 10);
      }
      std::printf("%8.1f %-3s %10.1f %10.2f\n", sz, unit, avg_us, bw_gbs);
      if (show_counters) {
        std::printf("         dev:%zu  ipc:%zu  rdma:%zu\n",
                    after.device - before.device, after.ipc - before.ipc,
                    after.rdma - before.rdma);
      }
      fflush(stdout);
    }
    std::printf(
        "\nSprayExecutor %s benchmark done\n",
        (coll_kind == CollKind::AllToAllPairwise) ? "AllToAll" : "AllReduce");
  } else {
    for (int si = 0; si < kSizes; ++si) {
      size_t bytes = sizes[si];
      size_t tile_bytes = adaptive_tile_bytes(bytes);
      CollectiveConfig ar;
      ar.nranks = 2;
      ar.rank = rank;
      ar.input_bytes = bytes;
      ar.output_bytes = bytes;
      ar.tile_bytes = tile_bytes;
      ar.kind = coll_kind;
      ar.signal_group_tiles = (uint32_t)sig_group;
      for (int i = 0; i < kWarmup + kIters; ++i) {
        sync_memset(d_in, 0, bytes);
        sync_memset(d_out, 0, bytes);
        auto h = ex->submit(ar, d_in, d_out);
        while (ex->status(h) != CollectiveOpStatus::Completed)
          std::this_thread::yield();
        ex->release(h);
      }
    }
  }

  // Destroy executor first — stops adapter workers that may hold
  // references to GPU buffers.
  ex.reset();
  GPU_RT_CHECK(gpuFree(d_in));
  if (!inplace) GPU_RT_CHECK(gpuFree(d_out));
  return 0;
}

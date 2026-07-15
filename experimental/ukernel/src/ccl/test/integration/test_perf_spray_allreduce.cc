// SprayExecutor AllReduce throughput benchmark.

#include "coll_config.h"
#include "executor.h"
#include "gpu_rt.h"
#include "transport.h"
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
    std::fprintf(stderr, "Usage: --role=server|client [--gpu GPU]\n");
    return 1;
  }
  int rank = (role == "server") ? 0 : 1;
  int gpu = get_int_arg(argc, argv, "--gpu", rank);
  int port = 16998;

  GPU_RT_CHECK(gpuSetDevice(gpu));

  SprayExecutorConfig cfg;
  cfg.gpu_id = gpu; cfg.rank = rank; cfg.world_size = 2;
  cfg.exchanger_ip = (rank == 0) ? "0.0.0.0" : "127.0.0.1";
  cfg.exchanger_port = port; cfg.local_id = rank;
  auto ex = SprayExecutor::create(cfg);

  // Synchronous memset helper — avoids async gpuMemset racing with RDMA writes.
  auto sync_memset = [](void* ptr, int val, size_t bytes) {
    gpuStream_t s;
    GPU_RT_CHECK(gpuStreamCreate(&s));
    GPU_RT_CHECK(gpuMemsetAsync(ptr, val, bytes, s));
    GPU_RT_CHECK(gpuStreamSynchronize(s));
    GPU_RT_CHECK(gpuStreamDestroy(s));
  };

  size_t sizes[] = {262144, 1048576, 4194304, 16777216, 67108864, 268435456, 536870912};
  constexpr int kSizes = sizeof(sizes) / sizeof(sizes[0]);
  constexpr int kWarmup = 5;
  constexpr int kIters = 20;
  size_t max_bytes = sizes[kSizes - 1];

  void *d_in = nullptr, *d_out = nullptr, *d_scr = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d_in, max_bytes));
  GPU_RT_CHECK(gpuMalloc(&d_out, max_bytes));
  GPU_RT_CHECK(gpuMalloc(&d_scr, max_bytes));

  // Register buffers with their full capacity so any sub-range
  // offset used by a smaller collective remains within the MR.
  ex->get_or_register_buf(d_in, max_bytes);
  ex->get_or_register_buf(d_out, max_bytes);
  ex->get_or_register_buf(d_scr, max_bytes);

  // Handshake: one warm AllReduce to establish peer paths on both sides.
  {
    sync_memset(d_in, 0, 65536);
    sync_memset(d_out, 0, 65536);
    sync_memset(d_scr, 0, 65536);
    CollectiveConfig hs;
    hs.nranks = 2; hs.rank = rank;
    hs.input_bytes = 65536; hs.output_bytes = 65536;
    hs.tile_bytes = 65536; hs.kind = CollKind::AllReduceRing;
    auto h = ex->submit(hs, d_in, d_out, d_scr);
    while (ex->status(h) != CollectiveOpStatus::Completed)
      std::this_thread::yield();
    ex->release(h);
  }

  if (rank == 1) {
    bool show_counters = (std::getenv("UK_CCL_PATH_COUNTERS") != nullptr);
    constexpr size_t kMaxTiles = 256;
    std::printf("%9s %10s %10s\n", "Size", "Lat(us)", "BW(GB/s)");
    for (int si = 0; si < kSizes; ++si) {
      size_t bytes = sizes[si];
      size_t tile_bytes = std::max((size_t)65536, (bytes + kMaxTiles - 1) / kMaxTiles);
      CollectiveConfig ar;
      ar.nranks = 2; ar.rank = rank;
      ar.input_bytes = bytes; ar.output_bytes = bytes;
      ar.tile_bytes = tile_bytes; ar.kind = CollKind::AllReduceRing;

      for (int w = 0; w < kWarmup; ++w) {
        sync_memset(d_in, 0, bytes);
        sync_memset(d_out, 0, bytes);
        sync_memset(d_scr, 0, bytes);
        auto h = ex->submit(ar, d_in, d_out, d_scr);
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
        sync_memset(d_scr, 0, bytes);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto h = ex->submit(ar, d_in, d_out, d_scr);
        while (ex->status(h) != CollectiveOpStatus::Completed)
          std::this_thread::yield();
        auto t1 = std::chrono::high_resolution_clock::now();
        ex->release(h);
        total_us +=
            std::chrono::duration<double, std::micro>(t1 - t0).count();
      }
      PathCounters after;
      if (show_counters) after = ex->get_path_counters();
      double avg_us = total_us / kIters;
      double bw_gbs = (bytes * 2.0) / (avg_us * 1e3);
      char const* unit;
      double sz;
      if (bytes >= 1ul << 30) { unit = "GB"; sz = bytes / (double)(1ul << 30); }
      else if (bytes >= 1ul << 20) { unit = "MB"; sz = bytes / (double)(1ul << 20); }
      else { unit = "KB"; sz = bytes / (double)(1ul << 10); }
      std::printf("%8.1f %-3s %10.1f %10.2f\n", sz, unit, avg_us, bw_gbs);
      if (show_counters) {
        std::printf("         dev:%zu  ipc:%zu  rdma:%zu\n",
                    after.device - before.device,
                    after.ipc - before.ipc,
                    after.rdma - before.rdma);
      }
      fflush(stdout);
    }
    std::printf("\nSprayExecutor AllReduce benchmark done\n");
  } else {
    constexpr size_t kMaxTiles = 256;
    for (int si = 0; si < kSizes; ++si) {
      size_t bytes = sizes[si];
      size_t tile_bytes = std::max((size_t)65536, (bytes + kMaxTiles - 1) / kMaxTiles);
      CollectiveConfig ar;
      ar.nranks = 2; ar.rank = rank;
      ar.input_bytes = bytes; ar.output_bytes = bytes;
      ar.tile_bytes = tile_bytes; ar.kind = CollKind::AllReduceRing;
      for (int i = 0; i < kWarmup + kIters; ++i) {
        sync_memset(d_in, 0, bytes);
        sync_memset(d_out, 0, bytes);
        sync_memset(d_scr, 0, bytes);
        auto h = ex->submit(ar, d_in, d_out, d_scr);
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
  GPU_RT_CHECK(gpuFree(d_out));
  GPU_RT_CHECK(gpuFree(d_scr));
  return 0;
}

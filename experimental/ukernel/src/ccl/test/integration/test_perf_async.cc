#include "algo/chunk_graph.h"
#include "backend/async_backend.h"
#include "backend/backend.h"
#include "backend/device_backend.h"
#include "coll_config.h"
#include "executor.h"
#include "lower.h"
#include "../../../include/gpu_rt.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;
using namespace UKernel::CCL;

static double to_ms(Clock::duration d) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(d).count();
}
static double to_us(Clock::duration d) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(d).count();
}
static const char* size_str(size_t b) {
  static char buf[32];
  if (b < 1024) snprintf(buf, sizeof(buf), "%zuB", b);
  else if (b < 1024 * 1024) snprintf(buf, sizeof(buf), "%.0fKB", b / 1024.0);
  else snprintf(buf, sizeof(buf), "%.0fMB", b / (1024.0 * 1024.0));
  return buf;
}

static void set_gpu(int gpu_id) {
  int dev;
  GPU_RT_CHECK(gpuGetDevice(&dev));
  if (dev != gpu_id) GPU_RT_CHECK(gpuSetDevice(gpu_id));
}

static void* gpu_alloc(size_t bytes) {
  void* p;
  GPU_RT_CHECK(gpuMalloc(&p, bytes));
  return p;
}

// ── Helper: measure direct DeviceBackend copy ───────────────────────────

struct BenchResult {
  size_t total_bytes;
  double total_ms;
  double gb_s;
  double us_per_op;
  size_t ops;
};

static BenchResult bench_direct(DeviceBackend& be, size_t data_size,
                                size_t num_ops, size_t batch_size = 0) {
  void* src = gpu_alloc(data_size);
  void* dst = gpu_alloc(data_size);
  GPU_RT_CHECK(gpuMemset(src, 0xAB, data_size));
  GPU_RT_CHECK(gpuMemset(dst, 0, data_size));
  BufSpec bufs[3] = {{src, data_size}, {dst, data_size}, {nullptr, 0}};
  be.init(bufs);

  std::vector<Cmd> cmds(num_ops);
  for (size_t i = 0; i < num_ops; ++i) {
    cmds[i].kind = OpKind::Copy;
    cmds[i].bytes = static_cast<uint32_t>(data_size);
    cmds[i].src_buf = 1; cmds[i].dst_buf = 2;
    cmds[i].src_off = 0; cmds[i].dst_off = 0;
    cmds[i].src_peer = ~0u; cmds[i].dst_peer = ~0u;
    cmds[i].redop = ReductionKind::None;
  }

  if (batch_size == 0) batch_size = num_ops;

  auto t0 = Clock::now();
  size_t submitted = 0, total_done = 0;
  while (total_done < num_ops) {
    while (submitted < num_ops) {
      size_t n = be.enqueue(cmds.data() + submitted,
                            std::min(num_ops - submitted, batch_size));
      submitted += n;
      if (n == 0) break;
    }
    uint32_t done[256];
    size_t n = be.drain(done, 256);
    total_done += n;
    if (n == 0 && submitted >= num_ops) std::this_thread::yield();
  }
  auto t1 = Clock::now();
  fprintf(stderr, "    bench_direct: done, %.2f ms\n", to_ms(t1 - t0));

  BenchResult r{};
  r.total_bytes = data_size * num_ops;
  r.total_ms = to_ms(t1 - t0);
  r.gb_s = (r.total_bytes / 1e9) / (r.total_ms / 1000.0);
  r.us_per_op = to_us(t1 - t0) / num_ops;
  r.ops = num_ops;

  GPU_RT_CHECK(gpuFree(dst));
  GPU_RT_CHECK(gpuFree(src));
  return r;
}

// ── Section A: DeviceBackend Raw ────────────────────────────────────────

static void section_a() {
  printf("\n=== Section A: DeviceBackend Raw Performance ===\n");
  fflush(stdout);

  std::vector<size_t> sizes = {256, 1024, 4096, 16384, 65536,
                               262144, 1048576, 4194304, 16777216};
  std::vector<uint32_t> threads_list = {32, 64, 128, 256, 512};
  std::vector<uint32_t> fifos_list = {1, 2, 4, 8};

  // A2: threads_per_block impact
  printf("\n  A1: threads_per_block vs Throughput (GB/s), 256KB copies\n");
  size_t tsz = 262144;
  for (auto th : threads_list) {
    DeviceBackendConfig cfg;
    cfg.threads_per_block = th;
    cfg.max_fifos = 4;
    DeviceBackend be(cfg);
    auto r = bench_direct(be, tsz, 64);
    printf("    threads=%3u  %.2f GB/s  %.1f us/op\n", th, r.gb_s, r.us_per_op);
  }

  // A3: max_fifos impact
  printf("\n  A2: max_fifos vs Throughput, 256KB copies (256 th/block)\n");
  for (auto nf : fifos_list) {
    DeviceBackendConfig cfg;
    cfg.max_fifos = nf;
    cfg.threads_per_block = 256;
    DeviceBackend be(cfg);
    auto r = bench_direct(be, tsz, 64);
    printf("    fifos=%u  %.2f GB/s  %.1f us/op\n", nf, r.gb_s, r.us_per_op);
  }

  // A1: throughput vs data_size
  printf("\n  A3: Copy Throughput vs Data Size (4 fifos, 256 th/block)\n");
  printf("    %-10s  %-10s  %-10s\n", "Size", "GB/s", "us/op");
  for (auto sz : sizes) {
    DeviceBackendConfig cfg;
    cfg.max_fifos = 4;
    cfg.threads_per_block = 256;
    DeviceBackend be(cfg);
    size_t n = std::max<size_t>(1, 16777216 / sz);
    auto r = bench_direct(be, sz, n);
    printf("    %-10s  %-10.2f  %-10.1f\n", size_str(sz), r.gb_s, r.us_per_op);
    fflush(stdout);
  }
}

// ── Section B: AsyncBackend Overhead ────────────────────────────────────

static void section_b() {
  printf("\n=== Section B: AsyncBackend Pipeline Overhead ===\n");
  fflush(stdout);

  std::vector<size_t> batch_sizes = {1, 4, 16, 64, 128, 256};
  size_t data_size = 65536;
  size_t num_ops = 1000;

  void* src = gpu_alloc(data_size);
  void* dst = gpu_alloc(data_size);
  GPU_RT_CHECK(gpuMemset(src, 0xCD, data_size));
  GPU_RT_CHECK(gpuMemset(dst, 0, data_size));

  printf("\n  Batch Size vs Throughput (64KB copies, %zu cmds):\n", num_ops);
  printf("    %-8s  %-12s  %-12s  %-8s\n", "Batch", "Direct(us)", "Async(us)", "Ovhd%");

  for (auto bs : batch_sizes) {
    // Direct baseline
    double dt_us = 0, at_us = 0;

    {
      DeviceBackendConfig cfg; cfg.max_fifos = 4; cfg.threads_per_block = 256;
      DeviceBackend be(cfg);
      BufSpec bufs[3] = {{src, data_size}, {dst, data_size}, {nullptr, 0}};
      be.init(bufs);

      std::vector<Cmd> cmds(num_ops);
      for (size_t i = 0; i < num_ops; ++i) {
        cmds[i].kind = OpKind::Copy; cmds[i].bytes = static_cast<uint32_t>(data_size);
        cmds[i].src_buf = 1; cmds[i].dst_buf = 2;
        cmds[i].src_peer = ~0u; cmds[i].dst_peer = ~0u;
        cmds[i].redop = ReductionKind::None;
      }

      auto t0 = Clock::now();
      size_t sub = 0, done = 0;
      while (done < num_ops) {
        while (sub < num_ops) {
          size_t n = be.enqueue(cmds.data() + sub, std::min(num_ops - sub, bs));
          sub += n; if (n == 0) break;
        }
        uint32_t dbuf[256];
        size_t n = be.drain(dbuf, 256);
        done += n;
        if (n == 0) std::this_thread::yield();
      }
      dt_us = to_us(Clock::now() - t0);
    }

    // Async via AsyncBackend
    {
      DeviceBackendConfig cfg2; cfg2.max_fifos = 4; cfg2.threads_per_block = 256;
      DeviceBackend be2(cfg2);
      BufSpec bufs2[3] = {{src, data_size}, {dst, data_size}, {nullptr, 0}};
      be2.init(bufs2);

      AsyncBackend async(&be2, 1024, 1024);
      async.start();

      std::vector<CmdWithId> acmds(num_ops);
      for (size_t i = 0; i < num_ops; ++i) {
        acmds[i].cmd.kind = OpKind::Copy; acmds[i].cmd.bytes = static_cast<uint32_t>(data_size);
        acmds[i].cmd.src_buf = 1; acmds[i].cmd.dst_buf = 2;
        acmds[i].cmd.src_peer = ~0u; acmds[i].cmd.dst_peer = ~0u;
        acmds[i].cmd.redop = ReductionKind::None;
        acmds[i].caller_id = static_cast<uint32_t>(i);
      }

      auto t2 = Clock::now();
      size_t asub = 0, adone = 0;
      while (adone < num_ops) {
        while (asub < num_ops) {
          size_t n = async.try_enqueue(acmds.data() + asub,
                                       std::min(num_ops - asub, bs));
          asub += n; if (n == 0) break;
        }
        uint32_t dbuf[256];
        size_t n = async.try_drain(dbuf, 256);
        adone += n;
        if (n == 0) std::this_thread::yield();
      }
      at_us = to_us(Clock::now() - t2);
      async.stop();
    }

    double overhead = (at_us - dt_us) / dt_us * 100.0;
    printf("    %-8zu  %-12.1f  %-12.1f  %+6.1f%%\n",
           bs, dt_us / num_ops, at_us / num_ops, overhead);
    fflush(stdout);
  }

  GPU_RT_CHECK(gpuFree(dst));
  GPU_RT_CHECK(gpuFree(src));
}

// ── Section C: SprayExecutor AllReduce ──────────────────────────────────

static void section_c() {
  printf("\n=== Section C: SprayExecutor AllReduce Latency ===\n");
  fflush(stdout);

  std::vector<size_t> sizes = {262144, 1048576, 4194304, 16777216, 67108864};

  printf("\n  AllReduce latency vs data size (4-rank ring, 4KB tile, 4 fifos):\n");
  printf("    %-10s  %-10s  %-10s\n", "Size", "Time(ms)", "BW(GB/s)");

  for (auto sz : sizes) {
    DeviceBackendConfig cfg; cfg.max_fifos = 4; cfg.threads_per_block = 256;
    DeviceBackend dev_be(cfg);

    auto ex = std::make_unique<SprayExecutor>(&dev_be, nullptr);

    std::vector<uint8_t> in(sz, 0xAA);
    std::vector<uint8_t> out(sz, 0);
    size_t scratch_sz = std::max<size_t>(sz / 2, 4096);
    std::vector<uint8_t> scratch(scratch_sz, 0);

    CollectiveConfig cc;
    cc.nranks = 4; cc.rank = 0;
    cc.input_bytes = sz; cc.output_bytes = sz;
    cc.tile_bytes = 4096;
    cc.kind = CollKind::AllReduceRing;

    auto t0 = Clock::now();
    auto h = ex->submit_allreduce(cc, in.data(), out.data(), scratch.data());
    ex->wait(h, std::chrono::milliseconds(30000));
    auto t1 = Clock::now();

    double ms = to_ms(t1 - t0);
    double gb_s = (sz / 1e9) / (ms / 1000.0);
    printf("    %-10s  %-10.2f  %-10.2f\n", size_str(sz), ms, gb_s);

    ex->release(h);
    fflush(stdout);
  }
}

// ── Section D: Dual GPU Round-Robin ─────────────────────────────────────

static void section_d() {
  printf("\n=== Section D: Dual GPU Round-Robin ===\n");
  fflush(stdout);

  size_t data_size = 1048576;
  size_t num_ops = 256;
  size_t batch_size = 16;

  // GPU 0 (CUDA_VISIBLE_DEVICES maps 6→0)
  set_gpu(0);
  DeviceBackendConfig cfg0; cfg0.max_fifos = 4; cfg0.threads_per_block = 256;
  DeviceBackend be0(cfg0);
  void* s0 = gpu_alloc(data_size); void* d0 = gpu_alloc(data_size);
  GPU_RT_CHECK(gpuMemset(s0, 0xEE, data_size));
  GPU_RT_CHECK(gpuMemset(d0, 0, data_size));
  BufSpec bufs0[3] = {{s0, data_size}, {d0, data_size}, {nullptr, 0}};
  be0.init(bufs0);
  AsyncBackend async0(&be0, 1024, 1024);
  async0.start();

  // GPU 1 (CUDA_VISIBLE_DEVICES maps 7→1)
  set_gpu(1);
  DeviceBackendConfig cfg1; cfg1.max_fifos = 4; cfg1.threads_per_block = 256;
  DeviceBackend be1(cfg1);
  void* s1 = gpu_alloc(data_size); void* d1 = gpu_alloc(data_size);
  GPU_RT_CHECK(gpuMemset(s1, 0xEE, data_size));
  GPU_RT_CHECK(gpuMemset(d1, 0, data_size));
  BufSpec bufs1[3] = {{s1, data_size}, {d1, data_size}, {nullptr, 0}};
  be1.init(bufs1);
  AsyncBackend async1(&be1, 1024, 1024);
  async1.start();

  // Build commands distributed round-robin
  std::vector<CmdWithId> cmds0, cmds1;
  for (size_t i = 0; i < num_ops; ++i) {
    CmdWithId cwi;
    cwi.cmd.kind = OpKind::Copy;
    cwi.cmd.bytes = static_cast<uint32_t>(data_size);
    cwi.cmd.src_buf = 1; cwi.cmd.dst_buf = 2;
    cwi.cmd.src_peer = ~0u; cwi.cmd.dst_peer = ~0u;
    cwi.cmd.redop = ReductionKind::None;
    cwi.caller_id = static_cast<uint32_t>(i);
    if (i % 2 == 0) cmds0.push_back(cwi);
    else cmds1.push_back(cwi);
  }

  printf("  Dual GPU round-robin: %zu ops (%s each), batch=%zu\n",
         num_ops, size_str(data_size), batch_size);

  auto t0 = Clock::now();

  size_t s0_done = 0, s0_sub = 0;
  size_t s1_done = 0, s1_sub = 0;

  while (s0_done < cmds0.size() || s1_done < cmds1.size()) {
    while (s0_sub < cmds0.size()) {
      size_t n = async0.try_enqueue(cmds0.data() + s0_sub,
                                    std::min(cmds0.size() - s0_sub, batch_size));
      s0_sub += n; if (n == 0) break;
    }
    while (s1_sub < cmds1.size()) {
      size_t n = async1.try_enqueue(cmds1.data() + s1_sub,
                                    std::min(cmds1.size() - s1_sub, batch_size));
      s1_sub += n; if (n == 0) break;
    }
    uint32_t dbuf[256];
    size_t n0 = async0.try_drain(dbuf, 256); s0_done += n0;
    size_t n1 = async1.try_drain(dbuf, 256); s1_done += n1;
    if (n0 == 0 && n1 == 0 && s0_sub >= cmds0.size() && s1_sub >= cmds1.size())
      std::this_thread::yield();
  }

  auto t1 = Clock::now();
  double ms = to_ms(t1 - t0);
  double total_gb = (data_size * num_ops) / 1e9;
  double gb_s = total_gb / (ms / 1000.0);

  printf("  Result: %.2f ms, %.2f GB/s combined\n", ms, gb_s);

  async0.stop(); async1.stop();

  set_gpu(0);
  GPU_RT_CHECK(gpuFree(d0)); GPU_RT_CHECK(gpuFree(s0));
  set_gpu(1);
  GPU_RT_CHECK(gpuFree(d1)); GPU_RT_CHECK(gpuFree(s1));
}

// ── main ────────────────────────────────────────────────────────────────

int main() {
  setbuf(stdout, NULL);
  setbuf(stderr, NULL);

  fprintf(stderr, "=== CCL Performance Benchmark ===\n");

  try { section_a(); } catch (std::exception const& e) {
    fprintf(stderr, "SECTION A ERROR: %s\n", e.what());
  }
  try { section_b(); } catch (std::exception const& e) {
    fprintf(stderr, "SECTION B ERROR: %s\n", e.what());
  }
  try { section_c(); } catch (std::exception const& e) {
    fprintf(stderr, "SECTION C ERROR: %s\n", e.what());
  }
  try { section_d(); } catch (std::exception const& e) {
    fprintf(stderr, "SECTION D ERROR: %s\n", e.what());
  }

  fprintf(stderr, "\n=== All perf tests complete ===\n");
  return 0;
}

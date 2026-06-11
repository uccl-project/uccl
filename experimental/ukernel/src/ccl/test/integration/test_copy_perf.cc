#include "../../include/gpu_rt.h"
#include "../backend/backend.h"
#include "../backend/device_backend.h"
#include "../backend/rdma_local_copy_backend.h"
#include "../../transport/adapter/transport_adapter.h"
#include "../../transport/adapter/rdma_adapter.h"
#include "../coll_types.h"
#include "../lower.h"
#include "../utils.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <infiniband/verbs.h>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>

namespace UKernel {
namespace CCL {
namespace {

using Clock = std::chrono::steady_clock;

struct BenchSize {
  size_t bytes;
  int lat_iters;
  int tp_iters;
};

static constexpr BenchSize kBenchSizes[] = {
    {4, 10000, 10000},          {16, 5000, 5000},
    {64, 5000, 5000},           {256, 2000, 2000},
    {1024, 1000, 1000},         {4096, 500, 500},
    {16384, 200, 200},          {65536, 100, 100},
    {262144, 50, 50},           {1048576, 50, 50},
    {4194304, 20, 20},          {16777216, 10, 10},
    {67108864, 5, 3},           {134217728, 3, 2},
};

static constexpr size_t kNumSizes = sizeof(kBenchSizes) / sizeof(kBenchSizes[0]);
static constexpr int kWarmupFactor = 10;
static constexpr size_t kMaxBuf = 256ULL * 1024 * 1024;

static const char* kHandleFile = "/tmp/p2p_bench_handle.bin";
static const char* kReadyFile  = "/tmp/p2p_bench_ready";
static const char* kDoneFile   = "/tmp/p2p_bench_done";
static const char* kSrvDoneFile = "/tmp/p2p_bench_srv_done";
static const char* kResultsFile = "/tmp/p2p_bench_results.txt";
static const char* kMrFile     = "/tmp/p2p_bench_mr.txt";
static const char* kSrvSpecFile = "/tmp/p2p_bench_srv_spec.bin";
static const char* kCliSpecFile = "/tmp/p2p_bench_cli_spec.bin";

void write_spec(UKernel::Transport::RdmaPeerConnectSpec const& s,
                char const* path) {
  std::ofstream of(path, std::ios::binary);
  of.write((char*)&s, sizeof(s));
}

UKernel::Transport::RdmaPeerConnectSpec read_spec(char const* path) {
  UKernel::Transport::RdmaPeerConnectSpec s{};
  std::ifstream inf(path, std::ios::binary);
  inf.read((char*)&s, sizeof(s));
  return s;
}

void write_handle(gpuIpcMemHandle_t const& h) {
  std::ofstream of(kHandleFile, std::ios::binary);
  of.write((char*)&h, sizeof(h));
}

gpuIpcMemHandle_t read_handle() {
  gpuIpcMemHandle_t h{};
  std::ifstream inf(kHandleFile, std::ios::binary);
  inf.read((char*)&h, sizeof(h));
  return h;
}

void signal_ready() { std::ofstream(kReadyFile).close(); }
void wait_ready() { while (!std::ifstream(kReadyFile).good()) { usleep(100000); } }
void signal_done() { std::ofstream(kDoneFile).close(); }
void wait_done() { while (!std::ifstream(kDoneFile).good()) { usleep(100000); } }
void signal_server_done() { std::ofstream(kSrvDoneFile).close(); }
void wait_server_done() { while (!std::ifstream(kSrvDoneFile).good()) { usleep(100000); } }
void wait_file(char const* path) {
  while (!std::ifstream(path).good()) { usleep(100000); }
}

// ── cudaMemcpyPeer (GPU 0 → GPU 1) ────────────────────────────────────

double bench_cuda_peer_latency(size_t bytes, int iters, void* src, void* dst,
                               int src_dev, int dst_dev) {
  int wu = std::max(1, iters / kWarmupFactor);
  GPU_RT_CHECK(gpuSetDevice(src_dev));
  for (int i = 0; i < wu; ++i) {
    GPU_RT_CHECK(gpuMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
    GPU_RT_CHECK(gpuDeviceSynchronize());
  }
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    GPU_RT_CHECK(gpuMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
    GPU_RT_CHECK(gpuDeviceSynchronize());
  }
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

double bench_cuda_peer_throughput(size_t bytes, int iters, void* src, void* dst,
                                  int src_dev, int dst_dev) {
  int wu = std::max(1, iters / kWarmupFactor);
  GPU_RT_CHECK(gpuSetDevice(src_dev));
  for (int i = 0; i < wu; ++i) {
    GPU_RT_CHECK(gpuMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
  }
  GPU_RT_CHECK(gpuDeviceSynchronize());
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    GPU_RT_CHECK(gpuMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
  }
  GPU_RT_CHECK(gpuDeviceSynchronize());
  auto t1 = Clock::now();
  double total_s = std::chrono::duration<double>(t1 - t0).count();
  return (bytes * iters * 1e-9) / total_s;
}

// ── DeviceBackend P2P via IPC (GPU 1) ──────────────────────────────────

double bench_device_latency(size_t bytes, int iters, void* ipc_src,
                            void* dst, int src_dev, int dst_dev,
                            uint32_t bytes_per_block = 0) {
  TiledResult tiled;
  tiled.input_bytes = bytes; tiled.output_bytes = bytes;
  tiled.rank = 0; tiled.nranks = 1;
  Op op; op.kind = OpKind::Copy; op.bytes = bytes;
  tiled.ops.push_back(op); tiled.chunk_of.push_back(0); tiled.layers = {{0}};

  GPU_RT_CHECK(gpuSetDevice(dst_dev));
  DeviceBackendConfig db_cfg;
  db_cfg.task_capacity = 20000;
  db_cfg.max_fifos = 64;
  db_cfg.bytes_per_block = bytes_per_block;
  DeviceBackend backend(db_cfg);
  backend.validate(tiled, ipc_src, dst, nullptr);

  int wu = std::max(1, iters / kWarmupFactor);
  for (int i = 0; i < wu; ++i) {
    OpBindings bnd;
    bnd.stream_index = 0;
    bnd.resolved_src = ipc_src; bnd.src_device = src_dev;
    bnd.resolved_dst = dst;    bnd.dst_device = dst_dev;
    BackendToken tok;
    while (true) {
      tok = backend.submit(tiled.ops[0], bnd, ipc_src, dst, nullptr);
      if (tok.value != 0) break;
      BackendToken tmp; backend.drain(&tmp, 1);
    }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {}
  }

  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    OpBindings bnd;
    bnd.stream_index = 0;
    bnd.resolved_src = ipc_src; bnd.src_device = src_dev;
    bnd.resolved_dst = dst;    bnd.dst_device = dst_dev;
    BackendToken tok;
    while (true) {
      tok = backend.submit(tiled.ops[0], bnd, ipc_src, dst, nullptr);
      if (tok.value != 0) break;
      BackendToken tmp; backend.drain(&tmp, 1);
    }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {}
  }
  auto t1 = Clock::now();
  backend.stop(0);
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

double bench_device_throughput(size_t bytes, int iters, void* ipc_src,
                               void* dst, int src_dev, int dst_dev,
                               uint32_t bytes_per_block = 0) {
  TiledResult tiled;
  tiled.input_bytes = bytes; tiled.output_bytes = bytes;
  tiled.rank = 0; tiled.nranks = 1;
  Op op; op.kind = OpKind::Copy; op.bytes = bytes;
  tiled.ops.push_back(op); tiled.chunk_of.push_back(0); tiled.layers = {{0}};

  GPU_RT_CHECK(gpuSetDevice(dst_dev));
  DeviceBackendConfig db_cfg;
  db_cfg.task_capacity = 20000;
  db_cfg.max_fifos = 64;
  db_cfg.bytes_per_block = bytes_per_block;
  DeviceBackend backend(db_cfg);
  backend.validate(tiled, ipc_src, dst, nullptr);

  int wu = std::max(1, iters / kWarmupFactor);
  {
    int sub = 0, comp = 0;
    std::vector<BackendToken> tokens(wu);
    while (comp < wu) {
      while (sub < wu) {
        OpBindings bnd; bnd.stream_index = 0;
        bnd.resolved_src = ipc_src; bnd.src_device = src_dev;
        bnd.resolved_dst = dst;    bnd.dst_device = dst_dev;
        auto tok = backend.submit(tiled.ops[0], bnd, ipc_src, dst, nullptr);
        if (tok.value == 0) break;
        tokens[sub++] = tok;
      }
      BackendToken out[64];
      size_t n = backend.drain(out, 64);
      for (size_t j = 0; j < n; ++j)
        for (int k = 0; k < sub; ++k)
          if (tokens[k].value == out[j].value) { ++comp; break; }
      if (sub == wu && comp < sub) std::this_thread::yield();
    }
  }
  backend.stop(0);
  backend.validate(tiled, ipc_src, dst, nullptr);

  auto t0 = Clock::now();
  int submitted = 0, completed = 0;
  std::vector<BackendToken> tokens(iters);
  while (completed < iters) {
    while (submitted < iters) {
      OpBindings bnd; bnd.stream_index = 0;
      bnd.resolved_src = ipc_src; bnd.src_device = src_dev;
      bnd.resolved_dst = dst;    bnd.dst_device = dst_dev;
      auto tok = backend.submit(tiled.ops[0], bnd, ipc_src, dst, nullptr);
      if (tok.value == 0) break;
      tokens[submitted++] = tok;
    }
    BackendToken out[64];
    size_t n = backend.drain(out, 64);
    for (size_t j = 0; j < n; ++j)
      for (int k = 0; k < submitted; ++k)
        if (tokens[k].value == out[j].value) { ++completed; break; }
    if (submitted == iters && completed < submitted) std::this_thread::yield();
  }
  auto t1 = Clock::now();
  backend.stop(0);
  double total_s = std::chrono::duration<double>(t1 - t0).count();
  return (bytes * iters * 1e-9) / total_s;
}

// ── RdmaLocalCopy P2P ──────────────────────────────────────────────────

double bench_rdma_latency(size_t bytes, int iters, void* src, void* dst) {
  RdmaLocalCopyBackendConfig cfg; cfg.gpu_id = 0;
  RdmaLocalCopyBackend backend(cfg);
  if (strcmp(backend.name(), "degraded") == 0) return -1.0;

  TiledResult tiled;
  tiled.input_bytes = bytes; tiled.output_bytes = bytes;
  tiled.rank = 0; tiled.nranks = 1;
  Op op; op.kind = OpKind::Copy; op.bytes = bytes;
  tiled.ops.push_back(op); tiled.chunk_of.push_back(0); tiled.layers = {{0}};

  backend.validate(tiled, src, dst, nullptr);
  if (strcmp(backend.name(), "degraded") == 0) return -1.0;

  int wu = std::max(1, iters / kWarmupFactor);
  for (int i = 0; i < wu; ++i) {
    OpBindings bnd;
    bnd.stream_index = 0; bnd.resolved_src = src; bnd.resolved_dst = dst;
    BackendToken tok;
    while (true) {
      tok = backend.submit(tiled.ops[0], bnd, src, dst, nullptr);
      if (tok.value != 0) break;
      if (strcmp(backend.name(), "degraded") == 0) return -1.0;
      BackendToken tmp; backend.drain(&tmp, 1);
    }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {}
  }

  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    OpBindings bnd;
    bnd.stream_index = 0; bnd.resolved_src = src; bnd.resolved_dst = dst;
    BackendToken tok;
    while (true) {
      tok = backend.submit(tiled.ops[0], bnd, src, dst, nullptr);
      if (tok.value != 0) break;
      if (strcmp(backend.name(), "degraded") == 0) return -1.0;
      BackendToken tmp; backend.drain(&tmp, 1);
    }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {}
  }
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

double bench_rdma_throughput(size_t bytes, int iters, void* src, void* dst) {
  RdmaLocalCopyBackendConfig cfg; cfg.gpu_id = 0;
  RdmaLocalCopyBackend backend(cfg);
  if (strcmp(backend.name(), "degraded") == 0) return -1.0;

  TiledResult tiled;
  tiled.input_bytes = bytes; tiled.output_bytes = bytes;
  tiled.rank = 0; tiled.nranks = 1;
  Op op; op.kind = OpKind::Copy; op.bytes = bytes;
  tiled.ops.push_back(op); tiled.chunk_of.push_back(0); tiled.layers = {{0}};

  backend.validate(tiled, src, dst, nullptr);
  if (strcmp(backend.name(), "degraded") == 0) return -1.0;

  int wu = std::max(1, iters / kWarmupFactor);
  {
    int sub = 0, comp = 0;
    std::vector<BackendToken> tokens(wu);
    while (comp < wu) {
      while (sub < wu) {
        OpBindings bnd; bnd.stream_index = 0;
        bnd.resolved_src = src; bnd.resolved_dst = dst;
        auto tok = backend.submit(tiled.ops[0], bnd, src, dst, nullptr);
        if (tok.value == 0) break;
        tokens[sub++] = tok;
      }
      BackendToken out[64];
      size_t n = backend.drain(out, 64);
      for (size_t j = 0; j < n; ++j)
        for (int k = 0; k < sub; ++k)
          if (tokens[k].value == out[j].value) { ++comp; break; }
      if (sub == wu && comp < sub) std::this_thread::yield();
    }
  }

  auto t0 = Clock::now();
  int submitted = 0, completed = 0;
  std::vector<BackendToken> tokens(iters);
  while (completed < iters) {
    while (submitted < iters) {
      OpBindings bnd; bnd.stream_index = 0;
      bnd.resolved_src = src; bnd.resolved_dst = dst;
      auto tok = backend.submit(tiled.ops[0], bnd, src, dst, nullptr);
      if (tok.value == 0) break;
      tokens[submitted++] = tok;
    }
    BackendToken out[64];
    size_t n = backend.drain(out, 64);
    for (size_t j = 0; j < n; ++j)
      for (int k = 0; k < submitted; ++k)
        if (tokens[k].value == out[j].value) { ++completed; break; }
    if (submitted == iters && completed < submitted)
      std::this_thread::yield();
  }
  auto t1 = Clock::now();
  double total_s = std::chrono::duration<double>(t1 - t0).count();
  return (bytes * iters * 1e-9) / total_s;
}

// Overloads that accept pre-configured backend (for P2P with remote MR)
double bench_rdma_latency(size_t bytes, int iters, void* src, void* dst,
                          RdmaLocalCopyBackend* backend) {
  if (strcmp(backend->name(), "degraded") == 0) return -1.0;
  TiledResult tiled;
  tiled.input_bytes = bytes; tiled.output_bytes = bytes;
  tiled.rank = 0; tiled.nranks = 1;
  Op op; op.kind = OpKind::Copy; op.bytes = bytes;
  tiled.ops.push_back(op); tiled.chunk_of.push_back(0); tiled.layers = {{0}};
  backend->validate(tiled, src, dst, nullptr);
  if (strcmp(backend->name(), "degraded") == 0) return -1.0;
  int wu = std::max(1, iters / kWarmupFactor);
  int errs = 0;
  for (int i = 0; i < wu; ++i) {
    OpBindings bnd; bnd.stream_index = 0;
    bnd.resolved_src = src; bnd.resolved_dst = dst;
    BackendToken tok;
    while (true) {
      tok = backend->submit(tiled.ops[0], bnd, src, dst, nullptr);
      if (tok.value != 0) break;
      BackendToken tmp; backend->drain(&tmp, 1);
    }
    BackendToken out;
    while (backend->drain(&out, 1) != 1 || out.value != tok.value) {}
    if (out.failed) ++errs;
  }
  if (errs > wu / 2) { std::fprintf(stderr, "[rdma] LAT %zuB: %d/%d FAILED, degrading\n", bytes, errs, wu); return -1.0; }
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    OpBindings bnd; bnd.stream_index = 0;
    bnd.resolved_src = src; bnd.resolved_dst = dst;
    BackendToken tok;
    while (true) {
      tok = backend->submit(tiled.ops[0], bnd, src, dst, nullptr);
      if (tok.value != 0) break;
      BackendToken tmp; backend->drain(&tmp, 1);
    }
    BackendToken out;
    while (backend->drain(&out, 1) != 1 || out.value != tok.value) {}
    if (out.failed) ++errs;
  }
  if (errs > iters / 2) { std::fprintf(stderr, "[rdma] LAT %zuB: %d/%d FAILED, degrading\n", bytes, errs, iters); return -1.0; }
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

double bench_rdma_throughput(size_t bytes, int iters, void* src, void* dst,
                             RdmaLocalCopyBackend* backend) {
  if (strcmp(backend->name(), "degraded") == 0) return -1.0;
  TiledResult tiled;
  tiled.input_bytes = bytes; tiled.output_bytes = bytes;
  tiled.rank = 0; tiled.nranks = 1;
  Op op; op.kind = OpKind::Copy; op.bytes = bytes;
  tiled.ops.push_back(op); tiled.chunk_of.push_back(0); tiled.layers = {{0}};
  backend->validate(tiled, src, dst, nullptr);
  if (strcmp(backend->name(), "degraded") == 0) return -1.0;
  int wu = std::max(1, iters / kWarmupFactor);
  int errs = 0;
  {
    int sub = 0, comp = 0;
    std::vector<BackendToken> tokens(wu);
    while (comp < wu) {
      while (sub < wu) {
        OpBindings bnd; bnd.stream_index = 0;
        bnd.resolved_src = src; bnd.resolved_dst = dst;
        auto tok = backend->submit(tiled.ops[0], bnd, src, dst, nullptr);
        if (tok.value == 0) break;
        tokens[sub++] = tok;
      }
      BackendToken out[64];
      size_t n = backend->drain(out, 64);
      for (size_t j = 0; j < n; ++j) {
        if (out[j].failed) ++errs;
        for (int k = 0; k < sub; ++k)
          if (tokens[k].value == out[j].value) { ++comp; break; }
      }
      if (sub == wu && comp < sub) std::this_thread::yield();
    }
  }
  if (errs > wu / 2) { std::fprintf(stderr, "[rdma] TP %zuB: %d/%d FAILED, degrading\n", bytes, errs, wu); return -1.0; }
  auto t0 = Clock::now();
  int submitted = 0, completed = 0;
  errs = 0;
  std::vector<BackendToken> tokens(iters);
  while (completed < iters) {
    while (submitted < iters) {
      OpBindings bnd; bnd.stream_index = 0;
      bnd.resolved_src = src; bnd.resolved_dst = dst;
      auto tok = backend->submit(tiled.ops[0], bnd, src, dst, nullptr);
      if (tok.value == 0) break;
      tokens[submitted++] = tok;
    }
    BackendToken out[64];
    size_t n = backend->drain(out, 64);
    for (size_t j = 0; j < n; ++j) {
      if (out[j].failed) ++errs;
      for (int k = 0; k < submitted; ++k)
        if (tokens[k].value == out[j].value) { ++completed; break; }
    }
    if (submitted == iters && completed < submitted)
      std::this_thread::yield();
  }
  if (errs > iters / 2) { std::fprintf(stderr, "[rdma] TP %zuB: %d/%d FAILED, degrading\n", bytes, errs, iters); return -1.0; }
  auto t1 = Clock::now();
  double total_s = std::chrono::duration<double>(t1 - t0).count();
  return (bytes * iters * 1e-9) / total_s;
}

const char* fmt_size(size_t bytes) {
  static char buf[32];
  if (bytes < 1024)       snprintf(buf, sizeof(buf), "%6zu B", bytes);
  else if (bytes < 1048576) snprintf(buf, sizeof(buf), " %5.1f KB", bytes/1024.0);
  else                     snprintf(buf, sizeof(buf), " %5.1f MB", bytes/(1024.0*1024.0));
  return buf;
}

// ── Roles ──────────────────────────────────────────────────────────────

void run_server(int gpu0, int gpu1) {
  unlink(kHandleFile); unlink(kReadyFile); unlink(kDoneFile);
  unlink(kResultsFile); unlink(kSrvDoneFile);
  unlink(kMrFile); unlink(kSrvSpecFile); unlink(kCliSpecFile);

  void* src = nullptr;
  GPU_RT_CHECK(gpuSetDevice(gpu0));
  GPU_RT_CHECK(gpuMalloc(&src, kMaxBuf));

  // IPC export for client's cudaMemcpy + DeviceBackend
  gpuIpcMemHandle_t h;
  GPU_RT_CHECK(gpuIpcGetMemHandle(&h, src));
  write_handle(h);
  signal_ready();

  // Create real RDMA peer (not self-peer)
  RdmaLocalCopyBackendConfig cfg; cfg.gpu_id = gpu0;
  RdmaLocalCopyBackend rdma_backend(cfg);

  if (strcmp(rdma_backend.name(), "degraded") != 0) {
    auto srv_spec = rdma_backend.get_connect_spec();
    write_spec(srv_spec, kSrvSpecFile);
    signal_ready();
    fprintf(stderr, "[server] QP spec written, waiting for client spec...\n");

    wait_file(kCliSpecFile);
    auto cli_spec = read_spec(kCliSpecFile);
    fprintf(stderr, "[server] client QP spec received\n");

    // Read MR info
    uint64_t remote_addr = 0;
    uint32_t remote_rkey = 0;
    { std::ifstream mf(kMrFile);
      if (mf.good()) mf >> remote_addr >> remote_rkey; }
    fprintf(stderr, "[server] remote dst MR: addr=0x%lx rkey=%u\n",
            remote_addr, remote_rkey);

    if (remote_rkey != 0) {
      rdma_backend.register_remote_buffer(2, (void const*)remote_addr, remote_rkey);
      rdma_backend.setup_external_peer(cli_spec);
    } else {
      fprintf(stderr, "[server] no MR info, rdma degraded\n");
    }
  }

  fprintf(stderr, "[server] waiting for client benchmarks...\n");
  wait_done();

  // Read client results (dev1_lat/dev1_tp stored in rdma slots)
  double cm_lat[kNumSizes], cm_tp[kNumSizes];
  double dev_lat[kNumSizes], dev_tp[kNumSizes];
  double dev1_lat[kNumSizes], dev1_tp[kNumSizes];
  { std::ifstream rf(kResultsFile);
    for (size_t i = 0; i < kNumSizes; ++i)
      rf >> cm_lat[i] >> cm_tp[i] >> dev_lat[i] >> dev_tp[i] >> dev1_lat[i] >> dev1_tp[i]; }

  // Run RdmaLocalCopy benchmarks
  double rdma_lat[kNumSizes], rdma_tp[kNumSizes];
  for (size_t i = 0; i < kNumSizes; ++i) rdma_lat[i] = rdma_tp[i] = -1;
  if (strcmp(rdma_backend.name(), "degraded") != 0) {
    for (size_t i = 0; i < kNumSizes; ++i) {
      auto& bs = kBenchSizes[i];
      rdma_lat[i] = bench_rdma_latency(bs.bytes, bs.lat_iters, src,
                                        nullptr, &rdma_backend);
      rdma_tp[i] = bench_rdma_throughput(bs.bytes, bs.tp_iters, src,
                                          nullptr, &rdma_backend);
    }
  }

  printf("\n=== P2P Copy Benchmark  GPU %d -> GPU %d ===\n\n", gpu0, gpu1);
  printf("%-9s | %8s  %8s  %8s  %8s | %8s  %8s  %8s  %8s\n",
         "Size", "cuda us", "dev-auto us", "dev-1blk us", "rdma us",
         "cuda GB/s", "dev-auto GB/s", "dev-1blk GB/s", "rdma GB/s");
  printf("----------|----------------------------------------|----------------------------------------\n");
  for (size_t i = 0; i < kNumSizes; ++i) {
    printf("%-9s | %8.2f  %8.2f  %8.2f  ",
           fmt_size(kBenchSizes[i].bytes), cm_lat[i], dev_lat[i], dev1_lat[i]);
    if (rdma_lat[i] < 0) printf(" degraded | ");
    else printf("%8.2f | ", rdma_lat[i]);
    printf("%8.2f  %8.2f  %8.2f  ", cm_tp[i], dev_tp[i], dev1_tp[i]);
    if (rdma_tp[i] < 0) printf(" degraded\n");
    else printf("%8.2f\n", rdma_tp[i]);
  }

  GPU_RT_CHECK(gpuFree(src));
  signal_server_done();
}

void run_client(int gpu0, int gpu1) {
  wait_ready();
  auto h = read_handle();

  // GPU 1: src via IPC, dst local
  GPU_RT_CHECK(gpuSetDevice(gpu1));
  GPU_RT_CHECK(gpuFree(nullptr));
  void* ipc_src = nullptr;
  gpuError_t e = gpuIpcOpenMemHandle(&ipc_src, h, gpuIpcMemLazyEnablePeerAccess);
  if (e != gpuSuccess) e = gpuIpcOpenMemHandle(&ipc_src, h, 0);
  GPU_RT_CHECK(e);
  fprintf(stderr, "[client] IPC src opened: %p\n", ipc_src);

  void* dst = nullptr;
  GPU_RT_CHECK(gpuMalloc(&dst, kMaxBuf));

  // Register dst for RDMA peer
  static ibv_context* g_ibv_ctx = nullptr;
  static ibv_pd* g_ibv_pd = nullptr;
  {
    int ndev = 0; ibv_device** devs = ibv_get_device_list(&ndev);
    fprintf(stderr, "[client] ibv devices: %d\n", ndev);
    if (devs && ndev > 0) {
      g_ibv_ctx = ibv_open_device(devs[0]);
      if (g_ibv_ctx) {
        g_ibv_pd = ibv_alloc_pd(g_ibv_ctx);
        if (g_ibv_pd) {
          ibv_mr* mr = ibv_reg_mr(g_ibv_pd, dst, kMaxBuf,
              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
          if (mr) {
            std::ofstream mf(kMrFile);
            mf << (uint64_t)dst << " " << mr->rkey; mf.close();
            fprintf(stderr, "[client] dst MR: addr=%p rkey=%u\n", dst, mr->rkey);
          } else {
            fprintf(stderr, "[client] ibv_reg_mr FAILED: %s (errno=%d)\n", strerror(errno), errno);
          }
        } else {
          fprintf(stderr, "[client] ibv_alloc_pd FAILED\n");
        }
      } else {
        fprintf(stderr, "[client] ibv_open_device FAILED\n");
      }
      ibv_free_device_list(devs);
    } else {
      fprintf(stderr, "[client] no ibv devices found\n");
    }
  }

  // Create RDMA adapter, send QP spec to server
  RdmaLocalCopyBackendConfig cfg; cfg.gpu_id = gpu1;
  RdmaLocalCopyBackend rdma_backend(cfg);

  wait_file(kSrvSpecFile);
  auto srv_spec = read_spec(kSrvSpecFile);
  fprintf(stderr, "[client] server QP spec received\n");

  if (strcmp(rdma_backend.name(), "degraded") != 0) {
    auto cli_spec = rdma_backend.get_connect_spec();
    write_spec(cli_spec, kCliSpecFile);
    signal_ready();

    // Client: wait path (receiver side)
    UKernel::Transport::PeerConnectSpec pcs;
    pcs.peer_rank = 0;
    pcs.type = UKernel::Transport::PeerConnectType::Accept;
    pcs.detail = srv_spec;
    rdma_backend.setup_external_peer_for_client(pcs);
  }

  // Run cudaMemcpy + DeviceBackend (via IPC)
  double cm_lat[kNumSizes], cm_tp[kNumSizes];
  double dev_lat[kNumSizes], dev_tp[kNumSizes];
  double rdma_lat[kNumSizes], rdma_tp[kNumSizes];
  for (size_t i = 0; i < kNumSizes; ++i) {
    auto& bs = kBenchSizes[i];
    // cudaMemcpy inline
    auto bench_cuda_lat = [&]() {
      int wu = std::max(1, bs.lat_iters / kWarmupFactor);
      GPU_RT_CHECK(gpuSetDevice(gpu1));
      for (int j = 0; j < wu; ++j) {
        GPU_RT_CHECK(gpuMemcpy(dst, ipc_src, bs.bytes, gpuMemcpyDeviceToDevice));
        GPU_RT_CHECK(gpuDeviceSynchronize());
      }
      auto t0 = Clock::now();
      for (int j = 0; j < bs.lat_iters; ++j) {
        GPU_RT_CHECK(gpuMemcpy(dst, ipc_src, bs.bytes, gpuMemcpyDeviceToDevice));
        GPU_RT_CHECK(gpuDeviceSynchronize());
      }
      auto t1 = Clock::now();
      return std::chrono::duration<double, std::micro>(t1 - t0).count() / bs.lat_iters;
    };
    auto bench_cuda_tp = [&]() {
      int wu = std::max(1, bs.tp_iters / kWarmupFactor);
      GPU_RT_CHECK(gpuSetDevice(gpu1));
      for (int j = 0; j < wu; ++j)
        GPU_RT_CHECK(gpuMemcpy(dst, ipc_src, bs.bytes, gpuMemcpyDeviceToDevice));
      GPU_RT_CHECK(gpuDeviceSynchronize());
      auto t0 = Clock::now();
      for (int j = 0; j < bs.tp_iters; ++j)
        GPU_RT_CHECK(gpuMemcpy(dst, ipc_src, bs.bytes, gpuMemcpyDeviceToDevice));
      GPU_RT_CHECK(gpuDeviceSynchronize());
      auto t1 = Clock::now();
      return (bs.bytes * bs.tp_iters * 1e-9) / std::chrono::duration<double>(t1 - t0).count();
    };
    cm_lat[i] = bench_cuda_lat();
    cm_tp[i] = bench_cuda_tp();
    dev_lat[i] = bench_device_latency(bs.bytes, bs.lat_iters, ipc_src, dst, gpu0, gpu1);
    dev_tp[i]  = bench_device_throughput(bs.bytes, bs.tp_iters, ipc_src, dst, gpu0, gpu1);
    // 1-block variant for comparison
    double     dev1_lat = bench_device_latency(bs.bytes, bs.lat_iters, ipc_src, dst, gpu0, gpu1, UINT32_MAX);
    double dev1_tp  = bench_device_throughput(bs.bytes, bs.tp_iters, ipc_src, dst, gpu0, gpu1, UINT32_MAX);
    rdma_lat[i] = dev1_lat;  // reuse rdma slot for 1-blk dev latency
    rdma_tp[i] = dev1_tp;
  }

  // ── Block count sweep ────────────────────────────────────────────
  {
    size_t sweep_sizes[] = {262144, 1048576, 4194304, 16777216, 67108864, 134217728};
    uint32_t block_counts[] = {1, 4, 8, 16, 32, 64, 128};
    int n_sizes = sizeof(sweep_sizes)/sizeof(sweep_sizes[0]);
    int n_blocks = sizeof(block_counts)/sizeof(block_counts[0]);

    printf("\n=== DeviceBackend Block Count Sweep (latency us) ===\n\n");
    printf("%-9s |", "Size");
    for (int b = 0; b < n_blocks; ++b) printf(" %4u-blk", block_counts[b]);
    printf("\n");
    printf("----------|");
    for (int b = 0; b < n_blocks; ++b) printf("-------");
    printf("\n");

    for (int s = 0; s < n_sizes; ++s) {
      size_t sz = sweep_sizes[s];
      int lat_iters = (sz < 1048576) ? 100 : ((sz < 16777216) ? 20 : 5);
      printf("%-9s |", fmt_size(sz));
      for (int b = 0; b < n_blocks; ++b) {
        uint32_t bp = block_counts[b];
        uint32_t bpb = 0;
        if (bp > 1) bpb = (uint32_t)((sz + bp - 1) / bp);
        else if (bp == 1) bpb = UINT32_MAX;
        double lat = bench_device_latency(sz, lat_iters, ipc_src, dst, gpu0, gpu1, bpb);
        printf(" %7.1f", lat);
      }
      printf("\n");
    }
  }

  { std::ofstream rf(kResultsFile);
    for (size_t i = 0; i < kNumSizes; ++i)
      rf << cm_lat[i] << " " << cm_tp[i] << " " << dev_lat[i] << " " << dev_tp[i]
         << " " << rdma_lat[i] << " " << rdma_tp[i] << "\n"; }

  GPU_RT_CHECK(gpuIpcCloseMemHandle(ipc_src));
  GPU_RT_CHECK(gpuFree(dst));
  signal_done();
  fprintf(stderr, "[client] benchmarks done, waiting for server RDMA...\n");
  wait_server_done();
  fprintf(stderr, "[client] server done, exiting\n");
}

}  // namespace
}  // namespace CCL
}  // namespace UKernel

int main(int argc, char** argv) {
  using namespace UKernel::CCL;
  try {
    if (argc < 2 || (strcmp(argv[1], "server") != 0 && strcmp(argv[1], "client") != 0)) {
      fprintf(stderr, "Usage: %s server|client\n", argv[0]);
      fprintf(stderr, "  server: GPU 0, cudaMemcpyPeer + RdmaLocalCopy\n");
      fprintf(stderr, "  client: GPU 1, DeviceBackend via IPC\n");
      fprintf(stderr, "  Example:\n");
      fprintf(stderr, "    CUDA_VISIBLE_DEVICES=6,7 %s server &\n", argv[0]);
      fprintf(stderr, "    CUDA_VISIBLE_DEVICES=6,7 %s client\n", argv[0]);
      return 1;
    }
    if (strcmp(argv[1], "server") == 0)
      run_server(0, 1);
    else
      run_client(0, 1);
    printf("\n=== P2P benchmark DONE ===\n");
    return 0;
  } catch (std::exception const& ex) {
    fprintf(stderr, "[p2p-perf] fatal: %s\n", ex.what());
    return 2;
  }
}

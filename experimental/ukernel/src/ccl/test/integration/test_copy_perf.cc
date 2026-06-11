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
#include <thread>
#include <vector>
#include <unistd.h>

namespace UKernel {
namespace CCL {
namespace {

using Clock = std::chrono::steady_clock;
using UKernel::Transport::RdmaPeerConnectSpec;
using UKernel::Transport::PeerConnectSpec;

// ── Configuration ───────────────────────────────────────────────────────

struct BenchSize { size_t bytes; int lat_n; int tp_n; };
static constexpr BenchSize kSizes[] = {
    {4, 10000, 10000},       {16, 5000, 5000},       {64, 5000, 5000},
    {256, 2000, 2000},       {1024, 1000, 1000},     {4096, 500, 500},
    {16384, 200, 200},       {65536, 100, 100},      {262144, 50, 50},
    {1048576, 50, 50},       {4194304, 20, 20},      {16777216, 10, 10},
    {67108864, 5, 3},        {134217728, 3, 2},
};
static constexpr size_t kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);
static constexpr size_t kMaxBuf = 256ULL * 1024 * 1024;
static constexpr int kWarmupFactor = 10;

// Sweep sizes for block count test
static constexpr size_t kSweepSizes[] = {262144, 1048576, 4194304, 16777216, 67108864, 134217728};
static constexpr uint32_t kBlockCounts[] = {1, 4, 8, 16, 32, 64, 128};
static constexpr int kNumSweepSizes = sizeof(kSweepSizes)/sizeof(kSweepSizes[0]);
static constexpr int kNumBlockCounts = sizeof(kBlockCounts)/sizeof(kBlockCounts[0]);

static bool g_block_sweep = false;
static bool g_no_gdr = false;

// ── Temp files for P2P coordination ─────────────────────────────────────

static const char* kHandleFile = "/tmp/p2p_handle";
static const char* kReadyFile  = "/tmp/p2p_ready";
static const char* kDoneFile   = "/tmp/p2p_done";
static const char* kRdmaFile   = "/tmp/p2p_rdma";
static const char* kSrvRdmaFile = "/tmp/p2p_srv_rdma";

// ── Results struct for P2P exchange ─────────────────────────────────────

struct BenchPoint { double lat_us; double tp_gbs; };
struct RdmaInfo {
  RdmaPeerConnectSpec spec;
  uint64_t dst_addr;
  uint32_t dst_rkey;
};

// ── Device buffer ───────────────────────────────────────────────────────

struct DevBuf {
  void* ptr = nullptr; size_t bytes = 0;
  explicit DevBuf(size_t n) : bytes(n) { GPU_RT_CHECK(gpuMalloc(&ptr, n)); }
  ~DevBuf() { if (ptr) gpuFree(ptr); }
  DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this == &o) return *this; if (ptr) gpuFree(ptr);
    ptr=o.ptr; bytes=o.bytes; o.ptr=nullptr; return *this;
  }
  DevBuf(DevBuf const&) = delete;
  DevBuf& operator=(DevBuf const&) = delete;
};

#ifdef __HIP_PLATFORM_AMD__
DevBuf alloc_rdma_buf(size_t sz) {
  void* p = nullptr;
  hipError_t e = hipExtMallocWithFlags(&p, sz, hipDeviceMallocUncached);
  if (e != hipSuccess) GPU_RT_CHECK(gpuMalloc(&p, sz));
  DevBuf b(0); b.ptr = p; b.bytes = sz; return b;
}
#else
DevBuf alloc_rdma_buf(size_t sz) { return DevBuf(sz); }
#endif

// ── IPC helpers ─────────────────────────────────────────────────────────

void write_handle(gpuIpcMemHandle_t const& h) {
  std::ofstream of(kHandleFile, std::ios::binary);
  of.write((char*)&h, sizeof(h));
}
gpuIpcMemHandle_t read_handle() {
  gpuIpcMemHandle_t h{};
  std::ifstream inf(kHandleFile, std::ios::binary);
  inf.read((char*)&h, sizeof(h)); return h;
}
void write_rdma_info(RdmaInfo const& ri) {
  std::ofstream of(kRdmaFile, std::ios::binary);
  of.write((char*)&ri, sizeof(ri));
}
RdmaInfo read_rdma_info() {
  RdmaInfo ri{};
  std::ifstream inf(kRdmaFile, std::ios::binary);
  inf.read((char*)&ri, sizeof(ri)); return ri;
}

void sig(char const* f) { std::ofstream(f).close(); }
void wait(char const* f) { while (!std::ifstream(f).good()) usleep(100000); }

// ── Timing helpers ──────────────────────────────────────────────────────

inline double dur_us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

// ── Generic latency / throughput measurement ────────────────────────────

// submit_drain: one complete op (submit + drain)
template<typename F>
double measure_lat(F submit_drain, int iters) {
  int wu = std::max(1, iters / kWarmupFactor);
  for (int i = 0; i < wu; ++i) submit_drain();
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) submit_drain();
  return dur_us(t0, Clock::now()) / iters;
}

// submit: enqueue one op.  drain_all: drain N pending ops (N >= 0, blocking)
template<typename S, typename D>
double measure_tp(S submit, D drain_all, int iters, size_t bytes) {
  int wu = std::max(1, iters / kWarmupFactor);
  for (int i = 0; i < wu; ++i) submit();
  drain_all(wu);
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) submit();
  drain_all(iters);
  double total_s = std::chrono::duration<double>(Clock::now() - t0).count();
  return (bytes * iters * 1e-9) / total_s;
}

// ── cudaMemcpy runner ───────────────────────────────────────────────────

BenchPoint run_cuda(size_t bytes, int lat_n, int tp_n, void* src, void* dst) {
  auto sub_drn = [&](){
    GPU_RT_CHECK(gpuMemcpy(dst, src, bytes, gpuMemcpyDeviceToDevice));
    GPU_RT_CHECK(gpuDeviceSynchronize());
  };
  return {measure_lat(sub_drn, lat_n),
          measure_tp(sub_drn, [&](int){ GPU_RT_CHECK(gpuDeviceSynchronize()); }, tp_n, bytes)};
}

// ── DeviceBackend runner ────────────────────────────────────────────────

BenchPoint run_device(size_t bytes, int lat_n, int tp_n,
                      void* src, void* dst, int src_dev, int dst_dev,
                      uint32_t bytes_per_block = 0) {
  TiledResult t;
  t.input_bytes = bytes; t.output_bytes = bytes; t.rank = 0; t.nranks = 1;
  Op o; o.kind = OpKind::Copy; o.bytes = bytes;
  t.ops.push_back(o); t.chunk_of.push_back(0); t.layers = {{0}};

  DeviceBackendConfig cfg;
  cfg.task_capacity = 20000; cfg.max_fifos = 64; cfg.bytes_per_block = bytes_per_block;
  cfg.no_gdr = g_no_gdr;
  DeviceBackend backend(cfg);
  backend.validate(t, src, dst, nullptr);

  auto sub = [&](){
    OpBindings b; b.stream_index=0;
    b.resolved_src=src; b.resolved_dst=dst; b.src_device=src_dev; b.dst_device=dst_dev;
    BackendToken tok;
    while(true) { tok=backend.submit(t.ops[0],b,src,dst,nullptr); if(tok.value!=0)break;
      BackendToken tmp; backend.drain(&tmp,1); }
    return tok;
  };
  auto sub_drn = [&](){
    auto t = sub();
    while(true){
      BackendToken out; size_t n=backend.drain(&out,1);
      if(n==1 && out.value==t.value) break;
      std::this_thread::yield();
    }
  };

  double lat = measure_lat(sub_drn, lat_n);

  double tp = measure_tp(
    [&](){
      OpBindings b; b.stream_index=0;
      b.resolved_src=src; b.resolved_dst=dst; b.src_device=src_dev; b.dst_device=dst_dev;
      BackendToken tok;
      while(true){tok=backend.submit(t.ops[0],b,src,dst,nullptr); if(tok.value!=0)break;
        BackendToken tmp; if(backend.drain(&tmp,1)==0) std::this_thread::yield();}
    },
    [&](int n) {
      int done=0;
      while(done < n) {
        BackendToken out[64]; size_t c=backend.drain(out,64);
        done += (int)c;
        if(done < n) std::this_thread::yield();
      }
    }, tp_n, bytes);

  backend.stop(0);
  return {lat, tp};
}

// ── RdmaLocalCopy runner ────────────────────────────────────────────────

BenchPoint run_rdma(size_t bytes, int lat_n, int tp_n,
                    void* src, void* dst, RdmaLocalCopyBackend* be) {
  if (!be || strcmp(be->name(), "degraded") == 0) return {-1, -1};
  TiledResult t;
  t.input_bytes = bytes; t.output_bytes = bytes; t.rank = 0; t.nranks = 1;
  Op o; o.kind = OpKind::Copy; o.bytes = bytes;
  t.ops.push_back(o); t.chunk_of.push_back(0); t.layers = {{0}};
  be->validate(t, src, dst, nullptr);
  if (strcmp(be->name(), "degraded") == 0) return {-1, -1};

  auto sub_drn = [&]() {
    OpBindings b; b.stream_index=0; b.resolved_src=src; b.resolved_dst=dst;
    BackendToken tok;
    while(true){tok=be->submit(t.ops[0],b,src,dst,nullptr); if(tok.value!=0)break;
      BackendToken tmp; be->drain(&tmp,1);}
    BackendToken out; while(be->drain(&out,1)!=1||out.value!=tok.value){}
  };
  double lat = measure_lat(sub_drn, lat_n);

  auto sub_tp = [&](){
    OpBindings b; b.stream_index=0; b.resolved_src=src; b.resolved_dst=dst;
    BackendToken tok;
    while(true){tok=be->submit(t.ops[0],b,src,dst,nullptr); if(tok.value!=0)break;
      BackendToken tmp; if(be->drain(&tmp,1)==0) std::this_thread::yield();}
  };
  double tp = measure_tp(sub_tp,
    [&](int n) {
      int done=0;
      while(done < n) {
        BackendToken out[64]; size_t c=be->drain(out,64);
        done += (int)c;
        if(done < n) std::this_thread::yield();
      }
    }, tp_n, bytes);
  return {lat, tp};
}

// ── Output ──────────────────────────────────────────────────────────────

const char* fmt(size_t b) {
  static char buf[32];
  if (b<1024) snprintf(buf,sizeof(buf),"%6zu B",b);
  else if(b<1048576) snprintf(buf,sizeof(buf)," %5.1f KB",b/1024.0);
  else snprintf(buf,sizeof(buf)," %5.1f MB",b/(1024.0*1024.0));
  return buf;
}

void print_header(const char* title, int g0, int g1) {
  printf("\n=== %s  GPU %d -> GPU %d ===\n\n", title, g0, g1);
  printf("%-9s | %8s  %8s  %8s | %8s  %8s  %8s\n",
         "Size","cuda us","device us","rdma us","cuda GB/s","device GB/s","rdma GB/s");
  printf("----------|----------------------------------|-------------------------------\n");
  fflush(stdout);
}

void print_row(size_t bytes, BenchPoint cm, BenchPoint dev, BenchPoint rdma) {
  printf("%-9s | %8.2f  %8.2f  ", fmt(bytes), cm.lat_us, dev.lat_us);
  if (rdma.lat_us < 0) printf(" degraded | ");
  else printf("%8.2f | ", rdma.lat_us);
  printf("%8.2f  %8.2f  ", cm.tp_gbs, dev.tp_gbs);
  if (rdma.tp_gbs < 0) printf(" degraded\n");
  else printf("%8.2f\n", rdma.tp_gbs);
  fflush(stdout);
}

void print_block_sweep(int gpu_id) {
  printf("\n=== DeviceBackend Block Sweep (GPU %d) ===\n\n", gpu_id);
  printf("%-9s |","Size");
  for (int i = 0; i < kNumBlockCounts; ++i) printf(" %4u-blk", kBlockCounts[i]);
  printf("\n----------|");
  for (int i = 0; i < kNumBlockCounts; ++i) printf("-------");
  printf("\n");

  DevBuf src(kMaxBuf), dst(kMaxBuf);
  for (int s = 0; s < kNumSweepSizes; ++s) {
    size_t sz = kSweepSizes[s];
    int lat_n = (sz < 1048576) ? 100 : ((sz < 16777216) ? 20 : 5);
    printf("%-9s |", fmt(sz));
    for (int b = 0; b < kNumBlockCounts; ++b) {
      uint32_t cnt = kBlockCounts[b];
      uint32_t bpb = (cnt == 1) ? UINT32_MAX : (uint32_t)((sz + cnt - 1) / cnt);
      double lat = run_device(sz, lat_n, lat_n/3, src.ptr, dst.ptr, 0, 0, bpb).lat_us;
      printf(" %7.1f", lat);
    }
    printf("\n");
  }
}

// ── P2P Server ──────────────────────────────────────────────────────────

void run_p2p_server(int gpu0, int gpu1) {
  unlink(kHandleFile); unlink(kReadyFile); unlink(kDoneFile);
  unlink(kRdmaFile); unlink(kSrvRdmaFile);

  DevBuf src(kMaxBuf);
  GPU_RT_CHECK(gpuSetDevice(gpu0));

  gpuIpcMemHandle_t h;
  GPU_RT_CHECK(gpuIpcGetMemHandle(&h, src.ptr));
  write_handle(h);

  // Write our QP spec for client
  RdmaLocalCopyBackendConfig rcfg; rcfg.gpu_id = gpu0;
  RdmaLocalCopyBackend rdma_backend(rcfg);
  if (strcmp(rdma_backend.name(), "degraded") != 0) {
    auto srv_spec = rdma_backend.get_connect_spec();
    std::ofstream sf(kSrvRdmaFile, std::ios::binary);
    sf.write((char*)&srv_spec, sizeof(srv_spec));
  }
  sig(kReadyFile);
  fprintf(stderr, "[server] IPC + QP spec exported, waiting for client...\n");

  // Read client's RDMA info
  wait(kRdmaFile);
  RdmaInfo ri = read_rdma_info();
  fprintf(stderr, "[server] client info: addr=0x%lx rkey=%u\n", ri.dst_addr, ri.dst_rkey);

  if (strcmp(rdma_backend.name(), "degraded") != 0 && ri.dst_rkey != 0) {
    rdma_backend.register_remote_buffer(2, (void const*)ri.dst_addr, ri.dst_rkey);
    rdma_backend.setup_external_peer(ri.spec);
  }

  // Wait for client benchmarks
  wait(kDoneFile);
  BenchPoint cm[kNumSizes], dev[kNumSizes];
  {
    std::ifstream rf(kDoneFile);
    for (size_t i = 0; i < kNumSizes; ++i)
      rf >> cm[i].lat_us >> cm[i].tp_gbs >> dev[i].lat_us >> dev[i].tp_gbs;
  }

  print_header("P2P Copy Benchmark", gpu0, gpu1);

  // Run RdmaLocalCopy + print row by row
  for (size_t i = 0; i < kNumSizes; ++i) {
    auto& bs = kSizes[i];
    BenchPoint rdma = run_rdma(bs.bytes, bs.lat_n, bs.tp_n, src.ptr, (void*)ri.dst_addr, &rdma_backend);
    print_row(bs.bytes, cm[i], dev[i], rdma);
  }

  if (g_block_sweep) print_block_sweep(gpu1);
}

// ── P2P Client ──────────────────────────────────────────────────────────

void run_p2p_client(int gpu0, int gpu1) {
  wait(kReadyFile);
  auto h = read_handle();

  // Read server's QP spec
  RdmaPeerConnectSpec srv_spec{};
  {
    std::ifstream sf(kSrvRdmaFile, std::ios::binary);
    sf.read((char*)&srv_spec, sizeof(srv_spec));
  }

  GPU_RT_CHECK(gpuSetDevice(gpu1));
  GPU_RT_CHECK(gpuFree(nullptr));
  void* ipc_src = nullptr;
  gpuError_t e = gpuIpcOpenMemHandle(&ipc_src, h, gpuIpcMemLazyEnablePeerAccess);
  if (e != gpuSuccess) e = gpuIpcOpenMemHandle(&ipc_src, h, 0);
  GPU_RT_CHECK(e);
  fprintf(stderr, "[client] IPC opened: %p\n", ipc_src);

  DevBuf dst = alloc_rdma_buf(kMaxBuf);

  // RDMA: create adapter, export spec + MR
  RdmaLocalCopyBackendConfig rcfg; rcfg.gpu_id = gpu1;
  RdmaLocalCopyBackend rdma_backend(rcfg);

  RdmaInfo ri{};
  if (strcmp(rdma_backend.name(), "degraded") != 0) {
    ri.spec = rdma_backend.get_connect_spec();

    int ndev=0; ibv_device** devs = ibv_get_device_list(&ndev);
    if (devs && ndev > 0) {
      ibv_context* ctx = ibv_open_device(devs[0]);
      if (ctx) {
        ibv_pd* pd = ibv_alloc_pd(ctx);
        if (pd) {
          ibv_mr* mr = ibv_reg_mr(pd, dst.ptr, kMaxBuf,
              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
          if (mr) { ri.dst_addr=(uint64_t)dst.ptr; ri.dst_rkey=mr->rkey;
            fprintf(stderr,"[client] MR: addr=0x%lx rkey=%u\n",ri.dst_addr,ri.dst_rkey);
          } else fprintf(stderr,"[client] ibv_reg_mr FAILED\n");
        }
      }
      ibv_free_device_list(devs);
    }
  }
  write_rdma_info(ri);
  fprintf(stderr, "[client] rdma info written, running benchmarks...\n");

  // Run cudaMemcpy + DeviceBackend
  BenchPoint cm[kNumSizes], dev[kNumSizes];
  for (size_t i = 0; i < kNumSizes; ++i) {
    auto& bs = kSizes[i];
    fprintf(stderr, "[client] size=%zu...\n", bs.bytes);
    cm[i] = run_cuda(bs.bytes, bs.lat_n, bs.tp_n, ipc_src, dst.ptr);
    fprintf(stderr, "[client]   cuda done\n");
    dev[i] = run_device(bs.bytes, bs.lat_n, bs.tp_n, ipc_src, dst.ptr, gpu0, gpu1);
    fprintf(stderr, "[client]   device done\n");
  }

  { std::ofstream rf(kDoneFile);
    for (size_t i = 0; i < kNumSizes; ++i)
      rf << cm[i].lat_us << " " << cm[i].tp_gbs << " "
         << dev[i].lat_us << " " << dev[i].tp_gbs << "\n"; }
  fprintf(stderr, "[client] results written, done\n");
  sig(kDoneFile);

  fprintf(stderr, "[client] done, waiting server...\n");
  GPU_RT_CHECK(gpuIpcCloseMemHandle(ipc_src));
}

// ── Single GPU G2G ──────────────────────────────────────────────────────

void run_single_gpu(int gpu) {
  GPU_RT_CHECK(gpuSetDevice(gpu));
  DevBuf src(kMaxBuf), dst(kMaxBuf);

  RdmaLocalCopyBackendConfig rcfg; rcfg.gpu_id = gpu;
  RdmaLocalCopyBackend rdma_backend(rcfg);

  print_header("Single-GPU G2G", gpu, gpu);

  for (size_t i = 0; i < kNumSizes; ++i) {
    auto& bs = kSizes[i];
    BenchPoint cm  = run_cuda(bs.bytes, bs.lat_n, bs.tp_n, src.ptr, dst.ptr);
    BenchPoint dev = run_device(bs.bytes, bs.lat_n, bs.tp_n, src.ptr, dst.ptr, 0, 0);
    BenchPoint rdma= run_rdma(bs.bytes, bs.lat_n, bs.tp_n, src.ptr, dst.ptr, &rdma_backend);
    print_row(bs.bytes, cm, dev, rdma);
  }
  if (g_block_sweep) print_block_sweep(gpu);
}

}  // namespace
}  // namespace CCL
}  // namespace UKernel

int main(int argc, char** argv) {
  using namespace UKernel::CCL;
  try {
    const char* mode = (argc >= 2) ? argv[1] : "server";
    for (int i = 2; i < argc; ++i) {
      if (strcmp(argv[i], "--block-sweep") == 0) g_block_sweep = true;
      if (strcmp(argv[i], "--no-gdr") == 0) g_no_gdr = true;
    }

    if (strcmp(mode, "server") == 0) {
      run_p2p_server(0, 1);
    } else if (strcmp(mode, "client") == 0) {
      run_p2p_client(0, 1);
    } else if (strcmp(mode, "g2g") == 0) {
      run_single_gpu(0);
    } else {
      fprintf(stderr, "Usage: %s [server|client|g2g] [--block-sweep]\n", argv[0]);
      fprintf(stderr, "  server  P2P server (default)\n");
      fprintf(stderr, "  client  P2P client\n");
      fprintf(stderr, "  g2g     Single GPU G2G\n");
      return 1;
    }
    printf("\n=== benchmark DONE ===\n");
    return 0;
  } catch (std::exception const& ex) {
    fprintf(stderr, "[perf] fatal: %s\n", ex.what());
    return 2;
  }
}

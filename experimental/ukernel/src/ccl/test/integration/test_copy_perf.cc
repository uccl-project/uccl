#include "../../../include/gpu_rt.h"
#include "../../backend/device_backend.h"
#include "../../coll_types.h"
#include "../../lower.h"
#include "../../utils.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
using UKernel::CCL::DeviceBackend;
using UKernel::CCL::DeviceBackendConfig;
using UKernel::CCL::TiledResult;
using UKernel::CCL::Op;
using UKernel::CCL::OpKind;
using UKernel::CCL::OpBindings;
using UKernel::CCL::BackendToken;

// ── Config ──────────────────────────────────────────────────────────────

struct Size { size_t bytes; int lat_n; int tp_n; };
static constexpr Size kSizes[] = {
    {4, 10000, 10000},     {16, 5000, 5000},      {64, 5000, 5000},
    {256, 2000, 2000},     {1024, 1000, 1000},    {4096, 500, 500},
    {16384, 200, 200},     {65536, 100, 100},     {262144, 50, 50},
    {1048576, 50, 50},     {4194304, 20, 20},     {16777216, 10, 10},
    {67108864, 5, 3},      {134217728, 3, 2},
};
static constexpr int kNumSizes = sizeof(kSizes)/sizeof(kSizes[0]);
static constexpr size_t kMaxBuf = 256ULL*1024*1024;
static constexpr int kWB = 10; // warmup factor

static constexpr size_t kSweepSz[] = {262144,1048576,4194304,16777216,67108864,134217728};
static constexpr uint32_t kBlks[]   = {1,4,8,16,32,64,128};
static constexpr int kSwN = sizeof(kSweepSz)/sizeof(kSweepSz[0]);
static constexpr int kBlkN = sizeof(kBlks)/sizeof(kBlks[0]);

static const char* kHandle  = "/tmp/p2p_handle";
static const char* kDone    = "/tmp/p2p_done";

// ── Helpers ─────────────────────────────────────────────────────────────

struct DevBuf { void* p; size_t n; DevBuf(size_t s):n(s){gpuMalloc(&p,n);}
  ~DevBuf(){if(p)gpuFree(p);} DevBuf(DevBuf&&o):p(o.p),n(o.n){o.p=nullptr;} };

void sig(const char* f) { std::ofstream(f).close(); }
void waitf(const char* f) { while(!std::ifstream(f).good()) usleep(100000); }
void wr_h(gpuIpcMemHandle_t const& h) {
  std::ofstream o(kHandle,std::ios::binary); o.write((char*)&h,sizeof(h)); sig(kHandle); }
gpuIpcMemHandle_t rd_h() { gpuIpcMemHandle_t h{}; std::ifstream i(kHandle,std::ios::binary); i.read((char*)&h,sizeof(h)); return h; }

inline double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::micro>(b-a).count(); }
const char* F(size_t b) { static char buf[32];
  if(b<1024)snprintf(buf,sizeof(buf),"%6zu B",b);
  else if(b<1048576)snprintf(buf,sizeof(buf)," %5.1f KB",b/1024.0);
  else snprintf(buf,sizeof(buf)," %5.1f MB",b/(1024.0*1024.0)); return buf; }

// ── Lat / Tp ────────────────────────────────────────────────────────────

template<typename F> double lat(F sd, int iters) { int w=std::max(1,iters/kWB);
  for(int i=0;i<w;++i)sd(); auto t0=Clock::now(); for(int i=0;i<iters;++i)sd(); return us(t0,Clock::now())/iters; }

struct BenchPoint { double l; double t; };

// ── cudaMemcpy ──────────────────────────────────────────────────────────

BenchPoint bench_cuda(size_t bytes, int ln, int tn, void* src, void* dst) {
  auto sd=[&](){ GPU_RT_CHECK(gpuMemcpy(dst,src,bytes,gpuMemcpyDeviceToDevice)); GPU_RT_CHECK(gpuDeviceSynchronize()); };
  return {lat(sd,ln), lat(sd,tn)*(tn*bytes*1e-9)/(us(Clock::now(),Clock::now())*1e-6+1e-9) };
  // simpler tp: measure total time for tn ops
}

BenchPoint bench_cuda2(size_t bytes, int ln, int tn, void* src, void* dst) {
  auto sd=[&](){ GPU_RT_CHECK(gpuMemcpy(dst,src,bytes,gpuMemcpyDeviceToDevice)); GPU_RT_CHECK(gpuDeviceSynchronize()); };
  // throughput: batch tn ops, measure total
  int w=std::max(1,tn/kWB);
  for(int i=0;i<w;++i) { GPU_RT_CHECK(gpuMemcpy(dst,src,bytes,gpuMemcpyDeviceToDevice)); }
  GPU_RT_CHECK(gpuDeviceSynchronize());
  auto t0=Clock::now();
  for(int i=0;i<tn;++i) GPU_RT_CHECK(gpuMemcpy(dst,src,bytes,gpuMemcpyDeviceToDevice));
  GPU_RT_CHECK(gpuDeviceSynchronize());
  double s=std::chrono::duration<double>(Clock::now()-t0).count();
  return {lat(sd,ln),(bytes*tn*1e-9)/s};
}

// ── DeviceBackend ───────────────────────────────────────────────────────

BenchPoint bench_device(size_t bytes, int ln, int tn,
                         void* src, void* dst, int sdv, int ddv, uint32_t bpb=0) {
  TiledResult t; t.input_bytes=bytes; t.output_bytes=bytes; t.rank=0; t.nranks=1;
  Op o; o.kind=OpKind::Copy; o.bytes=bytes;
  t.ops.push_back(o); t.chunk_of.push_back(0); t.layers={{0}};

  DeviceBackendConfig c; c.task_capacity=20000; c.max_fifos=64; c.bytes_per_block=bpb;
  fprintf(stderr,"[dev] creating backend...\n");
  DeviceBackend be(c);
  fprintf(stderr,"[dev] validating...\n");
  be.validate(t,src,dst,nullptr);
  fprintf(stderr,"[dev] validated\n");

  auto sub=[&](){ OpBindings b; b.stream_index=0; b.resolved_src=src; b.resolved_dst=dst;
    b.src_device=sdv; b.dst_device=ddv; BackendToken tok;
    while(true){tok=be.submit(t.ops[0],b,src,dst,nullptr); if(tok.value!=0)break;
      BackendToken tmp; be.drain(&tmp,1);} return tok; };
  auto drn=[&](BackendToken tok){ BackendToken o; while(be.drain(&o,1)!=1||o.value!=tok.value) std::this_thread::yield(); };

  auto sd=[&](){ auto t=sub(); drn(t); };

  // throughput pipeline
  auto tp_sub=[&](){ static int first=1; if(first){fprintf(stderr,"[dev] tp_sub...\n");first=0;}
    OpBindings b; b.stream_index=0; b.resolved_src=src; b.resolved_dst=dst;
    b.src_device=sdv; b.dst_device=ddv; BackendToken tok; while(true){
      tok=be.submit(t.ops[0],b,src,dst,nullptr); if(tok.value!=0)break;
      BackendToken tmp; if(be.drain(&tmp,1)==0) std::this_thread::yield(); }
    if(first==0){fprintf(stderr,"[dev] tp_sub ok\n");first=-1;}
  };
  auto tp_drn=[&](int n){int d=0;while(d<n){BackendToken o[64];size_t c=be.drain(o,64);d+=(int)c; if(d<n)std::this_thread::yield();}};

  // tp: batch submit, drain all
  int w=std::max(1,tn/kWB);
  fprintf(stderr,"[dev] tp warmup %d...\n",w);
  for(int i=0;i<w;++i) tp_sub(); tp_drn(w);
  fprintf(stderr,"[dev] tp warmup done\n");
  auto t0=Clock::now();
  fprintf(stderr,"[dev] tp timed %d...\n",tn);
  for(int i=0;i<tn;++i) tp_sub(); tp_drn(tn);
  fprintf(stderr,"[dev] tp timed done\n");
  double s=std::chrono::duration<double>(Clock::now()-t0).count();

  be.stop(0);
  return {lat(sd,ln),(bytes*tn*1e-9)/s};
}

// ── Output ──────────────────────────────────────────────────────────────

void pr_hdr(const char* t, int g0, int g1) {
  printf("\n=== %s  GPU %d -> GPU %d ===\n\n",t,g0,g1);
  printf("%-9s | %8s  %8s | %8s  %8s\n","Size","cuda us","device us","cuda GB/s","device GB/s");
  printf("----------|----------------------|----------------------\n"); fflush(stdout);
}
void pr_row(size_t b, BenchPoint c, BenchPoint d) {
  printf("%-9s | %8.2f  %8.2f | %8.2f  %8.2f\n",F(b),c.l,d.l,c.t,d.t); fflush(stdout);
}

void pr_sweep(int gpu, void* s, void* d) {
  printf("\n=== DeviceBackend Block Sweep (GPU %d) ===\n\n",gpu);
  printf("%-9s |","Size");
  for(int i=0;i<kBlkN;++i) printf(" %4u-blk",kBlks[i]);
  printf("\n----------|");
  for(int i=0;i<kBlkN;++i) printf("-------");
  printf("\n");
  for(int si=0;si<kSwN;++si) { size_t sz=kSweepSz[si];
    int n=(sz<1048576)?100:((sz<16777216)?20:5);
    printf("%-9s |",F(sz));
    for(int bi=0;bi<kBlkN;++bi) { uint32_t cnt=kBlks[bi];
      uint32_t bpb=(cnt==1)?UINT32_MAX:(uint32_t)((sz+cnt-1)/cnt);
      printf(" %7.1f",bench_device(sz,n,n/3,s,d,0,0,bpb).l); }
    printf("\n");
  }
}

// ── Server ──────────────────────────────────────────────────────────────

void run_server(int g0, int g1) {
  unlink(kHandle); unlink(kDone);
  DevBuf src(kMaxBuf);   GPU_RT_CHECK(gpuSetDevice(g0));
  gpuIpcMemHandle_t h; GPU_RT_CHECK(gpuIpcGetMemHandle(&h,src.p));
  wr_h(h);
  fprintf(stderr,"[server] handle exported, waiting client...\n");

  waitf(kDone);
  BenchPoint cm[kNumSizes], dev[kNumSizes];
  { std::ifstream rf(kDone);
    for(int i=0;i<kNumSizes;++i) rf>>cm[i].l>>cm[i].t>>dev[i].l>>dev[i].t; }

  pr_hdr("P2P Copy Benchmark",g0,g1);
  for(int i=0;i<kNumSizes;++i) pr_row(kSizes[i].bytes,cm[i],dev[i]);
}

// ── Client ──────────────────────────────────────────────────────────────

void run_client(int g0, int g1) {
  waitf(kHandle); auto h=rd_h();
  GPU_RT_CHECK(gpuSetDevice(g1)); GPU_RT_CHECK(gpuFree(nullptr));
  void* ipc=nullptr; gpuError_t e=gpuIpcOpenMemHandle(&ipc,h,gpuIpcMemLazyEnablePeerAccess);
  if(e!=gpuSuccess) e=gpuIpcOpenMemHandle(&ipc,h,0); GPU_RT_CHECK(e);
  fprintf(stderr,"[client] IPC: %p\n",ipc);
  DevBuf dst(kMaxBuf);

  BenchPoint cm[kNumSizes], dev[kNumSizes];
  for(int i=0;i<kNumSizes;++i) { auto& bs=kSizes[i];
    fprintf(stderr,"[client] sz=%zu...\n",bs.bytes);
    cm[i]=bench_cuda2(bs.bytes,bs.lat_n,bs.tp_n,ipc,dst.p);
    fprintf(stderr,"[client]   cuda done\n");
    dev[i]=bench_device(bs.bytes,bs.lat_n,bs.tp_n,ipc,dst.p,g0,g1);
    fprintf(stderr,"[client]   device done\n");
  }

  { std::ofstream rf(kDone);
    for(int i=0;i<kNumSizes;++i) rf<<cm[i].l<<" "<<cm[i].t<<" "<<dev[i].l<<" "<<dev[i].t<<"\n"; }
  sig(kDone);

  pr_hdr("P2P Copy Benchmark",g0,g1);
  for(int i=0;i<kNumSizes;++i) pr_row(kSizes[i].bytes,cm[i],dev[i]);
  pr_sweep(g1,ipc,dst.p);

  fprintf(stderr,"[client] done\n");
  GPU_RT_CHECK(gpuIpcCloseMemHandle(ipc));
}

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  try {
    const char* m=(argc>=2)?argv[1]:"server";
    if(strcmp(m,"server")==0)       run_server(0,1);
    else if(strcmp(m,"client")==0)  run_client(0,1);
    else { fprintf(stderr,"Usage: %s server|client\n",argv[0]); return 1; }
    printf("\n=== DONE ===\n");
    return 0;
  } catch(std::exception const& ex) { fprintf(stderr,"FATAL: %s\n",ex.what()); return 2; }
}

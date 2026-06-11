#include "../../../include/gpu_rt.h"
#include "../../backend/backend.h"
#include "../../backend/rdma_local_copy_backend.h"
#include "../../coll_types.h"
#include "../../lower.h"
#include "../../utils.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

using namespace UKernel::CCL;

int main(int argc, char** argv) {
  int gpu = 0, rank = 0, peer = 0;
  bool is_self = true;
  if (argc >= 2) {
    if (strcmp(argv[1], "client") == 0) { rank = 1; peer = 0; gpu = 1; is_self = false; }
    else if (strcmp(argv[1], "server") == 0) { rank = 0; peer = 1; gpu = 0; is_self = false; }
    // default/no-arg: self-peer test on GPU 0
  }

  fprintf(stderr, "[test] GPU=%d rank=%d peer=%d %s\n", gpu, rank, peer,
          is_self ? "(self-peer)" : "(P2P OOB)");

  GPU_RT_CHECK(gpuSetDevice(gpu));
  void* src; GPU_RT_CHECK(gpuMalloc(&src, 4096));
  void* dst; GPU_RT_CHECK(gpuMalloc(&dst, 4096));
  fprintf(stderr, "[test] buffers: src=%p dst=%p\n", src, dst);

  RdmaLocalCopyBackendConfig cfg;
  cfg.gpu_id = gpu; cfg.rank = rank; cfg.peer_rank = peer;
  RdmaLocalCopyBackend backend(cfg);
  fprintf(stderr, "[test] backend: %s\n", backend.name());

  TiledResult t;
  t.input_bytes = 4096; t.output_bytes = 4096;
  t.rank = 0; t.nranks = 1;
  Op o; o.kind = OpKind::Copy; o.bytes = 4096;
  t.ops.push_back(o); t.chunk_of.push_back(0); t.layers = {{0}};

  if (!is_self) fprintf(stderr, "[test] validating (OOB wait for peer)...\n");
  else fprintf(stderr, "[test] validating...\n");
  backend.validate(t, src, dst, nullptr);
  if (strcmp(backend.name(), "degraded") == 0) {
    fprintf(stderr, "[test] FAILED: backend degraded after validate\n");
    return 1;
  }
  fprintf(stderr, "[test] validated OK\n");

  // warmup
  for (int i = 0; i < 10; ++i) {
    OpBindings b; b.stream_index = 0; b.resolved_src = src; b.resolved_dst = dst;
    BackendToken tok;
    while (true) { tok = backend.submit(t.ops[0], b, src, dst, nullptr); if (tok.value != 0) break;
      BackendToken tmp; backend.drain(&tmp, 1); }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {}
  }
  fprintf(stderr, "[test] warmup done\n");

  // timed test
  auto t0 = std::chrono::steady_clock::now();
  int iters = 100;
  for (int i = 0; i < iters; ++i) {
    OpBindings b; b.stream_index = 0; b.resolved_src = src; b.resolved_dst = dst;
    BackendToken tok;
    while (true) { tok = backend.submit(t.ops[0], b, src, dst, nullptr); if (tok.value != 0) break;
      BackendToken tmp; backend.drain(&tmp, 1); }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {}
    if (out.failed) { fprintf(stderr, "[test] FAILED: CQ error on iter %d\n", i); return 1; }
  }
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  fprintf(stderr, "[test] PASSED: %.1f us/op\n", us);

  GPU_RT_CHECK(gpuFree(src));
  GPU_RT_CHECK(gpuFree(dst));
  return 0;
}

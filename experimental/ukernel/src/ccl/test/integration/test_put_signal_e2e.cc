// Fused PutSignal e2e: data + signal in one shot (IPC: shm ring write
// after copy; RDMA: write-with-imm). Verifies the core semantic — the
// peer observes the signal ONLY after the data has landed.
//
// Usage (two terminals):
//   server: ./test_put_signal_e2e --role=server --gpu=0 [--transport=rdma]
//   client: ./test_put_signal_e2e --role=client --gpu=1 [--transport=rdma]
#include "coll_types.h"
#include "gpu_rt.h"
#include "transport.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

using namespace UKernel;
using namespace UKernel::Transport;
static constexpr int kW = 2;
static std::string ga(int c, char** v, std::string n, std::string d) {
  for (int i = 1; i < c; ++i) {
    std::string s = v[i];
    if (s == n && i + 1 < c) return v[i + 1];
    if (s.rfind(n + "=", 0) == 0) return s.substr(n.size() + 1);
  }
  return d;
}
static int gi(int c, char** v, std::string n, int d) {
  return std::stoi(ga(c, v, n, std::to_string(d)));
}
static void cp(std::shared_ptr<Communicator> cm, int r, PeerTransportKind tpt) {
  int p = (r == 0) ? 1 : 0;
  // IPC paths are same-host only; skip them cross-node.
  if (cm->same_host(p)) {
    if (r < p) {
      cm->connect(p, PeerTransportKind::Ipc);
      cm->accept(p, PeerTransportKind::Ipc);
    } else {
      cm->accept(p, PeerTransportKind::Ipc);
      cm->connect(p, PeerTransportKind::Ipc);
    }
  }
  if (tpt == PeerTransportKind::Rdma) {
    if (r < p) {
      cm->connect(p, PeerTransportKind::Rdma);
      cm->accept(p, PeerTransportKind::Rdma);
    } else {
      cm->accept(p, PeerTransportKind::Rdma);
      cm->connect(p, PeerTransportKind::Rdma);
    }
  }
}

static void wait_put_rid(Communicator* comm, unsigned rid) {
  while (true) {
    CompletionResult r;
    if (comm->try_complete_put(&r, 1) && r.rid == rid) {
      if (r.failed) printf("  [WARN] put rid=%u completed FAILED\n", rid);
      return;
    }
    std::this_thread::yield();
  }
}
static void wait_sig_rid(Communicator* comm, unsigned rid) {
  while (true) {
    SignalCompletion s;
    if (comm->try_complete_sig_wait(&s, 1) && s.rid == rid) return;
    std::this_thread::yield();
  }
}

int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  auto role = ga(argc, argv, "--role", "");
  if (role != "server" && role != "client") return 1;
  int rank = (role == "server") ? 0 : 1, peer = (rank == 0) ? 1 : 0,
      gpu = gi(argc, argv, "--gpu", rank == 0 ? 0 : 1);
  auto ip = (role == "server") ? "0.0.0.0"
                               : ga(argc, argv, "--exchanger-ip", "127.0.0.1");
  int port = gi(argc, argv, "--exchanger-port", 16998);
  auto transport_str = ga(argc, argv, "--transport", "ipc");
  auto tpt = (transport_str == "rdma") ? PeerTransportKind::Rdma
                                       : PeerTransportKind::Ipc;
  printf("PutSignal e2e %s r%d g%d transport=%s\n", role.c_str(), rank, gpu,
         transport_str.c_str());
  GPU_RT_CHECK(gpuSetDevice(gpu));
  auto cfg = std::make_shared<CommunicatorConfig>();
  cfg->exchanger_ip = ip;
  cfg->exchanger_port = port;
  cfg->local_id = gpu;
  auto comm = std::make_shared<Communicator>(gpu, rank, kW, cfg);
  cp(comm, rank, tpt);
  printf("  peer ok, can_fuse_put_signal=%d\n",
         (int)comm->can_fuse_put_signal(peer, tpt));
  if (!comm->can_fuse_put_signal(peer, tpt)) {
    printf("  [FAIL] transport cannot fuse put+signal\n");
    return 1;
  }

  constexpr size_t B = 64ULL << 20;
  constexpr size_t kChunk = 1ULL << 20;
  constexpr int kN = 50;
  void* d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d, B));
  comm->register_buffer(1, d, B);
  comm->resolve_remote_buffer(peer, 1, 30000);

  int const sender = 1;  // client sends first, then roles swap
  for (int phase = 0; phase < 2; ++phase) {
    int const pattern = 0x5A + phase;
    bool const am_sender = (rank == (sender ^ phase));
    uint64_t const tag_base = 1000 + phase * 1000;
    if (am_sender) {
      GPU_RT_CHECK(gpuMemset(d, pattern, B));
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < kN; ++i) {
        size_t off = (size_t)i * kChunk;
        unsigned rid = comm->alloc_rid();
        // Exercise group QP affinity: puts of consecutive groups pin to
        // different QPs (i % 4); ignored by IPC.
        if (!comm->send_put_signal_async_with_rid(peer, 1, off, 1, off, kChunk,
                                                  tpt, tag_base + i, rid,
                                                  (uint32_t)(i % 4))) {
          printf("  [FAIL] send_put_signal_async_with_rid i=%d\n", i);
          return 1;
        }
        wait_put_rid(comm.get(), rid);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      printf("  [PutSignal %zuKB x%d] %.1f us/op\n", kChunk >> 10, kN,
             std::chrono::duration<double, std::micro>(t1 - t0).count() / kN);
    } else {
      GPU_RT_CHECK(gpuMemset(d, 0, B));
      for (int i = 0; i < kN; ++i) {
        unsigned rid = comm->wait_signal_async(peer, tag_base + i, tpt);
        if (!rid) {
          printf("  [FAIL] wait_signal_async i=%d\n", i);
          return 1;
        }
        wait_sig_rid(comm.get(), rid);
      }
      // The signal was observed => the data must already be there.
      size_t const last = (size_t)(kN - 1) * kChunk;
      auto* chk = new uint8_t[kChunk];
      GPU_RT_CHECK(
          gpuMemcpy(chk, (char*)d + last, kChunk, gpuMemcpyDeviceToHost));
      size_t bad = 0;
      for (size_t i = 0; i < kChunk; ++i)
        if (chk[i] != (uint8_t)pattern) ++bad;
      printf("  data-after-signal: %s (bad=%zu)\n",
             bad == 0 ? "verified" : "MISMATCH", bad);
      delete[] chk;
      if (bad) return 1;
    }
  }
  printf("  [PASS]\n");
  GPU_RT_CHECK(gpuFree(d));
  return 0;
}

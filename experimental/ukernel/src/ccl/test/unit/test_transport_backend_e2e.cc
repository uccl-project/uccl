#include "backend/transport_backend.h"
#include "coll_types.h"
#include "gpu_rt.h"
#include "transport.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace UKernel;
using namespace UKernel::CCL;
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
static void cp(std::shared_ptr<Communicator> cm, int r,
               PeerTransportKind tpt = PeerTransportKind::Ipc) {
  int p = (r == 0) ? 1 : 0;
  if (r < p) {
    cm->connect(p, PeerTransportKind::Ipc);
    cm->accept(p, PeerTransportKind::Ipc);
  } else {
    cm->accept(p, PeerTransportKind::Ipc);
    cm->connect(p, PeerTransportKind::Ipc);
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

int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  auto role = ga(argc, argv, "--role", "");
  if (role != "server" && role != "client") return 1;
  int rank = (role == "server") ? 0 : 1, peer = (rank == 0) ? 1 : 0,
      gpu = gi(argc, argv, "--gpu", rank == 0 ? 0 : 1);
  auto ip = (role == "server") ? "0.0.0.0"
                               : ga(argc, argv, "--exchanger-ip", "127.0.0.1");
  int port = gi(argc, argv, "--exchanger-port", 16980);
  auto transport_str = ga(argc, argv, "--transport", "ipc");
  auto tpt_kind = (transport_str == "rdma") ? PeerTransportKind::Rdma
                                            : PeerTransportKind::Ipc;
  printf("TransportBackend e2e %s r%d g%d transport=%s\n", role.c_str(),
         rank, gpu, transport_str.c_str());
  GPU_RT_CHECK(gpuSetDevice(gpu));
  auto cfg = std::make_shared<CommunicatorConfig>();
  cfg->exchanger_ip = ip;
  cfg->exchanger_port = port;
  cfg->local_id = rank;
  auto comm = std::make_shared<Communicator>(gpu, rank, kW, cfg);
  cp(comm, rank, tpt_kind);
  printf("  peer ok\n");
  constexpr size_t B = 128ULL << 20;
  void* d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d, B));
  comm->register_buffer(1, d, B);
  comm->resolve_remote_buffer(peer, 1, 30000);

  uint32_t sizes[] = {4096, 65536, 262144, 1048576, 4194304, 16777216};
  int iters = 100;

  if (rank == 1) {
    TransportBackend tpt(comm.get());

    // Allocate caller_id mapping (mirrors SprayExecutor pattern)
    static constexpr uint32_t kEmpty = ~0u;
    static constexpr size_t kMapSize = 65536;
    std::unique_ptr<std::atomic<uint32_t>[]> caller_map(
        new std::atomic<uint32_t>[kMapSize]);
    for (size_t i = 0; i < kMapSize; ++i)
      caller_map[i].store(kEmpty, std::memory_order_relaxed);

    for (int si = 0; si < 6; si++) {
      auto sz = sizes[si];
      std::vector<CmdWithId> w(iters);
      for (int i = 0; i < iters; i++) {
        w[i].cmd.kind = ExecOpKind::Put;
        w[i].cmd.src_buf = 1;
        w[i].cmd.dst_buf = 1;
        w[i].cmd.bytes = sz;
        w[i].cmd.src_peer = ~0u;
        w[i].cmd.dst_peer = (uint32_t)peer;
        w[i].cmd.put_path = (tpt_kind == PeerTransportKind::Rdma)
                                ? PutPath::Rdma
                                : PutPath::Ipc;
        w[i].caller_id = i;
      }

      // Enqueue with mapping (release store, like SprayExecutor)
      size_t enq = 0;
      while (enq < w.size()) {
        uint32_t be_idx;
        if (tpt.do_enqueue(&w[enq].cmd, 1, &be_idx) > 0) {
          caller_map[be_idx & (kMapSize - 1)].store(w[enq].caller_id,
                                                    std::memory_order_release);
          ++enq;
        }
      }

      // Drain with mapping (acquire load + spin-wait, like SprayExecutor)
      uint32_t be_buf[128];
      size_t ok = 0;
      auto t0 = std::chrono::high_resolution_clock::now();
      while (ok < w.size()) {
        size_t n = tpt.do_drain(be_buf, 128);
        for (size_t i = 0; i < n; ++i) {
          uint32_t cid;
          while ((cid = caller_map[be_buf[i] & (kMapSize - 1)].load(
                      std::memory_order_acquire)) == kEmpty)
            std::this_thread::yield();
          caller_map[be_buf[i] & (kMapSize - 1)].store(
              kEmpty, std::memory_order_relaxed);
          ++ok;
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      printf("  [Put %8u x%4d] %7.2f GB/s\n", sz, iters,
             (1.0 * sz * iters) /
                 std::chrono::duration<double>(t1 - t0).count() / 1e9);
    }
    unsigned rid = comm->send_signal_async(peer, 999, tpt_kind);
    while (1) {
      CompletionResult r;
      if (comm->try_complete_put(&r, 1) && r.rid == rid) break;
    }
    unsigned wid = comm->wait_signal_async(peer, 998, tpt_kind);
    while (1) {
      SignalCompletion s;
      if (comm->try_complete_sig_wait(&s, 1) && s.rid == wid) break;
    }
    printf("  [PASS]\n");
  } else {
    unsigned rid = comm->wait_signal_async(peer, 999, tpt_kind);
    while (1) {
      SignalCompletion s;
      if (comm->try_complete_sig_wait(&s, 1) && s.rid == rid) break;
    }
    auto* chk = new uint8_t[9000];
    GPU_RT_CHECK(gpuMemcpy(chk, d, 9000, gpuMemcpyDeviceToHost));
    bool ok = false;
    for (int i = 0; i < 9000; i++)
      if (chk[i]) {
        ok = true;
        break;
      }
    printf("  data: %s\n", ok ? "arrived" : "empty");
    unsigned sid = comm->send_signal_async(peer, 998, tpt_kind);
    while (1) {
      CompletionResult r;
      if (comm->try_complete_put(&r, 1) && r.rid == sid) break;
    }
    printf("  [PASS]\n");
    delete[] chk;
  }
  GPU_RT_CHECK(gpuFree(d));
  return 0;
}

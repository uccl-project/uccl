#include "backend/signal_backend.h"
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
static void cp(std::shared_ptr<Communicator> cm, int r) {
  int p = (r == 0) ? 1 : 0;
  if (r < p) {
    cm->connect(p, PeerTransportKind::Ipc);
    cm->accept(p, PeerTransportKind::Ipc);
  } else {
    cm->accept(p, PeerTransportKind::Ipc);
    cm->connect(p, PeerTransportKind::Ipc);
  }
  cm->connect(p, PeerTransportKind::Rdma);
}

int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  auto role = ga(argc, argv, "--role", "");
  if (role != "server" && role != "client") return 1;
  int rank = (role == "server") ? 0 : 1, peer = (rank == 0) ? 1 : 0,
      gpu = gi(argc, argv, "--gpu", rank == 0 ? 0 : 1);
  auto ip = (role == "server") ? "0.0.0.0"
                               : ga(argc, argv, "--exchanger-ip", "127.0.0.1");
  int port = gi(argc, argv, "--exchanger-port", 16982);
  printf("SignalBackend %s r%d g%d\n", role.c_str(), rank, gpu);
  GPU_RT_CHECK(gpuSetDevice(gpu));
  auto cfg = std::make_shared<CommunicatorConfig>();
  cfg->exchanger_ip = ip;
  cfg->exchanger_port = port;
  cfg->local_id = rank;
  auto comm = std::make_shared<Communicator>(gpu, rank, kW, cfg);
  cp(comm, rank);
  printf("  peer ok\n");

  SignalBackend sig;
  sig.set_comm(comm.get());
  int const N = 100;

  // Allocate caller_id mapping (mirrors SprayExecutor pattern)
  static constexpr uint32_t kEmpty = ~0u;
  static constexpr size_t kMapSize = 65536;
  std::unique_ptr<std::atomic<uint32_t>[]> caller_map(
      new std::atomic<uint32_t>[kMapSize]);
  for (size_t i = 0; i < kMapSize; ++i)
    caller_map[i].store(kEmpty, std::memory_order_relaxed);

  if (rank == 1) {
    std::vector<CmdWithId> snd(N);
    for (int i = 0; i < N; i++) {
      snd[i].cmd.kind = ExecOpKind::Signal;
      snd[i].cmd.dst_peer = (uint32_t)peer;
      snd[i].cmd.tag = 100 + i;
      snd[i].caller_id = i;
    }

    // Enqueue with mapping
    size_t enq = 0;
    while (enq < N) {
      uint32_t be_idx;
      if (sig.do_enqueue(&snd[enq].cmd, 1, &be_idx) > 0) {
        caller_map[be_idx & (kMapSize - 1)].store(snd[enq].caller_id,
                                                  std::memory_order_release);
        ++enq;
      }
    }

    // Drain with mapping
    uint32_t be_buf[128];
    size_t tot = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    while (tot < N) {
      size_t n = sig.do_drain(be_buf, 128);
      for (size_t i = 0; i < n; ++i) {
        uint32_t cid;
        while ((cid = caller_map[be_buf[i] & (kMapSize - 1)].load(
                    std::memory_order_acquire)) == kEmpty)
          std::this_thread::yield();
        caller_map[be_buf[i] & (kMapSize - 1)].store(kEmpty,
                                                     std::memory_order_relaxed);
        ++tot;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf(
        "  [SignalBatch x%d] %.0f us  %.1f us/sig\n", N,
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
            .count(),
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count() /
            N);

    std::vector<CmdWithId> wt(N);
    for (int i = 0; i < N; i++) {
      wt[i].cmd.kind = ExecOpKind::WaitSignal;
      wt[i].cmd.src_peer = (uint32_t)peer;
      wt[i].cmd.tag = 200 + i;
      wt[i].caller_id = 100 + i;
    }

    enq = 0;
    while (enq < N) {
      uint32_t be_idx;
      if (sig.do_enqueue(&wt[enq].cmd, 1, &be_idx) > 0) {
        caller_map[be_idx & (kMapSize - 1)].store(wt[enq].caller_id,
                                                  std::memory_order_release);
        ++enq;
      }
    }

    tot = 0;
    t0 = std::chrono::high_resolution_clock::now();
    while (tot < N) {
      size_t n = sig.do_drain(be_buf, 128);
      for (size_t i = 0; i < n; ++i) {
        uint32_t cid;
        while ((cid = caller_map[be_buf[i] & (kMapSize - 1)].load(
                    std::memory_order_acquire)) == kEmpty)
          std::this_thread::yield();
        caller_map[be_buf[i] & (kMapSize - 1)].store(kEmpty,
                                                     std::memory_order_relaxed);
        ++tot;
      }
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf(
        "  [WaitBatch  x%d] %.0f us  %.1f us/wait\n  [PASS]\n", N,
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
            .count(),
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count() /
            N);
  } else {
    std::vector<CmdWithId> wt(N);
    for (int i = 0; i < N; i++) {
      wt[i].cmd.kind = ExecOpKind::WaitSignal;
      wt[i].cmd.src_peer = (uint32_t)peer;
      wt[i].cmd.tag = 100 + i;
      wt[i].caller_id = i;
    }

    size_t enq = 0;
    while (enq < N) {
      uint32_t be_idx;
      if (sig.do_enqueue(&wt[enq].cmd, 1, &be_idx) > 0) {
        caller_map[be_idx & (kMapSize - 1)].store(wt[enq].caller_id,
                                                  std::memory_order_release);
        ++enq;
      }
    }

    uint32_t be_buf[128];
    size_t tot = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    while (tot < N) {
      size_t n = sig.do_drain(be_buf, 128);
      for (size_t i = 0; i < n; ++i) {
        uint32_t cid;
        while ((cid = caller_map[be_buf[i] & (kMapSize - 1)].load(
                    std::memory_order_acquire)) == kEmpty)
          std::this_thread::yield();
        caller_map[be_buf[i] & (kMapSize - 1)].store(kEmpty,
                                                     std::memory_order_relaxed);
        ++tot;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf(
        "  [WaitBatch  x%d] %.0f us  %.1f us/wait\n", N,
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
            .count(),
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count() /
            N);

    std::vector<CmdWithId> snd(N);
    for (int i = 0; i < N; i++) {
      snd[i].cmd.kind = ExecOpKind::Signal;
      snd[i].cmd.dst_peer = (uint32_t)peer;
      snd[i].cmd.tag = 200 + i;
      snd[i].caller_id = 100 + i;
    }

    enq = 0;
    while (enq < N) {
      uint32_t be_idx;
      if (sig.do_enqueue(&snd[enq].cmd, 1, &be_idx) > 0) {
        caller_map[be_idx & (kMapSize - 1)].store(snd[enq].caller_id,
                                                  std::memory_order_release);
        ++enq;
      }
    }

    tot = 0;
    t0 = std::chrono::high_resolution_clock::now();
    while (tot < N) {
      size_t n = sig.do_drain(be_buf, 128);
      for (size_t i = 0; i < n; ++i) {
        uint32_t cid;
        while ((cid = caller_map[be_buf[i] & (kMapSize - 1)].load(
                    std::memory_order_acquire)) == kEmpty)
          std::this_thread::yield();
        caller_map[be_buf[i] & (kMapSize - 1)].store(kEmpty,
                                                     std::memory_order_relaxed);
        ++tot;
      }
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf(
        "  [SignalBatch x%d] %.0f us  %.1f us/sig\n  [PASS]\n", N,
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
            .count(),
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count() /
            N);
  }
  return 0;
}

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
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (true) {
    CompletionResult r;
    if (comm->try_complete_put(&r, 1) && r.rid == rid) {
      if (r.failed) printf("  [WARN] put rid=%u completed FAILED\n", rid);
      return;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      printf("  [FAIL] put rid=%u timed out\n", rid);
      std::exit(2);
    }
    std::this_thread::yield();
  }
}
static void wait_sig_rid(Communicator* comm, unsigned rid) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (true) {
    SignalCompletion s;
    if (comm->try_complete_sig_wait(&s, 1) && s.rid == rid) return;
    if (std::chrono::steady_clock::now() >= deadline) {
      printf("  [FAIL] sig wait rid=%u timed out\n", rid);
      std::exit(2);
    }
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
  // RDMA fused puts deliver the tag as a 32-bit write-with-imm, which the
  // wait must match on the immediate path (force_imm). IPC fused puts go
  // through the 64-bit shm signal ring (plain wait).
  auto wait_fused = [&](int peer, uint64_t tag, unsigned rid,
                        uint32_t count = 1) {
    return comm->wait_signal_async_with_rid(
        peer, tag, tpt, rid, count, tpt == PeerTransportKind::Rdma);
  };
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
  // Ensure both sides have finished local buffer setup before either
  // starts memset/send, otherwise the receiver's zero-fill can race the
  // sender's first RDMA writes.
  if (!comm->barrier("put_signal_ready", 30000)) {
    fprintf(stderr, "[FAIL] barrier before phase loop failed\n");
    return 1;
  }

  int const sender = 1;  // client sends first, then roles swap
  for (int phase = 0; phase < 2; ++phase) {
    int const pattern = 0x5A + phase;
    bool const am_sender = (rank == (sender ^ phase));
    uint64_t const tag_base = 1000 + phase * 1000;
    if (am_sender) {
      GPU_RT_CHECK(gpuMemset(d, pattern, B));
      // gpuMemset runs on the legacy default stream while IPC copies run
      // on the adapter's non-blocking streams; sync so the copy sources
      // are stable before the first put is launched.
      GPU_RT_CHECK(gpuDeviceSynchronize());
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < kN; ++i) {
        size_t off = (size_t)i * kChunk;
        unsigned rid = comm->alloc_rid();
        // Fixed QP affinity: the receiver matches write-with-imm
        // arrivals per-peer FIFO in arrival order, so fused puts to one
        // peer must share a QP (the executor pins per peer). Ignored by
        // IPC.
        if (!comm->send_put_signal_async_with_rid(peer, 1, off, 1, off, kChunk,
                                                  tpt, tag_base + i, rid, 0)) {
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
      // The local pre-fill must be fully visible before any peer copy can
      // land (the peer's put completion only orders the copy against the
      // signal, not against this rank's own default-stream memset).
      GPU_RT_CHECK(gpuDeviceSynchronize());
      for (int i = 0; i < kN; ++i) {
        unsigned rid = comm->alloc_rid();
        if (!wait_fused(peer, tag_base + i, rid)) {
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

    // Section 1: cross-run identical tags. Two batches reuse the same tag
    // range (immediates carry only the unsalted low 32 bits, so "runs"
    // are indistinguishable by tag). The receiver must pair arrivals
    // with waits strictly in issue order — verified by writing each
    // batch to distinct offsets with distinct patterns and checking the
    // batch's OWN chunk right after each wait fires.
    {
      uint64_t const xtag = 2000;
      constexpr int kB = 10;
      for (int batch = 0; batch < 2; ++batch) {
        uint8_t const bp = (uint8_t)(0xA0 + 0x10 * batch + phase);
        size_t const base_off = (size_t)batch * kB * kChunk;
        if (am_sender) {
          GPU_RT_CHECK(gpuMemset((char*)d + base_off, bp, kB * kChunk));
          GPU_RT_CHECK(gpuDeviceSynchronize());
          for (int i = 0; i < kB; ++i) {
            size_t const off = base_off + (size_t)i * kChunk;
            unsigned rid = comm->alloc_rid();
            if (!comm->send_put_signal_async_with_rid(peer, 1, off, 1, off,
                                                      kChunk, tpt, xtag + i,
                                                      rid, 0)) {
              printf("  [FAIL] xrun send batch=%d i=%d\n", batch, i);
              return 1;
            }
            wait_put_rid(comm.get(), rid);
          }
        } else {
          auto* chk = new uint8_t[kChunk];
          for (int i = 0; i < kB; ++i) {
            unsigned rid = comm->alloc_rid();
            if (!wait_fused(peer, xtag + i, rid)) {
              printf("  [FAIL] xrun wait batch=%d i=%d\n", batch, i);
              return 1;
            }
            wait_sig_rid(comm.get(), rid);
            size_t const off = base_off + (size_t)i * kChunk;
            GPU_RT_CHECK(
                gpuMemcpy(chk, (char*)d + off, kChunk, gpuMemcpyDeviceToHost));
            size_t bad = 0;
            for (size_t k = 0; k < kChunk; ++k)
              if (chk[k] != bp) ++bad;
            if (bad) {
              printf("  [FAIL] xrun data batch=%d i=%d bad=%zu\n", batch, i,
                     bad);
              return 1;
            }
          }
          delete[] chk;
        }
      }
      if (!am_sender) printf("  cross-run same-tag FIFO: verified\n");
    }

    // Section 2: counted wait — kC fused puts carry the SAME tag, one
    // wait counts kC arrivals (the fused signal-group shape).
    {
      uint64_t const ctag = 3000 + phase;
      constexpr int kC = 4;
      size_t const base_off = 32ULL * kChunk;
      uint8_t const cpatt = (uint8_t)(0xC0 + phase);
      if (am_sender) {
        GPU_RT_CHECK(gpuMemset((char*)d + base_off, cpatt, kC * kChunk));
        GPU_RT_CHECK(gpuDeviceSynchronize());
        for (int i = 0; i < kC; ++i) {
          size_t const off = base_off + (size_t)i * kChunk;
          unsigned rid = comm->alloc_rid();
          if (!comm->send_put_signal_async_with_rid(peer, 1, off, 1, off,
                                                    kChunk, tpt, ctag, rid,
                                                    0)) {
            printf("  [FAIL] counted send i=%d\n", i);
            return 1;
          }
          wait_put_rid(comm.get(), rid);
        }
      } else {
        unsigned rid = comm->alloc_rid();
        if (!wait_fused(peer, ctag, rid, kC)) {
          printf("  [FAIL] counted wait\n");
          return 1;
        }
        wait_sig_rid(comm.get(), rid);
        auto* chk = new uint8_t[kC * kChunk];
        GPU_RT_CHECK(gpuMemcpy(chk, (char*)d + base_off, kC * kChunk,
                               gpuMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t k = 0; k < kC * kChunk; ++k)
          if (chk[k] != cpatt) ++bad;
        delete[] chk;
        if (bad) {
          printf("  [FAIL] counted data bad=%zu\n", bad);
          return 1;
        }
        printf("  counted wait (count=%d): verified\n", kC);
      }
    }
  }
  printf("  [PASS]\n");
  GPU_RT_CHECK(gpuFree(d));
  return 0;
}

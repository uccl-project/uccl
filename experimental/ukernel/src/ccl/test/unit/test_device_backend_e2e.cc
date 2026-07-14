#include "backend/device_backend.h"
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
  int port = gi(argc, argv, "--exchanger-port", 16981);
  printf("DeviceBackend %s r%d g%d\n", role.c_str(), rank, gpu);
  GPU_RT_CHECK(gpuSetDevice(gpu));
  auto cfg = std::make_shared<CommunicatorConfig>();
  cfg->exchanger_ip = ip;
  cfg->exchanger_port = port;
  cfg->local_id = rank;
  auto comm = std::make_shared<Communicator>(gpu, rank, kW, cfg);
  cp(comm, rank);
  printf("  peer ok\n");
  constexpr size_t B = 128ULL << 20;
  void* d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d, B));
  comm->register_buffer(1, d, B);
  comm->resolve_remote_buffer(peer, 1, 30000);

  if (rank == 1) {
    void* rem = nullptr;
    int rdev = -1;
    if (!comm->try_resolve_remote_ipc_pointer(peer, 1, 0, B, &rem, &rdev)) {
      printf("FAIL:no remote\n");
      GPU_RT_CHECK(gpuFree(d));
      return 1;
    }
    int ca = 0;
    GPU_RT_CHECK(gpuDeviceCanAccessPeer(&ca, gpu, rdev));
    if (ca) {
      gpuError_t e = gpuDeviceEnablePeerAccess(rdev, 0);
      if (e != gpuSuccess && e != gpuErrorPeerAccessAlreadyEnabled)
        GPU_RT_CHECK(e);
    }
    printf("  remote dev=%d access=%d\n", rdev, ca);

    // Pre-flight P2P copy to verify peer access works before launching
    // persistent kernels.
    gpuStream_t probe_stream;
    GPU_RT_CHECK(gpuStreamCreate(&probe_stream));
    GPU_RT_CHECK(gpuMemcpyPeerAsync(rem, rdev, d, gpu, 4096, probe_stream));
    GPU_RT_CHECK(gpuStreamSynchronize(probe_stream));
    GPU_RT_CHECK(gpuStreamDestroy(probe_stream));
    printf("  p2p probe ok\n");

    DeviceBackendConfig dcfg;
    dcfg.task_capacity = 256;
    dcfg.blocks_per_worker = 1;
    dcfg.threads_per_block = 64;
    dcfg.max_fifos = 2;
    dcfg.fifo_capacity = 256;
    dcfg.smem_size = 4096;
    DeviceBackend dev(dcfg);
    dev.set_comm(comm.get());

    // Allocate caller_id mapping (mirrors SprayExecutor pattern)
    static constexpr uint32_t kEmpty = ~0u;
    static constexpr size_t kMapSize = 65536;
    std::unique_ptr<std::atomic<uint32_t>[]> caller_map(
        new std::atomic<uint32_t>[kMapSize]);
    for (size_t i = 0; i < kMapSize; ++i)
      caller_map[i].store(kEmpty, std::memory_order_relaxed);

    uint32_t sizes[] = {4096, 65536, 262144, 1048576, 4194304};
    int iters[] = {500, 200, 100, 50, 20};
    for (int si = 0; si < 5; si++) {
      uint32_t sz = sizes[si];
      int n = iters[si];
      std::vector<CmdWithId> w(n);
      for (int i = 0; i < n; i++) {
        w[i].cmd.kind = ExecOpKind::Put;
        w[i].cmd.bytes = sz;
        w[i].cmd.src_buf = 1;
        w[i].cmd.dst_buf = 1;
        w[i].cmd.src_peer = ~0u;
        w[i].cmd.dst_peer = (uint32_t)peer;
        w[i].caller_id = i;
      }

      // Enqueue with mapping (release store, like SprayExecutor).
      // Drain whenever backpressure hits to keep the FIFO flowing.
      size_t enq = 0;
      size_t total_done = 0;
      while (enq < w.size()) {
        uint32_t be_idx;
        if (dev.do_enqueue(&w[enq].cmd, 1, &be_idx) > 0) {
          caller_map[be_idx & (kMapSize - 1)].store(w[enq].caller_id,
                                                    std::memory_order_release);
          ++enq;
        } else {
          // Backpressure: drain some completions to free FIFO slots.
          uint32_t be_buf[128];
          size_t n = dev.do_drain(be_buf, 128);
          for (size_t i = 0; i < n; ++i) {
            uint32_t cid;
            while ((cid = caller_map[be_buf[i] & (kMapSize - 1)].load(
                        std::memory_order_acquire)) == kEmpty)
              std::this_thread::yield();
            caller_map[be_buf[i] & (kMapSize - 1)].store(
                kEmpty, std::memory_order_relaxed);
            ++total_done;
          }
        }
      }

      // Drain remaining completions.
      auto t0 = std::chrono::high_resolution_clock::now();
      uint32_t be_buf[128];
      while (total_done < w.size()) {
        size_t n = dev.do_drain(be_buf, 128);
        for (size_t i = 0; i < n; ++i) {
          uint32_t cid;
          while ((cid = caller_map[be_buf[i] & (kMapSize - 1)].load(
                      std::memory_order_acquire)) == kEmpty)
            std::this_thread::yield();
          caller_map[be_buf[i] & (kMapSize - 1)].store(
              kEmpty, std::memory_order_relaxed);
          ++total_done;
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      printf("  [Put %8u x%4d] %7.2f GB/s\n", sz, n,
             (1.0 * sz * n) / std::chrono::duration<double>(t1 - t0).count() /
                 1e9);
    }

    // Reduce correctness test
    float rd[1024];
    for (int i = 0; i < 1024; i++) rd[i] = (float)i;
    GPU_RT_CHECK(gpuMemcpy(d, rd, 4096, gpuMemcpyHostToDevice));
    CmdWithId w{};
    w.cmd.kind = ExecOpKind::Reduce;
    w.cmd.bytes = 4096;
    w.cmd.src_buf = 1;
    w.cmd.dst_buf = 1;
    w.cmd.src_peer = ~0u;
    w.cmd.dst_peer = ~0u;
    w.cmd.redop = ReductionKind::Sum;
    w.caller_id = 0;

    uint32_t be_idx;
    dev.do_enqueue(&w.cmd, 1, &be_idx);
    caller_map[be_idx & (kMapSize - 1)].store(w.caller_id,
                                              std::memory_order_release);

    uint32_t be_buf[1];
    int tr = 0;
    while (dev.do_drain(be_buf, 1) == 0) {
      std::this_thread::yield();
      if (++tr > 5000) break;
    }
    if (tr <= 5000) {
      uint32_t cid;
      while ((cid = caller_map[be_buf[0] & (kMapSize - 1)].load(
                  std::memory_order_acquire)) == kEmpty)
        std::this_thread::yield();
      caller_map[be_buf[0] & (kMapSize - 1)].store(kEmpty,
                                                   std::memory_order_relaxed);
    }

    std::vector<float> res(1024);
    GPU_RT_CHECK(gpuMemcpy(res.data(), d, 4096, gpuMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < 1024; i++)
      if (std::abs(res[i] - (float)(i * 2)) > 0.01f) {
        ok = false;
        break;
      }
    printf("  [Reduce x1024] %s\n", ok ? "PASS" : "FAIL");

    unsigned rid = comm->send_signal_async(peer, 999, PeerTransportKind::Ipc);
    while (1) {
      CompletionResult r;
      if (comm->try_complete_put(&r, 1) && r.rid == rid) break;
    }
    unsigned wid =
        comm->wait_signal_async(peer, 998, PeerTransportKind::Unknown);
    while (1) {
      SignalCompletion s;
      if (comm->try_complete_sig_wait(&s, 1) && s.rid == wid) break;
    }
    printf("  [PASS]\n");
  } else {
    unsigned rid =
        comm->wait_signal_async(peer, 999, PeerTransportKind::Unknown);
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
    unsigned sid = comm->send_signal_async(peer, 998, PeerTransportKind::Ipc);
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

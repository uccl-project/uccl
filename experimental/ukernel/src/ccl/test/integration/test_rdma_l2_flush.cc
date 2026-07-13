// Test: verify GPU L2 cache coherence after RDMA write.
//
// Rank 0 writes a known float pattern via RDMA into rank 1's GPU buffer.
// Rank 1 waits for the signal, then uses DeviceBackend (SM CollCopy) to
// read the data into a second buffer and verifies it on the host.
// No SprayExecutor, no GDR flush — isolates RDMA write + SM kernel read.
//
// Run:
//   server: CUDA_VISIBLE_DEVICES=0,1 ./test_rdma_l2_flush --role=server --gpu=0
//   client: CUDA_VISIBLE_DEVICES=0,1 ./test_rdma_l2_flush --role=client --gpu=1
//
// IPC (same-host) should pass.  RDMA may fail on pre-Hopper GPUs due to
// stale L2 cache lines after the NIC writes directly to GPU DRAM.

#include "backend/device_backend.h"
#include "executor.h"
#include "gpu_rt.h"
#include "transport.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>

using namespace UKernel::CCL;
using UKernel::Transport::Communicator;
using UKernel::Transport::CommunicatorConfig;
using UKernel::Transport::PeerTransportKind;
using UKernel::Transport::CompletionResult;
using UKernel::Transport::SignalCompletion;

static std::string get_arg(int argc, char** argv, std::string const& name,
                           std::string const& def) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == name && i + 1 < argc) return argv[i + 1];
    std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
  }
  return def;
}

static int get_int_arg(int argc, char** argv, std::string const& name,
                       int def) {
  std::string s = get_arg(argc, argv, name, std::to_string(def));
  try {
    return std::stoi(s);
  } catch (...) {
    return def;
  }
}

// Register both buffers on both ranks.  Buffer 1 = send/recv, buffer 2 = verify.
static void setup_buffers(Communicator* comm, void* buf1, void* buf2,
                          size_t bytes, int rank, int world_size) {
  comm->reg_ipc(1, buf1, bytes, true);
  comm->reg_ipc(2, buf2, bytes, true);

  for (int p = 0; p < world_size; ++p) {
    if (p == rank) continue;
    if (!comm->resolve_remote_buffer(p, 1, 30000))
      throw std::runtime_error("resolve_remote_buffer(1) failed");
    if (!comm->resolve_remote_buffer(p, 2, 30000))
      throw std::runtime_error("resolve_remote_buffer(2) failed");
  }
}

// Spin-wait for a specific rid completion.
static void wait_completion(Communicator* comm, unsigned rid) {
  CompletionResult res;
  while (true) {
    size_t n = comm->try_complete(&res, 1);
    if (n > 0 && res.rid == rid) {
      if (res.failed) throw std::runtime_error("RDMA put failed");
      return;
    }
    std::this_thread::yield();
  }
}

// Spin-wait for a specific signal.
static void wait_signal(Communicator* comm, unsigned signal_rid) {
  SignalCompletion ev;
  while (true) {
    size_t n = comm->try_complete_signals(&ev, 1);
    if (n > 0 && ev.rid == signal_rid) {
      if (ev.failed) throw std::runtime_error("wait_signal failed");
      return;
    }
    std::this_thread::yield();
  }
}

int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  std::string role = get_arg(argc, argv, "--role", "");
  if (role != "server" && role != "client") {
    std::fprintf(stderr,
                 "Usage: --role=server|client [--gpu GPU] "
                 "[--transport ipc|rdma] [--exchanger-port PORT]\n");
    return 1;
  }

  int rank = (role == "server") ? 0 : 1;
  int gpu = get_int_arg(argc, argv, "--gpu", rank);
  int port = get_int_arg(argc, argv, "--exchanger-port", 16999);
  std::string transport_str = get_arg(argc, argv, "--transport", "rdma");
  PeerTransportKind transport_kind =
      (transport_str == "ipc") ? PeerTransportKind::Ipc
                               : PeerTransportKind::Rdma;

  std::printf("[l2flush] %s rank=%d gpu=%d transport=%s\n", role.c_str(), rank,
              gpu, transport_str.c_str());

  GPU_RT_CHECK(gpuSetDevice(gpu));

  // --- Communicator setup ---
  SprayExecutorConfig cfg;
  cfg.gpu_id = gpu;
  cfg.rank = rank;
  cfg.world_size = 2;
  cfg.exchanger_ip = (rank == 0) ? "0.0.0.0" : "127.0.0.1";
  cfg.exchanger_port = port;
  cfg.local_id = rank;
  auto comm_cfg = std::make_shared<CommunicatorConfig>();
  comm_cfg->exchanger_ip = cfg.exchanger_ip;
  comm_cfg->exchanger_port = cfg.exchanger_port;
  comm_cfg->local_id = cfg.local_id;
  auto comm = std::make_shared<Communicator>(cfg.gpu_id, cfg.rank,
                                             cfg.world_size, comm_cfg);

  // Connect RDMA (and IPC) paths
  for (int p = 0; p < 2; ++p) {
    if (p == rank) continue;
    if (transport_kind == PeerTransportKind::Ipc || comm->same_host(p)) {
      comm->connect(p, PeerTransportKind::Ipc);
      comm->accept(p, PeerTransportKind::Ipc);
    }
    comm->connect(p, PeerTransportKind::Rdma);
    comm->accept(p, PeerTransportKind::Rdma);
  }

  // --- GPU buffers ---
  constexpr size_t kBufBytes = 65536;  // one tile, ≤ BAR1 page
  constexpr size_t kFloats = kBufBytes / sizeof(float);
  void *d_send = nullptr, *d_recv = nullptr, *d_verify = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d_send, kBufBytes));
  GPU_RT_CHECK(gpuMalloc(&d_recv, kBufBytes));
  GPU_RT_CHECK(gpuMalloc(&d_verify, kBufBytes));

  // Register: id=1 for send/recv, id=2 for verify
  if (rank == 0) {
    setup_buffers(comm.get(), d_send, d_verify, kBufBytes, rank, 2);
  } else {
    setup_buffers(comm.get(), d_recv, d_verify, kBufBytes, rank, 2);
  }

  // --- Rank 0: fill pattern, RDMA put, signal ---
  if (rank == 0) {
    std::vector<float> host_send(kFloats);
    for (size_t i = 0; i < kFloats; ++i) host_send[i] = static_cast<float>(i + 1);
    GPU_RT_CHECK(
        gpuMemcpy(d_send, host_send.data(), kBufBytes, gpuMemcpyHostToDevice));

    std::printf("[l2flush] rank0: RDMA put %zu bytes to rank1\n", kBufBytes);
    unsigned put_rid = comm->send_put_async(1, 1, 0, 1, 0, kBufBytes,
                                            PeerTransportKind::Rdma);
    wait_completion(comm.get(), put_rid);
    std::printf("[l2flush] rank0: RDMA put done\n");

    unsigned sig_rid = comm->send_signal_async(1, 42, PeerTransportKind::Rdma);
    wait_signal(comm.get(), sig_rid);
    std::printf("[l2flush] rank0: signal sent\n");
  }

  // --- Rank 1: wait signal, SM CollCopy, verify ---
  if (rank == 1) {
    std::printf("[l2flush] rank1: waiting for signal...\n");
    unsigned signal_rid =
        comm->wait_signal_async(0, 42, PeerTransportKind::Rdma);
    wait_signal(comm.get(), signal_rid);
    std::printf("[l2flush] rank1: signal received\n");

    // SM CollCopy: read d_recv(id=1) into d_verify(id=2)
    DeviceBackendConfig dev_cfg;
    dev_cfg.max_fifos = 1;
    dev_cfg.fifo_capacity = 16;
    DeviceBackend dev_be(dev_cfg);
    dev_be.set_comm(comm.get());

    Cmd cmd{};
    cmd.kind = ExecOpKind::Put;
    cmd.src_buf = 1;
    cmd.dst_buf = 2;
    cmd.bytes = static_cast<uint32_t>(kBufBytes);
    cmd.src_peer = ~0u;
    cmd.dst_peer = ~0u;

    uint32_t be_idx = 0;
    size_t accepted = dev_be.do_enqueue(&cmd, 1, &be_idx);
    if (accepted != 1) {
      std::fprintf(stderr, "[l2flush] rank1: CollCopy enqueue failed\n");
      return 1;
    }
    std::printf("[l2flush] rank1: CollCopy enqueued\n");

    // Drain
    uint32_t completed = 0;
    while (dev_be.do_drain(&completed, 1) == 0) {
      std::this_thread::yield();
    }
    std::printf("[l2flush] rank1: CollCopy done\n");

    // Verify on host
    std::vector<float> host_verify(kFloats);
    GPU_RT_CHECK(gpuMemcpy(host_verify.data(), d_verify, kBufBytes,
                           gpuMemcpyDeviceToHost));

    bool pass = true;
    for (size_t i = 0; i < kFloats; ++i) {
      float expected = static_cast<float>(i + 1);
      if (std::fabs(host_verify[i] - expected) > 1e-5f) {
        std::fprintf(stderr,
                     "[l2flush] MISMATCH at [%zu]: got %.1f want %.1f\n", i,
                     host_verify[i], expected);
        pass = false;
        break;
      }
    }
    if (pass) {
      std::printf("[l2flush] rank1: PASSED — all %zu floats match\n", kFloats);
    } else {
      std::printf("[l2flush] rank1: FAILED\n");
    }
    fflush(stdout);
    return pass ? 0 : 1;
  }

  // Wait for rank 1 to finish before exiting
  if (rank == 0) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  GPU_RT_CHECK(gpuFree(d_send));
  GPU_RT_CHECK(gpuFree(d_recv));
  GPU_RT_CHECK(gpuFree(d_verify));
  std::printf("[l2flush] done\n");
  return 0;
}

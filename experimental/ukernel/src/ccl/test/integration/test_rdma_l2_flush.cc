// Test: verify GPU L2 cache coherence after RDMA write.
//
// Rank 0 writes a known float pattern via RDMA into rank 1's GPU buffer.
// Rank 1 waits for the signal, then uses DeviceBackend (SM CollCopy) to
// read the data into a second buffer and verifies it on the host.
// No SprayExecutor, no GDR flush — isolates RDMA write + SM kernel read.
//
// Run:
//   server: CUDA_VISIBLE_DEVICES=6,7 ./test_rdma_l2_flush --role=server --gpu=0
//   client: CUDA_VISIBLE_DEVICES=6,7 ./test_rdma_l2_flush --role=client --gpu=1
//
// IPC (same-host) should pass.  RDMA may fail on pre-Hopper GPUs due to
// stale L2 cache lines after the NIC writes directly to GPU DRAM.

#include "backend/device_backend.h"
#include "executor.h"
#include "gpu_rt.h"
#include "transport.h"
#if !defined(__HIP_PLATFORM_AMD__)
#include <gdrapi.h>
#endif
#include <chrono>
#include <cmath>
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
  comm->register_buffer(1, buf1, bytes);
  comm->register_buffer(2, buf2, bytes);

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
    size_t n = comm->try_complete_put(&res, 1);
    if (n > 0 && res.rid == rid) {
      if (res.failed) throw std::runtime_error("RDMA put failed");
      return;
    }
    std::this_thread::yield();
  }
}

// Spin-wait for a specific signal receive completion.
static void wait_signal_recv(Communicator* comm, unsigned signal_rid) {
  SignalCompletion ev;
  while (true) {
    size_t n = comm->try_complete_sig_wait(&ev, 1);
    if (n > 0 && ev.rid == signal_rid) {
      if (ev.failed) throw std::runtime_error("signal recv failed");
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
                 "[--transport ipc|rdma] [--exchanger-ip IP] [--exchanger-port PORT]\n");
    return 1;
  }

  int rank = (role == "server") ? 0 : 1;
  int gpu = get_int_arg(argc, argv, "--gpu", rank);
  int port = get_int_arg(argc, argv, "--exchanger-port", 16998);
  std::string xip = get_arg(argc, argv, "--exchanger-ip", "127.0.0.1");

  std::string test_case = get_arg(argc, argv, "--case", "gpuMemcpy");
  if (test_case != "gpuMemcpy" && test_case != "CollCopy" &&
      test_case != "Reduce") {
    std::fprintf(stderr,
                 "Usage: --role=server|client --case=gpuMemcpy|CollCopy|Reduce "
                 "[--gpu GPU] [--transport rdma] [--exchanger-port PORT]\n");
    return 1;
  }

  std::printf("[l2flush] %s rank=%d gpu=%d case=%s\n", role.c_str(), rank, gpu,
              test_case.c_str());

  GPU_RT_CHECK(gpuSetDevice(gpu));

  // --- Communicator setup ---
  SprayExecutorConfig cfg;
  cfg.gpu_id = gpu;
  cfg.rank = rank;
  cfg.world_size = 2;
  cfg.exchanger_ip = (rank == 0) ? "0.0.0.0" : xip;
  cfg.exchanger_port = port;
  cfg.local_id = gpu;
  auto comm_cfg = std::make_shared<CommunicatorConfig>();
  comm_cfg->exchanger_ip = cfg.exchanger_ip;
  comm_cfg->exchanger_port = cfg.exchanger_port;
  comm_cfg->local_id = cfg.local_id;
  auto comm = std::make_shared<Communicator>(cfg.gpu_id, cfg.rank,
                                             cfg.world_size, comm_cfg);

  // Connect RDMA (and IPC) paths — lower rank connects first to avoid
  // deadlock in the handshake (matches factory convention).
  for (int p = 0; p < 2; ++p) {
    if (p == rank) continue;
    bool same = comm->same_host(p);
    if (same) {
      if (rank < p) {
        comm->connect(p, PeerTransportKind::Ipc);
        comm->accept(p, PeerTransportKind::Ipc);
      } else {
        comm->accept(p, PeerTransportKind::Ipc);
        comm->connect(p, PeerTransportKind::Ipc);
      }
    }
    if (rank < p) {
      comm->connect(p, PeerTransportKind::Rdma);
      comm->accept(p, PeerTransportKind::Rdma);
    } else {
      comm->accept(p, PeerTransportKind::Rdma);
      comm->connect(p, PeerTransportKind::Rdma);
    }
  }

  // --- GPU buffers ---
  constexpr size_t kBufBytes = 65536;  // one tile, ≤ BAR1 page
  constexpr size_t kFloats = kBufBytes / sizeof(float);
  void *d_send = nullptr, *d_recv = nullptr, *d_verify = nullptr, *d_local = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d_send, kBufBytes));
  GPU_RT_CHECK(gpuMalloc(&d_recv, kBufBytes));
  GPU_RT_CHECK(gpuMalloc(&d_verify, kBufBytes));
  GPU_RT_CHECK(gpuMalloc(&d_local, kBufBytes));

  // Register: id=1 send/recv, id=2 verify/copy, id=3 local reduce dst
  if (rank == 0) {
    setup_buffers(comm.get(), d_send, d_verify, kBufBytes, rank, 2);
  } else {
    setup_buffers(comm.get(), d_recv, d_verify, kBufBytes, rank, 2);
  }
  // Buffer 3: local-only on rank 1, register on both for resolution
  comm->register_buffer(3, d_local, kBufBytes);
  comm->resolve_remote_buffer(rank == 0 ? 1 : 0, 3, 30000);

  // --- Rank 0: wait for rank 1 ready, then RDMA put + signal ---
  if (rank == 0) {
    // Wait for rank 1 to signal that its buffer is registered and zeroed
    std::printf("[l2flush] rank0: waiting for rank1 ready...\n");
    {
      unsigned rid = comm->wait_signal_async(1, 99, PeerTransportKind::Rdma);
      wait_signal_recv(comm.get(), rid);
    }
    std::printf("[l2flush] rank0: rank1 ready\n");

    // Dump remote MR info for debugging
    auto mr = comm->get_mr(1, 1);
    std::printf("[l2flush] rank0: remote MR buf=1 addr=0x%lx key=%u len=%lu\n",
                mr.address, mr.key, mr.length);

    std::vector<float> host_send(kFloats);
    for (size_t i = 0; i < kFloats; ++i)
      host_send[i] = static_cast<float>(i + 1) * 1.5f + 0.1f;
    {
      gpuStream_t ss;
      GPU_RT_CHECK(gpuStreamCreate(&ss));
      GPU_RT_CHECK(gpuMemcpyAsync(d_send, host_send.data(), kBufBytes,
                                   gpuMemcpyHostToDevice, ss));
      GPU_RT_CHECK(gpuStreamSynchronize(ss));
      GPU_RT_CHECK(gpuStreamDestroy(ss));
    }

    unsigned put_rid = comm->send_put_async(1, 1, 0, 1, 0, kBufBytes,
                                            PeerTransportKind::Rdma);
    if (put_rid == 0) {
      std::fprintf(stderr, "[l2flush] rank0: send_put_async returned 0 (path not ready)\n");
      return 1;
    }
    wait_completion(comm.get(), put_rid);
    std::printf("[l2flush] rank0: RDMA put done (rid=%u)\n", put_rid);

    unsigned sig_rid = comm->send_signal_async(1, 42, PeerTransportKind::Rdma);
    if (sig_rid == 0) {
      std::fprintf(stderr, "[l2flush] rank0: send_signal_async returned 0\n");
      return 1;
    }
    CompletionResult res;
    for (int tries = 0; tries < 1000; ++tries) {
      if (comm->try_complete_sig_send(&res, 1) > 0 && res.rid == sig_rid)
        break;
      comm->try_complete_put(&res, 1);
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    std::printf("[l2flush] rank0: signal sent\n");
  }

  // --- Rank 1: wait signal, then run selected test case ---
  if (rank == 1) {
    // Fill d_recv with marker and pin via GDRCopy for BAR1 read-tail.
    // After signal, read tail through BAR1 to force PCIe posted writes
    // (RDMA data from NIC) to commit to DRAM.
#if !defined(__HIP_PLATFORM_AMD__)
    gdr_t gdr = gdr_open();
    gdr_mh_t mh{};
    void* bar1_ptr = nullptr;
#endif
    {
      std::vector<float> marker(kFloats);
      for (size_t i = 0; i < kFloats; ++i) marker[i] = -999.0f;
      gpuStream_t ms;
      GPU_RT_CHECK(gpuStreamCreate(&ms));
      GPU_RT_CHECK(gpuMemcpyAsync(d_recv, marker.data(), kBufBytes,
                                   gpuMemcpyHostToDevice, ms));
      GPU_RT_CHECK(gpuStreamSynchronize(ms));
      GPU_RT_CHECK(gpuStreamDestroy(ms));
#if !defined(__HIP_PLATFORM_AMD__)
      if (gdr) {
        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(d_recv);
        if (gdr_pin_buffer(gdr, dptr, kBufBytes, 0, 0, &mh) == 0)
          gdr_map(gdr, mh, &bar1_ptr, kBufBytes);
      }
#endif
    }
    // Tell rank 0 we're ready (buffer registered and filled with marker)
    {
      unsigned rid = comm->send_signal_async(0, 99, PeerTransportKind::Rdma);
      CompletionResult res;
      for (int tries = 0; tries < 1000; ++tries) {
        if (comm->try_complete_sig_send(&res, 1) > 0 && res.rid == rid)
          break;
        comm->try_complete_put(&res, 1);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    std::printf("[l2flush] rank1: ready signal sent\n");

    std::printf("[l2flush] rank1: waiting for signal...\n");
    unsigned sid = comm->wait_signal_async(0, 42, PeerTransportKind::Rdma);
    wait_signal_recv(comm.get(), sid);
    std::printf("[l2flush] rank1: signal received\n");

#if 0
    // GDR read-tail: read tail 32 bytes via BAR1 mapping to force
    // PCIe posted writes (RDMA data) to commit to DRAM.
    if (bar1_ptr) {
      volatile char sink[32];
      std::memcpy(const_cast<char*>(sink),
                  static_cast<const char*>(bar1_ptr) + kBufBytes - 32, 32);
    }
#endif

    // Poll first float until RDMA data arrives
    {
      float v = 0.0f;
      int waited_ms = 0;
      float expected = 1.6f;  // first element of pattern
      while (waited_ms < 2000) {
        GPU_RT_CHECK(
            gpuMemcpy(&v, d_recv, sizeof(float), gpuMemcpyDeviceToHost));
        if (v == expected) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ++waited_ms;
      }
      std::printf("[l2flush] rank1: after %d ms, first=%.1f\n", waited_ms, v);
    }

    if (test_case == "gpuMemcpy") {
      GPU_RT_CHECK(
          gpuMemcpy(d_verify, d_recv, kBufBytes, gpuMemcpyDeviceToDevice));
      std::vector<float> hv(kFloats);
      GPU_RT_CHECK(
          gpuMemcpy(hv.data(), d_verify, kBufBytes, gpuMemcpyDeviceToHost));
      for (size_t i = 0; i < kFloats; ++i) {
        float expected = static_cast<float>(i + 1) * 1.5f + 0.1f;
        if (std::fabs(hv[i] - expected) > 1e-5f) {
          std::fprintf(stderr, "[l2flush] gpuMemcpy MISMATCH [%zu]: got %.1f want %.1f\n",
                       i, hv[i], expected);
          return 1;
        }
      }
      std::printf("[l2flush] gpuMemcpy PASSED\n");
    } else if (test_case == "CollCopy") {
      DeviceBackendConfig dev_cfg;
      dev_cfg.task_capacity = 4096;
      dev_cfg.max_fifos = 2;
      dev_cfg.threads_per_block = 64;
      dev_cfg.blocks_per_worker = 1;
      dev_cfg.fifo_capacity = 256;
      dev_cfg.smem_size = 4096;
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
      if (dev_be.do_enqueue(&cmd, 1, &be_idx) != 1) {
        std::fprintf(stderr, "[l2flush] CollCopy enqueue failed\n");
        return 1;
      }
      uint32_t done = 0;
      while (dev_be.do_drain(&done, 1) == 0) std::this_thread::yield();
      std::vector<float> hv(kFloats);
      GPU_RT_CHECK(
          gpuMemcpy(hv.data(), d_verify, kBufBytes, gpuMemcpyDeviceToHost));
      for (size_t i = 0; i < kFloats; ++i) {
        float expected = static_cast<float>(i + 1) * 1.5f + 0.1f;
        if (std::fabs(hv[i] - expected) > 1e-5f) {
          std::fprintf(stderr, "[l2flush] CollCopy MISMATCH [%zu]: got %.1f want %.1f\n",
                       i, hv[i], expected);
          return 1;
        }
      }
      std::printf("[l2flush] CollCopy PASSED\n");
    } else {
      // Reduce — use d_verify (buf=2) as reduce dst, same as CollCopy
      DeviceBackendConfig dev_cfg;
      dev_cfg.task_capacity = 4096;
      dev_cfg.max_fifos = 2;
      dev_cfg.threads_per_block = 64;
      dev_cfg.blocks_per_worker = 1;
      dev_cfg.fifo_capacity = 256;
      dev_cfg.smem_size = 4096;
      DeviceBackend dev_be(dev_cfg);
      dev_be.set_comm(comm.get());
      std::vector<float> local_init(kFloats);
      for (size_t i = 0; i < kFloats; ++i) local_init[i] = 10.0f + (float)i;
      {
        gpuStream_t rs;
        GPU_RT_CHECK(gpuStreamCreate(&rs));
        GPU_RT_CHECK(gpuMemcpyAsync(d_verify, local_init.data(), kBufBytes,
                                     gpuMemcpyHostToDevice, rs));
        GPU_RT_CHECK(gpuStreamSynchronize(rs));
        GPU_RT_CHECK(gpuStreamDestroy(rs));
      }

      Cmd rcmd{};
      rcmd.kind = ExecOpKind::Reduce;
      rcmd.src_buf = 1;
      rcmd.dst_buf = 2;
      rcmd.bytes = static_cast<uint32_t>(kBufBytes);
      rcmd.src_peer = ~0u;
      rcmd.dst_peer = ~0u;
      rcmd.redop = ReductionKind::Sum;
      uint32_t be_idx = 0;
      if (dev_be.do_enqueue(&rcmd, 1, &be_idx) != 1) {
        std::fprintf(stderr, "[l2flush] Reduce enqueue failed\n");
        return 1;
      }
      uint32_t done = 0;
      while (dev_be.do_drain(&done, 1) == 0) std::this_thread::yield();
      std::vector<float> hv(kFloats);
      GPU_RT_CHECK(
          gpuMemcpy(hv.data(), d_verify, kBufBytes, gpuMemcpyDeviceToHost));
      for (size_t i = 0; i < kFloats; ++i) {
        float expected = 11.6f + 2.5f * static_cast<float>(i);
        if (std::fabs(hv[i] - expected) > 1e-5f) {
          std::fprintf(stderr, "[l2flush] Reduce MISMATCH [%zu]: got %.1f want %.1f\n",
                       i, hv[i], expected);
          return 1;
        }
      }
      std::printf("[l2flush] Reduce PASSED\n");
    }
    return 0;
  }

  // Wait for rank 1 to finish before exiting
  if (rank == 0) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  GPU_RT_CHECK(gpuFree(d_send));
  GPU_RT_CHECK(gpuFree(d_recv));
  GPU_RT_CHECK(gpuFree(d_verify));
  GPU_RT_CHECK(gpuFree(d_local));
  std::printf("[l2flush] done\n");
  return 0;
}

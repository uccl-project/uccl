#include "../include/config.h"
#include "../include/transport.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace UKernel::Transport;

static inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// IPC can sustain a larger in-flight window in this benchmark, while the UCCL
// path becomes unstable when bidirectional send+recv pressure reaches the
// backend queue limit.
static constexpr int kIpcThroughputWindow = 8;
static constexpr int kUcclThroughputWindow = 4;

static PreferredTransport parse_transport(char const* value) {
  if (strcmp(value, "auto") == 0) return PreferredTransport::Auto;
  if (strcmp(value, "ipc") == 0) return PreferredTransport::Ipc;
  if (strcmp(value, "uccl") == 0) return PreferredTransport::Uccl;
  fprintf(stderr, "Error: unsupported transport '%s' (expected auto|ipc|uccl)\n",
          value);
  std::exit(1);
}

static int throughput_window_for(PeerTransportKind kind) {
  return kind == PeerTransportKind::Uccl ? kUcclThroughputWindow
                                         : kIpcThroughputWindow;
}

static void print_latency(std::vector<uint64_t> const& v) {
  if (v.empty()) return;

  std::vector<uint64_t> sorted = v;
  std::sort(sorted.begin(), sorted.end());

  auto p = [&](double q) -> uint64_t {
    size_t i = static_cast<size_t>(q * sorted.size());
    if (i >= sorted.size()) i = sorted.size() - 1;
    return sorted[i];
  };

  printf(
      "  Latency (us): min %.2f | p50 %.2f | p90 %.2f | p99 %.2f | max %.2f\n",
      sorted.front() / 1000.0, p(0.5) / 1000.0, p(0.9) / 1000.0,
      p(0.99) / 1000.0, sorted.back() / 1000.0);
}

static bool setup_bidirectional_peer(Communicator& comm, int rank,
                                     int peer_rank) {
  if (rank < peer_rank) {
    if (!comm.connect_to(peer_rank)) return false;
    if (!comm.accept_from(peer_rank)) return false;
    return true;
  }

  if (!comm.accept_from(peer_rank)) return false;
  if (!comm.connect_to(peer_rank)) return false;
  return true;
}

static bool wait_all(Communicator& comm, std::vector<unsigned>& reqs) {
  if (reqs.empty()) return true;
  bool ok = comm.wait_finish(reqs);
  reqs.clear();
  return ok;
}

static std::vector<uint8_t> make_pattern(size_t msg_size, uint8_t seed) {
  std::vector<uint8_t> buf(msg_size);
  for (size_t i = 0; i < msg_size; ++i) {
    buf[i] = static_cast<uint8_t>((i + seed) % 256);
  }
  return buf;
}

static bool allocate_device_slots(int count, size_t msg_size,
                                  std::vector<void*>& slots) {
  slots.assign(count, nullptr);
  for (int i = 0; i < count; ++i) {
    if (cudaMalloc(&slots[i], msg_size) != cudaSuccess || slots[i] == nullptr) {
      for (void* ptr : slots) {
        if (ptr != nullptr) cudaFree(ptr);
      }
      slots.clear();
      return false;
    }
  }
  return true;
}

static void free_device_slots(std::vector<void*>& slots) {
  for (void* ptr : slots) {
    if (ptr != nullptr) cudaFree(ptr);
  }
  slots.clear();
}

static std::vector<MR> register_slot_mrs(Communicator& comm,
                                         std::vector<void*> const& slots,
                                         size_t msg_size) {
  std::vector<MR> mrs;
  mrs.reserve(slots.size());
  for (void* ptr : slots) {
    mrs.push_back(comm.reg_mr(ptr, msg_size));
  }
  return mrs;
}

static void deregister_slot_mrs(Communicator& comm,
                                std::vector<void*> const& slots) {
  for (void* ptr : slots) {
    if (ptr != nullptr) comm.dereg_mr(ptr);
  }
}

static bool validate_recv_slots(char const* role, int rank,
                                std::vector<void*> const& slots,
                                std::vector<char> const& touched,
                                std::vector<uint8_t> const& expected) {
  std::vector<uint8_t> host(expected.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    if (!touched[i]) continue;
    if (cudaMemcpy(host.data(), slots[i], expected.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "[%s %d] Failed to copy recv slot %zu to host\n", role,
              rank, i);
      return false;
    }
    if (std::memcmp(host.data(), expected.data(), expected.size()) != 0) {
      fprintf(stderr, "[%s %d] Data validation failed for recv slot %zu\n",
              role, rank, i);
      return false;
    }
  }
  return true;
}

static bool exchange_remote_recv_mrs(Communicator& comm, int peer_rank,
                                     std::vector<MR> const& local_recv_mrs,
                                     std::vector<MR>& remote_recv_mrs,
                                     bool notify_first) {
  remote_recv_mrs.resize(local_recv_mrs.size());
  if (notify_first) {
    for (auto const& mr : local_recv_mrs) {
      MR copy = mr;
      comm.notify_mr(peer_rank, copy);
    }
    for (size_t i = 0; i < remote_recv_mrs.size(); ++i) {
      if (!comm.wait_mr_notify(peer_rank, remote_recv_mrs[i])) return false;
    }
    return true;
  }

  for (size_t i = 0; i < remote_recv_mrs.size(); ++i) {
    if (!comm.wait_mr_notify(peer_rank, remote_recv_mrs[i])) return false;
  }
  for (auto const& mr : local_recv_mrs) {
    MR copy = mr;
    comm.notify_mr(peer_rank, copy);
  }
  return true;
}

static uint32_t remote_recv_slot_id(PeerTransportKind kind,
                                    std::vector<MR> const& remote_recv_mrs,
                                    int slot) {
  if (kind != PeerTransportKind::Uccl) return 0;
  return remote_recv_mrs.at(static_cast<size_t>(slot)).id;
}

void run_sender(int gpu_id, int rank, int peer_rank, int world_size,
                size_t msg_size, int num_iterations, int num_warmup,
                std::string const& local_ip, uint16_t listen_port,
                PreferredTransport preferred_transport) {
  printf("[Sender %d] Initializing...\n", rank);

  // Create configuration
  auto config = std::make_shared<CommunicatorConfig>();
  config->exchanger_ip = local_ip;
  config->exchanger_port = listen_port;
  config->local_id = rank;
  config->preferred_transport = preferred_transport;

  // Create communicator
  Communicator comm(gpu_id, rank, world_size, config);

  // Establish both UCCL directions: connect flow for sends, accept flow for
  // receives.
  printf("[Sender %d] Establishing bidirectional peer flows with %d...\n", rank,
         peer_rank);
  if (!setup_bidirectional_peer(comm, rank, peer_rank)) {
    fprintf(
        stderr,
        "[Sender %d] Failed to establish bidirectional flows with peer %d\n",
        rank, peer_rank);
    return;
  }

  printf("[Sender %d] Bidirectional flows ready with peer %d\n", rank,
         peer_rank);
  PeerTransportKind transport_kind = comm.peer_transport_kind(peer_rank);
  int throughput_window = throughput_window_for(transport_kind);
  printf("[Sender %d] Active transport to peer %d: %s (throughput window=%d)\n",
         rank, peer_rank, peer_transport_kind_name(transport_kind),
         throughput_window);

  cudaSetDevice(gpu_id);

  void* send_buf = nullptr;
  std::vector<void*> recv_slots;
  if (cudaMalloc(&send_buf, msg_size) != cudaSuccess || send_buf == nullptr) {
    fprintf(stderr, "[Sender %d] Failed to allocate GPU memory\n", rank);
    return;
  }
  if (!allocate_device_slots(throughput_window, msg_size, recv_slots)) {
    fprintf(stderr, "[Sender %d] Failed to allocate GPU memory\n", rank);
    cudaFree(send_buf);
    return;
  }
  auto cleanup = [&]() {
    if (send_buf != nullptr) {
      comm.dereg_mr(send_buf);
      cudaFree(send_buf);
    }
    deregister_slot_mrs(comm, recv_slots);
    free_device_slots(recv_slots);
  };

  std::vector<uint8_t> host_buf = make_pattern(msg_size, 0);
  std::vector<uint8_t> expected_recv = make_pattern(msg_size, 97);
  cudaMemcpy(send_buf, host_buf.data(), msg_size, cudaMemcpyHostToDevice);
  for (void* recv_slot : recv_slots) {
    cudaMemset(recv_slot, 0, msg_size);
  }

  MR local_send_mr = comm.reg_mr(send_buf, msg_size);
  std::vector<MR> local_recv_mrs = register_slot_mrs(comm, recv_slots, msg_size);

  printf("[Sender %d] Memory registered: send_mr=%d, recv_slots=%d\n", rank,
         local_send_mr.id, throughput_window);

  std::vector<MR> remote_recv_mrs;
  if (transport_kind == PeerTransportKind::Uccl) {
    if (!exchange_remote_recv_mrs(comm, peer_rank, local_recv_mrs,
                                  remote_recv_mrs, true)) {
      fprintf(stderr, "[Sender %d] Failed to exchange remote receive MRs\n",
              rank);
      cleanup();
      return;
    }
  }

  printf("[Sender %d] Remote memory info received\n", rank);

  // Warmup phase
  printf("[Sender %d] Warmup (%d iterations)...\n", rank, num_warmup);
  std::vector<char> warmup_touched(static_cast<size_t>(throughput_window), 0);
  for (int i = 0; i < num_warmup; ++i) {
    int slot = i % throughput_window;
    warmup_touched[static_cast<size_t>(slot)] = 1;
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id,
                                   remote_recv_slot_id(transport_kind,
                                                       remote_recv_mrs, slot),
                                   true);
    comm.wait_finish(send_req);

    unsigned recv_req =
        comm.irecv(peer_rank, recv_slots[static_cast<size_t>(slot)], 0,
                   msg_size, true);
    comm.wait_finish(recv_req);
  }
  if (!validate_recv_slots("Sender", rank, recv_slots, warmup_touched,
                           expected_recv)) {
    cleanup();
    return;
  }
  printf("[Sender %d] Warmup complete\n", rank);

  // Latency test (ping-pong)
  printf("[Sender %d] Latency test (%d iterations)...\n", rank, num_iterations);
  std::vector<uint64_t> latencies;
  latencies.reserve(num_iterations);
  std::vector<char> latency_touched(static_cast<size_t>(throughput_window), 0);

  for (int i = 0; i < num_iterations; ++i) {
    int slot = i % throughput_window;
    latency_touched[static_cast<size_t>(slot)] = 1;
    uint64_t t0 = now_ns();

    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id,
                                   remote_recv_slot_id(transport_kind,
                                                       remote_recv_mrs, slot),
                                   true);
    comm.wait_finish(send_req);

    unsigned recv_req =
        comm.irecv(peer_rank, recv_slots[static_cast<size_t>(slot)], 0,
                   msg_size, true);
    comm.wait_finish(recv_req);

    uint64_t t1 = now_ns();
    latencies.push_back(t1 - t0);
  }
  if (!validate_recv_slots("Sender", rank, recv_slots, latency_touched,
                           expected_recv)) {
    cleanup();
    return;
  }

  printf("[Sender %d] Latency results:\n", rank);
  print_latency(latencies);

  // Throughput test (one-way send)
  printf("[Sender %d] Throughput test (%d iterations)...\n", rank,
         num_iterations);

  uint64_t t0 = now_ns();
  std::vector<unsigned> send_reqs;
  send_reqs.reserve(throughput_window);

  for (int i = 0; i < num_iterations; ++i) {
    int slot = i % throughput_window;
    unsigned req = comm.isend(peer_rank, send_buf, 0, msg_size,
                              local_send_mr.id,
                              remote_recv_slot_id(transport_kind,
                                                  remote_recv_mrs, slot),
                              true);
    if (req == 0) {
      fprintf(stderr,
              "[Sender %d] isend failed during throughput test at iter %d\n",
              rank, i);
      cleanup();
      return;
    }
    send_reqs.push_back(req);
    if (static_cast<int>(send_reqs.size()) == throughput_window &&
        !wait_all(comm, send_reqs)) {
      fprintf(stderr, "[Sender %d] wait_finish failed during throughput test\n",
              rank);
      cleanup();
      return;
    }
  }

  // Wait for all sends to complete
  if (!wait_all(comm, send_reqs)) {
    fprintf(stderr, "[Sender %d] wait_finish failed during throughput test\n",
            rank);
    cleanup();
    return;
  }

  uint64_t t1 = now_ns();
  double elapsed_sec = (t1 - t0) * 1e-9;
  double total_gb =
      (double)(msg_size * num_iterations) / (1024.0 * 1024.0 * 1024.0);
  double throughput_gbps =
      (double)(msg_size * num_iterations) * 8.0 / elapsed_sec / 1e9;

  printf("[Sender %d] Throughput results:\n", rank);
  printf("  Total data: %.2f GB\n", total_gb);
  printf("  Time: %.3f sec\n", elapsed_sec);
  printf("  Throughput: %.2f GB/s (%.2f Gbps)\n", total_gb / elapsed_sec,
         throughput_gbps);
  printf("  Messages/sec: %.2f M\n", num_iterations / elapsed_sec / 1e6);

  // Bidirectional throughput test
  printf("[Sender %d] Bidirectional throughput test (%d iterations)...\n", rank,
         num_iterations);

  t0 = now_ns();
  send_reqs.clear();
  std::vector<unsigned> recv_reqs;
  send_reqs.reserve(throughput_window);
  recv_reqs.reserve(throughput_window);
  std::vector<char> bidi_touched(static_cast<size_t>(throughput_window), 0);

  for (int i = 0; i < num_iterations; ++i) {
    int slot = i % throughput_window;
    bidi_touched[static_cast<size_t>(slot)] = 1;
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id,
                                   remote_recv_slot_id(transport_kind,
                                                       remote_recv_mrs, slot),
                                   true);
    unsigned recv_req =
        comm.irecv(peer_rank, recv_slots[static_cast<size_t>(slot)], 0,
                   msg_size, true);
    if (send_req == 0 || recv_req == 0) {
      fprintf(stderr,
              "[Sender %d] request submission failed during bidirectional "
              "throughput at iter %d\n",
              rank, i);
      cleanup();
      return;
    }
    send_reqs.push_back(send_req);
    recv_reqs.push_back(recv_req);
    if (static_cast<int>(send_reqs.size()) == throughput_window) {
      if (!wait_all(comm, send_reqs) || !wait_all(comm, recv_reqs)) {
        fprintf(
            stderr,
            "[Sender %d] wait_finish failed during bidirectional throughput\n",
            rank);
        cleanup();
        return;
      }
    }
  }

  if (!wait_all(comm, send_reqs) || !wait_all(comm, recv_reqs)) {
    fprintf(stderr,
            "[Sender %d] wait_finish failed during bidirectional throughput\n",
            rank);
    cleanup();
    return;
  }
  if (!validate_recv_slots("Sender", rank, recv_slots, bidi_touched,
                           expected_recv)) {
    cleanup();
    return;
  }

  t1 = now_ns();
  elapsed_sec = (t1 - t0) * 1e-9;
  total_gb =
      (double)(msg_size * num_iterations * 2) / (1024.0 * 1024.0 * 1024.0);
  throughput_gbps =
      (double)(msg_size * num_iterations * 2) * 8.0 / elapsed_sec / 1e9;

  printf("[Sender %d] Bidirectional throughput results:\n", rank);
  printf("  Total data: %.2f GB\n", total_gb);
  printf("  Time: %.3f sec\n", elapsed_sec);
  printf("  Throughput: %.2f GB/s (%.2f Gbps)\n", total_gb / elapsed_sec,
         throughput_gbps);

  cleanup();

  printf("[Sender %d] Done.\n", rank);
}

void run_receiver(int gpu_id, int rank, int peer_rank, int world_size,
                  size_t msg_size, int num_iterations, int num_warmup,
                  std::string const& local_ip, uint16_t listen_port,
                  PreferredTransport preferred_transport) {
  printf("[Receiver %d] Initializing...\n", rank);

  // Create configuration
  auto config = std::make_shared<CommunicatorConfig>();
  config->exchanger_ip = local_ip;
  config->exchanger_port = listen_port;
  config->local_id = rank;
  config->preferred_transport = preferred_transport;

  // Create communicator
  Communicator comm(gpu_id, rank, world_size, config);

  // Establish both UCCL directions: accept flow for receives, connect flow for
  // sends.
  printf("[Receiver %d] Establishing bidirectional peer flows with %d...\n",
         rank, peer_rank);
  if (!setup_bidirectional_peer(comm, rank, peer_rank)) {
    fprintf(
        stderr,
        "[Receiver %d] Failed to establish bidirectional flows with peer %d\n",
        rank, peer_rank);
    return;
  }

  printf("[Receiver %d] Bidirectional flows ready with peer %d\n", rank,
         peer_rank);
  PeerTransportKind transport_kind = comm.peer_transport_kind(peer_rank);
  int throughput_window = throughput_window_for(transport_kind);
  printf("[Receiver %d] Active transport to peer %d: %s (throughput window=%d)\n",
         rank, peer_rank, peer_transport_kind_name(transport_kind),
         throughput_window);

  cudaSetDevice(gpu_id);

  void* send_buf = nullptr;
  std::vector<void*> recv_slots;
  if (cudaMalloc(&send_buf, msg_size) != cudaSuccess || send_buf == nullptr) {
    fprintf(stderr, "[Receiver %d] Failed to allocate GPU memory\n", rank);
    return;
  }
  if (!allocate_device_slots(throughput_window, msg_size, recv_slots)) {
    fprintf(stderr, "[Receiver %d] Failed to allocate GPU memory\n", rank);
    cudaFree(send_buf);
    return;
  }
  auto cleanup = [&]() {
    if (send_buf != nullptr) {
      comm.dereg_mr(send_buf);
      cudaFree(send_buf);
    }
    deregister_slot_mrs(comm, recv_slots);
    free_device_slots(recv_slots);
  };

  std::vector<uint8_t> host_buf = make_pattern(msg_size, 97);
  std::vector<uint8_t> expected_recv = make_pattern(msg_size, 0);
  cudaMemcpy(send_buf, host_buf.data(), msg_size, cudaMemcpyHostToDevice);
  for (void* recv_slot : recv_slots) {
    cudaMemset(recv_slot, 0, msg_size);
  }

  MR local_send_mr = comm.reg_mr(send_buf, msg_size);
  std::vector<MR> local_recv_mrs = register_slot_mrs(comm, recv_slots, msg_size);

  printf("[Receiver %d] Memory registered: send_mr=%d, recv_slots=%d\n", rank,
         local_send_mr.id, throughput_window);

  std::vector<MR> remote_recv_mrs;
  if (transport_kind == PeerTransportKind::Uccl) {
    if (!exchange_remote_recv_mrs(comm, peer_rank, local_recv_mrs,
                                  remote_recv_mrs, false)) {
      fprintf(stderr, "[Receiver %d] Failed to exchange remote receive MRs\n",
              rank);
      cleanup();
      return;
    }
  }

  printf("[Receiver %d] Remote memory info received\n", rank);

  // Warmup phase
  printf("[Receiver %d] Warmup (%d iterations)...\n", rank, num_warmup);
  std::vector<char> warmup_touched(static_cast<size_t>(throughput_window), 0);
  for (int i = 0; i < num_warmup; ++i) {
    int slot = i % throughput_window;
    warmup_touched[static_cast<size_t>(slot)] = 1;
    unsigned recv_req =
        comm.irecv(peer_rank, recv_slots[static_cast<size_t>(slot)], 0,
                   msg_size, true);
    comm.wait_finish(recv_req);

    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id,
                                   remote_recv_slot_id(transport_kind,
                                                       remote_recv_mrs, slot),
                                   true);
    comm.wait_finish(send_req);
  }
  if (!validate_recv_slots("Receiver", rank, recv_slots, warmup_touched,
                           expected_recv)) {
    cleanup();
    return;
  }
  printf("[Receiver %d] Warmup complete\n", rank);

  // Latency test (ping-pong) - receiver side
  printf("[Receiver %d] Latency test (%d iterations)...\n", rank,
         num_iterations);
  std::vector<char> latency_touched(static_cast<size_t>(throughput_window), 0);
  for (int i = 0; i < num_iterations; ++i) {
    int slot = i % throughput_window;
    latency_touched[static_cast<size_t>(slot)] = 1;
    unsigned recv_req =
        comm.irecv(peer_rank, recv_slots[static_cast<size_t>(slot)], 0,
                   msg_size, true);
    comm.wait_finish(recv_req);

    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id,
                                   remote_recv_slot_id(transport_kind,
                                                       remote_recv_mrs, slot),
                                   true);
    comm.wait_finish(send_req);
  }
  if (!validate_recv_slots("Receiver", rank, recv_slots, latency_touched,
                           expected_recv)) {
    cleanup();
    return;
  }
  printf("[Receiver %d] Latency test complete\n", rank);

  // Throughput test - just receive
  printf("[Receiver %d] Throughput test (%d iterations)...\n", rank,
         num_iterations);
  std::vector<unsigned> recv_reqs;
  recv_reqs.reserve(throughput_window);
  std::vector<char> throughput_touched(static_cast<size_t>(throughput_window), 0);

  for (int i = 0; i < num_iterations; ++i) {
    int slot = i % throughput_window;
    throughput_touched[static_cast<size_t>(slot)] = 1;
    unsigned req =
        comm.irecv(peer_rank, recv_slots[static_cast<size_t>(slot)], 0,
                   msg_size, true);
    if (req == 0) {
      fprintf(stderr,
              "[Receiver %d] irecv failed during throughput test at iter %d\n",
              rank, i);
      cleanup();
      return;
    }
    recv_reqs.push_back(req);
    if (static_cast<int>(recv_reqs.size()) == throughput_window &&
        !wait_all(comm, recv_reqs)) {
      fprintf(stderr,
              "[Receiver %d] wait_finish failed during throughput test\n",
              rank);
      cleanup();
      return;
    }
  }

  // Wait for all receives
  if (!wait_all(comm, recv_reqs)) {
    fprintf(stderr, "[Receiver %d] wait_finish failed during throughput test\n",
            rank);
    cleanup();
    return;
  }
  if (!validate_recv_slots("Receiver", rank, recv_slots, throughput_touched,
                           expected_recv)) {
    cleanup();
    return;
  }
  printf("[Receiver %d] Throughput test complete\n", rank);

  // Bidirectional throughput test
  printf("[Receiver %d] Bidirectional throughput test (%d iterations)...\n",
         rank, num_iterations);
  recv_reqs.clear();
  std::vector<unsigned> send_reqs;
  recv_reqs.reserve(throughput_window);
  send_reqs.reserve(throughput_window);
  std::vector<char> bidi_touched(static_cast<size_t>(throughput_window), 0);

  for (int i = 0; i < num_iterations; ++i) {
    int slot = i % throughput_window;
    bidi_touched[static_cast<size_t>(slot)] = 1;
    unsigned recv_req =
        comm.irecv(peer_rank, recv_slots[static_cast<size_t>(slot)], 0,
                   msg_size, true);
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id,
                                   remote_recv_slot_id(transport_kind,
                                                       remote_recv_mrs, slot),
                                   true);
    if (send_req == 0 || recv_req == 0) {
      fprintf(stderr,
              "[Receiver %d] request submission failed during bidirectional "
              "throughput at iter %d\n",
              rank, i);
      cleanup();
      return;
    }
    recv_reqs.push_back(recv_req);
    send_reqs.push_back(send_req);
    if (static_cast<int>(recv_reqs.size()) == throughput_window) {
      if (!wait_all(comm, recv_reqs) || !wait_all(comm, send_reqs)) {
        fprintf(stderr,
                "[Receiver %d] wait_finish failed during bidirectional "
                "throughput\n",
                rank);
        cleanup();
        return;
      }
    }
  }

  if (!wait_all(comm, recv_reqs) || !wait_all(comm, send_reqs)) {
    fprintf(
        stderr,
        "[Receiver %d] wait_finish failed during bidirectional throughput\n",
        rank);
    cleanup();
    return;
  }
  if (!validate_recv_slots("Receiver", rank, recv_slots, bidi_touched,
                           expected_recv)) {
    cleanup();
    return;
  }
  printf("[Receiver %d] Bidirectional throughput test complete\n", rank);

  cleanup();

  printf("[Receiver %d] Done.\n", rank);
}

void print_usage(char const* prog) {
  printf("Usage: %s [options]\n\n", prog);
  printf("Options:\n");
  printf("  --rank <n>          Rank of this process (0 or 1)\n");
  printf("  --peer-rank <n>     Rank of peer process\n");
  printf("  --gpu-id <n>        GPU ID to use\n");
  printf("  --msg-size <bytes>  Message size (default: 1024)\n");
  printf("  --iterations <n>    Number of iterations (default: 1000)\n");
  printf("  --warmup <n>        Number of warmup iterations (default: 100)\n");
  printf("  --ip <addr>         Local IP address (default: 127.0.0.1)\n");
  printf("  --port <n>          Listen port (default: 6979)\n");
  printf("  --transport <kind>  Transport override: auto|ipc|uccl (default: auto)\n");
  printf("  --help              Show this help\n");
}

int main(int argc, char** argv) {
  int rank = -1;
  int peer_rank = -1;
  int gpu_id = 0;
  size_t msg_size = 1024;
  int num_iterations = 1000;
  int num_warmup = 100;
  std::string local_ip = "127.0.0.1";
  uint16_t listen_port = 6979;
  PreferredTransport preferred_transport = PreferredTransport::Auto;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
      rank = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--peer-rank") == 0 && i + 1 < argc) {
      peer_rank = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--gpu-id") == 0 && i + 1 < argc) {
      gpu_id = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--msg-size") == 0 && i + 1 < argc) {
      msg_size = atol(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      num_iterations = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      num_warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--ip") == 0 && i + 1 < argc) {
      local_ip = argv[++i];
    } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      listen_port = static_cast<uint16_t>(atoi(argv[++i]));
    } else if (strcmp(argv[i], "--transport") == 0 && i + 1 < argc) {
      preferred_transport = parse_transport(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  // Validate arguments
  if (rank < 0 || peer_rank < 0) {
    fprintf(stderr, "Error: --rank and --peer-rank must be specified\n");
    print_usage(argv[0]);
    return 1;
  }

  if (rank == peer_rank) {
    fprintf(stderr, "Error: rank and peer-rank must be different\n");
    return 1;
  }

  int world_size = 2;

  printf("============================================================\n");
  printf("Transport Benchmark\n");
  printf("============================================================\n");
  printf("Rank: %d, Peer: %d, World: %d\n", rank, peer_rank, world_size);
  printf("GPU: %d, Message size: %zu bytes\n", gpu_id, msg_size);
  printf("Iterations: %d, Warmup: %d\n", num_iterations, num_warmup);
  printf("IP: %s, Port: %d\n", local_ip.c_str(), listen_port);
  printf("Transport override: %s\n",
         preferred_transport == PreferredTransport::Auto
             ? "auto"
             : (preferred_transport == PreferredTransport::Ipc ? "ipc"
                                                               : "uccl"));
  printf("============================================================\n\n");

  // Run as sender or receiver
  if (rank < peer_rank) {
    // Lower rank acts as sender
    run_sender(gpu_id, rank, peer_rank, world_size, msg_size, num_iterations,
               num_warmup, local_ip, listen_port, preferred_transport);
  } else {
    // Higher rank acts as receiver
    run_receiver(gpu_id, rank, peer_rank, world_size, msg_size, num_iterations,
                 num_warmup, local_ip, listen_port, preferred_transport);
  }

  return 0;
}

#include "../include/transport.h"
#include "../include/config.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>

using namespace UKernel::Transport;

static inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void print_latency(const std::vector<uint64_t>& v) {
  if (v.empty()) return;
  
  std::vector<uint64_t> sorted = v;
  std::sort(sorted.begin(), sorted.end());
  
  auto p = [&](double q) -> uint64_t {
    size_t i = static_cast<size_t>(q * sorted.size());
    if (i >= sorted.size()) i = sorted.size() - 1;
    return sorted[i];
  };

  printf("  Latency (us): min %.2f | p50 %.2f | p90 %.2f | p99 %.2f | max %.2f\n",
         sorted.front() / 1000.0, 
         p(0.5) / 1000.0, 
         p(0.9) / 1000.0, 
         p(0.99) / 1000.0, 
         sorted.back() / 1000.0);
}

void run_sender(int gpu_id, int rank, int peer_rank, int world_size,
                size_t msg_size, int num_iterations, int num_warmup,
                const std::string& local_ip, uint16_t listen_port) {
  
  printf("[Sender %d] Initializing...\n", rank);
  
  // Create configuration
  auto config = std::make_shared<CommunicatorConfig>();
  config->exchanger_ip = local_ip;
  config->exchanger_port = listen_port;
  
  // Create communicator
  Communicator comm(gpu_id, rank, world_size, config);
  
  // Connect to peer
  printf("[Sender %d] Connecting to peer %d...\n", rank, peer_rank);
  if (!comm.connect_to(peer_rank)) {
    fprintf(stderr, "[Sender %d] Failed to connect to peer %d\n", rank, peer_rank);
    return;
  }
  
  printf("[Sender %d] Connected to peer %d\n", rank, peer_rank);
  
  // Allocate buffer
  void* send_buf = nullptr;
  void* recv_buf = nullptr;
  
  cudaMalloc(&send_buf, msg_size);
  cudaMalloc(&recv_buf, msg_size);
  
  if (!send_buf || !recv_buf) {
    fprintf(stderr, "[Sender %d] Failed to allocate GPU memory\n", rank);
    return;
  }
  
  // Initialize send buffer with pattern
  std::vector<uint8_t> host_buf(msg_size);
  for (size_t i = 0; i < msg_size; ++i) {
    host_buf[i] = static_cast<uint8_t>(i % 256);
  }
  cudaMemcpy(send_buf, host_buf.data(), msg_size, cudaMemcpyHostToDevice);
  cudaMemset(recv_buf, 0, msg_size);
  
  // Register memory regions
  MR local_send_mr = comm.reg_mr(send_buf, msg_size);
  MR local_recv_mr = comm.reg_mr(recv_buf, msg_size);
  
  printf("[Sender %d] Memory registered: send_mr=%d, recv_mr=%d\n", 
         rank, local_send_mr.id, local_recv_mr.id);
  
  // Notify peer of our memory regions
  comm.notify_mr(peer_rank, local_send_mr);
  comm.notify_mr(peer_rank, local_recv_mr);
  
  // Wait for peer's memory regions
  MR remote_send_mr;
  MR remote_recv_mr;
  comm.wait_mr_notify(peer_rank, remote_send_mr);
  comm.wait_mr_notify(peer_rank, remote_recv_mr);
  
  printf("[Sender %d] Remote memory info received\n", rank);
  
  // Warmup phase
  printf("[Sender %d] Warmup (%d iterations)...\n", rank, num_warmup);
  for (int i = 0; i < num_warmup; ++i) {
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id, remote_recv_mr.id, true);
    unsigned recv_req = comm.irecv(peer_rank, recv_buf, 0, msg_size, true);
    
    comm.wait_finish(send_req);
    comm.wait_finish(recv_req);
  }
  printf("[Sender %d] Warmup complete\n", rank);
  
  // Latency test (ping-pong)
  printf("[Sender %d] Latency test (%d iterations)...\n", rank, num_iterations);
  std::vector<uint64_t> latencies;
  latencies.reserve(num_iterations);
  
  for (int i = 0; i < num_iterations; ++i) {
    uint64_t t0 = now_ns();
    
    // Send to peer
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id, remote_recv_mr.id, true);
    // Receive from peer
    unsigned recv_req = comm.irecv(peer_rank, recv_buf, 0, msg_size, true);
    
    // Wait for both
    comm.wait_finish(send_req);
    comm.wait_finish(recv_req);
    
    uint64_t t1 = now_ns();
    latencies.push_back(t1 - t0);
  }
  
  printf("[Sender %d] Latency results:\n", rank);
  print_latency(latencies);
  
  // Throughput test (one-way send)
  printf("[Sender %d] Throughput test (%d iterations)...\n", rank, num_iterations);
  
  uint64_t t0 = now_ns();
  std::vector<unsigned> send_reqs;
  send_reqs.reserve(num_iterations);
  
  for (int i = 0; i < num_iterations; ++i) {
    unsigned req = comm.isend(peer_rank, send_buf, 0, msg_size,
                              local_send_mr.id, remote_recv_mr.id, true);
    send_reqs.push_back(req);
  }
  
  // Wait for all sends to complete
  comm.wait_finish(send_reqs);
  
  uint64_t t1 = now_ns();
  double elapsed_sec = (t1 - t0) * 1e-9;
  double total_gb = (double)(msg_size * num_iterations) / (1024.0 * 1024.0 * 1024.0);
  double throughput_gbps = (double)(msg_size * num_iterations) * 8.0 / elapsed_sec / 1e9;
  
  printf("[Sender %d] Throughput results:\n", rank);
  printf("  Total data: %.2f GB\n", total_gb);
  printf("  Time: %.3f sec\n", elapsed_sec);
  printf("  Throughput: %.2f GB/s (%.2f Gbps)\n", total_gb / elapsed_sec, throughput_gbps);
  printf("  Messages/sec: %.2f M\n", num_iterations / elapsed_sec / 1e6);
  
  // Bidirectional throughput test
  printf("[Sender %d] Bidirectional throughput test (%d iterations)...\n", rank, num_iterations);
  
  t0 = now_ns();
  send_reqs.clear();
  std::vector<unsigned> recv_reqs;
  send_reqs.reserve(num_iterations);
  recv_reqs.reserve(num_iterations);
  
  for (int i = 0; i < num_iterations; ++i) {
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id, remote_recv_mr.id, true);
    unsigned recv_req = comm.irecv(peer_rank, recv_buf, 0, msg_size, true);
    send_reqs.push_back(send_req);
    recv_reqs.push_back(recv_req);
  }
  
  // Wait for all operations
  comm.wait_finish(send_reqs);
  comm.wait_finish(recv_reqs);
  
  t1 = now_ns();
  elapsed_sec = (t1 - t0) * 1e-9;
  total_gb = (double)(msg_size * num_iterations * 2) / (1024.0 * 1024.0 * 1024.0);
  throughput_gbps = (double)(msg_size * num_iterations * 2) * 8.0 / elapsed_sec / 1e9;
  
  printf("[Sender %d] Bidirectional throughput results:\n", rank);
  printf("  Total data: %.2f GB\n", total_gb);
  printf("  Time: %.3f sec\n", elapsed_sec);
  printf("  Throughput: %.2f GB/s (%.2f Gbps)\n", total_gb / elapsed_sec, throughput_gbps);
  
  // Cleanup
  comm.dereg_mr(send_buf);
  comm.dereg_mr(recv_buf);
  cudaFree(send_buf);
  cudaFree(recv_buf);
  
  printf("[Sender %d] Done.\n", rank);
}

void run_receiver(int gpu_id, int rank, int peer_rank, int world_size,
                  size_t msg_size, int num_iterations, int num_warmup,
                  const std::string& local_ip, uint16_t listen_port) {
  
  printf("[Receiver %d] Initializing...\n", rank);
  
  // Create configuration
  auto config = std::make_shared<CommunicatorConfig>();
  config->exchanger_ip = local_ip;
  config->exchanger_port = listen_port;
  
  // Create communicator
  Communicator comm(gpu_id, rank, world_size, config);
  
  // Accept connection from peer
  printf("[Receiver %d] Accepting connection from peer %d...\n", rank, peer_rank);
  if (!comm.accept_from(peer_rank)) {
    fprintf(stderr, "[Receiver %d] Failed to accept from peer %d\n", rank, peer_rank);
    return;
  }
  
  printf("[Receiver %d] Connected to peer %d\n", rank, peer_rank);
  
  // Allocate buffer
  void* send_buf = nullptr;
  void* recv_buf = nullptr;
  
  cudaMalloc(&send_buf, msg_size);
  cudaMalloc(&recv_buf, msg_size);
  
  if (!send_buf || !recv_buf) {
    fprintf(stderr, "[Receiver %d] Failed to allocate GPU memory\n", rank);
    return;
  }
  
  // Initialize buffers
  cudaMemset(send_buf, 0, msg_size);
  cudaMemset(recv_buf, 0, msg_size);
  
  // Register memory regions
  MR local_send_mr = comm.reg_mr(send_buf, msg_size);
  MR local_recv_mr = comm.reg_mr(recv_buf, msg_size);
  
  printf("[Receiver %d] Memory registered: send_mr=%d, recv_mr=%d\n", 
         rank, local_send_mr.id, local_recv_mr.id);
  
  // Wait for peer's memory regions first
  MR remote_send_mr;
  MR remote_recv_mr;
  comm.wait_mr_notify(peer_rank, remote_send_mr);
  comm.wait_mr_notify(peer_rank, remote_recv_mr);
  
  // Then notify peer of ours
  comm.notify_mr(peer_rank, local_send_mr);
  comm.notify_mr(peer_rank, local_recv_mr);
  
  printf("[Receiver %d] Remote memory info received\n", rank);
  
  // Warmup phase
  printf("[Receiver %d] Warmup (%d iterations)...\n", rank, num_warmup);
  for (int i = 0; i < num_warmup; ++i) {
    unsigned recv_req = comm.irecv(peer_rank, recv_buf, 0, msg_size, true);
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id, remote_recv_mr.id, true);
    
    comm.wait_finish(recv_req);
    comm.wait_finish(send_req);
  }
  printf("[Receiver %d] Warmup complete\n", rank);
  
  // Latency test (ping-pong) - receiver side
  printf("[Receiver %d] Latency test (%d iterations)...\n", rank, num_iterations);
  for (int i = 0; i < num_iterations; ++i) {
    // Receive from sender
    unsigned recv_req = comm.irecv(peer_rank, recv_buf, 0, msg_size, true);
    // Send back (echo)
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id, remote_recv_mr.id, true);
    
    comm.wait_finish(recv_req);
    comm.wait_finish(send_req);
  }
  printf("[Receiver %d] Latency test complete\n", rank);
  
  // Throughput test - just receive
  printf("[Receiver %d] Throughput test (%d iterations)...\n", rank, num_iterations);
  std::vector<unsigned> recv_reqs;
  recv_reqs.reserve(num_iterations);
  
  for (int i = 0; i < num_iterations; ++i) {
    unsigned req = comm.irecv(peer_rank, recv_buf, 0, msg_size, true);
    recv_reqs.push_back(req);
  }
  
  // Wait for all receives
  comm.wait_finish(recv_reqs);
  printf("[Receiver %d] Throughput test complete\n", rank);
  
  // Bidirectional throughput test
  printf("[Receiver %d] Bidirectional throughput test (%d iterations)...\n", rank, num_iterations);
  recv_reqs.clear();
  std::vector<unsigned> send_reqs;
  recv_reqs.reserve(num_iterations);
  send_reqs.reserve(num_iterations);
  
  for (int i = 0; i < num_iterations; ++i) {
    unsigned recv_req = comm.irecv(peer_rank, recv_buf, 0, msg_size, true);
    unsigned send_req = comm.isend(peer_rank, send_buf, 0, msg_size,
                                   local_send_mr.id, remote_recv_mr.id, true);
    recv_reqs.push_back(recv_req);
    send_reqs.push_back(send_req);
  }
  
  // Wait for all operations
  comm.wait_finish(recv_reqs);
  comm.wait_finish(send_reqs);
  printf("[Receiver %d] Bidirectional throughput test complete\n", rank);
  
  // Cleanup
  comm.dereg_mr(send_buf);
  comm.dereg_mr(recv_buf);
  cudaFree(send_buf);
  cudaFree(recv_buf);
  
  printf("[Receiver %d] Done.\n", rank);
}

void print_usage(const char* prog) {
  printf("Usage: %s [options]\n\n", prog);
  printf("Options:\n");
  printf("  --rank <n>          Rank of this process (0 or 1)\n");
  printf("  --peer-rank <n>     Rank of peer process\n");
  printf("  --gpu-id <n>        GPU ID to use\n");
  printf("  --msg-size <bytes>  Message size (default: 1024)\n");
  printf("  --iterations <n>    Number of iterations (default: 10000)\n");
  printf("  --warmup <n>        Number of warmup iterations (default: 1000)\n");
  printf("  --ip <addr>         Local IP address (default: 127.0.0.1)\n");
  printf("  --port <n>          Listen port (default: 6979)\n");
  printf("  --help              Show this help\n");
}

int main(int argc, char** argv) {
  int rank = -1;
  int peer_rank = -1;
  int gpu_id = 0;
  size_t msg_size = 1024;
  int num_iterations = 10000;
  int num_warmup = 1000;
  std::string local_ip = "127.0.0.1";
  uint16_t listen_port = 6979;
  
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
  printf("============================================================\n\n");
  
  // Run as sender or receiver
  if (rank < peer_rank) {
    // Lower rank acts as sender
    run_sender(gpu_id, rank, peer_rank, world_size, msg_size, 
               num_iterations, num_warmup, local_ip, listen_port);
  } else {
    // Higher rank acts as receiver
    run_receiver(gpu_id, rank, peer_rank, world_size, msg_size,
                 num_iterations, num_warmup, local_ip, listen_port);
  }
  
  return 0;
}

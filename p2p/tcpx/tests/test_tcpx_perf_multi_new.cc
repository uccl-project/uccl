/**
 * @file test_tcpx_perf_multi_new.cc
 * @brief Multi-channel TCPX GPU-to-GPU performance benchmark (Phase 1 API)
 *
 * This is a simplified version of test_tcpx_perf_multi.cc that uses the new
 * Phase 1 API (TcpxSession + TcpxTransfer) instead of inline implementation.
 *
 * Goals:
 * 1. Validate the Phase 1 API with real workloads
 * 2. Simplify the benchmark code by removing inline implementation
 * 3. Provide a clean reference for NIXL plugin development (Phase 2)
 *
 * Usage:
 *   # Server (example: 10.65.74.150)
 *   UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864 \
 *   ./tests/test_tcpx_perf_multi server 0
 *
 *   # Client (example: 10.64.113.77)
 *   UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864 \
 *   ./tests/test_tcpx_perf_multi client 10.65.74.150 0
 *
 * Env vars:
 *   UCCL_TCPX_NUM_CHANNELS: number of channels (default 2)
 *   UCCL_TCPX_PERF_SIZE: total bytes per iteration (default 4MB)
 *   UCCL_TCPX_PERF_ITERS: iteration count (default 10)
 *   UCCL_TCPX_CHUNK_BYTES: chunk size (default 512KB)
 *   UCCL_TCPX_BOOTSTRAP_PORT_BASE: bootstrap port base (default 12345)
 */

#include "tcpx_perf_runner.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  // ============================================================================
  // Env overrides (TCPX config)
  // ============================================================================

  // Enable zero-copy (devmem-tcp from 4KB)
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);

  // Enable recv sync (data integrity)
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);

  // Enable TCPX wrapper debug logs
  setenv("UCCL_TCPX_DEBUG", "1", 0);

  // Disable kernel-launch debug by default
  setenv("UCCL_TCPX_KERNEL_DEBUG", "0", 0);

  // ============================================================================
  // Parse command line
  // ============================================================================

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <server|client> [server_ip] [gpu_id]" << std::endl;
    std::cerr << "  server mode: " << argv[0] << " server [gpu_id]" << std::endl;
    std::cerr << "  client mode: " << argv[0] << " client <server_ip> [gpu_id]" << std::endl;
    return 1;
  }

  std::string mode = argv[1];
  bool is_server = (mode == "server");
  std::string server_ip;
  int gpu_id = 0;

  if (is_server) {
    gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  } else {
    if (argc < 3) {
      std::cerr << "Error: client mode requires server_ip" << std::endl;
      return 1;
    }
    server_ip = argv[2];
    gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
  }

  // ============================================================================
  // Configuration
  // ============================================================================

  tcpx::PerfConfig config;
  config.is_server = is_server;
  config.server_ip = server_ip;
  config.gpu_id = gpu_id;

  // Number of channels (default 2)
  config.num_channels = tcpx::getEnvInt("UCCL_TCPX_NUM_CHANNELS", 2);

  // Total bytes per iteration (default 4MB)
  config.test_size = tcpx::getEnvSize("UCCL_TCPX_PERF_SIZE", 4 * 1024 * 1024);

  // Iteration count (default 10)
  config.iterations = tcpx::getEnvInt("UCCL_TCPX_PERF_ITERS", 10);

  // Chunk size (default 512KB)
  config.chunk_bytes = tcpx::getEnvSize("UCCL_TCPX_CHUNK_BYTES",
                                        tcpx::getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 512 * 1024));

  // Bootstrap port
  int bootstrap_port_base = tcpx::getEnvInt("UCCL_TCPX_BOOTSTRAP_PORT_BASE", 12345);
  config.bootstrap_port = bootstrap_port_base + gpu_id;

  // Print configuration
  std::cout << "========================================" << std::endl;
  std::cout << "TCPX Performance Benchmark (Phase 1 API)" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Mode:           " << (is_server ? "SERVER" : "CLIENT") << std::endl;
  std::cout << "GPU ID:         " << gpu_id << std::endl;
  std::cout << "Channels:       " << config.num_channels << std::endl;
  std::cout << "Size:           " << (config.test_size / 1024 / 1024) << " MB" << std::endl;
  std::cout << "Iterations:     " << config.iterations << std::endl;
  std::cout << "Chunk size:     " << (config.chunk_bytes / 1024 / 1024) << " MB" << std::endl;
  std::cout << "Bootstrap port: " << config.bootstrap_port << std::endl;
  if (!is_server) {
    std::cout << "Server IP:      " << server_ip << std::endl;
  }
  std::cout << "========================================" << std::endl;

  // ============================================================================
  // Run benchmark
  // ============================================================================

  tcpx::PerfRunner runner(config);

  // Initialize
  if (runner.initialize() != 0) {
    std::cerr << "[ERROR] Runner initialization failed" << std::endl;
    return 1;
  }

  // Run benchmark
  tcpx::PerfStats stats;
  if (runner.run(&stats) != 0) {
    std::cerr << "[ERROR] Benchmark run failed" << std::endl;
    runner.cleanup();
    return 1;
  }

  // Print results
  tcpx::printPerfStats(stats, config);

  // Cleanup
  runner.cleanup();

  std::cout << "[PERF] Benchmark completed successfully" << std::endl;
  return 0;
}


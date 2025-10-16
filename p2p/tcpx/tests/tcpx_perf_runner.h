/**
 * @file tcpx_perf_runner.h
 * @brief Lightweight wrapper for TCPX performance benchmarking using TcpxSession/TcpxTransfer API
 *
 * This runner encapsulates the server/client logic using the new Phase 1 API:
 * - TcpxSession: manages channels, memory registration, connection lifecycle
 * - TcpxTransfer: manages send/recv operations, completion tracking
 *
 * The goal is to:
 * 1. Validate the Phase 1 API with real workloads
 * 2. Simplify test_tcpx_perf_multi.cc by removing inline implementation
 * 3. Provide a clean reference for NIXL plugin development (Phase 2)
 */

#ifndef TCPX_PERF_RUNNER_H_
#define TCPX_PERF_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tcpx {

// ============================================================================
// Configuration
// ============================================================================

/**
 * @brief Performance benchmark configuration
 */
struct PerfConfig {
  // Mode
  bool is_server;           // true = server (recv), false = client (send)
  std::string server_ip;    // Server IP (client only)
  int gpu_id;               // GPU device ID

  // Connection
  int num_channels;         // Number of TCPX channels
  int bootstrap_port;       // Bootstrap port for handshake

  // Transfer
  size_t test_size;         // Total bytes per iteration
  size_t chunk_bytes;       // Chunk size
  int iterations;           // Number of iterations

  // Defaults
  PerfConfig()
      : is_server(false),
        server_ip(""),
        gpu_id(0),
        num_channels(2),
        bootstrap_port(12345),
        test_size(4 * 1024 * 1024),
        chunk_bytes(512 * 1024),
        iterations(10) {}
};

/**
 * @brief Performance statistics
 */
struct PerfStats {
  double total_time_ms;     // Total time (ms)
  double bandwidth_gbps;    // Bandwidth (GB/s)
  size_t total_bytes;       // Total bytes transferred
  int iterations;           // Number of iterations
  int num_channels;         // Number of channels used

  PerfStats()
      : total_time_ms(0.0),
        bandwidth_gbps(0.0),
        total_bytes(0),
        iterations(0),
        num_channels(0) {}
};

// ============================================================================
// Runner Interface
// ============================================================================

/**
 * @brief TCPX performance benchmark runner
 *
 * This class encapsulates the server/client logic using TcpxSession and TcpxTransfer.
 * It handles:
 * - Bootstrap handshake
 * - CUDA initialization
 * - Memory allocation and registration
 * - Transfer execution (chunked send/recv)
 * - Performance measurement
 */
class PerfRunner {
 public:
  /**
   * @brief Constructor
   * @param config Benchmark configuration
   */
  explicit PerfRunner(const PerfConfig& config);

  /**
   * @brief Destructor - cleans up resources
   */
  ~PerfRunner();

  /**
   * @brief Initialize the runner
   * @return 0 on success, non-zero on error
   *
   * This function:
   * 1. Initializes TCPX plugin
   * 2. Performs bootstrap handshake
   * 3. Creates TcpxSession
   * 4. Initializes CUDA
   * 5. Allocates and registers memory
   */
  int initialize();

  /**
   * @brief Run the benchmark
   * @param stats Output statistics
   * @return 0 on success, non-zero on error
   *
   * This function:
   * 1. Runs multiple iterations
   * 2. For each iteration:
   *    - Creates TcpxTransfer
   *    - Posts chunked send/recv operations
   *    - Waits for completion
   *    - Releases transfer
   * 3. Measures total time and bandwidth
   */
  int run(PerfStats* stats);

  /**
   * @brief Clean up resources
   */
  void cleanup();

 private:
  // Forward declaration of implementation
  struct Impl;
  Impl* impl_;

  // Disable copy and assignment
  PerfRunner(const PerfRunner&) = delete;
  PerfRunner& operator=(const PerfRunner&) = delete;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Print performance statistics
 * @param stats Statistics to print
 * @param config Configuration used
 */
void printPerfStats(const PerfStats& stats, const PerfConfig& config);

/**
 * @brief Get environment variable as int
 * @param name Variable name
 * @param def Default value
 * @return Value from environment or default
 */
int getEnvInt(const char* name, int def);

/**
 * @brief Get environment variable as size_t
 * @param name Variable name
 * @param def Default value
 * @return Value from environment or default
 */
size_t getEnvSize(const char* name, size_t def);

}  // namespace tcpx

#endif  // TCPX_PERF_RUNNER_H_


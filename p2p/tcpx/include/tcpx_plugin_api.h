#pragma once
// Stable external API for TCPX plugin (OOP style).
// Only include this header from external code. Implementation is hidden by PIMPL.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tcpx_plugin {

static constexpr int TCP_PLUGIN_API_VERSION = 1;

// Error codes (stable)
enum class Status : int {
  kOk = 0,
  kUnavailable = -1,
  kInvalidArg = -2,
  kInternal = -3,
  kTimeout = -4,
};

// Optional init options
struct InitOptions {
  // Path to NCCL TCPX plugin .so; nullptr => default loader path.
  const char* plugin_path = nullptr;
  // Reserved fields for future expansion.
};

// Lightweight RAII utility to switch CUDA device (best-effort)
class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int gpu_id);
  ~ScopedCudaDevice();
  ScopedCudaDevice(ScopedCudaDevice const&) = delete;
  ScopedCudaDevice& operator=(ScopedCudaDevice const&) = delete;
 private:
  int prev_{-1};
  bool active_{false};
};

// Initialization / Capability
Status init(InitOptions const& opts = {});
Status get_device_count(int* out_count);

// Connection identity for creating transfers (extensible)
struct ConnID {
  std::string remote;  // logical peer name, e.g. "server"/"client"
  int remote_gpu = -1; // optional: remote GPU id if known
  int nic_index = -1;  // optional: NIC index for multi-NIC setups
};

// Forward declare
class Transfer;

/**
 * @brief Session manages TCPX channels, memory registration and handshake.
 *        Not thread-safe; caller must synchronize.
 */
class Session {
 public:
  /**
   * @param gpu_id         CUDA device id
   * @param num_channels   number of TCPX channels
   * @param bootstrap_info free-form bootstrap config string (JSON/URI/etc.)
   * @param nic_device_id  optional NIC device id (future use)
   */
  Session(int gpu_id, int num_channels,
          std::string const& bootstrap_info = std::string(),
          int nic_device_id = -1);
  ~Session();

  Session(Session const&) = delete;
  Session& operator=(Session const&) = delete;

  int gpu_id() const;
  int num_channels() const;

  // Memory registration for send/recv buffers.
  // For now we default to CUDA memory type internally (see .cpp).
  // Returns >0 mem_id on success; 0 on failure.
  uint64_t register_memory(void* buffer, size_t size, bool is_recv);
  Status   deregister_memory(uint64_t mem_id);

  // Create a transfer object bound to a connection identity.
  // Ownership: caller owns returned pointer and must delete it.
  Transfer* create_transfer(ConnID const& conn);

  // -------- Bootstrap / Handshake helpers (string-based, stable) --------
  // Server path:
  // 1) listen_json() => returns serialized connection info
  // 2) accept(remote_name)
  std::string listen_json();
  Status      accept(std::string const& remote_name);

  // Client path:
  // 1) load_remote_json(remote_name, json)
  // 2) connect(remote_name)
  Status      load_remote_json(std::string const& remote_name,
                               std::string const& conn_info_json);
  Status      connect(std::string const& remote_name);

 private:
  struct Impl;
  Impl* pimpl_;
};

/**
 * @brief Represents one logical batch of posted transfers.
 *        Supports post_send/post_recv and wait/is_complete lifecycle.
 */
class Transfer {
 public:
  ~Transfer();

  Transfer(Transfer const&) = delete;
  Transfer& operator=(Transfer const&) = delete;

  // Non-blocking post APIs (offset within the registered buffer)
  Status post_send(uint64_t mem_id, size_t offset, size_t size, int tag);
  Status post_recv(uint64_t mem_id, size_t offset, size_t size, int tag);

  // Poll/Wait
  bool   is_complete();
  Status wait();  // blocks until all completed

  // Stats
  int total_chunks() const;
  int completed_chunks() const;

  // Release any transfer-scoped resources (safe after wait())
  Status release();

  // Convenience: split into equal chunks and post in a loop.
  // These only post; you still need wait() or is_complete().
  Status send_all(uint64_t mem_id, size_t total_size, size_t offset,
                  size_t chunk_bytes, int tag_base);
  Status recv_all(uint64_t mem_id, size_t total_size, size_t offset,
                  size_t chunk_bytes, int tag_base);

 private:
  friend class Session;
  struct Impl;
  Impl* pimpl_;
  explicit Transfer(Impl* impl);
};

// -------- High-level OOB helpers (optional, blocking socket I/O) --------

// Server: produce conn JSON via session->listen_json() and write (len + bytes)
// to sockfd. Returns kOk on success.
Status server_send_conn_json(Session* sess, int sockfd);

// Client: read (len + bytes) from sockfd, then call
// session->load_remote_json(remote, json). Returns kOk on success.
Status client_recv_conn_json(Session* sess, int sockfd,
                             std::string const& remote);

// Default knobs
struct Defaults {
  static constexpr int    kBootstrapPort      = 12347;
  static constexpr size_t kDefaultChunkBytes  = 512 * 1024;  // 512KB
};

} // namespace tcpx_plugin

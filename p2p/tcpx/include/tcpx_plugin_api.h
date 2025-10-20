#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace tcpx_plugin {

// -----------------------------------------------------------
// Status codes
// -----------------------------------------------------------
enum class Status : int {
  kOk          = 0,
  kUnavailable = -1,
  kInvalidArg  = -2,
  kInternal    = -3,
  kTimeout     = -4,
};

// -----------------------------------------------------------
// Opaque handles
// -----------------------------------------------------------
using ConnHandle = uint64_t;  // connection
using MrHandle   = uint64_t;  // registered memory region
using TxHandle   = uint64_t;  // async transfer

// -----------------------------------------------------------
// Transport configuration
// -----------------------------------------------------------
struct Config {
  const char* plugin_path = nullptr;    // optional plugin .so
  int         gpu_id      = 0;          // CUDA device id
  int         channels    = 1;          // number of TCPX channels
  size_t      chunk_bytes = 512 * 1024; // default chunk size
  bool        enable      = true;       // enable transport at start
};

// -----------------------------------------------------------
// Main transport facade (direct constructed instance)
// -----------------------------------------------------------
class Transport {
public:
  // Construct with explicit config
  explicit Transport(const Config& cfg);

  ~Transport();

  // ---------------------------------------------------------
  // Connection management (blocking)
  // ---------------------------------------------------------
  Status connect(const std::string& ip,
                 int                port,
                 const std::string& remote_name,
                 ConnHandle&        out_conn);

  Status accept(int                port,
                const std::string& remote_name,
                ConnHandle&        out_conn);

  // ---------------------------------------------------------
  // Memory registration (clean version with only 2 APIs)
  // ---------------------------------------------------------
  Status register_memory(void*    ptr,
                         size_t   size,
                         bool     is_recv,
                         MrHandle& out_mr);

  Status deregister_memory(MrHandle mr);

  // ---------------------------------------------------------
  // Blocking send/recv
  // ---------------------------------------------------------
  Status send(ConnHandle conn,
              MrHandle   mr,
              size_t     total_bytes,
              int        tag_base = 100);

  Status recv(ConnHandle conn,
              MrHandle   mr,
              size_t     total_bytes,
              int        tag_base = 100);

  // ---------------------------------------------------------
  // Async send/recv
  // ---------------------------------------------------------
  Status send_async(ConnHandle conn,
                    MrHandle   mr,
                    size_t     total_bytes,
                    int        tag_base,
                    TxHandle&  out_tx);

  Status recv_async(ConnHandle conn,
                    MrHandle   mr,
                    size_t     total_bytes,
                    int        tag_base,
                    TxHandle&  out_tx);

  // Poll transfer for completion
  Status poll_transfer(TxHandle tx, bool& is_done);

private:
  struct Impl;
  Impl* pimpl_;
};

} // namespace tcpx

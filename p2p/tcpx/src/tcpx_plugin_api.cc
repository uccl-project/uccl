#include "tcpx_plugin_api.h"
#include "bootstrap.h"
#include "session_manager.h"
#include "transfer_manager.h"
#include "tcpx_interface.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace tcpx_plugin {

// -------------------------------
// Small helpers
// -------------------------------
static inline Status rc_to_status(int rc) {
  return rc == 0 ? Status::kOk : Status::kInternal;
}
static inline Status invalid() { return Status::kInvalidArg; }

// Best-effort CUDA device guard (keeps previous device, restores on exit)
struct ScopedCudaDevice {
  int  prev{-1};
  bool active{false};
  explicit ScopedCudaDevice(int gpu_id) {
    int cur = -1;
    if (cudaGetDevice(&cur) == cudaSuccess) {
      prev = cur;
      if (cudaSetDevice(gpu_id) == cudaSuccess) active = true;
    }
  }
  ~ScopedCudaDevice() {
    if (active && prev >= 0) cudaSetDevice(prev);
  }
};

// Opaque internal ID used by TcpxSession for memory registration
using MemId = uint64_t;

// ===========================================================
// Transport::Impl
// ===========================================================
struct Transport::Impl {
  // Config (copied from user)
  Config cfg{};
  // Session (manages channels, registration, transfer creation)
  std::unique_ptr<tcpx::TcpxSession> sess;

  // Chunk bytes for posting; must be >0
  size_t chunk_bytes{512 * 1024};

  // Handle allocators
  std::atomic<uint64_t> next_conn{1};
  std::atomic<uint64_t> next_mr{1};
  std::atomic<uint64_t> next_tx{1};

  // Per-connection state
  struct ConnState {
    std::string remote_name;  // key used by session->createTransfer()
  };

  // Async transfer tracking
  struct TxState {
    std::unique_ptr<tcpx::TcpxTransfer> xfer;
  };

  // Mappings protected by mutex
  std::mutex mu;
  std::unordered_map<ConnHandle, ConnState> conns;
  std::unordered_map<MrHandle, MemId>       mr_map;
  std::unordered_map<TxHandle, TxState>     tx_map;

  // ---------- Bootstrap wire helpers (length-prefixed JSON) ----------
  static Status send_conn_json(tcpx::TcpxSession* s, int fd) {
    if (!s || fd < 0) return Status::kInvalidArg;
    std::string json = s->listen();
    if (json.empty()) return Status::kInternal;

    uint32_t len = static_cast<uint32_t>(json.size());
    ssize_t n = ::write(fd, &len, sizeof(len));
    if (n != static_cast<ssize_t>(sizeof(len))) return Status::kInternal;

    ssize_t m = ::write(fd, json.data(), len);
    if (m != static_cast<ssize_t>(len)) return Status::kInternal;

    return Status::kOk;
  }

  static Status recv_conn_json(tcpx::TcpxSession* s, int fd,
                               const std::string& remote) {
    if (!s || fd < 0) return Status::kInvalidArg;

    uint32_t len = 0;
    ssize_t n = ::read(fd, &len, sizeof(len));
    if (n != static_cast<ssize_t>(sizeof(len))) return Status::kInternal;
    if (len == 0) return Status::kInternal;

    std::string json(len, '\0');
    ssize_t m = ::read(fd, json.data(), len);
    if (m != static_cast<ssize_t>(len)) return Status::kInternal;

    int rc = s->loadRemoteConnInfo(remote, json);
    return rc_to_status(rc);
  }

  // Fresh transfer for a given remote
  static std::unique_ptr<tcpx::TcpxTransfer>
  make_transfer(tcpx::TcpxSession* s, const std::string& remote) {
    if (!s) return nullptr;
    auto* raw = s->createTransfer(remote);
    return std::unique_ptr<tcpx::TcpxTransfer>(raw);
  }
};

// ===========================================================
// Transport - public API
// ===========================================================

Transport::Transport(const Config& cfg) : pimpl_(new Impl) {
  pimpl_->cfg = cfg;
  pimpl_->chunk_bytes = cfg.chunk_bytes ? cfg.chunk_bytes : (512 * 1024);

  // Ensure plugin presence (optional explicit load)
  int ndev = tcpx_get_device_count();
  if (ndev < 0 && cfg.plugin_path && cfg.plugin_path[0]) {
    (void)tcpx_load_plugin(cfg.plugin_path);
    ndev = tcpx_get_device_count();
  }
  // CUDA driver best-effort init (TcpxSession also retains primary ctx)
  (void)cuInit(0);

  // Create session immediately (direct-construct design)
  pimpl_->sess.reset(new tcpx::TcpxSession(cfg.gpu_id, cfg.channels));
}

Transport::~Transport() {
  // Release async transfers and maps first (order matters)
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    pimpl_->tx_map.clear();
    pimpl_->mr_map.clear();
    pimpl_->conns.clear();
  }
  // Destroy session last
  pimpl_->sess.reset();
  delete pimpl_;
}

// -----------------------------------------------------------
// Connection management (blocking)
// -----------------------------------------------------------

Status Transport::connect(const std::string& ip,
                          int                port,
                          const std::string& remote_name,
                          ConnHandle&        out_conn) {
  if (ip.empty() || port <= 0 || remote_name.empty()) return invalid();

  int fd = -1;
  if (bootstrap_client_connect(ip.c_str(), port, &fd) != 0) {
    return Status::kUnavailable;
  }

  Status st = Impl::recv_conn_json(pimpl_->sess.get(), fd, remote_name);
  ::close(fd);
  if (st != Status::kOk) return st;

  int rc = pimpl_->sess->connect(remote_name);
  if (rc != 0) return Status::kInternal;

  ConnHandle ch = pimpl_->next_conn.fetch_add(1);
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    pimpl_->conns.emplace(ch, Impl::ConnState{remote_name});
  }
  out_conn = ch;
  return Status::kOk;
}

Status Transport::accept(int                port,
                         const std::string& remote_name,
                         ConnHandle&        out_conn) {
  if (port <= 0 || remote_name.empty()) return invalid();

  int fd = -1;
  if (bootstrap_server_create(port, &fd) != 0) {
    return Status::kUnavailable;
  }

  Status st = Impl::send_conn_json(pimpl_->sess.get(), fd);
  ::close(fd);
  if (st != Status::kOk) return st;

  int rc = pimpl_->sess->accept(remote_name);
  if (rc != 0) return Status::kInternal;

  ConnHandle ch = pimpl_->next_conn.fetch_add(1);
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    pimpl_->conns.emplace(ch, Impl::ConnState{remote_name});
  }
  out_conn = ch;
  return Status::kOk;
}

// -----------------------------------------------------------
// Memory registration (clean 2-API version)
// -----------------------------------------------------------

Status Transport::register_memory(void*    ptr,
                                  size_t   size,
                                  bool     is_recv,
                                  MrHandle& out_mr) {
  if (!ptr || size == 0) return Status::kInvalidArg;

  ScopedCudaDevice guard(pimpl_->cfg.gpu_id);

  // IMPORTANT: use NCCL_PTR_CUDA, don't hardcode constants
  uint64_t mem_id = pimpl_->sess->registerMemory(ptr, size, NCCL_PTR_CUDA, is_recv);
  if (mem_id == 0) return Status::kInternal;

  MrHandle h = pimpl_->next_mr.fetch_add(1);
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    pimpl_->mr_map.emplace(h, mem_id);
  }
  out_mr = h;
  return Status::kOk;
}

Status Transport::deregister_memory(MrHandle mr) {
  if (mr == 0) return invalid();

  MemId mem_id = 0;
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    auto it = pimpl_->mr_map.find(mr);
    if (it == pimpl_->mr_map.end()) return Status::kInvalidArg;
    mem_id = it->second;
    pimpl_->mr_map.erase(it);
  }

  int rc = pimpl_->sess->deregisterMemory(mem_id);
  return rc_to_status(rc);
}

// -----------------------------------------------------------
// Blocking send / recv
// -----------------------------------------------------------

Status Transport::send(ConnHandle conn,
                       MrHandle   mr,
                       size_t     total_bytes,
                       int        tag_base) {
  if (conn == 0 || mr == 0 || total_bytes == 0) return invalid();

  // Ensure we are on the configured CUDA device during posting
  ScopedCudaDevice guard(pimpl_->cfg.gpu_id);

  std::string remote;
  MemId mem_id = 0;
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    auto cit = pimpl_->conns.find(conn);
    if (cit == pimpl_->conns.end()) return Status::kInvalidArg;
    remote = cit->second.remote_name;

    auto mit = pimpl_->mr_map.find(mr);
    if (mit == pimpl_->mr_map.end()) return Status::kInvalidArg;
    mem_id = mit->second;
  }

  auto xfer = Impl::make_transfer(pimpl_->sess.get(), remote);
  if (!xfer) return Status::kInternal;

  const size_t chunk = pimpl_->chunk_bytes;
  if (chunk == 0) return Status::kInvalidArg;

  // Post all chunks using offset within the single registered MR.
  size_t posted = 0; int idx = 0;
  while (posted < total_bytes) {
    size_t n = std::min(chunk, total_bytes - posted);
    int rc = xfer->postSend(mem_id, /*offset=*/posted, /*size=*/n,
                            /*tag=*/tag_base + idx);
    if (rc != 0) return Status::kInternal;
    posted += n; ++idx;
  }

  if (xfer->wait() != 0) return Status::kInternal;
  if (xfer->release() != 0) return Status::kInternal;
  return Status::kOk;
}

Status Transport::recv(ConnHandle conn,
                       MrHandle   mr,
                       size_t     total_bytes,
                       int        tag_base) {
  if (conn == 0 || mr == 0 || total_bytes == 0) return invalid();

  // Ensure we are on the configured CUDA device during posting
  ScopedCudaDevice guard(pimpl_->cfg.gpu_id);

  std::string remote;
  MemId mem_id = 0;
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    auto cit = pimpl_->conns.find(conn);
    if (cit == pimpl_->conns.end()) return Status::kInvalidArg;
    remote = cit->second.remote_name;

    auto mit = pimpl_->mr_map.find(mr);
    if (mit == pimpl_->mr_map.end()) return Status::kInvalidArg;
    mem_id = mit->second;
  }

  auto xfer = Impl::make_transfer(pimpl_->sess.get(), remote);
  if (!xfer) return Status::kInternal;

  const size_t chunk = pimpl_->chunk_bytes;
  if (chunk == 0) return Status::kInvalidArg;

  // Post all chunks using offset within the single registered MR.
  size_t recvd = 0; int idx = 0;
  while (recvd < total_bytes) {
    size_t n = std::min(chunk, total_bytes - recvd);
    int rc = xfer->postRecv(mem_id, /*offset=*/recvd, /*size=*/n,
                            /*tag=*/tag_base + idx);
    if (rc != 0) return Status::kInternal;
    recvd += n; ++idx;
  }

  if (xfer->wait() != 0) return Status::kInternal;
  if (xfer->release() != 0) return Status::kInternal;
  return Status::kOk;
}

// -----------------------------------------------------------
// Async send / recv
// -----------------------------------------------------------

Status Transport::send_async(ConnHandle conn,
                             MrHandle   mr,
                             size_t     total_bytes,
                             int        tag_base,
                             TxHandle&  out_tx) {
  if (conn == 0 || mr == 0 || total_bytes == 0) return invalid();

  // Ensure we are on the configured CUDA device during posting
  ScopedCudaDevice guard(pimpl_->cfg.gpu_id);

  std::string remote;
  MemId mem_id = 0;
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    auto cit = pimpl_->conns.find(conn);
    if (cit == pimpl_->conns.end()) return Status::kInvalidArg;
    remote = cit->second.remote_name;

    auto mit = pimpl_->mr_map.find(mr);
    if (mit == pimpl_->mr_map.end()) return Status::kInvalidArg;
    mem_id = mit->second;
  }

  auto xfer = Impl::make_transfer(pimpl_->sess.get(), remote);
  if (!xfer) return Status::kInternal;

  const size_t chunk = pimpl_->chunk_bytes;
  if (chunk == 0) return Status::kInvalidArg;

  // Post all chunks only; do not wait here (non-blocking).
  size_t posted = 0; int idx = 0;
  while (posted < total_bytes) {
    size_t n = std::min(chunk, total_bytes - posted);
    int rc = xfer->postSend(mem_id, /*offset=*/posted, /*size=*/n,
                            /*tag=*/tag_base + idx);
    if (rc != 0) return Status::kInternal;
    posted += n; ++idx;
  }

  TxHandle th = pimpl_->next_tx.fetch_add(1);
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    pimpl_->tx_map.emplace(th, Impl::TxState{std::move(xfer)});
  }
  out_tx = th;
  return Status::kOk;
}

Status Transport::recv_async(ConnHandle conn,
                             MrHandle   mr,
                             size_t     total_bytes,
                             int        tag_base,
                             TxHandle&  out_tx) {
  if (conn == 0 || mr == 0 || total_bytes == 0) return invalid();

  // Ensure we are on the configured CUDA device during posting
  ScopedCudaDevice guard(pimpl_->cfg.gpu_id);

  std::string remote;
  MemId mem_id = 0;
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    auto cit = pimpl_->conns.find(conn);
    if (cit == pimpl_->conns.end()) return Status::kInvalidArg;
    remote = cit->second.remote_name;

    auto mit = pimpl_->mr_map.find(mr);
    if (mit == pimpl_->mr_map.end()) return Status::kInvalidArg;
    mem_id = mit->second;
  }

  auto xfer = Impl::make_transfer(pimpl_->sess.get(), remote);
  if (!xfer) return Status::kInternal;

  const size_t chunk = pimpl_->chunk_bytes;
  if (chunk == 0) return Status::kInvalidArg;

  // Post all chunks only; do not wait here (non-blocking).
  size_t recvd = 0; int idx = 0;
  while (recvd < total_bytes) {
    size_t n = std::min(chunk, total_bytes - recvd);
    int rc = xfer->postRecv(mem_id, /*offset=*/recvd, /*size=*/n,
                            /*tag=*/tag_base + idx);
    if (rc != 0) return Status::kInternal;
    recvd += n; ++idx;
  }

  TxHandle th = pimpl_->next_tx.fetch_add(1);
  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    pimpl_->tx_map.emplace(th, Impl::TxState{std::move(xfer)});
  }
  out_tx = th;
  return Status::kOk;
}

Status Transport::poll_transfer(TxHandle tx, bool& is_done) {
  if (tx == 0) return invalid();

  std::unique_ptr<tcpx::TcpxTransfer> xfer_to_finalize;

  {
    std::lock_guard<std::mutex> g(pimpl_->mu);
    auto it = pimpl_->tx_map.find(tx);
    if (it == pimpl_->tx_map.end()) return Status::kInvalidArg;

    // Non-blocking completion check across all channels
    is_done = it->second.xfer->isComplete();

    if (is_done) {
      // Drain and release before erasing
      if (it->second.xfer->wait() != 0) return Status::kInternal;
      if (it->second.xfer->release() != 0) return Status::kInternal;
      pimpl_->tx_map.erase(it);
    }
  }

  return Status::kOk;
}

}  // namespace tcpx_plugin

#include "tcpx_plugin_api.h"

// Internal headers (from your existing tcpx include directory)
#include "session_manager.h"
#include "transfer_manager.h"
#include "bootstrap.h"
#include "tcpx_interface.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tcpx_plugin {

struct Transfer::Impl {
  std::unique_ptr<tcpx::TcpxTransfer> xfer;
};

// ---------- small utils ----------
static inline Status to_status(bool ok) {
  return ok ? Status::kOk : Status::kInternal;
}
static inline Status fail(Status s = Status::kInternal) { return s; }

// Best-effort CUDA device guard
ScopedCudaDevice::ScopedCudaDevice(int gpu_id) {
  int cur = -1;
  if (cudaGetDevice(&cur) == cudaSuccess) {
    prev_ = cur;
    // Always set device to ensure context is current
    if (cudaSetDevice(gpu_id) == cudaSuccess) {
      active_ = true;
    }
  }
}
ScopedCudaDevice::~ScopedCudaDevice() {
  if (active_ && prev_ >= 0) {
    cudaSetDevice(prev_);
  }
}

// ---------- init / caps ----------
Status init(InitOptions const& opts) {
  // Try to ensure the TCPX plugin is available. If count < 0, optionally try load.
  int n = tcpx_get_device_count();
  if (n < 0) {
    if (opts.plugin_path && opts.plugin_path[0]) {
      if (tcpx_load_plugin(opts.plugin_path) != 0) return Status::kUnavailable;
    } else {
      // rely on default search; if still unavailable, report error
      if (tcpx_get_device_count() < 0) return Status::kUnavailable;
    }
  }
  // Initialize CUDA driver (best-effort)
  if (cuInit(0) != CUDA_SUCCESS) {
    return Status::kUnavailable;
  }
  return Status::kOk;
}

Status get_device_count(int* out_count) {
  if (!out_count) return Status::kInvalidArg;
  int n = tcpx_get_device_count();
  if (n < 0) return Status::kUnavailable;
  *out_count = n;
  return Status::kOk;
}

// ---------- Session PIMPL ----------
struct Session::Impl {
  int gpu_id = 0;
  int num_channels = 0;
  std::string bootstrap_info;
  int nic_device_id = -1;

  std::unique_ptr<tcpx::TcpxSession> sess;

  Impl(int gid, int nch, std::string const& bs, int nic)
      : gpu_id(gid), num_channels(nch), bootstrap_info(bs), nic_device_id(nic),
        sess(new tcpx::TcpxSession(gid, nch)) {}
};

Session::Session(int gpu_id, int num_channels,
                 std::string const& bootstrap_info,
                 int nic_device_id)
    : pimpl_(new Impl(gpu_id, num_channels, bootstrap_info, nic_device_id)) {}

Session::~Session() { delete pimpl_; }

int Session::gpu_id() const { return pimpl_->gpu_id; }
int Session::num_channels() const { return pimpl_->num_channels; }

uint64_t Session::register_memory(void* buffer, size_t size, bool is_recv) {
  return pimpl_->sess->registerMemory(buffer, size, NCCL_PTR_CUDA, is_recv);
}

Status Session::deregister_memory(uint64_t mem_id) {
  int rc = pimpl_->sess->deregisterMemory(mem_id);
  return rc == 0 ? Status::kOk : Status::kInternal;
}

Transfer* Session::create_transfer(ConnID const& conn) {
  (void)conn;
  auto* t = pimpl_->sess->createTransfer(conn.remote);
  if (!t) return nullptr;

  return new Transfer(new Transfer::Impl{std::unique_ptr<tcpx::TcpxTransfer>(t)});
}

// Bootstrap helpers (string JSON roundtrip)
std::string Session::listen_json() {
  return pimpl_->sess->listen();
}

Status Session::accept(std::string const& remote_name) {
  int rc = pimpl_->sess->accept(remote_name);
  return rc == 0 ? Status::kOk : Status::kInternal;
}

Status Session::load_remote_json(std::string const& remote_name,
                                 std::string const& conn_info_json) {
  int rc = pimpl_->sess->loadRemoteConnInfo(remote_name, conn_info_json);
  return rc == 0 ? Status::kOk : Status::kInvalidArg;
}

Status Session::connect(std::string const& remote_name) {
  int rc = pimpl_->sess->connect(remote_name);
  return rc == 0 ? Status::kOk : Status::kInternal;
}

// ---------- Transfer PIMPL ----------
Transfer::Transfer(Impl* impl) : pimpl_(impl) {}
Transfer::~Transfer() { delete pimpl_; }

Status Transfer::post_send(uint64_t mem_id, size_t offset, size_t size, int tag) {
  int rc = pimpl_->xfer->postSend(mem_id, offset, size, tag);
  return rc == 0 ? Status::kOk : Status::kInternal;
}

Status Transfer::post_recv(uint64_t mem_id, size_t offset, size_t size, int tag) {
  int rc = pimpl_->xfer->postRecv(mem_id, offset, size, tag);
  return rc == 0 ? Status::kOk : Status::kInternal;
}

bool Transfer::is_complete() {
  return pimpl_->xfer->isComplete();
}

Status Transfer::wait() {
  int rc = pimpl_->xfer->wait();
  return rc == 0 ? Status::kOk : Status::kInternal;
}

int Transfer::total_chunks() const {
  return pimpl_->xfer->getTotalChunks();
}

int Transfer::completed_chunks() const {
  return pimpl_->xfer->getCompletedChunks();
}

Status Transfer::release() {
  int rc = pimpl_->xfer->release();
  return rc == 0 ? Status::kOk : Status::kInternal;
}

Status Transfer::send_all(uint64_t mem_id, size_t total_size, size_t offset,
                          size_t chunk_bytes, int tag_base) {
  if (chunk_bytes == 0) return Status::kInvalidArg;
  size_t sent = 0; int idx = 0;
  while (sent < total_size) {
    size_t n = std::min(chunk_bytes, total_size - sent);
    auto st = post_send(mem_id, offset + sent, n, tag_base + idx);
    if (st != Status::kOk) return st;
    sent += n; ++idx;
  }
  return Status::kOk;
}

Status Transfer::recv_all(uint64_t mem_id, size_t total_size, size_t offset,
                          size_t chunk_bytes, int tag_base) {
  if (chunk_bytes == 0) return Status::kInvalidArg;
  size_t recvd = 0; int idx = 0;
  while (recvd < total_size) {
    size_t n = std::min(chunk_bytes, total_size - recvd);
    auto st = post_recv(mem_id, offset + recvd, n, tag_base + idx);
    if (st != Status::kOk) return st;
    recvd += n; ++idx;
  }
  return Status::kOk;
}

// ---------- High-level OOB helpers (blocking) ----------
Status server_send_conn_json(Session* sess, int sockfd) {
  if (!sess || sockfd < 0) return Status::kInvalidArg;
  std::string json = sess->listen_json();
  if (json.empty()) return Status::kInternal;

  uint32_t len = static_cast<uint32_t>(json.size());
  ssize_t n = ::write(sockfd, &len, sizeof(len));
  if (n != static_cast<ssize_t>(sizeof(len))) return Status::kInternal;

  ssize_t m = ::write(sockfd, json.data(), len);
  if (m != static_cast<ssize_t>(len)) return Status::kInternal;

  return Status::kOk;
}

Status client_recv_conn_json(Session* sess, int sockfd,
                             std::string const& remote) {
  if (!sess || sockfd < 0) return Status::kInvalidArg;

  uint32_t len = 0;
  ssize_t n = ::read(sockfd, &len, sizeof(len));
  if (n != static_cast<ssize_t>(sizeof(len))) return Status::kInternal;

  std::string json(len, '\0');
  ssize_t m = ::read(sockfd, json.data(), len);
  if (m != static_cast<ssize_t>(len)) return Status::kInternal;

  auto st = sess->load_remote_json(remote, json);
  if (st != Status::kOk) return st;

  return Status::kOk;
}

} // namespace tcpx_plugin

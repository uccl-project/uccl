#include "nccl/nccl_endpoint.h"
#include "util/gpu_rt.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

namespace tcp {
namespace {
// Control-plane handshake constants.
constexpr uint32_t kMagic = 0x4e43434c;  // "NCCL"
constexpr uint32_t kVersion = 1;

struct ClientHello {
  uint32_t magic;
  uint32_t version;
  int gpu_idx;
};

struct ServerHello {
  uint32_t magic;
  uint32_t version;
  int gpu_idx;
  ncclUniqueId uid_rank0;
  ncclUniqueId uid_rank1;
};

enum CtrlMsgType : uint8_t { kWriteReq = 1, kReadReq = 2 };

struct CtrlMsg {
  // type: kWriteReq => peer should recv into addr; kReadReq => peer should send
  // from addr. addr is the peer-local pointer from the FIFO descriptor.
  uint8_t type;
  uint8_t reserved[3];
  uint32_t size;
  uint64_t addr;
};
static_assert(sizeof(CtrlMsg) == 16, "CtrlMsg must be 16 bytes");

int get_env_int(char const* key, int def) {
  char const* v = std::getenv(key);
  if (!v || !*v) return def;
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v || parsed <= 0 || parsed > 65535) return def;
  return static_cast<int>(parsed);
}

// Record an event on record_stream and make wait_stream wait on it.
bool record_and_wait(gpuStream_t record_stream, gpuStream_t wait_stream) {
  gpuEvent_t evt = nullptr;
  if (gpuEventCreateWithFlags(&evt, gpuEventDisableTiming) != gpuSuccess) {
    return false;
  }
  if (gpuEventRecord(evt, record_stream) != gpuSuccess) {
    gpuEventDestroy(evt);
    return false;
  }
  if (gpuStreamWaitEvent(wait_stream, evt, 0) != gpuSuccess) {
    gpuEventDestroy(evt);
    return false;
  }
  gpuEventDestroy(evt);
  return true;
}

bool fence_default_stream(int device, gpuStream_t stream) {
#if defined(UCCL_P2P_USE_RCCL)
  // HIP does not guarantee legacy/PTDS semantics identical to CUDA.
  // Keep the RCCL path simple to avoid stream/event edge cases.
  if (!stream) return false;
  if (gpuSetDevice(device) != gpuSuccess) return false;
  return true;
#else
  if (!stream) return false;
  if (gpuSetDevice(device) != gpuSuccess) return false;

  // Fence both legacy and per-thread default streams.
  if (!record_and_wait(cudaStreamLegacy, stream)) return false;
  if (cudaStreamPerThread != cudaStreamLegacy) {
    if (!record_and_wait(cudaStreamPerThread, stream)) return false;
  }
  return true;
#endif
}

uccl::ConnID make_invalid_conn() {
  // Helper to return a consistent "invalid" connection.
  uccl::ConnID invalid{};
  invalid.context = nullptr;
  invalid.sock_fd = -1;
  invalid.flow_id = UINT64_MAX;
  invalid.peer_id = 0;
  invalid.dev = -1;
  return invalid;
}
}  // namespace

struct TCPEndpoint::Conn {
  // One TCP control socket + two NCCL communicators (one per direction).
  int sock_fd = -1;
  int rank = -1;
  int remote_rank = -1;
  int local_gpu_idx = 0;
  ncclComm_t comm[2] = {nullptr, nullptr};
  gpuStream_t stream[2] = {nullptr, nullptr};
  std::atomic<bool> stop{false};
  // Serialize control messages (read/write requests).
  std::mutex ctrl_send_mu;
  // Dedicated control-plane thread for this connection.
  std::thread ctrl_thread;
};

struct TCPEndpoint::AsyncHandle {
  // Completion event for a single NCCL op.
  gpuEvent_t event = nullptr;
};

TCPEndpoint::TCPEndpoint(int gpu_index, uint16_t port)
    : gpu_index_(gpu_index), listen_port_(0), listen_fd_(-1) {
  // Port selection priority: explicit port arg, then env overrides.
  uint16_t chosen_port = port;
  if (chosen_port == 0) {
    int env_port = get_env_int("UCCL_TCP_OOB_PORT", 0);
    if (env_port == 0) {
      env_port = get_env_int("UCCL_TCPX_OOB_PORT", 0);
    }
    chosen_port = static_cast<uint16_t>(env_port);
  }
  if (!setup_listener_(chosen_port)) {
    std::cerr << "[tcp] failed to set up listen socket" << std::endl;
  }
}

TCPEndpoint::~TCPEndpoint() {
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    // Stop control threads and release NCCL/CUDA resources.
    for (auto& kv : conn_map_) {
      cleanup_conn_(*kv.second);
    }
    conn_map_.clear();
  }
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
}

bool TCPEndpoint::setup_listener_(uint16_t port) {
  // Create the TCP control-plane listen socket.
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return false;

  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::cerr << "[tcp] bind failed: " << strerror(errno) << " port=" << port
              << std::endl;
    ::close(fd);
    return false;
  }
  if (listen(fd, 8) < 0) {
    std::cerr << "[tcp] listen failed: " << strerror(errno) << std::endl;
    ::close(fd);
    return false;
  }

  if (port == 0) {
    // Resolve ephemeral port chosen by the OS.
    sockaddr_in bound{};
    socklen_t len = sizeof(bound);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &len) == 0) {
      listen_port_ = ntohs(bound.sin_port);
    }
  } else {
    listen_port_ = port;
  }
  listen_fd_ = fd;
  return true;
}

bool TCPEndpoint::send_all_(int fd, void const* buf, size_t len) const {
  // Blocking send helper used by the control-plane.
  char const* p = static_cast<char const*>(buf);
  size_t sent = 0;
  while (sent < len) {
    ssize_t rc = ::send(fd, p + sent, len - sent, 0);
    if (rc <= 0) return false;
    sent += static_cast<size_t>(rc);
  }
  return true;
}

bool TCPEndpoint::recv_all_(int fd, void* buf, size_t len) const {
  // Blocking recv helper used by the control-plane.
  char* p = static_cast<char*>(buf);
  size_t recvd = 0;
  while (recvd < len) {
    ssize_t rc = ::recv(fd, p + recvd, len - recvd, 0);
    if (rc <= 0) return false;
    recvd += static_cast<size_t>(rc);
  }
  return true;
}

bool TCPEndpoint::init_comm_(Conn& conn, ncclUniqueId const& uid,
                             int comm_index) {
  // comm_index selects which NCCL communicator (0 or 1) to init.
  if (comm_index < 0 || comm_index > 1) return false;
  if (gpuSetDevice(conn.local_gpu_idx) != gpuSuccess) return false;
  if (gpuStreamCreateWithFlags(&conn.stream[comm_index],
                               gpuStreamNonBlocking) != gpuSuccess) {
    return false;
  }
  ncclResult_t rc = ncclCommInitRank(&conn.comm[comm_index], 2, uid, conn.rank);
  if (rc != ncclSuccess) {
    std::cerr << "[tcp] ncclCommInitRank failed: " << ncclGetErrorString(rc)
              << std::endl;
    return false;
  }
  return true;
}

bool TCPEndpoint::init_comms_(Conn& conn, ncclUniqueId const& uid_rank0,
                              ncclUniqueId const& uid_rank1) {
  // Two communicators: one is used for the "send" direction, one for "recv".
  if (!init_comm_(conn, uid_rank0, 0)) return false;
  if (!init_comm_(conn, uid_rank1, 1)) return false;
  return true;
}

void TCPEndpoint::control_loop_(Conn* conn) {
  if (!conn) return;
  // Control thread: handle one-sided read/write requests from the peer.
  while (!conn->stop.load(std::memory_order_acquire)) {
    CtrlMsg msg{};
    if (!recv_all_(conn->sock_fd, &msg, sizeof(msg))) {
      break;
    }
    if (conn->stop.load(std::memory_order_acquire)) break;

    size_t size = static_cast<size_t>(msg.size);
    if (size == 0) continue;

    uccl::ucclRequest ureq{};
    int comm_index = comm_index_for_recv_(*conn);
    bool ok = false;
    switch (msg.type) {
      case kWriteReq:
        ok = recv_internal_(*conn, reinterpret_cast<void*>(msg.addr), size,
                            comm_index, &ureq);
        break;
      case kReadReq:
        ok = send_internal_(*conn, reinterpret_cast<void*>(msg.addr), size,
                            comm_index, &ureq);
        break;
      default:
        std::cerr << "[tcp] unknown ctrl msg type: "
                  << static_cast<int>(msg.type) << std::endl;
        return;
    }
    if (!ok) {
      std::cerr << "[tcp] failed to handle ctrl msg type: "
                << static_cast<int>(msg.type) << std::endl;
      return;
    }

    // Block until the NCCL op completes before processing the next request.
    while (!uccl_poll_ureq_once(&ureq)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

int TCPEndpoint::comm_index_for_send_(Conn const& conn) const {
  // Each side sends on its own rank's communicator.
  if (conn.rank == 0 || conn.rank == 1) return conn.rank;
  return 0;
}

int TCPEndpoint::comm_index_for_recv_(Conn const& conn) const {
  // Each side receives on the peer rank's communicator.
  if (conn.remote_rank == 0 || conn.remote_rank == 1) return conn.remote_rank;
  return 1;
}

bool TCPEndpoint::send_internal_(Conn& conn, void const* data, size_t size,
                                 int comm_index, uccl::ucclRequest* ureq) {
  if (!ureq) return false;
  if (size == 0) {
    ureq->context = nullptr;
    ureq->engine_idx = 0;
    return true;
  }
  if (!data) return false;
  if (comm_index < 0 || comm_index > 1) return false;
  if (!conn.comm[comm_index] || !conn.stream[comm_index]) return false;
  if (!fence_default_stream(conn.local_gpu_idx, conn.stream[comm_index])) {
    return false;
  }
  // Enqueue NCCL send on the selected communicator/stream.
  ncclResult_t rc = ncclSend(data, size, ncclChar, conn.remote_rank,
                             conn.comm[comm_index], conn.stream[comm_index]);
  if (rc != ncclSuccess) {
    std::cerr << "[tcp] ncclSend failed: " << ncclGetErrorString(rc)
              << std::endl;
    return false;
  }

  gpuEvent_t evt = nullptr;
  if (gpuEventCreateWithFlags(&evt, gpuEventDisableTiming) != gpuSuccess) {
    return false;
  }
  if (gpuEventRecord(evt, conn.stream[comm_index]) != gpuSuccess) {
    gpuEventDestroy(evt);
    return false;
  }

  // Track completion via ucclRequest.
  auto* handle = new AsyncHandle();
  handle->event = evt;
  ureq->context = handle;
  ureq->engine_idx = comm_index;
  return true;
}

bool TCPEndpoint::recv_internal_(Conn& conn, void* data, size_t size,
                                 int comm_index, uccl::ucclRequest* ureq) {
  if (!ureq) return false;
  if (size == 0) {
    ureq->context = nullptr;
    ureq->engine_idx = 0;
    return true;
  }
  if (!data) return false;
  if (comm_index < 0 || comm_index > 1) return false;
  if (!conn.comm[comm_index] || !conn.stream[comm_index]) return false;
  if (!fence_default_stream(conn.local_gpu_idx, conn.stream[comm_index])) {
    return false;
  }
  // Enqueue NCCL recv on the selected communicator/stream.
  ncclResult_t rc = ncclRecv(data, size, ncclChar, conn.remote_rank,
                             conn.comm[comm_index], conn.stream[comm_index]);
  if (rc != ncclSuccess) {
    std::cerr << "[tcp] ncclRecv failed: " << ncclGetErrorString(rc)
              << std::endl;
    return false;
  }

  gpuEvent_t evt = nullptr;
  if (gpuEventCreateWithFlags(&evt, gpuEventDisableTiming) != gpuSuccess) {
    return false;
  }
  if (gpuEventRecord(evt, conn.stream[comm_index]) != gpuSuccess) {
    gpuEventDestroy(evt);
    return false;
  }

  // Track completion via ucclRequest.
  auto* handle = new AsyncHandle();
  handle->event = evt;
  ureq->context = handle;
  ureq->engine_idx = comm_index;
  return true;
}

void TCPEndpoint::cleanup_conn_(Conn& conn) {
  // Stop control thread, then destroy streams/comms and close the socket.
  conn.stop.store(true, std::memory_order_release);
  if (conn.sock_fd >= 0) {
    ::shutdown(conn.sock_fd, SHUT_RDWR);
  }
  if (conn.ctrl_thread.joinable()) {
    conn.ctrl_thread.join();
  }
  if (conn.local_gpu_idx >= 0) gpuSetDevice(conn.local_gpu_idx);
  for (int i = 0; i < 2; ++i) {
    if (conn.stream[i]) {
      gpuStreamDestroy(conn.stream[i]);
      conn.stream[i] = nullptr;
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (conn.comm[i]) {
      ncclCommDestroy(conn.comm[i]);
      conn.comm[i] = nullptr;
    }
  }
  if (conn.sock_fd >= 0) {
    ::close(conn.sock_fd);
    conn.sock_fd = -1;
  }
}

uccl::ConnID TCPEndpoint::uccl_connect(int dev, int local_gpuidx,
                                       int remote_dev, int remote_gpuidx,
                                       std::string remote_ip,
                                       uint16_t remote_port) {
  // remote_dev/gpuidx are unused for TCP but part of the common API.
  (void)remote_dev;
  (void)remote_gpuidx;
  int local_idx = local_gpuidx >= 0 ? local_gpuidx : gpu_index_;
  if (local_gpuidx >= 0) gpu_index_ = local_gpuidx;

  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return make_invalid_conn();

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(remote_port);
  if (inet_pton(AF_INET, remote_ip.c_str(), &addr.sin_addr) <= 0) {
    ::close(fd);
    return make_invalid_conn();
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return make_invalid_conn();
  }

  // Handshake: send client hello and receive two NCCL unique IDs.
  ClientHello ch{.magic = kMagic, .version = kVersion, .gpu_idx = local_idx};
  if (!send_all_(fd, &ch, sizeof(ch))) {
    ::close(fd);
    return make_invalid_conn();
  }

  ServerHello sh{};
  if (!recv_all_(fd, &sh, sizeof(sh))) {
    ::close(fd);
    return make_invalid_conn();
  }
  if (sh.magic != kMagic || sh.version != kVersion) {
    ::close(fd);
    return make_invalid_conn();
  }

  // Build connection state and initialize NCCL communicators.
  auto conn = std::make_unique<Conn>();
  conn->sock_fd = fd;
  conn->rank = 1;
  conn->remote_rank = 0;
  conn->local_gpu_idx = local_idx;

  if (!init_comms_(*conn, sh.uid_rank0, sh.uid_rank1)) {
    cleanup_conn_(*conn);
    return make_invalid_conn();
  }

  uint64_t flow_id = next_flow_id_.fetch_add(1);
  Conn* conn_ptr = conn.get();
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    conn_map_.emplace(flow_id, std::move(conn));
  }
  conn_ptr->ctrl_thread =
      std::thread(&TCPEndpoint::control_loop_, this, conn_ptr);

  uccl::ConnID conn_id{};
  conn_id.context = conn_ptr;
  conn_id.sock_fd = conn_ptr->sock_fd;
  conn_id.flow_id = flow_id;
  conn_id.peer_id = 0;
  conn_id.dev = dev;
  return conn_id;
}

uccl::ConnID TCPEndpoint::uccl_accept(int dev, int listen_fd, int local_gpuidx,
                                      std::string& remote_ip, int* remote_dev,
                                      int* remote_gpuidx) {
  int local_idx = local_gpuidx >= 0 ? local_gpuidx : gpu_index_;
  if (local_gpuidx >= 0) gpu_index_ = local_gpuidx;
  int fd = listen_fd >= 0 ? listen_fd : listen_fd_;
  if (fd < 0) return make_invalid_conn();

  sockaddr_in cli_addr{};
  socklen_t len = sizeof(cli_addr);
  int conn_fd = ::accept(fd, reinterpret_cast<sockaddr*>(&cli_addr), &len);
  if (conn_fd < 0) return make_invalid_conn();

  char ip_buf[INET_ADDRSTRLEN] = {};
  if (!inet_ntop(AF_INET, &cli_addr.sin_addr, ip_buf, sizeof(ip_buf))) {
    ::close(conn_fd);
    return make_invalid_conn();
  }
  remote_ip = std::string(ip_buf);

  // Receive client hello.
  ClientHello ch{};
  if (!recv_all_(conn_fd, &ch, sizeof(ch))) {
    ::close(conn_fd);
    return make_invalid_conn();
  }
  if (ch.magic != kMagic || ch.version != kVersion) {
    ::close(conn_fd);
    return make_invalid_conn();
  }
  if (remote_dev) *remote_dev = 0;
  if (remote_gpuidx) *remote_gpuidx = ch.gpu_idx;

  // Generate two NCCL unique IDs, one per direction.
  ncclUniqueId uid_rank0;
  ncclUniqueId uid_rank1;
  if (ncclGetUniqueId(&uid_rank0) != ncclSuccess) {
    ::close(conn_fd);
    return make_invalid_conn();
  }
  if (ncclGetUniqueId(&uid_rank1) != ncclSuccess) {
    ::close(conn_fd);
    return make_invalid_conn();
  }

  ServerHello sh{};
  sh.magic = kMagic;
  sh.version = kVersion;
  sh.gpu_idx = local_idx;
  sh.uid_rank0 = uid_rank0;
  sh.uid_rank1 = uid_rank1;
  if (!send_all_(conn_fd, &sh, sizeof(sh))) {
    ::close(conn_fd);
    return make_invalid_conn();
  }

  // Build connection state and initialize NCCL communicators.
  auto conn = std::make_unique<Conn>();
  conn->sock_fd = conn_fd;
  conn->rank = 0;
  conn->remote_rank = 1;
  conn->local_gpu_idx = local_idx;

  if (!init_comms_(*conn, uid_rank0, uid_rank1)) {
    cleanup_conn_(*conn);
    return make_invalid_conn();
  }

  uint64_t flow_id = next_flow_id_.fetch_add(1);
  Conn* conn_ptr = conn.get();
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    conn_map_.emplace(flow_id, std::move(conn));
  }
  conn_ptr->ctrl_thread =
      std::thread(&TCPEndpoint::control_loop_, this, conn_ptr);

  uccl::ConnID conn_id{};
  conn_id.context = conn_ptr;
  conn_id.sock_fd = conn_ptr->sock_fd;
  conn_id.flow_id = flow_id;
  conn_id.peer_id = 0;
  conn_id.dev = dev;
  return conn_id;
}

int TCPEndpoint::uccl_regmr(uccl::UcclFlow* flow, void* data, size_t len,
                            int type, struct uccl::Mhandle** mhandle) {
  // TCP path does not register memory; keep for API compatibility.
  (void)flow;
  (void)data;
  (void)len;
  (void)type;
  if (mhandle) *mhandle = nullptr;
  return 0;
}

int TCPEndpoint::uccl_regmr(void* data, size_t len, MRArray& mr_array) {
  // No-op for TCP.
  (void)data;
  (void)len;
  (void)mr_array;
  return 0;
}

int TCPEndpoint::uccl_regmr(int dev, void* data, size_t len, int type,
                            struct uccl::Mhandle** mhandle) {
  // No-op for TCP.
  (void)dev;
  (void)data;
  (void)len;
  (void)type;
  if (mhandle) *mhandle = nullptr;
  return 0;
}

void TCPEndpoint::uccl_deregmr(struct uccl::Mhandle* mhandle) {
  // No-op for TCP.
  (void)mhandle;
}

void TCPEndpoint::uccl_deregmr(MRArray const& mr_array) { (void)mr_array; }

int TCPEndpoint::uccl_send_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                                 void const* data, size_t size,
                                 struct uccl::ucclRequest* ureq) {
  // Two-sided send: enqueue ncclSend on the send communicator.
  (void)mh;
  if (!flow || !ureq) return -1;
  Conn* conn = reinterpret_cast<Conn*>(flow);
  int comm_index = comm_index_for_send_(*conn);
  return send_internal_(*conn, data, size, comm_index, ureq) ? 0 : -1;
}

int TCPEndpoint::uccl_recv_async(uccl::UcclFlow* flow,
                                 struct uccl::Mhandle** mhandles, void** data,
                                 int* sizes, int n,
                                 struct uccl::ucclRequest* ureq) {
  // Two-sided recv: enqueue ncclRecv on the recv communicator.
  (void)mhandles;
  if (!flow || !ureq || !data || !sizes || n != 1) return -1;
  Conn* conn = reinterpret_cast<Conn*>(flow);
  int comm_index = comm_index_for_recv_(*conn);
  size_t size = static_cast<size_t>(sizes[0]);
  return recv_internal_(*conn, data[0], size, comm_index, ureq) ? 0 : -1;
}

int TCPEndpoint::uccl_read_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                                 void* dst, size_t size,
                                 uccl::FifoItem const& slot_item,
                                 uccl::ucclRequest* ureq) {
  // One-sided read: tell the peer to send, then post a local recv.
  (void)mh;
  if (!flow || !ureq) return -1;
  if (size == 0) {
    ureq->context = nullptr;
    ureq->engine_idx = 0;
    return 0;
  }
  Conn* conn = reinterpret_cast<Conn*>(flow);
  size_t xfer_size = size;
  if (slot_item.size > 0 && slot_item.size < xfer_size) {
    xfer_size = slot_item.size;
  }

  CtrlMsg msg{};
  msg.type = kReadReq;
  msg.size = static_cast<uint32_t>(xfer_size);
  msg.addr = slot_item.addr;
  {
    std::lock_guard<std::mutex> lock(conn->ctrl_send_mu);
    if (!send_all_(conn->sock_fd, &msg, sizeof(msg))) {
      return -1;
    }
  }

  int comm_index = comm_index_for_send_(*conn);
  return recv_internal_(*conn, dst, xfer_size, comm_index, ureq) ? 0 : -1;
}

int TCPEndpoint::uccl_write_async(uccl::UcclFlow* flow,
                                  struct uccl::Mhandle* mh, void* src,
                                  size_t size, uccl::FifoItem const& slot_item,
                                  uccl::ucclRequest* ureq) {
  // One-sided write: tell the peer to recv, then post a local send.
  (void)mh;
  if (!flow || !ureq) return -1;
  if (size == 0) {
    ureq->context = nullptr;
    ureq->engine_idx = 0;
    return 0;
  }
  Conn* conn = reinterpret_cast<Conn*>(flow);
  size_t xfer_size = size;
  if (slot_item.size > 0 && slot_item.size < xfer_size) {
    xfer_size = slot_item.size;
  }

  CtrlMsg msg{};
  msg.type = kWriteReq;
  msg.size = static_cast<uint32_t>(xfer_size);
  msg.addr = slot_item.addr;
  {
    std::lock_guard<std::mutex> lock(conn->ctrl_send_mu);
    if (!send_all_(conn->sock_fd, &msg, sizeof(msg))) {
      return -1;
    }
  }

  int comm_index = comm_index_for_send_(*conn);
  return send_internal_(*conn, src, xfer_size, comm_index, ureq) ? 0 : -1;
}

bool TCPEndpoint::uccl_poll_ureq_once(struct uccl::ucclRequest* ureq) {
  if (!ureq) return true;
  auto* handle = reinterpret_cast<AsyncHandle*>(ureq->context);
  if (!handle) return true;
  // Poll CUDA event for completion.
  gpuError_t rc = gpuEventQuery(handle->event);
  if (rc == gpuSuccess) {
    gpuEventDestroy(handle->event);
    delete handle;
    ureq->context = nullptr;
    return true;
  }
  if (rc == gpuErrorNotReady) {
    return false;
  }
  std::cerr << "[tcp] gpuEventQuery failed: " << gpuGetErrorString(rc)
            << std::endl;
  gpuEventDestroy(handle->event);
  delete handle;
  ureq->context = nullptr;
  return true;
}

int TCPEndpoint::prepare_fifo_metadata(uccl::UcclFlow* flow,
                                       struct uccl::Mhandle** mhandle,
                                       void const* data, size_t size,
                                       char* out_buf) {
  // Encode addr/size into uccl::FifoItem and serialize.
  (void)flow;
  (void)mhandle;
  uccl::FifoItem item{};
  item.addr = reinterpret_cast<uint64_t>(data);
  item.size = static_cast<uint32_t>(size);
  std::memset(item.padding, 0, sizeof(item.padding));
  uccl::serialize_fifo_item(item, out_buf);
  return 0;
}

}  // namespace tcp

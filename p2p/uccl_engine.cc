#include "uccl_engine.h"
#include "engine.h"
#include "rdma/epoll_client.h"
#ifdef UCCL_P2P_USE_NCCL
#include "nccl/nccl_endpoint.h"
#else
#include "endpoint_wrapper.h"
#endif
#include "util/util.h"
#include <arpa/inet.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdbool>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

using Endpoint = ::Endpoint;

struct uccl_engine {
  std::unique_ptr<Endpoint> endpoint;
  std::thread local_accept_thread;
  std::atomic<bool> local_accept_started{false};
  // Standalone OOB client for cross-process local notifications when the
  // transport endpoint does not provide one (e.g. NCCL/TCP build).
  std::shared_ptr<EpollClient> local_oob_client;
};

struct uccl_conn {
  uint64_t conn_id;
  uccl_engine* engine;
  int sock_fd;               // Keep for backward compatibility
  std::string oob_conn_key;  // For epoll-based notifications
  std::thread* listener_thread;
  bool listener_running;
  std::mutex listener_mutex;
  bool is_local = false;      // True for intra-node (IPC) connections
  bool same_process = false;  // True for same-process local transfers
};

typedef struct {
  FifoItem fifo_item;
  bool is_valid;
} fifo_item_t;

typedef struct {
  uint64_t mr_id;
  size_t size;
  IpcTransferInfo ipc_info = {};  // Pre-computed IPC handle for GPU buffers
  bool has_ipc = false;           // True if IPC handle is valid (GPU memory)
} mem_reg_entry_t;

std::unordered_map<uintptr_t, mem_reg_entry_t> mem_reg_info;

// Helper function to find the base address and mr_id for any address within a
// registered region
bool find_mem_reg(uintptr_t addr, uintptr_t& base_addr, uint64_t& mr_id) {
  for (auto const& [base, entry] : mem_reg_info) {
    if (addr >= base && addr < base + entry.size) {
      base_addr = base;
      mr_id = entry.mr_id;
      return true;
    }
  }
  return false;
}

// Look up the pre-computed IpcTransferInfo for any address within a registered
// GPU buffer region.  Adjusts offset and size to match the requested range.
// Returns false if the address is not registered or has no IPC handle (e.g.
// host memory).
static bool get_ipc_info_for_addr(uintptr_t addr, size_t size,
                                  IpcTransferInfo& out_info) {
  uintptr_t base_addr;
  uint64_t mr_id_unused;
  if (!find_mem_reg(addr, base_addr, mr_id_unused)) return false;
  auto const& entry = mem_reg_info.at(base_addr);
  if (!entry.has_ipc) return false;
  out_info = entry.ipc_info;
  // Shift the offset within the IPC allocation to point at the sub-range.
  out_info.offset += (addr - base_addr);
  out_info.size = size;
  return true;
}

// Deserialize IpcTransferInfo from an opaque buffer.
static void deserialize_ipc_info(char const* buf, IpcTransferInfo& info) {
  memset(&info, 0, sizeof(info));
  size_t off = 0;
  memcpy(&info.handle, buf + off, sizeof(info.handle));
  off += sizeof(info.handle);
  memcpy(&info.offset, buf + off, sizeof(info.offset));
  off += sizeof(info.offset);
  memcpy(&info.size, buf + off, sizeof(info.size));
  off += sizeof(info.size);
  memcpy(&info.gpu_idx, buf + off, sizeof(info.gpu_idx));
  off += sizeof(info.gpu_idx);
}

// Check UCCL_P2P_DISABLE_IPC=1 to force all transfers through the network
// transport (RDMA or TCP) even for intra-node peers.  Useful for CI testing.
static bool ipc_disabled() {
  static bool disabled = [] {
    char const* val = std::getenv("UCCL_P2P_DISABLE_IPC");
    bool dis = val && std::string(val) == "1";
    if (dis)
      std::cerr << "UCCL P2P: IPC disabled, all transfers will use "
                   "network transport"
                << std::endl;
    return dis;
  }();
  return disabled;
}

uccl_engine_t* uccl_engine_create(int num_cpus, bool in_python) {
  (void)num_cpus;
  inside_python = in_python;
  uccl_engine_t* eng = new uccl_engine;
  eng->endpoint = std::unique_ptr<Endpoint>(new Endpoint());
  return eng;
}

void uccl_engine_destroy(uccl_engine_t* engine) {
  if (engine) {
    engine->endpoint.reset();
    delete engine;
  }
}

uccl_conn_t* uccl_engine_connect(uccl_engine_t* engine, char const* ip_addr,
                                 char const* remote_gpu, int remote_port,
                                 bool same_process) {
  if (!engine || !ip_addr) return nullptr;
  uccl_conn_t* conn = new uccl_conn;
  uint64_t conn_id;

  // Detect intra-node connection: if the target IP matches our own IP.
  // When UCCL_P2P_DISABLE_IPC=1, treat all connections as remote so that
  // data flows through the network transport instead of IPC.
  std::string local_ip = uccl::get_oob_ip();
  bool is_local = !ipc_disabled() && (std::string(ip_addr) == local_ip);

  // Resolve remote_gpu: accept either a PCI BDF string (e.g. "0000:4a:00.0")
  // or a CUDA device index string (e.g. "0").
  std::string remote_bdf;
  if (remote_gpu && std::strchr(remote_gpu, ':')) {
    remote_bdf = uccl::normalize_pci_bus_id(remote_gpu);
  } else {
    int gpu_idx = remote_gpu ? std::atoi(remote_gpu) : 0;
    char bdf_buf[64];
    GPU_RT_CHECK(gpuDeviceGetPCIBusId(bdf_buf, sizeof(bdf_buf), gpu_idx));
    remote_bdf = uccl::normalize_pci_bus_id(bdf_buf);
  }

  bool ok;
  if (is_local && same_process) {
    // Same process: use shm rings directly, skip network transport.
    // Transfers use direct_addr (no gpuIpcOpenMemHandle).
    ok = engine->endpoint->connect_local(remote_bdf, conn_id, true);
    conn->sock_fd = -1;
  } else {
    // Remote or cross-process local: use network connection.
    // Cross-process same-node still sets conn->is_local = true below so that
    // NIXL can use the IPC data path via ipc_bufs; notifications go through
    // the network connection (RDMA OOB or NCCL send_notification).
    ok = engine->endpoint->connect(std::string(ip_addr), 0, remote_port,
                                   conn_id);
    if (ok) {
      conn->sock_fd = engine->endpoint->get_sock_fd(conn_id);
#if !defined(UCCL_P2P_USE_NCCL)
      conn->oob_conn_key = engine->endpoint->get_oob_conn_key(conn_id);
#endif
    }
  }

  if (!ok) {
    delete conn;
    return nullptr;
  }
  conn->conn_id = conn_id;
  conn->engine = engine;
  conn->is_local = is_local;
  conn->same_process = same_process;
  conn->listener_thread = nullptr;
  conn->listener_running = false;
  return conn;
}

uccl_conn_t* uccl_engine_accept(uccl_engine_t* engine, char* ip_addr_buf,
                                size_t ip_addr_buf_len, int* remote_gpu_idx) {
  if (!engine || !ip_addr_buf || !remote_gpu_idx) return nullptr;
#if !defined(UCCL_P2P_USE_NCCL)
  // RDMA: start the local accept thread for same-process IPC peers that
  // may connect via connect_local before the blocking RDMA accept returns.
  engine->endpoint->start_passive_accept_local();
#endif
  uccl_conn_t* conn = new uccl_conn;
  std::string ip_addr;
  uint64_t conn_id;
  int gpu_idx;
  bool ok = engine->endpoint->accept(ip_addr, gpu_idx, conn_id);
  if (!ok) {
    delete conn;
    return nullptr;
  }
  std::strncpy(ip_addr_buf, ip_addr.c_str(), ip_addr_buf_len);
  *remote_gpu_idx = gpu_idx;
  conn->conn_id = conn_id;
  conn->sock_fd = engine->endpoint->get_sock_fd(conn_id);
#if !defined(UCCL_P2P_USE_NCCL)
  conn->oob_conn_key = engine->endpoint->get_oob_conn_key(conn_id);
#endif
  conn->engine = engine;
  conn->listener_thread = nullptr;
  conn->listener_running = false;
  return conn;
}

int uccl_engine_reg(uccl_engine_t* engine, uintptr_t data, size_t size,
                    uccl_mr_t& mr_id) {
  if (!engine || !data) return -1;
  bool ok = engine->endpoint->reg((void*)data, size, mr_id);
  if (!ok) {
    return -1;
  }
  mem_reg_entry_t entry;
  entry.mr_id = mr_id;
  entry.size = size;

  // Pre-compute IPC handle for GPU buffers so the local (IPC) transfer path
  // can look it up without a per-transfer gpuIpcGetMemHandle call.
  int dev_idx = uccl::get_dev_idx((void*)data);
  if (dev_idx >= 0) {
    gpuSetDevice(dev_idx);
    static constexpr size_t kIpcAlignment = 1ul << 20;
    uintptr_t aligned = data & ~(static_cast<uintptr_t>(kIpcAlignment - 1));
    gpuError_t err = gpuIpcGetMemHandle(&entry.ipc_info.handle,
                                        reinterpret_cast<void*>(aligned));
    if (err == gpuSuccess) {
      entry.ipc_info.offset = data - aligned;
      entry.ipc_info.size = size;
      entry.ipc_info.gpu_idx = dev_idx;
      entry.has_ipc = true;
    }
  }

  mem_reg_info[data] = entry;
  return 0;
}

int uccl_engine_read(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                     size_t size, FifoItem fifo_item, uint64_t* transfer_id) {
  if (!conn || !data) return -1;

  if (conn->is_local) {
    IpcTransferInfo info;
    if (!get_ipc_info_for_addr(fifo_item.addr, size, info)) {
      UCCL_LOG(ERROR) << "Failed to get IPC info";
      return -1;
    }
    if (conn->same_process) info.direct_addr = fifo_item.addr;
    return conn->engine->endpoint->read_ipc_async(
               conn->conn_id, const_cast<void*>(data), size, info, transfer_id)
               ? 0
               : -1;
  }

  return conn->engine->endpoint->read_async(conn->conn_id, mr,
                                            const_cast<void*>(data), size,
                                            fifo_item, transfer_id)
             ? 0
             : -1;
}

int uccl_engine_read_vector(uccl_conn_t* conn, std::vector<uccl_mr_t> mr_ids,
                            std::vector<void*> dst_v,
                            std::vector<size_t> size_v,
                            std::vector<FifoItem> fifo_items, int num_iovs,
                            uint64_t* transfer_id,
                            std::vector<char*> ipc_bufs) {
  if (!conn || num_iovs <= 0) return -1;

#if defined(UCCL_P2P_USE_NCCL)
  // The NIXL UCCL backend always uses the vector API, even for a single iov.
  // For the NCCL/TCPX path, bypass the generic readv proxy-thread machinery in
  // that case and issue the scalar read directly.
  if (num_iovs == 1 && mr_ids.size() == 1 && dst_v.size() == 1 &&
      size_v.size() == 1 && fifo_items.size() == 1 && ipc_bufs.empty()) {
    return conn->engine->endpoint->read_async(conn->conn_id, mr_ids[0],
                                              dst_v[0], size_v[0],
                                              fifo_items[0], transfer_id)
               ? 0
               : -1;
  }
#endif

  // Local IPC path (both same-process and cross-process)
  if ((conn->is_local || conn->same_process) && !ipc_bufs.empty()) {
    std::vector<IpcTransferInfo> info_v(num_iovs);
    for (int i = 0; i < num_iovs; i++) {
      deserialize_ipc_info(ipc_bufs[i], info_v[i]);
      if (conn->same_process) {
        info_v[i].direct_addr = fifo_items[i].addr;
      }
    }
    return conn->engine->endpoint->readv_ipc_async(
               conn->conn_id, dst_v, size_v, info_v, num_iovs, transfer_id)
               ? 0
               : -1;
  }

  // Remote RDMA
  return conn->engine->endpoint->readv_async(conn->conn_id, mr_ids, dst_v,
                                             size_v, fifo_items, num_iovs,
                                             transfer_id)
             ? 0
             : -1;
}

int uccl_engine_send(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                     size_t size, uint64_t* transfer_id) {
  if (!conn || !data) return -1;
  return conn->engine->endpoint->send_async(conn->conn_id, mr, data, size,
                                            transfer_id)
             ? 0
             : -1;
}

int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t mr, void* data,
                     size_t data_size) {
  if (!conn || !data) return -1;

  return conn->engine->endpoint->recv(conn->conn_id, mr, data, data_size) ? 0
                                                                          : -1;
}

int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                      size_t size, FifoItem fifo_item, uint64_t* transfer_id) {
  if (!conn || !data) return -1;

  if (conn->is_local) {
    // Same-process local path: both buffers are in our address space.
    // Set direct_addr so write_ipc_async skips gpuIpcOpenMemHandle and
    // uses the virtual address directly, while keeping its stream/event
    // optimizations.
    IpcTransferInfo info;
    if (!get_ipc_info_for_addr(fifo_item.addr, size, info)) {
      UCCL_LOG(ERROR) << "Failed to get IPC info";
      return -1;
    }
    if (conn->same_process) info.direct_addr = fifo_item.addr;
    return conn->engine->endpoint->write_ipc_async(conn->conn_id, data, size,
                                                   info, transfer_id)
               ? 0
               : -1;
  }

  return conn->engine->endpoint->write_async(conn->conn_id, mr,
                                             const_cast<void*>(data), size,
                                             fifo_item, transfer_id)
             ? 0
             : -1;
}

int uccl_engine_write_vector(uccl_conn_t* conn, std::vector<uccl_mr_t> mr_ids,
                             std::vector<void*> dst_v,
                             std::vector<size_t> size_v,
                             std::vector<FifoItem> fifo_items, int num_iovs,
                             uint64_t* transfer_id,
                             std::vector<char*> ipc_bufs) {
  if (!conn || num_iovs <= 0) return -1;

#if defined(UCCL_P2P_USE_NCCL)
  // Mirror the single-iov fast path for writes so the merged NCCL endpoint
  // does not force 1-buffer transfers through the vector proxy path.
  if (num_iovs == 1 && mr_ids.size() == 1 && dst_v.size() == 1 &&
      size_v.size() == 1 && fifo_items.size() == 1 && ipc_bufs.empty()) {
    return conn->engine->endpoint->write_async(conn->conn_id, mr_ids[0],
                                               dst_v[0], size_v[0],
                                               fifo_items[0], transfer_id)
               ? 0
               : -1;
  }
#endif

  // Local IPC path (both same-process and cross-process)
  if ((conn->is_local || conn->same_process) && !ipc_bufs.empty()) {
    std::vector<void const*> src_v(dst_v.begin(), dst_v.end());
    std::vector<IpcTransferInfo> info_v(num_iovs);
    for (int i = 0; i < num_iovs; i++) {
      deserialize_ipc_info(ipc_bufs[i], info_v[i]);
      if (conn->same_process) {
        info_v[i].direct_addr = fifo_items[i].addr;
      }
    }
    return conn->engine->endpoint->writev_ipc_async(
               conn->conn_id, src_v, size_v, info_v, num_iovs, transfer_id)
               ? 0
               : -1;
  }

  // Remote RDMA
  return conn->engine->endpoint->writev_async(conn->conn_id, mr_ids, dst_v,
                                              size_v, fifo_items, num_iovs,
                                              transfer_id)
             ? 0
             : -1;
}

bool uccl_engine_xfer_status(uccl_conn_t* conn, uint64_t transfer_id) {
  bool is_done;
  conn->engine->endpoint->poll_async(transfer_id, &is_done);
  return is_done;
}

int uccl_engine_start_listener(uccl_conn_t* conn) {
  if (!conn) return -1;

  if (conn->listener_running) {
    return -1;
  }

  conn->listener_running = true;
  // Notifications are handled by backend control channels; no separate
  // listener thread is needed.
  conn->listener_thread = nullptr;

  return 0;
}

int uccl_engine_stop_listener(uccl_conn_t* conn) {
  if (!conn) return -1;

  std::lock_guard<std::mutex> lock(conn->listener_mutex);

  if (!conn->listener_running) {
    return 0;
  }

  conn->listener_running = false;

  if (conn->sock_fd >= 0) {
    close(conn->sock_fd);
    conn->sock_fd = -1;
  }

  if (conn->listener_thread && conn->listener_thread->joinable()) {
    auto future = std::async(std::launch::async,
                             [conn]() { conn->listener_thread->join(); });

    if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
      std::cout << "Warning: Listener thread not responding, detaching..."
                << std::endl;
      conn->listener_thread->detach();
    }
  }

  delete conn->listener_thread;
  conn->listener_thread = nullptr;

  return 0;
}

bool uccl_engine_conn_is_local(uccl_conn_t* conn) {
  return conn && conn->is_local;
}

void uccl_engine_stop_accept(uccl_engine_t* engine) {
  if (engine && engine->endpoint) {
    engine->endpoint->stop_accepting();
  }
}

void uccl_engine_conn_destroy(uccl_conn_t* conn) {
  if (conn) {
    uccl_engine_stop_listener(conn);

    delete conn;
  }
}

void uccl_engine_mr_destroy(uccl_engine_t* engine, uccl_mr_t mr) {
  for (auto it = mem_reg_info.begin(); it != mem_reg_info.end(); ++it) {
    if (it->second.mr_id == mr) {
      engine->endpoint->dereg(mr);
      mem_reg_info.erase(it);
      break;
    }
  }
}

int uccl_engine_prepare_fifo(uccl_engine_t* engine, uccl_mr_t mr,
                             void const* data, size_t size, char* fifo_buf) {
  if (!engine || !data || !fifo_buf) return -1;

  return engine->endpoint->prepare_fifo(mr, const_cast<void*>(data), size,
                                        fifo_buf)
             ? 0
             : -1;
}

int uccl_engine_update_fifo(FifoItem& fifo_item, uint64_t remote_addr,
                            uint32_t size) {
  fifo_item.addr = remote_addr;
  fifo_item.size = size;

  return 0;
}

std::vector<notify_msg_t> uccl_engine_get_notifs() {
  std::lock_guard<std::mutex> lock(notify_mutex);

  std::vector<notify_msg_t> result;
  for (auto const& oob_msg : notify_list) {
    notify_msg_t msg;
    strncpy(msg.name, oob_msg.name, sizeof(msg.name) - 1);
    msg.name[sizeof(msg.name) - 1] = '\0';
    memcpy(msg.msg, oob_msg.msg, sizeof(msg.msg));
    result.push_back(msg);
  }

  notify_list.clear();

  return result;
}

int uccl_engine_send_notif(uccl_conn_t* conn, notify_msg_t* notify_msg) {
  if (!conn || !notify_msg) return -1;

  NotifyMsg oob_msg;
  oob_msg.magic = NOTIFY_MSG_MAGIC;
  strncpy(oob_msg.name, notify_msg->name, sizeof(oob_msg.name) - 1);
  oob_msg.name[sizeof(oob_msg.name) - 1] = '\0';
  memcpy(oob_msg.msg, notify_msg->msg, sizeof(oob_msg.msg));

  // Same-process local connection: push notification directly to the local
  // list — no network path needed regardless of transport.
  if (conn->same_process) {
    std::lock_guard<std::mutex> lock(notify_mutex);
    notify_list.push_back(oob_msg);
    return 0;
  }

  // Cross-process local connection: send via OOB client (works for both
  // RDMA and TCP builds).  The oob_conn_key was set up in uccl_engine_connect.
  if (conn->is_local && !conn->oob_conn_key.empty()) {
    auto oob_client = conn->engine->endpoint->get_oob_client();
    if (!oob_client) {
      oob_client = conn->engine->local_oob_client;
    }
    if (!oob_client) {
      std::cerr << "No OOB client available for local notification"
                << std::endl;
      return -1;
    }
    std::string payload(reinterpret_cast<char*>(&oob_msg), sizeof(NotifyMsg));
    return oob_client->send_meta(conn->oob_conn_key, payload)
               ? sizeof(NotifyMsg)
               : -1;
  }

#if defined(UCCL_P2P_USE_NCCL)
  return conn->engine->endpoint->send_notification(conn->conn_id, oob_msg);
#else
  if (conn->oob_conn_key.empty()) {
    std::cerr << "No OOB connection key available for notification"
              << std::endl;
    return -1;
  }

  auto oob_client = conn->engine->endpoint->get_oob_client();
  if (!oob_client) {
    std::cerr << "No OOB client available for notification" << std::endl;
    return -1;
  }

  std::string payload(reinterpret_cast<char*>(&oob_msg), sizeof(NotifyMsg));
  bool ok = oob_client->send_meta(conn->oob_conn_key, payload);

  return ok ? sizeof(NotifyMsg) : -1;
#endif
}

// Serialize IpcTransferInfo to an opaque buffer (IPC_INFO_SIZE bytes).
// Layout: handle(64) + offset(8) + size(8) + gpu_idx(4) = 84 bytes.
static void serialize_ipc_info(IpcTransferInfo const& info, char* buf) {
  memset(buf, 0, IPC_INFO_SIZE);
  size_t off = 0;
  memcpy(buf + off, &info.handle, sizeof(info.handle));
  off += sizeof(info.handle);  // 64
  memcpy(buf + off, &info.offset, sizeof(info.offset));
  off += sizeof(info.offset);  // 8
  memcpy(buf + off, &info.size, sizeof(info.size));
  off += sizeof(info.size);  // 8
  memcpy(buf + off, &info.gpu_idx, sizeof(info.gpu_idx));
  off += sizeof(info.gpu_idx);  // 4
}

int uccl_engine_get_ipc_info(uccl_engine_t* engine, uintptr_t addr,
                             char* ipc_buf, bool* has_ipc) {
  if (!engine || !ipc_buf || !has_ipc) return -1;
  *has_ipc = false;
  if (ipc_disabled()) return 0;
  auto it = mem_reg_info.find(addr);
  if (it == mem_reg_info.end()) return -1;
  if (!it->second.has_ipc) return 0;
  serialize_ipc_info(it->second.ipc_info, ipc_buf);
  *has_ipc = true;
  return 0;
}

int uccl_engine_update_ipc_info(char* ipc_buf, uintptr_t addr,
                                uintptr_t base_addr, size_t size) {
  if (!ipc_buf) return -1;
  IpcTransferInfo info;
  deserialize_ipc_info(ipc_buf, info);
  info.offset += (addr - base_addr);
  info.size = size;
  serialize_ipc_info(info, ipc_buf);
  return 0;
}

int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata) {
  if (!engine || !metadata) return -1;

  try {
    std::vector<uint8_t> metadata_vec = engine->endpoint->get_metadata();

    std::string result;
    // New metadata format: [IP (4/16)] [port (2)] [bdf_len (1)] [bdf_str (N)]
    if (metadata_vec.size() >= 7 && metadata_vec.size() <= 30) {
      // Try to detect IPv4 vs IPv6 by parsing the BDF length field
      // IPv4: offset 6 has bdf_len; IPv6: offset 18 has bdf_len
      size_t ip_len;
      std::string ip_addr;
      if (metadata_vec.size() >= 7) {
        uint8_t candidate_bdf_len = metadata_vec[6];
        if (7 + candidate_bdf_len == metadata_vec.size()) {
          // IPv4 format
          ip_len = 4;
          ip_addr = std::to_string(metadata_vec[0]) + "." +
                    std::to_string(metadata_vec[1]) + "." +
                    std::to_string(metadata_vec[2]) + "." +
                    std::to_string(metadata_vec[3]);
        } else if (metadata_vec.size() >= 19) {
          uint8_t candidate_bdf_len6 = metadata_vec[18];
          if (19 + candidate_bdf_len6 == metadata_vec.size()) {
            ip_len = 16;
            char ip6_str[INET6_ADDRSTRLEN];
            struct in6_addr ip6_a;
            std::memcpy(&ip6_a, metadata_vec.data(), 16);
            inet_ntop(AF_INET6, &ip6_a, ip6_str, sizeof(ip6_str));
            ip_addr = "[" + std::string(ip6_str) + "]";
          } else {
            ip_len = 0;
          }
        } else {
          ip_len = 0;
        }
      } else {
        ip_len = 0;
      }
      if (ip_len > 0) {
        uint16_t port = (metadata_vec[ip_len] << 8) | metadata_vec[ip_len + 1];
        uint8_t bdf_len = metadata_vec[ip_len + 2];
        std::string bdf(metadata_vec.begin() + ip_len + 3,
                        metadata_vec.begin() + ip_len + 3 + bdf_len);
        result = ip_addr + ":" + std::to_string(port) + "?" + bdf;
      } else {
        // Fallback: hex
        for (size_t i = 0; i < metadata_vec.size(); ++i) {
          char hex[3];
          snprintf(hex, sizeof(hex), "%02x", metadata_vec[i]);
          result += hex;
        }
      }
    } else {
      result = "";
      for (size_t i = 0; i < metadata_vec.size(); ++i) {
        char hex[3];
        snprintf(hex, sizeof(hex), "%02x", metadata_vec[i]);
        result += hex;
      }
    }

    *metadata = new char[result.length() + 1];
    std::strcpy(*metadata, result.c_str());

    return 0;
  } catch (...) {
    return -1;
  }
}

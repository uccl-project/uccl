#include "uccl_engine.h"
#ifdef UCCL_P2P_USE_TCPX
#include "nccl_tcpx_endpoint.h"
#elif defined(UCCL_P2P_USE_NCCL)
#include "engine.h"
#include "nccl/nccl_endpoint.h"
#else
#include "endpoint_wrapper.h"
#include "engine.h"
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

#ifdef UCCL_P2P_USE_TCPX
// nccl_tcpx_endpoint does not declare inside_python; define it here for
// uccl_engine.
thread_local bool inside_python = false;

using Endpoint = nccl_tcpx::Endpoint;
#else
using Endpoint = ::Endpoint;
#endif

struct uccl_engine {
  std::unique_ptr<Endpoint> endpoint;
};

struct uccl_conn {
  uint64_t conn_id;
  uccl_engine* engine;
  int sock_fd;               // Keep for backward compatibility
  std::string oob_conn_key;  // For epoll-based notifications
  std::thread* listener_thread;
  bool listener_running;
  std::mutex listener_mutex;
};

typedef struct {
  FifoItem fifo_item;
  bool is_valid;
} fifo_item_t;

typedef struct {
  uint64_t mr_id;
  size_t size;
} mem_reg_entry_t;

std::unordered_map<uintptr_t, mem_reg_entry_t> mem_reg_info;

std::vector<notify_msg_t> notify_msg_list;
std::mutex notify_msg_list_mutex;

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

// Helper function for the listener thread
void listener_thread_func(uccl_conn_t* conn) {
  std::cout << "Listener thread: Waiting for notifs" << std::endl;

  while (conn->listener_running) {
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(conn->sock_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
               sizeof(timeout));

    md_t md;
    ssize_t total_received = 0;
    ssize_t recv_size = 0;
    char* buffer = reinterpret_cast<char*>(&md);

    while (total_received < sizeof(md_t)) {
      recv_size = recv(conn->sock_fd, buffer + total_received,
                       sizeof(md_t) - total_received, 0);

      if (recv_size <= 0) {
        if (!conn->listener_running) {
          return;
        }
        break;
      }
      total_received += recv_size;
    }

    if (total_received != sizeof(md_t)) {
      if (!conn->listener_running) {
        return;
      }
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(notify_msg_list_mutex);
      notify_msg_t notify_msg = {};
      strncpy(notify_msg.name, md.notify_data.name,
              sizeof(notify_msg.name) - 1);
      memcpy(notify_msg.msg, md.notify_data.msg, sizeof(notify_msg.msg));
      notify_msg_list.push_back(notify_msg);
    }
  }
}

uccl_engine_t* uccl_engine_create(int num_cpus, bool in_python) {
  inside_python = in_python;
  uccl_engine_t* eng = new uccl_engine;
  eng->endpoint = std::unique_ptr<Endpoint>(new Endpoint(num_cpus));
  return eng;
}

void uccl_engine_destroy(uccl_engine_t* engine) {
  if (engine) {
    engine->endpoint.reset();
    delete engine;
  }
}

uccl_conn_t* uccl_engine_connect(uccl_engine_t* engine, char const* ip_addr,
                                 int remote_gpu_idx, int remote_port) {
  if (!engine || !ip_addr) return nullptr;
  uccl_conn_t* conn = new uccl_conn;
  uint64_t conn_id;
  bool ok = engine->endpoint->connect(std::string(ip_addr), remote_gpu_idx,
                                      remote_port, conn_id);
  if (!ok) {
    delete conn;
    return nullptr;
  }
  conn->conn_id = conn_id;
  conn->sock_fd = engine->endpoint->get_sock_fd(conn_id);
#if !defined(UCCL_P2P_USE_TCPX) && !defined(UCCL_P2P_USE_NCCL)
  conn->oob_conn_key = engine->endpoint->get_oob_conn_key(conn_id);
#endif
  conn->engine = engine;
  conn->listener_thread = nullptr;
  conn->listener_running = false;
  return conn;
}

uccl_conn_t* uccl_engine_accept(uccl_engine_t* engine, char* ip_addr_buf,
                                size_t ip_addr_buf_len, int* remote_gpu_idx) {
  if (!engine || !ip_addr_buf || !remote_gpu_idx) return nullptr;
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
#if !defined(UCCL_P2P_USE_TCPX) && !defined(UCCL_P2P_USE_NCCL)
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
  mem_reg_info[data] = entry;
  return 0;
}

int uccl_engine_read(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                     size_t size, FifoItem fifo_item, uint64_t* transfer_id) {
  if (!conn || !data) return -1;

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
                            uint64_t* transfer_id) {
  if (!conn || num_iovs <= 0) return -1;

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

#ifdef UCCL_P2P_USE_TCPX
  return -1;  // TODO: support write_rc for TCPX
#else

  return conn->engine->endpoint->write_async(conn->conn_id, mr,
                                             const_cast<void*>(data), size,
                                             fifo_item, transfer_id)
             ? 0
             : -1;
#endif
}

int uccl_engine_write_vector(uccl_conn_t* conn, std::vector<uccl_mr_t> mr_ids,
                             std::vector<void*> dst_v,
                             std::vector<size_t> size_v,
                             std::vector<FifoItem> fifo_items, int num_iovs,
                             uint64_t* transfer_id) {
  if (!conn || num_iovs <= 0) return -1;

#ifdef UCCL_P2P_USE_TCPX
  return -1;  // TODO: support write_rc for TCPX
#else
  return conn->engine->endpoint->writev_async(conn->conn_id, mr_ids, dst_v,
                                              size_v, fifo_items, num_iovs,
                                              transfer_id)
             ? 0
             : -1;
#endif
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
#if defined(UCCL_P2P_USE_TCPX)
  // TCPX uses a separate listener thread
  conn->listener_thread = new std::thread(listener_thread_func, conn);
#elif defined(UCCL_P2P_USE_NCCL)
  // NCCL handles notifications in the control thread, no separate listener
  // needed
  conn->listener_thread = nullptr;
#endif

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
#if defined(UCCL_P2P_USE_TCPX)
  std::lock_guard<std::mutex> lock(notify_msg_list_mutex);
  std::vector<notify_msg_t> result = std::move(notify_msg_list);
  notify_msg_list.clear();
  return result;
#else
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
#endif
}

int uccl_engine_send_notif(uccl_conn_t* conn, notify_msg_t* notify_msg) {
  if (!conn || !notify_msg) return -1;

#if defined(UCCL_P2P_USE_TCPX)
  md_t md;
  md.notify_data = *notify_msg;

  return send(conn->sock_fd, &md, sizeof(md_t), 0);
#elif defined(UCCL_P2P_USE_NCCL)
  NotifyMsg oob_msg;
  oob_msg.magic = NOTIFY_MSG_MAGIC;
  strncpy(oob_msg.name, notify_msg->name, sizeof(oob_msg.name) - 1);
  oob_msg.name[sizeof(oob_msg.name) - 1] = '\0';
  memcpy(oob_msg.msg, notify_msg->msg, sizeof(oob_msg.msg));

  auto tcp_endpoint = conn->engine->endpoint->get_endpoint();
  if (!tcp_endpoint) {
    LOG(ERROR) << "Failed to get TCP endpoint for notification";
    return -1;
  }

  uint64_t flow_id = conn->conn_id;
  return tcp_endpoint->send_notification(flow_id, oob_msg);
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

  NotifyMsg oob_msg;
  oob_msg.magic = NOTIFY_MSG_MAGIC;
  strncpy(oob_msg.name, notify_msg->name, sizeof(oob_msg.name) - 1);
  oob_msg.name[sizeof(oob_msg.name) - 1] = '\0';
  memcpy(oob_msg.msg, notify_msg->msg, sizeof(oob_msg.msg));

  std::string payload(reinterpret_cast<char*>(&oob_msg), sizeof(NotifyMsg));
  bool ok = oob_client->send_meta(conn->oob_conn_key, payload);

  return ok ? sizeof(NotifyMsg) : -1;
#endif
}

int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata) {
  if (!engine || !metadata) return -1;

  try {
    std::vector<uint8_t> metadata_vec =
        engine->endpoint->get_unified_metadata();

    std::string result;
    if (metadata_vec.size() == 10) {  // IPv4 format
      std::string ip_addr = std::to_string(metadata_vec[0]) + "." +
                            std::to_string(metadata_vec[1]) + "." +
                            std::to_string(metadata_vec[2]) + "." +
                            std::to_string(metadata_vec[3]);
      uint16_t port = (metadata_vec[4] << 8) | metadata_vec[5];
      int gpu_idx;
      std::memcpy(&gpu_idx, &metadata_vec[6], sizeof(int));

      result = "" + ip_addr + ":" + std::to_string(port) + "?" +
               std::to_string(gpu_idx);
    } else if (metadata_vec.size() == 22) {  // IPv6 format
      char ip6_str[INET6_ADDRSTRLEN];
      struct in6_addr ip6_addr;
      std::memcpy(&ip6_addr, metadata_vec.data(), 16);
      inet_ntop(AF_INET6, &ip6_addr, ip6_str, sizeof(ip6_str));

      uint16_t port = (metadata_vec[16] << 8) | metadata_vec[17];
      int gpu_idx;
      std::memcpy(&gpu_idx, &metadata_vec[18], sizeof(int));

      result = "" + std::string(ip6_str) + "]:" + std::to_string(port) + "?" +
               std::to_string(gpu_idx);
    } else {  // Fallback: return hex representation
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
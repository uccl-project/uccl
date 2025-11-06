#include "uccl_engine.h"
#ifdef USE_TCPX
#include "tcpx_engine.h"
#else
#include "engine.h"
#endif
#include "util/util.h"
#include <arpa/inet.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdbool>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef USE_TCPX
using FifoItem = tcpx::FifoItem;
using Endpoint = tcpx::Endpoint;
#else
using FifoItem = uccl::FifoItem;
using Endpoint = Endpoint;
#endif

struct uccl_engine {
  Endpoint* endpoint;
};

struct uccl_conn {
  uint64_t conn_id;
  uccl_engine* engine;
  int sock_fd;
  std::thread* listener_thread;
  bool listener_running;
  std::mutex listener_mutex;
};

struct uccl_mr {
  uint64_t mr_id;
  uccl_engine* engine;
};

typedef struct {
  FifoItem fifo_item;
  bool is_valid;
} fifo_item_t;

std::unordered_map<uintptr_t, uint64_t> mem_reg_info;
std::unordered_map<uccl_conn_t*, fifo_item_t*> fifo_item_map;
std::vector<notify_msg_t> notify_msg_list;

std::mutex fifo_item_map_mutex;
std::mutex notify_msg_list_mutex;

// Helper function for the listener thread
void listener_thread_func(uccl_conn_t* conn) {
  std::cout << "Listener thread: Waiting for metadata." << std::endl;

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

    uint64_t mr_id = 0;
    switch (md.op) {
      case UCCL_READ: {
        tx_msg_t tx_data = md.data.tx_data;
#ifdef USE_TCPX
        if (!conn->engine->endpoint->find_mr_by_addr(
                tx_data.data_ptr, tx_data.data_size, &mr_id)) {
          std::cerr << "Local memory not registered for address: "
                    << tx_data.data_ptr << std::endl;
          break;
        }
#else
        auto local_mem_iter = mem_reg_info.find(tx_data.data_ptr);
        if (local_mem_iter == mem_reg_info.end()) {
          std::cerr << "Local memory not registered for address: "
                    << tx_data.data_ptr << std::endl;
          break;
        }
        mr_id = local_mem_iter->second;
#endif

        char out_buf[sizeof(FifoItem)];
        conn->engine->endpoint->advertise(conn->conn_id, mr_id,
                                          (void*)tx_data.data_ptr,
                                          tx_data.data_size, out_buf);

        md_t response_md;
        response_md.op = UCCL_FIFO;
        fifo_msg_t fifo_data;
        memcpy(fifo_data.fifo_buf, out_buf, sizeof(FifoItem));
        response_md.data.fifo_data = fifo_data;

        ssize_t result = send(conn->sock_fd, &response_md, sizeof(md_t), 0);
        if (result < 0) {
          std::cerr << "Failed to send FifoItem data: " << strerror(errno)
                    << std::endl;
        }

#ifdef USE_TCPX
        FifoItem fifo_item;
        memcpy(&fifo_item, out_buf, sizeof(FifoItem));
        // Immediately push the data over TCPX so the passive reader only needs
        // the FIFO metadata to complete its read.
        if (!conn->engine->endpoint->queue_read_response(conn->conn_id,
                                                         fifo_item)) {
          std::cerr << "Failed to queue read response" << std::endl;
        }
#endif
        break;
      }
      case UCCL_WRITE: {
        tx_msg_t tx_data = md.data.tx_data;
#ifdef USE_TCPX
        if (!conn->engine->endpoint->find_mr_by_addr(
                tx_data.data_ptr, tx_data.data_size, &mr_id)) {
          std::cerr << "Local memory not registered for address: "
                    << tx_data.data_ptr << std::endl;
          break;
        }
#else
        auto local_mem_iter = mem_reg_info.find(tx_data.data_ptr);
        if (local_mem_iter == mem_reg_info.end()) {
          std::cerr << "Local memory not registered for address: "
                    << tx_data.data_ptr << std::endl;
          break;
        }
        mr_id = local_mem_iter->second;
#endif

        uccl_mr_t temp_mr;
        temp_mr.mr_id = mr_id;
        temp_mr.engine = conn->engine;
        int result = uccl_engine_recv(conn, &temp_mr, (void*)tx_data.data_ptr,
                                      tx_data.data_size);
        if (result < 0) {
          std::cerr << "Failed to perform uccl_engine_recv" << std::endl;
        }
        break;
      }
      case UCCL_VECTOR_READ: {
        size_t count = md.data.vector_data.count;

        tx_msg_t* tx_data_array = new tx_msg_t[count];
        ssize_t data_size = count * sizeof(tx_msg_t);

        ssize_t total_received = 0;
        ssize_t recv_data_size = 0;
        char* buffer = reinterpret_cast<char*>(tx_data_array);

        while (total_received < data_size) {
          recv_data_size = recv(conn->sock_fd, buffer + total_received,
                                data_size - total_received, 0);

          if (recv_data_size <= 0) {
            std::cerr << "Failed to receive tx_data array. Expected: "
                      << data_size << ", Received: " << total_received
                      << ", Last recv: " << recv_data_size << std::endl;
            delete[] tx_data_array;
            break;
          }

          total_received += recv_data_size;
        }

        if (total_received != data_size) {
          delete[] tx_data_array;
          break;
        }
        for (size_t i = 0; i < count; i++) {
          tx_msg_t tx_data = tx_data_array[i];
#ifdef USE_TCPX
          if (!conn->engine->endpoint->find_mr_by_addr(
                  tx_data.data_ptr, tx_data.data_size, &mr_id)) {
            std::cerr << "Local memory not registered for address: "
                      << tx_data.data_ptr << " (item " << i << ")" << std::endl;
            continue;
          }
#else
          auto local_mem_iter = mem_reg_info.find(tx_data.data_ptr);
          if (local_mem_iter == mem_reg_info.end()) {
            std::cerr << "Local memory not registered for address: "
                      << tx_data.data_ptr << " (item " << i << ")" << std::endl;
            continue;
          }
          mr_id = local_mem_iter->second;
#endif

          char out_buf[sizeof(FifoItem)];
          conn->engine->endpoint->advertise(conn->conn_id, mr_id,
                                            (void*)tx_data.data_ptr,
                                            tx_data.data_size, out_buf);

          md_t response_md;
          response_md.op = UCCL_FIFO;
          fifo_msg_t fifo_data;
          fifo_data.id = i;
          memcpy(fifo_data.fifo_buf, out_buf, sizeof(FifoItem));
          response_md.data.fifo_data = fifo_data;

          // TODO: Check if this has to sent in bulk or individually
          ssize_t result = send(conn->sock_fd, &response_md, sizeof(md_t), 0);
          if (result < 0) {
            std::cerr << "Failed to send FifoItem data for item " << i << ": "
                      << strerror(errno) << std::endl;
          }

          FifoItem fifo_item;
          memcpy(&fifo_item, out_buf, sizeof(FifoItem));
          // Each advertised slice triggers a corresponding send so the remote
          // side can simply post tagged receives.
#ifdef USE_TCPX
          if (!conn->engine->endpoint->queue_read_response(conn->conn_id,
                                                           fifo_item)) {
            std::cerr << "Failed to queue read response for item " << i
                      << std::endl;
          }
#endif
        }

        delete[] tx_data_array;
        break;
      }
      case UCCL_VECTOR_WRITE: {
        size_t count = md.data.vector_data.count;
        tx_msg_t* tx_data_array = new tx_msg_t[count];
        ssize_t data_size = count * sizeof(tx_msg_t);

        ssize_t total_received = 0;
        ssize_t recv_data_size = 0;
        char* buffer = reinterpret_cast<char*>(tx_data_array);

        while (total_received < data_size) {
          recv_data_size = recv(conn->sock_fd, buffer + total_received,
                                data_size - total_received, 0);

          if (recv_data_size <= 0) {
            std::cerr << "Failed to receive tx_data array. Expected: "
                      << data_size << ", Received: " << total_received
                      << ", Last recv: " << recv_data_size << std::endl;
            delete[] tx_data_array;
            break;
          }

          total_received += recv_data_size;
        }

        if (total_received != data_size) {
          delete[] tx_data_array;
          break;
        }

        for (size_t i = 0; i < count; i++) {
          tx_msg_t tx_data = tx_data_array[i];
#ifdef USE_TCPX
          if (!conn->engine->endpoint->find_mr_by_addr(
                  tx_data.data_ptr, tx_data.data_size, &mr_id)) {
            std::cerr << "Local memory not registered for address: "
                      << tx_data.data_ptr << " (item " << i << ")" << std::endl;
            continue;
          }
#else
          auto local_mem_iter = mem_reg_info.find(tx_data.data_ptr);
          if (local_mem_iter == mem_reg_info.end()) {
            std::cerr << "Local memory not registered for address: "
                      << tx_data.data_ptr << " (item " << i << ")" << std::endl;
            continue;
          }
          mr_id = local_mem_iter->second;
#endif

          uccl_mr_t temp_mr;
          temp_mr.mr_id = mr_id;
          temp_mr.engine = conn->engine;
          int result = uccl_engine_recv(conn, &temp_mr, (void*)tx_data.data_ptr,
                                        tx_data.data_size);
          if (result < 0) {
            std::cerr << "Failed to perform uccl_engine_recv for item " << i
                      << std::endl;
          }
        }

        delete[] tx_data_array;
        break;
      }
      case UCCL_FIFO: {
        fifo_msg_t fifo_data = md.data.fifo_data;
        FifoItem fifo_item;
        memcpy(&fifo_item, fifo_data.fifo_buf, sizeof(FifoItem));
        fifo_item_t* f_item = new fifo_item_t;
        f_item->fifo_item = fifo_item;
        f_item->is_valid = true;

        {
          std::lock_guard<std::mutex> lock(fifo_item_map_mutex);
          fifo_item_map[conn + fifo_data.id] = f_item;
        }
        break;
      }
      case UCCL_NOTIFY: {
        std::lock_guard<std::mutex> lock(notify_msg_list_mutex);
        notify_msg_t notify_msg = {};
        strncpy(notify_msg.name, md.data.notify_data.name,
                sizeof(notify_msg.name) - 1);
        strncpy(notify_msg.msg, md.data.notify_data.msg,
                sizeof(notify_msg.msg) - 1);
        notify_msg_list.push_back(notify_msg);
        break;
      }
      default:
        std::cerr << "Invalid operation type: " << md.op << std::endl;
        continue;
    }
  }
}

uccl_engine_t* uccl_engine_create(int num_cpus, bool in_python) {
  inside_python = in_python;
  uccl_engine_t* eng = new uccl_engine;
  eng->endpoint = new Endpoint(num_cpus);
  return eng;
}

void uccl_engine_destroy(uccl_engine_t* engine) {
  if (engine) {
    delete engine->endpoint;
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
  conn->engine = engine;
  conn->listener_thread = nullptr;
  conn->listener_running = false;
  return conn;
}

uccl_mr_t* uccl_engine_reg(uccl_engine_t* engine, uintptr_t data, size_t size) {
  if (!engine || !data) return nullptr;
  uccl_mr_t* mr = new uccl_mr;
  uint64_t mr_id;
  bool ok = engine->endpoint->reg((void*)data, size, mr_id);
  if (!ok) {
    delete mr;
    return nullptr;
  }
  mr->mr_id = mr_id;
  mr->engine = engine;
  mem_reg_info[data] = mr_id;
  return mr;
}

int uccl_engine_read(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                     size_t size, void* slot_item_ptr, uint64_t* transfer_id) {
  if (!conn || !mr || !data) return -1;

  FifoItem slot_item;
  slot_item = *static_cast<FifoItem*>(slot_item_ptr);

  return conn->engine->endpoint->read_async(conn->conn_id, mr->mr_id,
                                            const_cast<void*>(data), size,
                                            slot_item, transfer_id)
             ? 0
             : -1;
}

int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                      size_t size, uint64_t* transfer_id) {
  if (!conn || !mr || !data) return -1;
  return conn->engine->endpoint->send_async(conn->conn_id, mr->mr_id, data,
                                            size, transfer_id)
             ? 0
             : -1;
}

int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t* mr, void* data,
                     size_t data_size) {
  if (!conn || !mr || !data) return -1;

  uint64_t transfer_id = 0;
  if (!conn->engine->endpoint->recv_async(conn->conn_id, mr->mr_id, data,
                                          data_size, &transfer_id)) {
    return -1;
  }

  // Poll until the transfer completes
  // This drives the stage 1 (network -> bounce buffer) and stage 2 (unpack kernel) pipelines
  bool done = false;
  while (!done) {
    if (!conn->engine->endpoint->poll_async(transfer_id, &done)) {
      return -1;
    }
    if (!done) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  return 0;
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

  conn->listener_thread = new std::thread(listener_thread_func, conn);

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

void uccl_engine_conn_destroy(uccl_conn_t* conn) {
  if (conn) {
    uccl_engine_stop_listener(conn);
    {
      std::lock_guard<std::mutex> lock(fifo_item_map_mutex);
      auto it = fifo_item_map.find(conn);
      if (it != fifo_item_map.end()) {
        delete it->second;
        fifo_item_map.erase(it);
      }
    }
    delete conn;
  }
}

void uccl_engine_mr_destroy(uccl_mr_t* mr) {
  for (auto it = mem_reg_info.begin(); it != mem_reg_info.end(); ++it) {
    if (it->second == mr->mr_id) {
      mem_reg_info.erase(it);
      break;
    }
  }
  delete mr;
}

int uccl_engine_send_tx_md(uccl_conn_t* conn, md_t* md) {
  if (!conn || !md) return -1;

  return send(conn->sock_fd, md, sizeof(md_t), 0);
}

int uccl_engine_send_tx_md_vector(uccl_conn_t* conn, md_t* md_array,
                                  size_t count) {
  if (!conn || !md_array || count == 0) return -1;

  // Determine the operation type based on the first item
  uccl_msg_type op_type =
      (md_array[0].op == UCCL_READ) ? UCCL_VECTOR_READ : UCCL_VECTOR_WRITE;

  md_t vector_md;
  vector_md.op = op_type;
  vector_md.data.vector_data.count = count;

  ssize_t bytes_sent = send(conn->sock_fd, &vector_md, sizeof(md_t), 0);
  if (bytes_sent != sizeof(md_t)) {
    std::cerr << "Failed to send vector metadata header. Expected: "
              << sizeof(md_t) << ", Sent: " << bytes_sent << std::endl;
    return -1;
  }

  tx_msg_t* tx_data_array = new tx_msg_t[count];
  for (size_t i = 0; i < count; i++) {
    tx_data_array[i] = md_array[i].data.tx_data;
  }

  ssize_t data_size = count * sizeof(tx_msg_t);
  bytes_sent = send(conn->sock_fd, tx_data_array, data_size, 0);

  delete[] tx_data_array;

  if (bytes_sent != data_size) {
    std::cerr << "Failed to send vector tx_data array. Expected: " << data_size
              << ", Sent: " << bytes_sent << std::endl;
    return -1;
  }

  return sizeof(md_t) + data_size;
}

std::vector<notify_msg_t> uccl_engine_get_notifs() {
  std::lock_guard<std::mutex> lock(notify_msg_list_mutex);

  std::vector<notify_msg_t> result;
  result = std::move(notify_msg_list);

  notify_msg_list.clear();

  return result;
}

int uccl_engine_send_notif(uccl_conn_t* conn, notify_msg_t* notify_msg) {
  if (!conn || !notify_msg) return -1;
  md_t md;
  md.op = UCCL_NOTIFY;
  md.data.notify_data = *notify_msg;

  return send(conn->sock_fd, &md, sizeof(md_t), 0);
}

int uccl_engine_get_fifo_item(uccl_conn_t* conn, int id, void* fifo_item) {
  if (!conn || !fifo_item) return -1;

  std::lock_guard<std::mutex> lock(fifo_item_map_mutex);
  auto it = fifo_item_map.find(conn + id);
  if (it == fifo_item_map.end()) {
    return -1;
  }
  if (it->second->is_valid) {
    memcpy(fifo_item, &it->second->fifo_item, sizeof(FifoItem));
    it->second->is_valid = false;
    return 0;
  } else {
    return -1;
  }
}

int uccl_engine_get_sock_fd(uccl_conn_t* conn) {
  if (!conn) return -1;
  return conn->sock_fd;
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

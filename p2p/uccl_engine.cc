#include "uccl_engine.h"
#include "engine.h"
#include <arpa/inet.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdbool>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Threadpool to manage the receiver side
class ThreadPool {
 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop_flag;

 public:
  ThreadPool(size_t num_threads) : stop_flag(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(lock, [this] {
              return this->stop_flag || !this->tasks.empty();
            });

            if (this->stop_flag && this->tasks.empty()) return;

            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop_flag = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
      if (worker.joinable()) worker.join();
    }
  }

  // Force destroy without waiting for threads to finish
  void force_destroy() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop_flag = true;
    }
    condition.notify_all();

    // Detach all threads instead of joining
    for (std::thread& worker : workers) {
      if (worker.joinable()) {
        worker.detach();
      }
    }
  }

  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args) {
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop_flag.load()) return;
      tasks.emplace(std::move(task));
    }
    condition.notify_one();
  }
};

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
  ThreadPool* recv_thread_pool;
};

struct uccl_mr {
  uint64_t mr_id;
  uccl_engine* engine;
};

typedef struct {
  uccl::FifoItem fifo_item;
  bool is_valid;
} fifo_item_t;

std::unordered_map<uintptr_t, uint64_t> mem_reg_info;
std::unordered_map<uccl_conn_t*, fifo_item_t*> fifo_item_map;
std::mutex fifo_item_map_mutex;

// Forward declaration
void listener_thread_func(uccl_conn_t* conn);

// Simple async receive function
void async_recv_worker(uccl_conn_t* conn, uint64_t mr_id, void* data,
                       size_t data_size) {
  size_t recv_size = 0;
  uccl_mr_t temp_mr;
  temp_mr.mr_id = mr_id;
  temp_mr.engine = conn->engine;

  int result = uccl_engine_recv(conn, &temp_mr, data, data_size);

  if (result == 0) {
    // Notify something in conn
  }
}

uccl_engine_t* uccl_engine_create(int local_gpu_idx, int num_cpus) {
  uccl_engine_t* eng = new uccl_engine;
  eng->endpoint = new Endpoint(local_gpu_idx, num_cpus);
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
  conn->recv_thread_pool =
      new ThreadPool(4);  // Create thread pool with 4 threads
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
  conn->recv_thread_pool =
      new ThreadPool(4);  // Create thread pool with 4 threads
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

  uccl::FifoItem slot_item;
  slot_item = *static_cast<uccl::FifoItem*>(slot_item_ptr);

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
  uint64_t transfer_id;
  return conn->engine->endpoint->recv_async(conn->conn_id, mr->mr_id, data,
                                            data_size, &transfer_id)
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
    return -1;  // Listener already running
  }

  conn->listener_running = true;

  // Start the listener thread
  conn->listener_thread = new std::thread(listener_thread_func, conn);

  return 0;
}

int uccl_engine_stop_listener(uccl_conn_t* conn) {
  if (!conn) return -1;

  std::lock_guard<std::mutex> lock(conn->listener_mutex);

  if (!conn->listener_running) {
    return 0;  // Listener not running
  }

  // Signal the thread to stop
  conn->listener_running = false;

  // Close the socket to unblock the recv() call
  if (conn->sock_fd >= 0) {
    close(conn->sock_fd);
    conn->sock_fd = -1;
  }

  // Wait for the thread to finish with a timeout
  if (conn->listener_thread && conn->listener_thread->joinable()) {
    // Try to join with a timeout
    auto future = std::async(std::launch::async,
                             [conn]() { conn->listener_thread->join(); });

    if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
      // Thread is stuck, detach it and let it die naturally
      std::cout << "Warning: Listener thread not responding, detaching..."
                << std::endl;
      conn->listener_thread->detach();
    }
  }

  // Clean up
  delete conn->listener_thread;
  conn->listener_thread = nullptr;

  return 0;
}

void uccl_engine_conn_destroy(uccl_conn_t* conn) {
  if (conn) {
    uccl_engine_stop_listener(conn);
    if (conn->recv_thread_pool) {
      conn->recv_thread_pool->force_destroy();
      delete conn->recv_thread_pool;
    }
    // Clean up fifo_item for this connection
    {
      std::lock_guard<std::mutex> lock(fifo_item_map_mutex);
      auto it = fifo_item_map.find(conn);
      if (it != fifo_item_map.end()) {
        delete it->second;  // Free the fifo_item_t
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

// Helper function for the listener thread
void listener_thread_func(uccl_conn_t* conn) {
  const size_t buffer_size = 4096;  // 4KB buffer for receiving messages
  char* buffer = new char[buffer_size];
  std::cout << "Listener thread: Waiting for metadata." << std::endl;

  while (conn->listener_running) {
    // Set socket to non-blocking mode with timeout
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(conn->sock_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
               sizeof(timeout));

    metadata_t md;
    ssize_t recv_size = recv(conn->sock_fd, &md, sizeof(metadata_t), 0);
    if (recv_size != sizeof(metadata_t)) {
      if (!conn->listener_running) {
        break;
      }
      // Error receiving metadata or incomplete metadata, sleep briefly before
      // retrying
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    std::cout << "Received metadata: " << md.op << " " << md.data_ptr << " "
              << md.data_size << std::endl;
    auto local_mem_iter = mem_reg_info.find(md.data_ptr);
    if (local_mem_iter == mem_reg_info.end()) {
      std::cerr << "Local memory not registered for address: " << md.data_ptr
                << std::endl;
      continue;
    }
    std::cout << "Local memory registered for address: " << md.data_ptr
              << std::endl;
    auto mr_id = local_mem_iter->second;
    char out_buf[sizeof(uccl::FifoItem)];
    switch (md.op) {
      case UCCL_READ: {
        conn->engine->endpoint->advertise(
            conn->conn_id, mr_id, (void*)md.data_ptr, md.data_size, out_buf);
        ssize_t result =
            send(conn->sock_fd, out_buf, sizeof(uccl::FifoItem), 0);
        if (result < 0) {
          std::cerr << "Failed to send FifoItem data: " << strerror(errno)
                    << std::endl;
        }
        break;
      }
      case UCCL_WRITE:
        // Submit async receive task to thread pool
        conn->recv_thread_pool->enqueue(async_recv_worker, conn, mr_id,
                                        (void*)md.data_ptr, md.data_size);
        break;
      case UCCL_FIFO: {
        std::cout << "Received & Stored FIFO item" << std::endl;
        // Store fifo_item in the hashmap for this connection
        uccl::FifoItem fifo_item;
        memcpy(&fifo_item, md.fifo_buf, sizeof(uccl::FifoItem));
        fifo_item_t* f_item = new fifo_item_t;
        f_item->fifo_item = fifo_item;
        f_item->is_valid = true;

        {
          std::lock_guard<std::mutex> lock(fifo_item_map_mutex);
          fifo_item_map[conn] = f_item;
        }
        break;
      }
      default:
        std::cerr << "Invalid operation type: " << md.op << std::endl;
        continue;
    }
  }

  delete[] buffer;
}

int uccl_engine_get_sock_fd(uccl_conn_t* conn) {
  if (!conn) return -1;
  return conn->sock_fd;
}

int uccl_engine_get_fifo_item(uccl_conn_t* conn, void* fifo_item) {
  if (!conn || !fifo_item) return -1;

  std::lock_guard<std::mutex> lock(fifo_item_map_mutex);
  auto it = fifo_item_map.find(conn);
  if (it == fifo_item_map.end()) {  
    return -1;
  }
  if (it->second->is_valid) {
    memcpy(fifo_item, &it->second->fifo_item, sizeof(uccl::FifoItem));
    it->second->is_valid = false;
    return 0;
  } else {
    return -1;
  }
}

int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata) {
  if (!engine || !metadata) return -1;

  try {
    std::vector<uint8_t> metadata_vec =
        engine->endpoint->get_endpoint_metadata();

    // Build string representation
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
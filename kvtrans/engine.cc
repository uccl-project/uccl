#include "engine.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

Engine::Engine(std::string const& if_name, const uint32_t ncpus,
               const uint32_t nconn_per_cpu, const uint16_t listen_port)
    : if_name_(if_name),
      ncpus_(ncpus),
      nconn_per_cpu_(nconn_per_cpu),
      listen_port_(listen_port) {
  std::cout << "Creating Engine with interface: " << if_name
            << ", CPUs: " << ncpus << ", connections per CPU: " << nconn_per_cpu
            << ", listen port: " << listen_port << std::endl;

  // Initialize channels and threads
  channels_.resize(ncpus);
  engine_threads_.reserve(ncpus);

  // For this demo, we'll create a simplified version
  // In a real implementation, you'd initialize InfiniBand resources here
  for (uint32_t i = 0; i < ncpus; ++i) {
    // Initialize ring buffer for each CPU
    // jring_init(&channels_[i], 1024);  // Commented out as jring.h might not
    // be available

    // Create engine threads
    engine_threads_.emplace_back(&Engine::engine_thread, this);
  }

  std::cout << "Engine initialized successfully" << std::endl;
}

Engine::~Engine() {
  std::cout << "Destroying Engine..." << std::endl;

  // Clean up connections
  for (auto& [conn_id, conn] : conn_id_to_conn_) {
    if (conn) {
      // Clean up InfiniBand resources
      delete conn;
    }
  }

  // Wait for threads to finish
  for (auto& thread : engine_threads_) {
    thread.join();
  }

  std::cout << "Engine destroyed" << std::endl;
}

bool Engine::connect(std::string const& ip_addr, uint16_t const& port,
                     int& conn_id) {
  std::cout << "Attempting to connect to " << ip_addr << ":" << port
            << std::endl;

  // Create a new connection ID
  static int next_conn_id = 1;
  conn_id = next_conn_id++;

  // Create a new connection structure
  Conn* conn = new Conn();
  conn->conn_id_ = conn_id;
  conn->port_ = port;

  // In a real implementation, you would:
  // 1. Create TCP connection
  // 2. Exchange RDMA connection parameters
  // 3. Create InfiniBand QP, CQ, etc.

  // For demo purposes, simulate successful connection
  conn_id_to_conn_[conn_id] = conn;

  std::cout << "Connected successfully with conn_id: " << conn_id << std::endl;
  return true;
}

bool Engine::accept(std::string& ip_addr, uint16_t& port, int& conn_id) {
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // In a real implementation, you would:
  // 1. Accept TCP connection
  // 2. Extract client IP and port
  // 3. Exchange RDMA parameters
  // 4. Create InfiniBand resources

  // For demo purposes, simulate accepted connection
  static int next_conn_id = 1000;
  conn_id = next_conn_id++;
  ip_addr = "127.0.0.1";
  port = 12345;

  Conn* conn = new Conn();
  conn->conn_id_ = conn_id;
  conn->port_ = port;

  conn_id_to_conn_[conn_id] = conn;

  std::cout << "Accepted connection from " << ip_addr << ":" << port
            << " with conn_id: " << conn_id << std::endl;
  return true;
}

bool Engine::reg_kv(int conn_id, void const* data, size_t size,
                    uint64_t& kv_id) {
  std::cout << "Registering KV for conn_id: " << conn_id << ", size: " << size
            << " bytes" << std::endl;

  auto conn_it = conn_id_to_conn_.find(conn_id);
  if (conn_it == conn_id_to_conn_.end()) {
    std::cerr << "Connection ID " << conn_id << " not found" << std::endl;
    return false;
  }

  Conn* conn = conn_it->second;

  // Create memory region
  MR* mr = new MR();
  static int next_mr_id = 1;
  mr->mr_id_ = next_mr_id++;

  // In a real implementation, you would register memory with InfiniBand
  // mr->mr_ = ibv_reg_mr(conn->pd_, (void*)data, size, IBV_ACCESS_LOCAL_WRITE |
  // IBV_ACCESS_REMOTE_WRITE);

  // Generate KV ID
  static uint64_t next_kv_id = 1;
  kv_id = next_kv_id++;

  // Store the mapping
  kv_id_to_conn_and_mr_[kv_id] = std::make_tuple(conn, mr);

  std::cout << "KV registered with kv_id: " << kv_id << std::endl;
  return true;
}

bool Engine::send_kv(uint64_t kv_id, void const* data, size_t size) {
  std::cout << "Sending KV with kv_id: " << kv_id << ", size: " << size
            << " bytes" << std::endl;

  auto kv_it = kv_id_to_conn_and_mr_.find(kv_id);
  if (kv_it == kv_id_to_conn_and_mr_.end()) {
    std::cerr << "KV ID " << kv_id << " not found" << std::endl;
    return false;
  }

  auto [conn, mr] = kv_it->second;
  (void)conn;  // Suppress unused variable warning
  (void)mr;    // Suppress unused variable warning

  // In a real implementation, you would:
  // 1. Post RDMA write operation using conn and mr
  // 2. Wait for completion

  // For demo purposes, just acknowledge the data (no actual copy needed)
  (void)data;  // Suppress unused parameter warning

  std::cout << "KV sent successfully" << std::endl;
  return true;
}

bool Engine::recv_kv(uint64_t kv_id, void* data, size_t& size) {
  std::cout << "Receiving KV with kv_id: " << kv_id << std::endl;

  auto kv_it = kv_id_to_conn_and_mr_.find(kv_id);
  if (kv_it == kv_id_to_conn_and_mr_.end()) {
    std::cerr << "KV ID " << kv_id << " not found" << std::endl;
    return false;
  }

  auto [conn, mr] = kv_it->second;
  (void)conn;  // Suppress unused variable warning
  (void)mr;    // Suppress unused variable warning

  // In a real implementation, you would:
  // 1. Post RDMA read operation using conn and mr
  // 2. Wait for completion
  // 3. Copy data to output buffer

  // For demo purposes, simulate received data
  char const* demo_data = "Hello from remote KV store!";
  size_t demo_size = strlen(demo_data);

  if (size < demo_size) {
    size = demo_size;
    return false;
  }

  std::memcpy(data, demo_data, demo_size);
  size = demo_size;

  std::cout << "KV received successfully, size: " << size << " bytes"
            << std::endl;
  return true;
}

void* Engine::engine_thread(void* arg) {
  Engine* engine = static_cast<Engine*>(arg);
  (void)engine;  // Suppress unused variable warning

  std::cout << "Engine thread started" << std::endl;

  // In a real implementation, this thread would use engine to:
  // 1. Poll completion queues
  // 2. Handle RDMA operations
  // 3. Manage connection state

  // For demo purposes, just sleep
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    // Check for shutdown signal here
    break;  // For demo, exit immediately
  }

  std::cout << "Engine thread finished" << std::endl;
  return nullptr;
}
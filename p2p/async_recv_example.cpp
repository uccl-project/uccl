#include "uccl_engine.h"
#include <chrono>
#include <iostream>
#include <thread>

// Callback function for asynchronous receive operations
void recv_callback(void* data, size_t size, void* user_data) {
  std::cout << "Received " << size << " bytes asynchronously" << std::endl;

  // Process the received data here
  char* buffer = static_cast<char*>(data);
  std::cout << "Data: ";
  for (size_t i = 0; i < std::min(size, size_t(100)); ++i) {
    std::cout << buffer[i];
  }
  if (size > 100) {
    std::cout << "...";
  }
  std::cout << std::endl;

  // Signal completion if needed
  if (user_data) {
    bool* completed = static_cast<bool*>(user_data);
    *completed = true;
  }
}

int main() {
  // Create engine
  uccl_engine_t* engine = uccl_engine_create(0, 4);
  if (!engine) {
    std::cerr << "Failed to create engine" << std::endl;
    return -1;
  }

  // Create connection (example for server side)
  char ip_addr[256];
  int remote_gpu_idx;
  uccl_conn_t* conn =
      uccl_engine_accept(engine, ip_addr, sizeof(ip_addr), &remote_gpu_idx);
  if (!conn) {
    std::cerr << "Failed to accept connection" << std::endl;
    uccl_engine_destroy(engine);
    return -1;
  }

  std::cout << "Accepted connection from " << ip_addr
            << " (GPU: " << remote_gpu_idx << ")" << std::endl;

  // Set callback for asynchronous receive
  bool received = false;
  uccl_engine_set_recv_callback(conn, recv_callback, &received);

  // Start listener thread
  if (uccl_engine_start_listener(conn) != 0) {
    std::cerr << "Failed to start listener" << std::endl;
    uccl_engine_conn_destroy(conn);
    uccl_engine_destroy(engine);
    return -1;
  }

  std::cout << "Listener started. Waiting for data..." << std::endl;

  // Wait for some time to receive data
  for (int i = 0; i < 30; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (received) {
      std::cout << "Data received successfully!" << std::endl;
      break;
    }
    std::cout << "Waiting... (" << (i + 1) << "/30)" << std::endl;
  }

  // Cleanup
  uccl_engine_conn_destroy(conn);
  uccl_engine_destroy(engine);

  std::cout << "Example completed" << std::endl;
  return 0;
}
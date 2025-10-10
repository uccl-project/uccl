/**
 * @file test_modules.cc
 * @brief Unit tests for multi-channel modules
 *
 * Tests:
 * 1. SlidingWindow basic operations
 * 2. Bootstrap protocol (requires 2 processes)
 * 3. ChannelManager initialization
 */

#include "bootstrap.h"
#include "channel_manager.h"
#include "sliding_window.h"
#include "tcpx_handles.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>

// ============================================================================
// Test 1: SlidingWindow
// ============================================================================

void test_sliding_window() {
  std::cout << "\n=== Test 1: SlidingWindow ===" << std::endl;

  SlidingWindow window(16);

  // Test 1.1: Initial state
  assert(window.size() == 0);
  assert(!window.is_full());
  std::cout << "[PASS] Initial state" << std::endl;

  // Test 1.2: Add requests
  for (int i = 0; i < 10; i++) {
    void* fake_req = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
    window.add_request(fake_req, i);
  }
  assert(window.size() == 10);
  assert(!window.is_full());
  std::cout << "[PASS] Add 10 requests" << std::endl;

  // Test 1.3: Fill to capacity
  for (int i = 10; i < 16; i++) {
    void* fake_req = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
    window.add_request(fake_req, i);
  }
  assert(window.size() == 16);
  assert(window.is_full());
  std::cout << "[PASS] Fill to capacity (16)" << std::endl;

  // Test 1.4: Clear
  window.clear();
  assert(window.size() == 0);
  assert(!window.is_full());
  std::cout << "[PASS] Clear" << std::endl;

  std::cout << "[SUCCESS] SlidingWindow tests passed!" << std::endl;
}

// ============================================================================
// Test 2: Bootstrap Protocol (requires manual 2-process test)
// ============================================================================

void test_bootstrap_server() {
  std::cout << "\n=== Test 2: Bootstrap Server ===" << std::endl;

  // Create 4 fake handles
  std::vector<ncclNetHandle_v7> handles(4);
  for (int i = 0; i < 4; i++) {
    std::memset(handles[i].data, 'A' + i, 128);
  }

  // Create server
  int client_fd = -1;
  std::cout << "Waiting for client connection on port " << kBootstrapPort
            << "..." << std::endl;
  if (bootstrap_server_create(kBootstrapPort, &client_fd) != 0) {
    std::cerr << "[FAIL] bootstrap_server_create failed" << std::endl;
    return;
  }
  std::cout << "[PASS] Client connected" << std::endl;

  // Send handles
  if (bootstrap_server_send_handles(client_fd, handles) != 0) {
    std::cerr << "[FAIL] bootstrap_server_send_handles failed" << std::endl;
    close(client_fd);
    return;
  }
  std::cout << "[PASS] Sent 4 handles" << std::endl;

  close(client_fd);
  std::cout << "[SUCCESS] Bootstrap server test passed!" << std::endl;
}

void test_bootstrap_client(char const* server_ip) {
  std::cout << "\n=== Test 2: Bootstrap Client ===" << std::endl;

  // Connect to server
  int server_fd = -1;
  if (bootstrap_client_connect(server_ip, kBootstrapPort, &server_fd) != 0) {
    std::cerr << "[FAIL] bootstrap_client_connect failed" << std::endl;
    return;
  }
  std::cout << "[PASS] Connected to server" << std::endl;

  // Receive handles
  std::vector<ncclNetHandle_v7> handles;
  if (bootstrap_client_recv_handles(server_fd, handles) != 0) {
    std::cerr << "[FAIL] bootstrap_client_recv_handles failed" << std::endl;
    close(server_fd);
    return;
  }
  std::cout << "[PASS] Received " << handles.size() << " handles" << std::endl;

  // Verify handles
  assert(handles.size() == 4);
  for (int i = 0; i < 4; i++) {
    char expected = 'A' + i;
    bool all_match = true;
    for (int j = 0; j < 128; j++) {
      if (handles[i].data[j] != expected) {
        all_match = false;
        break;
      }
    }
    assert(all_match);
  }
  std::cout << "[PASS] Handles verified" << std::endl;

  close(server_fd);
  std::cout << "[SUCCESS] Bootstrap client test passed!" << std::endl;
}

// ============================================================================
// Test 3: ChannelManager Initialization
// ============================================================================

void test_channel_manager() {
  std::cout << "\n=== Test 3: ChannelManager ===" << std::endl;

  // Test 3.1: Create manager with 4 channels
  ChannelManager mgr(4, 0);

  int num_channels = mgr.get_num_channels();
  if (num_channels == 0) {
    std::cout << "[SKIP] TCPX library not available (expected on local machine)"
              << std::endl;
    std::cout << "[INFO] This test requires TCPX plugin to be installed"
              << std::endl;
    return;
  }

  std::cout << "[PASS] Created manager with " << num_channels << " channels"
            << std::endl;

  // Test 3.2: Get channel by index
  for (int i = 0; i < num_channels; i++) {
    ChannelResources& ch = mgr.get_channel(i);
    assert(ch.channel_id == i);
    assert(ch.net_dev == i);
    assert(ch.sliding_window != nullptr);
  }
  std::cout << "[PASS] Get channel by index" << std::endl;

  // Test 3.3: Round-robin channel selection
  for (int chunk_idx = 0; chunk_idx < 16; chunk_idx++) {
    ChannelResources& ch = mgr.get_channel_for_chunk(chunk_idx);
    int expected_channel = chunk_idx % num_channels;
    assert(ch.channel_id == expected_channel);
  }
  std::cout << "[PASS] Round-robin channel selection" << std::endl;

  std::cout << "[SUCCESS] ChannelManager tests passed!" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  std::cout << "=== Multi-Channel Module Tests ===" << std::endl;

  if (argc == 1) {
    // Run local tests only
    test_sliding_window();
    test_channel_manager();

    std::cout << "\n=== All Local Tests Passed! ===" << std::endl;
    std::cout << "\nTo test bootstrap protocol, run:" << std::endl;
    std::cout << "  Server: ./tests/test_modules server" << std::endl;
    std::cout << "  Client: ./tests/test_modules client <server_ip>"
              << std::endl;

  } else if (argc == 2 && std::string(argv[1]) == "server") {
    // Run bootstrap server test
    test_bootstrap_server();

  } else if (argc == 3 && std::string(argv[1]) == "client") {
    // Run bootstrap client test
    test_bootstrap_client(argv[2]);

  } else {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  Local tests:  " << argv[0] << std::endl;
    std::cerr << "  Server test:  " << argv[0] << " server" << std::endl;
    std::cerr << "  Client test:  " << argv[0] << " client <server_ip>"
              << std::endl;
    return 1;
  }

  return 0;
}

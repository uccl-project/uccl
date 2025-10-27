/**
 * @file bootstrap.cc
 * @brief Implementation of bootstrap protocol for multi-channel handshake
 */

#include "bootstrap.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>

// ============================================================================
// Helper functions
// ============================================================================

/**
 * @brief Send exact number of bytes (handles partial sends)
 */
static int send_exact(int fd, void const* data, size_t size) {
  size_t total_sent = 0;
  char const* ptr = static_cast<char const*>(data);

  while (total_sent < size) {
    ssize_t sent = send(fd, ptr + total_sent, size - total_sent, 0);
    if (sent <= 0) {
      std::cerr << "[Bootstrap] send failed: " << strerror(errno) << std::endl;
      return -1;
    }
    total_sent += sent;
  }

  return 0;
}

/**
 * @brief Receive exact number of bytes (handles partial receives)
 */
static int recv_exact(int fd, void* data, size_t size) {
  size_t total_received = 0;
  char* ptr = static_cast<char*>(data);

  while (total_received < size) {
    ssize_t received = recv(fd, ptr + total_received, size - total_received, 0);
    if (received <= 0) {
      std::cerr << "[Bootstrap] recv failed: " << strerror(errno) << std::endl;
      return -1;
    }
    total_received += received;
  }

  return 0;
}

// ============================================================================
// Server-side implementation
// ============================================================================

int bootstrap_server_create(int port, int* client_fd) {
  // Create TCP socket
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "[Bootstrap] socket() failed: " << strerror(errno)
              << std::endl;
    return -1;
  }

  // Set SO_REUSEADDR to allow quick restart
  int opt = 1;
  if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    std::cerr << "[Bootstrap] setsockopt(SO_REUSEADDR) failed: "
              << strerror(errno) << std::endl;
    close(listen_fd);
    return -1;
  }

  // Bind to all interfaces on specified port
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::cerr << "[Bootstrap] bind() failed on port " << port << ": "
              << strerror(errno) << std::endl;
    close(listen_fd);
    return -1;
  }

  // Listen for connections
  if (listen(listen_fd, 1) < 0) {
    std::cerr << "[Bootstrap] listen() failed: " << strerror(errno)
              << std::endl;
    close(listen_fd);
    return -1;
  }

  std::cout << "[Bootstrap] Server listening on port " << port << std::endl;

  // Accept one client connection
  sockaddr_in client_addr{};
  socklen_t client_len = sizeof(client_addr);
  int conn_fd =
      accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);

  if (conn_fd < 0) {
    std::cerr << "[Bootstrap] accept() failed: " << strerror(errno)
              << std::endl;
    close(listen_fd);
    return -1;
  }

  char client_ip[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
  std::cout << "[Bootstrap] Client connected from " << client_ip << std::endl;

  // Close listen socket (we only need one client)
  close(listen_fd);

  *client_fd = conn_fd;
  return 0;
}

int bootstrap_server_send_handles(
    int client_fd, std::vector<ncclNetHandle_v7> const& handles) {
  // Send channel count (NCCL-style protocol)
  uint32_t channel_count = static_cast<uint32_t>(handles.size());

  std::cout << "[Bootstrap] Sending " << channel_count << " handles to client"
            << std::endl;

  if (send_exact(client_fd, &channel_count, sizeof(channel_count)) != 0) {
    std::cerr << "[Bootstrap] Failed to send channel count" << std::endl;
    return -1;
  }

  // Send all handles in one batch
  size_t total_size = channel_count * sizeof(ncclNetHandle_v7);
  if (send_exact(client_fd, handles.data(), total_size) != 0) {
    std::cerr << "[Bootstrap] Failed to send handles" << std::endl;
    return -1;
  }

  std::cout << "[Bootstrap] Successfully sent " << channel_count << " handles"
            << std::endl;
  return 0;
}

// ============================================================================
// Client-side implementation
// ============================================================================

int bootstrap_client_connect(char const* server_ip, int port, int* server_fd) {
  // Create TCP socket
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    std::cerr << "[Bootstrap] socket() failed: " << strerror(errno)
              << std::endl;
    return -1;
  }

  // Connect to server
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);

  if (inet_pton(AF_INET, server_ip, &addr.sin_addr) <= 0) {
    std::cerr << "[Bootstrap] Invalid server IP: " << server_ip << std::endl;
    close(sock_fd);
    return -1;
  }

  std::cout << "[Bootstrap] Connecting to " << server_ip << ":" << port
            << std::endl;

  if (connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::cerr << "[Bootstrap] connect() failed: " << strerror(errno)
              << std::endl;
    close(sock_fd);
    return -1;
  }

  std::cout << "[Bootstrap] Connected to server" << std::endl;

  *server_fd = sock_fd;
  return 0;
}

int bootstrap_client_recv_handles(int server_fd,
                                  std::vector<ncclNetHandle_v7>& handles) {
  // Receive channel count (NCCL-style protocol)
  uint32_t channel_count = 0;

  if (recv_exact(server_fd, &channel_count, sizeof(channel_count)) != 0) {
    std::cerr << "[Bootstrap] Failed to receive channel count" << std::endl;
    return -1;
  }

  std::cout << "[Bootstrap] Receiving " << channel_count
            << " handles from server" << std::endl;

  if (channel_count == 0 || channel_count > 64) {
    std::cerr << "[Bootstrap] Invalid channel count: " << channel_count
              << std::endl;
    return -1;
  }

  // Resize vector and receive all handles in one batch
  handles.resize(channel_count);
  size_t total_size = channel_count * sizeof(ncclNetHandle_v7);

  if (recv_exact(server_fd, handles.data(), total_size) != 0) {
    std::cerr << "[Bootstrap] Failed to receive handles" << std::endl;
    return -1;
  }

  std::cout << "[Bootstrap] Successfully received " << channel_count
            << " handles" << std::endl;
  return 0;
}

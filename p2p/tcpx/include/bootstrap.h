/**
 * @file bootstrap.h
 * @brief Bootstrap protocol for multi-channel TCPX handshake
 *
 * Handles TCP-based exchange of TCPX handles between server and client.
 * Mirrors NCCL's multi-channel negotiation pattern (net.cc:626).
 *
 * Protocol:
 *   1. Server creates TCP listen socket
 *   2. Client connects to server
 *   3. Server sends: uint32_t channel_count + ncclNetHandle_v7[channel_count]
 *   4. Client receives handles and uses them to connect TCPX channels
 */

#pragma once

#include "tcpx_handles.h"
#include <cstddef>
#include <cstdint>
#include <vector>

// Bootstrap protocol constants
constexpr int kBootstrapPort = 12347;
constexpr size_t kHandleBytes = 128;

// ============================================================================
// Server-side functions
// ============================================================================

/**
 * @brief Create bootstrap server and wait for client connection
 *
 * Creates a TCP listen socket, binds to port, and accepts one client.
 * Returns the connected client socket.
 *
 * @param port TCP port to listen on
 * @param client_fd Output parameter for connected client socket
 * @return 0 on success, -1 on error
 */
int bootstrap_server_create(int port, int* client_fd);

/**
 * @brief Send multiple handles to client (NCCL-style batch protocol)
 *
 * Sends:
 *   1. uint32_t channel_count
 *   2. ncclNetHandle_v7[channel_count]
 *
 * @param client_fd Connected client socket
 * @param handles Vector of handles to send
 * @return 0 on success, -1 on error
 */
int bootstrap_server_send_handles(int client_fd,
                                  std::vector<ncclNetHandle_v7> const& handles);

// ============================================================================
// Client-side functions
// ============================================================================

/**
 * @brief Connect to bootstrap server
 *
 * @param server_ip Server IP address (e.g., "10.65.74.150")
 * @param port TCP port to connect to
 * @param server_fd Output parameter for connected server socket
 * @return 0 on success, -1 on error
 */
int bootstrap_client_connect(char const* server_ip, int port, int* server_fd);

/**
 * @brief Receive multiple handles from server (NCCL-style batch protocol)
 *
 * Receives:
 *   1. uint32_t channel_count
 *   2. ncclNetHandle_v7[channel_count]
 *
 * @param server_fd Connected server socket
 * @param handles Output vector of handles (resized to channel_count)
 * @return 0 on success, -1 on error
 */
int bootstrap_client_recv_handles(int server_fd,
                                  std::vector<ncclNetHandle_v7>& handles);

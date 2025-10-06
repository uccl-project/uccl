/**
 * @file channel_manager.h
 * @brief Multi-channel TCPX connection manager
 *
 * Manages lifecycle of multiple TCPX channels, each using a different NIC.
 * Mirrors NCCL's per-channel resource management (net.cc:84-142).
 *
 * Key responsibilities:
 * - Create listen/connect for N channels
 * - Map channel_id to netDev (NIC index)
 * - Register shared GPU memory across all channels
 * - Provide round-robin channel selection for chunks
 */

#pragma once

#include "tcpx_handles.h"
#include "tcpx_interface.h"
#include <array>
#include <string>
#include <vector>
#include <cuda.h>

// Forward declarations
class SlidingWindow;

/**
 * @brief Per-channel resources (mirrors NCCL's
 * sendNetResources/recvNetResources)
 */
struct ChannelResources {
  int channel_id;  // Logical channel index (0..N-1)
  int net_dev;     // TCPX device index resolved from iface list

  // Connection handles
  void* listen_comm;  // Server-side only
  void* recv_comm;    // Server-side only
  void* send_comm;    // Client-side only

  // TCPX device handle storage (required 16-byte alignment)
  alignas(16) std::array<uint8_t, 512> recv_dev_handle_storage;
  alignas(16) std::array<uint8_t, 512> send_dev_handle_storage;
  void* recv_dev_handle;  // Points into recv_dev_handle_storage
  void* send_dev_handle;  // Points into send_dev_handle_storage

  // Memory registration
  void* mhandle;

  // Sliding window helper (one per comm to mirror NCCL's inflight tracking)
  SlidingWindow* sliding_window;

  // Statistics / debugging
  uint64_t bytes_transferred;
  int chunks_processed;
};

/**
 * @brief Multi-channel manager
 */
class ChannelManager {
 public:
  /**
   * @brief Constructor
   * @param num_channels Number of channels to create
   * @param gpu_id GPU device ID (for CUDA context)
   */
  ChannelManager(int num_channels, int gpu_id);

  /**
   * @brief Destructor
   */
  ~ChannelManager();

  /**
   * @brief Get number of channels
   */
  int get_num_channels() const { return num_channels_; }

  /**
   * @brief Get channel by index
   * @param idx Channel index (0..num_channels-1)
   * @return Reference to channel resources
   */
  ChannelResources& get_channel(int idx);

  /**
   * @brief Get channel for a chunk (round-robin)
   * @param chunk_idx Chunk index
   * @return Reference to channel resources
   */
  ChannelResources& get_channel_for_chunk(int chunk_idx);

  // ========================================================================
  // Server-side methods
  // ========================================================================

  /**
   * @brief Create listen comms for all channels
   *
   * For each channel i:
   *   - Call tcpx_listen(net_dev=i, &handle, &listen_comm)
   *   - Store handle for bootstrap transmission
   *
   * @param handles Output vector of handles (one per channel)
   * @return 0 on success, -1 on error
   */
  int server_listen_all(std::vector<ncclNetHandle_v7>& handles);

  /**
   * @brief Accept connections for all channels
   *
   * For each channel:
   *   - Call tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle)
   *   - Retry with backoff if client hasn't connected yet
   *
   * @return 0 on success, -1 on error
   */
  int server_accept_all();

  // ========================================================================
  // Client-side methods
  // ========================================================================

  /**
   * @brief Connect to all channels
   *
   * For each channel i:
   *   - Call tcpx_connect_v5(net_dev=i, &handle[i], &send_comm,
   * &send_dev_handle)
   *
   * @param handles Input vector of handles (one per channel, from bootstrap)
   * @return 0 on success, -1 on error
   */
  int client_connect_all(std::vector<ncclNetHandle_v7> const& handles);

  // ========================================================================
  // Memory management
  // ========================================================================

  /**
   * @brief Register memory for all channels (shared buffer approach)
   *
   * All channels register the same GPU buffer.
   * Each chunk writes to: buffer + (chunk_idx * chunk_size)
   *
   * @param buffer GPU or host buffer pointer
   * @param size Buffer size in bytes
   * @param ptr_type NCCL_PTR_CUDA or NCCL_PTR_HOST
   * @param is_recv true for recv (server), false for send (client)
   * @return 0 on success, -1 on error
   */
  int register_memory(void* buffer, size_t size, int ptr_type, bool is_recv);

  /**
   * @brief Deregister memory for all channels
   * @param is_recv true for recv (server), false for send (client)
   * @return 0 on success, -1 on error
   */
  int deregister_memory(bool is_recv);

  // ========================================================================
  // Cleanup
  // ========================================================================

  /**
   * @brief Close all connections
   * @param is_recv true for recv (server), false for send (client)
   */
  void close_all(bool is_recv);

 private:
  int num_channels_;
  int gpu_id_;
  std::vector<ChannelResources> channels_;

  // Disable copy
  ChannelManager(ChannelManager const&) = delete;
  ChannelManager& operator=(ChannelManager const&) = delete;
};

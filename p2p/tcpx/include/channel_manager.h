/**
 * @file channel_manager.h
 * @brief Multi-channel TCPX connection manager
 */

#pragma once

#include "bootstrap.h"
#include "tcpx_interface.h"
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda.h>

struct ChannelResources {
  int channel_id;
  int net_dev;
  std::string nic_name;

  void* listen_comm;
  void* recv_comm;
  void* send_comm;

  alignas(16) std::array<uint8_t, 512> recv_dev_handle_storage;
  alignas(16) std::array<uint8_t, 512> send_dev_handle_storage;
  void* recv_dev_handle;
  void* send_dev_handle;

  // Multi-memory registration support
  // Maps mem_id -> mhandle for recv buffers
  std::unordered_map<uint64_t, void*> recv_mhandles;
  // Maps mem_id -> mhandle for send buffers
  std::unordered_map<uint64_t, void*> send_mhandles;

  uint64_t bytes_transferred;
  int chunks_processed;
};

class ChannelManager {
 public:
  ChannelManager(int num_channels, int gpu_id);
  ~ChannelManager();

  int get_num_channels() const { return num_channels_; }
  ChannelResources& get_channel(int idx);
  ChannelResources& get_channel_for_chunk(int chunk_idx);

  // Server-side
  int server_listen_all(std::vector<ncclNetHandle_v7>& handles);
  int server_accept_all();

  // Client-side
  int client_connect_all(std::vector<ncclNetHandle_v7> const& handles);

  // Memory management (multi-registration support)
  int register_memory(uint64_t mem_id, void* buffer, size_t size, int ptr_type,
                      bool is_recv);
  int deregister_memory(uint64_t mem_id, bool is_recv);
  void* get_mhandle(uint64_t mem_id, bool is_recv, int channel_id);
  void close_all(bool is_recv);

 private:
  int num_channels_;
  int gpu_id_;
  std::vector<ChannelResources> channels_;

  ChannelManager(ChannelManager const&) = delete;
  ChannelManager& operator=(ChannelManager const&) = delete;
};

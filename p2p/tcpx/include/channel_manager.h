/**
 * @file channel_manager.h
 * @brief Multi-channel TCPX connection manager
 */

#pragma once

#include "tcpx_handles.h"
#include "tcpx_interface.h"
#include <array>
#include <string>
#include <vector>
#include <cuda.h>

class SlidingWindow;

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

  void* mhandle;
  SlidingWindow* sliding_window;

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

  // Memory management
  int register_memory(void* buffer, size_t size, int ptr_type, bool is_recv);
  int deregister_memory(bool is_recv);
  void close_all(bool is_recv);

 private:
  int num_channels_;
  int gpu_id_;
  std::vector<ChannelResources> channels_;

  ChannelManager(ChannelManager const&) = delete;
  ChannelManager& operator=(ChannelManager const&) = delete;
};

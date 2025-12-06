#pragma once
#include "define.h"
#include "efa_channel_group.h"
#include "efa_ctrl_channel.h"
#include "epoll_client.h"
#include "epoll_server.h"
#include "memory_allocator.h"
#include "rdma_context.h"
#include "rdma_device.h"
#include <glog/logging.h>

#include "util/net.h"

// ChannelGroup manages multiple channels for a connection

// EFAEndpoint manages RDMA contexts and channel groups
class EFAEndpoint {
 public:
  // Constructor with required gpu_index and optional device_ids
  // If device_ids is empty, all available devices will be used
  explicit EFAEndpoint(
      int gpu_index, uint64_t rank_id, uint64_t port = 0,
      std::vector<size_t> const& device_ids = std::vector<size_t>())
      : gpu_index_(gpu_index), rank_id_(rank_id), port_(port), oob_meta_() {
    uccl::pin_thread_to_numa(0);
    std::vector<size_t> actual_device_ids;
    if (device_ids.size() == 0) {
      actual_device_ids =
          RdmaDeviceManager::instance().get_best_dev_idx(gpu_index);
    } else {
      actual_device_ids = device_ids;
    }
    initializeContexts(actual_device_ids);
    UCCL_LOG_EP << "EFAEndpoint initialized with " << contexts_.size()
                << " context(s) for GPU " << gpu_index_;

    oob_server_ = std::make_shared<EpollServer>(
        port_, [this](std::string const& input, std::string& output,
                      std::string const& ip, int port) {
          this->process_meta(input, output, ip, port);
        });
    oob_client_ = std::make_shared<EpollClient>();

    allocator_ = std::make_shared<MemoryAllocator>();
    assert(oob_server_->start());
    assert(oob_client_->start());

    // Initialize oob_meta_ with server information
    // oob_meta_.server_ip = uccl::get_oob_ip();  // Default to localhost
    oob_meta_.server_ip = uccl::get_oob_ip();
    oob_meta_.server_port = oob_server_->get_port();
    oob_meta_.gpu_id = gpu_index_;
  }

  // Destructor
  ~EFAEndpoint() {
    if (oob_client_) {
      oob_client_->stop();
    }
    if (oob_server_) {
      oob_server_->stop();
    }
  }

  // Getters
  int gpuIndex() const { return gpu_index_; }

  size_t contextCount() const { return contexts_.size(); }

  std::shared_ptr<RdmaContext> getContext(size_t index) const {
    if (index >= contexts_.size()) return nullptr;
    return contexts_[index];
  }

  std::vector<std::shared_ptr<RdmaContext>> const& contexts() const {
    return contexts_;
  }

  bool regMem(std::shared_ptr<RegMemBlock> reg_block) {
    if (unlikely(!reg_block)) {
      UCCL_LOG_ERROR << "Error: regMem called with null reg_block";
      return false;
    }

    for (size_t context_id = 0; context_id < contexts_.size(); ++context_id) {
      auto context = contexts_[context_id];
      if (unlikely(!context)) {
        UCCL_LOG_ERROR << "Error: context at context_id " << context_id
                       << " is null";
        return false;
      }

      // Register memory region for this context
      struct ibv_mr* mr = context->regMem(reg_block->addr, reg_block->size);

      if (unlikely(!mr)) {
        UCCL_LOG_ERROR << "Error: ibv_reg_mr failed for block at "
                       << reg_block->addr << " size " << reg_block->size
                       << " context_id " << context_id;
        return false;
      }

      // Store the MR in the RegMemBlock's mr_map using context_id as key
      reg_block->mr_map[context_id] = mr;

      UCCL_LOG_RE << "Registered memory block at " << reg_block->addr
                  << " size " << reg_block->size << " with context_id "
                  << context_id << " (lkey: 0x" << std::hex << mr->lkey
                  << ", rkey: 0x" << mr->rkey << std::dec << ")";
    }

    return true;
  }

  bool deregMem(std::shared_ptr<RegMemBlock> reg_block) {
    if (unlikely(!reg_block)) {
      UCCL_LOG_ERROR << "Error: deregMem called with null reg_block";
      return false;
    }

    // Deregister memory regions for all contexts in the mr_map
    for (auto const& [context_id, mr] : reg_block->mr_map) {
      if (unlikely(!mr)) {
        UCCL_LOG_ERROR << "Error: mr is null for context_id " << context_id;
        continue;
      }

      // Deregister memory region using RdmaContext's static method
      RdmaContext::deregMem(mr);

      UCCL_LOG_RE << "Deregistered memory block at " << reg_block->addr
                  << " size " << reg_block->size << " with context_id "
                  << context_id;
    }

    // Clear the mr_map after deregistration
    reg_block->mr_map.clear();

    return true;
  }

  std::shared_ptr<RecvChannelGroup> getOrCreateRecvGroup(uint64_t rank_id) {
    // Fast path: try shared (reader) lock first
    {
      std::shared_lock read_lock(recv_channel_mutex_);
      auto it = recv_channel_groups_.find(rank_id);
      if (it != recv_channel_groups_.end()) {
        return it->second;  // found -> return (shared_ptr keeps object alive)
      }
    }

    // Not found => acquire unique lock and insert (recheck to avoid race)
    {
      std::unique_lock write_lock(recv_channel_mutex_);
      auto [it, inserted] = recv_channel_groups_.try_emplace(
          rank_id,
          std::make_shared<RecvChannelGroup>());  // try_emplace constructs only
                                                  // if inserting
      // it->second now contains the group (either existing or newly created)
      return it->second;
    }
  }

  void addOneRecvChannel(uint64_t rank_id, uint32_t channel_id,
                         std::shared_ptr<EFAChannel> new_channel) {
    // 1) Get or create the group with minimal locking
    std::shared_ptr<RecvChannelGroup> group_ptr = getOrCreateRecvGroup(rank_id);
    group_ptr->addChannel(channel_id, new_channel);
  }

  std::shared_ptr<SendChannelGroup> getOrCreateSendGroup(uint64_t rank_id) {
    {
      std::shared_lock read_lock(send_channel_mutex_);
      auto it = send_channel_groups_.find(rank_id);
      if (it != send_channel_groups_.end()) return it->second;
    }
    {
      std::unique_lock write_lock(send_channel_mutex_);
      auto [it, inserted] = send_channel_groups_.try_emplace(
          rank_id, std::make_shared<SendChannelGroup>());
      return it->second;
    }
  }

  void addOneSendChannel(uint64_t rank_id, uint32_t channel_id,
                         std::shared_ptr<EFAChannel> new_channel) {
    auto group_ptr = getOrCreateSendGroup(rank_id);
    group_ptr->addChannel(channel_id, new_channel);
  }

  void setSendControlChannel(
      uint64_t rank_id, std::shared_ptr<SendControlChannel>&& ctrl_channel) {
    std::unique_lock write_lock(send_channel_mutex_);
    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      send_channel_groups_[rank_id] = std::make_shared<SendChannelGroup>();
    }
    send_channel_groups_[rank_id]->setControlChannel(
        std::forward<std::shared_ptr<SendControlChannel>>(ctrl_channel));
  }

  void setRecvControlChannel(
      uint64_t rank_id, std::shared_ptr<RecvControlChannel>&& ctrl_channel) {
    std::unique_lock write_lock(recv_channel_mutex_);
    auto it = recv_channel_groups_.find(rank_id);
    if (it == recv_channel_groups_.end()) {
      recv_channel_groups_[rank_id] = std::make_shared<RecvChannelGroup>();
    }
    recv_channel_groups_[rank_id]->setControlChannel(
        std::forward<std::shared_ptr<RecvControlChannel>>(ctrl_channel));
  }

  std::string const build_oob_connect(uint64_t rank_id) {
    auto const& item = rank_oob_meta_.find(rank_id);
    std::shared_ptr<OOBMetaData> ip_port_ptr = item->second;
    std::string oob_con;
    while (oob_con.empty()) {
      oob_con = oob_client_->connect_to_server(ip_port_ptr->server_ip,
                                               ip_port_ptr->server_port);
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    return oob_con;
  }

  void build_control_channel(std::string const& oob_con, uint64_t rank_id) {
    // Allocate memory for control channel ring buffer
    auto ctrl_mem =
        allocator_->allocate(kRingBufferSize, MemoryType::HOST, contexts_[0]);
    auto control_channel =
        std::make_shared<SendControlChannel>(contexts_[0], ctrl_mem, 0);
    std::shared_ptr<RemoteMemInfo> ctrl_info =
        std::make_shared<RemoteMemInfo>(ctrl_mem);
    MetaInfoToExchange ctrl_meta(rank_id_, 0,
                                 control_channel->get_local_meta(),
                                 ctrl_info, ChannelType::Control, gpu_index_);
    LOG(INFO)<<ctrl_meta<<std::endl;
    std::string ctrl_serialized_meta = serialize(ctrl_meta);
    bool ctrl_sent = oob_client_->send_meta(
        oob_con, ctrl_serialized_meta,
        [this, control_channel](std::string const& response) {
          uint64_t rank_id =
              this->handle_send_meta_response(control_channel, response);
          // Create a copy of shared_ptr and move it to setSendControlChannel
          auto ctrl_ch_copy = control_channel;
          this->setSendControlChannel(rank_id, std::move(ctrl_ch_copy));
        });
    LOG(INFO) << "Created control channel with QPN: "
                << control_channel->get_local_meta()->qpn;
  }

  // Synchronous version using future and promise
  uint64_t build_control_channel_sync(std::string const& oob_con,
                                      uint64_t rank_id,
                                      int timeout_ms = 10000) {
    // Allocate memory for control channel ring buffer
    auto ctrl_mem =
        allocator_->allocate(kRingBufferSize, MemoryType::HOST, contexts_[0]);
    auto control_channel =
        std::make_shared<SendControlChannel>(contexts_[0], ctrl_mem, 0);
    std::shared_ptr<RemoteMemInfo> ctrl_info =
        std::make_shared<RemoteMemInfo>(ctrl_mem);
    MetaInfoToExchange ctrl_meta(rank_id_, 0,
                                 control_channel->get_local_meta(),
                                 ctrl_info, ChannelType::Control, gpu_index_);
    LOG(INFO) <<ctrl_meta<<std::endl;
    std::string ctrl_serialized_meta = serialize(ctrl_meta);

    // Create promise and future for synchronization
    auto promise_ptr = std::make_shared<std::promise<void>>();
    std::future<void> future = promise_ptr->get_future();
    uint64_t recv_rank_id;
    bool ctrl_sent = oob_client_->send_meta(
        oob_con, ctrl_serialized_meta,
        [this, control_channel, promise_ptr,
         &recv_rank_id](std::string const& response) {
          try {
            uint64_t rank_id =
                this->handle_send_meta_response(control_channel, response);
            // Create a copy of shared_ptr and move it to setSendControlChannel
            auto ctrl_ch_copy = control_channel;
            this->setSendControlChannel(rank_id, std::move(ctrl_ch_copy));
            promise_ptr->set_value();  // Signal completion
            recv_rank_id = rank_id;
          } catch (...) {
            // Capture any exception and propagate through promise
            promise_ptr->set_exception(std::current_exception());
          }
        });

    if (!ctrl_sent) {
      LOG(ERROR) << "Failed to send control channel metadata for rank "
                     << rank_id;
      return false;
    }

    LOG(INFO) << "Created control channel with QPN: "
                << control_channel->get_local_meta()->qpn
                << ", waiting for handshake...";

    // Wait for callback to complete with timeout
    if (future.wait_for(std::chrono::milliseconds(timeout_ms)) ==
        std::future_status::timeout) {
      LOG(ERROR)
          << "Timeout waiting for control channel handshake for rank "
          << rank_id;
      return false;
    }

    try {
      future.get();  // Throws if callback had an exception
    } catch (std::exception const& e) {
      LOG(ERROR) << "Error in control channel callback for rank " << rank_id
                     << ": " << e.what();
      return false;
    }
    LOG(INFO) << "Control channel handshake completed successfully for rank "
                << recv_rank_id;
    return recv_rank_id;
  }

  bool build_normal_channels(std::string const& oob_con, uint64_t rank_id) {
    for (int i = 0; i < kQpNumPerChannel; i++) {
      uint32_t channel_id = i + 1;
      auto new_channel = std::make_shared<EFAChannel>(
          contexts_[i % contexts_.size()], channel_id);
      MetaInfoToExchange meta(rank_id_, channel_id,
                              new_channel->get_local_meta(), nullptr,
                              ChannelType::Normal, gpu_index_);
      LOG(INFO) <<meta<<std::endl;
      std::string serialized_meta = serialize(meta);
      bool sent = oob_client_->send_meta(
          oob_con, serialized_meta,
          [this, new_channel, channel_id](std::string const& response) {
            uint64_t rank_id =
                this->handle_send_meta_response(new_channel, response);
            addOneSendChannel(rank_id, channel_id, new_channel);
          });
    }
    return true;
  }

  // Synchronous version using future and promise
  bool build_normal_channels_sync(std::string const& oob_con, uint64_t rank_id,
                                  int timeout_ms = 10000) {
    // Vector to hold futures for all channels
    std::vector<std::future<void>> futures;
    futures.reserve(kQpNumPerChannel);

    for (int i = 0; i < kQpNumPerChannel; i++) {
      uint32_t channel_id = i + 1;
      auto new_channel = std::make_shared<EFAChannel>(
          contexts_[i % contexts_.size()], channel_id);

      MetaInfoToExchange meta(rank_id_, channel_id,
                              new_channel->get_local_meta(), nullptr,
                              ChannelType::Normal, gpu_index_);
      std::string serialized_meta = serialize(meta);

      // Create a shared_ptr to promise to ensure it outlives the callback
      auto promise_ptr = std::make_shared<std::promise<void>>();
      futures.push_back(promise_ptr->get_future());

      bool sent = oob_client_->send_meta(
          oob_con, serialized_meta,
          [this, new_channel, channel_id,
           promise_ptr](std::string const& response) {
            try {
              uint64_t rank_id =
                  this->handle_send_meta_response(new_channel, response);
              addOneSendChannel(rank_id, channel_id, new_channel);
              promise_ptr->set_value();  // Signal completion
            } catch (...) {
              // Capture any exception and propagate through promise
              promise_ptr->set_exception(std::current_exception());
            }
          });

      if (!sent) {
        UCCL_LOG_ERROR << "Failed to send metadata for channel " << channel_id;
        return false;
      }
    }

    // Wait for all channels to complete with timeout
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    for (size_t i = 0; i < futures.size(); i++) {
      auto remaining_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              deadline - std::chrono::steady_clock::now());

      if (remaining_time.count() <= 0 ||
          futures[i].wait_for(remaining_time) == std::future_status::timeout) {
        UCCL_LOG_ERROR << "Timeout waiting for channel " << (i + 1)
                       << " to complete";
        return false;
      }

      try {
        futures[i].get();  // Throws if callback had an exception
      } catch (std::exception const& e) {
        UCCL_LOG_ERROR << "Error in channel " << (i + 1)
                       << " callback: " << e.what();
        return false;
      }
    }

    UCCL_LOG_EP << "All " << kQpNumPerChannel
                << " normal channels built successfully for rank " << rank_id;
    return true;
  }
  bool build_connect(uint64_t rank_id) {
    if (rank_id == rank_id_) {
      return false;
    }
    std::string const oob_con = build_oob_connect(rank_id);
    this->build_control_channel(oob_con, rank_id);
    this->build_normal_channels(oob_con, rank_id);
    return true;
  }

  int build_connect_sync(uint64_t rank_id) {
    if (rank_id == rank_id_) {
      return -1;
    }
    std::string const oob_con = build_oob_connect(rank_id);
    uint64_t receved_rank_id =
        this->build_control_channel_sync(oob_con, rank_id);
    this->build_normal_channels_sync(oob_con, rank_id);
    return receved_rank_id;
  }
  // Blocking check for send completion
  void checkSendComplete(uint64_t rank_id, int64_t wr_id) {
    LOG(INFO) << "checkSendComplete - rank_id: " << rank_id
              << ", wr_id: " << wr_id;

    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    while (!send_group->check(wr_id)) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    LOG(INFO) << "checkSendComplete - Completed for rank_id: " << rank_id
              << ", wr_id: " << wr_id;
  }

  bool checkSendComplete_once(uint64_t rank_id, int64_t wr_id) {
    // LOG(INFO) << "checkSendComplete - rank_id: " << rank_id
    //           << ", wr_id: " << wr_id;

    auto it = send_channel_groups_.find(rank_id);
    if (unlikely(it == send_channel_groups_.end())) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    return send_group->check(wr_id);
  }

  bool checkRecvComplete_once(uint64_t rank_id, uint64_t index) {
    // LOG(INFO) << "checkRecvComplete - Checking for rank_id: " << rank_id
    //           << ", index: " << index;
    auto it = recv_channel_groups_.find(rank_id);
    if (unlikely(it == recv_channel_groups_.end())) {
      throw std::runtime_error("Recv channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto recv_group = it->second;
    return recv_group->check(index);
  }

  // Blocking check for recv completion
  void checkRecvComplete(uint64_t rank_id, uint64_t index) {
    LOG(INFO) << "checkRecvComplete - Checking for rank_id: " << rank_id
              << ", index: " << index;
    auto it = recv_channel_groups_.find(rank_id);
    if (it == recv_channel_groups_.end()) {
      throw std::runtime_error("Recv channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto recv_group = it->second;
    while (!recv_group->check(index)) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    LOG(INFO) << "checkRecvComplete - Completed for rank_id: " << rank_id
              << ", index: " << index;
  }

  // Check if connection is ready for a given rank_id
  // Waits until recv_channel_groups_[rank_id] exists and has kQpNumPerChannel+1
  // channels
  void connect_check(uint64_t rank_id) {
    while (true) {
      {
        std::shared_lock<std::shared_mutex> lock(recv_channel_mutex_);
        auto it = recv_channel_groups_.find(rank_id);
        if (it != recv_channel_groups_.end()) {
          auto group = it->second;
          if (group && group->channelCount() == kQpNumPerChannel + 1) {
            LOG(WARNING) << "connect_check - Connection ready for rank_id: "
                         << rank_id << " with " << group->channelCount()
                         << " channels";
            return;
          }
        }
      }  // Lock is released here

      // Wait a bit before checking again
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  int64_t write(std::shared_ptr<EFASendRequest> req){
    req->send_type = SendType::Write;
    uint64_t rank_id = req->to_rank_id;
    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    int64_t wr_id = -1;

    // Blocking call until send succeeds
    while (wr_id < 0) {
      LOG(INFO) << "EFAEndpoint::write - Attempting to send to rank_id: "
          << rank_id << ", peer rank_id " << rank_id;
      wr_id = send_group->write(req);

      if (wr_id < 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    return wr_id;
  }
  // Blocking send: wraps SendChannelGroup::send with rank_id parameter
  // Returns wr_id for checking completion later
  int64_t send(uint64_t rank_id, std::shared_ptr<EFASendRequest> req) {
    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    int64_t wr_id = -1;

    // Blocking call until send succeeds
    while (wr_id < 0) {
      LOG(INFO) << "EFAEndpoint::send - Attempting to send to rank_id: "
          << rank_id << ", peer rank_id " << rank_id;
      wr_id = send_group->send(req);

      if (wr_id < 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    return wr_id;
  }

  // Blocking recv: wraps RecvChannelGroup::recv with rank_id parameter
  // Returns index for checking completion later
  int64_t recv(uint64_t rank_id, std::shared_ptr<EFARecvRequest> req) {
    auto it = recv_channel_groups_.find(rank_id);
    if (it == recv_channel_groups_.end()) {
      throw std::runtime_error("Recv channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto recv_group = it->second;
    int64_t index = -1;
    // Blocking call until recv succeeds
    while (index < 0) {

      index = recv_group->recv(req);
      LOG(INFO) << "EFAEndpoint::recv - Attempting to recv from rank_id: "
                << rank_id <<  ", peer rank_id " << rank_id;
      if (index < 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    return index;
  }

  // Add or update rank OOB metadata from a given map
  void add_rank_oob_meta(
      std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> const&
          new_meta) {
    for (auto const& [rank_id, meta_ptr] : new_meta) {
      rank_oob_meta_[rank_id] = meta_ptr;
    }
  }
  uccl::ConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                            int remote_gpuidx, std::string remote_ip,
                            uint16_t remote_port) {
    add_rank_oob_meta({{kRankIDPlaceHolder, std::make_shared<OOBMetaData>(
                                                remote_ip, remote_port)}});
    std::cout <<"remote_gpuidx:"<<remote_gpuidx<<"remote_ip"<<remote_ip<<"remote_port"<<remote_port<<std::endl;
    int rank_id = build_connect_sync(kRankIDPlaceHolder);
    uccl::ConnID conn_id;
    conn_id.context = reinterpret_cast<void*>(static_cast<intptr_t>(rank_id));
    conn_id.flow_id = rank_id;
    return conn_id;
  };
  inline uint16_t get_p2p_listen_port(int dev) { return oob_server_->get_port(); };
  inline int get_p2p_listen_fd(int dev) {
    return oob_server_->get_listen_fd();
  };
  inline uccl::ConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                                  std::string& remote_ip, int* remote_dev,
                                  int* remote_gpuidx) {
    AcceptedMeta accepted;
    uint64_t rank_id = 0;

    // Block until there's an accepted connection
    while (true) {
      {
        {std::unique_lock<std::shared_mutex> lock(accepted_meta_mutex_);
        if (!accepted_meta_.empty()) {
          // Get the first accepted connection
          auto it = accepted_meta_.begin();
          rank_id = it->first;
          accepted = it->second;
          // Remove it from the map
          if(getOrCreateRecvGroup(rank_id)->channelCount() == kQpNumPerChannel + 1){
            accepted_meta_.erase(it);
            std::cout<< "Accepted connection: rank_id=" << rank_id
                        << ", ip=" << accepted.ip << ", port=" << accepted.port
                        << ", gpu_id=" << accepted.gpu_id; 
            UCCL_LOG_EP << "Accepted connection: rank_id=" << rank_id
                        << ", ip=" << accepted.ip << ", port=" << accepted.port
                        << ", gpu_id=" << accepted.gpu_id;
            break;
          }
        }
        }
      }
      // Wait before checking again
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout<< "Done Accepted connection: rank_id=" << rank_id
                << ", ip=" << accepted.ip << ", port=" << accepted.port
                << ", gpu_id=" << accepted.gpu_id;
    // Assign output parameters
    remote_ip = accepted.ip;
    if (remote_gpuidx != nullptr) {
      *remote_gpuidx = accepted.gpu_id;
    }
    if (remote_dev != nullptr) {
      *remote_dev = 0;  // Default device
    }

    // Create and return ConnID
    uccl::ConnID conn_id;
    conn_id.context = reinterpret_cast<void*>(static_cast<intptr_t>(rank_id));
    conn_id.flow_id = rank_id;
    return conn_id;
  }

  inline int uccl_regmr(uccl::UcclFlow* flow, void* data, size_t len, int type,
                        struct uccl::Mhandle** mhandle) {
    return 0;
  }
  inline int uccl_regmr(void* const data, const size_t len, std::unordered_map<int64_t, struct ibv_mr*>& mr_map) {
    if (unlikely(!data)) {
      UCCL_LOG_ERROR << "Error: uccl_regmr called with null data";
      return -1;
    }

    for (size_t context_id = 0; context_id < contexts_.size(); ++context_id) {
      auto context = contexts_[context_id];
      if (unlikely(!context)) {
        UCCL_LOG_ERROR << "Error: context at context_id " << context_id
                       << " is null";
        return -1;
      }

      // Register memory region for this context
      struct ibv_mr* mr = context->regMem(data, len);

      if (unlikely(!mr)) {
        UCCL_LOG_ERROR << "Error: ibv_reg_mr failed for data at "
                       << data << " size " << len
                       << " context_id " << context_id;
        return -1;
      }

      // Store the MR in the mr_map using context_id as key
      mr_map[context_id] = mr;

      UCCL_LOG_RE << "Registered memory at " << data
                  << " size " << len << " with context_id "
                  << context_id << " (lkey: 0x" << std::hex << mr->lkey
                  << ", rkey: 0x" << mr->rkey << std::dec << ")";
    }

    return 0;
  }

  inline int uccl_send_async(uccl::UcclFlow* flow,
                             struct uccl::Mhandle* mhandle, void const* data,
                             size_t const size,
                             struct uccl::ucclRequest* ureq) {
    return 0;
  }

  inline int uccl_recv_async(uccl::UcclFlow* flow,
                             struct uccl::Mhandle** mhandles, void** data,
                             int* size, int n, struct uccl::ucclRequest* ureq) {
    return 0;
  }
  inline bool uccl_poll_ureq_once(struct uccl::ucclRequest* ureq) { return false; }
  inline int uccl_read_async(uccl::UcclFlow* flow, struct uccl::Mhandle* local_mh,
                             void* dst, size_t size,
                             uccl::FifoItem const& slot_item,
                             uccl::ucclRequest* ureq) { return 0; }
  inline int uccl_write_async(uccl::UcclFlow* flow, struct uccl::Mhandle* local_mh,
                              void* src, size_t size,
                              uccl::FifoItem const& slot_item,
                              uccl::ucclRequest* ureq) { return 0; }
  inline int prepare_fifo_metadata(uint64_t rank_id,
                                   const std::unordered_map<int64_t, struct ibv_mr*>& mr_map,
                                   void const* data, size_t size,
                                    uint32_t rkeys[]) {


    // Get the recv group for this rank_id
    auto recv_group = getOrCreateRecvGroup(rank_id);
    if (!recv_group) {
      UCCL_LOG_ERROR << "Failed to get recv group for rank_id " << rank_id;
      return -1;
    }

    // Collect rkeys from all channels
    if (!recv_group->collectAllRkeys(mr_map, rkeys)) {
      UCCL_LOG_ERROR << "Failed to collect rkeys for rank_id " << rank_id;
      return -1;
    }

    // Serialize the remote_m

    return 0;
  }
  inline void uccl_deregmr(std::unordered_map<int64_t, struct ibv_mr*>& mr_map) {
    for (auto const& [context_id, mr] : mr_map) {
      if (unlikely(!mr)) {
        UCCL_LOG_ERROR << "Error: mr is null for context_id " << context_id;
        continue;
      }

      // Deregister memory region using RdmaContext's static method
      RdmaContext::deregMem(mr);

      UCCL_LOG_RE << "Deregistered memory with context_id " << context_id;
    }

    // Clear the mr_map after deregistration
    mr_map.clear();
  }

  int get_best_dev_idx(int gpu_idx) { return 0; }

  bool initialize_engine_by_dev(int dev, bool enable_p2p_listen) {
    return true;
  }

  void create_unified_p2p_socket() {}

 private:
  void initializeContexts(std::vector<size_t> const& device_ids) {
    auto& device_manager = RdmaDeviceManager::instance();

    // Determine which devices to use
    std::vector<size_t> target_device_ids;
    if (device_ids.empty()) {
      // Use all available devices
      for (size_t i = 0; i < device_manager.deviceCount(); ++i) {
        target_device_ids.push_back(i);
      }
    } else {
      target_device_ids = device_ids;
    }

    // Create context for each device
    for (size_t device_id : target_device_ids) {
      auto device = device_manager.getDevice(device_id);
      if (!device) {
        UCCL_LOG_ERROR << "Warning: Device " << device_id
                       << " not found, skipping";
        continue;
      }

      try {
        auto context = std::make_shared<RdmaContext>(device, contexts_.size());
        contexts_.push_back(context);
        UCCL_LOG_EP << "EFAEndpoint: Created context for device " << device_id
                    << " (" << device->name() << ")";
      } catch (std::exception const& e) {
        UCCL_LOG_ERROR << "Error creating context for device " << device_id
                       << ": " << e.what();
      }
    }

    if (contexts_.empty()) {
      throw std::runtime_error("EFAEndpoint: No contexts created");
    }
  }

  void process_meta(std::string const& input, std::string& output,
                    std::string const& client_ip, int client_port) {
    MetaInfoToExchange meta = deserialize<MetaInfoToExchange>(input);
    LOG(INFO) << "Received from " << client_ip << ":" << client_port
                   << " - " << meta;

    auto context_id = meta.channel_id % contexts_.size();
    std::shared_ptr<RdmaContext> ctx_ptr = contexts_[context_id];

    if (meta.flag == ChannelType::Control) {
      // This is a control channel, allocate memory and create
      // RecvControlChannel
      auto ctrl_mem =
          allocator_->allocate(kRingBufferSize, MemoryType::HOST, ctx_ptr);
      LOG(INFO) << "process_meta: Allocated " << ctrl_mem->size
                  << " bytes for recv control channel ring buffer at "
                  << ctrl_mem->addr;

      auto recv_ctrl_channel = std::make_shared<RecvControlChannel>(
          ctx_ptr, meta, ctrl_mem, meta.channel_id);

      // Create respons
      RemoteMemInfo ctrl_info(ctrl_mem);
      MetaInfoToExchange response(rank_id_, meta.channel_id,
                                  recv_ctrl_channel->get_local_meta(),
                                  nullptr, ChannelType::Control, gpu_index_);
      response.mem_meta = ctrl_info;
      LOG(INFO) << "response (control channel):::::::" << response;
      output = serialize(response);

      // Set the control channel
      auto ctrl_ch_copy = recv_ctrl_channel;
      setRecvControlChannel(meta.rank_id, std::move(ctrl_ch_copy));

      // Store accepted connection metadata
      {
        std::unique_lock<std::shared_mutex> lock(accepted_meta_mutex_);
        AcceptedMeta accepted;
        accepted.ip = client_ip;
        accepted.port = static_cast<uint16_t>(client_port);
        accepted.gpu_id = meta.gpu_id;
        accepted.rank_id = meta.rank_id;
        accepted_meta_[meta.rank_id] = accepted;
        LOG(INFO) << "Stored accepted connection: rank_id=" << meta.rank_id
                    << ", ip=" << client_ip << ", port=" << client_port
                    << ", gpu_id=" << meta.gpu_id;
      }
    } else {
      // Normal channel
      std::shared_ptr<EFAChannel> new_channel = std::make_shared<EFAChannel>(
          ctx_ptr, meta.channel_meta, meta.channel_id);
      // Create response (echo back the same data)
      MetaInfoToExchange response(rank_id_, meta.channel_id,
                                  new_channel->get_local_meta(), nullptr,
                                  ChannelType::Normal, gpu_index_);
      LOG(INFO) << "response:::::::" << response;
      output = serialize(response);
      addOneRecvChannel(meta.rank_id, meta.channel_id, new_channel);
    }
  }

  // Handle response from send_meta operation
  uint64_t handle_send_meta_response(std::shared_ptr<EFAChannel> channel,
                                     std::string const& response) {
    // Deserialize response as MetaInfoToExchange
    MetaInfoToExchange response_meta =
        deserialize<MetaInfoToExchange>(response);
    UCCL_LOG_EP << response_meta;
    channel->connect(response_meta.channel_meta);
    return response_meta.rank_id;
  }

  uint64_t rank_id_;
  int gpu_index_;
  std::vector<std::shared_ptr<RdmaContext>> contexts_;
  mutable std::shared_mutex recv_channel_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<RecvChannelGroup>>
      recv_channel_groups_;
  mutable std::shared_mutex send_channel_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<SendChannelGroup>>
      send_channel_groups_;

  uint64_t port_;
  std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> rank_oob_meta_;
  std::shared_ptr<EpollClient> oob_client_;
  std::shared_ptr<EpollServer> oob_server_;
  std::shared_ptr<MemoryAllocator> allocator_;
  OOBMetaData oob_meta_;
  mutable std::shared_mutex accepted_meta_mutex_;
  std::unordered_map<uint64_t, AcceptedMeta> accepted_meta_;
};
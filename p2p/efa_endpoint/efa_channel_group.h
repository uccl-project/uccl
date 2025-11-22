#pragma once
#include "define.h"
#include "efa_channel.h"
#include "efa_ctrl_channel.h"
#include <shared_mutex>
#include <glog/logging.h>

class ChannelGroup {
 public:
  ChannelGroup() = default;
  virtual ~ChannelGroup() = default;

  virtual void addChannel(uint32_t channel_id,
                          std::shared_ptr<EFAChannel> channel) {
    if (!channel) {
      throw std::invalid_argument("addChannel called with null channel");
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);

    channels_[channel_id] = std::move(channel);
  }

  virtual std::shared_ptr<EFAChannel> getChannel(uint32_t channel_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = channels_.find(channel_id);
    if (it == channels_.end()) return nullptr;
    return it->second;
  }

  virtual size_t channelCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return channels_.size();
  }

  virtual std::unordered_map<uint32_t, std::shared_ptr<EFAChannel>> const&
  channels() const {
    mutex_.lock_shared();
    mutex_.unlock_shared();  // just to annotate read lock expected
    return channels_;
  }

 protected:
  mutable std::shared_mutex mutex_;
  std::unordered_map<uint32_t, std::shared_ptr<EFAChannel>> channels_;
};

class SendChannelGroup : public ChannelGroup {
 public:
  SendChannelGroup() : running_(false), poll_thread_(nullptr) {}

  ~SendChannelGroup() { stopPolling(); }

  void addChannel(uint32_t channel_id,
                  std::shared_ptr<EFAChannel> channel) override {
    ChannelGroup::addChannel(channel_id, channel);
    // Add SendChannelGroup specific logic here
  }

  std::shared_ptr<EFAChannel> getChannel(uint32_t channel_id) const override {
    auto result = ChannelGroup::getChannel(channel_id);
    // Add SendChannelGroup specific logic here
    return result;
  }

  size_t channelCount() const override {
    auto result = ChannelGroup::channelCount();
    // Add SendChannelGroup specific logic here
    return result;
  }

  std::unordered_map<uint32_t, std::shared_ptr<EFAChannel>> const& channels()
      const override {
    // Add SendChannelGroup specific logic here
    return ChannelGroup::channels();
  }

  // Set the control channel
  template <typename T>
  void setControlChannel(T&& ctrl_channel) {
    if (ctrl_channel_) {
      throw std::runtime_error(
          "SendChannelGroup: Control channel has already been set");
    }
    ctrl_channel_ = std::forward<T>(ctrl_channel);
    startPolling();
  }

  bool setupRecvRequest(std::shared_ptr<EFASendRequest> req) {
    if (unlikely(!req || !req->local_mem)) {
      return false;
    }

    auto channel = getChannel(req->channel_id);
    if (unlikely(!channel)) {
      return false;
    }
    uint64_t context_id = channel->getContextID();

    auto& mr_map = req->local_mem->mr_map;
    auto it = mr_map.find(context_id);
    if (unlikely(it == mr_map.end())) {
      return false;
    }

    req->local_mem->mr = it->second;

    return true;
    
  }
  // Send function: channel_id and EFASendRequest
  int64_t send(std::shared_ptr<EFASendRequest> req) {
    // Get additional info from SendControlChannel
    if (!ctrl_channel_) {
      return -1;
    }

    if (!ctrl_channel_->getOneSendRequest(req)) {
      return -1;
    }

    // Find channel by channel_id
    LOG(INFO) << "SendChannelGroup: Sending on channel_id " << req->channel_id;
    auto channel = getChannel(req->channel_id);
    setupRecvRequest(req);
    if (!channel) {
      LOG(WARNING) << "SendChannelGroup: Channel not found for channel_id " << req->channel_id;
      return -1;
    }

    // Call send on the channel
    return channel->send(req);
  }

  // Start polling thread
  void startPolling() {
    if (running_.load()) {
      return;
    }
    running_.store(true);
    poll_thread_ =
        std::make_unique<std::thread>(&SendChannelGroup::pollingLoop, this);
  }

  bool check(uint32_t channel_id, int64_t wr_id) {
    auto channel = getChannel(channel_id);
    if (!channel) {
      return false;
    }
    return channel->isAcknowledged(wr_id);
  }
  // Stop polling thread
  void stopPolling() {
    if (!running_.load()) {
      return;
    }
    running_.store(false);
    if (poll_thread_ && poll_thread_->joinable()) {
      poll_thread_->join();
    }
  }

 private:
  std::shared_ptr<SendControlChannel> ctrl_channel_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;

  // Polling loop
  // void pollingLoop() {
  //   while (running_.load()) {
  //     // Poll all channels
  //     CQMeta cq_data;
  //     for (auto& [channel_id, channel] : channels_) {
  //       channel->poll_once(cq_data);
  //     }

  //     // Poll control channel
  //     if (ctrl_channel_) {
  //       ctrl_channel_->noblockingPoll();
  //     }
  //   }
  // }
  void pollingLoop() {
    LOG(INFO) << "SendChannelGroup::pollingLoop - Started";
    uint64_t iteration_count = 0;
    while (running_.load(std::memory_order_acquire)) {
      // ---- Step 1: Copy shared data under read lock ----
      std::vector<std::shared_ptr<EFAChannel>> local_channels;
      std::shared_ptr<SendControlChannel> local_ctrl;

      {  // scope for lock
        std::shared_lock lock(mutex_);
        local_channels.reserve(channels_.size());
        for (auto& kv : channels_) {
          local_channels.push_back(kv.second);
        }
        local_ctrl = ctrl_channel_;
      }

      // ---- Step 2: unlock, do slow polling outside lock ----
      CQMeta cq_data;
      for (size_t i = 0; i < local_channels.size(); ++i) {
        auto& ch = local_channels[i];
        if (ch && ch->poll_once(cq_data)) {
          LOG(INFO) << "SendChannelGroup::pollingLoop - Channel " << i
                    << " polled completion: " << cq_data;
        }
      }

      if (local_ctrl) {
        if (local_ctrl->noblockingPoll()) {
          LOG(INFO) << "SendChannelGroup::pollingLoop - Control channel polled successfully";
        }
      }

      // Print periodic status every 10000 iterations
      iteration_count++;
      if (iteration_count % 10000 == 0) {
        LOG(INFO) << "SendChannelGroup::pollingLoop - Still running, iteration: " << iteration_count
                  << ", channels: " << local_channels.size();
      }
    }
    LOG(INFO) << "SendChannelGroup::pollingLoop - Stopped";
  }
};

class RecvChannelGroup : public ChannelGroup {
 public:
  RecvChannelGroup() : running_(false), poll_thread_(nullptr), last_channel_id_(0) {}

  ~RecvChannelGroup() { stopPolling(); }

  void addChannel(uint32_t channel_id,
                  std::shared_ptr<EFAChannel> channel) override {
    ChannelGroup::addChannel(channel_id, channel);
    // Add RecvChannelGroup specific logic here
  }

  std::shared_ptr<EFAChannel> getChannel(uint32_t channel_id) const override {
    auto result = ChannelGroup::getChannel(channel_id);
    // Add RecvChannelGroup specific logic here
    return result;
  }

  size_t channelCount() const override {
    auto result = ChannelGroup::channelCount();
    // Add RecvChannelGroup specific logic here
    return result;
  }

  std::unordered_map<uint32_t, std::shared_ptr<EFAChannel>> const& channels()
      const override {
    // Add RecvChannelGroup specific logic here
    return ChannelGroup::channels();
  }

  // Set the control channel
  template <typename T>
  void setControlChannel(T&& ctrl_channel) {
    if (ctrl_channel_) {
      throw std::runtime_error(
          "RecvChannelGroup: Control channel has already been set");
    }
    ctrl_channel_ = std::forward<T>(ctrl_channel);
    startPolling();
  }

  // Start polling thread
  void startPolling() {
    if (running_.load()) {
      return;
    }
    running_.store(true);
    poll_thread_ =
        std::make_unique<std::thread>(&RecvChannelGroup::pollingLoop, this);
  }

  // Stop polling thread
  void stopPolling() {
    if (!running_.load()) {
      return;
    }
    running_.store(false);
    if (poll_thread_ && poll_thread_->joinable()) {
      poll_thread_->join();
    }
  }

  
  int64_t recv(std::shared_ptr<EFARecvRequest> req) {
    if (unlikely(!setupRecvRequestWithRoundRobin(req))) {
      return -1;
    }
    return ctrl_channel_->postSendReq(req);
  }

  bool check(uint64_t index) { return ctrl_channel_->check_done(index); }

  // Round-robin channel selection and MR setup
  bool setupRecvRequestWithRoundRobin(std::shared_ptr<EFARecvRequest> req) {
    if (unlikely(!req || !req->local_mem)) {
      return false;
    }

    // Get the total number of channels
    size_t num_channels = channelCount();
    if (unlikely(num_channels == 0)) {
      return false;
    }

    // Round-robin: get next channel_id
    uint32_t current_id = last_channel_id_.load(std::memory_order_relaxed);
    uint32_t next_id = (current_id + 1) % num_channels;
    last_channel_id_.store(next_id, std::memory_order_relaxed);

    // Get the channel by channel_id
    auto channel = getChannel(next_id);
    if (unlikely(!channel)) {
      return false;
    }

    uint64_t context_id = channel->getContextID();

    // Get MR from mr_map using context_id as key
    auto& mr_map = req->local_mem->mr_map;
    auto it = mr_map.find(context_id);
    if (unlikely(it == mr_map.end())) {
      return false;
    }

    // Set the MR
    req->local_mem->mr = it->second;
    req->channel_id = next_id;
    LOG(INFO) << "RecvChannelGroup: Assigned channel_id " << next_id << " to recv request";

    return true;
  }

 private:
  std::shared_ptr<RecvControlChannel> ctrl_channel_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  std::atomic<uint32_t> last_channel_id_;

  // Polling loop
  // void pollingLoop() {
  //   while (running_.load()) {
  //     // Poll all channels
  //     CQMeta cq_data;
  //     std::shared_lock<std::shared_mutex> lock(mutex_);
  //     for (auto& [channel_id, channel] : channels_) {
  //       if(channel->poll_once(cq_data)){
  //         if(cq_data.hasIMM()){
  //           ctrl_channel_->recv_done(cq_data.imm);
  //         }
  //       }
  //     }

  //   }
  // }
  // 修改后的 pollingLoop（RecvChannelGroup）
  void pollingLoop() {
    LOG(INFO) << "RecvChannelGroup::pollingLoop - Started";
    uint64_t iteration_count = 0;
    while (running_.load(std::memory_order_acquire)) {
      CQMeta cq_data;
      std::vector<std::shared_ptr<EFAChannel>> snapshot;
      {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        snapshot.reserve(channels_.size());
        for (auto& kv : channels_) {
          snapshot.push_back(kv.second);
        }
      }

      for (size_t i = 0; i < snapshot.size(); ++i) {
        auto& channel = snapshot[i];
        if (!channel) continue;
        bool polled = false;
        try {
          polled = channel->poll_once(cq_data);
        } catch (...) {
          LOG(ERROR) << "RecvChannelGroup::pollingLoop - Exception in poll_once for channel " << i;
        }
        if (polled) {
          LOG(INFO) << "RecvChannelGroup::pollingLoop - Channel " << i
                    << " polled completion: " << cq_data;

          if (cq_data.hasIMM()) {
            LOG(INFO) << "RecvChannelGroup::pollingLoop - Completion has IMM data: " << cq_data.imm;
            if (ctrl_channel_) {
              ctrl_channel_->recv_done(cq_data.imm);
              LOG(INFO) << "RecvChannelGroup::pollingLoop - Called recv_done(" << cq_data.imm << ")";
            } else {
              LOG(WARNING) << "RecvChannelGroup::pollingLoop - ctrl_channel_ is null, cannot call recv_done";
            }
          }
        }
      }

      // Print periodic status every 10000 iterations
      iteration_count++;
      if (iteration_count % 10000000 == 0) {
        LOG(INFO) << "RecvChannelGroup::pollingLoop - Still running, iteration: " << iteration_count
                  << ", channels: " << snapshot.size();
      }

      // optional small sleep/yield to avoid busy-looping if desired:
      // std::this_thread::yield();
    }
    LOG(INFO) << "RecvChannelGroup::pollingLoop - Stopped";
  }
};

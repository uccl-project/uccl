#pragma once
#include "define.h"
#include "efa_channel.h"
#include "efa_ctrl_channel.h"
#include <shared_mutex>

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

  // Send function: channel_id and EFASendRequest
  int64_t send(uint32_t channel_id, std::shared_ptr<EFASendRequest> req) {
    // Get additional info from SendControlChannel
    if (!ctrl_channel_) {
      return -1;
    }

    if (!ctrl_channel_->getOneSendRequest(req)) {
      return -1;
    }

    // Find channel by channel_id
    auto channel = getChannel(channel_id);
    if (!channel) {
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
      for (auto& ch : local_channels) {
        if (ch) ch->poll_once(cq_data);
      }

      if (local_ctrl) {
        local_ctrl->noblockingPoll();
      }
    }
  }
};

class RecvChannelGroup : public ChannelGroup {
 public:
  RecvChannelGroup() : running_(false), poll_thread_(nullptr) {}

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

  int64_t recv(uint32_t channel_id, std::shared_ptr<EFARecvRequest> req) {
    return ctrl_channel_->postSendReq(req);
  }

  bool check(uint64_t index) { return ctrl_channel_->check_done(index); }

 private:
  std::shared_ptr<RecvControlChannel> ctrl_channel_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;

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

      for (auto& channel : snapshot) {
        if (!channel) continue;
        bool polled = false;
        try {
          polled = channel->poll_once(cq_data);
        } catch (...) {
          std::cout << "????????????????????????" << std::endl;
        }
        if (polled && cq_data.hasIMM()) {
          if (ctrl_channel_) {
            ctrl_channel_->recv_done(cq_data.imm);
          }
        }
      }

      // optional small sleep/yield to avoid busy-looping if desired:
      // std::this_thread::yield();
    }
  }
};

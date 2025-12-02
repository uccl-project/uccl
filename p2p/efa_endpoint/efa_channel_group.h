#pragma once
#include "define.h"
#include "efa_channel.h"
#include "efa_ctrl_channel.h"

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
  SendChannelGroup() : running_(false), poll_thread_(nullptr) {
    tracker_ = std::make_shared<AtomicBitmapPacketTracker>();
    request_queue_ = std::make_unique<
        RingBuffer<std::shared_ptr<EFASendRequest>, kRingCapacity>>();
  }

  ~SendChannelGroup() { stopPolling(); }

  void addChannel(uint32_t channel_id,
                  std::shared_ptr<EFAChannel> channel) override {
    ChannelGroup::addChannel(channel_id, channel);
  }

  std::shared_ptr<EFAChannel> getChannel(uint32_t channel_id) const override {
    auto result = ChannelGroup::getChannel(channel_id);
    return result;
  }

  size_t channelCount() const override {
    auto result = ChannelGroup::channelCount();
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    if (ctrl_channel_) {
      result += 1;
    }
    return result;
  }

  size_t normalChannelCount() const { return ChannelGroup::channelCount(); }

  std::unordered_map<uint32_t, std::shared_ptr<EFAChannel>> const& channels()
      const override {
    return ChannelGroup::channels();
  }

  template <typename T>
  void setControlChannel(T&& ctrl_channel) {
    std::unique_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    if (ctrl_channel_) {
      throw std::runtime_error(
          "SendChannelGroup: Control channel has already been set");
    }
    ctrl_channel_ = std::forward<T>(ctrl_channel);
    lock.unlock();
    startPolling();
  }

  int64_t send(std::shared_ptr<EFASendRequest> req) {
    int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
    req->wr_id = wr_id;
    if (unlikely(request_queue_->push(req) < 0)) {
      LOG(WARNING) << "SendChannelGroup: isend request queue is full, wr_id="
                   << wr_id;
      return -1;
    }
    return wr_id;
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

  bool check(int64_t wr_id) { return tracker_->isAcknowledged(wr_id); }
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
  mutable std::shared_mutex ctrl_channel_mutex_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  std::unique_ptr<RingBuffer<std::shared_ptr<EFASendRequest>, kRingCapacity>>
      request_queue_;
  std::shared_ptr<AtomicBitmapPacketTracker> tracker_;
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

  // Send a request through the appropriate channel
  // Returns true on success, false on failure
  bool sendRequestOnChannel(std::shared_ptr<EFASendRequest> req) {
    auto channel = getChannel(req->channel_id);
    if (unlikely(!channel)) {
      LOG(WARNING) << "SendChannelGroup: Channel not found for channel_id "
                   << req->channel_id;
      return false;
    }

    if (!setupRecvRequest(req)) {
      LOG(WARNING)
          << "SendChannelGroup: Failed to setup request for channel_id "
          << req->channel_id;
      return false;
    }

    int64_t send_ret = channel->send(req);
    if (send_ret < 0) {
      LOG(WARNING) << "SendChannelGroup: Failed to send on channel_id "
                   << req->channel_id;
      return false;
    }

    return true;
  }

  void pollControlChannel() {
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    if (ctrl_channel_) {
      if (ctrl_channel_->noblockingPoll()) {
        LOG(INFO) << "SendChannelGroup::pollingLoop - Control channel "
                     "polled successfully";
      }
    }
  }

  void processSendRequests() {
    SendReqMeta meta;
    int index;
    {
      std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
      index = ctrl_channel_ ? ctrl_channel_->getOneSendRequestMeta(meta) : -1;
    }

    while (index >= 0) {
      LOG(INFO) << "SendChannelGroup: Processing send request meta: " << meta;
      std::shared_ptr<EFASendRequest> req;
      while (tracker_->getTotalInflightBytes() > kInFlightMaxSizeKB * 1024 ||
             !request_queue_->pop(req)) {
        if (tracker_->getTotalInflightBytes() > kInFlightMaxSizeKB * 1024) {
          LOG(WARNING) << "SendChannelGroup: In-flight bytes exceed limit,pausing sending."<< tracker_->getTotalInflightBytes() << " bytes in-flight.";

          // std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
      }

      // Check if message needs to be split into chunks
      if (meta.expected_chunk_count == 1) {
        // Original logic: send as a single message
        auto remote_mem = std::make_shared<RemoteMemInfo>(meta.remote_mem);
        req->remote_mem = remote_mem;
        req->channel_id = meta.channel_id;
        req->imm_data = index;

        // Send the request
        sendRequestOnChannel(req);
      } else {
        // Split message into chunks
        size_t message_size = req->local_mem->size;
        auto chunks = splitMessageToChunks(message_size);

        LOG(INFO) << "SendChannelGroup: Splitting message into "
                  << chunks.size() << " chunks (message_size: " << message_size
                  << ")";

        size_t num_channels = normalChannelCount();
        uint32_t base_channel_id = meta.channel_id;

        for (size_t i = 0; i < chunks.size(); ++i) {
          auto const& chunk = chunks[i];

          // Use different channel for each chunk: round-robin
          uint32_t chunk_channel_id =
              ((base_channel_id - 1 + i) % num_channels) + 1;

          // Create RegMemBlock for this chunk
          auto chunk_local_mem = std::make_shared<RegMemBlock>(
              static_cast<char*>(req->local_mem->addr) + chunk.offset,
              chunk.size, req->local_mem->type, req->local_mem->mr, false);
          chunk_local_mem->mr_map = req->local_mem->mr_map;

          // Create RemoteMemInfo for this chunk
          auto chunk_remote_mem = std::make_shared<RemoteMemInfo>(
              meta.remote_mem.addr + chunk.offset,
              meta.remote_mem.rkeys[chunk_channel_id], chunk.size,
              meta.remote_mem.type);

          // Create send request for this chunk
          // Only the last chunk needs signaled for completion notification
          bool is_last_chunk = (i == chunks.size() - 1);
          auto chunk_req = std::make_shared<EFASendRequest>(
              chunk_local_mem, chunk_remote_mem, index, is_last_chunk);
          chunk_req->channel_id = chunk_channel_id;
          chunk_req->from_rank_id = req->from_rank_id;
          chunk_req->to_rank_id = req->to_rank_id;
          chunk_req->wr_id = req->wr_id;

          // Send the chunk
          if (sendRequestOnChannel(chunk_req)) {
            LOG(INFO) << "SendChannelGroup: Sent chunk " << i << "/"
                      << chunks.size() << " (offset: " << chunk.offset
                      << ", size: " << chunk.size
                      << ", channel_id: " << chunk_channel_id << ")";
          } else {
            LOG(WARNING) << "SendChannelGroup: Failed to send chunk " << i
                         << " (offset: " << chunk.offset
                         << ", size: " << chunk.size
                         << ", channel_id: " << chunk_channel_id << ")";
          }
        }
      }

      {
        std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
        index = ctrl_channel_ ? ctrl_channel_->getOneSendRequestMeta(meta) : -1;
      }
    }
  }

  void pollDataChannels() {
    CQMeta cq_data;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (auto& [channel_id, channel] : channels_) {
      if (channel && channel->poll_once(cq_data)) {
        LOG(INFO) << "SendChannelGroup::pollingLoop - Channel " << channel_id
                  << " polled completion: " << cq_data;
        tracker_->acknowledge(cq_data.wr_id);
      }
    }
  }

  void pollingLoop() {
    LOG(INFO) << "SendChannelGroup::pollingLoop - Started";
    uccl::pin_thread_to_numa(0);
    while (running_.load(std::memory_order_acquire)) {
      pollControlChannel();
      processSendRequests();
      pollDataChannels();

      LOG_EVERY_N_ENDPOINT(INFO, 100000000)
          << "SendChannelGroup::pollingLoop - Still running";
    }
    LOG(INFO) << "SendChannelGroup::pollingLoop - Stopped";
  }
};

class RecvChannelGroup : public ChannelGroup {
 public:
  RecvChannelGroup()
      : running_(false), poll_thread_(nullptr), last_channel_id_(0) {}

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
    if (ctrl_channel_) {
      result += 1;
    }
    return result;
  }
  size_t normalChannelCount() const { return ChannelGroup::channelCount(); }
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
      LOG(WARNING)
          << "RecvChannelGroup: Failed to setup recv request with round robin";
      return -1;
    }
    return ctrl_channel_->postSendReq(req);
  }

  bool check(uint64_t index) { return ctrl_channel_->check_done(index); }

 private:
  std::shared_ptr<RecvControlChannel> ctrl_channel_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  std::atomic<uint32_t> last_channel_id_;

  void pollAndProcessCompletions() {
    CQMeta cq_data;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if(ctrl_channel_){
      ctrl_channel_->noblockingPoll();
    }
    for (auto& [channel_id, channel] : channels_) {
      if (!channel) continue;
      bool polled = false;
      try {
        polled = channel->poll_once(cq_data);
      } catch (...) {
        LOG(ERROR) << "RecvChannelGroup::pollAndProcessCompletions - Exception "
                      "in poll_once for channel "
                   << channel_id;
      }
      if (polled) {
        LOG(INFO) << "RecvChannelGroup::pollAndProcessCompletions - Channel "
                  << channel_id << " polled completion: " << cq_data;

        if (cq_data.hasIMM()) {
          LOG(INFO) << "RecvChannelGroup::pollAndProcessCompletions - "
                       "Completion has IMM data: "
                    << cq_data.imm;
          if (ctrl_channel_) {
            ctrl_channel_->recv_done(cq_data.imm);
            LOG(INFO) << "RecvChannelGroup::pollAndProcessCompletions - Called "
                         "recv_done("
                      << cq_data.imm << ")";
          } else {
            LOG(WARNING) << "RecvChannelGroup::pollAndProcessCompletions - "
                            "ctrl_channel_ is null, cannot call recv_done";
          }
        }
      }
    }
    LOG_EVERY_N_ENDPOINT(INFO, 100000000)
        << "RecvChannelGroup::pollingLoop - Still running, channels: "
        << channels_.size();
  }

  void pollingLoop() {
    LOG(INFO) << "RecvChannelGroup::pollingLoop - Started";
    uccl::pin_thread_to_numa(0);
    while (running_.load(std::memory_order_acquire)) {
      pollAndProcessCompletions();

      // optional small sleep/yield to avoid busy-looping if desired:
      // std::this_thread::yield();
    }
    LOG(INFO) << "RecvChannelGroup::pollingLoop - Stopped";
  }

  // Round-robin channel selection and MR setup
  bool setupRecvRequestWithRoundRobin(std::shared_ptr<EFARecvRequest> req) {
    if (unlikely(!req || !req->local_mem)) {
      return false;
    }

    // Get the total number of channels
    size_t num_channels = normalChannelCount();
    std::cout<<"num_channels::"<<num_channels<<std::endl<<std::flush;
    if (unlikely(num_channels == 0)) {
      LOG(WARNING)
          << "RecvChannelGroup: No channels available for recv request";
      return false;
    }

    // Round-robin: get next channel_id
    uint32_t current_id = last_channel_id_.load(std::memory_order_relaxed);
    uint32_t next_id = (current_id ) % num_channels + 1;
    last_channel_id_.store(next_id, std::memory_order_relaxed);

    // Get the channel by channel_id
    auto channel = getChannel(next_id);
    if (unlikely(!channel)) {
      LOG(WARNING) << "RecvChannelGroup: Channel not found for channel_id "
                   << next_id;
      return false;
    }

    uint64_t context_id = channel->getContextID();

    // Get MR from mr_map using context_id as key
    auto& mr_map = req->local_mem->mr_map;
    auto it = mr_map.find(context_id);
    if (unlikely(it == mr_map.end())) {
      LOG(WARNING) << "RecvChannelGroup: MR not found for context_id "
                   << context_id;
      return false;
    }

    // Set the MR
    req->local_mem->mr = it->second;
    req->channel_id = next_id;
    LOG(INFO) << "RecvChannelGroup: Assigned channel_id " << next_id
              << " to recv request";

    for (int i = 1; i < kQpNumPerChannel + 1; ++i) {
      auto channel = getChannel(i);
      if (unlikely(!channel)) {
        LOG(WARNING) << "RecvChannelGroup: Channel not found for channel_id "
                     << next_id;
        return false;
      }
      uint64_t context_id = channel->getContextID();
      auto& mr_map = req->local_mem->mr_map;
      auto it = mr_map.find(context_id);
      if (unlikely(it == mr_map.end())) {
        LOG(WARNING) << "RecvChannelGroup: MR not found for context_id "
                     << context_id;
        return false;
      }
      req->local_mem->rkeys[i] = it->second->rkey;
    }

    return true;
  }
};

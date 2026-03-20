#pragma once
#include "compression.h"
#include "define.h"
#include "rdma_ctrl_channel.h"
#include "rdma_data_channel.h"
#include <chrono>
#include <random>

class RDMAConnection {
 public:
  RDMAConnection() : last_channel_id_(0) {}
  virtual ~RDMAConnection() = default;

  virtual void addChannel(uint32_t channel_id,
                          std::shared_ptr<RDMADataChannel> channel) {
    if (!channel) {
      throw std::invalid_argument("addChannel called with null channel");
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);

    channels_[channel_id] = std::move(channel);
  }

  virtual std::shared_ptr<RDMADataChannel> getChannel(
      uint32_t channel_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = channels_.find(channel_id);
    if (it == channels_.end()) return nullptr;
    return it->second;
  }

  virtual size_t channelCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return channels_.size();
  }

  virtual std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
  channels() const {
    mutex_.lock_shared();
    mutex_.unlock_shared();  // just to annotate read lock expected
    return channels_;
  }

  // Select next channel using round-robin algorithm
  // Returns: pair<channel_id, context_id>, or pair<0, 0> on failure
  std::pair<uint32_t, uint64_t> selectNextChannelRoundRobin() {
    // Get the total number of channels
    size_t num_channels = RDMAConnection::channelCount();
    if (unlikely(num_channels == 0)) {
      UCCL_LOG(WARN)
          << "RDMAConnection: No channels available for round-robin selection";
      return {0, 0};
    }

    // Round-robin: get next channel_id
    uint32_t current_id = last_channel_id_.load(std::memory_order_relaxed);
    uint32_t next_id = (current_id) % num_channels + 1;
    last_channel_id_.store(next_id, std::memory_order_relaxed);

    // Get the channel by channel_id
    auto channel = getChannel(next_id);
    if (unlikely(!channel)) {
      UCCL_LOG(WARN) << "RDMAConnection: Channel not found for channel_id "
                     << next_id << " num_channels " << num_channels;
      return {0, 0};
    }

    uint64_t context_id = channel->getContextID();
    return {next_id, context_id};
  }

  // Select next channel using random selection algorithm
  // Returns: pair<channel_id, context_id>, or pair<0, 0> on failure
  std::pair<uint32_t, uint64_t> selectNextChannelRandom() {
    // Get the total number of channels
    size_t num_channels = RDMAConnection::channelCount();
    if (unlikely(num_channels == 0)) {
      UCCL_LOG(WARN)
          << "RDMAConnection: No channels available for random selection";
      return {0, 0};
    }

    // Random selection: generate random channel_id in range [1, num_channels]
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(1, num_channels);
    uint32_t random_id = dist(rng);

    // Get the channel by channel_id
    auto channel = getChannel(random_id);
    if (unlikely(!channel)) {
      UCCL_LOG(WARN) << "RDMAConnection: Channel not found for channel_id "
                     << random_id << " num_channels " << num_channels;
      return {0, 0};
    }

    uint64_t context_id = channel->getContextID();
    return {random_id, context_id};
  }

 protected:
  mutable std::shared_mutex mutex_;
  std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> channels_;
  std::atomic<uint32_t> last_channel_id_;
};

class SendConnection : public RDMAConnection {
 public:
  SendConnection(int numa_node, bool auto_start_polling = true)
      : numa_node_(numa_node),
        running_(false),
        poll_thread_(nullptr),
        auto_start_polling_(auto_start_polling) {
    tracker_ = std::make_shared<AtomicBitmapPacketTrackerMultiAck>();
    request_queue_ = std::make_unique<
        RingBuffer<std::shared_ptr<RDMASendRequest>, kRingCapacity>>();
  }

  ~SendConnection() { stopPolling(); }

  void addChannel(uint32_t channel_id,
                  std::shared_ptr<RDMADataChannel> channel) override {
    RDMAConnection::addChannel(channel_id, channel);
  }

  std::shared_ptr<RDMADataChannel> getChannel(
      uint32_t channel_id) const override {
    auto result = RDMAConnection::getChannel(channel_id);
    return result;
  }

  size_t channelCount() const override {
    auto result = RDMAConnection::channelCount();
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    if (ctrl_channel_) {
      result += 1;
    }
    return result;
  }

  size_t normalChannelCount() const { return RDMAConnection::channelCount(); }

  std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
  channels() const override {
    return RDMAConnection::channels();
  }

  template <typename T>
  void setControlChannel(T&& ctrl_channel) {
    std::unique_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    if (ctrl_channel_) {
      throw std::runtime_error(
          "SendConnection: Control channel has already been set");
    }
    ctrl_channel_ = std::forward<T>(ctrl_channel);
    lock.unlock();
    if (auto_start_polling_) {
      startPolling();
    }
  }

  int64_t send(std::shared_ptr<RDMASendRequest> req) {
    int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
    req->wr_id = wr_id;
    if (unlikely(request_queue_->push(req) < 0)) {
      UCCL_LOG(WARN) << "SendConnection: isend request queue is full, wr_id="
                     << wr_id;
      return -1;
    }
    return wr_id;
  }

  int64_t postWriteOrRead(std::shared_ptr<RDMASendRequest> req) {
    if (unlikely(req->send_type != SendType::Write &&
                 req->send_type != SendType::Read)) {
      UCCL_LOG(ERROR) << "SendConnection::write - Invalid send_type, expected "
                         "SendType::Write";
      return -1;
    }
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
    req->wr_id = wr_id;

    auto [channel_id, context_id] = selectNextChannelRoundRobin();
    if (unlikely(channel_id == 0)) {
      UCCL_LOG(ERROR) << "SendConnection::write - Failed to select channel";
      return -1;
    }

    req->channel_id = channel_id;
    postChunkedRequest(req);

    return wr_id;
  }

  int64_t read(std::shared_ptr<RDMASendRequest> req) {
    if (unlikely(req->send_type != SendType::Read)) {
      UCCL_LOG(ERROR) << "SendConnection::read - Invalid send_type, expected "
                         "SendType::Read";
      return -1;
    }
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
    req->wr_id = wr_id;

    auto [channel_id, context_id] = selectNextChannelRoundRobin();
    if (unlikely(channel_id == 0)) {
      UCCL_LOG(ERROR) << "SendConnection::read - Failed to select channel";
      return -1;
    }

    req->channel_id = channel_id;
    postChunkedRequest(req);

    return wr_id;
  }

  // Start polling thread
  void startPolling() {
    if (running_.load()) {
      return;
    }
    running_.store(true);
    poll_thread_ =
        std::make_unique<std::thread>(&SendConnection::pollingLoop, this);
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

  void pollingLoopForMeta() {
    pollControlChannel();
    pollDataChannels();
    LOG_EVERY_N_ENDPOINT(INFO, 100000000)
        << "SendConnection::pollingLoop - Still running";
  }

  int processSendRequests(std::shared_ptr<RDMASendRequest> req) {
    pollControlChannel();
    if (unlikely(ctrl_channel_ == nullptr)) {
      return -1;
    }
    SendReqMeta meta;
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    int index = ctrl_channel_->getOneSendRequestMeta(meta);
    if (index < 0) {
      return -1;
    }
    int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
    req->wr_id = wr_id;
    UCCL_LOG(INFO, UCCL_RDMA)
        << "SendConnection: Processing send request meta: " << meta;
    processOnceSendRequests(req, meta, index);
    return wr_id;
  }

 private:
  std::shared_ptr<SendControlChannel> ctrl_channel_;
  mutable std::shared_mutex ctrl_channel_mutex_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  std::unique_ptr<RingBuffer<std::shared_ptr<RDMASendRequest>, kRingCapacity>>
      request_queue_;
  std::shared_ptr<AtomicBitmapPacketTrackerMultiAck> tracker_;
  bool auto_start_polling_;
  int numa_node_ = 0;

  // Send a request through the appropriate channel
  // Returns true on success, false on failure
  bool postRequestOnChannel(std::shared_ptr<RDMASendRequest> req) {
    auto channel = getChannel(req->channel_id);
    if (unlikely(!channel)) {
      UCCL_LOG(WARN) << "SendConnection: Channel not found for channel_id "
                     << req->channel_id;
      return false;
    }

    int64_t send_ret = channel->submitRequest(req);
    if (send_ret < 0) {
      UCCL_LOG(WARN) << "SendConnection: Failed to send on channel_id "
                     << req->channel_id;
      return false;
    }

    return true;
  }

  void pollControlChannel() {
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    if (ctrl_channel_) {
      if (ctrl_channel_->noblockingPoll()) {
        UCCL_LOG(INFO, UCCL_RDMA)
            << "SendConnection::pollingLoop - Control channel polled "
               "successfully";
      }
    }
  }

  std::vector<std::shared_ptr<RDMASendRequest>> divideRequest(
      int expected_chunk_count,
      std::shared_ptr<RDMASendRequest> const& req,
      int chunk_offset = 0) {
    std::vector<std::shared_ptr<RDMASendRequest>> result;
    if (expected_chunk_count <= 0) {
      UCCL_LOG(ERROR) << "SendConnection::divideRequest - "
                         "expected_chunk_count <= 0: "
                      << expected_chunk_count;
      return result;
    }
    result.reserve(expected_chunk_count);
    size_t num_channels = normalChannelCount();
    size_t message_size = req->local_mem->size;
    size_t chunk_size =
        ChunkSplitStrategy::getRegularChunkSize(message_size,
                                                expected_chunk_count);

    for (int i = 0; i < expected_chunk_count; ++i) {
      uint64_t offset = static_cast<uint64_t>(i) * chunk_size;
      size_t size = std::min(chunk_size, message_size - offset);

      // Use different channel for each chunk: round-robin
      uint32_t chunk_channel_id =
          ((req->channel_id - 1 + i) % num_channels) + 1;

      // Create RegMemBlock for this chunk
      auto chunk_local_mem = std::make_shared<RegMemBlock>(
          static_cast<char*>(req->local_mem->addr) + offset, size,
          req->local_mem->mr_array, req->local_mem->type);

      // Create RemoteMemInfo for this chunk
      auto chunk_remote_mem = std::make_shared<RemoteMemInfo>(
          req->remote_mem->addr + offset, size,
          req->remote_mem->rkey_array, req->remote_mem->type);

      // Only the last chunk needs signaled for completion notification
      bool is_last_chunk = (i == expected_chunk_count - 1);
      auto chunk_req = std::make_shared<RDMASendRequest>(
          chunk_local_mem, chunk_remote_mem, req->imm_data, is_last_chunk);

      chunk_req->imm_data.set_chunk_range(chunk_offset + i, chunk_offset + i);
      chunk_req->channel_id = chunk_channel_id;
      chunk_req->from_rank_id = req->from_rank_id;
      chunk_req->to_rank_id = req->to_rank_id;
      chunk_req->wr_id = req->wr_id;
      chunk_req->send_type = req->send_type;
      result.push_back(std::move(chunk_req));
    }
    return result;
  }
  
  void postChunkedRequest(std::shared_ptr<RDMASendRequest> req,
                          int expected_chunk_count = 0,
                          int chunk_offset = 0) {
    size_t message_size = req->local_mem->size;
    if (expected_chunk_count == 0) {
      expected_chunk_count =
          ChunkSplitStrategy::getMessageChunkCount(message_size);
      tracker_->updateExpectedAckCount(req->wr_id, expected_chunk_count);
    }
    UCCL_LOG(INFO, UCCL_RDMA)
        << "SendConnection: Splitting message into " << expected_chunk_count
        << " chunks (message_size: " << message_size << ")";

    auto divided_reqs =
        divideRequest(expected_chunk_count, req, chunk_offset);
    for (auto const& divided_req : divided_reqs) {
      if (!postRequestOnChannel(divided_req)) {
        UCCL_LOG(WARN) << "SendConnection: Failed to send chunk"
                       << " (channel_id: " << divided_req->channel_id << ")";
      }
    }
  }

  void processSendRequests() {
    if (unlikely(ctrl_channel_ == nullptr)) {
      return;
    }
    SendReqMeta meta;
    bool has_meta = false;
    int index = -1;
    // {
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    has_meta = ctrl_channel_->hasSendRequest();
    // }
    while (has_meta) {
      std::shared_ptr<RDMASendRequest> req;
      if (tracker_->getTotalInflightBytes() > kInFlightMaxSizeKB * 1024 ||
          !request_queue_->pop(req)) {
        if (tracker_->getTotalInflightBytes() > kInFlightMaxSizeKB * 1024) {
          UCCL_LOG(WARN) << "SendConnection: In-flight bytes exceed "
                            "limit,pausing sending."
                         << tracker_->getTotalInflightBytes()
                         << " bytes in-flight.";
        }
        break;
      }
      index = ctrl_channel_->getOneSendRequestMeta(meta);
      UCCL_LOG(INFO, UCCL_RDMA)
          << "SendConnection: Processing send request meta: " << meta;
      processOnceSendRequests(req, meta, index);
      has_meta = ctrl_channel_->hasSendRequest();
    }
  }

  inline void compressSendRequest(std::shared_ptr<RDMASendRequest> req) {
    Compressor::getInstance().compress(req);
  }

  inline void compressSendRequestSplitFirst(
      std::shared_ptr<RDMASendRequest> req, size_t expected_chunk_count) {
    auto t0 = std::chrono::high_resolution_clock::now();
    Compressor::getInstance().compressSplitOneBatch(req);
    auto t1 = std::chrono::high_resolution_clock::now();

    // compressed data / chunk size = chunk count
    uint32_t send_chunks_first =
        req->local_mem->size /
        ChunkSplitStrategy::getRegularChunkSize(req->compress_ctx->maxSize,
                                                expected_chunk_count);
    if (send_chunks_first > 0) {
      postChunkedRequest(req, send_chunks_first, 0);
    }
    // std::cout<< "compressSendRequestSplitFirst: send_chunks_first=" << send_chunks_first
    //           << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    uint32_t uncompressed_size =
        Compressor::getInstance().compressEncodeOneBatch(req);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto split_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto encode_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    UCCL_LOG(INFO) << "compressSendRequestSplitFirst: split=" << split_us
                    << "us, encode=" << encode_us << "us";

    postChunkedRequest(req, expected_chunk_count - send_chunks_first,
                       send_chunks_first);
    tracker_->updateExpectedAckCount(
        req->wr_id,
        ChunkSplitStrategy::getMessageChunkCount(uncompressed_size));
  }

  inline void compressSendRequestPipelineEncode(
      std::shared_ptr<RDMASendRequest> req, size_t expected_chunk_count) {
    auto divided_reqs = divideRequest(expected_chunk_count, req);
    tracker_->updateExpectedAckCount(req->wr_id, expected_chunk_count);

    size_t offset = 0;
    for (auto& chunk_req : divided_reqs) {
      chunk_req->compress_ctx = req->compress_ctx;
      auto chunk_size = chunk_req->local_mem->size;
      // std::cout << "compressSendRequestPipelineEncode: chunk_size=" << chunk_size
      //           << std::endl;
      Compressor::getInstance().compress(chunk_req, offset);
      offset += chunk_size;
      if (!postRequestOnChannel(chunk_req)) {
        UCCL_LOG(WARN)
            << "SendConnection::compressSendRequestPipelineEncode - "
               "Failed to send chunk (channel_id: "
            << chunk_req->channel_id << ")";
      }
    }
  }

  inline void processOnceSendRequests(std::shared_ptr<RDMASendRequest> req,
                                      SendReqMeta& meta, int index) {
    req->imm_data.set_index(index);
    req->channel_id = meta.channel_id;
    req->remote_mem = std::make_shared<RemoteMemInfo>(meta.remote_mem);
    if (Compressor::getInstance().shouldCompressAndPipelineEncode(
            req->local_mem->size)) {
      // std::cout << "processOnceSendRequests: pipeline encode, expected_chunk_count="
      //           << meta.expected_chunk_count << std::endl;
      compressSendRequestPipelineEncode(req, meta.expected_chunk_count);
      return;
    }
    if (Compressor::getInstance().shouldCompressAndSplitFirst(
            req->local_mem->size)) {
      compressSendRequestSplitFirst(req, meta.expected_chunk_count);
      return;
    }
    if (Compressor::getInstance().shouldCompress(req->local_mem->size)) {
      compressSendRequest(req);
    }
    tracker_->updateExpectedAckCount(
        req->wr_id,
        ChunkSplitStrategy::getMessageChunkCount(req->local_mem->size));
    postChunkedRequest(req, meta.expected_chunk_count);
  }

  void pollDataChannels() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (auto& [channel_id, channel] : channels_) {
      std::vector<CQMeta> cq_datas;
      if (channel && channel->pollOnce(cq_datas)) {
        for (auto const& cq_data : cq_datas) {
          // UCCL_LOG(INFO, UCCL_RDMA) << "SendConnection::pollingLoop -
          // Channel "
          // << channel_id
          //           << " polled completion: " << cq_data;
          tracker_->acknowledge(cq_data.wr_id);
        }
      }
    }
  }

  void pollingLoop() {
    UCCL_LOG(INFO, UCCL_RDMA) << "SendConnection::pollingLoop - Started";
    uccl::pin_thread_to_numa(numa_node_);
    while (running_.load(std::memory_order_acquire)) {
      pollControlChannel();
      processSendRequests();
      pollDataChannels();

      LOG_EVERY_N_ENDPOINT(INFO, 100000000)
          << "SendConnection::pollingLoop - Still running";
    }
    UCCL_LOG(INFO, UCCL_RDMA) << "SendConnection::pollingLoop - Stopped";
  }
};

class RecvConnection : public RDMAConnection {
 public:
  RecvConnection(int numa_node, bool auto_start_polling = true)
      : numa_node_(numa_node),
        running_(false),
        poll_thread_(nullptr),
        auto_start_polling_(auto_start_polling) {}

  ~RecvConnection() { stopPolling(); }

  void addChannel(uint32_t channel_id,
                  std::shared_ptr<RDMADataChannel> channel) override {
    RDMAConnection::addChannel(channel_id, channel);
  }

  std::shared_ptr<RDMADataChannel> getChannel(
      uint32_t channel_id) const override {
    auto result = RDMAConnection::getChannel(channel_id);
    return result;
  }

  size_t channelCount() const override {
    auto result = RDMAConnection::channelCount();
    if (ctrl_channel_) {
      result += 1;
    }
    return result;
  }

  size_t normalChannelCount() const { return RDMAConnection::channelCount(); }

  std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
  channels() const override {
    return RDMAConnection::channels();
  }

  // Set the control channel
  template <typename T>
  void setControlChannel(T&& ctrl_channel) {
    if (ctrl_channel_) {
      throw std::runtime_error(
          "RecvConnection: Control channel has already been set");
    }
    ctrl_channel_ = std::forward<T>(ctrl_channel);
    if (auto_start_polling_) {
      startPolling();
    }
  }

  // Start polling thread
  void startPolling() {
    if (running_.load()) {
      return;
    }
    running_.store(true);
    poll_thread_ =
        std::make_unique<std::thread>(&RecvConnection::pollingLoop, this);
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

  int64_t recv(std::shared_ptr<RDMARecvRequest> req) {
    if (unlikely(!setupRecvRequestChannelAndMemoryRegion(req))) {
      UCCL_LOG(WARN)
          << "RecvConnection: Failed to setup recv request with round robin";
      return -1;
    }
    if (Compressor::getInstance().shouldCompress(req->local_mem->size)) {
      Compressor::getInstance().prepareDecompress(req);
    }
    return ctrl_channel_->postSendReq(req);
  }

  bool check(uint64_t index) { return ctrl_channel_->check_done(index); }

  void pollAndProcessCompletions() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (ctrl_channel_) {
      ctrl_channel_->noblockingPoll();
    }
    for (auto& [channel_id, channel] : channels_) {
      if (!channel) continue;
      bool polled = false;
      std::vector<CQMeta> cq_datas;
      polled = channel->pollOnce(cq_datas);
      if (polled) {
        for (auto const& cq_data : cq_datas) {
          if (cq_data.hasIMM()) {
            UCCL_LOG(INFO, UCCL_RDMA)
                << "RecvConnection::pollAndProcessCompletions - Channel "
                << channel_id << " polled completion: " << cq_data;
            if (ctrl_channel_) {
              std::shared_ptr<SendReqMeta> req_meta;
              for (int i = 0; i < cq_data.imm.chunk_count(); ++i) {
                req_meta = ctrl_channel_->recv_done(cq_data.imm.index());
                UCCL_LOG(INFO, UCCL_RDMA)
                    << "RecvConnection::pollAndProcessCompletions - Called "
                       "recv_done("
                    << cq_data.imm.index() << ")";
              }
              if (req_meta && Compressor::getInstance().shouldCompress(
                                  req_meta->local_mem.size)) {
                if (Compressor::getInstance().shouldCompressAndPipelineEncode(
                        req_meta->local_mem.size)) {
                  // Pipeline encode: decompress all blocks sequentially
                  size_t total_size = req_meta->local_mem.size;
                  size_t block_size =
                      CompressChunkSplitStrategy::kCompressionBlockSize;
                  size_t block_count =
                      CompressChunkSplitStrategy::getMessageChunkCount(
                          total_size);
                  size_t src_offset = 0;
                  for (size_t b = 0; b < block_count; ++b) {
                    size_t dst_offset = b * block_size;
                    size_t cur_block_size =
                        std::min(block_size, total_size - dst_offset);
                    RegMemBlock chunk_output(
                        static_cast<char*>(req_meta->local_mem.addr) +
                            dst_offset,
                        cur_block_size, req_meta->local_mem.mr_array,
                        req_meta->local_mem.type);
                    RemoteMemInfo chunk_input(
                        req_meta->remote_mem.addr + src_offset,
                        cur_block_size, req_meta->remote_mem.rkey_array,
                        req_meta->remote_mem.type);
                    Compressor::getInstance().decompress(
                        chunk_input, chunk_output, req_meta->float_type);
                    src_offset += cur_block_size;
                    dst_offset += cur_block_size;
                  }
                } else {
                  Compressor::getInstance().decompress(req_meta->remote_mem,
                                                       req_meta->local_mem,
                                                       req_meta->float_type);
                }
              }

            } else {
              UCCL_LOG(WARN) << "RecvConnection::pollAndProcessCompletions - "
                                "ctrl_channel_ is null, cannot call recv_done";
            }
          }
        }
      }
    }
    LOG_EVERY_N_ENDPOINT(INFO, 100000000)
        << "RecvConnection::pollingLoop - Still running, channels: "
        << channels_.size();
  }

  void pollingLoop() {
    UCCL_LOG(INFO, UCCL_RDMA) << "RecvConnection::pollingLoop - Started";
    uccl::pin_thread_to_numa(numa_node_);
    while (running_.load(std::memory_order_acquire)) {
      pollAndProcessCompletions();
      // optional small sleep/yield to avoid busy-looping if desired:
      // std::this_thread::yield();
    }
    UCCL_LOG(INFO, UCCL_RDMA) << "RecvConnection::pollingLoop - Stopped";
  }

 private:
  std::shared_ptr<RecvControlChannel> ctrl_channel_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  bool auto_start_polling_;
  int numa_node_ = 0;

  // Collect rkey for a specific channel
  // Returns: true on success, false on failure
  bool collectRkeyForChannel(
      int channel_id, std::unordered_map<int64_t, struct ibv_mr*> const& mr_map,
      uint32_t& rkey) {
    auto channel = getChannel(channel_id);
    if (unlikely(!channel)) {
      UCCL_LOG(WARN) << "RecvConnection: Channel not found for channel_id "
                     << channel_id;
      return false;
    }

    uint64_t context_id = channel->getContextID();
    auto it = mr_map.find(context_id);
    if (unlikely(it == mr_map.end())) {
      UCCL_LOG(WARN) << "RecvConnection: MR not found for context_id "
                     << context_id;
      return false;
    }

    rkey = it->second->rkey;
    return true;
  }

  // Round-robin channel selection and MR setup
  bool setupRecvRequestChannelAndMemoryRegion(
      std::shared_ptr<RDMARecvRequest> req) {
    if (unlikely(!req || !req->local_mem)) {
      return false;
    }

    // Select next channel using round-robin
    auto [channel_id, context_id] = selectNextChannelRoundRobin();
    // auto [channel_id, context_id] = selectNextChannelRandom();
    if (channel_id == 0) {
      return false;
    }
    req->channel_id = channel_id;
    return true;
  }
};

#pragma once
#include "define.h"
#include "rdma_channel.h"
#include "rdma_ctrl_channel.h"
#include <random>

class ChannelGroup {
 public:
  ChannelGroup() : last_channel_id_(0) {
    auto allocator = std::make_shared<MemoryAllocator>();
    buffer_compression_ = allocator->allocate(kCompressBufferSize,
                                           MemoryType::GPU, nullptr);
  }
  virtual ~ChannelGroup() = default;

  virtual void addChannel(uint32_t channel_id,
                          std::shared_ptr<RDMAChannel> channel) {
    if (!channel) {
      throw std::invalid_argument("addChannel called with null channel");
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto ctx_ptr = channel->getContext();
    if (!ctx_ptr) {
      throw std::invalid_argument("addChannel called with channel having null RdmaContext");
    }else{
      ctx_ptr->regMem(buffer_compression_->addr,
                      buffer_compression_->size);
      buffer_compression_->setMRByChannelID(
          channel_id,
          ctx_ptr->regMem(buffer_compression_->addr,
                          buffer_compression_->size));
    }
    channels_[channel_id] = std::move(channel);
  }

  virtual std::shared_ptr<RDMAChannel> getChannel(uint32_t channel_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = channels_.find(channel_id);
    if (it == channels_.end()) return nullptr;
    return it->second;
  }

  virtual size_t channelCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return channels_.size();
  }

  virtual std::unordered_map<uint32_t, std::shared_ptr<RDMAChannel>> const&
  channels() const {
    mutex_.lock_shared();
    mutex_.unlock_shared();  // just to annotate read lock expected
    return channels_;
  }

  // Select next channel using round-robin algorithm
  // Returns: pair<channel_id, context_id>, or pair<0, 0> on failure
  std::pair<uint32_t, uint64_t> selectNextChannelRoundRobin() {
    // Get the total number of channels
    size_t num_channels = ChannelGroup::channelCount();
    if (unlikely(num_channels == 0)) {
      LOG(WARNING)
          << "ChannelGroup: No channels available for round-robin selection";
      return {0, 0};
    }

    // Round-robin: get next channel_id
    uint32_t current_id = last_channel_id_.load(std::memory_order_relaxed);
    uint32_t next_id = (current_id) % num_channels + 1;
    last_channel_id_.store(next_id, std::memory_order_relaxed);

    // Get the channel by channel_id
    auto channel = getChannel(next_id);
    if (unlikely(!channel)) {
      LOG(WARNING) << "ChannelGroup: Channel not found for channel_id "
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
    size_t num_channels = ChannelGroup::channelCount();
    if (unlikely(num_channels == 0)) {
      LOG(WARNING)
          << "ChannelGroup: No channels available for random selection";
      return {0, 0};
    }

    // Random selection: generate random channel_id in range [1, num_channels]
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(1, num_channels);
    uint32_t random_id = dist(rng);

    // Get the channel by channel_id
    auto channel = getChannel(random_id);
    if (unlikely(!channel)) {
      LOG(WARNING) << "ChannelGroup: Channel not found for channel_id "
                   << random_id << " num_channels " << num_channels;
      return {0, 0};
    }

    uint64_t context_id = channel->getContextID();
    return {random_id, context_id};
  }

 protected:
  mutable std::shared_mutex mutex_;
  std::unordered_map<uint32_t, std::shared_ptr<RDMAChannel>> channels_;
  std::atomic<uint32_t> last_channel_id_;

  std::shared_ptr<RegMemBlock> buffer_compression_;

};

class SendChannelGroup : public ChannelGroup {
 public:
  SendChannelGroup(int numa_node, bool auto_start_polling = true)
      : numa_node_(numa_node),
        running_(false),
        poll_thread_(nullptr),
        auto_start_polling_(auto_start_polling) {
    tracker_ = std::make_shared<AtomicBitmapPacketTrackerMultiAck>();
    request_queue_ = std::make_unique<
        RingBuffer<std::shared_ptr<RDMASendRequest>, kRingCapacity>>();
    // Initialize CudaStream for compression
    compress_stream_ = std::make_shared<dietgpu::CudaStream>(
        dietgpu::CudaStream::make());
    // Initialize StackDeviceMemory for compression
    compress_res_ = std::make_shared<dietgpu::StackDeviceMemory>(
        dietgpu::makeStackMemory());
  }

  ~SendChannelGroup() { stopPolling(); }

  void addChannel(uint32_t channel_id,
                  std::shared_ptr<RDMAChannel> channel) override {
    ChannelGroup::addChannel(channel_id, channel);
  }
  std::shared_ptr<RDMAChannel> getChannel(uint32_t channel_id) const override {
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

  std::unordered_map<uint32_t, std::shared_ptr<RDMAChannel>> const& channels()
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
    if (auto_start_polling_) {
      startPolling();
    }
  }

  int64_t send(std::shared_ptr<RDMASendRequest> req) {
    int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
    req->wr_id = wr_id;
    if (unlikely(request_queue_->push(req) < 0)) {
      LOG(WARNING) << "SendChannelGroup: isend request queue is full, wr_id="
                   << wr_id;
      return -1;
    }
    return wr_id;
  }

  int64_t postWriteOrRead(std::shared_ptr<RDMASendRequest> req) {
    if (unlikely(req->send_type != SendType::Write &&
                 req->send_type != SendType::Read)) {
      LOG(ERROR) << "SendChannelGroup::write - Invalid send_type, expected "
                    "SendType::Write";
      return -1;
    }

    int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
    req->wr_id = wr_id;

    auto [channel_id, context_id] = selectNextChannelRoundRobin();
    if (unlikely(channel_id == 0)) {
      LOG(ERROR) << "SendChannelGroup::write - Failed to select channel";
      return -1;
    }

    req->channel_id = channel_id;
    postChunkedRequest(req);

    return wr_id;
  }

  int64_t read(std::shared_ptr<RDMASendRequest> req) {
    if (unlikely(req->send_type != SendType::Read)) {
      LOG(ERROR) << "SendChannelGroup::read - Invalid send_type, expected "
                    "SendType::Read";
      return -1;
    }

    int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
    req->wr_id = wr_id;

    auto [channel_id, context_id] = selectNextChannelRoundRobin();
    if (unlikely(channel_id == 0)) {
      LOG(ERROR) << "SendChannelGroup::read - Failed to select channel";
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

  void pollingLoopForMeta() {
    pollControlChannel();
    pollDataChannels();
    LOG_EVERY_N_ENDPOINT(INFO, 100000000)
        << "SendChannelGroup::pollingLoop - Still running";
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
    LOG(INFO) << "SendChannelGroup: Processing send request meta: " << meta;
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

  // Compression resources
  std::shared_ptr<dietgpu::CudaStream> compress_stream_;
  std::shared_ptr<dietgpu::StackDeviceMemory> compress_res_;

  // Send a request through the appropriate channel
  // Returns true on success, false on failure
  bool postRequestOnChannel(std::shared_ptr<RDMASendRequest> req) {
    auto channel = getChannel(req->channel_id);
    if (unlikely(!channel)) {
      LOG(WARNING) << "SendChannelGroup: Channel not found for channel_id "
                   << req->channel_id;
      return false;
    }

    int64_t send_ret = channel->submitRequest(req);
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
        LOG(INFO) << "SendChannelGroup::pollingLoop - Control channel polled "
                     "successfully";
      }
    }
  }

  void postChunkedRequest(std::shared_ptr<RDMASendRequest> req) {
    // Split message into chunks
    size_t message_size = req->local_mem->size;
    auto chunks = splitMessageToChunks(message_size);

    LOG(INFO) << "SendChannelGroup: Splitting message into " << chunks.size()
              << " chunks (message_size: " << message_size << ")";
    size_t num_channels = normalChannelCount();
    tracker_->updateExpectedAckCount(req->wr_id, chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
      auto const& chunk = chunks[i];

      // Use different channel for each chunk: round-robin
      uint32_t chunk_channel_id =
          ((req->channel_id - 1 + i) % num_channels) + 1;

      // Create RegMemBlock for this chunk
      auto chunk_local_mem = std::make_shared<RegMemBlock>(
          static_cast<char*>(req->local_mem->addr) + chunk.offset, chunk.size,
          req->local_mem->mr_array, req->local_mem->type);

      // Create RemoteMemInfo for this chunk
      auto chunk_remote_mem = std::make_shared<RemoteMemInfo>(
          req->remote_mem->addr + chunk.offset, chunk.size,
          req->remote_mem->rkey_array, req->remote_mem->type);

      // Create send request for this chunk
      // Only the last chunk needs signaled for completion notification
      bool is_last_chunk = (i == chunks.size() - 1);
      auto chunk_req = std::make_shared<RDMASendRequest>(
          chunk_local_mem, chunk_remote_mem, req->imm_data, is_last_chunk);
      chunk_req->channel_id = chunk_channel_id;
      chunk_req->from_rank_id = req->from_rank_id;
      chunk_req->to_rank_id = req->to_rank_id;
      chunk_req->wr_id = req->wr_id;
      // Inherit the send type from the original request.
      chunk_req->send_type = req->send_type;
      // Send the chunk
      if (postRequestOnChannel(chunk_req)) {
        LOG(INFO) << "SendChannelGroup: Sent chunk " << i << "/"
                  << chunks.size() << " (offset: " << chunk.offset
                  << ", size: " << chunk.size
                  << ", channel_id: " << chunk_channel_id << ")" << std::endl;
      } else {
        LOG(WARNING) << "SendChannelGroup: Failed to send chunk " << i
                     << " (offset: " << chunk.offset << ", size: " << chunk.size
                     << ", channel_id: " << chunk_channel_id << ")";
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
          LOG(WARNING) << "SendChannelGroup: In-flight bytes exceed "
                          "limit,pausing sending."
                       << tracker_->getTotalInflightBytes()
                       << " bytes in-flight.";
        }
        break;
      }
      index = ctrl_channel_->getOneSendRequestMeta(meta);
      LOG(INFO) << "SendChannelGroup: Processing send request meta: " << meta;
      processOnceSendRequests(req, meta, index);
      has_meta = ctrl_channel_->hasSendRequest();
    }
  }

  inline void compressSendRequest(std::shared_ptr<RDMASendRequest> req) {
    if (!req || !compress_stream_ || !compress_res_) {
      return;
    }

    // Setup compression config
    dietgpu::FloatCompressConfig compressConfig;
    compressConfig.floatType = req->float_type;
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    // Calculate element count from bytes
    uint32_t numFloats = getElementCountFromBytes(
        req->float_type, req->local_mem->size);

    // Setup batch (single element batch)
    const void* inPtrs[1] = {req->local_mem->addr};
    uint32_t inSizes[1] = {numFloats};
    void* outPtrs[1] = {buffer_compression_->addr};

    // Allocate device memory for compressed size output
    uint32_t* devCompressedSize = nullptr;
    GPU_CHECK(hipMalloc(&devCompressedSize, sizeof(uint32_t)));

    // Compress
    dietgpu::floatCompress(
        *compress_res_,
        compressConfig,
        1,  // numInBatch
        inPtrs,
        inSizes,
        outPtrs,
        devCompressedSize,
        compress_stream_->get());

    // Get compressed size
    uint32_t compressedSize = 0;
    GPU_CHECK(hipMemcpyAsync(&compressedSize, devCompressedSize,
                              sizeof(uint32_t), hipMemcpyDeviceToHost,
                              compress_stream_->get()));
    GPU_CHECK(hipStreamSynchronize(compress_stream_->get()));

    LOG(INFO) << "SendChannelGroup: Compressed " << req->local_mem->size
              << " bytes to " << compressedSize << " bytes, ratio: "
              << static_cast<float>(req->local_mem->size) / compressedSize << "x";

    // Update request to use compressed buffer
    req->local_mem = std::make_shared<RegMemBlock>(
        buffer_compression_->addr, compressedSize,
        buffer_compression_->mr_array, buffer_compression_->type);

    // Cleanup
    GPU_CHECK(hipFree(devCompressedSize));
  }

  inline void processOnceSendRequests(std::shared_ptr<RDMASendRequest> req,
                                      SendReqMeta& meta, int index) {
    req->imm_data = index;
    req->channel_id = meta.channel_id;
    req->remote_mem = std::make_shared<RemoteMemInfo>(meta.remote_mem);
    if (req->local_mem->size >= kMinCompressBytes) {
      compressSendRequest(req);
    }
    if (meta.expected_chunk_count > 1) {
      postChunkedRequest(req);
    } else {
      postRequestOnChannel(req);
    }
  }

  void pollDataChannels() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (auto& [channel_id, channel] : channels_) {
      std::vector<CQMeta> cq_datas;
      if (channel && channel->poll_once(cq_datas)) {
        for (auto const& cq_data : cq_datas) {
          LOG(INFO) << "SendChannelGroup::pollingLoop - Channel " << channel_id
                    << " polled completion: " << cq_data;
          tracker_->acknowledge(cq_data.wr_id);
        }
      }
    }
  }

  void pollingLoop() {
    LOG(INFO) << "SendChannelGroup::pollingLoop - Started";
    uccl::pin_thread_to_numa(numa_node_);
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
  RecvChannelGroup(int numa_node, bool auto_start_polling = true)
      : numa_node_(numa_node),
        running_(false),
        poll_thread_(nullptr),
        auto_start_polling_(auto_start_polling) {
    // Initialize CudaStream for decompression
    decompress_stream_ = std::make_shared<dietgpu::CudaStream>(
        dietgpu::CudaStream::make());
    // Initialize StackDeviceMemory for decompression
    decompress_res_ = std::make_shared<dietgpu::StackDeviceMemory>(
        dietgpu::makeStackMemory());
  }

  ~RecvChannelGroup() { stopPolling(); }

  void addChannel(uint32_t channel_id,
                  std::shared_ptr<RDMAChannel> channel) override {
    ChannelGroup::addChannel(channel_id, channel);
  }

  std::shared_ptr<RDMAChannel> getChannel(uint32_t channel_id) const override {
    auto result = ChannelGroup::getChannel(channel_id);
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

  std::unordered_map<uint32_t, std::shared_ptr<RDMAChannel>> const& channels()
      const override {
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

  int64_t recv(std::shared_ptr<RDMARecvRequest> req) {
    if (unlikely(!setupRecvRequestChannelAndMemoryRegion(req))) {
      LOG(WARNING)
          << "RecvChannelGroup: Failed to setup recv request with round robin";
      return -1;
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
      polled = channel->poll_once(cq_datas);
      if (polled) {
        for (auto const& cq_data : cq_datas) {
          if (cq_data.hasIMM()) {
            LOG(INFO)
                << "RecvChannelGroup::pollAndProcessCompletions - Channel "
                << channel_id << " polled completion: " << cq_data;
            if (ctrl_channel_) {
              ctrl_channel_->recv_done(cq_data.imm);
              LOG(INFO)
                  << "RecvChannelGroup::pollAndProcessCompletions - Called "
                     "recv_done("
                  << cq_data.imm << ")";
            } else {
              LOG(WARNING) << "RecvChannelGroup::pollAndProcessCompletions - "
                              "ctrl_channel_ is null, cannot call recv_done";
            }
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
    uccl::pin_thread_to_numa(numa_node_);
    while (running_.load(std::memory_order_acquire)) {
      pollAndProcessCompletions();
      // optional small sleep/yield to avoid busy-looping if desired:
      // std::this_thread::yield();
    }
    LOG(INFO) << "RecvChannelGroup::pollingLoop - Stopped";
  }

 private:
  std::shared_ptr<RecvControlChannel> ctrl_channel_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  bool auto_start_polling_;
  int numa_node_ = 0;

  // Decompression resources
  std::shared_ptr<dietgpu::CudaStream> decompress_stream_;
  std::shared_ptr<dietgpu::StackDeviceMemory> decompress_res_;

  inline void decompressRecvRequest(std::shared_ptr<RDMARecvRequest> req) {
    if (unlikely(!req) || !decompress_stream_ || !decompress_res_) {
      return;
    }

    // Setup decompression config
    dietgpu::FloatDecompressConfig decompressConfig;
    decompressConfig.floatType = req->float_type;
    decompressConfig.useChecksum = false;
    decompressConfig.is16ByteAligned = true;

    // Calculate element count from bytes for output capacity
    uint32_t numFloats = getElementCountFromBytes(req->float_type, req->local_mem->size);

    // Setup batch for decompression
    // Input is the compressed data in buffer_compression_
    // Output is the original local_mem buffer
    const void* compInPtrs[1] = {buffer_compression_->addr};
    void* decompOutPtrs[1] = {req->local_mem->addr};
    uint32_t outCapacities[1] = {numFloats};

    // Decompress
    dietgpu::FloatDecompressStatus status = dietgpu::floatDecompress(
        *decompress_res_,
        decompressConfig,
        1,  // numInBatch
        compInPtrs,
        decompOutPtrs,
        outCapacities,
        nullptr,  // outSuccess_dev (optional)
        nullptr,  // outSize_dev (optional)
        decompress_stream_->get());

    GPU_CHECK(hipStreamSynchronize(decompress_stream_->get()));

    if (status.error != dietgpu::FloatDecompressError::None) {
      LOG(ERROR) << "RecvChannelGroup: Decompression failed!";
      return;
    }

    LOG(INFO) << "RecvChannelGroup: Decompressed data to "
              << req->local_mem->size << " bytes";
  }

  // Collect rkey for a specific channel
  // Returns: true on success, false on failure
  bool collectRkeyForChannel(
      int channel_id, std::unordered_map<int64_t, struct ibv_mr*> const& mr_map,
      uint32_t& rkey) {
    auto channel = getChannel(channel_id);
    if (unlikely(!channel)) {
      LOG(WARNING) << "RecvChannelGroup: Channel not found for channel_id "
                   << channel_id;
      return false;
    }

    uint64_t context_id = channel->getContextID();
    auto it = mr_map.find(context_id);
    if (unlikely(it == mr_map.end())) {
      LOG(WARNING) << "RecvChannelGroup: MR not found for context_id "
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

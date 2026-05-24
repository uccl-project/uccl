#include "rdma_connection.h"
#include "util/debug.h"
#include "util/util.h"

RDMAConnection::RDMAConnection() : last_channel_id_(0) {}

RDMAConnection::~RDMAConnection() = default;

void RDMAConnection::addChannel(uint32_t channel_id,
                                std::shared_ptr<RDMADataChannel> channel) {
  if (!channel) {
    throw std::invalid_argument("addChannel called with null channel");
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);

  channels_[channel_id] = std::move(channel);
}

std::shared_ptr<RDMADataChannel> RDMAConnection::getChannel(
    uint32_t channel_id) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = channels_.find(channel_id);
  if (it == channels_.end()) return nullptr;
  return it->second;
}

size_t RDMAConnection::channelCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return channels_.size();
}

std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
RDMAConnection::channels() const {
  mutex_.lock_shared();
  mutex_.unlock_shared();  // just to annotate read lock expected
  return channels_;
}

std::pair<uint32_t, uint64_t> RDMAConnection::selectNextChannelRoundRobin() {
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

std::pair<uint32_t, uint64_t> RDMAConnection::selectNextChannelRandom() {
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

RDMADataChannel* RDMAConnection::getChannelFast(uint32_t channel_id) const {
  if (likely(fast_channels_ready_.load(std::memory_order_acquire))) {
    if (likely(channel_id >= 1 &&
               channel_id <=
                   fast_channel_count_.load(std::memory_order_relaxed))) {
      return fast_channels_[channel_id - 1];
    }
  }
  auto sp = getChannel(channel_id);
  return sp.get();
}

std::pair<uint32_t, RDMADataChannel*>
RDMAConnection::selectNextChannelRoundRobinFast() {
  if (unlikely(!fast_channels_ready_.load(std::memory_order_acquire))) {
    buildFastChannelCache();
  }
  size_t n = fast_channel_count_.load(std::memory_order_relaxed);
  if (unlikely(n == 0)) return {0, nullptr};
  uint32_t prev = last_channel_id_.load(std::memory_order_relaxed);
  uint32_t next = (prev % n) + 1;
  last_channel_id_.store(next, std::memory_order_relaxed);
  return {next, fast_channels_[next - 1]};
}

void RDMAConnection::buildFastChannelCache() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  size_t n = channels_.size();
  fast_channels_.assign(n, nullptr);
  for (auto const& [cid, ch] : channels_) {
    if (cid >= 1 && cid <= n) fast_channels_[cid - 1] = ch.get();
  }
  fast_channel_count_.store(n, std::memory_order_release);
  fast_channels_ready_.store(true, std::memory_order_release);
}

SendConnection::SendConnection(int numa_node, bool auto_start_polling,
                               double link_bandwidth_bps)
    : numa_node_(numa_node),
      running_(false),
      poll_thread_(nullptr),
      auto_start_polling_(auto_start_polling),
      cc_(uccl::cc::CongestionControlState::parseMode("UCCL_P2P_RDMA_CC"),
          uccl::freq_ghz, link_bandwidth_bps) {
  tracker_ = std::make_shared<AtomicBitmapPacketTrackerMultiAck>();
  request_queue_ = std::make_unique<
      RingBuffer<std::shared_ptr<RDMASendRequest>, kRingCapacity>>();
}

SendConnection::~SendConnection() { stopPolling(); }

void SendConnection::addChannel(uint32_t channel_id,
                                std::shared_ptr<RDMADataChannel> channel) {
  RDMAConnection::addChannel(channel_id, channel);
}

std::shared_ptr<RDMADataChannel> SendConnection::getChannel(
    uint32_t channel_id) const {
  auto result = RDMAConnection::getChannel(channel_id);
  return result;
}

size_t SendConnection::channelCount() const {
  auto result = RDMAConnection::channelCount();
  std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
  if (ctrl_channel_) {
    result += 1;
  }
  return result;
}

size_t SendConnection::normalChannelCount() const {
  return RDMAConnection::channelCount();
}

std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
SendConnection::channels() const {
  return RDMAConnection::channels();
}

int64_t SendConnection::send(std::shared_ptr<RDMASendRequest> req) {
  // Allocate seq_num without counting bytes — actual size is registered
  // later by the polling thread after popping from the queue, so that
  // getTotalInflightBytes() only reflects requests actually being sent.
  int64_t wr_id = tracker_->sendPacket(0);
  req->wr_id = wr_id;
  while (unlikely(request_queue_->push(req) < 0)) {
    UCCL_LOG(WARN) << "SendConnection: isend request queue is full, wr_id="
                   << wr_id;
    std::this_thread::yield();
  }
  return wr_id;
}

int64_t SendConnection::postWriteOrRead(std::shared_ptr<RDMASendRequest> req) {
  if (unlikely(req->send_type != SendType::Write &&
               req->send_type != SendType::Read)) {
    UCCL_LOG(ERROR) << "SendConnection::write - Invalid send_type, expected "
                       "SendType::Write";
    return -1;
  }

  // Enforce CC window before posting
  if (cc_.enabled()) {
    size_t inflight_limit_bytes = currentInflightLimitBytes();
    while (currentInflightBytes() > inflight_limit_bytes) {
      std::this_thread::yield();
      inflight_limit_bytes = currentInflightLimitBytes();
    }
  }

  std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
  int64_t wr_id = tracker_->sendPacket(req->getLocalLen());
  req->wr_id = wr_id;

  // Use compressed write path when all preconditions are met.
  bool c_send_type = (req->send_type == SendType::Write);
  bool c_ack = static_cast<bool>(ack_ring_);
  bool c_rdb = remote_decompress_buf_.length > 0;
  bool c_ctx = static_cast<bool>(req->compress_ctx);
  bool c_ft = c_ctx && (req->compress_ctx->getFloatType() !=
                        uccl::FloatType::kUndefined);
  bool c_fit = static_cast<uint64_t>(req->local_mem->size) <=
               remote_decompress_buf_.length;
  bool c_should = Compressor::getInstance().shouldCompressAndSplitFirst(
      req->local_mem->size);
  UCCL_LOG(INFO, UCCL_RDMA)
      << "compressedWriteCheck size=" << req->local_mem->size
      << " send_type=" << c_send_type << " ack=" << c_ack
      << " rdb_len=" << remote_decompress_buf_.length << " (>" << 0 << ")"
      << " ctx=" << c_ctx << " ft_ok=" << c_ft << " fit=" << c_fit
      << " should=" << c_should;
  if (c_send_type && c_ack && c_rdb && c_ctx && c_ft && c_fit && c_should) {
    return compressWriteRequestSplitFirst(req);
  }

  // Lock-free channel selection + direct pointer.
  auto [channel_id, ch_ptr] = selectNextChannelRoundRobinFast();
  if (unlikely(channel_id == 0 || ch_ptr == nullptr)) {
    UCCL_LOG(ERROR) << "SendConnection::write - Failed to select channel";
    return -1;
  }

  req->channel_id = channel_id;

  // Fast single-chunk path: skip postChunkedRequest/postSingleChunk and
  // submit directly to the resolved channel pointer.
  if (likely(ChunkSplitStrategy::getMessageChunkCount(req->local_mem->size) ==
             1)) {
    req->imm_data.set_chunk_count(1);

    int64_t saved_wr_id = req->wr_id;
    if (cc_.enabled()) {
      uint32_t tsc_id =
          chunk_tsc_counter_.fetch_add(1, std::memory_order_relaxed);
      req->wr_id = (static_cast<int64_t>(tsc_id) << 32) |
                   static_cast<uint32_t>(req->wr_id);
      cc_.recordSendTsc(tsc_id);
    }

    int64_t send_ret = ch_ptr->submitRequest(req);
    req->wr_id = saved_wr_id;

    if (send_ret >= 0 && cc_.enabled()) {
      cc_inflight_bytes_.fetch_add(req->getLocalLen(),
                                   std::memory_order_relaxed);
    }
    return wr_id;
  }

  postChunkedRequest(req);

  // Since postChunkedRequest() is non-blocking — if the CC
  // window is exhausted mid-message it saves the remaining chunks
  // and returns immediately.
  // Draining them here.
  while (!drainPendingChunks()) {
    std::this_thread::yield();
  }

  return wr_id;
}

int64_t SendConnection::read(std::shared_ptr<RDMASendRequest> req) {
  if (unlikely(req->send_type != SendType::Read)) {
    UCCL_LOG(ERROR) << "SendConnection::read - Invalid send_type, expected "
                       "SendType::Read";
    return -1;
  }

  // Enforce CC window before posting
  if (cc_.enabled()) {
    size_t inflight_limit_bytes = currentInflightLimitBytes();
    while (currentInflightBytes() > inflight_limit_bytes) {
      std::this_thread::yield();
      inflight_limit_bytes = currentInflightLimitBytes();
    }
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

  // Draining any remaining chunks, as in postWriteOrRead()
  while (!drainPendingChunks()) {
    std::this_thread::yield();
  }

  return wr_id;
}

void SendConnection::startPolling() {
  if (running_.load()) {
    return;
  }
  running_.store(true);
  poll_thread_ =
      std::make_unique<std::thread>(&SendConnection::pollingLoop, this);
}

bool SendConnection::check(int64_t wr_id) {
  return tracker_->isAcknowledged(wr_id);
}

bool SendConnection::canUseRawOneSidedBatch(SendType send_type) {
  if (cc_.enabled()) return false;
  if (send_type == SendType::Write &&
      Compressor::getInstance().getCompressStrategy() !=
          CompressStrategy::kNone) {
    return false;
  }
  return true;
}

bool SendConnection::postWriteOrReadBatch(SendType send_type,
                                          OneSidedBatchOp const* ops,
                                          size_t num_ops, int64_t* wr_ids) {
  if (unlikely(send_type != SendType::Write && send_type != SendType::Read)) {
    UCCL_LOG(ERROR) << "SendConnection::postWriteOrReadBatch - invalid "
                       "send_type";
    return false;
  }
  if (unlikely(!canUseRawOneSidedBatch(send_type))) return false;
  if (unlikely((num_ops > 0 && ops == nullptr) || wr_ids == nullptr)) {
    UCCL_LOG(ERROR) << "SendConnection::postWriteOrReadBatch - null input";
    return false;
  }

  size_t const num_channels = normalChannelCount();
  if (unlikely(num_channels == 0 || num_channels > kQpNumPerChannel)) {
    UCCL_LOG(ERROR) << "SendConnection::postWriteOrReadBatch - invalid "
                       "channel count: "
                    << num_channels;
    return false;
  }

  for (size_t i = 0; i < num_ops; ++i) {
    auto const& op = ops[i];
    if (unlikely(op.local_addr == nullptr || op.size == 0 ||
                 op.size > std::numeric_limits<uint32_t>::max() ||
                 op.local_mr_array == nullptr || op.slot_item == nullptr)) {
      UCCL_LOG(ERROR) << "SendConnection::postWriteOrReadBatch - invalid op "
                      << i;
      return false;
    }
    if (unlikely(ChunkSplitStrategy::getMessageChunkCount(op.size) != 1)) {
      UCCL_LOG(ERROR) << "SendConnection::postWriteOrReadBatch - op " << i
                      << " requires chunking";
      return false;
    }
  }

  struct PreparedBatchOp {
    size_t op_index = 0;
    uint32_t channel_id = 0;
    RDMADataChannel* channel = nullptr;
    uint32_t local_key = 0;
    uint32_t remote_key = 0;
  };

  std::vector<PreparedBatchOp> prepared_ops;
  prepared_ops.reserve(num_ops);
  for (size_t i = 0; i < num_ops; ++i) {
    auto const& op = ops[i];
    auto [channel_id, ch_ptr] = selectNextChannelRoundRobinFast();
    if (unlikely(channel_id == 0 || ch_ptr == nullptr ||
                 channel_id > num_channels || channel_id > kQpNumPerChannel)) {
      UCCL_LOG(ERROR) << "SendConnection::postWriteOrReadBatch - failed to "
                         "select channel";
      return false;
    }

    auto* local_mr = op.local_mr_array->getKeyByChannelID(channel_id);
    if (unlikely(local_mr == nullptr)) {
      UCCL_LOG(ERROR) << "SendConnection::postWriteOrReadBatch - missing "
                         "local MR for op "
                      << i << " channel " << channel_id;
      return false;
    }

    RKeyArray remote_rkeys;
    remote_rkeys.copyFrom(static_cast<char const*>(op.slot_item->padding));

    PreparedBatchOp prepared{};
    prepared.op_index = i;
    prepared.channel_id = channel_id;
    prepared.channel = ch_ptr;
    prepared.local_key = local_mr->lkey;
    prepared.remote_key = remote_rkeys.getKeyByChannelID(channel_id);
    prepared_ops.push_back(prepared);
  }

  for (size_t i = 0; i < num_ops; ++i) wr_ids[i] = -1;
  std::array<std::vector<RDMADataChannel::RawSendRequest>, kQpNumPerChannel>
      channel_batches;
  std::array<RDMADataChannel*, kQpNumPerChannel> channel_ptrs{};
  size_t const reserve_per_channel =
      (num_ops + num_channels - 1) / num_channels;
  for (size_t i = 0; i < num_channels; ++i) {
    channel_batches[i].reserve(reserve_per_channel);
  }

  for (auto const& prepared : prepared_ops) {
    auto const& op = ops[prepared.op_index];

    int64_t wr_id = tracker_->sendPacket(op.size);
    if (unlikely(wr_id < 0)) {
      UCCL_LOG(ERROR) << "SendConnection::postWriteOrReadBatch - failed to "
                         "allocate tracker wr_id for op "
                      << prepared.op_index;
      return false;
    }
    wr_ids[prepared.op_index] = wr_id;

    RDMADataChannel::RawSendRequest raw{};
    raw.send_type = send_type;
    raw.wr_id = static_cast<uint64_t>(wr_id);
    raw.local_addr = reinterpret_cast<uint64_t>(op.local_addr);
    raw.local_len = static_cast<uint32_t>(op.size);
    raw.local_key = prepared.local_key;
    raw.remote_addr = op.slot_item->addr;
    raw.remote_key = prepared.remote_key;

    size_t const channel_idx = prepared.channel_id - 1;
    channel_ptrs[channel_idx] = prepared.channel;
    channel_batches[channel_idx].push_back(raw);
  }

  for (size_t i = 0; i < num_channels; ++i) {
    if (channel_batches[i].empty()) continue;
    if (unlikely(channel_ptrs[i] == nullptr ||
                 channel_ptrs[i]->postRawBatch(channel_batches[i]) != 0)) {
      UCCL_LOG(FATAL) << "SendConnection::postWriteOrReadBatch - failed to "
                         "post channel batch "
                      << (i + 1)
                      << "; raw one-sided batch post failure is "
                         "unrecoverable after tracker wr_ids are allocated";
      return false;
    }
  }
  return true;
}

void SendConnection::setRemoteDecompressBuf(RemoteMemInfo const& m) {
  if (m.length == 0) return;
  remote_decompress_buf_ = m;
  decompress_arena_.size = m.length;
  if (!running_.load(std::memory_order_acquire)) startPolling();
}

void SendConnection::setLocalAckRing(std::shared_ptr<RegMemBlock> ring) {
  ack_ring_ = ring;
}

void SendConnection::stopPolling() {
  if (!running_.load()) {
    return;
  }
  running_.store(false);
  if (poll_thread_ && poll_thread_->joinable()) {
    poll_thread_->join();
  }
}

void SendConnection::pollingLoopForMeta() {
  pollControlChannel();
  pollDataChannels();
  UCCL_LOG_EVERY_N(INFO, UCCL_RDMA, 100000000)
      << "SendConnection::pollingLoop - Still running";
}

int SendConnection::processSendRequests(std::shared_ptr<RDMASendRequest> req) {
  pollControlChannel();
  if (unlikely(ctrl_channel_ == nullptr)) {
    return -1;
  }
  // Enforce CC window before accepting a new request
  size_t inflight_limit_bytes = currentInflightLimitBytes();
  if (currentInflightBytes() > inflight_limit_bytes) {
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

void SendConnection::flushBatches() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  for (auto& [cid, channel] : channels_) {
    if (channel) channel->flushBatch();
  }
}

size_t SendConnection::currentInflightLimitBytes() {
  return cc_.enabled() ? cc_.getWindowBytes() : kInFlightMaxSizeKB * 1024;
}

size_t SendConnection::currentInflightBytes() {
  return cc_.enabled() ? cc_inflight_bytes_.load(std::memory_order_relaxed)
                       : tracker_->getTotalInflightBytes();
}

bool SendConnection::postRequestOnChannel(
    std::shared_ptr<RDMASendRequest> req) {
  auto channel = getChannel(req->channel_id);
  if (unlikely(!channel)) {
    UCCL_LOG(WARN) << "SendConnection: Channel not found for channel_id "
                   << req->channel_id;
    return false;
  }

  // Per-chunk CC: assign a unique TSC ID and record send timestamp
  // close to the actual ibv_post_send.  The TSC ID is encoded in the
  // upper 32 bits of wr_id; the lower 32 bits keep the message seq
  // used by the tracker.  We save/restore req->wr_id so that callers
  // (e.g. updateExpectedAckCount) still see the original message seq.
  int64_t saved_wr_id = req->wr_id;
  if (cc_.enabled()) {
    uint32_t tsc_id =
        chunk_tsc_counter_.fetch_add(1, std::memory_order_relaxed);
    req->wr_id = (static_cast<int64_t>(tsc_id) << 32) |
                 static_cast<uint32_t>(req->wr_id);
    cc_.recordSendTsc(tsc_id);
  }

  int64_t send_ret = channel->submitRequest(req);
  req->wr_id = saved_wr_id;
  if (send_ret >= 0 && cc_.enabled()) {
    cc_inflight_bytes_.fetch_add(req->getLocalLen(), std::memory_order_relaxed);
  }
  if (send_ret < 0) {
    UCCL_LOG(WARN) << "SendConnection: Failed to send on channel_id "
                   << req->channel_id;
    return false;
  }

  return true;
}

void SendConnection::pollControlChannel() {
  std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
  if (ctrl_channel_) {
    if (ctrl_channel_->noblockingPoll()) {
      UCCL_LOG(INFO, UCCL_RDMA)
          << "SendConnection::pollingLoop - Control channel polled "
             "successfully";
    }
  }
}

bool SendConnection::postSingleChunk(
    std::shared_ptr<RDMASendRequest> const& req, MessageChunk const& chunk,
    size_t chunk_index, size_t total_chunks, size_t num_channels,
    int& expected_chunk_count) {
  uint32_t chunk_channel_id =
      ((req->channel_id - 1 + chunk_index) % num_channels) + 1;

  auto chunk_local_mem = std::make_shared<RegMemBlock>(
      static_cast<char*>(req->local_mem->addr) + chunk.offset, chunk.size,
      req->local_mem->mr_array, req->local_mem->type);

  auto chunk_remote_mem = std::make_shared<RemoteMemInfo>(
      req->remote_mem->addr + chunk.offset, chunk.size,
      req->remote_mem->rkey_array, req->remote_mem->type);

  bool is_last_chunk = (chunk_index == total_chunks - 1);
  auto chunk_req = std::make_shared<RDMASendRequest>(
      chunk_local_mem, chunk_remote_mem, req->imm_data, is_last_chunk);

  // Due to compression, the chunk count may differ from the original
  // split, so set the expected chunk count for each chunk request.
  if (expected_chunk_count > 0) {
    if (is_last_chunk && expected_chunk_count > 1) {
      chunk_req->imm_data.set_chunk_count(expected_chunk_count);
    } else {
      chunk_req->imm_data.set_chunk_count(1);
    }
    expected_chunk_count -= 1;
  }

  chunk_req->channel_id = chunk_channel_id;
  chunk_req->from_rank_id = req->from_rank_id;
  chunk_req->to_rank_id = req->to_rank_id;
  chunk_req->wr_id = req->wr_id;
  chunk_req->send_type = req->send_type;

  return postRequestOnChannel(chunk_req);
}

bool SendConnection::drainPendingChunks() {
  if (!pending_chunked_) return true;

  auto& ps = *pending_chunked_;
  size_t num_channels = normalChannelCount();

  while (ps.next_chunk_idx < ps.chunks.size()) {
    // Per-chunk CC: check window before each chunk.
    if (cc_.enabled()) {
      size_t inflight_limit_bytes = currentInflightLimitBytes();
      if (currentInflightBytes() > inflight_limit_bytes) {
        return false;  // Yield back to polling loop.
      }
    }

    if (!postSingleChunk(ps.req, ps.chunks[ps.next_chunk_idx],
                         ps.next_chunk_idx, ps.chunks.size(), num_channels,
                         ps.remaining_expected_count)) {
      UCCL_LOG(WARN) << "SendConnection: Failed to send pending chunk "
                     << ps.next_chunk_idx;
    }
    ps.next_chunk_idx++;
  }

  pending_chunked_.reset();
  return true;
}

void SendConnection::postChunkedRequest(std::shared_ptr<RDMASendRequest> req,
                                        int expected_chunk_count) {
  // Fast path: single-chunk message. The default caller passes
  // expected_chunk_count=0, in which case we compute chunk count from the
  // message size; for messages that fit in a single chunk we post `req`
  // directly and skip the chunk-wrapper allocations done by
  // postSingleChunk().
  if (expected_chunk_count == 1 ||
      (expected_chunk_count == 0 &&
       ChunkSplitStrategy::getMessageChunkCount(req->local_mem->size) == 1)) {
    req->imm_data.set_chunk_count(1);
    if (!postRequestOnChannel(req)) {
      UCCL_LOG(WARN) << "SendConnection: Failed to send request on channel_id "
                     << req->channel_id;
    }
    return;
  }
  // Split message into chunks
  size_t message_size = req->local_mem->size;
  auto chunks = ChunkSplitStrategy::splitMessageToChunks(message_size);
  if (expected_chunk_count == 0) {
    expected_chunk_count = chunks.size();
    tracker_->updateExpectedAckCount(req->wr_id, expected_chunk_count);
  }
  UCCL_LOG(INFO, UCCL_RDMA)
      << "SendConnection: Splitting message into " << chunks.size()
      << " chunks (message_size: " << message_size << ")";
  size_t num_channels = normalChannelCount();

  for (size_t i = 0; i < chunks.size(); ++i) {
    // Per-chunk CC: if over budget, save remaining chunks and return.
    if (cc_.enabled()) {
      size_t inflight_limit_bytes = currentInflightLimitBytes();
      if (currentInflightBytes() > inflight_limit_bytes) {
        pending_chunked_ = PendingChunkedState{req, std::move(chunks), i,
                                               expected_chunk_count};
        return;
      }
    }

    if (!postSingleChunk(req, chunks[i], i, chunks.size(), num_channels,
                         expected_chunk_count)) {
      UCCL_LOG(WARN) << "SendConnection: Failed to send chunk " << i;
    }
  }
}

void SendConnection::processSendRequests() {
  if (unlikely(ctrl_channel_ == nullptr)) {
    return;
  }

  // First, try to drain any pending chunks from a previous request
  // that was paused due to CC window limits.
  if (!drainPendingChunks()) {
    return;  // Still CC-blocked, don't dequeue new requests.
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
    size_t inflight_limit_bytes = currentInflightLimitBytes();
    if (currentInflightBytes() > inflight_limit_bytes ||
        !request_queue_->pop(req)) {
      if (currentInflightBytes() > inflight_limit_bytes) {
        UCCL_LOG(WARN) << "SendConnection: In-flight bytes exceed "
                          "limit,pausing sending."
                       << currentInflightBytes() << " bytes in-flight.";
      }
      break;
    }
    // Register actual packet size now that we are about to send it.
    tracker_->updatePacketSize(req->wr_id, req->getLocalLen());
    index = ctrl_channel_->getOneSendRequestMeta(meta);
    UCCL_LOG(INFO, UCCL_RDMA)
        << "SendConnection: Processing send request meta: " << meta;
    processOnceSendRequests(req, meta, index);
    has_meta = ctrl_channel_->hasSendRequest();
  }
}

void SendConnection::compressSendRequest(std::shared_ptr<RDMASendRequest> req) {
  Compressor::getInstance().compress(req);
}

void SendConnection::postCompressedSegment(
    std::shared_ptr<RDMASendRequest> const& req, size_t seg_size,
    size_t num_chunks, size_t num_channels) {
  if (num_chunks == 0 || seg_size == 0 || num_channels == 0) return;
  size_t chunk_size = (seg_size + num_chunks - 1) / num_chunks;
  int dummy_expected = static_cast<int>(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    uint64_t offset = static_cast<uint64_t>(i) * chunk_size;
    if (offset >= seg_size) break;
    size_t this_size = std::min(chunk_size, seg_size - offset);
    MessageChunk chunk(offset, this_size);
    if (!postSingleChunk(req, chunk, i, num_chunks, num_channels,
                         dummy_expected)) {
      UCCL_LOG(WARN) << "postCompressedSegment: chunk " << i << " post failed";
    }
  }
}

int64_t SendConnection::compressWriteRequestSplitFirst(
    std::shared_ptr<RDMASendRequest> req) {
  int64_t wr_id = req->wr_id;
  uint32_t total_uncomp = req->local_mem->size;
  uint64_t user_remote_addr = req->remote_mem->addr;
  uint32_t user_remote_rkey =
      req->remote_mem->getKeyByContextID(0);  // diagnostic / logging only

  // Reserve space in the peer's decompress_buffer; released on ack receipt.
  uint64_t arena_bytes = static_cast<uint64_t>(total_uncomp);
  uint64_t arena_offset = decompress_arena_.reserve(arena_bytes);
  uint32_t ack_slot = static_cast<uint32_t>(
      next_ack_slot_.fetch_add(1, std::memory_order_relaxed) % kAckRingDepth);
  static_cast<AckSlot*>(ack_ring_->addr)[ack_slot].value.store(
      0, std::memory_order_relaxed);
  uint64_t slot_base = remote_decompress_buf_.addr + arena_offset;

  // Phase 1: split into compress_buffer, post to slot_base.
  Compressor::getInstance().compressSplitOneBatch(req);
  uint32_t first_seg = req->local_mem->size;
  req->remote_mem = std::make_shared<RemoteMemInfo>(
      slot_base, arena_bytes, remote_decompress_buf_.rkey_array,
      remote_decompress_buf_.type);

  auto [channel_id, _ch_ptr] = selectNextChannelRoundRobinFast();
  req->channel_id = channel_id ? channel_id : 1;
  size_t num_channels = normalChannelCount();
  if (num_channels == 0) num_channels = 1;
  // 2 chunks per channel keeps WR density similar to non-compressed writes.
  size_t first_chunks = std::min<size_t>(
      num_channels * 2, static_cast<size_t>(ChunkSplitStrategy::kMaxSplitNum));
  if (first_chunks > first_seg) first_chunks = first_seg > 0 ? 1 : 0;
  PendingCompressed entry{};
  entry.ack_slot = ack_slot;
  entry.arena_offset = arena_offset;
  entry.arena_bytes = arena_bytes;
  entry.meta.user_remote_addr = user_remote_addr;
  entry.meta.user_rkey = user_remote_rkey;
  entry.meta.total_uncomp_size = total_uncomp;
  entry.meta.decompress_offset = arena_offset;
  entry.meta.float_type = static_cast<uint32_t>(
      req->compress_ctx ? req->compress_ctx->getFloatType()
                        : uccl::FloatType::kUndefined);
  entry.meta.ack_slot = ack_slot;
  entry.meta.wr_id = wr_id;
  {
    std::lock_guard<std::mutex> lk(pending_compressed_mu_);
    pending_compressed_.emplace(wr_id, entry);
    pending_compressed_count_.fetch_add(1, std::memory_order_relaxed);
  }
  tracker_->updateExpectedAckCount(wr_id, std::numeric_limits<uint32_t>::max());

  // No DeferGuard: per-chunk auto-flush gives the round-robin interleaving
  // required on Broadcom bnxt_re (batching causes LOC_QP_OP_ERR vendor=0x3).
  postCompressedSegment(req, first_seg, first_chunks, num_channels);
  flushBatches();

  // Phase 2: encode tail, post as a single chunk on one channel.
  // Mixing phase-2 chunks across QPs with phase 1 halves throughput;
  // one chunk lets the NIC's per-QP bandwidth arbitration rebalance.
  uint32_t compressed_size =
      Compressor::getInstance().compressEncodeOneBatch(req);
  uint32_t second_seg = req->local_mem->size;
  size_t second_chunks = second_seg > 0 ? 1 : 0;

  {
    std::lock_guard<std::mutex> lk(pending_compressed_mu_);
    pending_compressed_[wr_id].meta.compressed_size = compressed_size;
  }

  // Advance past phase-1 channels so phase-2 lands on the next one.
  req->channel_id = ((req->channel_id - 1 + first_chunks) % num_channels) + 1;
  postCompressedSegment(req, second_seg, second_chunks, num_channels);
  tracker_->updateExpectedAckCount(wr_id, first_chunks + second_chunks);
  return wr_id;
}

void SendConnection::compressSendRequestSplitFirst(
    std::shared_ptr<RDMASendRequest> req, size_t expected_chunk_count) {
  Compressor::getInstance().compressSplitOneBatch(req);

  uint32_t send_chunks_first =
      req->local_mem->size /
      ChunkSplitStrategy::getRegularChunkSize(req->compress_ctx->getMaxSize(),
                                              expected_chunk_count);
  if (send_chunks_first > 0) {
    postChunkedRequest(req, send_chunks_first);
  }
  uint32_t uncompressed_size =
      Compressor::getInstance().compressEncodeOneBatch(req);
  postChunkedRequest(req, expected_chunk_count - send_chunks_first);
  tracker_->updateExpectedAckCount(
      req->wr_id, ChunkSplitStrategy::getMessageChunkCount(uncompressed_size));
}

void SendConnection::processOnceSendRequests(
    std::shared_ptr<RDMASendRequest> req, SendReqMeta& meta, int index) {
  req->imm_data.set_index(index);
  req->channel_id = meta.channel_id;
  req->remote_mem = std::make_shared<RemoteMemInfo>(meta.remote_mem);
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

void SendConnection::pollDataChannels() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  for (auto& [channel_id, channel] : channels_) {
    std::vector<CQMeta> cq_datas;
    if (channel && channel->pollOnce(cq_datas)) {
      std::vector<uint64_t> acks;
      for (auto const& cq_data : cq_datas) {
        // A signaled CQE produced by batched flush represents completion
        // of itself + all preceding unsignaled WRs on this QP. Expand
        // the wr_id list accordingly.
        acks.clear();
        channel->expandCompletion(cq_data.wr_id, acks);
        for (uint64_t wid : acks) {
          uint32_t msg_seq;
          if (cc_.enabled()) {
            // Decode: low 32 bits = message seq (tracker), high 32 = TSC ID
            msg_seq = static_cast<uint32_t>(wid);
            uint32_t tsc_id = static_cast<uint32_t>(wid >> 32);
            tracker_->acknowledge(msg_seq);
            cc_.onAck(tsc_id, cq_data.len);
            size_t prev = cc_inflight_bytes_.load(std::memory_order_relaxed);
            size_t sub = std::min(prev, static_cast<size_t>(cq_data.len));
            cc_inflight_bytes_.fetch_sub(sub, std::memory_order_relaxed);
          } else {
            msg_seq = static_cast<uint32_t>(wid);
            tracker_->acknowledge(msg_seq);
          }
          maybePushCompressedMeta(static_cast<int64_t>(msg_seq));
        }
      }
    }
  }
  pollAckRing();
}

void SendConnection::pollAckRing() {
  if (!ack_ring_) return;
  if (pending_compressed_count_.load(std::memory_order_relaxed) == 0) return;
  auto* slots = static_cast<AckSlot*>(ack_ring_->addr);
  std::vector<uint64_t> released_offsets;
  {
    std::lock_guard<std::mutex> lk(pending_compressed_mu_);
    for (auto it = pending_compressed_.begin();
         it != pending_compressed_.end();) {
      uint32_t ack_slot = it->second.ack_slot;
      uint64_t v = slots[ack_slot].value.load(std::memory_order_acquire);
      if (v == 0) {
        ++it;
        continue;
      }
      slots[ack_slot].value.store(0, std::memory_order_relaxed);
      if (it->second.arena_bytes > 0)
        released_offsets.push_back(it->second.arena_offset);
      it = pending_compressed_.erase(it);
      pending_compressed_count_.fetch_sub(1, std::memory_order_relaxed);
    }
  }
  for (auto off : released_offsets) decompress_arena_.release(off);
}

void SendConnection::maybePushCompressedMeta(int64_t wr_id) {
  WriteReqMeta meta_to_push{};
  uint32_t push_slot = 0;
  {
    std::lock_guard<std::mutex> lk(pending_compressed_mu_);
    auto it = pending_compressed_.find(wr_id);
    if (it == pending_compressed_.end() || it->second.meta_pushed) return;
    if (!tracker_->isAcknowledged(wr_id)) return;
    it->second.meta_pushed = true;
    meta_to_push = it->second.meta;
    push_slot = static_cast<uint32_t>(static_cast<uint64_t>(wr_id) %
                                      kWriteMetaRingCapacity);
  }
  UCCL_LOG(INFO, UCCL_RDMA)
      << "SendConnection::maybePushCompressedMeta - pushing wr_id=" << wr_id
      << " slot=" << push_slot
      << " decompress_offset=" << meta_to_push.decompress_offset
      << " compressed_size=" << meta_to_push.compressed_size
      << " ack_slot=" << meta_to_push.ack_slot << " user_remote_addr=0x"
      << std::hex << meta_to_push.user_remote_addr << std::dec;
  std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
  if (ctrl_channel_) ctrl_channel_->pushWriteMeta(meta_to_push, push_slot);
}

void SendConnection::pollingLoop() {
  UCCL_LOG(INFO, UCCL_RDMA) << "SendConnection::pollingLoop - Started";
  uccl::pin_thread_to_numa(numa_node_);
  while (running_.load(std::memory_order_acquire)) {
    pollControlChannel();
    processSendRequests();
    pollDataChannels();

    UCCL_LOG_EVERY_N(INFO, UCCL_RDMA, 100000000)
        << "SendConnection::pollingLoop - Still running";
  }
  UCCL_LOG(INFO, UCCL_RDMA) << "SendConnection::pollingLoop - Stopped";
}

uint64_t SendConnection::DecompressArena::reserve(uint64_t bytes) {
  std::unique_lock<std::mutex> lk(mu);
  while (true) {
    if (head + bytes > size) head = 0;
    uint64_t start = head, end = head + bytes;
    auto it = inflight.lower_bound(start);
    bool overlap = false;
    if (it != inflight.end() && it->first < end) overlap = true;
    if (!overlap && it != inflight.begin()) {
      auto prev = std::prev(it);
      if (prev->first + prev->second > start) overlap = true;
    }
    if (!overlap) {
      inflight.emplace(start, bytes);
      head = end;
      return start;
    }
    cv.wait(lk);
  }
}

void SendConnection::DecompressArena::release(uint64_t offset) {
  std::lock_guard<std::mutex> lk(mu);
  inflight.erase(offset);
  cv.notify_all();
}

RecvConnection::RecvConnection(int numa_node, bool auto_start_polling)
    : numa_node_(numa_node),
      running_(false),
      poll_thread_(nullptr),
      auto_start_polling_(auto_start_polling) {}

RecvConnection::~RecvConnection() { stopPolling(); }

void RecvConnection::addChannel(uint32_t channel_id,
                                std::shared_ptr<RDMADataChannel> channel) {
  RDMAConnection::addChannel(channel_id, channel);
}

std::shared_ptr<RDMADataChannel> RecvConnection::getChannel(
    uint32_t channel_id) const {
  auto result = RDMAConnection::getChannel(channel_id);
  return result;
}

size_t RecvConnection::channelCount() const {
  auto result = RDMAConnection::channelCount();
  if (ctrl_channel_) {
    result += 1;
  }
  return result;
}

size_t RecvConnection::normalChannelCount() const {
  return RDMAConnection::channelCount();
}

std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
RecvConnection::channels() const {
  return RDMAConnection::channels();
}

void RecvConnection::startPolling() {
  if (running_.load()) {
    return;
  }
  running_.store(true);
  poll_thread_ =
      std::make_unique<std::thread>(&RecvConnection::pollingLoop, this);
}

void RecvConnection::stopPolling() {
  if (!running_.load()) {
    return;
  }
  running_.store(false);
  if (poll_thread_ && poll_thread_->joinable()) {
    poll_thread_->join();
  }
}

int64_t RecvConnection::recv(std::shared_ptr<RDMARecvRequest> req) {
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

bool RecvConnection::check(uint64_t index) {
  return ctrl_channel_->check_done(index);
}

void RecvConnection::setRemoteAckRing(RemoteMemInfo const& m) {
  if (m.length == 0) return;
  remote_ack_ring_ = m;
  // The compressed-write path needs the receive side to actively poll its
  // control-channel CQ for incoming WriteReqMeta IMMs — but for pure
  // one-sided writes the benchmark/user never calls recv_async +
  // poll_async, so user-driven recvRoutine() never runs. Force the
  // polling thread up here regardless of auto_start_polling_.
  if (!running_.load(std::memory_order_acquire)) startPolling();
}

void RecvConnection::pollAndProcessCompletions() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (ctrl_channel_) {
    ctrl_channel_->noblockingPoll();
    for (auto const& m : ctrl_channel_->drainPendingWriteMetas()) {
      handleCompressedWriteArrival(m);
    }
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
              Compressor::getInstance().decompress(req_meta->remote_mem,
                                                   req_meta->local_mem,
                                                   req_meta->float_type);
            }

          } else {
            UCCL_LOG(WARN) << "RecvConnection::pollAndProcessCompletions - "
                              "ctrl_channel_ is null, cannot call recv_done";
          }
        }
      }
    }
  }
  UCCL_LOG_EVERY_N(INFO, UCCL_RDMA, 100000000)
      << "RecvConnection::pollingLoop - Still running, channels: "
      << channels_.size();
}

void RecvConnection::pollingLoop() {
  UCCL_LOG(INFO, UCCL_RDMA) << "RecvConnection::pollingLoop - Started";
  uccl::pin_thread_to_numa(numa_node_);
  while (running_.load(std::memory_order_acquire)) {
    pollAndProcessCompletions();
    // optional small sleep/yield to avoid busy-looping if desired:
    // std::this_thread::yield();
  }
  UCCL_LOG(INFO, UCCL_RDMA) << "RecvConnection::pollingLoop - Stopped";
}

void RecvConnection::postAckHostFn(void* user_data) {
  std::unique_ptr<AsyncAckCtx> ctx(static_cast<AsyncAckCtx*>(user_data));
  ctx->ctrl_channel->postAckWrite(ctx->remote_ack_addr, ctx->remote_ack_rkey,
                                  ctx->ack_slot,
                                  static_cast<uint64_t>(ctx->wr_id) + 1);
}

void RecvConnection::handleCompressedWriteArrival(WriteReqMeta const& m) {
  UCCL_LOG(INFO, UCCL_RDMA)
      << "RecvConnection::handleCompressedWriteArrival - wr_id=" << m.wr_id
      << " decompress_offset=" << m.decompress_offset
      << " compressed_size=" << m.compressed_size
      << " total_uncomp=" << m.total_uncomp_size
      << " float_type=" << m.float_type << " ack_slot=" << m.ack_slot;
  auto decomp_buf = Compressor::getInstance().getDecompressBuffer();
  if (unlikely(!decomp_buf || remote_ack_ring_.length == 0)) {
    UCCL_LOG(WARN) << "handleCompressedWriteArrival: missing infra";
    return;
  }
  uint64_t slot_base =
      reinterpret_cast<uint64_t>(decomp_buf->addr) + m.decompress_offset;
  RemoteMemInfo in(slot_base, m.compressed_size, decomp_buf->mr_array,
                   MemoryType::GPU);
  RegMemBlock out(reinterpret_cast<void*>(m.user_remote_addr),
                  m.total_uncomp_size, MemoryType::GPU);

  // Ack immediately: meta arrival implies data is already in
  // decompress_buffer (sender posts meta only after all chunk WCs). wr_id+1
  // avoids writing the zero sentinel that pollAckRing uses to mean "not yet
  // acked".
  ctrl_channel_->postAckWrite(
      remote_ack_ring_.addr + m.ack_slot * sizeof(AckSlot),
      remote_ack_ring_.getKeyByContextID(0), m.ack_slot,
      static_cast<uint64_t>(m.wr_id) + 1);

  Compressor::getInstance().decompressAsync(
      in, out, static_cast<uccl::FloatType>(m.float_type),
      /*on_done=*/nullptr, /*user_data=*/nullptr);
}

bool RecvConnection::collectRkeyForChannel(
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

bool RecvConnection::setupRecvRequestChannelAndMemoryRegion(
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

#include "rdma_connection.h"
#include "util/debug.h"
#include "util/util.h"

// ── RDMAConnection: Lifecycle ────────────────────────────────────────────────

RDMAConnection::RDMAConnection() : last_channel_id_(0) {}

RDMAConnection::~RDMAConnection() = default;

// ── RDMAConnection: Channel registry ─────────────────────────────────────────

void RDMAConnection::add_channel(uint32_t channel_id,
                                 std::shared_ptr<RDMADataChannel> channel) {
  if (!channel) {
    throw std::invalid_argument("add_channel called with null channel");
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);

  channels_[channel_id] = std::move(channel);
}

std::shared_ptr<RDMADataChannel> RDMAConnection::get_channel(
    uint32_t channel_id) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = channels_.find(channel_id);
  if (it == channels_.end()) return nullptr;
  return it->second;
}

size_t RDMAConnection::channel_count() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return channels_.size();
}

std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
RDMAConnection::channels() const {
  mutex_.lock_shared();
  mutex_.unlock_shared();  // just to annotate read lock expected
  return channels_;
}

// ── RDMAConnection: Channel selection ────────────────────────────────────────

std::pair<uint32_t, uint64_t>
RDMAConnection::select_next_channel_round_robin() {
  // Get the total number of channels
  size_t num_channels = RDMAConnection::channel_count();
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
  auto channel = get_channel(next_id);
  if (unlikely(!channel)) {
    UCCL_LOG(WARN) << "RDMAConnection: Channel not found for channel_id "
                   << next_id << " num_channels " << num_channels;
    return {0, 0};
  }

  uint64_t context_id = channel->get_context_id();
  return {next_id, context_id};
}

std::pair<uint32_t, uint64_t> RDMAConnection::select_next_channel_random() {
  // Get the total number of channels
  size_t num_channels = RDMAConnection::channel_count();
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
  auto channel = get_channel(random_id);
  if (unlikely(!channel)) {
    UCCL_LOG(WARN) << "RDMAConnection: Channel not found for channel_id "
                   << random_id << " num_channels " << num_channels;
    return {0, 0};
  }

  uint64_t context_id = channel->get_context_id();
  return {random_id, context_id};
}

// ── RDMAConnection: Fast channel cache ───────────────────────────────────────

RDMADataChannel* RDMAConnection::get_channel_fast(uint32_t channel_id) const {
  if (likely(fast_channels_ready_.load(std::memory_order_acquire))) {
    if (likely(channel_id >= 1 &&
               channel_id <=
                   fast_channel_count_.load(std::memory_order_relaxed))) {
      return fast_channels_[channel_id - 1];
    }
  }
  auto sp = get_channel(channel_id);
  return sp.get();
}

std::pair<uint32_t, RDMADataChannel*>
RDMAConnection::select_next_channel_round_robin_fast() {
  if (unlikely(!fast_channels_ready_.load(std::memory_order_acquire))) {
    build_fast_channel_cache();
  }
  size_t n = fast_channel_count_.load(std::memory_order_relaxed);
  if (unlikely(n == 0)) return {0, nullptr};
  uint32_t prev = last_channel_id_.load(std::memory_order_relaxed);
  uint32_t next = (prev % n) + 1;
  last_channel_id_.store(next, std::memory_order_relaxed);
  return {next, fast_channels_[next - 1]};
}

void RDMAConnection::build_fast_channel_cache() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  size_t n = channels_.size();
  fast_channels_.assign(n, nullptr);
  for (auto const& [cid, ch] : channels_) {
    if (cid >= 1 && cid <= n) fast_channels_[cid - 1] = ch.get();
  }
  fast_channel_count_.store(n, std::memory_order_release);
  fast_channels_ready_.store(true, std::memory_order_release);
}

// ── SendConnection: Lifecycle ────────────────────────────────────────────────

SendConnection::SendConnection(double link_bandwidth_bps)
    : cc_(uccl::cc::CongestionControlState::parseMode("UCCL_P2P_RDMA_CC"),
          uccl::freq_ghz, link_bandwidth_bps) {
  tracker_ = std::make_shared<AtomicBitmapPacketTrackerMultiAck>();
}

SendConnection::~SendConnection() = default;

// ── SendConnection: Channel registry ─────────────────────────────────────────

void SendConnection::add_channel(uint32_t channel_id,
                                 std::shared_ptr<RDMADataChannel> channel) {
  RDMAConnection::add_channel(channel_id, channel);
}

std::shared_ptr<RDMADataChannel> SendConnection::get_channel(
    uint32_t channel_id) const {
  auto result = RDMAConnection::get_channel(channel_id);
  return result;
}

size_t SendConnection::channel_count() const {
  auto result = RDMAConnection::channel_count();
  std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
  if (ctrl_channel_) {
    result += 1;
  }
  return result;
}

size_t SendConnection::data_channel_count() const {
  return RDMAConnection::channel_count();
}

std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
SendConnection::channels() const {
  return RDMAConnection::channels();
}

// ── SendConnection: One-sided transfer ───────────────────────────────────────

int64_t SendConnection::post_write_or_read(
    std::shared_ptr<RDMASendRequest> req) {
  if (unlikely(req->send_type != SendType::Write &&
               req->send_type != SendType::Read)) {
    UCCL_LOG(ERROR) << "SendConnection::write - Invalid send_type, expected "
                       "SendType::Write";
    return -1;
  }

  // Enforce CC window before posting
  if (cc_.enabled()) {
    size_t inflight_limit_bytes = current_inflight_limit_bytes();
    while (current_inflight_bytes() > inflight_limit_bytes) {
      std::this_thread::yield();
      inflight_limit_bytes = current_inflight_limit_bytes();
    }
  }

  int64_t wr_id = tracker_->send_packet(req->get_local_len());
  req->wr_id = wr_id;

  // Size gate first: small messages must not touch compression state or call
  // Compressor helpers even when compression is configured for the connection.
  size_t const msg_size = req->local_mem->size;
  if (msg_size >= kMinCompressBytes && req->send_type == SendType::Write) {
    bool const c_ack = static_cast<bool>(ack_ring_);
    bool const c_rdb = remote_decompress_buf_.length > 0;
    bool const c_ctx = static_cast<bool>(req->compress_ctx);
    bool const c_ft =
        c_ctx && (req->compress_ctx->get_float_type() != FloatType::kUndefined);
    bool const c_fit =
        static_cast<uint64_t>(msg_size) <= remote_decompress_buf_.length;
    if (c_ack && c_rdb && c_ctx && c_ft && c_fit &&
        Compressor::get_instance().should_compress_and_split_first(msg_size)) {
      return compress_write_request_split_first(req);
    }
  }

  // Lock-free channel selection + direct pointer.
  auto [channel_id, ch_ptr] = select_next_channel_round_robin_fast();
  if (unlikely(channel_id == 0 || ch_ptr == nullptr)) {
    UCCL_LOG(ERROR) << "SendConnection::write - Failed to select channel";
    return -1;
  }

  req->channel_id = channel_id;

  // Fast single-chunk path: skip post_chunked_request/post_single_chunk and
  // submit directly to the resolved channel pointer.
  if (likely(ChunkSplitStrategy::get_message_chunk_count(
                 req->local_mem->size) == 1)) {
    req->imm_data.set_chunk_count(1);

    int64_t saved_wr_id = req->wr_id;
    if (cc_.enabled()) {
      uint32_t tsc_id =
          chunk_tsc_counter_.fetch_add(1, std::memory_order_relaxed);
      req->wr_id = (static_cast<int64_t>(tsc_id) << 32) |
                   static_cast<uint32_t>(req->wr_id);
      cc_.recordSendTsc(tsc_id);
    }

    int64_t send_ret = ch_ptr->submit_request(req);
    req->wr_id = saved_wr_id;

    if (send_ret >= 0 && cc_.enabled()) {
      cc_inflight_bytes_.fetch_add(req->get_local_len(),
                                   std::memory_order_relaxed);
    }
    return wr_id;
  }

  post_chunked_request(req);

  // Since post_chunked_request() is non-blocking — if the CC
  // window is exhausted mid-message it saves the remaining chunks
  // and returns immediately.
  // Draining them here.
  while (!drain_pending_chunks()) {
    std::this_thread::yield();
  }

  return wr_id;
}

// ── SendConnection: One-sided batch transfer ─────────────────────────────────

bool SendConnection::can_use_raw_one_sided_batch(SendType send_type,
                                                 size_t max_iov_bytes) {
  if (cc_.enabled()) return false;
  if (send_type != SendType::Write) return true;
  if (max_iov_bytes < kMinCompressBytes) return true;
  if (Compressor::get_instance().get_compress_strategy() !=
      CompressStrategy::kNone) {
    return false;
  }
  return true;
}

bool SendConnection::post_write_or_read_batch(SendType send_type,
                                              OneSidedBatchOp const* ops,
                                              size_t num_ops, int64_t* wr_ids,
                                              RawBatchWait* waits,
                                              size_t* num_waits) {
  if (num_waits != nullptr) *num_waits = 0;
  if (unlikely(send_type != SendType::Write && send_type != SendType::Read)) {
    UCCL_LOG(ERROR) << "SendConnection::post_write_or_read_batch - invalid "
                       "send_type";
    return false;
  }
  if (unlikely((num_ops > 0 && ops == nullptr) || wr_ids == nullptr ||
               (waits != nullptr && num_waits == nullptr))) {
    UCCL_LOG(ERROR) << "SendConnection::post_write_or_read_batch - null input";
    return false;
  }

  size_t const num_channels = data_channel_count();
  if (unlikely(num_channels == 0 || num_channels > kQpNumPerChannel)) {
    UCCL_LOG(ERROR) << "SendConnection::post_write_or_read_batch - invalid "
                       "channel count: "
                    << num_channels;
    return false;
  }

  for (size_t i = 0; i < num_ops; ++i) {
    auto const& op = ops[i];
    if (unlikely(op.local_addr == nullptr || op.size == 0 ||
                 op.size > std::numeric_limits<uint32_t>::max() ||
                 op.local_mr_array == nullptr || op.slot_item == nullptr)) {
      UCCL_LOG(ERROR)
          << "SendConnection::post_write_or_read_batch - invalid op " << i;
      return false;
    }
    if (unlikely(ChunkSplitStrategy::get_message_chunk_count(op.size) != 1)) {
      UCCL_LOG(ERROR) << "SendConnection::post_write_or_read_batch - op " << i
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
    auto [channel_id, ch_ptr] = select_next_channel_round_robin_fast();
    if (unlikely(channel_id == 0 || ch_ptr == nullptr ||
                 channel_id > num_channels || channel_id > kQpNumPerChannel)) {
      UCCL_LOG(ERROR) << "SendConnection::post_write_or_read_batch - failed to "
                         "select channel";
      return false;
    }

    auto* local_mr = op.local_mr_array->get_key_by_channel_id(channel_id);
    if (unlikely(local_mr == nullptr)) {
      UCCL_LOG(ERROR) << "SendConnection::post_write_or_read_batch - missing "
                         "local MR for op "
                      << i << " channel " << channel_id;
      return false;
    }

    RKeyArray remote_rkeys;
    remote_rkeys.copy_from(static_cast<char const*>(op.slot_item->padding));

    PreparedBatchOp prepared{};
    prepared.op_index = i;
    prepared.channel_id = channel_id;
    prepared.channel = ch_ptr;
    prepared.local_key = local_mr->lkey;
    prepared.remote_key = remote_rkeys.get_key_by_channel_id(channel_id);
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

    int64_t wr_id = tracker_->send_packet(op.size);
    if (unlikely(wr_id < 0)) {
      UCCL_LOG(ERROR) << "SendConnection::post_write_or_read_batch - failed to "
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

  bool const wait_all_wr_ids = is_efa_transport();
  for (size_t i = 0; i < num_channels; ++i) {
    if (channel_batches[i].empty()) continue;
    if (unlikely(channel_ptrs[i] == nullptr ||
                 channel_ptrs[i]->post_raw_batch(channel_batches[i]) != 0)) {
      UCCL_LOG(FATAL) << "SendConnection::post_write_or_read_batch - failed to "
                         "post channel batch "
                      << (i + 1)
                      << "; raw one-sided batch post failure is "
                         "unrecoverable after tracker wr_ids are allocated";
      return false;
    }
    if (waits != nullptr) {
      if (wait_all_wr_ids) {
        for (auto const& raw : channel_batches[i]) {
          size_t const wait_idx = (*num_waits)++;
          waits[wait_idx].wr_id = static_cast<int64_t>(raw.wr_id);
          waits[wait_idx].iov_count = 1;
        }
      } else {
        size_t const wait_idx = (*num_waits)++;
        waits[wait_idx].wr_id =
            static_cast<int64_t>(channel_batches[i].back().wr_id);
        waits[wait_idx].iov_count = channel_batches[i].size();
      }
    }
  }
  return true;
}

// ── SendConnection: Compression setup ────────────────────────────────────────

void SendConnection::set_remote_decompress_buf(RemoteMemInfo const& m) {
  if (m.length == 0) return;
  remote_decompress_buf_ = m;
  decompress_arena_.size = m.length;
}

void SendConnection::set_local_ack_ring(std::shared_ptr<RegMemBlock> ring) {
  ack_ring_ = ring;
}

// ── SendConnection: Completion ───────────────────────────────────────────────

bool SendConnection::check_completion(int64_t wr_id) {
  return tracker_->is_acknowledged(wr_id);
}

// ── SendConnection: Polling ──────────────────────────────────────────────────

void SendConnection::send_routine() {
  std::lock_guard<std::mutex> guard(send_routine_mu_);
  poll_control_channel();
  poll_data_channels();
  UCCL_LOG_EVERY_N(INFO, UCCL_RDMA, 100000000)
      << "SendConnection::send_routine - Still running";
}

void SendConnection::flush_batches() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  for (auto& [cid, channel] : channels_) {
    if (channel) channel->flush_batch();
  }
}

// ── SendConnection: Congestion control ───────────────────────────────────────

size_t SendConnection::current_inflight_limit_bytes() {
  return cc_.enabled() ? cc_.getWindowBytes() : kInFlightMaxSizeKB * 1024;
}

size_t SendConnection::current_inflight_bytes() {
  return cc_.enabled() ? cc_inflight_bytes_.load(std::memory_order_relaxed)
                       : tracker_->get_total_inflight_bytes();
}

// ── SendConnection: Internal posting ─────────────────────────────────────────

bool SendConnection::post_request_on_channel(
    std::shared_ptr<RDMASendRequest> req) {
  auto channel = get_channel(req->channel_id);
  if (unlikely(!channel)) {
    UCCL_LOG(WARN) << "SendConnection: Channel not found for channel_id "
                   << req->channel_id;
    return false;
  }

  // Per-chunk CC: assign a unique TSC ID and record send timestamp
  // close to the actual ibv_post_send.  The TSC ID is encoded in the
  // upper 32 bits of wr_id; the lower 32 bits keep the message seq
  // used by the tracker.  We save/restore req->wr_id so that callers
  // (e.g. update_expected_ack_count) still see the original message seq.
  int64_t saved_wr_id = req->wr_id;
  if (cc_.enabled()) {
    uint32_t tsc_id =
        chunk_tsc_counter_.fetch_add(1, std::memory_order_relaxed);
    req->wr_id = (static_cast<int64_t>(tsc_id) << 32) |
                 static_cast<uint32_t>(req->wr_id);
    cc_.recordSendTsc(tsc_id);
  }

  int64_t send_ret = channel->submit_request(req);
  req->wr_id = saved_wr_id;
  if (send_ret >= 0 && cc_.enabled()) {
    cc_inflight_bytes_.fetch_add(req->get_local_len(),
                                 std::memory_order_relaxed);
  }
  if (send_ret < 0) {
    UCCL_LOG(WARN) << "SendConnection: Failed to send on channel_id "
                   << req->channel_id;
    return false;
  }

  return true;
}

// ── SendConnection: Polling internals ────────────────────────────────────────

void SendConnection::poll_control_channel() {
  std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
  if (ctrl_channel_) {
    if (ctrl_channel_->noblocking_poll()) {
      UCCL_LOG(INFO, UCCL_RDMA)
          << "SendConnection::polling_loop - Control channel polled "
             "successfully";
    }
  }
}

// ── SendConnection: Chunked posting ──────────────────────────────────────────

bool SendConnection::post_single_chunk(
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
  chunk_req->from_peer_id = req->from_peer_id;
  chunk_req->to_peer_id = req->to_peer_id;
  chunk_req->wr_id = req->wr_id;
  chunk_req->send_type = req->send_type;

  return post_request_on_channel(chunk_req);
}

bool SendConnection::drain_pending_chunks() {
  if (!pending_chunked_) return true;

  auto& ps = *pending_chunked_;
  size_t num_channels = data_channel_count();

  while (ps.next_chunk_idx < ps.chunks.size()) {
    // Per-chunk CC: check window before each chunk.
    if (cc_.enabled()) {
      size_t inflight_limit_bytes = current_inflight_limit_bytes();
      if (current_inflight_bytes() > inflight_limit_bytes) {
        return false;  // Yield back to polling loop.
      }
    }

    if (!post_single_chunk(ps.req, ps.chunks[ps.next_chunk_idx],
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

void SendConnection::post_chunked_request(
    std::shared_ptr<RDMASendRequest> req) {
  // Fast path: single-chunk message. Post `req` directly and skip the
  // chunk-wrapper allocations done by post_single_chunk().
  if (ChunkSplitStrategy::get_message_chunk_count(req->local_mem->size) == 1) {
    req->imm_data.set_chunk_count(1);
    if (!post_request_on_channel(req)) {
      UCCL_LOG(WARN) << "SendConnection: Failed to send request on channel_id "
                     << req->channel_id;
    }
    return;
  }
  // Split message into chunks
  size_t message_size = req->local_mem->size;
  std::vector<MessageChunk> chunks =
      ChunkSplitStrategy::split_message_to_chunks(message_size);
  int expected_chunk_count = static_cast<int>(chunks.size());
  tracker_->update_expected_ack_count(req->wr_id, expected_chunk_count);
  UCCL_LOG(INFO, UCCL_RDMA)
      << "SendConnection: Splitting message into " << chunks.size()
      << " chunks (message_size: " << message_size << ")";
  size_t num_channels = data_channel_count();

  for (size_t i = 0; i < chunks.size(); ++i) {
    // Per-chunk CC: if over budget, save remaining chunks and return.
    if (cc_.enabled()) {
      size_t inflight_limit_bytes = current_inflight_limit_bytes();
      if (current_inflight_bytes() > inflight_limit_bytes) {
        pending_chunked_ = PendingChunkedState{req, std::move(chunks), i,
                                               expected_chunk_count};
        return;
      }
    }

    if (!post_single_chunk(req, chunks[i], i, chunks.size(), num_channels,
                           expected_chunk_count)) {
      UCCL_LOG(WARN) << "SendConnection: Failed to send chunk " << i;
    }
  }
}

// ── SendConnection: Compressed write path ────────────────────────────────────

void SendConnection::post_compressed_segment(
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
    if (!post_single_chunk(req, chunk, i, num_chunks, num_channels,
                           dummy_expected)) {
      UCCL_LOG(WARN) << "post_compressed_segment: chunk " << i
                     << " post failed";
    }
  }
}

int64_t SendConnection::compress_write_request_split_first(
    std::shared_ptr<RDMASendRequest> req) {
  int64_t wr_id = req->wr_id;
  uint32_t total_uncomp = req->local_mem->size;
  uint64_t user_remote_addr = req->remote_mem->addr;
  uint32_t user_remote_rkey =
      req->remote_mem->get_key_by_context_id(0);  // diagnostic / logging only

  // Reserve space in the peer's decompress_buffer; released on ack receipt.
  uint64_t arena_bytes = static_cast<uint64_t>(total_uncomp);
  if (unlikely(arena_bytes > decompress_arena_.size)) {
    UCCL_LOG(ERROR) << "compress_write_request_split_first: request size "
                    << arena_bytes << " exceeds decompress arena size "
                    << decompress_arena_.size;
    return -1;
  }
  uint64_t arena_offset = 0;
  while (!decompress_arena_.try_reserve(arena_bytes, &arena_offset)) {
    send_routine();
    std::this_thread::yield();
  }
  uint32_t ack_slot = static_cast<uint32_t>(
      next_ack_slot_.fetch_add(1, std::memory_order_relaxed) % kAckRingDepth);
  static_cast<AckSlot*>(ack_ring_->addr)[ack_slot].value.store(
      0, std::memory_order_relaxed);
  uint64_t slot_base = remote_decompress_buf_.addr + arena_offset;

  // Phase 1: split into compress_buffer, post to slot_base.
  Compressor::get_instance().compress_split_one_batch(req);
  uint32_t first_seg = req->local_mem->size;
  req->remote_mem = std::make_shared<RemoteMemInfo>(
      slot_base, arena_bytes, remote_decompress_buf_.rkey_array,
      remote_decompress_buf_.type);

  auto [channel_id, _ch_ptr] = select_next_channel_round_robin_fast();
  req->channel_id = channel_id ? channel_id : 1;
  size_t num_channels = data_channel_count();
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
      req->compress_ctx ? req->compress_ctx->get_float_type()
                        : FloatType::kUndefined);
  entry.meta.ack_slot = ack_slot;
  entry.meta.wr_id = wr_id;
  {
    std::lock_guard<std::mutex> lk(pending_compressed_mu_);
    pending_compressed_.emplace(wr_id, entry);
    pending_compressed_count_.fetch_add(1, std::memory_order_relaxed);
  }
  tracker_->update_expected_ack_count(wr_id,
                                      std::numeric_limits<uint32_t>::max());

  // Per-chunk auto-flush gives the round-robin interleaving required on
  // Broadcom bnxt_re (batching causes LOC_QP_OP_ERR vendor=0x3).
  post_compressed_segment(req, first_seg, first_chunks, num_channels);
  flush_batches();

  // Phase 2: encode tail, post as a single chunk on one channel.
  // Mixing phase-2 chunks across QPs with phase 1 halves throughput;
  // one chunk lets the NIC's per-QP bandwidth arbitration rebalance.
  uint32_t compressed_size =
      Compressor::get_instance().compress_encode_one_batch(req);
  uint32_t second_seg = req->local_mem->size;
  size_t second_chunks = second_seg > 0 ? 1 : 0;

  {
    std::lock_guard<std::mutex> lk(pending_compressed_mu_);
    pending_compressed_[wr_id].meta.compressed_size = compressed_size;
  }

  // Advance past phase-1 channels so phase-2 lands on the next one.
  req->channel_id = ((req->channel_id - 1 + first_chunks) % num_channels) + 1;
  post_compressed_segment(req, second_seg, second_chunks, num_channels);
  tracker_->update_expected_ack_count(wr_id, first_chunks + second_chunks);
  return wr_id;
}

void SendConnection::poll_data_channels() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  for (auto& [channel_id, channel] : channels_) {
    std::vector<CQMeta> cq_datas;
    if (channel && channel->poll_once(cq_datas)) {
      std::vector<uint64_t> acks;
      for (auto const& cq_data : cq_datas) {
        // A signaled CQE produced by batched flush represents completion
        // of itself + all preceding unsignaled WRs on this QP. Expand
        // the wr_id list accordingly.
        acks.clear();
        channel->expand_completion(cq_data.wr_id, acks);
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
          maybe_push_compressed_meta(static_cast<int64_t>(msg_seq));
        }
      }
    }
  }
  poll_ack_ring();
}

void SendConnection::poll_ack_ring() {
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

void SendConnection::maybe_push_compressed_meta(int64_t wr_id) {
  WriteReqMeta meta_to_push{};
  uint32_t push_slot = 0;
  {
    std::lock_guard<std::mutex> lk(pending_compressed_mu_);
    auto it = pending_compressed_.find(wr_id);
    if (it == pending_compressed_.end() || it->second.meta_pushed) return;
    if (!tracker_->is_acknowledged(wr_id)) return;
    it->second.meta_pushed = true;
    meta_to_push = it->second.meta;
    push_slot = static_cast<uint32_t>(static_cast<uint64_t>(wr_id) %
                                      kWriteMetaRingCapacity);
  }
  UCCL_LOG(INFO, UCCL_RDMA)
      << "SendConnection::maybe_push_compressed_meta - pushing wr_id=" << wr_id
      << " slot=" << push_slot
      << " decompress_offset=" << meta_to_push.decompress_offset
      << " compressed_size=" << meta_to_push.compressed_size
      << " ack_slot=" << meta_to_push.ack_slot << " user_remote_addr=0x"
      << std::hex << meta_to_push.user_remote_addr << std::dec;
  std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
  if (ctrl_channel_) ctrl_channel_->push_write_meta(meta_to_push, push_slot);
}

bool SendConnection::DecompressArena::try_reserve(uint64_t bytes,
                                                  uint64_t* offset) {
  std::lock_guard<std::mutex> lk(mu);
  if (offset == nullptr || bytes == 0 || bytes > size) return false;
  if (head + bytes > size) head = 0;
  uint64_t start = head, end = head + bytes;
  auto it = inflight.lower_bound(start);
  bool overlap = false;
  if (it != inflight.end() && it->first < end) overlap = true;
  if (!overlap && it != inflight.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second > start) overlap = true;
  }
  if (overlap) return false;
  inflight.emplace(start, bytes);
  head = end;
  *offset = start;
  return true;
}

void SendConnection::DecompressArena::release(uint64_t offset) {
  std::lock_guard<std::mutex> lk(mu);
  inflight.erase(offset);
}

// ── RecvConnection: Lifecycle ────────────────────────────────────────────────

RecvConnection::RecvConnection() = default;

RecvConnection::~RecvConnection() = default;

// ── RecvConnection: Channel registry ─────────────────────────────────────────

void RecvConnection::add_channel(uint32_t channel_id,
                                 std::shared_ptr<RDMADataChannel> channel) {
  RDMAConnection::add_channel(channel_id, channel);
}

std::shared_ptr<RDMADataChannel> RecvConnection::get_channel(
    uint32_t channel_id) const {
  auto result = RDMAConnection::get_channel(channel_id);
  return result;
}

size_t RecvConnection::channel_count() const {
  auto result = RDMAConnection::channel_count();
  if (ctrl_channel_) {
    result += 1;
  }
  return result;
}

size_t RecvConnection::data_channel_count() const {
  return RDMAConnection::channel_count();
}

std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
RecvConnection::channels() const {
  return RDMAConnection::channels();
}

// ── RecvConnection: Compression recv path ────────────────────────────────────

void RecvConnection::set_remote_ack_ring(RemoteMemInfo const& m) {
  if (m.length == 0) return;
  remote_ack_ring_ = m;
}

void RecvConnection::handle_compressed_write_arrival(WriteReqMeta const& m) {
  UCCL_LOG(INFO, UCCL_RDMA)
      << "RecvConnection::handle_compressed_write_arrival - wr_id=" << m.wr_id
      << " decompress_offset=" << m.decompress_offset
      << " compressed_size=" << m.compressed_size
      << " total_uncomp=" << m.total_uncomp_size
      << " float_type=" << m.float_type << " ack_slot=" << m.ack_slot;
  auto decomp_buf = Compressor::get_instance().get_decompress_buffer();
  if (unlikely(!decomp_buf || remote_ack_ring_.length == 0)) {
    UCCL_LOG(WARN) << "handle_compressed_write_arrival: missing infra";
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
  // avoids writing the zero sentinel that poll_ack_ring uses to mean "not yet
  // acked".
  ctrl_channel_->post_ack_write(
      remote_ack_ring_.addr + m.ack_slot * sizeof(AckSlot),
      remote_ack_ring_.get_key_by_context_id(0), m.ack_slot,
      static_cast<uint64_t>(m.wr_id) + 1);

  Compressor::get_instance().decompress_async(
      in, out, static_cast<FloatType>(m.float_type),
      /*on_done=*/nullptr, /*user_data=*/nullptr);
}

void RecvConnection::post_ack_host_fn(void* user_data) {
  std::unique_ptr<AsyncAckCtx> ctx(static_cast<AsyncAckCtx*>(user_data));
  ctx->ctrl_channel->post_ack_write(ctx->remote_ack_addr, ctx->remote_ack_rkey,
                                    ctx->ack_slot,
                                    static_cast<uint64_t>(ctx->wr_id) + 1);
}

// ── RecvConnection: Polling loop ─────────────────────────────────────────────

void RecvConnection::poll_and_process_completions() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (ctrl_channel_) {
    ctrl_channel_->noblocking_poll();
    for (auto const& meta : ctrl_channel_->drain_pending_write_metas()) {
      handle_compressed_write_arrival(meta);
    }
  }
  for (auto& [channel_id, channel] : channels_) {
    if (!channel) continue;
    bool polled = false;
    std::vector<CQMeta> cq_datas;
    polled = channel->poll_once(cq_datas);
    if (polled) {
      for (auto const& cq_data : cq_datas) {
        UCCL_LOG(INFO, UCCL_RDMA)
            << "RecvConnection::poll_and_process_completions - Channel "
            << channel_id << " polled completion: " << cq_data;
      }
    }
  }
  UCCL_LOG_EVERY_N(INFO, UCCL_RDMA, 100000000)
      << "RecvConnection::polling_loop - Still running, channels: "
      << channels_.size();
}

void RecvConnection::recv_routine() {
  std::lock_guard<std::mutex> guard(recv_routine_mu_);
  poll_and_process_completions();
}

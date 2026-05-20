#pragma once
#include "compression.h"
#include "define.h"
#include "rdma_ctrl_channel.h"
#include "rdma_data_channel.h"
#include <cc/cc_state.h>
#include <cc/link_bandwidth.h>
#include <optional>
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

  // Lock-free hot-path channel cache. `channels_` is append-only during
  // connection setup, so once cached the snapshot stays valid for the
  // connection's lifetime. Falls back through the locked path if not built.
  RDMADataChannel* getChannelFast(uint32_t channel_id) const {
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

  // Select next channel via round-robin without acquiring `mutex_` in the
  // common case. Returns (channel_id, channel_ptr); on first call this
  // primes the cache under a shared lock.
  std::pair<uint32_t, RDMADataChannel*> selectNextChannelRoundRobinFast() {
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

 protected:
  void buildFastChannelCache() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t n = channels_.size();
    fast_channels_.assign(n, nullptr);
    for (auto const& [cid, ch] : channels_) {
      if (cid >= 1 && cid <= n) fast_channels_[cid - 1] = ch.get();
    }
    fast_channel_count_.store(n, std::memory_order_release);
    fast_channels_ready_.store(true, std::memory_order_release);
  }

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> channels_;
  std::atomic<uint32_t> last_channel_id_;

  // Lock-free channel pointer cache (populated lazily on first hot-path use).
  mutable std::vector<RDMADataChannel*> fast_channels_;
  mutable std::atomic<size_t> fast_channel_count_{0};
  mutable std::atomic<bool> fast_channels_ready_{false};
};

class SendConnection : public RDMAConnection {
 public:
  SendConnection(int numa_node, bool auto_start_polling = true,
                 double link_bandwidth_bps = 400.0 * 1e9 / 8.0)
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

  int64_t postWriteOrRead(std::shared_ptr<RDMASendRequest> req) {
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

    // Compressed-write path: split → encode pipelined, data targets a
    // variable-size region in the peer's decompress_buffer, and the
    // WriteReqMeta is pushed via the control channel after all data chunks
    // are locally ACKed. Requires a fully-formed compress_ctx (dietgpu
    // CHECK-fails on kUndefined) and the message must fit in the peer's
    // decompress_buffer (so reserve() doesn't deadlock).
    bool c_send_type = (req->send_type == SendType::Write);
    bool c_ack = static_cast<bool>(ack_ring_);
    bool c_rdb = remote_decompress_buf_.length > 0;
    bool c_ctx = static_cast<bool>(req->compress_ctx);
    bool c_ft = c_ctx && (req->compress_ctx->getFloatType() !=
                          uccl::FloatType::kUndefined);
    bool c_fit = static_cast<uint64_t>(req->local_mem->size) <=
                 remote_decompress_buf_.length;
    bool c_should =
        Compressor::getInstance().shouldCompressAndSplitFirst(req->local_mem->size);
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

  int64_t read(std::shared_ptr<RDMASendRequest> req) {
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

  // Start polling thread
  void startPolling() {
    if (running_.load()) {
      return;
    }
    running_.store(true);
    poll_thread_ =
        std::make_unique<std::thread>(&SendConnection::pollingLoop, this);
  }

  bool check(int64_t wr_id) {
    // Compressed and non-compressed writes alike complete when the local
    // data WCs have all arrived. For RC RDMA WRITE, local WC = peer NIC
    // ACK, which means the bytes are durably in the receiver's GPU memory
    // (in our case, the decompress_buffer slot). Receiver-side decompress
    // runs asynchronously after the WriteReqMeta IMM arrives, and the
    // receiver acks the sender immediately (before decompress kernel
    // finishes) — that ack drives pollAckRing() for arena slot recycling
    // but is OFF this critical path.
    return tracker_->isAcknowledged(wr_id);
  }

  void setRemoteDecompressBuf(RemoteMemInfo const& m) {
    remote_decompress_buf_ = m;
    decompress_arena_.size = m.length;
    // The compressed-write path needs continuous background pollAckRing()
    // to release arena slots — user threads can block in arena.reserve()
    // and never get back to poll_async() (which is what otherwise drives
    // pollingLoopForMeta in non-auto-polling mode). Force the polling
    // thread up here, regardless of cc_polling.
    if (!running_.load(std::memory_order_acquire)) startPolling();
  }
  void setLocalAckRing(std::shared_ptr<RegMemBlock> ring) { ack_ring_ = ring; }

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

  // Flush any batched send WRs on all channels of this connection. Used to
  // amortize doorbell cost across many small RDMA writes/reads posted via
  // g_uccl_batch_post.
  void flushBatches() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (auto& [cid, channel] : channels_) {
      if (channel) channel->flushBatch();
    }
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

  uccl::cc::CongestionControlState cc_;
  std::atomic<uint32_t> chunk_tsc_counter_{0};

  // ── Compressed-write state ─────────────────────────────────────────────
  // ack_ring_ is this endpoint's local host ack ring (peer writes here).
  // remote_decompress_buf_ is the peer's GPU decompress buffer (we write here).
  std::shared_ptr<RegMemBlock> ack_ring_;
  RemoteMemInfo remote_decompress_buf_;
  std::atomic<uint64_t> next_ack_slot_{0};
  struct PendingCompressed {
    WriteReqMeta meta;
    uint32_t ack_slot = 0;
    uint64_t arena_offset = 0;
    uint64_t arena_bytes = 0;
    bool meta_pushed = false;
  };
  std::mutex pending_compressed_mu_;
  std::unordered_map<int64_t, PendingCompressed> pending_compressed_;
  std::atomic<size_t> pending_compressed_count_{0};

  // Variable-size FIFO-ish allocator over the peer's decompress_buffer.
  // Each compressed write reserves enough contiguous bytes for its compressed
  // payload (upper bound = total_uncomp_size, since dietgpu output ≤ input
  // for the float types we support). Released by pollAckRing() (background
  // polling thread) once the receiver's ack lands in our ack_ring slot.
  // Bump-pointer with wrap-around + linear in-flight overlap check — small
  // N keeps this cheap.
  struct DecompressArena {
    uint64_t size = 0;
    uint64_t head = 0;
    std::map<uint64_t, uint64_t> inflight;  // offset → bytes
    std::mutex mu;
    std::condition_variable cv;

    uint64_t reserve(uint64_t bytes) {
      // Diagnostic mode: UCCL_P2P_COMPRESS_FIX_ARENA_OFFSET=1 forces every
      // compressed write to land at decompress_buffer[0..]. Used to test
      // whether varying destination address per iter is what's costing the
      // compressed path its NIC throughput (NIC/HBM cache locality
      // hypothesis). Not safe for production — successive writes overlap
      // and the receiver's in-flight decompress may read corrupted bytes
      // — but the bench doesn't verify payload integrity.
      static bool const fix_offset = []() {
        char const* env = std::getenv("UCCL_P2P_COMPRESS_FIX_ARENA_OFFSET");
        return env && env[0] == '1';
      }();
      if (fix_offset) return 0;

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
    void release(uint64_t offset) {
      std::lock_guard<std::mutex> lk(mu);
      inflight.erase(offset);
      cv.notify_all();
    }
  };
  DecompressArena decompress_arena_;
  // Per-chunk inflight byte counter for CC window checks.
  // Unlike tracker_->getTotalInflightBytes() which only decreases when ALL
  // chunks of a message are acked, this counter decreases on each chunk CQE.
  std::atomic<size_t> cc_inflight_bytes_{0};

  // Pending chunked request state for per-chunk CC pacing.
  struct PendingChunkedState {
    std::shared_ptr<RDMASendRequest> req;
    std::vector<MessageChunk> chunks;
    size_t next_chunk_idx = 0;
    int remaining_expected_count = 0;
  };
  std::optional<PendingChunkedState> pending_chunked_;

  inline size_t currentInflightLimitBytes() {
    return cc_.enabled() ? cc_.getWindowBytes() : kInFlightMaxSizeKB * 1024;
  }

  // Return the inflight byte count, depends on CC enablement status
  inline size_t currentInflightBytes() {
    return cc_.enabled() ? cc_inflight_bytes_.load(std::memory_order_relaxed)
                         : tracker_->getTotalInflightBytes();
  }

  // Send a request through the appropriate channel
  // Returns true on success, false on failure
  bool postRequestOnChannel(std::shared_ptr<RDMASendRequest> req) {
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
      cc_inflight_bytes_.fetch_add(req->getLocalLen(),
                                   std::memory_order_relaxed);
    }
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

  // Build and post a single chunk from a split message.
  bool postSingleChunk(std::shared_ptr<RDMASendRequest> const& req,
                       MessageChunk const& chunk, size_t chunk_index,
                       size_t total_chunks, size_t num_channels,
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

  // Post remaining chunks from a previously paused request.
  // Returns true if all chunks are sent, false if still CC-blocked.
  bool drainPendingChunks() {
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

  void postChunkedRequest(std::shared_ptr<RDMASendRequest> req,
                          int expected_chunk_count = 0) {
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
        UCCL_LOG(WARN)
            << "SendConnection: Failed to send request on channel_id "
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

  void processSendRequests() {
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

  inline void compressSendRequest(std::shared_ptr<RDMASendRequest> req) {
    Compressor::getInstance().compress(req);
  }

  // Post a compressed-write segment as exactly `num_chunks` ~equal-sized
  // chunks, round-robin across data channels starting from req->channel_id.
  // Bypasses ChunkSplitStrategy's 512 KB granularity / kMaxSplitNum cap so
  // we can keep per-write WR count low — per-WR NIC overhead is the
  // dominant cost for compressed payloads on a fast NIC.
  void postCompressedSegment(std::shared_ptr<RDMASendRequest> const& req,
                             size_t seg_size, size_t num_chunks,
                             size_t num_channels) {
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
        UCCL_LOG(WARN) << "postCompressedSegment: chunk " << i
                       << " post failed";
      }
    }
  }

  // Two-phase compressed RDMA WRITE. Both phases land in the peer's
  // decompress_buffer at one slot, then a WriteReqMeta is pushed via the
  // control channel after all data WCs are observed locally (see
  // pollDataChannels).
  int64_t compressWriteRequestSplitFirst(
      std::shared_ptr<RDMASendRequest> req) {
    int64_t wr_id = req->wr_id;
    uint32_t total_uncomp = req->local_mem->size;
    uint64_t user_remote_addr = req->remote_mem->addr;
    uint32_t user_remote_rkey =
        req->remote_mem->getKeyByContextID(0);  // diagnostic / logging only

    // Reserve a contiguous region in the peer's decompress_buffer big
    // enough to hold the compressed payload (compressed_size ≤ total_uncomp
    // for FP32/BF16/FP16). Released on ack receipt in check().
    uint64_t arena_bytes = static_cast<uint64_t>(total_uncomp);
    uint64_t arena_offset = decompress_arena_.reserve(arena_bytes);
    uint32_t ack_slot = static_cast<uint32_t>(
        next_ack_slot_.fetch_add(1, std::memory_order_relaxed) % kAckRingDepth);
    static_cast<AckSlot*>(ack_ring_->addr)[ack_slot].value.store(
        0, std::memory_order_relaxed);
    uint64_t slot_base = remote_decompress_buf_.addr + arena_offset;

    // Phase 1: split → compress_buffer[0..first_seg). req->local_mem is
    // rewritten to point at it; redirect req->remote_mem to slot_base.
    Compressor::getInstance().compressSplitOneBatch(req);
    uint32_t first_seg = req->local_mem->size;
    req->remote_mem = std::make_shared<RemoteMemInfo>(
        slot_base, arena_bytes, remote_decompress_buf_.rkey_array,
        remote_decompress_buf_.type);

    auto [channel_id, _ch_ptr] = selectNextChannelRoundRobinFast();
    req->channel_id = channel_id ? channel_id : 1;
    size_t num_channels = normalChannelCount();
    if (num_channels == 0) num_channels = 1;
    // Phase 1 fans out to 2 chunks per channel (=8 WRs for 4 channels), which
    // matches the WR density used by non-compressed RDMA writes (~4 WR per
    // channel for 100 MB) — per-WR NIC setup cost would otherwise dominate.
    // Phase 2 by contrast is a single chunk on one channel (see comment
    // below where second_chunks is set).
    size_t first_chunks =
        std::min<size_t>(num_channels * 2, static_cast<size_t>(ChunkSplitStrategy::kMaxSplitNum));
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

    // Defer per-chunk flushes for the entire compressed write — submitRequest
    // would otherwise call flushBatch() per chunk (each chunk > the 32KB
    // threshold), turning the writev batch into one doorbell per chunk and
    // halving effective NIC throughput. We manually flush at the end of
    // phase 1 so kSplitOnly pipelining (NIC drains phase-1 while GPU runs
    // encode) is preserved; phase 2 is left in the batch for the outer
    // writev / write_async caller's flush to drain.
    struct DeferGuard {
      DeferGuard() { g_uccl_defer_flush = true; }
      ~DeferGuard() { g_uccl_defer_flush = false; }
    } defer_guard;

    postCompressedSegment(req, first_seg, first_chunks, num_channels);
    flushBatches();  // 1 doorbell per channel for phase 1, kicks NIC DMA

    // Phase 2: encode. compressEncodeOneBatch advances req->remote_mem->addr
    // by first_seg so the encoded tail lands at slot_base + first_seg.
    // (encode must run even if we skip the NIC post afterwards — it owns
    // releasing the split kernel's toComp_dev/histogram_dev temps in LIFO
    // order from dietgpu's StackDeviceMemory.)
    uint32_t compressed_size =
        Compressor::getInstance().compressEncodeOneBatch(req);

    // Diagnostic: UCCL_P2P_COMPRESS_SKIP_PHASE2=1 short-circuits ONLY the
    // phase-2 NIC post. encode kernel still runs (to clean up split temps).
    // Bench sees iter completion after phase 1's first_chunks WCs land —
    // useful for isolating whether phase 2 (or its interaction with phase 1
    // in NIC SQ) is what's costing the compressed path its throughput. NOT
    // functionally correct (receiver gets only the raw split bytes, can't
    // decompress) but bench doesn't verify payload.
    static bool const skip_phase2 = []() {
      char const* env = std::getenv("UCCL_P2P_COMPRESS_SKIP_PHASE2");
      return env && env[0] == '1';
    }();
    if (skip_phase2) {
      tracker_->updateExpectedAckCount(wr_id, first_chunks);
      return wr_id;
    }
    uint32_t second_seg = req->local_mem->size;
    // Phase 2 routed to a SINGLE chunk on one channel. Empirical finding:
    // when phase 2 chunks share QPs with phase 1 (mixed sized WRs on the
    // same SQ), NIC effective per-byte throughput drops to ~half. With one
    // chunk only one channel carries phase 2; the other channels run a
    // homogeneous phase-1-only stream at line rate, and the NIC's dynamic
    // per-QP bandwidth arbitration backfills the lone-phase-2 QP after the
    // others idle. Measured: 100MB random compressed 30 → 68 GB/s when
    // collapsing phase 2 from 8 chunks to 1.
    size_t second_chunks = second_seg > 0 ? 1 : 0;

    {
      std::lock_guard<std::mutex> lk(pending_compressed_mu_);
      pending_compressed_[wr_id].meta.compressed_size = compressed_size;
    }

    // Advance the round-robin channel pointer past phase 1's chunks so the
    // lone phase 2 chunk lands on the channel right after phase 1's last,
    // not back on phase 1's first channel. Affects which single channel
    // bears the extra phase-2 work (NIC's dynamic per-QP bandwidth
    // arbitration handles the imbalance from there).
    req->channel_id =
        ((req->channel_id - 1 + first_chunks) % num_channels) + 1;

    postCompressedSegment(req, second_seg, second_chunks, num_channels);
    // Strict semantic: wait for ALL chunk WCs (= data durable in receiver
    // GPU memory). Set expected to the real post count.
    tracker_->updateExpectedAckCount(wr_id, first_chunks + second_chunks);
    return wr_id;
  }

  inline void compressSendRequestSplitFirst(
      std::shared_ptr<RDMASendRequest> req, size_t expected_chunk_count) {
    Compressor::getInstance().compressSplitOneBatch(req);

    // compressed data / chunk size = chunk count
    uint32_t send_chunks_first =
        req->local_mem->size /
        ChunkSplitStrategy::getRegularChunkSize(req->compress_ctx->maxSize,
                                                expected_chunk_count);
    if (send_chunks_first > 0) {
      postChunkedRequest(req, send_chunks_first);
    }
    uint32_t uncompressed_size =
        Compressor::getInstance().compressEncodeOneBatch(req);
    postChunkedRequest(req, expected_chunk_count - send_chunks_first);
    tracker_->updateExpectedAckCount(
        req->wr_id,
        ChunkSplitStrategy::getMessageChunkCount(uncompressed_size));
  }

  inline void processOnceSendRequests(std::shared_ptr<RDMASendRequest> req,
                                      SendReqMeta& meta, int index) {
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

  void pollDataChannels() {
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

  // Scan the host ack ring for completed acks and release the corresponding
  // arena slot. Only checks ack_slots of currently in-flight compressed
  // writes (≈ tens of entries) — much cheaper than scanning all
  // kAckRingDepth (4096) slots per polling tick, which previously cost
  // ~256 KB of cache-line reads each tick and stole CPU/L3 from the user
  // thread.
  void pollAckRing() {
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

  // If wr_id corresponds to a compressed write whose data chunks have all
  // been locally ACKed (≡ RC-acknowledged by the peer NIC ≡ data is durable
  // in the peer's decompress_buffer), push its WriteReqMeta via the control
  // channel exactly once.
  void maybePushCompressedMeta(int64_t wr_id) {
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
        << " ack_slot=" << meta_to_push.ack_slot
        << " user_remote_addr=0x" << std::hex << meta_to_push.user_remote_addr
        << std::dec;
    std::shared_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    if (ctrl_channel_) ctrl_channel_->pushWriteMeta(meta_to_push, push_slot);
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

  void setRemoteAckRing(RemoteMemInfo const& m) {
    remote_ack_ring_ = m;
    // The compressed-write path needs the receive side to actively poll its
    // control-channel CQ for incoming WriteReqMeta IMMs — but for pure
    // one-sided writes the benchmark/user never calls recv_async +
    // poll_async, so user-driven recvRoutine() never runs. Force the
    // polling thread up here regardless of auto_start_polling_.
    if (!running_.load(std::memory_order_acquire)) startPolling();
  }

  void pollAndProcessCompletions() {
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

  // Run decompress on a freshly-arrived WriteReqMeta and post an 8-byte
  // RDMA WRITE INLINE ack back to the sender's ack_ring at the indicated
  // slot. The decompress call is synchronous (stream_sync inside), so by
  // the time we post the ack the user_remote_addr already holds the
  // reconstructed original data.
  // Heap-allocated context shipped from the receiver polling thread to the
  // CUDA host-callback thread. The callback posts the ack RDMA WRITE and
  // deletes the context.
  struct AsyncAckCtx {
    int64_t wr_id;
    uint64_t remote_ack_addr;
    uint32_t remote_ack_rkey;
    std::shared_ptr<RecvControlChannel> ctrl_channel;
    uint64_t value;  // 8-byte ack payload kept alive until ibv_post_send copies it inline
  };

  static void postAckHostFn(void* user_data) {
    std::unique_ptr<AsyncAckCtx> ctx(static_cast<AsyncAckCtx*>(user_data));
    ctx->value = static_cast<uint64_t>(ctx->wr_id);
    ctx->ctrl_channel->postAckWrite(ctx->remote_ack_addr, ctx->remote_ack_rkey,
                                    &ctx->value, sizeof(ctx->value));
  }

  void handleCompressedWriteArrival(WriteReqMeta const& m) {
    UCCL_LOG(INFO, UCCL_RDMA)
        << "RecvConnection::handleCompressedWriteArrival - wr_id=" << m.wr_id
        << " decompress_offset=" << m.decompress_offset
        << " compressed_size=" << m.compressed_size
        << " total_uncomp=" << m.total_uncomp_size
        << " float_type=" << m.float_type
        << " ack_slot=" << m.ack_slot;
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

    // ACK IMMEDIATELY — when the WriteReqMeta IMM arrives, the sender's data
    // chunks have already landed in our decompress_buffer (RC guarantees
    // order: sender only pushed the meta after all chunk WCs were local,
    // which means receiver NIC already wrote the data). We can recycle the
    // sender's arena slot NOW; we trust the buffer is large enough that the
    // sender's arena head won't wrap around before our decompress kernel
    // drains the slot (2 GB arena vs single-digit-ms decompress → tens of
    // ms before any wrap collision could occur).
    //
    // postAckWrite is INLINE so the 8-byte payload is copied into the WQE
    // before ibv_post_send returns — stack lifetime is safe.
    uint64_t ack_value = static_cast<uint64_t>(m.wr_id);
    ctrl_channel_->postAckWrite(
        remote_ack_ring_.addr + m.ack_slot * sizeof(AckSlot),
        remote_ack_ring_.getKeyByContextID(0), &ack_value, sizeof(ack_value));

    // Queue decompress kernel; no callback needed (ack already sent).
    Compressor::getInstance().decompressAsync(
        in, out, static_cast<uccl::FloatType>(m.float_type),
        /*on_done=*/nullptr, /*user_data=*/nullptr);
  }

 private:
  std::shared_ptr<RecvControlChannel> ctrl_channel_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  bool auto_start_polling_;
  int numa_node_ = 0;
  RemoteMemInfo remote_ack_ring_;  // peer (sender) ack ring — write target

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

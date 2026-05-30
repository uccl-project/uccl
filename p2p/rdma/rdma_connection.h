#pragma once
#include "common.h"
#include "compression.h"
#include "rdma_ctrl_channel.h"
#include "rdma_data_channel.h"
#include <cc/cc_state.h>
#include <cc/link_bandwidth.h>
#include <array>
#include <limits>
#include <optional>
#include <random>

class RDMAConnection {
 public:
  RDMAConnection();

  virtual ~RDMAConnection();

  virtual void add_channel(uint32_t channel_id,
                           std::shared_ptr<RDMADataChannel> channel);

  virtual std::shared_ptr<RDMADataChannel> get_channel(
      uint32_t channel_id) const;

  virtual size_t channel_count() const;

  virtual std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
  channels() const;

  // Select next channel using round-robin algorithm
  // Returns: pair<channel_id, context_id>, or pair<0, 0> on failure
  std::pair<uint32_t, uint64_t> select_next_channel_round_robin();

  // Select next channel using random selection algorithm
  // Returns: pair<channel_id, context_id>, or pair<0, 0> on failure
  std::pair<uint32_t, uint64_t> select_next_channel_random();

  // Lock-free hot-path channel cache. `channels_` is append-only during
  // connection setup, so once cached the snapshot stays valid for the
  // connection's lifetime. Falls back through the locked path if not built.
  RDMADataChannel* get_channel_fast(uint32_t channel_id) const;

  // Select next channel via round-robin without acquiring `mutex_` in the
  // common case. Returns (channel_id, channel_ptr); on first call this
  // primes the cache under a shared lock.
  std::pair<uint32_t, RDMADataChannel*> select_next_channel_round_robin_fast();

 protected:
  void build_fast_channel_cache();

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
  struct OneSidedBatchOp {
    void* local_addr = nullptr;
    size_t size = 0;
    MRArray const* local_mr_array = nullptr;
    FifoItem const* slot_item = nullptr;
  };

  SendConnection(int numa_node, bool auto_start_polling = true,
                 double link_bandwidth_bps = 400.0 * 1e9 / 8.0);

  ~SendConnection();

  void add_channel(uint32_t channel_id,
                   std::shared_ptr<RDMADataChannel> channel) override;

  std::shared_ptr<RDMADataChannel> get_channel(
      uint32_t channel_id) const override;

  size_t channel_count() const override;

  size_t normal_channel_count() const;

  std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
  channels() const override;

  template <typename T>
  void set_control_channel(T&& ctrl_channel) {
    std::unique_lock<std::shared_mutex> lock(ctrl_channel_mutex_);
    if (ctrl_channel_) {
      throw std::runtime_error(
          "SendConnection: Control channel has already been set");
    }
    ctrl_channel_ = std::forward<T>(ctrl_channel);
    lock.unlock();
    if (auto_start_polling_) {
      start_polling();
    }
  }

  int64_t post_write_or_read(std::shared_ptr<RDMASendRequest> req);

  int64_t read(std::shared_ptr<RDMASendRequest> req);

  // Start polling thread
  void start_polling();

  bool is_running() const { return running_.load(std::memory_order_acquire); }

  bool check(int64_t wr_id);

  bool can_use_raw_one_sided_batch(SendType send_type);

  bool post_write_or_read_batch(SendType send_type, OneSidedBatchOp const* ops,
                                size_t num_ops, int64_t* wr_ids);

  void set_remote_decompress_buf(RemoteMemInfo const& m);

  void set_local_ack_ring(std::shared_ptr<RegMemBlock> ring);

  // Stop polling thread
  void stop_polling();

  void polling_loop_for_meta();

  // Flush any batched send WRs on all channels of this connection. Used to
  // amortize doorbell cost across many small RDMA writes/reads posted via
  // g_uccl_batch_post.
  void flush_batches();

 private:
  std::shared_ptr<SendControlChannel> ctrl_channel_;
  mutable std::shared_mutex ctrl_channel_mutex_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  std::shared_ptr<AtomicBitmapPacketTrackerMultiAck> tracker_;
  bool auto_start_polling_;
  int numa_node_ = 0;

  uccl::cc::CongestionControlState cc_;
  std::atomic<uint32_t> chunk_tsc_counter_{0};

  // Compressed-write state
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

  // Bump-pointer allocator over the peer's decompress_buffer. Released by
  // poll_ack_ring() once the receiver's ack lands in ack_ring_.
  struct DecompressArena {
    uint64_t size = 0;
    uint64_t head = 0;
    std::map<uint64_t, uint64_t> inflight;  // offset → bytes
    std::mutex mu;
    std::condition_variable cv;

    uint64_t reserve(uint64_t bytes);

    void release(uint64_t offset);
  };
  DecompressArena decompress_arena_;
  // Per-chunk inflight byte counter for CC window checks.
  // Unlike tracker_->get_total_inflight_bytes() which only decreases when ALL
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

  size_t current_inflight_limit_bytes();

  // Return the inflight byte count, depends on CC enablement status
  size_t current_inflight_bytes();

  // Send a request through the appropriate channel
  // Returns true on success, false on failure
  bool post_request_on_channel(std::shared_ptr<RDMASendRequest> req);

  void poll_control_channel();

  // Build and post a single chunk from a split message.
  bool post_single_chunk(std::shared_ptr<RDMASendRequest> const& req,
                         MessageChunk const& chunk, size_t chunk_index,
                         size_t total_chunks, size_t num_channels,
                         int& expected_chunk_count);

  // Post remaining chunks from a previously paused request.
  // Returns true if all chunks are sent, false if still CC-blocked.
  bool drain_pending_chunks();

  void post_chunked_request(std::shared_ptr<RDMASendRequest> req,
                            int expected_chunk_count = 0);

  void compress_send_request(std::shared_ptr<RDMASendRequest> req);

  // Post `num_chunks` equal-sized chunks of a compressed segment, round-robin
  // across data channels. Bypasses ChunkSplitStrategy to keep WR count low.
  void post_compressed_segment(std::shared_ptr<RDMASendRequest> const& req,
                               size_t seg_size, size_t num_chunks,
                               size_t num_channels);

  // Two-phase compressed RDMA WRITE into one decompress_buffer slot.
  // WriteReqMeta is pushed after all data WCs land (see poll_data_channels).
  int64_t compress_write_request_split_first(
      std::shared_ptr<RDMASendRequest> req);

  void compress_send_request_split_first(std::shared_ptr<RDMASendRequest> req,
                                         size_t expected_chunk_count);

  void poll_data_channels();

  // Release arena slots for completed acks. Only scans in-flight entries.
  void poll_ack_ring();

  // Push WriteReqMeta once all data WCs for wr_id have arrived.
  void maybe_push_compressed_meta(int64_t wr_id);

  void polling_loop();
};

class RecvConnection : public RDMAConnection {
 public:
  RecvConnection(int numa_node, bool auto_start_polling = true);

  ~RecvConnection();

  void add_channel(uint32_t channel_id,
                   std::shared_ptr<RDMADataChannel> channel) override;

  std::shared_ptr<RDMADataChannel> get_channel(
      uint32_t channel_id) const override;

  size_t channel_count() const override;

  size_t normal_channel_count() const;

  std::unordered_map<uint32_t, std::shared_ptr<RDMADataChannel>> const&
  channels() const override;

  // Set the control channel
  template <typename T>
  void set_control_channel(T&& ctrl_channel) {
    if (ctrl_channel_) {
      throw std::runtime_error(
          "RecvConnection: Control channel has already been set");
    }
    ctrl_channel_ = std::forward<T>(ctrl_channel);
    if (auto_start_polling_) {
      start_polling();
    }
  }

  // Start polling thread
  void start_polling();

  // Stop polling thread
  void stop_polling();

  bool is_running() const { return running_.load(std::memory_order_acquire); }

  void set_remote_ack_ring(RemoteMemInfo const& m);

  void poll_and_process_completions();

  void polling_loop();

  // Context for the async-decompress callback path (currently unused).
  struct AsyncAckCtx {
    int64_t wr_id;
    uint32_t ack_slot;
    uint64_t remote_ack_addr;
    uint32_t remote_ack_rkey;
    std::shared_ptr<RecvControlChannel> ctrl_channel;
  };

  static void post_ack_host_fn(void* user_data);

  void handle_compressed_write_arrival(WriteReqMeta const& m);

 private:
  std::shared_ptr<RecvControlChannel> ctrl_channel_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> poll_thread_;
  bool auto_start_polling_;
  int numa_node_ = 0;
  RemoteMemInfo remote_ack_ring_;  // peer (sender) ack ring — write target
};

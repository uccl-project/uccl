#pragma once
#include "common.h"
#include "compression.h"
#include "epoll_client.h"
#include "epoll_server.h"
#include "memory_allocator.h"
#include "rdma_connection.h"
#include "rdma_context.h"
#include "rdma_device.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util/net.h"
#include <cc/link_bandwidth.h>
#include <unordered_map>
#include <unordered_set>

class RDMAEndpoint {
 public:
  // ── Lifecycle ──────────────────────────────────────────────────────────────
  explicit RDMAEndpoint(
      int gpu_index = INVALID_GPU, uint64_t port = 0,
      bool auto_start_polling = true,
      std::vector<size_t> const& device_ids = std::vector<size_t>());
  ~RDMAEndpoint();

  // ── Initialization ─────────────────────────────────────────────────────────
  bool initialize_rdma_ctx_for_gpu(
      int gpu_index,
      std::vector<size_t> const& device_ids = std::vector<size_t>());

  void init_compressor();

  void create_unified_p2p_socket();

  // ── Accessors ──────────────────────────────────────────────────────────────
  int gpu_index() const;

  size_t context_count() const;

  std::shared_ptr<RegMemBlock> ack_ring() const;

  std::shared_ptr<RegMemBlock> write_meta_ring() const;

  uint16_t get_p2p_listen_port();

  int get_p2p_listen_fd();

  std::shared_ptr<EpollClient> get_oob_client();

  std::string get_oob_conn_key(uint64_t peer_id);

  // ── Connection setup ───────────────────────────────────────────────────────
  void add_peer_oob_meta(
      std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> const&
          new_meta);

  ConnID uccl_connect(int remote_gpuidx, std::string remote_ip,
                      uint16_t remote_port);

  ConnID uccl_accept(std::string& remote_ip, int* remote_gpuidx);

  int build_connect(uint64_t peer_id, bool sync = true, int timeout_ms = 10000);

  void stop_accept();

  // ── Memory registration ────────────────────────────────────────────────────
  int uccl_regmr(void* const data, size_t const len, MRArray& mr_array,
                 std::vector<MrCacheHandleRef>& cache_refs,
                 CompressCtx compress_ctx = nullptr);

  void uccl_deregmr(std::vector<MrCacheHandleRef> const& cache_refs);

  // Register a buffer once per unique RdmaContext and broadcast the resulting
  // MR pointer to every context slot that shares that context. Context slots
  // can share an underlying device; naïvely using get_context_id() as the slot
  // key would clobber all slots onto context 0 and leave the others null.
  bool reg_mem(std::shared_ptr<RegMemBlock> reg_block);

  bool dereg_mem(std::shared_ptr<RegMemBlock> reg_block);

  // Populate compression-related RemoteMemInfo fields in a handshake message.
  // No-op if compression is disabled.
  void fill_compression_meta(MetaInfoToExchange& m) const;

  // ── One-sided transfer and completion ──────────────────────────────────────
  int64_t write_or_read(std::shared_ptr<RDMASendRequest> req);

  void check_send_complete(uint64_t peer_id, int64_t wr_id);

  bool check_send_complete_once(uint64_t peer_id, int64_t wr_id);

  // Resolve the SendConnection for a peer_id once so callers can poll many
  // completions without re-acquiring the mutex + map lookup per check.
  SendConnection* get_send_group_raw(uint64_t peer_id);

  // ── Polling and batching ───────────────────────────────────────────────────
  // Used when auto_start_polling_ is false.
  void recv_routine();

  void send_routine();

  // Flush batched send WRs across all send connections (g_uccl_batch_post).
  void flush_all_sends();

 private:
  // ── Context slots ──────────────────────────────────────────────────────────
  void initialize_contexts(std::vector<size_t> const& device_ids);

  std::shared_ptr<RdmaContext> get_context_by_channel_id(
      uint32_t channel_id) const;

  // ── Active-side connection handshake (outbound) ────────────────────────────
  std::string const build_oob_connect(uint64_t peer_id);

  int build_control_channel(std::string const& oob_con, uint64_t peer_id,
                            bool sync = true, int timeout_ms = 10000);

  bool build_data_channels(std::string const& oob_con, uint64_t peer_id,
                           bool sync = true, int timeout_ms = 10000);

  uint64_t handle_send_meta_response(std::shared_ptr<RDMADataChannel> channel,
                                     std::string const& response);

  // ── Passive-side connection handshake (inbound OOB server) ─────────────────
  void process_meta(std::string const& input, std::string& output,
                    std::string const& client_ip, int client_port);

  // ── SendConnection group registry ──────────────────────────────────────────
  std::shared_ptr<SendConnection> get_or_create_send_group(uint64_t peer_id);

  void add_one_send_channel(uint64_t peer_id, uint32_t channel_id,
                            std::shared_ptr<RDMADataChannel> new_channel);

  void set_send_control_channel(
      uint64_t peer_id, std::shared_ptr<SendControlChannel>&& ctrl_channel);

  void set_send_compression_peer_meta(uint64_t peer_id,
                                      MetaInfoToExchange const& peer);

  // ── RecvConnection group registry ──────────────────────────────────────────
  std::shared_ptr<RecvConnection> get_or_create_recv_group(uint64_t peer_id);

  void add_one_recv_channel(uint64_t peer_id, uint32_t channel_id,
                            std::shared_ptr<RDMADataChannel> new_channel);

  void set_recv_control_channel(
      uint64_t peer_id, std::shared_ptr<RecvControlChannel>&& ctrl_channel);

  void set_recv_compression_peer_meta(uint64_t peer_id,
                                      MetaInfoToExchange const& peer);

  // ── Members ────────────────────────────────────────────────────────────────
  int gpu_index_;
  std::vector<std::shared_ptr<RdmaContext>> contexts_;
  mutable std::shared_mutex recv_channel_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<RecvConnection>>
      recv_channel_groups_;
  mutable std::shared_mutex send_channel_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<SendConnection>>
      send_channel_groups_;

  std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> peer_oob_meta_;
  mutable std::shared_mutex peer_oob_conn_keys_mutex_;
  std::unordered_map<uint64_t, std::string> peer_oob_conn_keys_;
  std::shared_ptr<EpollClient> oob_client_;
  std::shared_ptr<EpollServer> oob_server_;
  std::shared_ptr<MemoryAllocator> allocator_;
  std::shared_ptr<RegMemBlock> ack_ring_;
  std::shared_ptr<RegMemBlock> write_meta_ring_;
  mutable std::shared_mutex accepted_meta_mutex_;
  std::unordered_map<uint64_t, AcceptedMeta> accepted_meta_;
  bool auto_start_polling_;
  std::atomic<int32_t> next_send_peer_id_;
  std::atomic<int32_t> next_recv_peer_id_;

  std::atomic<bool> stop_accept_{false};
};

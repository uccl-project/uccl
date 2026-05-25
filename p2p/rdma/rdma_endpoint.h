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
  explicit RDMAEndpoint(
      int gpu_index = INVALID_GPU, uint64_t port = 0,
      bool auto_start_polling = true,
      std::vector<size_t> const& device_ids = std::vector<size_t>());
  // Destructor
  ~RDMAEndpoint();

  void initCompressor();

  // Register a buffer once per UNIQUE RdmaContext and broadcast the resulting
  // MR pointer to every context slot that shares that context. Mirrors the
  // dedup logic in uccl_regmr — context slots can share an underlying device,
  // and naïvely using getContextID() as the slot key would clobber all slots
  // onto context 0 and leave the others null.
  void regMrForAllSlots(RegMemBlock& blk);

  std::shared_ptr<RegMemBlock> ackRing() const;
  std::shared_ptr<RegMemBlock> writeMetaRing() const;

  // Populate the three RemoteMemInfo fields in a Control-channel
  // MetaInfoToExchange. No-op if compression is disabled.
  void fillCompressionMeta(MetaInfoToExchange& m) const;
  int gpuIndex() const;

  size_t contextCount() const;

  bool regMem(std::shared_ptr<RegMemBlock> reg_block);

  bool deregMem(std::shared_ptr<RegMemBlock> reg_block);

  int build_connect(uint64_t peer_id, bool sync = true, int timeout_ms = 10000);

  // Blocking check for send completion
  void checkSendComplete(uint64_t peer_id, int64_t wr_id);

  bool checkSendComplete_once(uint64_t peer_id, int64_t wr_id);

  // Resolve the SendConnection for a given peer_id once so the caller can
  // perform many completion checks without re-acquiring the mutex + doing a
  // map lookup per check.
  SendConnection* getSendGroupRaw(uint64_t peer_id);

  bool checkRecvComplete_once(uint64_t peer_id, uint64_t index);

  // Blocking check for recv completion
  void checkRecvComplete(uint64_t peer_id, uint64_t index);

  int64_t writeOrRead(std::shared_ptr<RDMASendRequest> req);

  // Blocking send: wraps SendConnection::send with peer_id parameter
  // Returns wr_id for checking completion later
  int64_t send(uint64_t peer_id, std::shared_ptr<RDMASendRequest> req);

  // Blocking recv: wraps RecvConnection::recv with peer_id parameter
  // Returns index for checking completion later
  int64_t recv(uint64_t peer_id, std::shared_ptr<RDMARecvRequest> req);

  // Add or update peer OOB metadata from a given map
  void add_peer_oob_meta(
      std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> const&
          new_meta);
  ConnID uccl_connect(int remote_gpuidx, std::string remote_ip,
                      uint16_t remote_port);

  uint16_t get_p2p_listen_port();

  int get_p2p_listen_fd();

  std::shared_ptr<EpollClient> get_oob_client();

  std::string get_oob_conn_key(uint64_t peer_id);

  ConnID uccl_accept(std::string& remote_ip, int* remote_gpuidx);

  int uccl_regmr(void* const data, size_t const len, MRArray& mr_array,
                 std::vector<MrCacheHandleRef>& cache_refs,
                 CompressCtx compress_ctx = nullptr);

  void uccl_deregmr(std::vector<MrCacheHandleRef> const& cache_refs);

  bool initialize_rdma_ctx_for_gpu(
      int gpu_index,
      std::vector<size_t> const& device_ids = std::vector<size_t>());

  void create_unified_p2p_socket();

  // Manual polling routine for recv channels when auto_start_polling_ is false
  void recvRoutine();

  void sendRoutine();

  // Flush any batched send WRs across all send connections. Used after
  // posting many small one-sided RDMA requests in g_uccl_batch_post mode.
  void flushAllSends();

  // Manual polling routine for send channels when auto_start_polling_ is false
  int sendWithoutInnerQueue(std::shared_ptr<RDMASendRequest> req);

  void stop_accept();

 private:
  // Get context from channel_id
  std::shared_ptr<RdmaContext> getContextByChannelId(uint32_t channel_id) const;

  void initializeContexts(std::vector<size_t> const& device_ids);

  void process_meta(std::string const& input, std::string& output,
                    std::string const& client_ip, int client_port);

  // Handle response from send_meta operation
  uint64_t handle_send_meta_response(std::shared_ptr<RDMADataChannel> channel,
                                     std::string const& response);

  std::shared_ptr<RecvConnection> getOrCreateRecvGroup(uint64_t peer_id);

  void addOneRecvChannel(uint64_t peer_id, uint32_t channel_id,
                         std::shared_ptr<RDMADataChannel> new_channel);

  void setRecvControlChannel(
      uint64_t peer_id, std::shared_ptr<RecvControlChannel>&& ctrl_channel);

  std::shared_ptr<SendConnection> getOrCreateSendGroup(uint64_t peer_id);

  void addOneSendChannel(uint64_t peer_id, uint32_t channel_id,
                         std::shared_ptr<RDMADataChannel> new_channel);

  void setSendControlChannel(
      uint64_t peer_id, std::shared_ptr<SendControlChannel>&& ctrl_channel);

  // Hand the peer-side compression descriptors to the corresponding
  // connection so the compressed-write data path can target them.
  void setSendCompressionPeerMeta(uint64_t peer_id,
                                  MetaInfoToExchange const& peer);

  void setRecvCompressionPeerMeta(uint64_t peer_id,
                                  MetaInfoToExchange const& peer);

  std::string const build_oob_connect(uint64_t peer_id);

  int build_control_channel(std::string const& oob_con, uint64_t peer_id,
                            bool sync = true, int timeout_ms = 10000);

  bool build_data_channels(std::string const& oob_con, uint64_t peer_id,
                           bool sync = true, int timeout_ms = 10000);

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
  std::unordered_map<uint64_t, std::string>
      peer_oob_conn_keys_;  // Track conn_key per peer
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

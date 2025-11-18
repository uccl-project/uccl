#pragma once

#include "tcpx/device/unpack_launch.h"
#include "tcpx/include/tcpx_interface.h"
#include "tcpx/include/unpack_descriptor.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern thread_local bool inside_python;

namespace tcpx {

// ============================================================================
// Connection State
// ============================================================================
// Conn aggregates everything that lives for the life of a TCPX connection:
// control plane metadata, TCPX comm handles and the CUDA event pool used to
// synchronize GPU unpack work on the receive path. Events are preallocated and
// reused to keep the hot path free of cudaEventCreate/Destroy churn.
class Endpoint;

// Note: shared_from_this is not needed; connections are always managed via
// std::shared_ptr in conn_map_.
struct Conn {
  Conn() {
    recv_dev_handle = recv_dev_handle_storage.data();
    send_dev_handle = send_dev_handle_storage.data();
    std::memset(recv_dev_handle_storage.data(), 0,
                recv_dev_handle_storage.size());
    std::memset(send_dev_handle_storage.data(), 0,
                send_dev_handle_storage.size());
  }

  // Basic connection metadata
  uint64_t conn_id = 0;     // Unique connection identifier.
  std::string ip_addr;      // Remote IP string (hostname already resolved).
  int remote_gpu_idx = -1;  // Remote GPU ordinal.
  int remote_port = -1;     // Remote TCP control port.
  int ctrl_sock_fd = -1;    // TCP control-socket descriptor.

  // CUDA event pool used by recv-side GPU unpack.
  // Events are recycled via round-robin to avoid create/destroy overhead.
  // Each chunk borrows an event during Stage 2, then returns it when
  // stage2_done=true.
  std::vector<cudaEvent_t> recv_events;
  uint64_t event_counter = 0;  // Round-robin cursor for recv_events.

  // TCPX comm state
  void* send_comm = nullptr;  // TCPX send comm for this peer.
  void* recv_comm = nullptr;  // TCPX recv comm for this peer.
  // Note: send_dev_handle is not referenced by the engine today but is kept to
  // satisfy the tcpx_connect_v5 API contract (device-handle lifetime).
  void* send_dev_handle = nullptr;  // Points into send_dev_handle_storage.
  void* recv_dev_handle = nullptr;  // Points into recv_dev_handle_storage.

  // Host-side cached copy of the recv device handle
  tcpx::plugin::unpackNetDeviceHandle recv_dev_handle_host{};
  bool recv_dev_handle_cached = false;

  // Backing storage for device handles (aligned for NCCL requirements).
  alignas(16) std::array<uint8_t, 512> recv_dev_handle_storage;
  alignas(16) std::array<uint8_t, 512> send_dev_handle_storage;
};

// FIFO metadata item
// A fixed-size record describing a remotely readable slice. The passive side
// produces it via advertise(), passes it through the UCCL listener FIFO, and
// the active side feeds it into queue_read_response().
struct FifoItem {
  uint64_t mr_id;    // Memory registration identifier.
  uint32_t size;     // Payload size in bytes.
  uint32_t tag;      // Pre-allocated TCPX tag that the requester must use.
  uint64_t offset;   // Byte offset inside the registered region.
  uint64_t token;    // Reserved for future extensions/alignment.
  char padding[32];  // Pads the struct to 64 bytes as expected by UCCL.
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem size is not 64 bytes");

// Memory registration entry
// Tracks the host-visible representation of a registered buffer plus all TCPX
// handles created for individual connections. Registration happens lazily on
// first use per (connection, direction).
struct MrEntry {
  void* base = nullptr;          // Base address of the registered region.
  size_t size = 0;               // Length in bytes.
  int ptr_type = NCCL_PTR_CUDA;  // NCCL pointer type (defaults to device).

  // Cached TCPX handles keyed by conn_id.
  std::unordered_map<uint64_t, void*> send_handles;
  std::unordered_map<uint64_t, void*> recv_handles;
};

// Pending transfer
// Represents a logical transfer that has been sliced into
// UCCL_TCPX_CHUNK_BYTES. Each chunk progresses independently through:
//   Stage 0: posted to TCPX (schedule_*_chunks_locked)
//   Stage 1: network completion (poll_chunk_request_)
//   Stage 2: GPU completion/unpack (recv path only)
// Per-connection sliding windows bound the number of inflight chunks on both
// send and recv sides.
//
// Pipeline flow:
//   Send: Stage 0 -> send_queue -> Stage 1 (tcpx_test) -> release_send_slot ->
//   done Recv: Stage 0 -> recv_stage1_queue -> Stage 1 (tcpx_test) ->
//   recv_stage2_queue ->
//         Stage 2 (cudaEventQuery) -> release_recv_slot -> done
struct PendingTransfer {
  // Chunk-level bookkeeping.
  struct ChunkState {
    size_t offset = 0;          // Byte offset within the full payload.
    size_t bytes = 0;           // Chunk length.
    uint32_t tag = 0;           // TCPX tag (base_tag + chunk_idx).
    void* request = nullptr;    // TCPX request handle for this chunk.
    void* dst_ptr = nullptr;    // Destination ptr (recv target or send src).
    bool needs_unpack = false;  // Whether the recv path must run GPU unpack.

    // State progression flags:
    //   posted=false -> Stage 0 schedules -> posted=true
    //   posted=true, stage1_done=false -> Stage 1 polls -> stage1_done=true
    //   stage1_done=true, stage2_done=false -> Stage 2 polls ->
    //   stage2_done=true
    bool posted = false;       // Has been submitted to TCPX.
    bool stage1_done = false;  // Network completion observed (tcpx_test).
    bool stage2_done =
        false;  // GPU completion observed (cudaEventQuery) or send acked.

    // Bounce-buffer metadata (populated after Stage 1 on the recv path).
    rx::UnpackDescriptorBlock desc_block{};

    // CUDA event borrowed from Conn::recv_events to track kernel completion.
    cudaEvent_t event = nullptr;

    size_t event_idx = 0;  // Debug aid: index inside the pool.
  };

  enum class Kind {
    kSend,  // send_async pipeline (pure TX).
    kRecv,  // recv_async pipeline (requires unpack).
    kRead   // read_async pipeline (RX without unpack).
  };

  Kind kind = Kind::kRecv;
  uint64_t transfer_id = 0;
  uint64_t conn_id = 0;
  uint64_t mr_id = 0;
  size_t total_bytes = 0;
  uint32_t base_tag = 0;  // Starting tag; chunks add their index.
  size_t next_chunk_to_post = 0;
  void* mhandle = nullptr;  // TCPX memory handle cached in MrEntry.

  std::vector<ChunkState> chunks;
  size_t chunks_completed = 0;

  // Queues that mirror the pipeline stages.
  // Send path uses only send_queue; recv path uses both stage1 and stage2
  // queues.
  std::deque<size_t> send_queue;         // Post-send chunks waiting for test.
  std::deque<size_t> recv_stage1_queue;  // Posted recvs awaiting TCPX test.
  std::deque<size_t> recv_stage2_queue;  // Chunks waiting for CUDA completion.
};

// TCPX transport engine
// Endpoint exposes the TCPX-based backend used by the higher-level UCCL shim.
// It owns:
//   * Connection management (connect/accept + control-plane handshake)
//   * Memory registration and per-connection handle caching
//   * Asynchronous send/recv/read APIs with chunking and pipelining
//   * User-driven progress via explicit poll_async calls
class Endpoint {
 public:
  explicit Endpoint(uint32_t const num_cpus);
  ~Endpoint();

  // Connection management

  /**
   * Connect to a remote peer (client role).
   * Performs the TCP control-plane handshake, exchanges listen handles,
   * creates the send/recv comm pair and initializes CUDA-side resources.
   *
   * @param ip_addr          Remote IP address string.
   * @param remote_gpu_idx   Remote GPU ordinal (for logging/topology only).
   * @param remote_port      Remote TCP control port (negative to use default).
   * @param conn_id          [out] New connection identifier.
   * @return                 true on success, false on failure.
   */
  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);

  /**
   * Accept an inbound connection (server role).
   * Mirrors connect() in reverse order and exposes the peer metadata.
   *
   * @param ip_addr          [out] Peer IP address string.
   * @param remote_gpu_idx   [out] Peer GPU ordinal.
   * @param conn_id          [out] New connection identifier.
   * @return                 true on success, false on failure.
   */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  /**
   * Get endpoint metadata for connection establishment (IP/port/GPU tuple).
   *
   * @return                 Serialized metadata bytes.
   */
  std::vector<uint8_t> get_unified_metadata();

  /**
   * Parse endpoint metadata produced by get_unified_metadata().
   *
   * @param metadata         Serialized bytes.
   * @return                 (ip, port, gpu) tuple.
   */
  static std::tuple<std::string, uint16_t, int> parse_metadata(
      std::vector<uint8_t> const& metadata);

  // READ/FIFO helpers.
  /**
   * Advertise a remote-readable slice via FIFO.
   * The helper writes a fixed-width FifoItem using a fresh TCPX tag.
   *
   * @param conn_id          Connection identifier (informational).
   * @param mr_id            Memory registration id.
   * @param addr             Start address of the slice.
   * @param len              Length in bytes.
   * @param out_buf          [out] Pointer to a 64-byte buffer to receive the
   * item.
   * @return                 true on success, false on failure.
   */
  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);
  /**
   * Respond to a FIFO item by posting the matching send.
   *
   * @param conn_id          Connection identifier.
   * @param fifo_item        FIFO item produced by the passive peer.
   * @return                 true on success, false on failure.
   */
  bool queue_read_response(uint64_t conn_id, FifoItem const& fifo_item);

  uint32_t allocate_tag() { return next_tag_.fetch_add(1); }

  /**
   * Get the TCP control-socket file descriptor for a connection.
   *
   * @param conn_id          Connection identifier.
   * @return                 File descriptor, or -1 if not found.
   */
  int get_sock_fd(uint64_t conn_id) const {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return -1;
    return it->second->ctrl_sock_fd;
  }

  // Memory registration.
  /**
   * Register device memory with the TCPX backend.
   *
   * @param data             Base device pointer.
   * @param size             Size in bytes.
   * @param mr_id            [out] Assigned memory registration id.
   * @return                 true on success, false on failure.
   */
  bool reg(void const* data, size_t size, uint64_t& mr_id);
  /**
   * Deregister a previously registered region and release per-connection
   * handles.
   *
   * @param mr_id            Memory registration id to deregister.
   * @return                 true on success, false on failure.
   */
  bool dereg(uint64_t mr_id);

  // Asynchronous data path.
  /**
   * Issue a tagged receive described by a passive-side FIFO item.
   *
   * @param conn_id          Connection identifier.
   * @param mr_id            Memory registration containing destination buffer.
   * @param dst              Destination pointer.
   * @param size             Destination capacity in bytes.
   * @param slot_item        FIFO item (provides size and tag).
   * @param transfer_id      [out] Transfer id to use with poll_async.
   * @return                 true on success, false on failure.
   */
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  FifoItem const& slot_item, uint64_t* transfer_id);
  /**
   * Start an asynchronous send with an auto-assigned tag.
   *
   * @param conn_id          Connection identifier.
   * @param mr_id            Memory registration containing source buffer.
   * @param data             Source pointer.
   * @param size             Number of bytes to send.
   * @param transfer_id      [out] Transfer id to use with poll_async.
   * @return                 true on success, false on failure.
   */
  bool send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                  size_t size, uint64_t* transfer_id);
  /**
   * Start an asynchronous send with an explicit TCPX tag.
   *
   * @param conn_id          Connection identifier.
   * @param mr_id            Memory registration containing source buffer.
   * @param data             Source pointer.
   * @param size             Number of bytes to send.
   * @param tag              TCPX tag to use for this transfer.
   * @param transfer_id      [out] Transfer id to use with poll_async.
   * @return                 true on success, false on failure.
   */
  bool send_async_with_tag(uint64_t conn_id, uint64_t mr_id, void const* data,
                           size_t size, uint32_t tag, uint64_t* transfer_id);
  /**
   * Start an asynchronous receive with an auto-assigned tag.
   *
   * @param conn_id          Connection identifier.
   * @param mr_id            Memory registration containing destination buffer.
   * @param data             Destination pointer.
   * @param size             Number of bytes to receive.
   * @param transfer_id      [out] Transfer id to use with poll_async.
   * @return                 true on success, false on failure.
   */
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);
  /**
   * Start a tagged asynchronous receive (used by read_async).
   *
   * @param conn_id          Connection identifier.
   * @param mr_id            Memory registration containing destination buffer.
   * @param data             Destination pointer.
   * @param size             Number of bytes to receive.
   * @param tag              TCPX tag to match the peer's send.
   * @param transfer_id      [out] Transfer id to use with poll_async.
   * @return                 true on success, false on failure.
   */
  bool recv_async_with_tag(uint64_t conn_id, uint64_t mr_id, void* data,
                           size_t size, uint32_t tag, uint64_t* transfer_id);

  /**
   * Drive progress for a single transfer.
   * Steps Stage 0 (submission), Stage 1 (TCPX completion) and Stage 2 (CUDA
   * completion for recv paths). Callers poll until is_done becomes true.
   *
   * @param transfer_id      Transfer id returned by an async call.
   * @param is_done          [out] Set to true when the transfer has finished.
   * @return                 true on success, false on failure.
   */
  bool poll_async(uint64_t transfer_id, bool* is_done);

 private:
  // Device / control-plane state.
  int dev_id_ = -1;
  int ctrl_listen_fd_ = -1;
  void* listen_comms_ = nullptr;
  uint32_t local_gpu_idx_ = 0;
  int ctrl_port_ = 0;
  ncclNetHandle_v7 listen_handle_{};

  // Monotonic id/tag generators.
  std::atomic<uint64_t> next_conn_id_{0};
  std::atomic<uint64_t> next_mr_id_{1};
  std::atomic<uint64_t> next_transfer_id_{1};
  std::atomic<uint32_t> next_tag_{1};

  // Shared connection table guarded by conn_mu_.
  mutable std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, std::shared_ptr<Conn>> conn_map_;

  // Registered-memory cache guarded by mr_mu_.
  mutable std::mutex mr_mu_;
  std::unordered_map<uint64_t, MrEntry> mr_map_;

  // Pending transfers guarded by transfer_mu_.
  mutable std::mutex transfer_mu_;
  std::unordered_map<uint64_t, PendingTransfer> transfer_map_;

  // Recv-side CUDA resources.
  cudaStream_t unpack_stream_ = nullptr;
  std::unique_ptr<device::UnpackLauncher> unpack_launcher_;

  // Tunables sourced from the environment.
  size_t chunk_bytes_ = 0;
  size_t max_send_inflight_ = 0;
  size_t max_recv_inflight_ = 0;
  bool debug_enabled_ = false;
  CUdevice cu_device_ = 0;
  CUcontext cu_context_ = nullptr;

  // Per-connection inflight counters guarded by window_mu_.
  // Sliding window flow control:
  //   - send_inflight_chunks_: incremented in reserve_send_slot (Stage 0),
  //                            decremented in release_send_slot (Stage 1 done)
  //   - recv_inflight_chunks_: incremented in reserve_recv_slot (Stage 0),
  //                            decremented in release_recv_slot (Stage 2 done)
  mutable std::mutex window_mu_;
  std::unordered_map<uint64_t, size_t> send_inflight_chunks_;
  std::unordered_map<uint64_t, size_t> recv_inflight_chunks_;

  enum class ScheduleOutcome { kNoProgress, kProgress, kError };

  /**
   * Release all resources associated with a connection.
   *
   * @param conn             Shared pointer to the connection to free.
   */
  void free_conn_(std::shared_ptr<Conn> const& conn);

  // Stage-0 chunk scheduling (submit TCPX requests while respecting inflight
  // windows).
  /**
   * Stage 0 (TX): post send chunks while respecting the send window.
   *
   * @param conn             Connection (transfer_mu_ held by caller).
   * @param transfer         Transfer state to update.
   * @return                 Progress outcome.
   */
  ScheduleOutcome schedule_send_chunks_locked(Conn& conn,
                                              PendingTransfer& transfer);
  /**
   * Stage 0 (RX): post recv chunks while respecting the recv window.
   *
   * @param conn             Connection (transfer_mu_ held by caller).
   * @param transfer         Transfer state to update.
   * @return                 Progress outcome.
   */
  ScheduleOutcome schedule_recv_chunks_locked(Conn& conn,
                                              PendingTransfer& transfer);

  // Sliding window helpers. These keep per-connection inflight limits for
  // Stage 0. reserve_* returns false when the window is full; release_* frees
  // slots once Stage 2 concludes.
  //
  // Timing:
  //   Send: reserve (Stage 0) -> release (Stage 1 done)
  //   Recv: reserve (Stage 0) -> release (Stage 2 done, after GPU unpack)
  /**
   * Reserve a unit of the per-connection send window.
   * @return                 true if reserved, false if window is full.
   */
  bool reserve_send_slot(uint64_t conn_id, size_t limit);
  /**
   * Reserve a unit of the per-connection recv window.
   * @return                 true if reserved, false if window is full.
   */
  bool reserve_recv_slot(uint64_t conn_id, size_t limit);
  /**
   * Release a unit of the per-connection send window (called after Stage 1).
   */
  void release_send_slot(uint64_t conn_id);
  /**
   * Release a unit of the per-connection recv window (called after Stage 2).
   */
  void release_recv_slot(uint64_t conn_id);

  // Memory-handle helpers.
  /**
   * Ensure a TCPX memory handle exists for (conn, mr_id, direction).
   *
   * @param conn             Connection to use.
   * @param mr_id            Memory registration id.
   * @param is_recv          Direction: false=send, true=recv.
   * @param mhandle_out      [out] Resulting TCPX handle pointer.
   * @return                 true on success, false on failure.
   */
  bool populate_conn_handles_(Conn& conn, uint64_t mr_id, bool is_recv,
                              void** mhandle_out);
  /**
   * Cache the device handle used by the recv path (host copy).
   */
  bool ensure_recv_dev_handle_cached_(Conn& conn);

  // Stage-1/2 progress (network completion + GPU completion). Stage 1 is
  // handled inside progress_transfer_locked by polling TCPX; Stage 2 applies
  // only to recv/read paths and tracks CUDA events associated with each chunk.
  /**
   * Stage 1/2 progression for a single transfer.
   * Polls network completions (Stage 1) and GPU completions (Stage 2 for recv).
   * Returns hints to re-run Stage 0 if window slots were freed.
   *
   * @param conn             Connection (transfer_mu_ held by caller).
   * @param transfer         Transfer to progress.
   * @param schedule_send    [out] Hint to re-run TX Stage 0.
   * @param schedule_recv    [out] Hint to re-run RX Stage 0.
   * @return                 true on success, false on failure.
   */
  bool progress_transfer_locked(Conn& conn, PendingTransfer& transfer,
                                bool* schedule_send, bool* schedule_recv);
  /**
   * Central driver: Stage 0 followed by Stage 1/2.
   * Called by poll_async() to advance a transfer through all pipeline stages.
   * Implements pipelining: if Stage 1/2 free window slots, Stage 0 runs again.
   *
   * @param conn             Connection (transfer_mu_ held by caller).
   * @param transfer         Transfer to progress.
   * @param transfer_complete [out] Set when all chunks finished.
   * @return                 true on success, false on failure.
   */
  bool advance_transfer_locked(Conn& conn, PendingTransfer& transfer,
                               bool* transfer_complete);

  /**
   * Remove a completed transfer from the map (transfer_mu_ held).
   */
  void finalize_transfer_locked(
      std::unordered_map<uint64_t, PendingTransfer>::iterator it);
  /**
   * Clear per-connection window counters after a transfer completes.
   */
  void reset_conn_window_counters_(uint64_t conn_id);

  // Chunk-level helpers. poll_chunk_request_ is the Stage-1 poller, enqueue /
  // finalize cover Stage-2 for receive paths (launching + completing GPU
  // unpack).
  /**
   * Stage 1 poller: check TCPX request completion for a chunk.
   * Calls tcpx_test() to detect network completion.
   *
   * @param transfer         Owning transfer.
   * @param chunk            Chunk state to update.
   * @param done             [out] Completion flag.
   * @param received_size    [out] Size reported by TCPX (RX only).
   * @return                 true on success, false on failure.
   */
  bool poll_chunk_request_(PendingTransfer& transfer,
                           PendingTransfer::ChunkState& chunk, bool* done,
                           int* received_size);
  /**
   * Stage 2 kickoff (RX): enqueue GPU unpack for a completed chunk.
   * Extracts bounce buffer metadata, launches GPU kernel, records CUDA event.
   */
  bool enqueue_chunk_unpack_(PendingTransfer& transfer,
                             PendingTransfer::ChunkState& chunk,
                             tcpx::plugin::tcpxRequest* request, Conn& conn);
  /**
   * Stage 2 epilogue (RX): notify TCPX the chunk was consumed and clear state.
   * Calls tcpx_irecv_consumed() to release bounce buffer.
   */
  bool finalize_recv_chunk_(Conn& conn, PendingTransfer::ChunkState& chunk);

  // Transfer submission helpers.
  /**
   * Build and register a send transfer, then run Stage 0 to post initial
   * chunks. Splits transfer into chunks, assigns tags, and calls
   * schedule_send_chunks_locked.
   */
  bool post_send_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                  void const* data, size_t size, int tag,
                  uint64_t& transfer_id);
  /**
   * Build and register a recv/read transfer, then run Stage 0 to post initial
   * chunks. The needs_unpack flag determines whether Stage 2 GPU unpack is
   * required.
   */
  bool post_recv_(Conn& conn, uint64_t mr_id, MrEntry const& mr, void* data,
                  size_t size, int tag, uint64_t& transfer_id,
                  bool needs_unpack);
};

}  // namespace tcpx

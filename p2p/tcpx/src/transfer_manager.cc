/**
 * @file transfer_manager.cc
 * @brief Implementation of TcpxTransfer class
 *
 * This file implements the TCPX transfer logic, faithfully following the
 * validated implementation from test_tcpx_perf_multi.cc (lines 523-705).
 */

#include "transfer_manager.h"
#include "channel_manager.h"
#include "device/unpack_launch.h"
#include "session_manager.h"
#include "tcpx_logging.h"
#include "transfer_flow.h"
#include "unpack_descriptor.h"
#include <chrono>
#include <deque>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// TCPX plugin API (already declared in tcpx_interface.h)
// No need to redeclare here

namespace tcpx {

// ============================================================================
// TcpxTransfer::Impl - Private implementation
// ============================================================================

struct TcpxTransfer::Impl {
  // Parent session (not owned)
  TcpxSession* session_;
  std::string remote_name_;

  // Per-channel sliding window state (for recv only)
  std::vector<ChannelWindow> channel_windows_;

  // Per-channel send inflight counters
  std::vector<int> send_inflight_count_;

  // Send queue per channel (head-of-line polling)
  struct SendRequest {
    void* request;
    int global_idx;
  };
  std::vector<std::deque<SendRequest>> send_queues_;

  // Transfer state
  int total_chunks_ = 0;      // Total chunks posted (send + recv)
  int completed_chunks_ = 0;  // Completed chunks (send + recv)
  bool completed_ = false;    // Transfer complete flag
  int next_channel_ = 0;      // Round-robin channel selection

  static constexpr int kMaxSendInflightPerChannel = 12;
  bool error_ = false;

  // ============================================================================
  // Constructor / Destructor
  // ============================================================================

  Impl(TcpxSession* session, std::string const& remote_name);
  ~Impl();

  // ============================================================================
  // Two-Stage Pipeline (from test_tcpx_perf_multi.cc)
  // ============================================================================

  /**
   * @brief Stage 1: Process inflight_recvs → pending_reqs
   *
   * This implements the first part of process_completed_chunk():
   * - Poll tcpx_test() for completed receives
   * - Build unpack descriptor from TCPX metadata
   * - Launch GPU unpack kernel
   * - Record CUDA event
   * - Move from inflight_recvs to pending_reqs
   *
   * Based on test_tcpx_perf_multi.cc lines 530-658.
   *
   * @param channel_id Channel to process
   * @param blocking If true, block until at least one chunk completes
   * @return true on success, false on error
   */
  bool processInflightRecv(int channel_id, bool blocking);

  /**
   * @brief Stage 2: Process pending_reqs → completed (recv only)
   *
   * This wraps tcpx::drainCompletedKernels() helper:
   * - Query CUDA events for completed kernels
   * - Call tcpx_irecv_consumed() to release TCPX resources
   * - Remove from pending_reqs
   *
   * Based on test_tcpx_perf_multi.cc Stage 2 logic (extracted to helper).
   *
   * @param channel_id Channel to process
   * @return true on success, false on error
   */
  bool drainCompletedKernels(int channel_id);

  /**
   * @brief Poll send completions (send only)
   *
   * This polls tcpx_test() for completed send requests:
   * - Check each pending send request
   * - Remove completed sends from per-channel send queues
   * - Update completed_chunks_ counter
   *
   * @return true on success, false on error
   */
  bool drainCompletedSends();

  /**
   * @brief Wait for capacity in the sliding window
   *
   * This implements wait_for_channel_capacity():
   * - If window full, drain completed kernels (Stage 2)
   * - If still full, process inflight recvs (Stage 1)
   * - Block until capacity available
   *
   * Based on test_tcpx_perf_multi.cc lines 663-705.
   *
   * @param channel_id Channel to wait on
   * @return true on success, false on error
   */
  bool waitForCapacity(int channel_id);

  /**
   * @brief Wait for send capacity on a channel
   *
   * Blocks until send inflight count falls below limit.
   */
  bool waitForSendCapacity(int channel_id);

  bool progressSendWindow(int channel_id);

  bool progressSendQueue(int channel_id, bool blocking);
};

// ============================================================================
// TcpxTransfer::Impl Implementation
// ============================================================================

TcpxTransfer::Impl::Impl(TcpxSession* session, std::string const& remote_name)
    : session_(session), remote_name_(remote_name) {
  // Get number of channels from session
  int num_channels = session->getNumChannels();

  // Initialize channel windows
  // Based on test_tcpx_perf_multi.cc lines 496-509
  channel_windows_.resize(num_channels);
  for (int ch = 0; ch < num_channels; ++ch) {
    channel_windows_[ch].events.resize(MAX_INFLIGHT_PER_CHANNEL);
    for (int i = 0; i < MAX_INFLIGHT_PER_CHANNEL; ++i) {
      cudaError_t cuda_rc = cudaEventCreate(&channel_windows_[ch].events[i]);
      if (cuda_rc != cudaSuccess) {
        LOG_ERROR("cudaEventCreate failed for channel %d, event %d: %s", ch, i,
                  cudaGetErrorString(cuda_rc));
        // Clean up already-created events
        for (int cleanup_ch = 0; cleanup_ch <= ch; ++cleanup_ch) {
          int max_evt = (cleanup_ch == ch) ? i : MAX_INFLIGHT_PER_CHANNEL;
          for (int cleanup_i = 0; cleanup_i < max_evt; ++cleanup_i) {
            cudaEventDestroy(channel_windows_[cleanup_ch].events[cleanup_i]);
          }
        }
        throw std::runtime_error("Failed to create CUDA events");
      }
    }
  }

  send_inflight_count_.assign(num_channels, 0);
  send_queues_.resize(num_channels);

  LOG_INFO("TcpxTransfer::Impl initialized with %d channels", num_channels);
}

TcpxTransfer::Impl::~Impl() {
  // Destroy all CUDA events
  // Based on test_tcpx_perf_multi.cc cleanup logic
  for (auto& win : channel_windows_) {
    for (auto& evt : win.events) {
      if (evt) {
        cudaEventDestroy(evt);
      }
    }
  }
}

bool TcpxTransfer::Impl::processInflightRecv(int channel_id, bool blocking) {
  // Faithfully implements test_tcpx_perf_multi.cc lines 525-660
  // This is Stage 1 of the two-stage pipeline

  auto& win = channel_windows_[channel_id];
  auto* mgr = static_cast<ChannelManager*>(session_->getChannelManager());
  auto& ch = mgr->get_channel(channel_id);
  int const kSleepMicros = 10;

  while (!win.inflight_recvs.empty()) {
    auto& entry = win.inflight_recvs.front();
    int done = 0;
    int received_size = 0;
    int test_rc = tcpx_test(entry.request, &done, &received_size);

    // Handle errors (based on lines 536-576)
    if (test_rc != 0) {
      // Special case: rc=2 (connection closed by peer)
      if (test_rc == 2) {
        if (done == 1) {
          // Data was received successfully before connection closed
          LOG_INFO(
              "Connection closed by peer after chunk %d completed on channel "
              "%d",
              entry.global_idx, channel_id);
          // Continue processing this chunk
        } else {
          // Connection closed but data not yet complete – transient
          LOG_WARN(
              "Connection closed by peer while chunk %d on channel %d still in "
              "progress (done=0, will retry)",
              entry.global_idx, channel_id);
          if (blocking) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(kSleepMicros));
            continue;
          } else {
            break;
          }
        }
      } else {
        // Other errors are real errors
        LOG_ERROR("tcpx_test failed (rc=%d, done=%d) for channel %d chunk %d",
                  test_rc, done, channel_id, entry.global_idx);
        error_ = true;
        return false;
      }
    }

    // If not done yet, handle based on blocking mode (lines 578-586)
    if (!done) {
      if (blocking) {
        std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicros));
        continue;
      }
      break;
    }

    LOG_DEBUG("Chunk %d recv completed (received_size=%d)", entry.global_idx,
              received_size);

    // Extract TCPX metadata for unpack (lines 592-615)
    auto* rx_req = reinterpret_cast<tcpx::plugin::tcpxRequest*>(entry.request);
    auto* dev_handle_struct =
        reinterpret_cast<tcpx::plugin::NcclNetDeviceHandle*>(
            ch.recv_dev_handle);

    if (!rx_req || !dev_handle_struct || !rx_req->unpack_slot.mem ||
        !rx_req->unpack_slot.cnt) {
      LOG_ERROR("Missing TCPX metadata for unpack (channel %d, chunk %d)",
                channel_id, entry.global_idx);
      error_ = true;
      return false;
    }

    uint64_t frag_count = *(rx_req->unpack_slot.cnt);
    LOG_DEBUG("Chunk %d has %lu fragments", entry.global_idx, frag_count);

    if (frag_count == 0 || frag_count > MAX_UNPACK_DESCRIPTORS) {
      LOG_ERROR("Invalid fragment count: %lu (cnt_cache=%lu)", frag_count,
                rx_req->unpack_slot.cnt_cache);
      error_ = true;
      return false;
    }

    // Copy device handle from GPU memory (lines 617-624)
    tcpx::plugin::unpackNetDeviceHandle dev_handle{};
    CUresult cu_rc = cuMemcpyDtoH(
        &dev_handle, reinterpret_cast<CUdeviceptr>(dev_handle_struct->handle),
        sizeof(dev_handle));
    if (cu_rc != CUDA_SUCCESS) {
      LOG_ERROR("cuMemcpyDtoH device handle failed: %d", cu_rc);
      error_ = true;
      return false;
    }

    // Build unpack descriptor block (lines 626-633)
    auto* meta_entries =
        static_cast<tcpx::plugin::loadMeta*>(rx_req->unpack_slot.mem);
    tcpx::rx::UnpackDescriptorBlock desc_block;
    tcpx::rx::buildDescriptorBlock(
        meta_entries, static_cast<uint32_t>(frag_count), dev_handle.bounce_buf,
        entry.dst_ptr, desc_block);
    desc_block.ready_flag = rx_req->unpack_slot.cnt;
    desc_block.ready_threshold = frag_count;

    // Launch GPU kernel to unpack scattered fragments (lines 635-645)
    LOG_DEBUG(
        "Launching unpack kernel for chunk %d (channel %d, descriptors=%u)",
        entry.global_idx, channel_id, desc_block.count);

    auto* launcher = static_cast<tcpx::device::UnpackLauncher*>(
        session_->getUnpackLauncher());
    int lrc = launcher->launch(desc_block);
    if (lrc != 0) {
      LOG_ERROR("Unpack kernel launch failed: %d", lrc);
      error_ = true;
      return false;
    }

    // Record CUDA event (lines 647-651)
    int event_idx = win.chunk_counter % MAX_INFLIGHT_PER_CHANNEL;
    cudaStream_t unpack_stream =
        static_cast<cudaStream_t>(session_->getUnpackStream());
    cudaError_t cuda_rc = cudaEventRecord(win.events[event_idx], unpack_stream);
    if (cuda_rc != cudaSuccess) {
      LOG_ERROR("cudaEventRecord failed: %s", cudaGetErrorString(cuda_rc));
      error_ = true;
      return false;
    }

    // Move to pending queue (lines 653-657)
    win.pending_reqs.push_back(entry.request);
    win.pending_indices.push_back(win.chunk_counter);
    win.chunk_counter++;
    win.inflight_recvs.pop_front();
  }

  return true;
}

bool TcpxTransfer::Impl::drainCompletedKernels(int channel_id) {
  // Wrap the helper function from tcpx_helpers.cc
  // Based on Stage 2 of process_completed_chunk

  auto& win = channel_windows_[channel_id];
  auto* mgr = static_cast<ChannelManager*>(session_->getChannelManager());
  auto& ch = mgr->get_channel(channel_id);

  // Use the helper function from tcpx_helpers.cc
  int completed = 0;
  bool success = tcpx::drainCompletedKernels(win, ch.recv_comm, completed);

  if (success) {
    completed_chunks_ += completed;
  } else {
    error_ = true;
  }

  return success;
}

bool TcpxTransfer::Impl::drainCompletedSends() {
  for (int ch = 0; ch < session_->getNumChannels(); ++ch) {
    if (!progressSendQueue(ch, /*blocking=*/false)) {
      return false;
    }
  }
  return true;
}

bool TcpxTransfer::Impl::waitForCapacity(int channel_id) {
  // Based on test_tcpx_perf_multi.cc lines 663-705
  // This implements the complete wait_for_channel_capacity logic

  auto& win = channel_windows_[channel_id];
  auto* mgr = static_cast<ChannelManager*>(session_->getChannelManager());
  auto& ch = mgr->get_channel(channel_id);

  while (win.pending_reqs.size() + win.inflight_recvs.size() >=
         MAX_INFLIGHT_PER_CHANNEL) {
    // First, try to drain completed kernels (Stage 2)
    if (!win.pending_reqs.empty()) {
      int oldest_idx = win.pending_indices.front();
      void* oldest_req = win.pending_reqs.front();
      cudaEvent_t oldest_event =
          win.events[oldest_idx % MAX_INFLIGHT_PER_CHANNEL];

      LOG_DEBUG(
          "Channel %d sliding window FULL (%zu+%zu/%d), waiting for chunk %d "
          "kernel to complete...",
          channel_id, win.pending_reqs.size(), win.inflight_recvs.size(),
          MAX_INFLIGHT_PER_CHANNEL, oldest_idx);

      // Block until kernel completes
      cudaError_t cuda_rc = cudaEventSynchronize(oldest_event);
      if (cuda_rc != cudaSuccess) {
        LOG_ERROR("cudaEventSynchronize failed: %s",
                  cudaGetErrorString(cuda_rc));
        return false;
      }

      // Release TCPX resources
      tcpx_irecv_consumed(ch.recv_comm, 1, oldest_req);
      win.pending_reqs.erase(win.pending_reqs.begin());
      win.pending_indices.erase(win.pending_indices.begin());
      completed_chunks_++;

      LOG_DEBUG("Channel %d window now has %zu+%zu outstanding", channel_id,
                win.pending_reqs.size(), win.inflight_recvs.size());
      continue;
    }

    // If pending is empty but inflight is not, process inflight (Stage 1)
    if (!win.inflight_recvs.empty()) {
      if (!processInflightRecv(channel_id, /*blocking=*/true)) {
        return false;
      }
      continue;
    }

    // Both queues empty, we have capacity
    break;
  }

  return true;
}

bool TcpxTransfer::Impl::waitForSendCapacity(int channel_id) {
  int const kSleepMicros = 10;
  auto& queue = send_queues_[channel_id];

  while (queue.size() >= kMaxSendInflightPerChannel) {
    if (!progressSendQueue(channel_id, /*blocking=*/true)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicros));
  }
  return true;
}

bool TcpxTransfer::Impl::progressSendWindow(int channel_id) {
  bool ok = progressSendQueue(channel_id, /*blocking=*/false);
  int total = session_->getNumChannels();
  for (int ch = 0; ch < total; ++ch) {
    if (ch == channel_id) continue;
    ok = progressSendQueue(ch, /*blocking=*/false) && ok;
  }
  return ok;
}

bool TcpxTransfer::Impl::progressSendQueue(int channel_id, bool blocking) {
  if (channel_id < 0 || channel_id >= static_cast<int>(send_queues_.size())) {
    return true;
  }

  auto& queue = send_queues_[channel_id];
  int const kSleepMicros = 10;

  while (!queue.empty()) {
    auto& entry = queue.front();

    int done = 0;
    int size = 0;
    int rc = tcpx_test(entry.request, &done, &size);

    if (rc < 0) {
      LOG_ERROR("tcpx_test failed for send request (chunk %d): rc=%d",
                entry.global_idx, rc);
      error_ = true;
      return false;
    }

    if (rc == 2) {
      if (done == 1) {
        LOG_INFO(
            "Connection closed by peer after send chunk %d completed on "
            "channel %d",
            entry.global_idx, channel_id);
      } else {
        LOG_WARN(
            "Connection closed by peer while send chunk %d on channel %d still "
            "in progress (done=0, will retry)",
            entry.global_idx, channel_id);
        if (blocking) {
          std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicros));
          continue;
        } else {
          break;
        }
      }
    } else if (rc > 0) {
      LOG_ERROR(
          "tcpx_test returned unexpected rc=%d for send chunk %d on channel %d",
          rc, entry.global_idx, channel_id);
      error_ = true;
      return false;
    }

    if (!done) {
      if (blocking) {
        std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicros));
        continue;
      }
      break;
    }

    LOG_DEBUG("Send chunk %d completed (channel %d)", entry.global_idx,
              channel_id);

    completed_chunks_++;
    if (channel_id >= 0 &&
        channel_id < static_cast<int>(send_inflight_count_.size()) &&
        send_inflight_count_[channel_id] > 0) {
      send_inflight_count_[channel_id]--;
    }

    queue.pop_front();
  }

  return true;
}

// ============================================================================
// TcpxTransfer - Public API
// ============================================================================

TcpxTransfer::TcpxTransfer(TcpxSession* session, std::string const& remote_name)
    : impl_(new Impl(session, remote_name)) {
  LOG_INFO("TcpxTransfer created for remote: %s", remote_name.c_str());
}

TcpxTransfer::~TcpxTransfer() { LOG_INFO("TcpxTransfer destroyed"); }

int TcpxTransfer::postRecv(uint64_t mem_id, size_t offset, size_t size,
                           int tag) {
  // Based on test_tcpx_perf_multi.cc lines 760-790

  // 1. Get memory handle
  auto* mem = impl_->session_->getMemoryHandle(mem_id);
  if (!mem || !mem->is_recv) {
    LOG_ERROR("Invalid memory handle or not a recv buffer: mem_id=%lu", mem_id);
    return -1;
  }

  // 2. Select channel (round-robin)
  int ch_id = impl_->next_channel_;
  impl_->next_channel_ =
      (impl_->next_channel_ + 1) % impl_->session_->getNumChannels();

  auto* mgr =
      static_cast<ChannelManager*>(impl_->session_->getChannelManager());
  auto& ch = mgr->get_channel(ch_id);
  auto& win = impl_->channel_windows_[ch_id];

  // 3. Wait for capacity (combines Stage 1 + Stage 2)
  if (!impl_->waitForCapacity(ch_id)) {
    return -1;
  }

  // 4. Get mhandle for this channel
  void* mhandle = mgr->get_mhandle(mem_id, true, ch_id);
  if (!mhandle) {
    LOG_ERROR("Failed to get mhandle for mem_id=%lu, channel=%d", mem_id,
              ch_id);
    return -1;
  }

  // 5. Post tcpx_irecv
  void* request = nullptr;
  void* dst_ptr = (char*)mem->buffer + offset;
  void* dst_ptrs[1] = {dst_ptr};
  int sizes[1] = {(int)size};
  int tags[1] = {tag};
  void* mhandles[1] = {mhandle};

  if (tcpx_irecv(ch.recv_comm, 1, dst_ptrs, sizes, tags, mhandles, &request) !=
      0) {
    LOG_ERROR("tcpx_irecv failed");
    return -1;
  }

  // 6. Record request
  int global_idx = impl_->total_chunks_++;

  PostedChunk chunk;
  chunk.request = request;
  chunk.dst_ptr = dst_ptr;
  chunk.bytes = size;
  chunk.offset = offset;
  chunk.tag = tag;
  chunk.global_idx = global_idx;

  win.inflight_recvs.push_back(chunk);

  LOG_DEBUG(
      "Posted recv: mem_id=%lu, offset=%zu, size=%zu, tag=%d, channel=%d, "
      "chunk=%d",
      mem_id, offset, size, tag, ch_id, global_idx);

  // Opportunistically progress current channel (non-blocking)
  if (!impl_->processInflightRecv(ch_id, /*blocking=*/false)) {
    return -1;
  }
  if (!impl_->drainCompletedKernels(ch_id)) {
    return -1;
  }

  // Opportunistically progress other channels to keep metadata queues short
  int num_channels = impl_->session_->getNumChannels();
  for (int other = 0; other < num_channels; ++other) {
    if (other == ch_id) continue;
    if (!impl_->processInflightRecv(other, /*blocking=*/false)) {
      return -1;
    }
    if (!impl_->drainCompletedKernels(other)) {
      return -1;
    }
  }

  return 0;
}

int TcpxTransfer::postSend(uint64_t mem_id, size_t offset, size_t size,
                           int tag) {
  // Based on test_tcpx_perf_multi.cc lines 1180-1210

  // 1. Get memory handle
  auto* mem = impl_->session_->getMemoryHandle(mem_id);
  if (!mem || mem->is_recv) {
    LOG_ERROR("Invalid memory handle or not a send buffer: mem_id=%lu", mem_id);
    return -1;
  }

  // 2. Select channel (round-robin)
  int ch_id = impl_->next_channel_;
  impl_->next_channel_ =
      (impl_->next_channel_ + 1) % impl_->session_->getNumChannels();

  auto* mgr =
      static_cast<ChannelManager*>(impl_->session_->getChannelManager());
  auto& ch = mgr->get_channel(ch_id);

  // 3. Wait for send capacity
  if (!impl_->waitForSendCapacity(ch_id)) {
    return -1;
  }

  // 3. Get mhandle for this channel
  void* mhandle = mgr->get_mhandle(mem_id, false, ch_id);
  if (!mhandle) {
    LOG_ERROR("Failed to get mhandle for mem_id=%lu, channel=%d", mem_id,
              ch_id);
    return -1;
  }

  // 4. Post tcpx_isend
  void* request = nullptr;
  void* src_ptr = (char*)mem->buffer + offset;

  if (tcpx_isend(ch.send_comm, src_ptr, (int)size, tag, mhandle, &request) !=
      0) {
    LOG_ERROR("tcpx_isend failed");
    return -1;
  }

  // 5. Record request for completion tracking
  int global_idx = impl_->total_chunks_++;

  Impl::SendRequest send_req;
  send_req.request = request;
  send_req.global_idx = global_idx;
  impl_->send_queues_[ch_id].push_back(send_req);
  if (ch_id >= 0 &&
      ch_id < static_cast<int>(impl_->send_inflight_count_.size())) {
    impl_->send_inflight_count_[ch_id]++;
  }

  LOG_DEBUG(
      "Posted send: mem_id=%lu, offset=%zu, size=%zu, tag=%d, channel=%d, "
      "chunk=%d",
      mem_id, offset, size, tag, ch_id, global_idx);

  // Opportunistically progress send window
  if (!impl_->progressSendWindow(ch_id)) {
    return -1;
  }

  return 0;
}

bool TcpxTransfer::isComplete() {
  bool ok = true;

  int num_channels = impl_->session_->getNumChannels();
  for (int ch = 0; ch < num_channels; ++ch) {
    if (!impl_->processInflightRecv(ch, /*blocking=*/false)) {
      ok = false;
    }
    if (!impl_->drainCompletedKernels(ch)) {
      ok = false;
    }
  }

  if (!impl_->drainCompletedSends()) {
    ok = false;
  }

  if (impl_->error_) {
    return false;
  }

  impl_->completed_ = (impl_->completed_chunks_ >= impl_->total_chunks_);
  return ok && impl_->completed_;
}

int TcpxTransfer::wait() {
  // Block until complete
  int const kSleepMicros = 100;
  while (!isComplete()) {
    if (impl_->error_) {
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicros));
  }
  return impl_->error_ ? -1 : 0;
}

int TcpxTransfer::getCompletedChunks() const {
  return impl_->completed_chunks_;
}

int TcpxTransfer::getTotalChunks() const { return impl_->total_chunks_; }

int TcpxTransfer::release() {
  // Resources are already released incrementally during:
  // - drainCompletedKernels() calls tcpx_irecv_consumed() for recv
  // - drainCompletedSends() polls tcpx_test() for send
  //
  // This is a no-op for NIXL plugin compatibility.
  // NIXL expects a releaseReqH() method to clean up request handles.

  LOG_DEBUG(
      "TcpxTransfer::release() called (no-op, resources already released)");
  return 0;
}

}  // namespace tcpx

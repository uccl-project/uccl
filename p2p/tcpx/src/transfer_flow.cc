/**
 * @file transfer_flow.cc
 * @brief Helper functions for TCPX plugin API
 *
 * This file contains helper functions extracted from test_tcpx_perf_multi.cc
 * for managing CUDA events and sliding window flow control.
 *
 * IMPORTANT DESIGN NOTE:
 * ======================
 * The original test_tcpx_perf_multi.cc has a two-stage pipeline:
 *
 * Stage 1 (inflight_recvs → pending_reqs):
 *   - tcpx_test() polling to check if recv completed
 *   - GPU unpack kernel launch
 *   - cudaEventRecord() to track kernel completion
 *   - Move from inflight_recvs to pending_reqs
 *
 * Stage 2 (pending_reqs → completed):
 *   - cudaEventQuery() to check if kernel completed
 *   - tcpx_irecv_consumed() to release TCPX slot
 *   - Remove from pending_reqs
 *
 * This file ONLY implements Stage 2 (pending queue management).
 * Stage 1 is application-specific (depends on UnpackLauncher, descriptor
 * building, etc.) and will be implemented in TcpxTransfer class.
 *
 * This separation allows:
 * - Reusable low-level helpers (this file)
 * - Application-specific logic in higher layers (TcpxTransfer)
 */

#include "transfer_flow.h"
#include "tcpx_logging.h"
#include "tcpx_interface.h"  // For tcpx_irecv_consumed
#include <cuda_runtime.h>

namespace tcpx {

// ============================================================================
// CUDA Event Management
// ============================================================================

/**
 * @brief Initialize CUDA events for all channels
 * @param windows Vector of channel windows to initialize
 * @param num_channels Number of channels
 * @return true on success, false on error
 *
 * Creates MAX_INFLIGHT_PER_CHANNEL events per channel for tracking
 * kernel completion in the sliding window flow control.
 *
 * This function is extracted from test_tcpx_perf_multi.cc lines 496-509.
 */
bool initChannelEvents(std::vector<ChannelWindow>& windows, int num_channels) {
  windows.resize(num_channels);
  for (int ch = 0; ch < num_channels; ++ch) {
    windows[ch].events.resize(MAX_INFLIGHT_PER_CHANNEL);
    for (int i = 0; i < MAX_INFLIGHT_PER_CHANNEL; ++i) {
      cudaError_t err = cudaEventCreate(&windows[ch].events[i]);
      if (err != cudaSuccess) {
        LOG_ERROR("Failed to create CUDA event for channel %d, index %d: %s",
                  ch, i, cudaGetErrorString(err));
        // Clean up already created events
        for (int cleanup_ch = 0; cleanup_ch <= ch; ++cleanup_ch) {
          int max_idx = (cleanup_ch == ch) ? i : MAX_INFLIGHT_PER_CHANNEL;
          for (int cleanup_i = 0; cleanup_i < max_idx; ++cleanup_i) {
            cudaEventDestroy(windows[cleanup_ch].events[cleanup_i]);
          }
        }
        return false;
      }
    }
    // Pre-allocate capacity for pending queues (from original implementation)
    windows[ch].pending_reqs.reserve(MAX_INFLIGHT_PER_CHANNEL);
    windows[ch].pending_indices.reserve(MAX_INFLIGHT_PER_CHANNEL);
  }
  return true;
}

/**
 * @brief Destroy CUDA events for all channels
 * @param windows Vector of channel windows to clean up
 *
 * Destroys all CUDA events created by initChannelEvents().
 * Safe to call even if some events were not created.
 */
void destroyChannelEvents(std::vector<ChannelWindow>& windows) {
  for (auto& win : windows) {
    for (auto& evt : win.events) {
      if (evt != nullptr) {
        cudaEventDestroy(evt);
        evt = nullptr;
      }
    }
  }
}

// ============================================================================
// Sliding Window Flow Control
// ============================================================================

/**
 * @brief Drain completed kernels from the pending queue
 * @param win Channel window state
 * @param recv_comm TCPX receive communicator (for tcpx_irecv_consumed)
 * @param completed_chunks Output: number of chunks completed
 * @return true on success, false on error
 *
 * Checks the oldest pending kernel for completion. If complete, calls
 * tcpx_irecv_consumed() to release the TCPX request slot.
 *
 * This function is non-blocking: it only processes kernels that have
 * already completed (cudaEventQuery returns cudaSuccess).
 */
bool drainCompletedKernels(ChannelWindow& win, void* recv_comm, int& completed_chunks) {
  completed_chunks = 0;
  
  while (!win.pending_reqs.empty()) {
    int oldest_idx = win.pending_indices.front();
    void* oldest_req = win.pending_reqs.front();
    cudaEvent_t oldest_event = win.events[oldest_idx % MAX_INFLIGHT_PER_CHANNEL];
    
    // Non-blocking check: has the kernel completed?
    cudaError_t err = cudaEventQuery(oldest_event);
    
    if (err == cudaSuccess) {
      // Kernel completed - consume the TCPX request
      int rc = tcpx_irecv_consumed(recv_comm, 1, oldest_req);
      if (rc != 0) {
        LOG_ERROR("tcpx_irecv_consumed failed: rc=%d", rc);
        return false;
      }
      
      // Remove from pending queue
      win.pending_reqs.erase(win.pending_reqs.begin());
      win.pending_indices.erase(win.pending_indices.begin());
      completed_chunks++;
      
      LOG_DEBUG("Drained completed chunk %d (pending: %zu)",
                oldest_idx, win.pending_reqs.size());
    } else if (err == cudaErrorNotReady) {
      // Kernel still running - stop draining
      break;
    } else {
      // CUDA error
      LOG_ERROR("cudaEventQuery failed: %s", cudaGetErrorString(err));
      return false;
    }
  }
  
  return true;
}

/**
 * @brief Wait for channel capacity by blocking on oldest pending kernel
 * @param win Channel window state
 * @param recv_comm TCPX receive communicator
 * @return true on success, false on error
 *
 * Blocks until the oldest pending kernel completes, then releases its TCPX slot.
 *
 * This implements the "pending queue drain" part of wait_for_channel_capacity
 * from test_tcpx_perf_multi.cc lines 665-689.
 *
 * IMPORTANT: This function assumes pending_reqs is NOT empty.
 * Caller must check this condition before calling.
 *
 * Strategy (from original code):
 * 1. Synchronize on oldest kernel (cudaEventSynchronize)
 * 2. Call tcpx_irecv_consumed() to release the TCPX slot
 * 3. Remove from pending queue
 */
bool waitForPendingKernel(ChannelWindow& win, void* recv_comm) {
  if (win.pending_reqs.empty()) {
    LOG_ERROR("waitForPendingKernel called with empty pending queue");
    return false;
  }

  int oldest_idx = win.pending_indices.front();
  void* oldest_req = win.pending_reqs.front();
  cudaEvent_t oldest_event = win.events[oldest_idx % MAX_INFLIGHT_PER_CHANNEL];

  LOG_DEBUG("Waiting for pending kernel: synchronizing on chunk %d (pending: %zu, inflight: %zu)",
            oldest_idx, win.pending_reqs.size(), win.inflight_recvs.size());

  // Block until kernel completes
  cudaError_t err = cudaEventSynchronize(oldest_event);
  if (err != cudaSuccess) {
    LOG_ERROR("cudaEventSynchronize failed: %s", cudaGetErrorString(err));
    return false;
  }

  // Consume the TCPX request
  int rc = tcpx_irecv_consumed(recv_comm, 1, oldest_req);
  if (rc != 0) {
    LOG_ERROR("tcpx_irecv_consumed failed: rc=%d", rc);
    return false;
  }

  // Remove from pending queue
  win.pending_reqs.erase(win.pending_reqs.begin());
  win.pending_indices.erase(win.pending_indices.begin());

  LOG_DEBUG("Pending kernel consumed: chunk %d (pending: %zu)",
            oldest_idx, win.pending_reqs.size());

  return true;
}

/**
 * @brief Check if channel window has capacity for a new request
 * @param win Channel window state
 * @return true if capacity is available, false if window is full
 */
bool hasCapacity(const ChannelWindow& win) {
  size_t total_inflight = win.pending_reqs.size() + win.inflight_recvs.size();
  return total_inflight < MAX_INFLIGHT_PER_CHANNEL;
}

/**
 * @brief Get the number of available slots in the channel window
 * @param win Channel window state
 * @return Number of available slots
 */
int getAvailableSlots(const ChannelWindow& win) {
  size_t total_inflight = win.pending_reqs.size() + win.inflight_recvs.size();
  return MAX_INFLIGHT_PER_CHANNEL - static_cast<int>(total_inflight);
}

}  // namespace tcpx

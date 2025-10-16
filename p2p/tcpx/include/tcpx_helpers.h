/**
 * @file tcpx_helpers.h
 * @brief Helper functions for TCPX plugin API
 *
 * This header declares helper functions extracted from test_tcpx_perf_multi.cc
 * for managing CUDA events and sliding window flow control.
 */

#pragma once

#include "tcpx_types.h"
#include <vector>

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
 * Extracted from test_tcpx_perf_multi.cc lines 496-509.
 */
bool initChannelEvents(std::vector<ChannelWindow>& windows, int num_channels);

/**
 * @brief Destroy CUDA events for all channels
 * @param windows Vector of channel windows to clean up
 *
 * Destroys all CUDA events created by initChannelEvents().
 * Safe to call even if some events were not created.
 */
void destroyChannelEvents(std::vector<ChannelWindow>& windows);

// ============================================================================
// Sliding Window Flow Control - Low Level
// ============================================================================

/**
 * @brief Drain completed kernels from the pending queue (non-blocking)
 * @param win Channel window state
 * @param recv_comm TCPX receive communicator (for tcpx_irecv_consumed)
 * @param completed_chunks Output: number of chunks completed
 * @return true on success, false on error
 *
 * Checks pending kernels for completion using cudaEventQuery().
 * For each completed kernel, calls tcpx_irecv_consumed() to release
 * the TCPX request slot.
 *
 * This function is non-blocking: it only processes kernels that have
 * already completed.
 *
 * NOTE: This only handles the pending queue (stage 2).
 * To process inflight_recvs (stage 1), use processInflightRecv().
 */
bool drainCompletedKernels(ChannelWindow& win, void* recv_comm, int& completed_chunks);

/**
 * @brief Wait for channel capacity by blocking on oldest pending kernel
 * @param win Channel window state
 * @param recv_comm TCPX receive communicator
 * @return true on success, false on error
 *
 * Blocks until at least one slot is available in the sliding window.
 * This is called when the window is full (pending + inflight >= MAX_INFLIGHT_PER_CHANNEL).
 *
 * Strategy (from wait_for_channel_capacity in test_tcpx_perf_multi.cc):
 * 1. If pending queue is not empty, synchronize on oldest kernel
 * 2. Call tcpx_irecv_consumed() to release the slot
 * 3. Return to caller (who can now post a new request)
 *
 * NOTE: This function does NOT process inflight_recvs.
 * Caller must ensure inflight_recvs are processed separately.
 */
bool waitForPendingKernel(ChannelWindow& win, void* recv_comm);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if channel window has capacity for a new request
 * @param win Channel window state
 * @return true if capacity is available, false if window is full
 */
bool hasCapacity(const ChannelWindow& win);

/**
 * @brief Get the number of available slots in the channel window
 * @param win Channel window state
 * @return Number of available slots
 */
int getAvailableSlots(const ChannelWindow& win);

}  // namespace tcpx


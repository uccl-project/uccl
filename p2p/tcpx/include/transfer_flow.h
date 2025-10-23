/**
 * @file transfer_flow.h
 * @brief Core types and constants for TCPX plugin API
 *
 * This file contains the fundamental data structures and constants used
 * throughout the TCPX plugin implementation. These types were extracted
 * from test_tcpx_perf_multi.cc to enable code reuse across the API layer.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>
#include <cuda_runtime.h>

namespace tcpx {

// ============================================================================
// Constants
// ============================================================================

/**
 * Maximum number of inflight requests per TCPX channel.
 * This is a hard limit imposed by the TCPX plugin (MAX_REQUESTS=16).
 * Server recv uses 16; client send uses 12 for safety headroom.
 */
constexpr int MAX_INFLIGHT_PER_CHANNEL = 16;

/**
 * Default number of channels per GPU.
 * Can be overridden via UCCL_TCPX_NUM_CHANNELS environment variable.
 */
constexpr int DEFAULT_NUM_CHANNELS = 2;

// ============================================================================
// Core Data Structures
// ============================================================================

/**
 * @struct PostedChunk
 * @brief Represents a single posted TCPX transfer request
 *
 * Tracks metadata for an inflight send or receive operation.
 * Used in sliding window flow control to manage request lifecycle.
 */
struct PostedChunk {
  void* request = nullptr;  ///< TCPX request handle
  void* dst_ptr = nullptr;  ///< Destination pointer (for recv)
  size_t bytes = 0;         ///< Transfer size in bytes
  size_t offset = 0;        ///< Offset within the buffer
  int tag = 0;              ///< Transfer tag (must be unique)
  int global_idx = 0;       ///< Global chunk index (for debugging)
};

/**
 * @struct ChannelWindow
 * @brief Per-channel sliding window state for flow control
 *
 * Each TCPX channel maintains independent sliding window state to prevent
 * exhausting the request pool (MAX_REQUESTS=16 per comm).
 *
 * Lifecycle:
 * 1. Request posted → added to inflight_recvs
 * 2. tcpx_test(done=1) → kernel launched → moved to pending_reqs
 * 3. cudaEventSynchronize → tcpx_irecv_consumed → removed from pending_reqs
 *
 * Invariant: pending_reqs.size() + inflight_recvs.size() <=
 * MAX_INFLIGHT_PER_CHANNEL
 */
struct ChannelWindow {
  std::vector<cudaEvent_t> events;   ///< CUDA events (track kernel completion)
  std::vector<void*> pending_reqs;   ///< Kernel submitted but not yet consumed
  std::vector<int> pending_indices;  ///< Pending chunk indices (for debugging)
  int chunk_counter = 0;             ///< Chunks handled by this channel
  std::deque<PostedChunk> inflight_recvs;  ///< Posted but not yet unpacked
  std::vector<uint64_t> pending_desc_id;   /// persistent_kernel to deal
};

// ============================================================================
// Helper Function Declarations (Stage 2 pipeline utilities)
// ============================================================================

bool initChannelEvents(std::vector<ChannelWindow>& windows,
                       int num_channels);                        // TODO delete?
void destroyChannelEvents(std::vector<ChannelWindow>& windows);  // TODO delete?
bool drainCompletedKernels(ChannelWindow& win, void* recv_comm,
                           int& completed_chunks);
bool waitForPendingKernel(ChannelWindow& win, void* recv_comm);  // TODO delete?
bool hasCapacity(ChannelWindow const& win);                      // TODO delete?
int getAvailableSlots(ChannelWindow const& win);                 // TODO delete?

}  // namespace tcpx

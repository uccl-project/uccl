/**
 * @file tcpx_transfer.h
 * @brief TCPX Transfer Management API
 *
 * This file defines the TcpxTransfer class, which manages data transfers
 * (send/recv operations) over established TCPX connections.
 *
 * Design based on test_tcpx_perf_multi.cc transfer logic (lines 523-705).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tcpx {

// Forward declaration
class TcpxSession;

/**
 * @brief TCPX Transfer - manages send/recv operations over a connection
 *
 * This class encapsulates the complete TCPX transfer flow:
 * - postRecv/postSend: Initiate non-blocking transfers
 * - isComplete: Poll for completion (non-blocking)
 * - wait: Block until all transfers complete
 *
 * Lifecycle:
 * 1. Creation: TcpxSession::createTransfer()
 * 2. Post operations: postRecv() / postSend()
 * 3. Completion: isComplete() polling or wait() blocking
 * 4. Destruction: RAII cleanup (destroy CUDA events)
 *
 * Thread-safety: NOT thread-safe. Caller must synchronize access.
 *
 * Implementation details:
 * - Uses sliding window flow control per channel
 * - Two-stage pipeline:
 *   * Stage 1: inflight_recvs → pending_reqs (tcpx_test + kernel launch)
 *   * Stage 2: pending_reqs → completed (cudaEventQuery + tcpx_irecv_consumed)
 * - Round-robin channel selection for load balancing
 */
class TcpxTransfer {
 public:
  /**
   * @brief Construct a new transfer object
   * @param session Parent session (must outlive this transfer)
   * @param remote_name Identifier for the remote peer
   *
   * This constructor:
   * - Initializes per-channel sliding windows
   * - Creates CUDA events for kernel completion tracking
   * - Sets up round-robin channel selection
   *
   * Based on test_tcpx_perf_multi.cc lines 496-509.
   */
  TcpxTransfer(TcpxSession* session, const std::string& remote_name);

  /**
   * @brief Destroy the transfer object
   *
   * RAII cleanup:
   * - Destroy all CUDA events
   * - Clear all pending requests
   */
  ~TcpxTransfer();

  // Disable copy and move
  TcpxTransfer(const TcpxTransfer&) = delete;
  TcpxTransfer& operator=(const TcpxTransfer&) = delete;
  TcpxTransfer(TcpxTransfer&&) = delete;
  TcpxTransfer& operator=(TcpxTransfer&&) = delete;

  // ============================================================================
  // Transfer Operations
  // ============================================================================

  /**
   * @brief Post a non-blocking receive operation
   * @param mem_id Memory registration ID (from TcpxSession::registerMemory)
   * @param offset Offset within the registered buffer
   * @param size Number of bytes to receive
   * @param tag Transfer tag (must match sender's tag)
   * @return 0 on success, non-zero on error
   *
   * This function:
   * 1. Validates memory handle (must be recv buffer)
   * 2. Selects a channel (round-robin)
   * 3. Waits for capacity (drains completed kernels if needed)
   * 4. Posts tcpx_irecv
   * 5. Adds to inflight_recvs queue
   *
   * Based on test_tcpx_perf_multi.cc lines 760-790.
   */
  int postRecv(uint64_t mem_id, size_t offset, size_t size, int tag);

  /**
   * @brief Post a non-blocking send operation
   * @param mem_id Memory registration ID (from TcpxSession::registerMemory)
   * @param offset Offset within the registered buffer
   * @param size Number of bytes to send
   * @param tag Transfer tag (must match receiver's tag)
   * @return 0 on success, non-zero on error
   *
   * This function:
   * 1. Validates memory handle (must be send buffer)
   * 2. Selects a channel (round-robin)
   * 3. Posts tcpx_isend
   * 4. Tracks request for completion polling
   *
   * Based on test_tcpx_perf_multi.cc lines 1180-1210.
   */
  int postSend(uint64_t mem_id, size_t offset, size_t size, int tag);

  /**
   * @brief Check if all posted transfers are complete (non-blocking)
   * @return true if all transfers complete, false otherwise
   *
   * This function:
   * 1. Polls all channels (drainCompletedKernels)
   * 2. Checks if completed_chunks >= total_chunks
   *
   * Based on test_tcpx_perf_multi.cc completion check logic.
   */
  bool isComplete();

  /**
   * @brief Block until all posted transfers complete
   * @return 0 on success, non-zero on error
   *
   * This function:
   * 1. Repeatedly calls isComplete() until true
   * 2. Sleeps briefly between polls to avoid busy-waiting
   *
   * Based on test_tcpx_perf_multi.cc wait logic.
   */
  int wait();

  /**
   * @brief Get the number of completed chunks
   * @return Number of chunks that have completed
   */
  int getCompletedChunks() const;

  /**
   * @brief Get the total number of posted chunks
   * @return Total number of chunks posted
   */
  int getTotalChunks() const;

  /**
   * @brief Release resources associated with this transfer
   * @return 0 on success, non-zero on error
   *
   * This function should be called after wait() completes to release
   * any remaining TCPX resources. For NIXL plugin compatibility.
   *
   * Note: Currently a no-op since resources are released incrementally
   * during drainCompletedKernels() and drainCompletedSends().
   */
  int release();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // Allow TcpxSession to access private constructor
  friend class TcpxSession;
};

}  // namespace tcpx


/**
 * @file sliding_window.h
 * @brief Per-channel sliding window manager for TCPX requests
 *
 * TCPX plugin has a fixed limit of MAX_REQUESTS=16 per comm.
 * This class manages a sliding window to ensure we never exceed this limit.
 *
 * Design mirrors NCCL's inflight tracking (net.cc:688+).
 */

#pragma once

#include <vector>
#include <cuda_runtime.h>

class SlidingWindow {
 public:
  /**
   * @brief Constructor
   * @param max_inflight Maximum number of in-flight requests (typically 16)
   */
  explicit SlidingWindow(int max_inflight);

  /**
   * @brief Destructor - cleans up CUDA events
   */
  ~SlidingWindow();

  /**
   * @brief Check if window is full
   * @return true if window has reached max_inflight capacity
   */
  bool is_full() const;

  /**
   * @brief Get current window size
   * @return Number of pending requests
   */
  int size() const;

  /**
   * @brief Add a new request to the window
   * @param request TCPX request handle
   * @param chunk_idx Chunk index for tracking/debugging
   * @param event CUDA event (optional, for server recv to track kernel
   * completion)
   */
  void add_request(void* request, int chunk_idx, cudaEvent_t event = nullptr);

  /**
   * @brief Try to release oldest request if it's ready
   *
   * CRITICAL: This function respects TCPX's internal queue order.
   * TCPX requires tcpx_test() to be called on rq.next_transmitting() only.
   * If the request isn't at the front of TCPX's internal queue yet,
   * tcpx_test() will return tcpxInternalError - this is NOT a real error,
   * just means "not your turn yet".
   *
   * This function will NOT force-wait if the request isn't ready.
   * Caller should retry later (with a small sleep) if return value is 1.
   *
   * For server (is_recv=true):
   *   1. Call tcpx_test() to check if request is done
   *   2. If tcpx_test() returns error (not at front of queue), return 1
   *   3. If tcpx_test() succeeds but done==0, return 1
   *   4. If done==1, wait for CUDA event (if any) and call
   * tcpx_irecv_consumed()
   *
   * For client (is_recv=false):
   *   1. Call tcpx_test() to check if request is done
   *   2. If not ready (error or done==0), return 1
   *   3. If done==1, remove from window (TCPX auto-releases send requests)
   *
   * Return values (THREE states):
   *   0  = Success: request released, window has space now
   *   1  = Not ready: request not at front of TCPX queue or not done yet
   *        (NOT an error - caller should sleep and retry)
   *   -1 = Real error: cudaEventSynchronize failed, tcpx_irecv_consumed failed,
   * etc.
   *
   * Example usage:
   *   while (win->is_full()) {
   *     int rc = win->try_release_oldest(comm, is_recv);
   *     if (rc == 0) break;              // Success, window has space
   *     if (rc == 1) sleep(10us);        // Not ready, retry later
   *     if (rc == -1) handle_error();    // Real error
   *   }
   *
   * @param comm TCPX comm handle
   * @param is_recv true for recv (server), false for send (client)
   * @return 0=released, 1=not ready (retry later), -1=real error
   */
  int try_release_oldest(void* comm, bool is_recv);

  /**
   * @brief Drain all pending requests (blocking)
   *
   * Called at end of iteration to ensure all requests complete.
   * Internally calls try_release_oldest() in a loop:
   *   - If rc==0 (released), continue to next request
   *   - If rc==1 (not ready), sleep 10us and retry
   *   - If rc==-1 (error), return -1 immediately
   *
   * This function will block until all requests are released or an error
   * occurs.
   *
   * @param comm TCPX comm handle
   * @param is_recv true for recv (server), false for send (client)
   * @return 0 on success (all drained), -1 on real error
   */
  int drain_all(void* comm, bool is_recv);

  /**
   * @brief Clear all requests without waiting (for cleanup)
   */
  void clear();

 private:
  int max_inflight_;
  std::vector<void*> pending_reqs_;
  std::vector<int> pending_indices_;
  std::vector<cudaEvent_t> events_;

  // Disable copy
  SlidingWindow(SlidingWindow const&) = delete;
  SlidingWindow& operator=(SlidingWindow const&) = delete;
};

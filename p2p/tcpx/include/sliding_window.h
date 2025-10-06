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
   * @brief Wait for oldest request and remove it from window
   *
   * For server (is_recv=true):
   *   - Wait for CUDA event (kernel completion)
   *   - Call tcpx_irecv_consumed to release TCPX slot
   *
   * For client (is_recv=false):
   *   - Poll tcpx_test until done
   *   - Request is automatically released by TCPX
   *
   * @param comm TCPX comm handle
   * @param is_recv true for recv (server), false for send (client)
   * @return 0 on success, -1 on error
   */
  int wait_and_release_oldest(void* comm, bool is_recv);

  /**
   * @brief Drain all pending requests
   *
   * Called at end of iteration to ensure all requests complete.
   *
   * @param comm TCPX comm handle
   * @param is_recv true for recv (server), false for send (client)
   * @return 0 on success, -1 on error
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

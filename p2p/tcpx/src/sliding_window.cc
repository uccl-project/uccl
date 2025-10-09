/**
 * @file sliding_window.cc
 * @brief Implementation of per-channel sliding window manager
 */

#include "sliding_window.h"
#include "tcpx_interface.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <cuda_runtime.h>

SlidingWindow::SlidingWindow(int max_inflight) : max_inflight_(max_inflight) {
  pending_reqs_.reserve(max_inflight);
  pending_indices_.reserve(max_inflight);
  events_.reserve(max_inflight);
}

SlidingWindow::~SlidingWindow() {
  // Clean up via clear() to avoid double-destroy
  clear();
}

bool SlidingWindow::is_full() const {
  return static_cast<int>(pending_reqs_.size()) >= max_inflight_;
}

int SlidingWindow::size() const {
  return static_cast<int>(pending_reqs_.size());
}

void SlidingWindow::add_request(void* request, int chunk_idx,
                                cudaEvent_t event) {
  pending_reqs_.push_back(request);
  pending_indices_.push_back(chunk_idx);
  events_.push_back(event);
}

int SlidingWindow::try_release_oldest(void* comm, bool is_recv) {
  if (pending_reqs_.empty()) {
    return 0;  // Nothing to release
  }

  void* oldest_req = pending_reqs_.front();
  int oldest_idx = pending_indices_.front();
  cudaEvent_t oldest_event = events_.front();

  // Step 1: Check if request is ready via tcpx_test()
  // CRITICAL: tcpx_test() requires the request to be rq.next_transmitting()
  // If it's not ready yet, tcpx_test() will return tcpxInternalError
  // This is NOT a real error - just means "not your turn yet"
  int done = 0;
  int size = 0;
  int rc = tcpx_test(oldest_req, &done, &size);

  static int debug_count = 0;
  if (debug_count < 50) {
    std::cout << "[DEBUG][SlidingWindow] tcpx_test: chunk_idx=" << oldest_idx
              << " rc=" << rc << " done=" << done << " size=" << size
              << " is_recv=" << is_recv << std::endl;
    debug_count++;
  }

  if (rc != 0) {
    // Request not ready yet (not at front of TCPX's internal queue)
    // This is expected behavior - just return "not ready"
    return 1;  // Not ready, try again later
  }

  if (!done) {
    // tcpx_test() succeeded but request not complete yet
    // This is also expected - need to wait more
    return 1;  // Not ready, try again later
  }

  // Step 2: Request is done! Now handle recv-specific cleanup
  if (is_recv) {
    // Wait for GPU kernel (if applicable)
    if (oldest_event) {
      cudaError_t err = cudaEventSynchronize(oldest_event);
      if (err != cudaSuccess) {
        std::cerr << "[SlidingWindow] cudaEventSynchronize failed for chunk "
                  << oldest_idx << ": " << cudaGetErrorString(err) << std::endl;
        return -1;  // Real error
      }

      cudaError_t destroy_err = cudaEventDestroy(oldest_event);
      if (destroy_err != cudaSuccess) {
        std::cerr << "[SlidingWindow] cudaEventDestroy failed: "
                  << cudaGetErrorString(destroy_err) << std::endl;
      }
    }

    // Release TCPX request slot
    if (tcpx_irecv_consumed(comm, 1, oldest_req) != 0) {
      std::cerr << "[SlidingWindow] tcpx_irecv_consumed failed for chunk "
                << oldest_idx << std::endl;
      return -1;  // Real error
    }
  }
  // For send: TCPX automatically releases when done=1

  // Step 3: Remove from window
  pending_reqs_.erase(pending_reqs_.begin());
  pending_indices_.erase(pending_indices_.begin());
  events_.erase(events_.begin());

  return 0;  // Successfully released
}

int SlidingWindow::drain_all(void* comm, bool is_recv) {
  while (!pending_reqs_.empty()) {
    int rc = try_release_oldest(comm, is_recv);

    if (rc == 0) {
      // Successfully released, continue
      continue;
    } else if (rc == 1) {
      // Not ready yet, sleep and retry
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      continue;
    } else {
      // Real error (rc == -1)
      std::cerr << "[SlidingWindow] Failed to drain request" << std::endl;
      return -1;
    }
  }
  return 0;
}

void SlidingWindow::clear() {
  pending_reqs_.clear();
  pending_indices_.clear();

  // Clean up events without waiting
  for (auto event : events_) {
    if (event) {
      cudaError_t err = cudaEventDestroy(event);
      if (err != cudaSuccess) {
        std::cerr << "[SlidingWindow] cudaEventDestroy failed in clear(): "
                  << cudaGetErrorString(err) << std::endl;
      }
    }
  }
  events_.clear();
}

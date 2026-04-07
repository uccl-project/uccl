#include "request.h"

namespace UKernel {
namespace Transport {

void Request::mark_queued(uint32_t completion_count) {
  remaining_completions.store(completion_count, std::memory_order_release);
  state.store(RequestState::Queued, std::memory_order_release);
}

void Request::mark_running() {
  state.store(RequestState::Running, std::memory_order_release);
}

void Request::mark_failed() {
  if (load_state(std::memory_order_acquire) == RequestState::Completed) return;
  remaining_completions.store(0, std::memory_order_release);
  state.store(RequestState::Failed, std::memory_order_release);
}

void Request::complete_one() {
  uint32_t prev = remaining_completions.fetch_sub(1, std::memory_order_acq_rel);
  if (prev == 0) {
    // Underflow means completion signaling protocol is broken.
    remaining_completions.store(0, std::memory_order_release);
    state.store(RequestState::Failed, std::memory_order_release);
    return;
  }
  if (prev == 1) {
    state.store(RequestState::Completed, std::memory_order_release);
  }
}

}  // namespace Transport
}  // namespace UKernel

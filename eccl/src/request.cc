#include "request.h"

void Request::on_comm_done(bool done) {
  if (!done) {
    failed.store(true, std::memory_order_release);
    return;
  }
  int prev = pending_signaled.fetch_sub(1, std::memory_order_acq_rel);
  if (do_reduction) {
    // TODO: post compute req to gpu kernel
    return;
  } else {
    if (prev == 1) {
      finished.store(true, std::memory_order_release);
    }
  }
}

void Request::on_compute_done() {
  { finished.store(true, std::memory_order_release); }
}
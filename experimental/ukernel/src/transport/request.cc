#include "request.h"

namespace UKernel {
namespace Transport {

void Request::on_comm_done() {
  int prev = pending_signaled.fetch_sub(1, std::memory_order_acq_rel);
  if (prev == 1) {
    finished.store(true, std::memory_order_release);
  }
}

}  // namespace Transport
}  // namespace UKernel
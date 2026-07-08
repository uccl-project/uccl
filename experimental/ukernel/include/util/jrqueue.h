#pragma once

extern "C" {
#include "jring.h"
}
#include <thread>

namespace UKernel {

// Blocking: spin until element is enqueued.
template <typename T>
static inline void jrpush(jring_t* ring, T const& elem) {
  while (jring_mp_enqueue_bulk(ring, &elem, 1, nullptr) != 1)
    std::this_thread::yield();
}

// Non-blocking: enqueue up to n elements, returns count pushed.
// Unpushed elements (n - returned) remain in caller's array for retry.
template <typename T>
static inline size_t jrtrypush(jring_t* ring, T const* elems, size_t n) {
  size_t done = 0;
  while (done < n) {
    unsigned int pushed =
        jring_mp_enqueue_bulk(ring, elems + done, n - done, nullptr);
    if (pushed == 0) break;
    done += pushed;
  }
  return done;
}

}  // namespace UKernel

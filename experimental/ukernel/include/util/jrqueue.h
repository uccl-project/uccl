#pragma once

extern "C" {
#include "jring.h"
}
#include <thread>

namespace UKernel {

// Blocking push — uses jring_enqueue_bulk which auto-selects SP/MP
// based on the ring's nprod setting.
template <typename T>
static inline void jrpush(jring_t* ring, T const& elem) {
  while (jring_enqueue_bulk(ring, &elem, 1, nullptr) != 1)
    std::this_thread::yield();
}

// Non-blocking push: enqueue up to n elements, returns count pushed.
template <typename T>
static inline size_t jrtrypush(jring_t* ring, T const* elems, size_t n) {
  size_t done = 0;
  while (done < n) {
    unsigned int pushed =
        jring_enqueue_bulk(ring, elems + done, n - done, nullptr);
    if (pushed == 0) break;
    done += pushed;
  }
  return done;
}

// Blocking pop — uses jring_dequeue_bulk which auto-selects SC/MC
// based on the ring's ncons setting.
template <typename T>
static inline T jrpop(jring_t* ring) {
  T elem;
  while (jring_dequeue_bulk(ring, &elem, 1, nullptr) != 1)
    std::this_thread::yield();
  return elem;
}

}  // namespace UKernel

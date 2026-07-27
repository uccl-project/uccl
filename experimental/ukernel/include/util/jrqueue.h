#pragma once
// Thin inline helper: blocking push onto a jring.
// All other ring operations should use raw jring_* functions directly
// to keep the producer/consumer configuration explicit at each call site.

extern "C" {
#include "jring.h"
}
#include <thread>

namespace UKernel {

template <typename T>
static inline void jrpush(jring_t* ring, T const& elem) {
  while (jring_mp_enqueue_bulk(ring, &elem, 1, nullptr) != 1)
    std::this_thread::yield();
}

}  // namespace UKernel

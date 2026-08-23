// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

namespace mscclpp::lite {

class SyncPrimitives {
 public:
  template <typename Atomic>
  static void waitEpoch(Atomic const& value, uint64_t epoch) {
    int spins = 0;
    while (value.load(std::memory_order_acquire) < epoch) {
      if (spins++ < 65536) {
#if defined(__x86_64__) || defined(__i386__)
        asm volatile("pause" ::: "memory");
#endif
      } else {
        std::this_thread::yield();
      }
    }
  }

  template <typename Atomic>
  static void publishEpoch(Atomic& value, uint64_t epoch) {
    value.store(epoch, std::memory_order_release);
  }
};

}  // namespace mscclpp::lite

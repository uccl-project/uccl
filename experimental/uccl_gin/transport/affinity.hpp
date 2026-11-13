// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <pthread.h>
#include <sched.h>

// Distribute proxy threads across the CPU budget inherited from their launcher.
inline bool pin_thread_to_allowed_cpu(bool reserve_submit_cpu = false) {
  cpu_set_t allowed;
  if (pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed))
    return false;
  int count = CPU_COUNT(&allowed) - int(reserve_submit_cpu);
  if (count <= 0) return false;
  static std::atomic<unsigned> next{0};
  unsigned ordinal =
      next.fetch_add(1, std::memory_order_relaxed) % unsigned(count);
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) continue;
    if (ordinal) {
      --ordinal;
      continue;
    }
    cpu_set_t selected;
    CPU_ZERO(&selected);
    CPU_SET(cpu, &selected);
    return !pthread_setaffinity_np(pthread_self(), sizeof(selected), &selected);
  }
  return false;
}

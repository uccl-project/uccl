// SPDX-License-Identifier: Apache-2.0
#include "experimental/uccl_gin/transport/affinity.hpp"
#include <cassert>
#include <cstdio>
#include <map>
#include <thread>
#include <vector>

int main() {
  cpu_set_t original, budget;
  assert(!pthread_getaffinity_np(pthread_self(), sizeof(original), &original));
  CPU_ZERO(&budget);
  int count = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE && count < 4; ++cpu) {
    if (CPU_ISSET(cpu, &original)) { CPU_SET(cpu, &budget); ++count; }
  }
  assert(count > 0);
  assert(!pthread_setaffinity_np(pthread_self(), sizeof(budget), &budget));
  int reserved_cpu = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &budget)) reserved_cpu = cpu;
  for (bool reserve : {false, true}) {
    int proxy_count = count - int(reserve);
    if (!proxy_count) {
      assert(!pin_thread_to_allowed_cpu(reserve));
      continue;
    }
    std::vector<int> observed(4 * proxy_count);
    std::vector<std::thread> workers;
    for (size_t i = 0; i < observed.size(); ++i) workers.emplace_back([&, i] {
      assert(pin_thread_to_allowed_cpu(reserve));
      cpu_set_t selected;
      assert(!pthread_getaffinity_np(pthread_self(), sizeof(selected), &selected));
      assert(CPU_COUNT(&selected) == 1);
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &selected)) {
        assert(CPU_ISSET(cpu, &budget));
        assert(!reserve || cpu != reserved_cpu);
        observed[i] = cpu;
      }
    });
    for (auto& worker : workers) worker.join();
    std::map<int, int> counts;
    for (int cpu : observed) ++counts[cpu];
    assert(int(counts.size()) == proxy_count);
    for (auto const& entry : counts) assert(entry.second == 4);
  }
  assert(!pthread_setaffinity_np(pthread_self(), sizeof(original), &original));
  std::puts("Inherited CPU budget, distribution and submit-CPU exclusion PASS");
}

// SPDX-License-Identifier: Apache-2.0
#include "device/ht_scheduler.hpp"
#include <cassert>
#include <cstdio>

int main() {
  using namespace nccl_ep::hybridep;
  KernelResources kernel{0, 132, 64, 192, 16, 4, 4096, 61440, 233472};
  ScheduleResources resources{kernel, kernel};
  // SMEM can cap an optimistic occupancy estimate; a lower API result can cap SMEM.
  assert(group_capacity(kernel) == 24);
  resources.combine.resident_blocks = 1;
  assert(admission_limit(resources, 64, 64) == 8);
  assert(admission_limit(resources, 2, 64) == 2);
  assert(admission_limit(resources, 64, 4) == 4);
  resources.combine.dynamic_smem = 240000;
  assert(admission_limit(resources, 64, 64) == 0);
  resources.combine = kernel;
  resources.combine.smem_per_block = 65535;
  assert(admission_limit(resources, 64, 64) == 0);
  resources.combine.smem_per_block = 65536;
  assert(admission_limit(resources, 64, 64) == 24);
  for (int smem = 1024; smem <= 240000; smem += 1024) {
    resources.dispatch.dynamic_smem = smem;
    int capacity = admission_limit(resources, 64, 64);
    assert(capacity >= 0 && capacity <= 64);
    assert(capacity * kernel.blocks <= kernel.sms * kernel.resident_blocks);
    assert(size_t(capacity * kernel.blocks) * (smem + kernel.static_smem) <=
           size_t(kernel.sms) * kernel.smem_per_sm);
  }
  assert(memory_capacity(2, 32, 72, 3) == 26);
  assert(memory_capacity(2, 16, 72, 3) == 16);
  assert(memory_capacity(16, 32, 2, 3) == 16);
  assert(memory_capacity(16, 32, 0, 3) == 16);
  assert(memory_capacity(16, 32, 3, 3) == 17);
  assert(memory_capacity(32, 32, 1024, 3) == 32);
  assert(memory_capacity(1, 32, size_t(-1), 1) == 32);
  KernelResources expert{0, 148, 32, 128, 74, 1, 2048, 0, 233472, 232448};
  resources = {kernel, kernel};
  assert(admission_limit(resources, expert, 64, 32) == 2);
  expert.blocks = 37;
  assert(admission_limit(resources, expert, 64, 32) == 4);
  expert.blocks = 18;
  assert(admission_limit(resources, expert, 64, 32) == 8);
  expert.blocks = 16;
  assert(admission_limit(resources, expert, 64, 32) == 9);
  assert(admission_limit(resources, expert, 1, 32) == 1);
  expert.static_smem = 232449;
  assert(admission_limit(resources, expert, 64, 32) == 0);
  std::puts("SMEM, occupancy and GPU memory admission boundaries PASS");
}

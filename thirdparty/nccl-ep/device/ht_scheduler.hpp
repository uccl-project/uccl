// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>

namespace nccl_ep::hybridep {

struct LaunchPolicy {
  int stages = 0;  // 0 selects the original topology-specific stage counts.
  int blocks = 16;
};

struct KernelResources {
  int device = 0;
  int sms = 0;
  int registers = 0;
  int threads = 0;
  int blocks = 0;
  int resident_blocks = 0;
  std::size_t static_smem = 0;
  std::size_t dynamic_smem = 0;
  std::size_t smem_per_sm = 0;
  std::size_t smem_per_block = 0;
};

struct ScheduleResources {
  KernelResources dispatch, combine;
};

// Queries and policy changes run before submission, with this thread's work drained.
// The default policy retains the original dispatch and topology-specific combine.
__attribute__((visibility("default"))) bool set_launch_policy(LaunchPolicy policy);
__attribute__((visibility("default"))) void query_schedule_resources(
    int hidden, int experts_per_rank, int local_ranks, int nodes,
    ScheduleResources& resources, bool backward_combine = false);

inline int group_capacity(KernelResources const& kernel) {
  auto smem = kernel.static_smem + kernel.dynamic_smem;
  if (kernel.smem_per_block && smem > kernel.smem_per_block) return 0;
  int smem_blocks = smem ? int(kernel.smem_per_sm / smem) : kernel.resident_blocks;
  int resident = std::min(kernel.resident_blocks, smem_blocks);
  return kernel.blocks > 0 ? kernel.sms * resident / kernel.blocks : 0;
}

// A logical queue can exceed this pool. Each group owns one in-flight handle.
// Both ranks agree on the minimum capacity before using deterministic windows.
inline int admission_limit(ScheduleResources const& resources, int queued, int requested) {
  return std::min({queued, requested, group_capacity(resources.dispatch),
                   group_capacity(resources.combine)});
}

// A staged expert operation uses the same resource budget as communication.
// This bounds admission; it neither reserves SMs nor promises co-residency.
inline int admission_limit(ScheduleResources const& resources, KernelResources const& expert,
                           int queued, int requested) {
  return std::min(admission_limit(resources, queued, requested), group_capacity(expert));
}

// Use the observed CUDA footprint of a complete group, including opaque buffers.
// Existing groups are resident; only additional groups consume the free budget.
// The caller supplies a positive measured cost and resident <= limit.
inline int memory_capacity(int resident, int limit, std::size_t free_bytes,
                           std::size_t group_bytes) {
  return resident + int(std::min(std::size_t(limit - resident), free_bytes / group_bytes));
}

}  // namespace nccl_ep::hybridep

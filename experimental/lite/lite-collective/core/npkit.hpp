// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NPKIT_H_
#define NPKIT_H_

#include "device.hpp"
#include "gpu_utils.hpp"
#include "npkit_event.hpp"
#include "npkit_struct.hpp"
#include <string>
#include <thread>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
#define NPKIT_GET_GPU_TIMESTAMP wall_clock64
#define NPKIT_MAX_NUM_GPU_THREADBLOCKS 64
#define NPKIT_CPU_TIMESTAMP_SLOT_SIZE 128
#define NPKIT_LOAD_CPU_TIMESTAMP_PER_BLOCK(buf, blk) \
  *(buf + NPKIT_CPU_TIMESTAMP_SLOT_SIZE * blk / sizeof(uint64_t))
#define NPKIT_STORE_CPU_TIMESTAMP_PER_BLOCK(buf, val, blk) \
  *reinterpret_cast<uint64_t volatile*>(                   \
      buf + NPKIT_CPU_TIMESTAMP_SLOT_SIZE * blk / sizeof(uint64_t)) = val

#else
#define NPKIT_GET_GPU_TIMESTAMP clock64
#endif

#define NPKIT_SHM_NUM_EVENTS 64

class NpKit {
 public:
  static const uint64_t kNumGpuEventBuffers = 1024;

  static const uint64_t kNumCpuEventBuffers = 64;

  static void Init(int rank);

  static void Dump(std::string const& dump_dir);

  static void Shutdown();

  static NpKitEventCollectContext* GetGpuEventCollectContexts();

#if defined(MSCCLPP_DEVICE_COMPILE)
  static MSCCLPP_DEVICE_INLINE void CollectGpuEventShm(
      uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
      NpKitEvent* event_buffer, uint64_t* event_buffer_head) {
    if (*event_buffer_head < NPKIT_SHM_NUM_EVENTS) {
      if (threadIdx.x == 0) {
        NpKitEvent& event = event_buffer[*event_buffer_head];
        event.fields.type = type;
        event.fields.size = size;
        event.fields.rsvd = rsvd;
        event.fields.timestamp = timestamp;
      }
      (*event_buffer_head)++;
    }
  }

  static MSCCLPP_DEVICE_INLINE void StoreGpuEventShm(
      NpKitEventCollectContext* npKitEventCollectContexts,
      NpKitEvent* event_buffer, uint64_t event_buffer_head) {
    __syncshm();
    NpKitEventCollectContext* npKitCtx = npKitEventCollectContexts + blockIdx.x;
    NpKitEvent* global_event_buffer = npKitCtx->event_buffer;
    uint64_t global_event_buffer_head = npKitCtx->event_buffer_head;
    for (size_t i = threadIdx.x;
         i < event_buffer_head * sizeof(NpKitEvent) / sizeof(int4);
         i += blockDim.x) {
      ((int4*)(global_event_buffer + global_event_buffer_head))[i] =
          ((int4*)event_buffer)[i];
    }
    if (threadIdx.x == 0) {
      npKitCtx->event_buffer_head += event_buffer_head;
    }
  }
#endif

  static void CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd,
                              uint64_t timestamp, int channel_id);

  static uint64_t* GetCpuTimestamp();

 private:
  static void CpuTimestampUpdateThread();

  // 64K * 1024 * 16B = 1GB per GPU
  static const uint64_t kMaxNumGpuEventsPerBuffer = 1ULL << 16;

  // 64K * 2 (send/recv) * (1024/64) = 2M, 2M * 64 * 16B = 2GB per CPU
  static const uint64_t kMaxNumCpuEventsPerBuffer = 1ULL << 21;

  static std::vector<mscclpp::detail::UniqueGpuPtr<NpKitEvent>>
      gpu_event_buffers_;
  static std::vector<std::unique_ptr<NpKitEvent[]>> cpu_event_buffers_;

  static mscclpp::detail::UniqueGpuPtr<NpKitEventCollectContext>
      gpu_collect_contexts_;
  static std::unique_ptr<NpKitEventCollectContext[]> cpu_collect_contexts_;

  static uint64_t rank_;

#if defined(__HIP_PLATFORM_AMD__)
  static mscclpp::detail::UniqueGpuHostPtr<uint64_t[]> cpu_timestamp_;
#else
  static mscclpp::detail::UniqueGpuHostPtr<uint64_t> cpu_timestamp_;
#endif
  static std::unique_ptr<std::thread> cpu_timestamp_update_thread_;
  static bool volatile cpu_timestamp_update_thread_should_stop_;
};

#endif

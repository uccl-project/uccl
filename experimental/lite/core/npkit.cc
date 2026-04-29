// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "npkit.hpp"
#include "debug.h"
#include "gpu.hpp"
#include <chrono>
#include <fstream>
#include <unistd.h>

uint64_t NpKit::rank_ = 0;

std::vector<mscclpp::detail::UniqueGpuPtr<NpKitEvent>>
    NpKit::gpu_event_buffers_;
std::vector<std::unique_ptr<NpKitEvent[]>> NpKit::cpu_event_buffers_;

mscclpp::detail::UniqueGpuPtr<NpKitEventCollectContext>
    NpKit::gpu_collect_contexts_;
std::unique_ptr<NpKitEventCollectContext[]> NpKit::cpu_collect_contexts_;

#if defined(__HIP_PLATFORM_AMD__)
mscclpp::detail::UniqueGpuHostPtr<uint64_t[]> NpKit::cpu_timestamp_;
#else
mscclpp::detail::UniqueGpuHostPtr<uint64_t> NpKit::cpu_timestamp_;
#endif
std::unique_ptr<std::thread> NpKit::cpu_timestamp_update_thread_;
bool volatile NpKit::cpu_timestamp_update_thread_should_stop_ = false;

void NpKit::CpuTimestampUpdateThread() {
  uint64_t init_system_clock =
      std::chrono::system_clock::now().time_since_epoch().count();
  uint64_t init_steady_clock =
      std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t curr_steady_clock = 0;
  while (!cpu_timestamp_update_thread_should_stop_) {
#if defined(__HIP_PLATFORM_AMD__)
    for (int i = 0; i < NPKIT_MAX_NUM_GPU_THREADBLOCKS; i++) {
      curr_steady_clock =
          std::chrono::steady_clock::now().time_since_epoch().count();
      NPKIT_STORE_CPU_TIMESTAMP_PER_BLOCK(
          cpu_timestamp_.get(),
          init_system_clock + (curr_steady_clock - init_steady_clock), i);
    }
#else
    curr_steady_clock =
        std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t volatile* volatile_cpu_timestamp_ = cpu_timestamp_.get();
    *volatile_cpu_timestamp_ =
        init_system_clock + (curr_steady_clock - init_steady_clock);
#endif
  }
}

void NpKit::Init(int rank) {
#if defined(ENABLE_NPKIT)
  uint64_t i = 0;
  NpKitEventCollectContext ctx;
  ctx.event_buffer_head = 0;
  rank_ = rank;

  // Init event data structures
  gpu_collect_contexts_ =
      mscclpp::detail::gpuCallocUnique<NpKitEventCollectContext>(
          NpKit::kNumGpuEventBuffers);
  for (i = 0; i < NpKit::kNumGpuEventBuffers; i++) {
    gpu_event_buffers_.emplace_back(
        mscclpp::detail::gpuCallocUnique<NpKitEvent>(
            kMaxNumGpuEventsPerBuffer));
    ctx.event_buffer = gpu_event_buffers_[i].get();
    mscclpp::gpuMemcpy(gpu_collect_contexts_.get() + i, &ctx, 1);
  }

  cpu_collect_contexts_ =
      std::make_unique<NpKitEventCollectContext[]>(NpKit::kNumCpuEventBuffers);
  for (i = 0; i < NpKit::kNumCpuEventBuffers; i++) {
    cpu_event_buffers_.emplace_back(
        std::make_unique<NpKitEvent[]>(kMaxNumCpuEventsPerBuffer));
    ctx.event_buffer = cpu_event_buffers_[i].get();
    cpu_collect_contexts_[i] = ctx;
  }

#if defined(__HIP_PLATFORM_AMD__)
  // Init timestamp. Allocates MAXCHANNELS*128 bytes buffer for GPU
  cpu_timestamp_ = mscclpp::detail::gpuCallocHostUnique<uint64_t[]>(
      NPKIT_MAX_NUM_GPU_THREADBLOCKS * NPKIT_CPU_TIMESTAMP_SLOT_SIZE /
      sizeof(uint64_t));
  for (int i = 0; i < NPKIT_MAX_NUM_GPU_THREADBLOCKS; i++) {
    NPKIT_STORE_CPU_TIMESTAMP_PER_BLOCK(
        cpu_timestamp_.get(),
        std::chrono::system_clock::now().time_since_epoch().count(), i);
  }
#else
  // Init timestamp
  cpu_timestamp_ = mscclpp::detail::gpuCallocHostUnique<uint64_t>();
  uint64_t volatile* volatile_cpu_timestamp = cpu_timestamp_.get();
  *volatile_cpu_timestamp =
      std::chrono::system_clock::now().time_since_epoch().count();
#endif
  cpu_timestamp_update_thread_should_stop_ = false;
  cpu_timestamp_update_thread_ =
      std::make_unique<std::thread>(CpuTimestampUpdateThread);
#else
  WARN("NpKit::Init(%d) : MSCCLPP library was not built with NPKit enabled.",
       rank);
#endif
}

#if defined(ENABLE_NPKIT)
static int GetGpuClockRateInKhz() {
  int dev_id;
#if defined(__HIP_PLATFORM_AMD__)
  cudaDeviceProp dev_prop;
  char gcn_arch[256];
  MSCCLPP_CUDATHROW(cudaGetDevice(&dev_id));
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&dev_prop, dev_id));
  char* gcnArchNameToken = strtok(dev_prop.gcnArchName, ":");
  strcpy(gcn_arch, gcnArchNameToken);
  if (strncmp("gfx94", gcn_arch, 5) == 0)
    return 100000;
  else
    return 25000;
#else
  cudaDeviceProp dev_prop;
  MSCCLPP_CUDATHROW(cudaGetDevice(&dev_id));
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&dev_prop, dev_id));
  return dev_prop.clockRate;
#endif
}
#endif

void NpKit::Dump(std::string const& dump_dir) {
#if defined(ENABLE_NPKIT)
  uint64_t i = 0;
  std::string dump_file_path;

  // Dump CPU events
  for (i = 0; i < NpKit::kNumCpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/cpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_channel_";
    dump_file_path += std::to_string(i);
    auto cpu_trace_file =
        std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    cpu_trace_file.write(
        reinterpret_cast<char*>(cpu_event_buffers_[i].get()),
        cpu_collect_contexts_[i].event_buffer_head * sizeof(NpKitEvent));
    cpu_trace_file.close();
  }

  // Dump CPU clock info
  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_num_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_num_str =
      std::to_string(std::chrono::steady_clock::duration::period::num);
  auto clock_period_num_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_num_file.write(clock_period_num_str.c_str(),
                              clock_period_num_str.length());
  clock_period_num_file.close();

  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_den_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_den_str =
      std::to_string(std::chrono::steady_clock::duration::period::den);
  auto clock_period_den_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_den_file.write(clock_period_den_str.c_str(),
                              clock_period_den_str.length());
  clock_period_den_file.close();

  // Dump GPU events, reuse CPU struct
  for (i = 0; i < NpKit::kNumGpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/gpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_buf_";
    dump_file_path += std::to_string(i);
    mscclpp::gpuMemcpy(cpu_event_buffers_[0].get(), gpu_event_buffers_[i].get(),
                       kMaxNumGpuEventsPerBuffer);
    mscclpp::gpuMemcpy(cpu_collect_contexts_.get(),
                       gpu_collect_contexts_.get() + i, 1);
    auto gpu_trace_file =
        std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    gpu_trace_file.write(
        reinterpret_cast<char*>(cpu_event_buffers_[0].get()),
        cpu_collect_contexts_[0].event_buffer_head * sizeof(NpKitEvent));
    gpu_trace_file.close();
  }

  // Dump GPU clockRate
  dump_file_path = dump_dir;
  dump_file_path += "/gpu_clock_rate_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_rate_str = std::to_string(GetGpuClockRateInKhz());
  auto gpu_clock_rate_file = std::fstream(dump_file_path, std::ios::out);
  gpu_clock_rate_file.write(clock_rate_str.c_str(), clock_rate_str.length());
  gpu_clock_rate_file.close();
#else
  WARN("NpKit::Dump(%s) : MSCCLPP library was not built with NPKit enabled.",
       dump_dir.c_str());
#endif
}

void NpKit::Shutdown() {
#if defined(ENABLE_NPKIT)
  // Stop CPU timestamp updating thread
  cpu_timestamp_update_thread_should_stop_ = true;
  cpu_timestamp_update_thread_->join();

  // Free CPU event data structures
  cpu_event_buffers_.clear();
  cpu_collect_contexts_.reset();

  // Free GPU event data structures
  gpu_event_buffers_.clear();
  gpu_collect_contexts_.reset();

  // Free timestamp
  cpu_timestamp_update_thread_.reset();
  cpu_timestamp_.reset();
#endif
}

NpKitEventCollectContext* NpKit::GetGpuEventCollectContexts() {
  return gpu_collect_contexts_.get();
}

void NpKit::CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd,
                            uint64_t timestamp, int channel_id) {
  uint64_t event_buffer_head =
      cpu_collect_contexts_[channel_id].event_buffer_head;
  if (event_buffer_head < kMaxNumCpuEventsPerBuffer) {
    NpKitEvent& event =
        cpu_collect_contexts_[channel_id].event_buffer[event_buffer_head];
    event.fields.type = type;
    event.fields.size = size;
    event.fields.rsvd = rsvd;
    event.fields.timestamp = timestamp;
    cpu_collect_contexts_[channel_id].event_buffer_head++;
  }
}

uint64_t* NpKit::GetCpuTimestamp() { return cpu_timestamp_.get(); }

#pragma once
#include "../multinode/shared_host_packets.h"
#include "uccl_lite/cpu_router/cpu_executor.hpp"
#include <atomic>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>
#include <sched.h>

namespace uccl_lite {
// Optional CPU lanes only: no NCCL, CUDA calls, or application callbacks in
// the copying loop. Shared storage and ownership come from the peer runtime.
class LocalHostCopy {
 public:
#ifndef UCCL_LITE_LOCAL_HOST_COPY_PARTS
#define UCCL_LITE_LOCAL_HOST_COPY_PARTS 2
#endif
  static constexpr int parts = UCCL_LITE_LOCAL_HOST_COPY_PARTS;
  static_assert(parts == 1 || parts == 2 || parts == 4);
  static constexpr int workers = 2 * parts;

 private:
  struct Job {
    LocalHostCopy* owner;
    uint64_t base;
    size_t count;
    std::atomic<int> done{0};
    Job(LocalHostCopy* owner, uint64_t base, size_t count)
        : owner(owner), base(base), count(count) {}
  };
  multinode::SharedHostPackets& storage_;
  int group_, slots_;
  size_t bytes_;
  std::atomic<bool> stopping_{false};
  std::deque<std::unique_ptr<Job>> jobs_;
  std::unique_ptr<cpu_router::CpuExecutor> executor_;
  uint64_t* flag(int group, int slot, int index) {
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
    return reinterpret_cast<uint64_t*>(storage_.host(group) + 4096 +
                                       (slot * 32 + index) * 64);
#else
    return storage_.flag(group, slot, index);
#endif
  }
  bool wait(uint64_t* flag, uint64_t epoch) {
    while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) < epoch) {
      if (stopping_.load(std::memory_order_relaxed)) return false;
#ifdef UCCL_LITE_LOCAL_HOST_COPY_PAUSE
      asm volatile("pause" ::: "memory");
#else
      std::this_thread::yield();
#endif
    }
    return true;
  }
  static void run(void* context, int lane) {
    auto& job = *static_cast<Job*>(context);
    auto& self = *job.owner;
    int source = (1 - self.group_) * 2 + lane / parts, part = lane % parts;
    size_t elements = self.bytes_ / 4;
    for (size_t offset = 0, tile = 0; offset < job.count;
         offset += elements, ++tile) {
      uint64_t epoch = job.base + tile + 1;
      int slot = (epoch - 1) % self.slots_;
      if (!self.wait(self.flag(1 - self.group_, slot, source), epoch)) break;
      if (epoch > uint64_t(self.slots_)) {
        bool ready = true;
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
        int creditGroup = 1 - self.group_, creditBase = 4 + source * 4;
#else
        int creditGroup = self.group_, creditBase = 4;
#endif
        for (int reader = 0; reader < 4; ++reader)
          ready &= self.wait(self.flag(creditGroup, slot, creditBase + reader),
                             epoch - self.slots_);
        if (!ready) break;
      }
      size_t length = std::min(elements, job.count - offset) * 4;
      size_t half = length / parts;
      size_t payload = self.storage_.payloadOffset(0) + slot * 4 * self.bytes_;
      char const* input = self.storage_.host(1 - self.group_) + payload +
                          (source % 2) * self.bytes_;
      char* output = self.storage_.host(self.group_) + payload +
                     (2 + source % 2) * self.bytes_;
      std::memcpy(output + part * half, input + part * half, half);
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
      constexpr int replicaReady = 20;
#else
      constexpr int replicaReady = 10;
#endif
      __atomic_store_n(self.flag(self.group_, slot,
                                 replicaReady + (source % 2) * parts + part),
                       epoch, __ATOMIC_RELEASE);
    }
    job.done.fetch_add(1, std::memory_order_release);
  }

 public:
  LocalHostCopy(multinode::SharedHostPackets& storage, int group, size_t bytes,
                int slots, int numaNode)
      : storage_(storage), group_(group), slots_(slots), bytes_(bytes) {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
      throw std::runtime_error("read CPU affinity failed");
    std::ifstream file("/sys/devices/system/node/node" +
                       std::to_string(numaNode) + "/cpulist");
    std::string list;
    if (!std::getline(file, list))
      throw std::runtime_error("read NUMA CPU list failed");
    std::istringstream input(list);
    std::string range;
    std::vector<int> cpus;
    while (std::getline(input, range, ',')) {
      auto dash = range.find('-');
      int first = std::stoi(range),
          last = dash == std::string::npos ? first
                                           : std::stoi(range.substr(dash + 1));
      for (int cpu = first; cpu <= last; ++cpu)
        if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    }
    if (cpus.size() < 4)
      throw std::runtime_error(
          "host replica requires four allowed CPUs per NUMA group");
    std::vector<int> lanes;
    for (int i = 0; i < workers; ++i) lanes.push_back(cpus[i % cpus.size()]);
    executor_ = std::make_unique<cpu_router::CpuExecutor>(lanes, 128);
    std::ostringstream binding;
    binding << "UCCL Lite host replica group=" << group << " numa=" << numaNode
            << " workers=" << workers << " cpus=";
    for (int cpu : lanes) binding << cpu << ',';
    std::fprintf(stderr, "%s\n", binding.str().c_str());
  }
  void enqueue(uint64_t base, size_t count) {
    while (!jobs_.empty() &&
           jobs_.front()->done.load(std::memory_order_acquire) == workers)
      jobs_.pop_front();
    jobs_.push_back(std::make_unique<Job>(this, base, count));
    std::array<size_t, workers> lanes{};
    for (int i = 0; i < workers; ++i) lanes[i] = i;
    executor_->enqueueBatch(lanes.data(), workers, run, jobs_.back().get());
  }
  ~LocalHostCopy() {
    stopping_.store(true, std::memory_order_release);
    if (executor_) executor_->stop();
  }
};
}  // namespace uccl_lite

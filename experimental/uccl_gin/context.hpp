#pragma once

#include "uccl_gin/resources.cuh"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class UcclProxy;

namespace uccl_gin {

struct ContextConfig {
  int rank = 0;
  int world_size = 1;
  int local_world_size = 8;
  std::size_t max_message_bytes = 0;
  const char* ifname = "enp71s0";
};

class Context {
 public:
  explicit Context(ContextConfig cfg);
  ~Context();

  Context(Context const&) = delete;
  Context& operator=(Context const&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  UCCLGinResources const& resources() const noexcept { return resources_; }
  void* window() const noexcept { return d_window_; }
  void* send_ptr() const noexcept { return d_window_; }
  void* recv_ptr() const noexcept {
    return static_cast<char*>(d_window_) + max_message_bytes_;
  }
  void* counter_ptr() const noexcept {
    return reinterpret_cast<void*>(resources_.atomic_tail_base);
  }
  std::size_t max_message_bytes() const noexcept { return max_message_bytes_; }
  std::size_t window_bytes() const noexcept { return window_bytes_; }
  int num_queues() const noexcept { return num_queues_; }

 private:
  void setup(ContextConfig cfg);
  void teardown();

  std::vector<std::unique_ptr<UcclProxy>> proxies_;
  void* d_window_ = nullptr;
  std::size_t max_message_bytes_ = 0;
  std::size_t window_bytes_ = 0;
  d2hq::D2HHandle** d_handles_ = nullptr;
  int num_queues_ = 0;
  UCCLGinResources resources_{};
};

}  // namespace uccl_gin

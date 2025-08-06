#pragma once
#include "bench_utils.hpp"
#include "proxy.hpp"
#include "ring_buffer.cuh"
#include <atomic>
#include <memory>
#include <string>
#include <thread>

class PeerCopyManager;

class UcclProxy {
  friend class PeerCopyManager;

 public:
  UcclProxy(uintptr_t rb_addr, int block_idx, uintptr_t gpu_buffer_addr,
            size_t total_size, int rank, std::string const& peer_ip = {});
  ~UcclProxy();

  void start_sender();
  void start_remote();
  void start_local();
  void start_dual();
  void stop();

 private:
  enum class Mode { None, Sender, Remote, Local, Dual };
  void start(Mode m);

  std::string peer_ip_storage_;
  std::unique_ptr<Proxy> proxy_;
  std::thread thread_;
  Mode mode_;
  std::atomic<bool> running_;
};
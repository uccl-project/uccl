#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

// Only the external SDK/transport boundary is substituted. BuffPool and all
// MCCL callback bodies are compiled from their production sources.
struct ibv_mr {
  void* addr;
  uint32_t lkey;
};
#define UCCL_INIT_CHECK(condition, message) assert(condition)
#include "util_buffpool.h"

class ConnectionGate {
 public:
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mu_);
    entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [&] { return released_; });
  }
  void wait_for_worker() {
    std::unique_lock<std::mutex> lock(mu_);
    assert(
        cv_.wait_for(lock, std::chrono::seconds(5), [&] { return entered_; }));
  }
  void release() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool entered_ = false;
  bool released_ = false;
};

inline int gpuGetDevice(int* gpu) {
  *gpu = 0;
  return 0;
}
inline int gpuMalloc(void** data, size_t size) {
  *data = malloc(size);
  return *data ? 0 : 1;
}
inline void gpuFree(void* data) { free(data); }
constexpr int gpuSuccess = 0;
inline int ucclParamNUM_ENGINES() { return 4; }
inline uint32_t str_to_ip(std::string const&) { return 0; }
inline std::string ip_to_str(uint32_t) { return "127.0.0.1"; }

namespace uccl {
constexpr int kMaxReq = 8;
enum ReqType { ReqTx, ReqRx, ReqFlush, ReqTxRC, ReqRxRC };
struct ucclRequest {
  ReqType type;
  void* req_pool;
  void* context;
  int n;
  struct {
    int data_len;
  } send;
  struct {
    int data_len[1];
  } recv;
};
struct UcclFlow {};
struct ConnID {
  void* context = nullptr;
  int sock_fd = -1;
};
struct Mhandle {
  ibv_mr* mr = nullptr;
};
struct FactoryDevice {
  struct {
    uint64_t sys_image_guid = 1;
  } dev_attr;
  double link_bw = 25e9;
  int ib_port_num = 1;
  char const* ib_name = "test";
  std::string local_ip_str = "127.0.0.1";
};
struct RDMAFactory {
  static FactoryDevice* get_factory_dev(int) {
    static FactoryDevice device;
    return &device;
  }
};
class RDMAEndpoint {
 public:
  explicit RDMAEndpoint(int) {}
  ConnectionGate* gate = nullptr;
  bool fail_connection = false;
  UcclFlow flow;
  int get_best_dev_idx(int) { return 0; }
  void initialize_engine_by_dev(int, bool) {}
  int get_num_devices() { return 1; }
  ConnID uccl_connect(int, int, int, int, std::string, uint16_t) {
    gate->arrive_and_wait();
    if (fail_connection) throw std::runtime_error("connect failed");
    return {&flow, 42};
  }
  ConnID uccl_accept(int, int, int, std::string& ip, int* dev, int* gpu) {
    gate->arrive_and_wait();
    if (fail_connection) throw std::runtime_error("accept failed");
    ip = "192.0.2.17";
    *dev = 7;
    *gpu = 3;
    return {&flow, 43};
  }
  template <typename Device>
  int uccl_regmr(Device, void*, size_t, int, Mhandle**) {
    return -1;
  }
  int uccl_regmr_dmabuf(UcclFlow*, void*, size_t, int, int, int, Mhandle**) {
    return -1;
  }
  void uccl_deregmr(Mhandle*) {}
  int uccl_send_async(UcclFlow*, Mhandle*, void*, size_t, ucclRequest*) {
    return -1;
  }
  int uccl_recv_async(UcclFlow*, Mhandle**, void**, int*, int, ucclRequest*) {
    return -1;
  }
  int uccl_flush(UcclFlow*, Mhandle**, void**, int*, int, ucclRequest*) {
    return -1;
  }
  bool uccl_poll_ureq_once(ucclRequest*) { return false; }
};
}  // namespace uccl

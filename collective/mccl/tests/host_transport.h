#pragma once

#include <arpa/inet.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Only the external SDK/transport boundary is substituted. BuffPool and all
// MCCL callback bodies are compiled from their production sources.
struct ibv_mr {
  void* addr;
  uint32_t lkey;
  uint32_t rkey;
};
struct ibv_sge {
  uint64_t addr;
  uint32_t length;
  uint32_t lkey;
};
struct ibv_send_wr {
  uint64_t wr_id;
  struct {
    struct {
      uint64_t remote_addr;
      uint32_t rkey;
    } rdma;
  } wr;
  ibv_sge* sg_list;
  int num_sge;
  int opcode;
  int send_flags;
};
struct ibv_wc {
  int opcode;
  uint64_t wr_id;
  uint32_t imm_data;
  uint32_t qp_num;
};
struct ibv_cq {
  std::deque<ibv_wc> completions;
  int polls = 0;
};
struct ibv_qp {
  uint32_t qp_num = 1;
  std::vector<ibv_send_wr> pending;
  bool pending_at_post = false;
};
constexpr int IBV_WC_RDMA_WRITE = 1;
constexpr int IBV_WC_RECV_RDMA_WITH_IMM = 2;
constexpr int IBV_WC_RDMA_READ = 3;
constexpr int IBV_WR_RDMA_READ = 4;
constexpr int IBV_SEND_SIGNALED = 1;
inline int ibv_poll_cq(ibv_cq* cq, int count, ibv_wc* wcs) {
  ++cq->polls;
  int n = 0;
  while (n < count && !cq->completions.empty()) {
    wcs[n++] = cq->completions.front();
    cq->completions.pop_front();
  }
  return n;
}
inline int ibv_post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr**) {
  assert(wr->opcode == IBV_WR_RDMA_READ);
  qp->pending_at_post = *reinterpret_cast<uint64_t*>(wr->wr_id) == 0;
  qp->pending.push_back(*wr);
  return 0;
}
#define UCCL_INIT_CHECK(condition, message) assert(condition)
#define UCCL_CHECK(condition) assert(condition)
inline std::ostringstream host_log;
#define UCCL_LOG_EP host_log
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
constexpr int kMaxRecv = 1;
constexpr int kMaxBatchCQ = 16;
constexpr int kRCSize = 0;
constexpr int NCCL_MIN_POST_RECV = 65536;
constexpr int kEngineLBPolicy = 1;
constexpr int ENGINE_POLICY_LOAD = 3;
struct PollCtx {
  bool done;
};
#include "request.inc"
struct ConnID {
  void* context = nullptr;
  int sock_fd = -1;
};
struct Mhandle {
  ibv_mr* mr = nullptr;
};
struct UcclFlow {
  ibv_cq cq;
  ibv_qp qp;
  struct {
    struct {
      ibv_cq* flow_cq;
      ibv_qp* rc_qp;
    } base;
    ibv_sge gpu_flush_sge;
    ibv_qp* gpu_flush_qp;
  } recv_comm_{{&cq, &qp}, {}, &qp};
  int flow_cq_cnt_ = 0;
  void poll_flow_cq();
  void post_flush(Mhandle**, void**, int*, int, uint64_t*, int);
  void dec_outstanding_reqs() {}
  void set_last_rc_size(int) {}
  int get_last_rc_size() { return 0; }
#include "flush_check.inc"

  void deliver_completions() {
    for (auto const& wr : qp.pending) {
      cq.completions.push_back({IBV_WC_RDMA_READ, wr.wr_id, 0, qp.qp_num});
    }
    qp.pending.clear();
  }
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
  int uccl_flush(UcclFlow*, Mhandle**, void**, int*, int, ucclRequest*);
  bool uccl_poll_ureq_once(ucclRequest*);
  bool uccl_poll_once(PollCtx* ctx) { return ctx->done; }
  void dec_load_on_engine(uint32_t) {}
};
#include "endpoint_completion.inc"
#include "flow_completion.inc"
}  // namespace uccl

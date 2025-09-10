#pragma once

#include <atomic>
#include <memory>

enum class RequestType { SEND, RECV };
enum class ReductionType { NONE, SUM, MAX };

struct Request {
  unsigned id;
  void* buf;
  size_t offset;  // default 0
  size_t len;
  uint32_t local_mr_id;
  uint32_t remote_mr_id;
  bool on_gpu;
  bool do_reduction;
  ReductionType reduction_op;
  RequestType reqtype;

  std::atomic<bool> running{false};
  std::atomic<bool> finished{false};
  std::atomic<bool> failed{false};
  static std::atomic<unsigned> global_id_counter;

  std::atomic<int> pending_signaled{0};  // how many qps we used
  std::atomic<int> pending_computed{0};  // how many compute events

  void on_comm_done(bool done = true);
  void on_compute_done();

  Request(void* buf, size_t offset, size_t len, uint32_t local_mr_id,
          uint32_t remote_mr_id, bool gpu,
          RequestType reqtype = RequestType::SEND, bool reduction = false,
          ReductionType op = ReductionType::NONE)
      : id(global_id_counter.fetch_add(1, std::memory_order_relaxed)),
        buf(buf),
        offset(offset),
        len(len),
        local_mr_id(local_mr_id),
        remote_mr_id(remote_mr_id),
        on_gpu(gpu),
        do_reduction(reduction),
        reduction_op(op),
        reqtype(reqtype) {}
};

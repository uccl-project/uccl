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
  bool on_gpu;
  bool do_reduction;
  ReductionType reduction_op;
  RequestType reqtype;

  std::atomic<bool> running{false};
  std::atomic<bool> finished{false};
  static std::atomic<unsigned> global_id_counter;

  void onCommCompletion();
  void onComputeCompletion();

  Request(void* buf, size_t len, bool gpu,
          RequestType reqtype = RequestType::SEND, bool reduction = false,
          ReductionType op = ReductionType::NONE, size_t offset = 0)
      : id(global_id_counter.fetch_add(1, std::memory_order_relaxed)),
        buf(buf),
        offset(offset),
        len(len),
        on_gpu(gpu),
        do_reduction(reduction),
        reduction_op(op),
        reqtype(reqtype) {}
};

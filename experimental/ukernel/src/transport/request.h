#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace UKernel {
namespace Transport {

enum class RequestType { NONE, SEND, RECV };

struct Request {
  unsigned id;
  uint64_t match_seq;
  void* buf;
  size_t offset;  // default 0
  size_t len;
  uint32_t local_mr_id;
  uint32_t remote_mr_id;
  bool on_gpu;
  RequestType request_type;

  std::atomic<bool> running{false};
  std::atomic<bool> finished{false};
  std::atomic<bool> failed{false};
  std::atomic<int> pending_signaled{0};  // How many qps we used

  void on_comm_done();

  Request(unsigned id, uint64_t match_seq, void* buf, size_t offset, size_t len,
          uint32_t local_mr_id, uint32_t remote_mr_id, bool gpu,
          RequestType reqtype = RequestType::SEND)
      : id(id),
        match_seq(match_seq),
        buf(buf),
        offset(offset),
        len(len),
        local_mr_id(local_mr_id),
        remote_mr_id(remote_mr_id),
        on_gpu(gpu),
        request_type(reqtype) {}
};

}  // namespace Transport
}  // namespace UKernel

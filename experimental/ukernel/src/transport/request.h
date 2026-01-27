#pragma once

#include <atomic>
#include <memory>

namespace UKernel {
namespace Transport {

enum class RequestType { NONE, SEND, RECV };

struct Request {
  unsigned id;
  void* buf;
  size_t offset;  // default 0
  size_t len;
  uint16_t local_mr_id;
  uint16_t remote_mr_id;
  bool on_gpu;
  RequestType request_type;

  std::atomic<bool> running{false};
  std::atomic<bool> finished{false};
  std::atomic<bool> failed{false};
  std::atomic<bool> notified{false};
  static std::atomic<unsigned> global_id_counter;

  std::atomic<int> pending_signaled{0};  // How many qps we used

  void on_comm_done();

  Request(unsigned id, void* buf, size_t offset, size_t len,
          uint16_t local_mr_id, uint16_t remote_mr_id, bool gpu,
          RequestType reqtype = RequestType::SEND)
      : id(id),
        buf(buf),
        offset(offset),
        len(len),
        local_mr_id(local_mr_id),
        remote_mr_id(remote_mr_id),
        on_gpu(gpu),
        request_type(reqtype) {}
};

static inline unsigned make_request_id(uint16_t receiver_rank, uint8_t mr_id,
                                       uint16_t seq, bool is_red = false) {
  unsigned base = ((static_cast<unsigned>(receiver_rank) & 0xFFF)
                   << 20) |  // [30:20] 11 bits → receiver_rank (2048)
                  ((static_cast<unsigned>(mr_id) & 0xFF)
                   << 12) |       // [19:12] 8 bits  → mr_id (255)
                  (seq & 0xFFF);  // [11:0]  12 bits → seq (4096)

  return is_red ? (0x80000000u | base) : base;
}

static inline void parse_request_id(unsigned req_id, uint16_t& receiver_rank,
                                    uint8_t& mr_id, uint16_t& seq) {
  receiver_rank = (req_id >> 20) & 0xFFF;
  mr_id = (req_id >> 12) & 0xFF;
  seq = req_id & 0xFFF;
}

}  // namespace Transport
}  // namespace UKernel
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
class Endpoint;
struct FifoItem;
namespace uccl::device {
class NetworkStaging {
 public:
  explicit NetworkStaging(Endpoint& endpoint) : endpoint_(endpoint) {}
  ~NetworkStaging();
  void request_stop() { stop_.store(true, std::memory_order_release); }
  void shutdown();
  bool advertise(uint64_t mr_id, void* dst, size_t len, char* out_buf);
  bool write(uint64_t conn_id, void* src, size_t size, FifoItem const& item);
  bool read(uint64_t conn_id, void* dst, size_t size, FifoItem const& item);

 private:
  Endpoint& endpoint_;
  std::atomic<bool> stop_{false};
  static constexpr size_t kMluStageChunk = 32ul << 20;
  struct MluStageRx {
    void* staging = nullptr;
    uint64_t staging_mr_id = 0;
    void* dst = nullptr;  // user buffer: write dst / read src
    size_t total = 0;
    uint64_t done_seq = 0;    // write path: seq already copied back by poller
    uint64_t served_req = 0;  // read path: req already served by poller
  };
  std::unordered_map<uint64_t, MluStageRx*> mlu_rx_;  // key = user dst addr
  std::mutex mlu_rx_mu_;
  std::thread mlu_poller_thread_;
  std::atomic<bool> mlu_poller_started_{false};
  std::mutex mlu_tx_mu_;  // serialize sender-side staging
  void* mlu_tx_staging_ = nullptr;
  uint64_t mlu_tx_staging_mr_ = 0;
  uint64_t* mlu_tx_ctrl_ = nullptr;  // host: [off, len, seq, ack]
  uint64_t mlu_tx_ctrl_mr_ = 0;
  uint64_t mlu_tx_next_seq_ = 0;
  bool mlu_tx_ready_ = false;
  bool ensure_tx();
  void release_tx();
  bool rdma_blocking(uint64_t conn_id, uint64_t mr_id, void* buf, size_t size,
                     FifoItem const& item, bool is_write);
  void poller_func();
};
}  // namespace uccl::device

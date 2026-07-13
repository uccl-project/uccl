// Cambricon MLU (CNRT) Plan A staging: members/methods of class Endpoint.
// Included inside the Endpoint class body in engine.h under
// __CAMBRICON_PLATFORM_MLU__; implementations live in mlu_staging.cc.
// Plain cnrtMalloc tensors aren't peer-able; relay them through a bounded
// peer-able staging buffer in chunks.
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
bool mlu_ensure_tx();
bool mlu_rdma_blocking(uint64_t conn_id, uint64_t mr_id, void* buf, size_t size,
                       FifoItem const& item, bool is_write);
bool mlu_setup_rx(uint64_t mr_id, void* dst, size_t len, char* out_buf);
bool mlu_staged_write(uint64_t conn_id, void* src, size_t size,
                      FifoItem const& staging_item);
bool mlu_staged_read(uint64_t conn_id, void* dst, size_t size,
                     FifoItem const& staging_item);
void mlu_poller_func();
// cnrtAcquireMemHandle succeeds only once per allocation; cache by base ptr.
std::mutex mlu_ipc_handle_mu_;
std::unordered_map<void*, gpuIpcMemHandle_t> mlu_ipc_handle_cache_;
bool mlu_acquire_ipc_handle(void* base_ptr, gpuIpcMemHandle_t* out);

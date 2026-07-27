// Cambricon MLU (CNRT) Plan A staging: implementations of Endpoint::mlu_*
// (declared in mlu_staging.inc). Compiled only by Makefile.cnrt.
#if defined(__CAMBRICON_PLATFORM_MLU__)
#include "endpoint_wrapper.h"
#include "engine.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util/util.h"
#include <chrono>
#include <thread>
#include <Python.h>

// Local copy of engine.cc's helper to keep this TU self-contained.
static inline void check_python_signals() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  if (PyErr_CheckSignals() != 0) {
    UCCL_LOG(FATAL) << "Python signal caught, exiting...";
  }
  PyGILState_Release(gstate);
}
#endif

#if defined(__CAMBRICON_PLATFORM_MLU__)
// staging tail, 8 bytes each: write path off/len/seq/ack, read path
// roff/rlen/req/ready.
static constexpr size_t kMluTailOff = 0, kMluTailLen = 8, kMluTailSeq = 16,
                        kMluTailAck = 24, kMluTailROff = 32, kMluTailRLen = 40,
                        kMluTailReq = 48, kMluTailReady = 56,
                        kMluTailBytes = 64;

static bool mlu_staging_alloc(size_t bytes, void** out) {
  // 6.2.10: GDR reg needs plain device memory, not cnMallocPeerAble.
  return cnrtMalloc(out, bytes) == cnrtSuccess;
}

bool Endpoint::mlu_ensure_tx() {
  if (mlu_tx_ready_) return true;
  cnrtSetDevice(local_gpu_idx_);  // cnrtMalloc binds to current device
  size_t blk = kMluStageChunk + kMluTailBytes;
  if (!mlu_staging_alloc(blk, &mlu_tx_staging_)) return false;
  if (!reg(mlu_tx_staging_, blk, mlu_tx_staging_mr_)) return false;
  mlu_tx_ctrl_ = new uint64_t[4]();
  if (!reg(mlu_tx_ctrl_, 4 * sizeof(uint64_t), mlu_tx_ctrl_mr_)) return false;
  mlu_tx_ready_ = true;
  return true;
}

bool Endpoint::mlu_rdma_blocking(uint64_t conn_id, uint64_t mr_id, void* buf,
                                 size_t size, FifoItem const& item,
                                 bool is_write) {
  Conn* conn = get_conn(conn_id);
  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (conn == nullptr || mhandle == nullptr) return false;
  UcclRequest ureq = {};
  FifoItem slot = item;
  slot.size = size;
  bool done = false;
  if (is_write) {
    SendConnection* send_group =
        uccl_resolve_send_group(ep_, conn->uccl_conn_id_.peer_id);
    while ((send_group != nullptr
                ? uccl_write_async_on_group(send_group, conn, mhandle, buf,
                                            size, slot, &ureq)
                : uccl_write_async(ep_, conn, mhandle, buf, size, slot,
                                   &ureq)) == -1)
      ;
    while (!done) {
      if (inside_python) check_python_signals();
      if (uccl_poll_ureq_once(ep_, &ureq)) done = true;
    }
  } else {
    while (uccl_read_async(ep_, conn, mhandle, buf, size, slot, &ureq) == -1)
      ;
    while (!done) {
      if (inside_python) check_python_signals();
      uccl_drive_recv(ep_);
      if (uccl_poll_ureq_once(ep_, &ureq)) done = true;
    }
  }
  return true;
}

bool Endpoint::mlu_setup_rx(uint64_t /*mr_id*/, void* dst, size_t len,
                            char* out_buf) {
  MluStageRx* e = nullptr;
  {
    std::lock_guard<std::mutex> lk(mlu_rx_mu_);
    auto it = mlu_rx_.find((uint64_t)dst);
    if (it != mlu_rx_.end()) {
      e = it->second;
    } else {
      e = new MluStageRx();
      cnrtSetDevice(local_gpu_idx_);  // cnrtMalloc binds to current device
      size_t blk = kMluStageChunk + kMluTailBytes;
      if (!mlu_staging_alloc(blk, &e->staging)) {
        delete e;
        return false;
      }
      cnrtMemset((char*)e->staging + kMluStageChunk, 0, kMluTailBytes);
      if (!reg(e->staging, blk, e->staging_mr_id)) {
        delete e;
        return false;
      }
      mlu_rx_[(uint64_t)dst] = e;
    }
    e->dst = dst;
    e->total = len;
  }
  P2PMhandle* mh = get_mhandle(e->staging_mr_id);
  if (mh == nullptr ||
      prepare_fifo_metadata(ep_, mh, e->staging, kMluStageChunk + kMluTailBytes,
                            out_buf) == -1)
    return false;
  if (!mlu_poller_started_.exchange(true))
    mlu_poller_thread_ = std::thread(&Endpoint::mlu_poller_func, this);
  return true;
}

bool Endpoint::mlu_staged_write(uint64_t conn_id, void* src, size_t size,
                                FifoItem const& staging_item) {
  std::lock_guard<std::mutex> lk(mlu_tx_mu_);
  if (!mlu_ensure_tx()) return false;
  cnrtSetDevice(local_gpu_idx_);
  uint64_t const base = staging_item.addr;
  size_t off = 0;
  while (off < size) {
    size_t len = (kMluStageChunk < size - off) ? kMluStageChunk : (size - off);
    if (cnrtMemcpy(mlu_tx_staging_, (char*)src + off, len,
                   cnrtMemcpyDevToDev) != 0)
      return false;
    uint64_t seq = ++mlu_tx_next_seq_;
    FifoItem d = staging_item;  // data chunk -> head of remote staging
    d.addr = base;
    if (!mlu_rdma_blocking(conn_id, mlu_tx_staging_mr_, mlu_tx_staging_, len, d,
                           true))
      return false;
    mlu_tx_ctrl_[0] =
        off;  // land off/len first (data acked remote), then seq to commit
    mlu_tx_ctrl_[1] = len;
    FifoItem ol = staging_item;
    ol.addr = base + kMluStageChunk + kMluTailOff;
    if (!mlu_rdma_blocking(conn_id, mlu_tx_ctrl_mr_, mlu_tx_ctrl_,
                           2 * sizeof(uint64_t), ol, true))
      return false;
    mlu_tx_ctrl_[2] = seq;
    FifoItem sq = staging_item;
    sq.addr = base + kMluStageChunk + kMluTailSeq;
    if (!mlu_rdma_blocking(conn_id, mlu_tx_ctrl_mr_, &mlu_tx_ctrl_[2],
                           sizeof(uint64_t), sq, true))
      return false;
    FifoItem ak = staging_item;  // RDMA-READ poll until ack == seq
    ak.addr = base + kMluStageChunk + kMluTailAck;
    for (;;) {
      mlu_tx_ctrl_[3] = 0;
      if (!mlu_rdma_blocking(conn_id, mlu_tx_ctrl_mr_, &mlu_tx_ctrl_[3],
                             sizeof(uint64_t), ak, false))
        return false;
      if (mlu_tx_ctrl_[3] >= seq) break;
    }
    off += len;
  }
  return true;
}

bool Endpoint::mlu_staged_read(uint64_t conn_id, void* dst, size_t size,
                               FifoItem const& staging_item) {
  std::lock_guard<std::mutex> lk(mlu_tx_mu_);
  if (!mlu_ensure_tx()) return false;
  cnrtSetDevice(local_gpu_idx_);
  uint64_t const base = staging_item.addr;
  size_t off = 0;
  while (off < size) {
    size_t len = (kMluStageChunk < size - off) ? kMluStageChunk : (size - off);
    uint64_t req = ++mlu_tx_next_seq_;
    mlu_tx_ctrl_[0] =
        off;  // land roff/rlen first, then req to commit the request
    mlu_tx_ctrl_[1] = len;
    FifoItem rl = staging_item;
    rl.addr = base + kMluStageChunk + kMluTailROff;
    if (!mlu_rdma_blocking(conn_id, mlu_tx_ctrl_mr_, mlu_tx_ctrl_,
                           2 * sizeof(uint64_t), rl, true))
      return false;
    mlu_tx_ctrl_[2] = req;
    FifoItem rq = staging_item;
    rq.addr = base + kMluStageChunk + kMluTailReq;
    if (!mlu_rdma_blocking(conn_id, mlu_tx_ctrl_mr_, &mlu_tx_ctrl_[2],
                           sizeof(uint64_t), rq, true))
      return false;
    FifoItem rdy = staging_item;  // poll until remote ready == req
    rdy.addr = base + kMluStageChunk + kMluTailReady;
    for (;;) {
      mlu_tx_ctrl_[3] = 0;
      if (!mlu_rdma_blocking(conn_id, mlu_tx_ctrl_mr_, &mlu_tx_ctrl_[3],
                             sizeof(uint64_t), rdy, false))
        return false;
      if (mlu_tx_ctrl_[3] >= req) break;
    }
    FifoItem d =
        staging_item;  // remote data ready -> RDMA-READ local -> D2D to dst
    d.addr = base;
    if (!mlu_rdma_blocking(conn_id, mlu_tx_staging_mr_, mlu_tx_staging_, len, d,
                           false))
      return false;
    if (cnrtMemcpy((char*)dst + off, mlu_tx_staging_, len,
                   cnrtMemcpyDevToDev) != 0)
      return false;
    off += len;
  }
  return true;
}

void Endpoint::mlu_poller_func() {
  cnrtSetDevice(local_gpu_idx_);
  while (!stop_.load(std::memory_order_acquire)) {
    std::vector<MluStageRx*> snap;
    {
      std::lock_guard<std::mutex> lk(mlu_rx_mu_);
      snap.reserve(mlu_rx_.size());
      for (auto& kv : mlu_rx_) snap.push_back(kv.second);
    }
    bool any = false;
    for (auto* e : snap) {
      char* tail = (char*)e->staging + kMluStageChunk;
      uint64_t seq = 0, ol[2] = {0, 0};
      // write path: peer wrote staging -> copy back to dst, set ack
      if (cnrtMemcpy(&seq, tail + kMluTailSeq, sizeof(uint64_t),
                     cnrtMemcpyDevToHost) == 0 &&
          seq > e->done_seq &&
          cnrtMemcpy(ol, tail + kMluTailOff, 2 * sizeof(uint64_t),
                     cnrtMemcpyDevToHost) == 0 &&
          ol[1] > 0 && ol[0] + ol[1] <= e->total &&
          cnrtMemcpy((char*)e->dst + ol[0], e->staging, ol[1],
                     cnrtMemcpyDevToDev) == 0) {
        e->done_seq = seq;
        cnrtMemcpy(tail + kMluTailAck, &seq, sizeof(uint64_t),
                   cnrtMemcpyHostToDev);
        any = true;
      }
      uint64_t req = 0, rol[2] = {0, 0};
      // read path: peer requested -> move src into staging for its RDMA-READ,
      // set ready
      if (cnrtMemcpy(&req, tail + kMluTailReq, sizeof(uint64_t),
                     cnrtMemcpyDevToHost) == 0 &&
          req > e->served_req &&
          cnrtMemcpy(rol, tail + kMluTailROff, 2 * sizeof(uint64_t),
                     cnrtMemcpyDevToHost) == 0 &&
          rol[1] > 0 && rol[0] + rol[1] <= e->total &&
          cnrtMemcpy(e->staging, (char*)e->dst + rol[0], rol[1],
                     cnrtMemcpyDevToDev) == 0) {
        e->served_req = req;
        cnrtMemcpy(tail + kMluTailReady, &req, sizeof(uint64_t),
                   cnrtMemcpyHostToDev);
        any = true;
      }
    }
    if (!any) std::this_thread::sleep_for(std::chrono::microseconds(5));
  }
}
#endif

#if defined(__CAMBRICON_PLATFORM_MLU__)
// cnrtAcquireMemHandle succeeds only once per allocation; cache so repeat
// advertise reuses it.
bool Endpoint::mlu_acquire_ipc_handle(void* base_ptr, gpuIpcMemHandle_t* out) {
  std::lock_guard<std::mutex> lk(mlu_ipc_handle_mu_);
  auto it = mlu_ipc_handle_cache_.find(base_ptr);
  if (it != mlu_ipc_handle_cache_.end()) {
    *out = it->second;
    return true;
  }
  if (gpuIpcGetMemHandle(out, base_ptr) != gpuSuccess) return false;
  mlu_ipc_handle_cache_[base_ptr] = *out;
  return true;
}
#endif

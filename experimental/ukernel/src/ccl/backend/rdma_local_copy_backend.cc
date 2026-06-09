#include "rdma_local_copy_backend.h"
#include "../../include/gpu_rt.h"
#include "../coll_types.h"
#include "../lower.h"
#include "../utils.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace UKernel {
namespace CCL {

RdmaLocalCopyBackend::RdmaLocalCopyBackend(
    RdmaLocalCopyBackendConfig const& config)
    : config_(config) {
  if (!init_device()) {
    degraded_ = true;
    return;
  }
  if (!init_qps()) {
    degraded_ = true;
    return;
  }
  for (int i = 0; i < kMaxReq; ++i) {
    reqs_[i].wr_id = i + 1;
    reqs_[i].done.store(true, std::memory_order_relaxed);
  }
  poll_thread_ = std::make_unique<std::thread>(
      &RdmaLocalCopyBackend::poll_loop, this);
}

RdmaLocalCopyBackend::~RdmaLocalCopyBackend() {
  poll_stop_ = true;
  if (poll_thread_ && poll_thread_->joinable()) poll_thread_->join();
  for (auto& [id, mr] : mr_map_) {
    if (mr) ibv_dereg_mr(mr);
  }
  if (qp_send_) ibv_destroy_qp(qp_send_);
  if (cq_send_) ibv_destroy_cq(cq_send_);
  if (qp_recv_) ibv_destroy_qp(qp_recv_);
  if (cq_recv_) ibv_destroy_cq(cq_recv_);
  if (pd_) ibv_dealloc_pd(pd_);
  if (ctx_) ibv_close_device(ctx_);
}

char const* RdmaLocalCopyBackend::name() const {
  return degraded_ ? "degraded" : "rdma-local-copy";
}

bool RdmaLocalCopyBackend::is_degraded() const {
  return degraded_;
}

bool RdmaLocalCopyBackend::supports(OpKind kind) const {
  return !is_degraded() && kind == OpKind::Copy;
}

bool RdmaLocalCopyBackend::init_device() {
  int num_devs = 0;
  ibv_device** dev_list = ibv_get_device_list(&num_devs);
  if (!dev_list || num_devs == 0) {
    std::fprintf(stderr, "[rdma] no RDMA devices found\n");
    return false;
  }

  for (int i = 0; i < num_devs; ++i) {
    char const* dev_name = ibv_get_device_name(dev_list[i]);
    ibv_context* ctx = ibv_open_device(dev_list[i]);
    if (!ctx) {
      std::fprintf(stderr, "[rdma] %s: open failed\n", dev_name);
      continue;
    }

    ibv_port_attr pa;
    if (ibv_query_port(ctx, 1, &pa) != 0) {
      std::fprintf(stderr, "[rdma] %s: query_port failed\n", dev_name);
      ibv_close_device(ctx);
      continue;
    }
    if (pa.state != IBV_PORT_ACTIVE) {
      std::fprintf(stderr, "[rdma] %s port 1: state=%d (need %d=ACTIVE)\n",
                   dev_name, pa.state, IBV_PORT_ACTIVE);
      ibv_close_device(ctx);
      continue;
    }

    ibv_device_attr da;
    if (ibv_query_device(ctx, &da) != 0) {
      std::fprintf(stderr, "[rdma] %s: query_device failed\n", dev_name);
      ibv_close_device(ctx);
      continue;
    }

    union ibv_gid gid;
    int gid_idx = -1;
    for (int p = 0; p < da.phys_port_cnt; ++p) {
      for (int j = 0; j < 16; ++j) {
        if (ibv_query_gid(ctx, (uint8_t)(p + 1), j, &gid) == 0) {
          if (gid.global.interface_id) {
            gid_idx = j;
            goto found_gid;
          }
        }
      }
    }
  found_gid:
    if (gid_idx < 0) {
      std::fprintf(stderr, "[rdma] %s: no RoCE GID found\n", dev_name);
      ibv_close_device(ctx);
      continue;
    }

    ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
      std::fprintf(stderr, "[rdma] %s: alloc_pd failed\n", dev_name);
      ibv_close_device(ctx);
      continue;
    }

    ctx_ = ctx;
    pd_ = pd;
    dev_attr_ = da;
    gid_ = gid;
    gid_index_ = (uint8_t)gid_idx;

    std::fprintf(stderr, "[rdma] using %s, gid_idx=%d\n", dev_name, gid_idx);
    ibv_free_device_list(dev_list);
    return true;
  }

  std::fprintf(stderr, "[rdma] no usable RDMA device found\n");
  ibv_free_device_list(dev_list);
  return false;
}

bool RdmaLocalCopyBackend::init_qps() {
  cq_send_ = ibv_create_cq(ctx_, 256, nullptr, nullptr, 0);
  if (!cq_send_) {
    std::fprintf(stderr, "[rdma] create cq_send failed\n");
    return false;
  }

  cq_recv_ = ibv_create_cq(ctx_, 256, nullptr, nullptr, 0);
  if (!cq_recv_) {
    std::fprintf(stderr, "[rdma] create cq_recv failed\n");
    return false;
  }

  ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = cq_send_;
  qp_attr.recv_cq = cq_send_;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = kMaxReq;
  qp_attr.cap.max_recv_wr = 1;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;

  qp_send_ = ibv_create_qp(pd_, &qp_attr);
  if (!qp_send_) {
    std::fprintf(stderr, "[rdma] create qp_send failed\n");
    return false;
  }

  qp_attr.send_cq = cq_recv_;
  qp_attr.recv_cq = cq_recv_;
  qp_recv_ = ibv_create_qp(pd_, &qp_attr);
  if (!qp_recv_) {
    std::fprintf(stderr, "[rdma] create qp_recv failed\n");
    return false;
  }
  std::fprintf(stderr, "[rdma] QPs created: send=%u recv=%u\n",
               qp_send_->qp_num, qp_recv_->qp_num);

  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
              IBV_QP_ACCESS_FLAGS;

  if (ibv_modify_qp(qp_send_, &attr, flags)) {
    std::fprintf(stderr, "[rdma] send INIT failed: %s\n", strerror(errno));
    return false;
  }
  if (ibv_modify_qp(qp_recv_, &attr, flags)) {
    std::fprintf(stderr, "[rdma] recv INIT failed: %s\n", strerror(errno));
    return false;
  }
  std::fprintf(stderr, "[rdma] INIT OK\n");

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = qp_recv_->qp_num;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.is_global = 1;
  attr.ah_attr.grh.dgid = gid_;
  attr.ah_attr.grh.sgid_index = gid_index_;
  attr.ah_attr.grh.hop_limit = 64;
  attr.ah_attr.port_num = 1;
  flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  if (ibv_modify_qp(qp_send_, &attr, flags)) {
    std::fprintf(stderr, "[rdma] send RTR failed: %s\n", strerror(errno));
    return false;
  }

  attr.dest_qp_num = qp_send_->qp_num;
  if (ibv_modify_qp(qp_recv_, &attr, flags)) {
    std::fprintf(stderr, "[rdma] recv RTR failed: %s\n", strerror(errno));
    return false;
  }
  std::fprintf(stderr, "[rdma] RTR OK\n");

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = 0;
  attr.max_rd_atomic = 1;
  flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

  if (ibv_modify_qp(qp_send_, &attr, flags)) {
    std::fprintf(stderr, "[rdma] send RTS failed: %s\n", strerror(errno));
    return false;
  }
  if (ibv_modify_qp(qp_recv_, &attr, flags)) {
    std::fprintf(stderr, "[rdma] recv RTS failed: %s\n", strerror(errno));
    return false;
  }
  std::fprintf(stderr, "[rdma] RTS OK\n");

  return true;
}

void RdmaLocalCopyBackend::poll_loop() {
  ibv_wc wc[64];
  while (!poll_stop_.load(std::memory_order_relaxed)) {
    int n = ibv_poll_cq(cq_send_, 64, wc);
    for (int i = 0; i < n; ++i) {
      uint64_t wr_id = wc[i].wr_id;
      if (wr_id >= 1 && wr_id <= (uint64_t)kMaxReq) {
        auto& r = reqs_[wr_id - 1];
        r.failed = (wc[i].status != IBV_WC_SUCCESS);
        r.done.store(true, std::memory_order_release);
      }
    }
    if (n == 0) std::this_thread::yield();
  }
}

void RdmaLocalCopyBackend::validate(TiledResult const& tiled,
                                    void* input_ptr, void* output_ptr,
                                    void* scratch_ptr) {
  if (is_degraded()) return;

  buf_count_ = 0;
  auto reg_buf = [&](uint32_t id, void* ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0) return;
    if (mr_map_.count(id)) return;

    ibv_mr* mr = ibv_reg_mr(
        pd_, ptr, bytes,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ);
    if (!mr) {
      std::fprintf(stderr, "[rdma] reg_mr id=%u FAILED: %s (errno=%d)\n",
                   id, strerror(errno), errno);
    }
    if (mr) {
      mr_map_[id] = mr;
      registered_bufs_[buf_count_] = {id, ptr};
      ++buf_count_;
      return;
    }

    if (errno != EFAULT || bytes == 0) {
      degraded_ = true;
      return;
    }

    static gpuMemGetHandleForAddressRange_fn dma_export_fn = nullptr;
    static bool dma_init = false;
    if (!dma_init) {
      dma_init = true;
      void* lib = dlopen(GPU_DRIVER_LIB_NAME, RTLD_NOW);
      if (!lib)
        lib = dlopen(GPU_DRIVER_LIB_NAME_FALLBACK, RTLD_NOW);
      if (lib) {
        dma_export_fn = (gpuMemGetHandleForAddressRange_fn)dlsym(
            lib, GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME);
      }
    }

    if (!dma_export_fn) {
      std::fprintf(stderr, "[rdma] DMABUF not available (%s not found in %s)\n",
                   GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME,
                   GPU_DRIVER_LIB_NAME);
      degraded_ = true;
      return;
    }

    int dmabuf_fd = -1;
    gpuDriverResult_t r = dma_export_fn(
        &dmabuf_fd, (gpuDevicePtr_t)ptr, bytes,
        (gpuMemRangeHandleType)GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (r != gpuDriverSuccess || dmabuf_fd < 0) {
      std::fprintf(stderr, "[rdma] DMABUF export failed: result=%d fd=%d\n",
                   (int)r, dmabuf_fd);
      degraded_ = true;
      return;
    }

    mr = ibv_reg_dmabuf_mr(pd_, (uint64_t)ptr, bytes, (uint64_t)ptr,
                            dmabuf_fd,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ);
    close(dmabuf_fd);
    if (!mr) {
      std::fprintf(stderr, "[rdma] ibv_reg_dmabuf_mr FAILED: %s\n",
                   strerror(errno));
      degraded_ = true;
      return;
    }
    std::fprintf(stderr, "[rdma] DMABUF reg OK lkey=%u\n", mr->lkey);
    mr_map_[id] = mr;
    registered_bufs_[buf_count_] = {id, ptr};
    ++buf_count_;
  };

  reg_buf(1, input_ptr, tiled.input_bytes);
  if (is_degraded()) return;
  reg_buf(2, output_ptr, tiled.output_bytes);
  if (is_degraded()) return;
  reg_buf(3, scratch_ptr, tiled.staging_bytes_required);
  if (is_degraded()) return;

  buf_id_cache_.clear();
  buf_id_cache_.reserve(tiled.ops.size());
  for (auto const& op : tiled.ops) {
    OpBufInfo info = {0, 0};
    if (op.kind == OpKind::Copy) {
      CollectiveBufferRole src_role =
          buf_role(OpKind::Copy, true, op.copy_from_staging);
      CollectiveBufferRole dst_role =
          buf_role(OpKind::Copy, false, op.copy_from_staging);
      info.src_buf_id = static_cast<uint32_t>(src_role) + 1;
      info.dst_buf_id = static_cast<uint32_t>(dst_role) + 1;
    }
    buf_id_cache_.push_back(info);
  }

  validated_ = true;
}

BackendToken RdmaLocalCopyBackend::submit(Op const& op, OpBindings const& bind,
                                          void* input_ptr, void* output_ptr,
                                          void* scratch_ptr) {
  (void)input_ptr;
  (void)output_ptr;
  (void)scratch_ptr;
  if (is_degraded() || !validated_) return BackendToken{0};

  if (op.kind != OpKind::Copy) return BackendToken{0};

  uint32_t op_idx = bind.stream_index;
  OpBufInfo const& info = buf_id_cache_[op_idx];

  auto mr_src = mr_map_.find(info.src_buf_id);
  auto mr_dst = mr_map_.find(info.dst_buf_id);
  if (mr_src == mr_map_.end() || mr_dst == mr_map_.end())
    return BackendToken{0};

  uint32_t idx = req_tail_.fetch_add(1, std::memory_order_relaxed) % kMaxReq;
  auto& r = reqs_[idx];
  if (!r.done.load(std::memory_order_acquire)) {
    req_tail_.fetch_sub(1, std::memory_order_relaxed);
    return BackendToken{0};
  }
  r.done.store(false, std::memory_order_release);
  r.failed = false;

  uint64_t token = next_token_++;
  r.token = token;

  ibv_sge sge = {};
  sge.addr = (uint64_t)bind.resolved_src;
  sge.length = (uint32_t)op.bytes;
  sge.lkey = mr_src->second->lkey;

  ibv_send_wr wr = {};
  ibv_send_wr* bad = nullptr;
  wr.wr_id = r.wr_id;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uint64_t)bind.resolved_dst;
  wr.wr.rdma.rkey = mr_dst->second->rkey;

  if (ibv_post_send(qp_send_, &wr, &bad)) {
    r.done.store(true, std::memory_order_release);
    r.failed = true;
    return BackendToken{r.token, true};
  }

  return BackendToken{token};
}

size_t RdmaLocalCopyBackend::drain(BackendToken* out, size_t max_count) {
  if (is_degraded()) return 0;

  size_t count = 0;
  uint32_t head = req_head_.load(std::memory_order_relaxed);
  uint32_t tail = req_tail_.load(std::memory_order_relaxed);

  while (count < max_count && head != tail) {
    auto& r = reqs_[head % kMaxReq];
    if (r.done.load(std::memory_order_acquire)) {
      out[count].value = r.token;
      out[count].failed = r.failed;
      ++count;
      ++head;
    } else {
      break;
    }
  }
  req_head_.store(head, std::memory_order_relaxed);
  return count;
}

}  // namespace CCL
}  // namespace UKernel

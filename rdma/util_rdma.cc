#include "util_rdma.h"
#include "eqds.h"
#include "transport.h"
#include "transport_config.h"
#include "util/util.h"
#include "util_timer.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sys/mman.h>

namespace uccl {

// RDMAFactory rdma_ctl;
std::shared_ptr<RDMAFactory> rdma_ctl;

static int ibvWidths[] = {1, 4, 8, 12, 2};
static int ibvSpeeds[] = {2500,  /* SDR */
                          5000,  /* DDR */
                          10000, /* QDR */
                          10000, /* QDR */
                          14000, /* FDR */
                          25000, /* EDR */
                          50000, /* HDR */
                          100000 /* NDR */};

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i < max && ((val & (1 << i)) == 0)) i++;
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths) / sizeof(int) - 1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds) / sizeof(int) - 1)];
}

void RDMAFactory::init_dev(int devname_suffix) {
  struct FactoryDevice dev;
  struct ibv_device** device_list;
  struct ibv_context* context;
  struct ibv_device_attr dev_attr;
  struct ibv_port_attr port_attr;
  int i, nb_devices;

  static std::once_flag init_flag;
  std::call_once(init_flag,
                 []() { rdma_ctl = std::make_shared<RDMAFactory>(); });

  // Get Infiniband name from GID index.
  DCHECK(util_rdma_get_ib_name_from_suffix(devname_suffix, dev.ib_name) == 0);

  // Get IP address from Infiniband name.
  if (!SINGLE_CTRL_NIC.empty())
    dev.local_ip_str = get_dev_ip(SINGLE_CTRL_NIC.c_str());
  else
    DCHECK(util_rdma_get_ip_from_ib_name(dev.ib_name, &dev.local_ip_str) == 0);

  // Get the list of RDMA devices.
  device_list = ibv_get_device_list(&nb_devices);
  if (device_list == nullptr || nb_devices == 0) {
    perror("ibv_get_device_list");
    goto error;
  }

  // Find the device by name.
  for (i = 0; i < nb_devices; i++) {
    if (strcmp(ibv_get_device_name(device_list[i]), dev.ib_name) == 0) {
      break;
    }
  }
  if (i == nb_devices) {
    fprintf(stderr, "No device found for %s\n", dev.ib_name);
    goto free_devices;
  }

  // Open the device.
  memset(&dev_attr, 0, sizeof(dev_attr));
  if ((context = ibv_open_device(device_list[i])) == nullptr) {
    perror("ibv_open_device");
    goto free_devices;
  }

  if (ibv_query_device(context, &dev_attr)) {
    perror("ibv_query_device");
    goto close_device;
  }

  // Currently, we only use one port.
  if (dev_attr.phys_port_cnt != IB_PORT_NUM /* 1 */) {
    fprintf(stderr, "Only one port is supported\n");
    goto close_device;
  }

  // Port number starts from 1.
  if (ibv_query_port(context, 1, &port_attr)) {
    perror("ibv_query_port");
    goto close_device;
  }

  if (port_attr.state != IBV_PORT_ACTIVE) {
    fprintf(stderr, "Port is not active\n");
    goto close_device;
  }

  if (ROCE_NET && port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
    fprintf(stderr, "RoCE is not supported\n");
    goto close_device;
  } else if (!ROCE_NET && port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND) {
    fprintf(stderr, "IB is not supported\n");
    goto close_device;
  }

  dev.dev_attr = dev_attr;
  dev.port_attr = port_attr;
  dev.ib_port_num = IB_PORT_NUM;
  dev.gid_idx = GID_IDX;
  dev.context = context;

  if (ibv_query_gid(context, IB_PORT_NUM, dev.gid_idx, &dev.gid)) {
    perror("ibv_query_gid");
    goto close_device;
  }

  // Allocate a PD for this device.
  dev.pd = ibv_alloc_pd(context);
  if (dev.pd == nullptr) {
    perror("ibv_alloc_pd");
    goto close_device;
  }

  // Detect DMA-BUF support.
  {
    struct ibv_pd* pd;
    pd = ibv_alloc_pd(context);
    if (pd == nullptr) {
      perror("ibv_alloc_pd");
      goto close_device;
    }
    // Test kernel DMA-BUF support with a dummy call (fd=-1)
    (void)ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/,
                            -1 /*fd*/, 0 /*flags*/);
    dev.dma_buf_support =
        !((errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT));
    ibv_dealloc_pd(pd);

    UCCL_LOG_RE << "DMA-BUF support: " << dev.dma_buf_support;
  }

  rdma_ctl->devices_.push_back(dev);

  return;

close_device:
  ibv_close_device(context);

free_devices:
  ibv_free_device_list(device_list);
error:
  throw std::runtime_error("Failed to initialize RDMAFactory");
}

/**
 * @brief Create a new RDMA context for a given device running on a specific
 * engine.
 *
 * @param dev
 * @param meta
 * @return RDMAContext*
 */
RDMAContext* RDMAFactory::CreateContext(TimerManager* rto,
                                        uint32_t* engine_unacked_bytes,
                                        eqds::EQDS* eqds, int dev,
                                        uint32_t engine_offset,
                                        union CtrlMeta meta,
                                        SharedIOContext* io_ctx) {
  RDMAContext* ctx = nullptr;

  if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS)
    ctx = new EQDSRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                              engine_offset, meta, io_ctx);
  else if constexpr (kSenderCCA == SENDER_CCA_TIMELY)
    ctx = new TimelyRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                                engine_offset, meta, io_ctx);
  else if constexpr (kSenderCCA == SENDER_CCA_SWIFT)
    ctx = new SwiftRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                               engine_offset, meta, io_ctx);

  CHECK(ctx != nullptr);
  return ctx;
}

std::pair<uint64_t, uint32_t> TXTracking::ack_rc_transmitted_chunks(
    void* subflow_context, RDMAContext* rdma_ctx, UINT_CSN csn, uint64_t now,
    uint32_t* flow_unacked_bytes, uint32_t* engine_outstanding_bytes) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(subflow_context);
  uint64_t tx_timestamp;
  uint32_t qpidx;

  uint32_t acked_bytes = 0;

  // Traverse unacked_chunks_
  // TODO: we can do more efficiently here.
  for (auto chunk = unacked_chunks_.begin(); chunk != unacked_chunks_.end();
       chunk++) {
    if (chunk->csn == csn.to_uint32()) {
      // We find it!
      chunk->ureq->send.acked_bytes += chunk->wr_ex->sge.length;

      acked_bytes += chunk->wr_ex->sge.length;

      if (chunk->ureq->send.acked_bytes == chunk->ureq->send.data_len) {
        auto poll_ctx = chunk->ureq->poll_ctx;
        // Wakeup app thread waiting one endpoint
        uccl_wakeup(poll_ctx);
        UCCL_LOG_IO << "RC TX message complete";
      }

      *flow_unacked_bytes -= chunk->wr_ex->sge.length;
      *engine_outstanding_bytes -= chunk->wr_ex->sge.length;

      tx_timestamp = chunk->timestamp;
      qpidx = chunk->wr_ex->qpidx;

      // Free wr_ex here.
      rdma_ctx->wr_ex_pool_->free_buff(
          reinterpret_cast<uint64_t>(chunk->wr_ex));

      unacked_chunks_.erase(chunk);
      break;
    }
  }

  auto newrtt_tsc = now - tx_timestamp;

  subflow->pcb.timely_cc.update_rate(now, newrtt_tsc, kEwmaAlpha);

  subflow->pcb.swift_cc.adjust_wnd(to_usec(newrtt_tsc, freq_ghz), acked_bytes);

  return std::make_pair(tx_timestamp, qpidx);
}

uint64_t TXTracking::ack_transmitted_chunks(void* subflow_context,
                                            RDMAContext* rdma_ctx,
                                            uint32_t num_acked_chunks,
                                            uint64_t t5, uint64_t t6,
                                            uint64_t remote_queueing_tsc,
                                            uint32_t* flow_unacked_bytes) {
  DCHECK(num_acked_chunks <= unacked_chunks_.size());

  auto* subflow = reinterpret_cast<SubUcclFlow*>(subflow_context);

  uint64_t t1 = 0;
  uint32_t seg_size = 0;

  while (num_acked_chunks) {
    auto& chunk = unacked_chunks_.front();
    if (chunk.last_chunk) {
      auto poll_ctx = chunk.ureq->poll_ctx;
      // Wakeup app thread waiting one endpoint
      uccl_wakeup(poll_ctx);
      UCCL_LOG_IO << "UC Tx message complete";
    }

    // Record timestamp of the oldest unacked chunk.
    if (t1 == 0) t1 = chunk.timestamp;

    seg_size += chunk.wr_ex->sge.length;

    *flow_unacked_bytes -= chunk.wr_ex->sge.length;

    // Free wr_ex here.
    rdma_ctx->wr_ex_pool_->free_buff(reinterpret_cast<uint64_t>(chunk.wr_ex));

    unacked_chunks_.erase(unacked_chunks_.begin());
    num_acked_chunks--;
  }

  if (unlikely(t5 <= t1)) {
    // Invalid timestamp.
    // We have found that t5 (transferred from NIC timestamp) may be
    // occasionally smaller than t1 (timestamp of the oldest unacked chunk).
    // When this happens, we use software timestamp to fix it.
    t5 = rdtsc();
  }

  auto endpoint_delay_tsc = t6 - t5 + remote_queueing_tsc;
  auto fabric_delay_tsc = (t6 - t1) - endpoint_delay_tsc;
  // Make RTT independent of segment size.
  auto serial_delay_tsc =
      us_to_cycles(seg_size * 1e6 / LINK_BANDWIDTH, freq_ghz);
  if (fabric_delay_tsc > serial_delay_tsc ||
      to_usec(fabric_delay_tsc, freq_ghz) < kMAXRTTUS)
    fabric_delay_tsc -= serial_delay_tsc;
  else {
    // Invalid timestamp.
    // Recalculate delay.
    t5 = rdtsc();
    endpoint_delay_tsc = t6 - t5 + remote_queueing_tsc;
    fabric_delay_tsc = (t6 - t1) - endpoint_delay_tsc;
    if (fabric_delay_tsc > serial_delay_tsc)
      fabric_delay_tsc -= serial_delay_tsc;
    else {
      // This may be caused by clock synchronization.
      fabric_delay_tsc = 0;
    }
  }

  UCCL_LOG_IO << "Total: " << to_usec(t6 - t1, freq_ghz)
              << ", Endpoint delay: " << to_usec(endpoint_delay_tsc, freq_ghz)
              << ", Fabric delay: " << to_usec(fabric_delay_tsc, freq_ghz);

  // LOG_EVERY_N(INFO, 10000) << "Host: " <<
  // std::round(to_usec(endpoint_delay_tsc, freq_ghz)) <<
  //     ", Fabric: " << std::round(to_usec(fabric_delay_tsc, freq_ghz));

#ifdef TEST_TURNAROUND_ESTIMATION
  static bool first = true;
  static double avg_turnaround_delay = 0.0;
  static int count = 0;
  auto turnaround_delay = to_usec(remote_queueing_tsc, freq_ghz);

  if (turnaround_delay <
          500 /* filter wrong values (probabaly due to clock sync) */
      && count++ > 5000 /* warmup */) {
    if (first) {
      avg_turnaround_delay = turnaround_delay;
      first = false;
    } else {
      avg_turnaround_delay =
          (avg_turnaround_delay * count + turnaround_delay) / (count + 1);
    }
    LOG_EVERY_N(INFO, 1000)
        << "Turnaround delay: " << turnaround_delay
        << "us, Average turnaround delay: " << avg_turnaround_delay << "us";
  }
#endif

  if (fabric_delay_tsc) {
    // Update global cwnd.
    subflow->pcb.timely_cc.update_rate(t6, fabric_delay_tsc, kEwmaAlpha);
    // TODO: seperate enpoint delay and fabric delay.
    subflow->pcb.swift_cc.adjust_wnd(to_usec(fabric_delay_tsc, freq_ghz),
                                     seg_size);
  }

  return fabric_delay_tsc;
}

void SharedIOContext::check_ctrl_rq(bool force) {
  auto n_post_ctrl_rq = get_post_ctrl_rq_cnt();
  if (!force && n_post_ctrl_rq < kPostRQThreshold) return;

  int post_batch = std::min(kPostRQThreshold, (uint32_t)n_post_ctrl_rq);

  for (int i = 0; i < post_batch; i++) {
    auto chunk_addr = pop_ctrl_chunk();
    ctrl_recv_wrs_.recv_sges[i].addr = chunk_addr;

    CQEDesc* cqe_desc = pop_cqe_desc();
    cqe_desc->data = (uint64_t)chunk_addr;
    ctrl_recv_wrs_.recv_wrs[i].wr_id = (uint64_t)cqe_desc;
    ctrl_recv_wrs_.recv_wrs[i].next =
        (i == post_batch - 1) ? nullptr : &ctrl_recv_wrs_.recv_wrs[i + 1];
  }

  struct ibv_recv_wr* bad_wr;
  DCHECK(ibv_post_recv(ctrl_qp_, &ctrl_recv_wrs_.recv_wrs[0], &bad_wr) == 0);
  UCCL_LOG_IO << "Posted " << post_batch << " recv requests for Ctrl QP";
  dec_post_ctrl_rq(post_batch);
}

void SharedIOContext::check_srq(bool force) {
  auto n_post_srq = get_post_srq_cnt();
  if (!force && n_post_srq < kPostRQThreshold) return;

  int post_batch = std::min(kPostRQThreshold, (uint32_t)n_post_srq);

  for (int i = 0; i < post_batch; i++) {
    if constexpr (!kRCMode) {
      auto chunk_addr = pop_retr_chunk();
      dp_recv_wrs_.recv_sges[i].addr = chunk_addr;
      dp_recv_wrs_.recv_sges[i].length = RetrChunkBuffPool::kRetrChunkSize;
      dp_recv_wrs_.recv_sges[i].lkey = get_retr_chunk_lkey();
      dp_recv_wrs_.recv_wrs[i].num_sge = 1;
      dp_recv_wrs_.recv_wrs[i].sg_list = &dp_recv_wrs_.recv_sges[i];
      dp_recv_wrs_.recv_wrs[i].next =
          (i == post_batch - 1) ? nullptr : &dp_recv_wrs_.recv_wrs[i + 1];

      CQEDesc* cqe_desc = pop_cqe_desc();
      cqe_desc->data = (uint64_t)chunk_addr;
      dp_recv_wrs_.recv_wrs[i].wr_id = (uint64_t)cqe_desc;
    } else {
      dp_recv_wrs_.recv_wrs[i].num_sge = 0;
      dp_recv_wrs_.recv_wrs[i].sg_list = nullptr;
      dp_recv_wrs_.recv_wrs[i].next =
          (i == post_batch - 1) ? nullptr : &dp_recv_wrs_.recv_wrs[i + 1];
      dp_recv_wrs_.recv_wrs[i].wr_id = 0;
    }
  }

  struct ibv_recv_wr* bad_wr;
  DCHECK(ibv_post_srq_recv(srq_, &dp_recv_wrs_.recv_wrs[0], &bad_wr) == 0);
  UCCL_LOG_IO << "Posted " << post_batch << " recv requests for SRQ";
  dec_post_srq(post_batch);
}

int SharedIOContext::poll_ctrl_cq(void) {
  auto cq_ex = ctrl_cq_ex_;
  int work = 0;

  int budget = kMaxBatchCQ << 1;

  while (1) {
    struct ibv_poll_cq_attr poll_cq_attr = {};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return work;
    int cq_budget = 0;

    while (1) {
      if (cq_ex->status != IBV_WC_SUCCESS) {
        DCHECK(false) << "Ctrl CQ state error: " << cq_ex->status << ", "
                      << ibv_wc_read_opcode(cq_ex)
                      << ", ctrl_chunk_pool_size: " << ctrl_chunk_pool_->size();
      }

      CQEDesc* cqe_desc = reinterpret_cast<CQEDesc*>(cq_ex->wr_id);
      auto chunk_addr = (uint64_t)cqe_desc->data;

      auto opcode = ibv_wc_read_opcode(cq_ex);
      if (opcode == IBV_WC_RECV) {
        auto imm_data = ntohl(ibv_wc_read_imm_data(cq_ex));
        auto num_ack = imm_data;
        UCCL_LOG_IO << "Receive " << num_ack
                    << " ACKs, Chunk addr: " << chunk_addr
                    << ", byte_len: " << ibv_wc_read_byte_len(cq_ex);
        auto base_addr = chunk_addr + UD_ADDITION;
        for (int i = 0; i < num_ack; i++) {
          auto pkt_addr = base_addr + i * CtrlChunkBuffPool::kPktSize;

          auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr);
          auto fid = ucclsackh->fid.value();
          auto peer_id = ucclsackh->peer_id.value();
          auto* rdma_ctx = find_rdma_ctx(peer_id, fid);

          rdma_ctx->uc_rx_ack(cq_ex, ucclsackh);
        }
        inc_post_ctrl_rq();
      } else {
        inflight_ctrl_wrs_--;
      }

      push_ctrl_chunk(chunk_addr);

      push_cqe_desc(cqe_desc);

      if (++cq_budget == budget || ibv_next_poll(cq_ex)) break;

      if (opcode == IBV_WC_SEND) {
        // We don't count send WRs in budget.
        cq_budget--;
      }
    }
    ibv_end_poll(cq_ex);

    work += cq_budget;

    check_ctrl_rq(false);

    if (cq_budget < budget) break;
  }

  return work;
}

void SharedIOContext::flush_acks() {
  if (nr_tx_ack_wr_ == 0) return;

  tx_ack_wr_[nr_tx_ack_wr_ - 1].next = nullptr;

  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(ctrl_qp_, tx_ack_wr_, &bad_wr);
  DCHECK(ret == 0) << ret << ", nr_tx_ack_wr_: " << nr_tx_ack_wr_;

  UCCL_LOG_IO << "Flush " << nr_tx_ack_wr_ << " ACKs";

  inflight_ctrl_wrs_ += nr_tx_ack_wr_;

  nr_tx_ack_wr_ = 0;
}

int SharedIOContext::rc_poll_recv_cq(void) {
  auto cq_ex = recv_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      DCHECK(false) << "data path CQ state error: " << cq_ex->status
                    << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    rdma_ctx->rc_rx_chunk(cq_ex);

    inc_post_srq();

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::rc_poll_send_cq(void) {
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      DCHECK(false) << "data path CQ state error: " << cq_ex->status
                    << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    rdma_ctx->rc_rx_ack(cq_ex);

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::uc_poll_send_cq(void) {
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;
  int budget = kMaxBatchCQ << 1;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      DCHECK(false) << "data path CQ state error: " << cq_ex->status
                    << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* cqe_desc = (CQEDesc*)cq_ex->wr_id;

    if (cqe_desc) {
      // Completion signal from rtx.
      auto retr_hdr = (uint64_t)cqe_desc->data;
      push_retr_hdr(retr_hdr);
      push_cqe_desc(cqe_desc);
    }

    if (++cq_budget == budget || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::uc_poll_recv_cq(void) {
  auto cq_ex = recv_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  std::vector<RDMAContext*> rdma_ctxs;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      DCHECK(false) << "data path CQ state error: " << cq_ex->status
                    << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    auto* cqe_desc = (CQEDesc*)cq_ex->wr_id;
    auto chunk_addr = (uint64_t)cqe_desc->data;
    auto opcode = ibv_wc_read_opcode(cq_ex);

    if (likely(opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
      // Common case.
      rdma_ctx->uc_rx_chunk(cq_ex);
    } else {
      // Rare case.
      rdma_ctx->uc_rx_rtx_chunk(cq_ex, chunk_addr);
    }

    rdma_ctxs.push_back(rdma_ctx);

    push_retr_chunk(chunk_addr);

    push_cqe_desc(cqe_desc);

    inc_post_srq();

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  for (auto rdma_ctx : rdma_ctxs) {
    rdma_ctx->uc_post_acks();
  }

  flush_acks();

  return cq_budget;
}

}  // namespace uccl

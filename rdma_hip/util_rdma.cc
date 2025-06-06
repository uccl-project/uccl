#include "util_rdma.h"
#include "eqds.h"
#include "transport.h"
#include "transport_config.h"
#include "util.h"
#include "util_timer.h"
#include <glog/logging.h>
#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sys/mman.h>

namespace uccl {

// RDMAFactory rdma_ctl;
std::shared_ptr<RDMAFactory> rdma_ctl;

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
RDMAContext* RDMAFactory::CreateContext(PeerID peer_id, TimerManager* rto,
                                        uint32_t* engine_unacked_bytes,
                                        eqds::EQDS* eqds, int dev,
                                        uint32_t engine_offset,
                                        union CtrlMeta meta) {
  RDMAContext* ctx = nullptr;

  if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS)
    ctx = new EQDSRDMAContext(peer_id, rto, engine_unacked_bytes, eqds, dev,
                              engine_offset, meta);
  else if constexpr (kSenderCCA == SENDER_CCA_TIMELY)
    ctx = new TimelyRDMAContext(peer_id, rto, engine_unacked_bytes, eqds, dev,
                                engine_offset, meta);
  else if constexpr (kSenderCCA == SENDER_CCA_SWIFT)
    ctx = new SwiftRDMAContext(peer_id, rto, engine_unacked_bytes, eqds, dev,
                               engine_offset, meta);

  CHECK(ctx != nullptr);
  return ctx;
}

RDMAContext::RDMAContext(PeerID peer_id, TimerManager* rto,
                         uint32_t* engine_unacked_bytes, eqds::EQDS* eqds,
                         int dev, uint32_t engine_offset, union CtrlMeta meta)
    : peer_id_(peer_id),
      rto_(rto),
      engine_unacked_bytes_(engine_unacked_bytes),
      eqds_(eqds),
      dev_(dev),
      engine_offset_(engine_offset),
      wheel_({freq_ghz, us_to_cycles(kWheelSlotWidthUs, freq_ghz),
              us_to_cycles(kWheelHorizonUs, freq_ghz), kBktPoolSize}) {
  auto* factory_dev = RDMAFactory::get_factory_dev(dev);

  context_ = factory_dev->context;

  remote_ctx_.remote_gid = meta.install_ctx.remote_gid;
  remote_ctx_.remote_port_attr = meta.install_ctx.remote_port_attr;

  mtu_bytes_ =
      util_rdma_get_mtu_from_ibv_mtu(factory_dev->port_attr.active_mtu);

  // Create PD.
  pd_ = factory_dev->pd;

  // Create a SRQ for all data path QPs.
  struct ibv_srq_init_attr srq_init_attr;
  memset(&srq_init_attr, 0, sizeof(srq_init_attr));
  srq_init_attr.attr.max_sge = 1;
  srq_init_attr.attr.max_wr = kMaxSRQ;
  srq_init_attr.attr.srq_limit = 0;
  srq_ = ibv_create_srq(pd_, &srq_init_attr);
  UCCL_INIT_CHECK(srq_ != nullptr, "ibv_create_srq failed");

  // Create seperate send/recv CQ for data path QPs.
  struct ibv_cq_init_attr_ex cq_ex_attr;
  cq_ex_attr.cqe = kCQSize;
  cq_ex_attr.cq_context = nullptr;
  cq_ex_attr.channel = nullptr;
  cq_ex_attr.comp_vector = 0;
  cq_ex_attr.wc_flags =
      IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM | IBV_WC_EX_WITH_QP_NUM |
      IBV_WC_EX_WITH_SRC_QP |
      IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;  // Timestamp support.
  cq_ex_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
  cq_ex_attr.flags =
      IBV_CREATE_CQ_ATTR_SINGLE_THREADED | IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;

  if constexpr (kTestNoHWTimestamp)
    cq_ex_attr.wc_flags &= ~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;

  send_cq_ex_ = ibv_create_cq_ex(context_, &cq_ex_attr);
  UCCL_INIT_CHECK(send_cq_ex_ != nullptr, "ibv_create_cq_ex failed");

  recv_cq_ex_ = ibv_create_cq_ex(context_, &cq_ex_attr);
  UCCL_INIT_CHECK(recv_cq_ex_ != nullptr, "ibv_create_cq_ex failed");

  // Configure CQ moderation.
  struct ibv_modify_cq_attr cq_attr;
  cq_attr.attr_mask = IBV_CQ_ATTR_MODERATE;
  cq_attr.moderate.cq_count = kCQMODCount;
  cq_attr.moderate.cq_period = kCQMODPeriod;

  UCCL_INIT_CHECK(ibv_modify_cq(ibv_cq_ex_to_cq(send_cq_ex_), &cq_attr) == 0,
                  "ibv_modify_cq failed");
  UCCL_INIT_CHECK(ibv_modify_cq(ibv_cq_ex_to_cq(recv_cq_ex_), &cq_attr) == 0,
                  "ibv_modify_cq failed");

  // Create data path QPs. (UC/RC)
  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.qp_context = this;
  qp_init_attr.send_cq = ibv_cq_ex_to_cq(send_cq_ex_);
  qp_init_attr.recv_cq = ibv_cq_ex_to_cq(recv_cq_ex_);
  if constexpr (!kRCMode)
    qp_init_attr.qp_type = IBV_QPT_UC;
  else
    qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = 2 * kMaxReq * kMaxRecv;
  qp_init_attr.cap.max_send_sge = kMaxSge;
  qp_init_attr.cap.max_inline_data = 0;
  qp_init_attr.srq = srq_;

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = IB_PORT_NUM;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;

  for (int i = 0; i < kPortEntropy; i++) {
    struct ibv_qp* qp = ibv_create_qp(pd_, &qp_init_attr);
    UCCL_INIT_CHECK(qp != nullptr, "ibv_create_qp failed for data path QP");

    // Modify QP state to INIT.
    UCCL_INIT_CHECK(ibv_modify_qp(qp, &qpAttr,
                                  IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                      IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                    "ibv_modify_qp failed");

    dp_qps_[i].local_psn = BASE_PSN;
    dp_qps_[i].qp = qp;
    qpn2idx_.insert({qp->qp_num, i});
  }

  // Initialize work request extension buffer pool.
  wr_ex_pool_.emplace();

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));

  // Create Credit QP, SCQ/RCQ and MR for engine or pacer.
  credit_local_psn_ = BASE_PSN;
  util_rdma_create_qp_seperate_cq(context_, &credit_qp_, IBV_QPT_UC, true,
                                  false, (struct ibv_cq**)&pacer_credit_cq_ex_,
                                  (struct ibv_cq**)&engine_credit_cq_ex_, false,
                                  kCQSize, pd_,
                                  eqds::CreditChunkBuffPool::kNumChunk,
                                  eqds::CreditChunkBuffPool::kNumChunk, 1, 1);

  auto addr = mmap(nullptr, eqds::CreditChunkBuffPool::kCreditMRSize,
                   PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  UCCL_INIT_CHECK(addr != MAP_FAILED, "mmap failed");
  engine_credit_mr_ =
      ibv_reg_mr(pd_, addr, eqds::CreditChunkBuffPool::kCreditMRSize,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  UCCL_INIT_CHECK(engine_credit_mr_ != nullptr,
                  "ibv_reg_mr failed for engine credit MR");

  addr = mmap(nullptr, eqds::CreditChunkBuffPool::kCreditMRSize,
              PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  UCCL_INIT_CHECK(addr != MAP_FAILED, "mmap failed");
  pacer_credit_mr_ =
      ibv_reg_mr(pd_, addr, eqds::CreditChunkBuffPool::kCreditMRSize,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  UCCL_INIT_CHECK(pacer_credit_mr_ != nullptr,
                  "ibv_reg_mr failed for pacer credit MR");

  // Initialize Credit packet buffer pool for engine and pacer, respectively.
  engine_credit_chunk_pool_.emplace(engine_credit_mr_);
  pacer_credit_chunk_pool_.emplace(pacer_credit_mr_);

  pc_qpw_.credit_qp_ = credit_qp_;
  pc_qpw_.pacer_credit_cq_ = pacer_credit_cq_ex_;
  pc_qpw_.pacer_credit_chunk_pool_ = &(*pacer_credit_chunk_pool_);

  INIT_LIST_HEAD(&pc_qpw_.poll_item.poll_link);
  pc_qpw_.poll_item.pc_qpw = &pc_qpw_;

  // Populate recv work requests on Credit QP for consuming credit packets.
  {
    struct ibv_sge sge;
    for (int i = 0; i < (eqds::CreditChunkBuffPool::kNumChunk - 1) / 2; i++) {
      uint64_t chunk_addr;
      if (engine_credit_chunk_pool_->alloc_buff(&chunk_addr))
        throw std::runtime_error("Failed to allocate buffer for credit packet");
      sge.addr = chunk_addr;
      sge.length = eqds::CreditChunkBuffPool::kChunkSize;
      sge.lkey = engine_credit_chunk_pool_->get_lkey();
      wr.wr_id = chunk_addr;
      wr.next = nullptr;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      struct ibv_recv_wr* bad_wr;
      if (ibv_post_recv(credit_qp_, &wr, &bad_wr))
        throw std::runtime_error("ibv_post_recv failed");
    }
  }

  // Initialize resources needed when using UC.
  if constexpr (!kRCMode) {
    // Create Ctrl QP, CQ, and MR.
    ctrl_local_psn_ = BASE_PSN;
    util_rdma_create_qp(context_, &ctrl_qp_, IBV_QPT_UC, true, true,
                        (struct ibv_cq**)&ctrl_cq_ex_, false, kCQSize, pd_,
                        &ctrl_mr_, nullptr, kCtrlMRSize,
                        CtrlChunkBuffPool::kNumChunk,
                        CtrlChunkBuffPool::kNumChunk, 1, 1);

    // Initialize Control packet buffer pool.
    ctrl_chunk_pool_.emplace(ctrl_mr_);

    // Create Retr hdr/chunk MR.
    void* retr_mr_addr = mmap(nullptr, kRetrMRSize, PROT_READ | PROT_WRITE,
                              MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (retr_mr_addr == MAP_FAILED) throw std::runtime_error("mmap failed");
    retr_mr_ = ibv_reg_mr(pd_, retr_mr_addr, kRetrMRSize,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (retr_mr_ == nullptr) throw std::runtime_error("ibv_reg_mr failed");

    void* retr_hdr =
        mmap(nullptr, RetrHdrBuffPool::kNumHdr * RetrHdrBuffPool::kHdrSize,
             PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (retr_hdr == MAP_FAILED) throw std::runtime_error("mmap failed");
    retr_hdr_mr_ = ibv_reg_mr(
        pd_, retr_hdr, RetrHdrBuffPool::kNumHdr * RetrHdrBuffPool::kHdrSize,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (retr_hdr_mr_ == nullptr) throw std::runtime_error("ibv_reg_mr failed");

    // Initialize retransmission chunk and header buffer pool.
    {
      retr_chunk_pool_.emplace(retr_mr_);
      retr_hdr_pool_.emplace(retr_hdr_mr_);
    }

    // Populate recv work requests on Ctrl QP for consuming control packets.
    {
      struct ibv_sge sge;
      for (int i = 0; i < (CtrlChunkBuffPool::kNumChunk - 1) / 2; i++) {
        uint64_t chunk_addr;
        if (ctrl_chunk_pool_->alloc_buff(&chunk_addr))
          throw std::runtime_error(
              "Failed to allocate buffer for control packet");
        sge.addr = chunk_addr;
        sge.length = CtrlChunkBuffPool::kChunkSize;
        sge.lkey = ctrl_chunk_pool_->get_lkey();
        wr.wr_id = chunk_addr;
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        struct ibv_recv_wr* bad_wr;
        if (ibv_post_recv(ctrl_qp_, &wr, &bad_wr))
          throw std::runtime_error("ibv_post_recv failed");
      }
    }

    for (int i = 0; i < kMaxBatchCQ; i++) {
      retr_wrs_[i].num_sge = 1;
      retr_wrs_[i].sg_list = nullptr;
      retr_wrs_[i].next = (i == kMaxBatchCQ - 1) ? nullptr : &retr_wrs_[i + 1];
    }

    for (int i = 0; i < kPostRQThreshold; i++) {
      rx_ack_sges_[i].lkey = ctrl_chunk_pool_->get_lkey();
      rx_ack_sges_[i].length = CtrlChunkBuffPool::kChunkSize;
      rx_ack_wrs_[i].sg_list = &rx_ack_sges_[i];
      rx_ack_wrs_[i].num_sge = 1;
      rx_ack_wrs_[i].next =
          (i == kPostRQThreshold - 1) ? nullptr : &rx_ack_wrs_[i + 1];
    }

    tx_ack_wr_.num_sge = 1;
    tx_ack_wr_.next = nullptr;
    tx_ack_wr_.opcode = IBV_WR_SEND_WITH_IMM;
    tx_ack_wr_.send_flags = IBV_SEND_SIGNALED;
  }

  for (int i = 0; i < kPostRQThreshold; i++) {
    imm_wrs_[i].num_sge = 0;
    imm_wrs_[i].sg_list = nullptr;
    imm_wrs_[i].next = (i == kPostRQThreshold - 1) ? nullptr : &imm_wrs_[i + 1];

    rx_credit_sges_[i].lkey = engine_credit_chunk_pool_->get_lkey();
    rx_credit_sges_[i].length = eqds::CreditChunkBuffPool::kChunkSize;
    rx_credit_wrs_[i].sg_list = &rx_credit_sges_[i];
    rx_credit_wrs_[i].num_sge = 1;
    rx_credit_wrs_[i].next =
        (i == kPostRQThreshold - 1) ? nullptr : &rx_credit_wrs_[i + 1];
  }

  // Populate recv work requests to SRQ for consuming immediate data.
  for (int i = 0; i < kMaxSRQ; i++) {
    struct ibv_recv_wr* bad_wr;
    struct ibv_sge sge;
    if constexpr (!kRCMode) {
      uint64_t chunk_addr;
      if (retr_chunk_pool_->alloc_buff(&chunk_addr))
        throw std::runtime_error(
            "Failed to allocate buffer for retransmission chunk");
      sge.addr = chunk_addr;
      sge.length = RetrChunkBuffPool::kRetrChunkSize;
      sge.lkey = retr_chunk_pool_->get_lkey();
      wr.wr_id = chunk_addr;
      wr.next = nullptr;
      wr.sg_list = &sge;
      wr.num_sge = 1;
    }
    DCHECK(ibv_post_srq_recv(srq_, &wr, &bad_wr) == 0);
  }

  // Timing wheel.
  wheel_.catchup();
}

RDMAContext::~RDMAContext() {
  if constexpr (!kRCMode) {
    if (ctrl_mr_ != nullptr) {
      munmap(ctrl_mr_->addr, ctrl_mr_->length);
      ibv_dereg_mr(ctrl_mr_);
    }
    if (ibv_cq_ex_to_cq(ctrl_cq_ex_) != nullptr) {
      ibv_destroy_cq(ibv_cq_ex_to_cq(ctrl_cq_ex_));
    }
    if (credit_qp_ != nullptr) {
      ibv_destroy_qp(credit_qp_);
    }
    if (engine_credit_mr_ != nullptr) {
      munmap(engine_credit_mr_->addr, engine_credit_mr_->length);
      ibv_dereg_mr(engine_credit_mr_);
    }
    if (pacer_credit_mr_ != nullptr) {
      munmap(pacer_credit_mr_->addr, pacer_credit_mr_->length);
      ibv_dereg_mr(pacer_credit_mr_);
    }
    if (ctrl_qp_ != nullptr) {
      ibv_destroy_qp(ctrl_qp_);
    }
    if (retr_mr_ != nullptr) {
      munmap(retr_mr_->addr, retr_mr_->length);
      ibv_dereg_mr(retr_mr_);
    }
    if (retr_hdr_mr_ != nullptr) {
      munmap(retr_hdr_mr_->addr, retr_hdr_mr_->length);
      ibv_dereg_mr(retr_hdr_mr_);
    }
  }

  for (int i = 0; i < kPortEntropy; i++) {
    ibv_destroy_qp(dp_qps_[i].qp);
  }
  if (srq_ != nullptr) {
    ibv_destroy_srq(srq_);
  }
  if (ibv_cq_ex_to_cq(send_cq_ex_) != nullptr) {
    ibv_destroy_cq(ibv_cq_ex_to_cq(send_cq_ex_));
  }
  if (ibv_cq_ex_to_cq(recv_cq_ex_) != nullptr) {
    ibv_destroy_cq(ibv_cq_ex_to_cq(recv_cq_ex_));
  }

  if (pd_ != nullptr) {
    ibv_dealloc_pd(pd_);
  }

  UCCL_LOG_RE << "RDMAContext destroyed";
}

int RDMAContext::supply_rx_buff(struct ucclRequest* ureq) {
  auto* elems = ureq->recv.elems;

  auto req = alloc_recvreq();
  if (req == nullptr) return -1;

  for (int i = 0; i < ureq->n; i++) {
    // For sender to encode the request id in the immediate data.
    elems[i].rid = get_recvreq_id(req);
  }

  struct ibv_send_wr* bad_wr;
  DCHECK(ibv_post_send(ureq->recv.qp, &ureq->recv.wr, &bad_wr) == 0);

  req->type = RecvRequest::RECV;
  req->ureq = ureq;
  memset(req->received_bytes, 0, sizeof(uint32_t) * kMaxRecv);
  req->fin_msg = 0;

  return 0;
}

bool RDMAContext::receiverCC_tx_message(struct ucclRequest* ureq) {
  auto* flow = reinterpret_cast<UcclFlow*>(ureq->context);
  auto* subflow = flow->sub_flows_[engine_offset_];
  auto* eqds = &subflow->pcb.eqds_cc;

  auto size = ureq->send.data_len;
  auto laddr = ureq->send.laddr;
  auto raddr = ureq->send.raddr;
  auto lkey = ureq->send.lkey;
  auto rkey = ureq->send.rkey;
  auto* sent_offset = &ureq->send.sent_offset;
  uint64_t wr_addr;
  bool queued = false;
  uint32_t chunk_size;

  auto now = rdtsc();

  if (ureq->send.inc_backlog == 0) {
    ureq->send.inc_backlog = 1;
    subflow->backlog_bytes_ += size;
  }

  if (subflow->in_rtx) {
    // We have to wait for the retransmission to finish.
    // Drain the retransmission queue.
    drain_rtx_queue(subflow);
    return false;
  }

  while (*sent_offset < size || size == 0 /* zero-length message */) {
    chunk_size = EventOnChunkSize(subflow, size - *sent_offset);

    if (chunk_size == 0 && size) return false;

    subflow->backlog_bytes_ -= chunk_size;

    auto pull_target = eqds->compute_pull_target(subflow, chunk_size);

    DCHECK(wr_ex_pool_->alloc_buff(&wr_addr) == 0);
    struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(wr_addr);
    auto wr = &wr_ex->wr;

    wr_ex->sge.addr = laddr + *sent_offset;
    wr_ex->sge.lkey = lkey;
    wr_ex->sge.length = chunk_size;

    wr->wr.rdma.remote_addr = raddr + *sent_offset;
    wr->wr.rdma.rkey = rkey;

    IMMDataEQDS imm_data(0);

    imm_data.SetTarget(pull_target);

    imm_data.SetFID(flow->flowid());
    if ((*sent_offset + chunk_size == size)) {
      // Last chunk of the message.
      imm_data.SetHINT(1);
    }
    imm_data.SetRID(ureq->send.rid);

    imm_data.SetCSN(subflow->pcb.get_snd_nxt().to_uint32());

    wr->imm_data = htonl(imm_data.GetImmData());

    // We use high 8 bits of wr_id to store CSN.
    // Lower 56 bits to store subflow pointer.
    if constexpr (kRCMode)
      wr->wr_id = (1ULL * imm_data.GetCSN()) << 56 | (uint64_t)subflow;
    else
      wr->wr_id = 0;

    // Select QP.
    auto qpidx = EventOnSelectPath(subflow, chunk_size);
    auto qpw = &dp_qps_[qpidx];

    wr->send_flags = 0;
    if (qpw->signal_cnt_++ % kSignalInterval == 0) {
      wr->send_flags = IBV_SEND_SIGNALED;
      pending_signal_poll_++;
    }
    wr_ex->qpidx = qpidx;

    struct ibv_send_wr* bad_wr;
    DCHECK(ibv_post_send(qpw->qp, wr, &bad_wr) == 0);

    // Track this chunk.
    subflow->txtracking.track_chunk(ureq, wr_ex, now, imm_data.GetCSN(),
                                    imm_data.GetHINT());
    if constexpr (!kRCMode) {
      // Arm timer for TX
      arm_timer_for_flow(subflow);
    }

    *sent_offset += chunk_size;

    UCCL_LOG_IO << "Tx: flow#" << flow->flowid() << ", req id#"
                << ureq->send.rid << ", msg id#" << ureq->mid
                << ", csn:" << imm_data.GetCSN()
                << ", remaining bytes:" << size - *sent_offset
                << ", pull target:" << (uint32_t)pull_target << " with QP#"
                << qpidx;

    subflow->unacked_bytes_ += chunk_size;
    *engine_unacked_bytes_ += chunk_size;

    /* zero-length message */
    if (size == 0) break;
  }

  return true;
}

bool RDMAContext::senderCC_tx_message(struct ucclRequest* ureq) {
  auto* flow = reinterpret_cast<UcclFlow*>(ureq->context);
  DCHECK(flow);
  auto* subflow = flow->sub_flows_[engine_offset_];

  auto size = ureq->send.data_len;
  auto laddr = ureq->send.laddr;
  auto raddr = ureq->send.raddr;
  auto lkey = ureq->send.lkey;
  auto rkey = ureq->send.rkey;
  auto* sent_offset = &ureq->send.sent_offset;
  uint64_t wr_addr;
  bool queued = false;
  uint32_t chunk_size;
  uint32_t qpidx;

  auto now = rdtsc();

  while (*sent_offset < size || size == 0 /* zero-length message */) {
    chunk_size = EventOnChunkSize(subflow, size - *sent_offset);

    if (chunk_size == 0 && size) return false;

    if constexpr (kBypassPacing) {
      DCHECK(wr_ex_pool_->alloc_buff(&wr_addr) == 0);
      struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(wr_addr);
      auto wr = &wr_ex->wr;

      wr_ex->sge.addr = laddr + *sent_offset;
      wr_ex->sge.lkey = lkey;
      wr_ex->sge.length = chunk_size;

      wr->wr.rdma.remote_addr = raddr + *sent_offset;
      wr->wr.rdma.rkey = rkey;

      IMMData imm_data(0);

      imm_data.SetFID(flow->flowid());
      if ((*sent_offset + chunk_size == size)) {
        // Last chunk of the message.
        imm_data.SetHINT(1);
      }
      imm_data.SetRID(ureq->send.rid);

      imm_data.SetCSN(subflow->pcb.get_snd_nxt().to_uint32());

      wr->imm_data = htonl(imm_data.GetImmData());

      // We use high 8 bits of wr_id to store CSN.
      // Lower 56 bits to store subflow pointer.
      if constexpr (kRCMode)
        wr->wr_id = (1ULL * imm_data.GetCSN()) << 56 | (uint64_t)subflow;
      else
        wr->wr_id = 0;

      // Select QP.
      qpidx = select_qpidx_pot(chunk_size, subflow);
      auto qpw = &dp_qps_[qpidx];

      wr->send_flags = 0;
      if (qpw->signal_cnt_++ % kSignalInterval == 0) {
        wr->send_flags = IBV_SEND_SIGNALED;
        pending_signal_poll_++;
      }
      wr_ex->qpidx = qpidx;

      struct ibv_send_wr* bad_wr;
      DCHECK(ibv_post_send(qpw->qp, wr, &bad_wr) == 0);

      // Track this chunk.
      subflow->txtracking.track_chunk(ureq, wr_ex, now, imm_data.GetCSN(),
                                      imm_data.GetHINT());
      if constexpr (!kRCMode) {
        // Arm timer for TX
        arm_timer_for_flow(subflow);
      }

      *sent_offset += chunk_size;

      UCCL_LOG_IO << "Tx: flow#" << flow->flowid() << ", req id#"
                  << ureq->send.rid << ", msg id#" << ureq->mid
                  << ", csn:" << imm_data.GetCSN()
                  << ", remaining bytes:" << size - *sent_offset << " with QP#"
                  << qpidx;

      subflow->unacked_bytes_ += chunk_size;
      *engine_unacked_bytes_ += chunk_size;
      /* zero-length message */
      if (size == 0) break;

      continue;
    }

    // Prepare SGE.
    DCHECK(wr_ex_pool_->alloc_buff(&wr_addr) == 0);
    struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(wr_addr);
    auto wr = &wr_ex->wr;
    wr_ex->sge.addr = laddr + *sent_offset;
    wr_ex->sge.lkey = lkey;
    wr_ex->sge.length = chunk_size;

    // wr->sg_list/num_sge/next/opcode are already set.

    wr->wr.rdma.remote_addr = raddr + *sent_offset;
    wr->wr.rdma.rkey = rkey;

    UCCL_LOG_IO << "remote_addr: " << wr->wr.rdma.remote_addr
                << ", rkey: " << wr->wr.rdma.rkey;

    IMMData imm_data(0);

    imm_data.SetFID(flow->flowid());
    if ((*sent_offset + chunk_size == size)) {
      // Last chunk of the message.
      imm_data.SetHINT(1);
    }
    imm_data.SetRID(ureq->send.rid);

    imm_data.SetCSN(subflow->pcb.get_snd_nxt().to_uint32());

    wr->imm_data = htonl(imm_data.GetImmData());

    // We use high 8 bits of wr_id to store CSN.
    // Lower 56 bits to store subflow pointer.
    if constexpr (kRCMode)
      wr->wr_id = (1ULL * imm_data.GetCSN()) << 56 | (uint64_t)subflow;
    else
      wr->wr_id = 0;

    {
      auto wheel = &wheel_;
      uint32_t hdr_overhead;
      if (likely(chunk_size == kChunkSize && mtu_bytes_ == 4096)) {
        hdr_overhead = ROCE_NET ? MAX_CHUNK_IB_4096_HDR_OVERHEAD
                                : MAX_CHUNK_ROCE_IPV4_4096_HDR_OVERHEAD;
      } else {
        auto num_mtu = (chunk_size + mtu_bytes_) / mtu_bytes_;
        hdr_overhead =
            num_mtu * (ROCE_NET ? ROCE_IPV4_HDR_OVERHEAD : IB_HDR_OVERHEAD);
      }

      // Enforce global cwnd.
      queued = EventOnQueueData(subflow, wr_ex, chunk_size + hdr_overhead, now);

      if (queued) {
        // Queue the SGE on the timing wheel.
        subflow->in_wheel_cnt_++;
        // For future tracking.
        wr_ex->ureq = ureq;
        UCCL_LOG_IO << "Queued " << chunk_size
                    << " bytes to timing wheel for flow#" << flow->flowid();
      } else {
        // Transmit this chunk directly.
        // Select QP.
        auto qpidx = select_qpidx_pot(chunk_size, subflow);
        auto qpw = &dp_qps_[qpidx];
        // There is no need to signal every WQE since we don't handle TX
        // completions. But we still need occasionally post a request
        // with the IBV_SEND_SIGNALED flag. See
        // https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/.
        wr_ex->wr.send_flags = 0;
        if (qpw->signal_cnt_++ % kSignalInterval == 0) {
          wr_ex->wr.send_flags = IBV_SEND_SIGNALED;
          pending_signal_poll_++;
        }
        wr_ex->qpidx = qpidx;
        struct ibv_send_wr* bad_wr;
        auto ret = ibv_post_send(qpw->qp, wr, &bad_wr);
        DCHECK(ret == 0) << pending_signal_poll_ << ", " << ret;

        // Track this chunk.
        subflow->txtracking.track_chunk(ureq, wr_ex, now, imm_data.GetCSN(),
                                        imm_data.GetHINT());
        if constexpr (!kRCMode) {
          // Arm timer for TX
          arm_timer_for_flow(subflow);
        }

        UCCL_LOG_IO << "Directly sent " << chunk_size << " bytes to QP#"
                    << qpidx;
      }
    }

    *sent_offset += chunk_size;

    UCCL_LOG_IO << "Tx: flow#" << flow->flowid() << ", req id#"
                << ureq->send.rid << ", msg id#" << ureq->mid
                << ", csn:" << imm_data.GetCSN()
                << ", remaining bytes:" << size - *sent_offset << " with QP#"
                << qpidx;

    subflow->unacked_bytes_ += chunk_size;
    *engine_unacked_bytes_ += chunk_size;
    /* zero-length message */
    if (size == 0) break;
  }

  return true;
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

int RDMAContext::receiver_poll_rc_cq(void) {
  auto cq_ex = recv_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status == IBV_WC_SUCCESS) {
      rc_rx_data();
      post_srq_cnt_++;
    } else {
      LOG(ERROR) << "data path CQ state error: " << cq_ex->status
                 << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return cq_budget;
}

int RDMAContext::receiver_poll_uc_cq(void) {
  auto cq_ex = recv_cq_ex_;
  LIST_HEAD(ack_list);
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status == IBV_WC_SUCCESS) {
      auto opcode = ibv_wc_read_opcode(cq_ex);
      if (likely(opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
        if constexpr (!kTestLoss)
          rx_data(&ack_list);
        else {
          DCHECK(kTestLossRate > 0);
          auto drop_period = (uint32_t)(1 / kTestLossRate);
          static uint32_t drop_cnt = 0;
          if (drop_cnt++ % drop_period == 0) {
            UCCL_LOG_IO << "Drop a chunk";
          } else {
            rx_data(&ack_list);
          }
        }
      } else {
        DCHECK(opcode == IBV_WC_RECV);
        rx_rtx_data(&ack_list);
      }

      auto chunk_addr = (uint64_t)cq_ex->wr_id;
      retr_chunk_pool_->free_buff(chunk_addr);

      post_srq_cnt_++;
    } else {
      LOG(ERROR) << "data path CQ state error: " << cq_ex->status
                 << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  // Send coalescing ACKs.
  int num_ack = 0;
  struct list_head *pos, *n;
  uint64_t chunk_addr;
  DCHECK(ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
  list_for_each_safe(pos, n, &ack_list) {
    auto ack_item = list_entry(pos, struct ack_item, ack_link);
    auto subflow = ack_item->subflow;
    craft_ack(subflow, chunk_addr, num_ack++);
    list_del(pos);
  }
  flush_acks(num_ack, chunk_addr);
  if (num_ack == 0) ctrl_chunk_pool_->free_buff(chunk_addr);

  return cq_budget;
}

void RDMAContext::rc_rx_ack(void) {
  auto cq_ex = send_cq_ex_;
  auto now = rdtsc();

  auto wr_id = cq_ex->wr_id;
  auto csn = (wr_id >> 56) & 0xff;
  auto subflow = reinterpret_cast<SubUcclFlow*>((wr_id & 0xffffffffffffff));

  auto pair = subflow->txtracking.ack_rc_transmitted_chunks(
      subflow, this, csn, now, &subflow->unacked_bytes_, engine_unacked_bytes_);

#ifdef TEST_CC_REACTION
  auto hw_ts = convert_nic_to_host(ibv_wc_read_completion_ts(cq_ex));

  static bool first = true;
  static double avg_react_delay = 0.0;
  static int count = 0;
  auto reaction_delay = to_usec(now - hw_ts, freq_ghz);

  if (reaction_delay <
          500 /* filter wrong values (probabaly due to clock sync) */
      && count++ > 5000 /* warmup */) {
    if (first) {
      avg_react_delay = reaction_delay;
      first = false;
    } else {
      avg_react_delay =
          (avg_react_delay * count + reaction_delay) / (count + 1);
    }
    LOG_EVERY_N(INFO, 1000)
        << "CC decision delay: " << reaction_delay
        << "us, Average CC decision delay: " << avg_react_delay << "us";
  }
#endif

  subflow->update_scoreboard_rtt(pair.first, pair.second);

  UCCL_LOG_IO << "Received ACK for csn: " << csn;
}

int RDMAContext::sender_poll_rc_cq(void) {
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status == IBV_WC_SUCCESS) {
      rc_rx_ack();
    } else {
      LOG(ERROR) << "data path CQ state error: " << cq_ex->status
                 << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return cq_budget;
}

int RDMAContext::sender_poll_uc_cq(void) {
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;

  if (likely(pending_signal_poll_ == 0)) return 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS)
      LOG(ERROR) << "data path CQ state error: " << cq_ex->status
                 << " from QP:" << ibv_wc_read_qp_num(cq_ex);

    auto retr_hdr = (uint64_t)cq_ex->wr_id;
    if (retr_hdr) retr_hdr_pool_->free_buff(retr_hdr);

    DCHECK(pending_signal_poll_ > 0);
    pending_signal_poll_--;

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  return cq_budget;
}

void RDMAContext::check_credit_rq(bool force) {
  // Populate recv work requests for consuming credit packets.
  while (post_credit_rq_cnt_ >= kPostRQThreshold) {
    struct ibv_recv_wr* bad_wr;
    for (int i = 0; i < kPostRQThreshold; i++) {
      uint64_t chunk_addr;
      DCHECK(engine_credit_chunk_pool_->alloc_buff(&chunk_addr) == 0);
      rx_credit_sges_[i].addr = chunk_addr;
      rx_credit_wrs_[i].wr_id = chunk_addr;
    }
    DCHECK(ibv_post_recv(credit_qp_, &rx_credit_wrs_[0], &bad_wr) == 0);
    UCCL_LOG_IO << "Posted " << post_credit_rq_cnt_
                << " recv requests for Credit QP";
    post_credit_rq_cnt_ -= kPostRQThreshold;
  }

  if (force && post_credit_rq_cnt_) {
    struct ibv_recv_wr* bad_wr;
    for (int i = 0; i < post_credit_rq_cnt_; i++) {
      uint64_t chunk_addr;
      DCHECK(engine_credit_chunk_pool_->alloc_buff(&chunk_addr) == 0);
      rx_credit_sges_[i].addr = chunk_addr;
      rx_credit_wrs_[i].wr_id = chunk_addr;
    }
    rx_credit_wrs_[post_credit_rq_cnt_ - 1].next = nullptr;
    DCHECK(ibv_post_recv(credit_qp_, &rx_credit_wrs_[0], &bad_wr) == 0);
    UCCL_LOG_IO << "Posted " << post_credit_rq_cnt_
                << " recv requests for Credit QP";
    rx_credit_wrs_[post_credit_rq_cnt_ - 1].next =
        &rx_credit_wrs_[post_credit_rq_cnt_];
    post_credit_rq_cnt_ = 0;
  }
}

void RDMAContext::check_ctrl_rq(bool force) {
  // Populate recv work requests for consuming control packets.
  while (post_ctrl_rq_cnt_ >= kPostRQThreshold) {
    struct ibv_recv_wr* bad_wr;
    for (int i = 0; i < kPostRQThreshold; i++) {
      uint64_t chunk_addr;
      DCHECK(ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
      rx_ack_sges_[i].addr = chunk_addr;
      rx_ack_wrs_[i].wr_id = chunk_addr;
    }
    DCHECK(ibv_post_recv(ctrl_qp_, &rx_ack_wrs_[0], &bad_wr) == 0);
    UCCL_LOG_IO << "Posted " << post_ctrl_rq_cnt_
                << " recv requests for Ctrl QP";
    post_ctrl_rq_cnt_ -= kPostRQThreshold;
  }

  if (force && post_ctrl_rq_cnt_) {
    struct ibv_recv_wr* bad_wr;
    for (int i = 0; i < post_ctrl_rq_cnt_; i++) {
      uint64_t chunk_addr;
      DCHECK(ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
      rx_ack_sges_[i].addr = chunk_addr;
      rx_ack_wrs_[i].wr_id = chunk_addr;
    }
    rx_ack_wrs_[post_ctrl_rq_cnt_ - 1].next = nullptr;
    DCHECK(ibv_post_recv(ctrl_qp_, &rx_ack_wrs_[0], &bad_wr) == 0);
    UCCL_LOG_IO << "Posted " << post_ctrl_rq_cnt_
                << " recv requests for Ctrl QP";
    rx_ack_wrs_[post_ctrl_rq_cnt_ - 1].next = &rx_ack_wrs_[post_ctrl_rq_cnt_];
    post_ctrl_rq_cnt_ = 0;
  }
}

void RDMAContext::check_srq(bool force) {
  struct ibv_sge sge[kPostRQThreshold];
  while (post_srq_cnt_ >= kPostRQThreshold) {
    struct ibv_recv_wr* bad_wr;

    if constexpr (!kRCMode) {
      for (int i = 0; i < kPostRQThreshold; i++) {
        uint64_t chunk_addr;
        DCHECK(retr_chunk_pool_->alloc_buff(&chunk_addr) == 0);
        sge[i].addr = chunk_addr;
        sge[i].length = RetrChunkBuffPool::kRetrChunkSize;
        sge[i].lkey = retr_chunk_pool_->get_lkey();
        imm_wrs_[i].sg_list = &sge[i];
        imm_wrs_[i].num_sge = 1;
        imm_wrs_[i].wr_id = chunk_addr;
      }
    }

    DCHECK(ibv_post_srq_recv(srq_, &imm_wrs_[0], &bad_wr) == 0);
    UCCL_LOG_IO << "Posted " << kPostRQThreshold << " recv requests for SRQ";
    post_srq_cnt_ -= kPostRQThreshold;
  }

  if (force && post_srq_cnt_) {
    struct ibv_recv_wr* bad_wr;
    if constexpr (!kRCMode) {
      for (int i = 0; i < post_srq_cnt_; i++) {
        uint64_t chunk_addr;
        DCHECK(retr_chunk_pool_->alloc_buff(&chunk_addr) == 0);
        sge[i].addr = chunk_addr;
        sge[i].length = RetrChunkBuffPool::kRetrChunkSize;
        sge[i].lkey = retr_chunk_pool_->get_lkey();
        imm_wrs_[i].sg_list = &sge[i];
        imm_wrs_[i].num_sge = 1;
        imm_wrs_[i].wr_id = chunk_addr;
      }
    }

    imm_wrs_[post_srq_cnt_ - 1].next = nullptr;
    DCHECK(ibv_post_srq_recv(srq_, &imm_wrs_[0], &bad_wr) == 0);
    UCCL_LOG_IO << "Posted " << post_srq_cnt_ << " recv requests for SRQ";
    imm_wrs_[post_srq_cnt_ - 1].next = &imm_wrs_[post_srq_cnt_];
    post_srq_cnt_ = 0;
  }
}

void RDMAContext::drain_rtx_queue(SubUcclFlow* subflow) {
  fast_retransmit_for_flow(subflow);
}

bool RDMAContext::try_retransmit_chunk(SubUcclFlow* subflow,
                                       struct wr_ex* wr_ex) {
  if (!EventOnTxRTXData(subflow, wr_ex)) return false;

  auto* lossy_qpw = &dp_qps_[wr_ex->qpidx];
  struct ibv_send_wr retr_wr, *bad_wr;

  // Use SEND/RECV for retransmission through the original lossy QP.
  memset(&retr_wr, 0, sizeof(retr_wr));
  struct ibv_sge retr_sge[2];

  uint64_t retr_hdr;

  if (retr_hdr_pool_->alloc_buff(&retr_hdr)) return false;

  struct retr_chunk_hdr* hdr =
      reinterpret_cast<struct retr_chunk_hdr*>(retr_hdr);
  hdr->remote_addr = wr_ex->wr.wr.rdma.remote_addr;
  // Network byte order.
  hdr->imm_data = wr_ex->wr.imm_data;

  retr_sge[0].addr = retr_hdr;
  retr_sge[0].length = sizeof(struct retr_chunk_hdr);
  retr_sge[0].lkey = retr_hdr_pool_->get_lkey();

  retr_sge[1] = wr_ex->sge;

  retr_wr.wr_id = retr_hdr;
  retr_wr.sg_list = retr_sge;
  retr_wr.num_sge = 2;
  retr_wr.opcode = IBV_WR_SEND;
  retr_wr.send_flags = IBV_SEND_SIGNALED;
  retr_wr.next = nullptr;

  pending_signal_poll_++;

  int ret = ibv_post_send(lossy_qpw->qp, &retr_wr, &bad_wr);
  DCHECK(ret == 0) << ret;

  UCCL_LOG_IO << "successfully retransmit chunk for QP#"
              << (lossy_qpw - dp_qps_)
              << ", remote_addr: " << wr_ex->wr.wr.rdma.remote_addr
              << ", chunk_size: " << wr_ex->sge.length
              << ", csn: " << IMMData(ntohl(wr_ex->wr.imm_data)).GetCSN()
              << " for flow: " << subflow->fid_;

  return true;
}

void RDMAContext::rx_credit(uint64_t pkt_addr) {
  auto* ucclpullh = reinterpret_cast<UcclPullHdr*>(pkt_addr);

  auto fid = ucclpullh->fid.value();
  auto pullno = ucclpullh->pullno.value();

  auto* flow = reinterpret_cast<UcclFlow*>(sender_flow_tbl_[fid]);

  if (unlikely(!flow)) {
    // No Bug. This only happens during connection setup when the sender
    // is not ready while the receiver has already started to send credits.
    return;
  }

  auto* subflow = flow->sub_flows_[engine_offset_];

  EventOnRxCredit(subflow, pullno);
}

void RDMAContext::rx_ack(uint64_t pkt_addr) {
  auto cq_ex = ctrl_cq_ex_;
  uint64_t t5;
  auto t6 = rdtsc();
  auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr);

  auto fid = ucclsackh->fid.value();
  auto qpidx = ucclsackh->path.value();
  auto ackno = ucclsackh->ackno.value();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(sender_flow_tbl_[fid]);
  auto* subflow = flow->sub_flows_[engine_offset_];

  bool update_sackbitmap = false;

  if (UINT_CSN::uintcsn_seqno_lt(ackno, subflow->pcb.snd_una)) {
    UCCL_LOG_IO << "Received old ACK " << ackno << " for flow" << fid
                << " by Ctrl QP";
  } else if (UINT_CSN::uintcsn_seqno_gt(ackno, subflow->pcb.snd_nxt)) {
    UCCL_LOG_IO << "Received ACK for untransmitted data "
                << "ackno: " << ackno
                << ", snd_nxt: " << subflow->pcb.snd_nxt.to_uint32()
                << " for flow" << fid << " by Ctrl QP";
  } else if (UINT_CSN::uintcsn_seqno_eq(ackno, subflow->pcb.snd_una)) {
    UCCL_LOG_IO << "Received duplicate ACK " << ackno << " for flow" << fid
                << " by Ctrl QP";

    EventOnRxNACK(subflow, ucclsackh);

    update_sackbitmap = true;

    subflow->pcb.duplicate_acks++;
    subflow->pcb.snd_ooo_acks = ucclsackh->sack_bitmap_count.value();

    if (subflow->pcb.duplicate_acks < kFastRexmitDupAckThres) {
      // We have not reached the threshold yet, so we do not do
      // retransmission.
    } else if (subflow->pcb.duplicate_acks == kFastRexmitDupAckThres) {
      // Fast retransmit.
      fast_retransmit_for_flow(subflow);
    } else {
      // We have already done the fast retransmit, so we are now
      // in the fast recovery phase.
      auto sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
      // We check the SACK bitmap to see if there are more undelivered
      // chunks. In fast recovery mode we get after a fast
      // retransmit, we will retransmit all missing chunks that we
      // find from the SACK bitmap, when enumerating the SACK bitmap
      // for up to sack_bitmap_count ACKs.
      uint32_t index = 0;
      while (sack_bitmap_count && index < kSackBitmapSize &&
             !subflow->txtracking.empty()) {
        auto bucket_idx = index / PCB::kSackBitmapBucketSize;
        auto sack_bitmap = ucclsackh->sack_bitmap[bucket_idx].value();

        auto cursor = index % PCB::kSackBitmapBucketSize;

        if ((sack_bitmap & (1ULL << cursor)) == 0) {
          // We found a hole.
          auto seqno = subflow->pcb.snd_una + index;
          auto chunk = subflow->txtracking.get_unacked_chunk_from_idx(index);
          if (seqno == chunk.csn) {
            auto wr_ex = chunk.wr_ex;
            if (try_retransmit_chunk(subflow, wr_ex)) {
              subflow->pcb.stats_fast_rexmits++;
            } else {
              // We can't retransmit the chunk due to lack of
              // credits. Quit the loop.
              index = kSackBitmapSize;
            }
          }
          // Rearm timer for Retransmission.
          rearm_timer_for_flow(subflow);
        } else {
          sack_bitmap_count--;
        }
        index++;
      }
    }

  } else {
    UCCL_LOG_IO << "Received valid ACK " << ackno << " for flow" << fid
                << " by Ctrl QP";

    EventOnRxACK(subflow, ucclsackh);

    update_sackbitmap = true;
    auto num_acked_chunks = UINT_CSN(ackno) - subflow->pcb.snd_una;
    auto remote_queueing_tsc =
        us_to_cycles((ucclsackh->remote_queueing.value()), freq_ghz);
    if constexpr (kTestNoHWTimestamp)
      t5 = t6;
    else
      t5 = convert_nic_to_host(ibv_wc_read_completion_ts(cq_ex));

    DCHECK(engine_offset_ < NUM_ENGINES);
    auto reduced_bytes = subflow->unacked_bytes_;
    auto newrtt_tsc = subflow->txtracking.ack_transmitted_chunks(
        subflow, this, num_acked_chunks.to_uint32(), t5, t6,
        remote_queueing_tsc, &subflow->unacked_bytes_);
    reduced_bytes -= subflow->unacked_bytes_;
    *engine_unacked_bytes_ -= reduced_bytes;
    if (qpidx < kPortEntropy)
      subflow->update_scoreboard_rtt(newrtt_tsc, qpidx);
    else {
      // This ack is for retransmitted chunk.
      // Don't update scoreboard for retransmitted chunks.
    }

    subflow->pcb.snd_una = ackno;
    subflow->pcb.duplicate_acks = 0;
    subflow->pcb.snd_ooo_acks = 0;
    subflow->pcb.rto_rexmits_consectutive = 0;
    if (!subflow->txtracking.empty()) {
      // Rearm timer if we still have unacked chunks.
      rearm_timer_for_flow(subflow);
    } else {
      disarm_timer_for_flow(subflow);
    }
  }

  // For duplicate ACKs and valid ACKs, we may need to update the SACK bitmap
  // at the sender side.
  if (update_sackbitmap) {
    for (int i = 0; i < kSackBitmapSize / PCB::kSackBitmapBucketSize; i++)
      subflow->pcb.tx_sack_bitmap[i] = ucclsackh->sack_bitmap[i].value();
    subflow->pcb.tx_sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
    subflow->pcb.tx_sack_bitmap_base = ackno;
  }
}

int RDMAContext::poll_credit_cq(void) {
  uint64_t chunk_addr;
  int work = 0;
  while (1) {
    struct ibv_poll_cq_attr poll_cq_attr = {};
    auto cq_ex = engine_credit_cq_ex_;
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return work;

    int cq_budget = 0;

    while (1) {
      if (cq_ex->status == IBV_WC_SUCCESS) {
        // Completion for receiving ACKs.
        chunk_addr = cq_ex->wr_id;
        if (ibv_wc_read_opcode(cq_ex) == IBV_WC_RECV) {
          rx_credit(chunk_addr);
          post_credit_rq_cnt_++;
        }
        engine_credit_chunk_pool_->free_buff(chunk_addr);
      } else {
        LOG(ERROR) << "Credit CQ state error: " << cq_ex->status;
      }

      if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }

    ibv_end_poll(cq_ex);

    work += cq_budget;

    if (cq_budget < kMaxBatchCQ) break;
  }

  return work;
}

int RDMAContext::poll_ctrl_cq(void) {
  uint64_t chunk_addr;
  int work = 0;
  while (1) {
    struct ibv_poll_cq_attr poll_cq_attr = {};
    auto cq_ex = ctrl_cq_ex_;
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return work;

    int cq_budget = 0;

    while (1) {
      if (cq_ex->status == IBV_WC_SUCCESS) {
        // Completion for receiving ACKs.
        chunk_addr = cq_ex->wr_id;
        if (ibv_wc_read_opcode(cq_ex) == IBV_WC_RECV) {
          auto imm_data = ntohl(ibv_wc_read_imm_data(cq_ex));
          auto num_ack = imm_data;
          UCCL_LOG_IO << "Receive " << num_ack
                      << " ACKs in one CtrlChunk, Chunk addr: " << cq_ex->wr_id;
          for (int i = 0; i < num_ack; i++) {
            auto pkt_addr = chunk_addr + i * CtrlChunkBuffPool::kPktSize;
            rx_ack(pkt_addr);
          }
          post_ctrl_rq_cnt_++;
        }
        ctrl_chunk_pool_->free_buff(chunk_addr);
      } else {
        LOG(ERROR) << "Ctrl CQ state error: " << cq_ex->status;
      }

      if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }

    ibv_end_poll(cq_ex);

    work += cq_budget;

    if (cq_budget < kMaxBatchCQ) break;
  }

  return work;
}

void RDMAContext::burst_timing_wheel(void) {
  auto wheel = &wheel_;
  struct ibv_send_wr* bad_wr;

  wheel->reap(rdtsc());

  auto num_chunks = std::min(kMaxBurstTW, (uint32_t)wheel->ready_queue_.size());

  for (auto i = 0; i < num_chunks; i++) {
    struct wr_ex* wr_ex =
        reinterpret_cast<struct wr_ex*>(wheel->ready_queue_.front().sslot_);
    auto* wr = &wr_ex->wr;
    auto* flow = reinterpret_cast<UcclFlow*>(wr_ex->ureq->context);
    auto* subflow = flow->sub_flows_[engine_offset_];
    // Select QP.
    auto qpidx = select_qpidx_pot(wr_ex->sge.length, subflow);
    auto qpw = &dp_qps_[qpidx];

    wr->send_flags = 0;
    if (qpw->signal_cnt_++ % kSignalInterval == 0) {
      wr->send_flags = IBV_SEND_SIGNALED;
      pending_signal_poll_++;
    }
    wr_ex->qpidx = qpidx;

    auto ret = ibv_post_send(qpw->qp, &wr_ex->wr, &bad_wr);
    DCHECK(ret == 0) << pending_signal_poll_ << ", " << ret;

    IMMData imm_data(ntohl(wr_ex->wr.imm_data));
    // Track this chunk.
    subflow->txtracking.track_chunk(wr_ex->ureq, wr_ex, rdtsc(),
                                    imm_data.GetCSN(), imm_data.GetHINT());
    if constexpr (!kRCMode) {
      // Arm timer for TX
      arm_timer_for_flow(subflow);
    }
    UCCL_LOG_IO << "Burst send: csn: " << imm_data.GetCSN() << " with QP#"
                << wr_ex->qpidx;

    subflow->in_wheel_cnt_--;

    wheel->ready_queue_.pop_front();

    if (*engine_unacked_bytes_ >= kMaxUnAckedBytesPerEngineHigh) {
      // The code is here because we want to at least send one chunk.
      // Push the message to the pending transmit queue.
      return;
    }
  }
}

void RDMAContext::try_update_csn(SubUcclFlow* subflow) {
  while (!subflow->rxtracking.ready_csn_.empty() &&
         subflow->rxtracking.ready_csn_.begin()->first.to_uint32() ==
             subflow->pcb.rcv_nxt.to_uint32()) {
    struct RecvRequest* req = reinterpret_cast<struct RecvRequest*>(
        subflow->rxtracking.ready_csn_.begin()->second);
    if (req) {
      // This is the last chunk of a message whose size is mismatched with
      // the expected size. I.e., send size < recv size. Fix
      // req->ureq->data_len[0] and wakeup application.
      req->ureq->recv.data_len[0] = req->received_bytes[0];
      // Wakeup app thread.
      uccl_wakeup(req->ureq->poll_ctx);
      // Free the request.
      free_recvreq(req);
    }

    subflow->rxtracking.ready_csn_.erase(
        subflow->rxtracking.ready_csn_.begin());

    // Data is already DMAed to the application buffer.
    // Nothing more to do.

    subflow->pcb.advance_rcv_nxt();
    UCCL_LOG_IO << "try_update_csn:"
                << " rcv_nxt: " << subflow->pcb.rcv_nxt.to_uint32();

    if constexpr (!kRCMode) {
      subflow->pcb.sack_bitmap_shift_left_one();
    }
  }
}

void RDMAContext::rx_rtx_data(struct list_head* ack_list) {
  UCCL_LOG_IO << "rx_rtx_data";
  auto cq_ex = recv_cq_ex_;
  auto now = rdtsc();

  auto chunk_addr = cq_ex->wr_id;
  auto chunk_len = ibv_wc_read_byte_len(cq_ex) - sizeof(struct retr_chunk_hdr);

  struct retr_chunk_hdr* hdr =
      reinterpret_cast<struct retr_chunk_hdr*>(chunk_addr);

  auto imm_data = IMMData(ntohl(hdr->imm_data));

  auto last_chunk = imm_data.GetHINT();
  auto csn = imm_data.GetCSN();
  auto rid = imm_data.GetRID();
  auto fid = imm_data.GetFID();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
  auto* subflow = flow->sub_flows_[engine_offset_];

  UCCL_LOG_IO << "Received retransmission chunk: (csn, rid, fid): " << csn
              << ", " << rid << ", " << fid;

  // Locate request by rid
  DCHECK(rid < kMaxReq);
  auto req = get_recvreq_by_id(rid);
  if (req->type != RecvRequest::RECV || req->ureq->context != flow) {
    UCCL_LOG_IO << "Can't find corresponding request or this request is "
                   "invalid for this retransmission chunk. Dropping.";
    subflow->pcb.stats_retr_chunk_drop++;
    return;
  }

  // Compare CSN with the expected CSN.
  auto ecsn = subflow->pcb.rcv_nxt;
  auto distance = UINT_CSN(csn) - ecsn;

  if (UINT_CSN::uintcsn_seqno_lt(UINT_CSN(csn), ecsn)) {
    // Original chunk is already received.
    UCCL_LOG_IO << "Original chunk is already received. Dropping "
                   "retransmission chunk for flow"
                << fid;
    subflow->pcb.stats_retr_chunk_drop++;
    return;
  }

  if (distance.to_uint32() > kReassemblyMaxSeqnoDistance) {
    UCCL_LOG_IO << "Chunk too far ahead. Dropping as we can't handle SACK. "
                << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    subflow->pcb.stats_retr_chunk_drop++;
    return;
  }

  auto bitmap_bucket_idx = distance.to_uint32() / PCB::kSackBitmapBucketSize;
  auto cursor = distance.to_uint32() % PCB::kSackBitmapBucketSize;
  auto sack_bitmap = &subflow->pcb.sack_bitmap[bitmap_bucket_idx];

  if ((*sack_bitmap & (1ULL << cursor))) {
    // Original chunk is already received.
    UCCL_LOG_IO << "Original chunk is already received. Dropping "
                   "retransmission chunk for flow"
                << fid;
    subflow->pcb.stats_retr_chunk_drop++;
    return;
  }

  UCCL_LOG_IO << "This retransmission chunk is accepted!!!";
// Accept this retransmission chunk.
#ifdef CPU_MEMORY
  memcpy(reinterpret_cast<void*>(hdr->remote_addr),
         reinterpret_cast<void*>(chunk_addr + sizeof(struct retr_chunk_hdr)),
         chunk_len);
#else
  DCHECK(hipMemcpy(reinterpret_cast<void*>(hdr->remote_addr),
                   reinterpret_cast<void*>(chunk_addr +
                                           sizeof(struct retr_chunk_hdr)),
                   chunk_len, hipMemcpyHostToDevice) == hipSuccess);
#endif

  subflow->pcb.stats_accept_retr++;

  subflow->pcb.sack_bitmap_bit_set(distance.to_uint32());

  auto* msg_size = &req->ureq->recv.elems[0].size;
  uint32_t* received_bytes = req->received_bytes;
  received_bytes[0] += chunk_len;

  if (!last_chunk) {
    req = nullptr;
  }

  subflow->rxtracking.ready_csn_.insert({csn, req});

  try_update_csn(subflow);

  /// FIXME: Should we send ACK immediately here?
  if (list_empty(&subflow->ack.ack_link))
    list_add_tail(&subflow->ack.ack_link, ack_list);
  // Don't let sender update the path's rtt.
  subflow->next_ack_path_ = std::numeric_limits<uint16_t>::max();

  EventOnRxRTXData(subflow, &imm_data);

  return;
}

void RDMAContext::rc_rx_data(void) {
  auto cq_ex = recv_cq_ex_;
  auto byte_len = ibv_wc_read_byte_len(cq_ex);
  auto imm_data = IMMData(ntohl(ibv_wc_read_imm_data(cq_ex)));

  auto last_chunk = imm_data.GetHINT();
  auto csn = imm_data.GetCSN();
  auto rid = imm_data.GetRID();
  auto fid = imm_data.GetFID();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
  DCHECK(flow) << fid << ", RDMAContext ptr: " << this;
  auto* subflow = flow->sub_flows_[engine_offset_];

  UCCL_LOG_IO << "Received chunk: (byte_len, csn, rid, fid): " << byte_len
              << ", " << csn << ", " << rid << ", " << fid;

  // Locate request by rid
  DCHECK(rid < kMaxReq);
  auto req = get_recvreq_by_id(rid);
  DCHECK(req->ureq);

  if (req->type != RecvRequest::RECV || req->ureq->context != flow) {
    LOG(ERROR) << "Can't find corresponding request or this request is "
                  "invalid for this chunk. Dropping.";
    // This should never happen.
    CHECK(0);
    return;
  }

  // There is no need to check CSN as RC provides reliable delivery.
  auto* msg_size = &req->ureq->recv.elems[0].size;
  uint32_t* received_bytes = req->received_bytes;
  received_bytes[0] += byte_len;

  if (!last_chunk) {
    req = nullptr;
  }

  subflow->rxtracking.ready_csn_.insert({csn, req});

  try_update_csn(subflow);

  EventOnRxData(subflow, &imm_data);
}

void RDMAContext::rx_data(struct list_head* ack_list) {
  auto cq_ex = recv_cq_ex_;
  auto now = rdtsc();
  auto byte_len = ibv_wc_read_byte_len(cq_ex);
  auto imm_data = IMMData(ntohl(ibv_wc_read_imm_data(cq_ex)));
  auto qp_num = ibv_wc_read_qp_num(cq_ex);
  auto qpidx = qpn2idx_[qp_num];

  auto last_chunk = imm_data.GetHINT();
  auto csn = imm_data.GetCSN();
  auto rid = imm_data.GetRID();
  auto fid = imm_data.GetFID();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
  auto* subflow = flow->sub_flows_[engine_offset_];

  UCCL_LOG_IO << "Received chunk: (byte_len, csn, rid, fid): " << byte_len
              << ", " << csn << ", " << rid << ", " << fid << " from QP#"
              << qpidx;

  // Locate request by rid
  DCHECK(rid < kMaxReq);
  auto req = get_recvreq_by_id(rid);
  if (req->type != RecvRequest::RECV || req->ureq->context != flow) {
    UCCL_LOG_IO << "Can't find corresponding request or this request is "
                   "invalid for this chunk. Dropping.";
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  // Compare CSN with the expected CSN.
  auto ecsn = subflow->pcb.rcv_nxt;
  auto distance = UINT_CSN(csn) - ecsn;

  if (UINT_CSN::uintcsn_seqno_lt(UINT_CSN(csn), ecsn)) {
    UCCL_LOG_IO << "Chunk lag behind. Dropping as we can't handle SACK. "
                << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  if (distance.to_uint32() > kReassemblyMaxSeqnoDistance) {
    UCCL_LOG_IO << "Chunk too far ahead. Dropping as we can't handle SACK. "
                << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  // Always use the latest timestamp.
  if constexpr (kTestNoHWTimestamp)
    subflow->pcb.t_remote_nic_rx = now;
  else
    subflow->pcb.t_remote_nic_rx = ibv_wc_read_completion_ts(cq_ex);

  subflow->pcb.sack_bitmap_bit_set(distance.to_uint32());

  auto* msg_size = &req->ureq->recv.elems[0].size;
  uint32_t* received_bytes = req->received_bytes;
  received_bytes[0] += byte_len;

  if (!last_chunk) {
    req = nullptr;
  }

  subflow->rxtracking.ready_csn_.insert({csn, req});

  try_update_csn(subflow);

  if (distance.to_uint32()) {
    subflow->rxtracking.encounter_ooo();
#ifdef STATS
    subflow->pcb.stats_ooo++;
    subflow->pcb.stats_maxooo =
        std::max(subflow->pcb.stats_maxooo, distance.to_uint32());
    if (subflow->rxtracking.real_ooo()) subflow->pcb.stats_real_ooo++;
#endif
  }

  subflow->rxtracking.cumulate_wqe();
  subflow->rxtracking.cumulate_bytes(byte_len);

  if (list_empty(&subflow->ack.ack_link))
    list_add_tail(&subflow->ack.ack_link, ack_list);
  subflow->next_ack_path_ = qpidx;

  // Send ACK if needed.
  if (subflow->rxtracking.need_imm_ack()) {
    uint64_t chunk_addr;
    DCHECK(ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
    craft_ack(subflow, chunk_addr, 0);
    flush_acks(1, chunk_addr);

    subflow->rxtracking.clear_imm_ack();
    list_del(&subflow->ack.ack_link);
  }

  EventOnRxData(subflow, &imm_data);
}

void RDMAContext::flush_acks(int num_ack, uint64_t chunk_addr) {
  if (num_ack == 0) return;

  struct ibv_sge sge = {
      .addr = chunk_addr,
      .length = CtrlChunkBuffPool::kPktSize * num_ack,
      .lkey = ctrl_chunk_pool_->get_lkey(),
  };

  tx_ack_wr_.sg_list = &sge;

  // For future free.
  tx_ack_wr_.wr_id = chunk_addr;

  // Tell sender how many acks are in this wqe.
  tx_ack_wr_.imm_data = htonl(num_ack);

  struct ibv_send_wr* bad_wr;
  DCHECK(ibv_post_send(ctrl_qp_, &tx_ack_wr_, &bad_wr) == 0);

  UCCL_LOG_IO << "Flush " << num_ack << " ACKs";
}

void RDMAContext::craft_ack(SubUcclFlow* subflow, uint64_t chunk_addr,
                            int num_sge) {
  uint64_t pkt_addr = chunk_addr + CtrlChunkBuffPool::kPktSize * num_sge;
  auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr);

  ucclsackh->ackno = be16_t(subflow->pcb.ackno().to_uint32());
  ucclsackh->fid = be16_t(subflow->fid_);
  ucclsackh->path = be16_t(subflow->next_ack_path_);

  auto t4 = rdtsc();
  uint64_t t2;
  if constexpr (kTestNoHWTimestamp)
    t2 = subflow->pcb.t_remote_nic_rx;
  else
    t2 = convert_nic_to_host(subflow->pcb.t_remote_nic_rx);

  ucclsackh->remote_queueing = be64_t(to_usec(t4 - t2, freq_ghz));

  for (size_t i = 0; i < sizeof(UcclSackHdr::sack_bitmap) /
                             sizeof(UcclSackHdr::sack_bitmap[0]);
       ++i) {
    ucclsackh->sack_bitmap[i] = be64_t(subflow->pcb.sack_bitmap[i]);
  }
  ucclsackh->sack_bitmap_count = be16_t(subflow->pcb.sack_bitmap_count);

  UCCL_LOG_IO << "craft_ack ackno: " << subflow->pcb.ackno().to_uint32()
              << " for flow: " << subflow->fid_;
}

void RDMAContext::__retransmit_for_flow(void* context, bool rto) {
  SubUcclFlow* subflow = reinterpret_cast<SubUcclFlow*>(context);

  if (subflow->txtracking.empty()) {
    UCCL_LOG_IO << "No unacked chunk to retransmit for flow" << subflow->fid_;
    return;
  }

  if (subflow->pcb.rto_rexmits_consectutive >= kRTOAbortThreshold) {
    LOG_FIRST_N(ERROR, 1) << "RTO retransmission threshold reached."
                          << subflow->fid_;
  }

  // Case#1: SACK bitmap at the sender side is empty. Retransmit the oldest
  // unacked chunk.
  auto sack_bitmap_count = subflow->pcb.tx_sack_bitmap_count;
  if (!sack_bitmap_count) {
    auto chunk = subflow->txtracking.get_oldest_unacked_chunk();
    auto wr_ex = chunk.wr_ex;
    try_retransmit_chunk(subflow, wr_ex);
    // Arm timer for Retransmission
    rearm_timer_for_flow(subflow);
    if (rto) {
      subflow->pcb.stats_rto_rexmits++;
      subflow->pcb.rto_rexmits_consectutive++;
    } else {
      subflow->pcb.stats_fast_rexmits++;
    }
    return;
  }

  // Case#2: Retransmit the unacked chunks according to the SACK bitmap.
  bool done = false;
  auto tx_sack_bitmap_base = UINT_CSN(subflow->pcb.tx_sack_bitmap_base);

  uint32_t index = 0;
  while (sack_bitmap_count && index < kSackBitmapSize &&
         !subflow->txtracking.empty()) {
    auto bucket_idx = index / PCB::kSackBitmapBucketSize;
    auto sack_bitmap = subflow->pcb.tx_sack_bitmap[bucket_idx];

    auto cursor = index % PCB::kSackBitmapBucketSize;

    if ((sack_bitmap & (1ULL << cursor)) == 0) {
      // We found a hole.
      auto seqno = tx_sack_bitmap_base + index;
      DCHECK(index < subflow->txtracking.track_size());
      auto chunk = subflow->txtracking.get_unacked_chunk_from_idx(index);
      if (seqno == chunk.csn) {
        auto wr_ex = chunk.wr_ex;
        if (try_retransmit_chunk(subflow, wr_ex)) {
          done = true;
        } else {
          // We can't retransmit the chunk due to lack of credits.
          // Quit the loop.
          index = kSackBitmapSize;
        }
      } else {
        // This bit is stale and its corresponding chunk is already
        // acked. Do nothing.
        UCCL_LOG_IO << "Stale SACK bit for seqno: " << seqno.to_uint32()
                    << ", chunk.csn: " << chunk.csn << ", tx_sack_bitmap_base: "
                    << tx_sack_bitmap_base.to_uint32();
      }
    } else {
      sack_bitmap_count--;
    }
    index++;
  }

  // Arm timer for Retransmission
  rearm_timer_for_flow(subflow);
  if (done) {
    if (rto) {
      subflow->pcb.stats_rto_rexmits++;
      subflow->pcb.rto_rexmits_consectutive++;
    } else {
      subflow->pcb.stats_fast_rexmits++;
    }
  }
}

uint32_t RDMAContext::select_qpidx_pot(uint32_t msize, void* subflow_context) {
  if (can_use_last_choice(msize)) return last_qp_choice_;

  auto* sublfow = reinterpret_cast<SubUcclFlow*>(subflow_context);
  auto q1 = select_qpidx_rand();
  auto q2 = select_qpidx_rand();

  // Return the QP with lower RTT.
  auto qpidx =
      sublfow->scoreboard_rtt_[q1] < sublfow->scoreboard_rtt_[q2] ? q1 : q2;
  last_qp_choice_ = qpidx;
  return qpidx;
}

// Try to arm a timer for the given flow. If the timer is already armed, do
// nothing.
void RDMAContext::arm_timer_for_flow(void* context) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(context);
  if (!subflow->rto_armed) {
    if constexpr (kConstRTO) {
      rto_->arm_timer({this, subflow});
    } else {
      rto_->arm_timer({this, subflow},
                      std::max(kRTORTT * subflow->pcb.timely_cc.get_avg_rtt(),
                               kMinRTOUsec));
    }
    subflow->rto_armed = true;
  }
}

// Try to rearm a timer for the given flow. If the timer is not armed, arm it.
// If the timer is already armed, rearm it.
void RDMAContext::rearm_timer_for_flow(void* context) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(context);
  if (subflow->rto_armed) {
    if constexpr (kConstRTO) {
      rto_->rearm_timer({this, subflow});
    } else {
      rto_->rearm_timer({this, subflow},
                        std::max(kRTORTT * subflow->pcb.timely_cc.get_avg_rtt(),
                                 kMinRTOUsec));
    }
  } else {
    arm_timer_for_flow(subflow);
  }
}

void RDMAContext::mark_flow_timeout(void* context) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(context);
  subflow->rto_armed = false;
}

void RDMAContext::disarm_timer_for_flow(void* context) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(context);
  if (subflow->rto_armed) {
    rto_->disarm_timer({this, subflow});
    subflow->rto_armed = false;
  }
}

std::string RDMAContext::to_string() {
  std::string s;
  s.clear();

  uint32_t stats_rto_rexmits = 0;
  uint32_t stats_fast_rexmits = 0;
  uint32_t stats_accept_retr = 0;

  uint32_t stats_chunk_drop = 0;
  uint32_t stats_retr_chunk_drop = 0;
  uint32_t stats_ooo = 0;
  uint32_t stats_real_ooo = 0;
  uint32_t stats_maxooo = 0;

  for (int fid = 0; fid < flow_cnt_; fid++) {
    {
      // Sender flows
      auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
      if (flow) {
        auto* subflow = flow->sub_flows_[engine_offset_];
        stats_accept_retr += subflow->pcb.stats_accept_retr;

        stats_chunk_drop += subflow->pcb.stats_chunk_drop;
        stats_retr_chunk_drop += subflow->pcb.stats_retr_chunk_drop;
        stats_ooo += subflow->pcb.stats_ooo;
        stats_real_ooo += subflow->pcb.stats_real_ooo;
        stats_maxooo = std::max(stats_maxooo, subflow->pcb.stats_maxooo);
        subflow->pcb.stats_maxooo = 0;  // Inaccurate is fine.
      }
    }
    {
      // Receiver flows
      auto* flow = reinterpret_cast<UcclFlow*>(sender_flow_tbl_[fid]);
      if (flow) {
        auto* subflow = flow->sub_flows_[engine_offset_];
        stats_rto_rexmits += subflow->pcb.stats_rto_rexmits;
        stats_fast_rexmits += subflow->pcb.stats_fast_rexmits;
      }
    }
  }

  s += "\tRTO retr:" + std::to_string(stats_rto_rexmits) +
       "/Fast retr:" + std::to_string(stats_fast_rexmits) +
       "/Eat retr:" + std::to_string(stats_accept_retr) +
       "/Chunk drop:" + std::to_string(stats_chunk_drop) +
       "/Retr drop:" + std::to_string(stats_retr_chunk_drop) +
       "/OOO: " + std::to_string(stats_ooo) +
       "/ROOO: " + std::to_string(stats_real_ooo) +
       "/MAXOOO: " + std::to_string(stats_maxooo);

  s += "\n";

  return s;
}

}  // namespace uccl

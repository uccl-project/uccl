#ifndef UTIL_RDMA_H
#define UTIL_RDMA_H

#include "transport_config.h"
#include "util/util.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>

namespace uccl {
// LRH (Local Routing Header) + GRH (Global Routing Header) + BTH (Base
// Transport Header)
static constexpr uint32_t IB_HDR_OVERHEAD = (8 + 40 + 12);
// Ethernet + IPv4 + UDP + BTH
static constexpr uint32_t ROCE_IPV4_HDR_OVERHEAD = (14 + 20 + 8 + 12);
// Ethernet + IPv6 + UDP + BTH
static constexpr uint32_t ROCE_IPV6_HDR_OVERHEAD = (14 + 40 + 8 + 12);
// Headroom for UD packets.
static constexpr uint32_t UD_ADDITION = 40;

static constexpr uint32_t BASE_PSN = 0;

// For quick computation at MTU 4096
static uint32_t MAX_CHUNK_ROCE_IPV4_4096_HDR_OVERHEAD =
    (((ucclParamCHUNK_SIZE_KB() << 10) + 4096) / 4096) * ROCE_IPV4_HDR_OVERHEAD;
static uint32_t MAX_CHUNK_ROCE_IPV6_4096_HDR_OVERHEAD =
    (((ucclParamCHUNK_SIZE_KB() << 10) + 4096) / 4096) * ROCE_IPV6_HDR_OVERHEAD;
static uint32_t MAX_CHUNK_IB_4096_HDR_OVERHEAD =
    (((ucclParamCHUNK_SIZE_KB() << 10) + 4096) / 4096) * IB_HDR_OVERHEAD;

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

static inline int util_rdma_get_link_speed_from_ibv_speed(int active_speed,
                                                          int active_width) {
  return (ncclIbSpeed(active_speed) * ncclIbWidth(active_width)) * 1e6 / 8;
}

static inline uint16_t util_rdma_extract_local_subnet_prefix(
    uint64_t subnet_prefix) {
  return (be64toh(subnet_prefix) & 0xffff);
}

static inline void util_rdma_create_qp_seperate_cq(
    struct ibv_context* context, struct ibv_qp** qp, enum ibv_qp_type qp_type,
    bool cq_ex, bool ts, struct ibv_cq** scq, struct ibv_cq** rcq,
    bool share_cq, uint32_t cqsize, struct ibv_pd* pd, int port,
    uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sge,
    uint32_t max_recv_sge) {
  // Creating SCQ and RCQ
  if (!share_cq) {
    if (cq_ex) {
      struct ibv_cq_init_attr_ex cq_ex_attr;
      cq_ex_attr.cqe = cqsize;
      cq_ex_attr.cq_context = nullptr;
      cq_ex_attr.channel = nullptr;
      cq_ex_attr.comp_vector = 0;
      cq_ex_attr.wc_flags =
          IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM | IBV_WC_EX_WITH_QP_NUM |
          IBV_WC_EX_WITH_SRC_QP |
          IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;  // Timestamp support.
      if constexpr (kTestNoHWTimestamp)
        cq_ex_attr.wc_flags &= ~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;
      cq_ex_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
      cq_ex_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED |
                         IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
      auto scq_ex = (struct ibv_cq_ex**)scq;
      *scq_ex = ibv_create_cq_ex(context, &cq_ex_attr);
      UCCL_INIT_CHECK(*scq_ex != nullptr, "ibv_create_cq_ex failed");

      auto rcq_ex = (struct ibv_cq_ex**)rcq;
      *rcq_ex = ibv_create_cq_ex(context, &cq_ex_attr);
      UCCL_INIT_CHECK(*rcq_ex != nullptr, "ibv_create_cq_ex failed");
    } else {
      *scq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
      UCCL_INIT_CHECK(*scq != nullptr, "ibv_create_cq failed");

      *rcq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
      UCCL_INIT_CHECK(*rcq != nullptr, "ibv_create_cq failed");
    }
  }

  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));

  qp_init_attr.send_cq = *scq;
  qp_init_attr.recv_cq = *rcq;
  qp_init_attr.qp_type = qp_type;

  qp_init_attr.cap.max_send_wr = max_send_wr;
  qp_init_attr.cap.max_recv_wr = max_recv_wr;
  qp_init_attr.cap.max_send_sge = max_send_sge;
  qp_init_attr.cap.max_recv_sge = max_recv_sge;
  // kMaxRecv * sizeof(struct FifoItem)
  qp_init_attr.cap.max_inline_data = kMaxInline;

  // Creating QP
  *qp = ibv_create_qp(pd, &qp_init_attr);
  UCCL_INIT_CHECK(*qp != nullptr, "ibv_create_qp failed");

  // Modifying QP state to INIT
  struct ibv_qp_attr qp_attr;
  int attr_mask =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  memset(&qp_attr, 0, sizeof(qp_attr));
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = port;
  qp_attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE |
      ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0);

  UCCL_INIT_CHECK(ibv_modify_qp(*qp, &qp_attr, attr_mask) == 0,
                  "ibv_modify_qp failed");
}

static inline void util_rdma_create_qp(
    struct ibv_context* context, struct ibv_qp** qp, enum ibv_qp_type qp_type,
    bool cq_ex, bool ts, struct ibv_cq** cq, bool share_cq, uint32_t cqsize,
    struct ibv_pd* pd, int port, struct ibv_mr** mr, void* addr, size_t mr_size,
    uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sge,
    uint32_t max_recv_sge) {
  // Creating CQ
  if (!share_cq) {
    if (cq_ex) {
      struct ibv_cq_init_attr_ex cq_ex_attr;
      cq_ex_attr.cqe = cqsize;
      cq_ex_attr.cq_context = nullptr;
      cq_ex_attr.channel = nullptr;
      cq_ex_attr.comp_vector = 0;
      cq_ex_attr.wc_flags =
          IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM | IBV_WC_EX_WITH_QP_NUM |
          IBV_WC_EX_WITH_SRC_QP |
          IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;  // Timestamp support.
      if constexpr (kTestNoHWTimestamp)
        cq_ex_attr.wc_flags &= ~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;
      cq_ex_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
      cq_ex_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED |
                         IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
      auto cq_ex = (struct ibv_cq_ex**)cq;
      *cq_ex = ibv_create_cq_ex(context, &cq_ex_attr);
      UCCL_INIT_CHECK(*cq_ex != nullptr, "ibv_create_cq_ex failed");
    } else {
      *cq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
      UCCL_INIT_CHECK(*cq != nullptr, "ibv_create_cq failed");
    }
  }

  // Creating MR
  if (addr == nullptr) {
    addr = mmap(nullptr, mr_size, PROT_READ | PROT_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    UCCL_INIT_CHECK(addr != MAP_FAILED, "mmap failed");
  }
  memset(addr, 0, mr_size);

  *mr = ibv_reg_mr(pd, addr, mr_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0));
  UCCL_INIT_CHECK(*mr != nullptr, "ibv_reg_mr failed");

  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));

  qp_init_attr.send_cq = *cq;
  qp_init_attr.recv_cq = *cq;
  qp_init_attr.qp_type = qp_type;

  qp_init_attr.cap.max_send_wr = max_send_wr;
  qp_init_attr.cap.max_recv_wr = max_recv_wr;
  qp_init_attr.cap.max_send_sge = max_send_sge;
  qp_init_attr.cap.max_recv_sge = max_recv_sge;
  // kMaxRecv * sizeof(struct FifoItem)
  qp_init_attr.cap.max_inline_data = kMaxInline;

  // Creating QP
  *qp = ibv_create_qp(pd, &qp_init_attr);
  UCCL_INIT_CHECK(*qp != nullptr, "ibv_create_qp failed");

  // Modifying QP state to INIT
  struct ibv_qp_attr qp_attr;
  int attr_mask =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  memset(&qp_attr, 0, sizeof(qp_attr));
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = port;
  qp_attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE |
      ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0);

  if (qp_type == IBV_QPT_UD) {
    // Use QP number as qkey.
    qp_attr.qkey = (*qp)->qp_num;
    attr_mask &= ~IBV_QP_ACCESS_FLAGS;
    attr_mask |= IBV_QP_QKEY;
  }

  UCCL_INIT_CHECK(ibv_modify_qp(*qp, &qp_attr, attr_mask) == 0,
                  "ibv_modify_qp failed");
}

static inline struct ibv_srq* util_rdma_create_srq(struct ibv_pd* pd,
                                                   uint32_t max_wr,
                                                   uint32_t max_sge,
                                                   uint32_t srq_limit) {
  struct ibv_srq* srq = nullptr;
  struct ibv_srq_init_attr srq_init_attr;
  memset(&srq_init_attr, 0, sizeof(srq_init_attr));
  srq_init_attr.attr.max_wr = max_wr;
  srq_init_attr.attr.max_sge = max_sge;
  srq_init_attr.attr.srq_limit = srq_limit;
  srq = ibv_create_srq(pd, &srq_init_attr);
  return srq;
}

static inline struct ibv_ah* util_rdma_create_ah(
    struct ibv_pd* pd, uint8_t port, union ibv_gid remote_gid,
    struct ibv_port_attr remote_port_attr, bool roce) {
  struct ibv_ah_attr ah_attr = {};

  if (roce) {
    ah_attr.is_global = 1;
    ah_attr.grh.dgid = remote_gid;
    ah_attr.grh.traffic_class = ucclParamROCE_TRAFFIC_CLASS();
    ah_attr.grh.sgid_index = ucclParamROCE_GID_IDX();
    ah_attr.grh.flow_label = 0;
    ah_attr.grh.hop_limit = 0xff;
    ah_attr.sl = ucclParamROCE_SERVICE_LEVEL();
  } else {
    ah_attr.is_global = 0;
    ah_attr.dlid = remote_port_attr.lid;
    ah_attr.sl = ucclParamIB_SERVICE_LEVEL();
  }

  ah_attr.port_num = port;

  struct ibv_ah* ah = ibv_create_ah(pd, &ah_attr);

  return ah;
}

static inline struct ibv_mr* util_rdma_create_host_memory_mr(struct ibv_pd* pd,
                                                             size_t size) {
  struct ibv_mr* mr = nullptr;
  void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  UCCL_INIT_CHECK(addr != MAP_FAILED, "mmap failed");
  mr = ibv_reg_mr(pd, addr, size,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  UCCL_INIT_CHECK(mr != nullptr, "ibv_reg_mr failed");
  return mr;
}

static inline struct ibv_cq* util_rdma_create_cq(struct ibv_context* context,
                                                 uint32_t cqsize) {
  struct ibv_cq* cq = nullptr;
  cq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
  return cq;
}

static inline struct ibv_cq_ex* util_rdma_create_cq_ex(
    struct ibv_context* context, uint32_t cqsize) {
  struct ibv_cq_ex* cq_ex = nullptr;
  struct ibv_cq_init_attr_ex cq_ex_attr;
  cq_ex_attr.cqe = cqsize;
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

  cq_ex = ibv_create_cq_ex(context, &cq_ex_attr);
  return cq_ex;
}

static inline int util_rdma_modify_cq_attr(struct ibv_cq_ex* cq_ex,
                                           uint32_t cq_count,
                                           uint32_t cq_period) {
  struct ibv_modify_cq_attr cq_attr;
  cq_attr.attr_mask = IBV_CQ_ATTR_MODERATE;
  cq_attr.moderate.cq_count = cq_count;
  cq_attr.moderate.cq_period = cq_period;

  return ibv_modify_cq(ibv_cq_ex_to_cq(cq_ex), &cq_attr);
}

/**
 * @brief This helper function converts an Infiniband name (e.g., mlx5_0) to an
 * Ethernet name (e.g., eth0)
 * @return int -1 on error, 0 on success
 */
static inline int util_rdma_ib2eth_name(char const* ib_name,
                                        char* ethernet_name) {
  char command[512];
  snprintf(command, sizeof(command),
           "ls -l /sys/class/infiniband/%s/device/net | sed -n '2p' | sed "
           "'s/.* //'",
           ib_name);
  FILE* fp = popen(command, "r");
  if (fp == nullptr) {
    perror("popen");
    return -1;
  }
  if (fgets(ethernet_name, 64, fp) == NULL) {
    pclose(fp);
    return -1;
  }
  pclose(fp);
  // Remove newline character if present
  ethernet_name[strcspn(ethernet_name, "\n")] = '\0';
  return 0;
}

/**
 * @brief This helper function gets the IP address of the device from Infiniband
 * name.
 *
 * @param ib_name
 * @param ip
 * @return int
 */
static inline int util_rdma_get_ip_from_ib_name(char const* ib_name,
                                                std::string* ip) {
  char ethernet_name[64];
  if (util_rdma_ib2eth_name(ib_name, ethernet_name)) {
    return -1;
  }

  *ip = get_dev_ip(ethernet_name);

  return *ip == "" ? -1 : 0;
}

static inline int util_rdma_get_mtu_from_ibv_mtu(ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:
      return 256;
    case IBV_MTU_512:
      return 512;
    case IBV_MTU_1024:
      return 1024;
    case IBV_MTU_2048:
      return 2048;
    case IBV_MTU_4096:
      return 4096;
    default:
      return 0;
  }
}

}  // namespace uccl

#endif
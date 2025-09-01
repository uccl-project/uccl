#include "rdma.hpp"
#include "common.hpp"
#include "peer_copy.cuh"
#include "peer_copy_worker.hpp"
#include "rdma_util.hpp"
#include "util/gpu_rt.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <fcntl.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "util/util.h"
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#define MAX_RETRIES 20
#define RETRY_DELAY_MS 200
#define TCP_PORT 18515
#define QKEY 0x11111111u

void exchange_connection_info(int rank, char const* peer_ip, int tid,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote) {
  int sockfd;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  printf(
      "Rank %d exchanging RDMA connection info with peer %s, "
      "local.addr=0x%lx\n",
      rank, peer_ip, local->addr);
  if (rank == 0) {
    // Listen
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT + tid);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(listenfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      perror("bind failed");
      exit(1);
    }
    listen(listenfd, 1);

    socklen_t len = sizeof(addr);
    sockfd = accept(listenfd, (struct sockaddr*)&addr, &len);
    printf("Rank %d accepted connection from peer %s on port %d\n", rank,
           peer_ip, TCP_PORT + tid);
    close(listenfd);
  } else {
    // Connect
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT + tid);
    inet_pton(AF_INET, peer_ip, &addr.sin_addr);

    int retry = 0;
    while (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      if (errno == ECONNREFUSED || errno == ENETUNREACH) {
        if (++retry > MAX_RETRIES) {
          fprintf(stderr, "Rank %d: failed to connect after %d retries\n", rank,
                  retry);
          exit(1);
        }
        usleep(RETRY_DELAY_MS * 1000);  // sleep 200 ms
        continue;
      } else {
        perror("connect failed");
        exit(1);
      }
    }
    printf("Rank %d connected to peer %s on port %d\n", rank, peer_ip,
           TCP_PORT + tid);
  }

  // Exchange info
  // send(sockfd, local, sizeof(*local), 0);
  // recv(sockfd, remote, sizeof(*remote), MSG_WAITALL);
  uccl::send_message(sockfd, local, sizeof(*local));
  uccl::receive_message(sockfd, remote, sizeof(*remote));
  close(sockfd);

  printf(
      "Rank %d exchanged RDMA info: addr=0x%lx, rkey=0x%x, "
      "qp_num=%u, psn=%u\n",
      rank, remote->addr, remote->rkey, remote->qp_num, remote->psn);
}

void per_thread_rdma_init(ProxyCtx& S, void* gpu_buf, size_t bytes, int rank,
                          int block_idx) {
  printf("Rank %d, Block %d: Initializing RDMA for GPU buffer %p, size %zu\n",
         rank, block_idx, gpu_buf, bytes);
  if (S.context) return;  // already initialized

  struct ibv_device** dev_list = ibv_get_device_list(NULL);
  if (!dev_list) {
    perror("Failed to get IB devices list");
    exit(1);
  }

  // Get GPU idx
  int gpu_idx = 0;
  auto gpu_cards = uccl::get_gpu_cards();
  auto ib_nics = uccl::get_rdma_nics();
  auto gpu_device_path = gpu_cards[gpu_idx];
  auto ib_nic_it = std::min_element(
      ib_nics.begin(), ib_nics.end(), [&](auto const& a, auto const& b) {
        return uccl::cal_pcie_distance(gpu_device_path, a.second) <
               uccl::cal_pcie_distance(gpu_device_path, b.second);
      });
  int selected_idx = ib_nic_it - ib_nics.begin();
  printf("[RDMA] Selected NIC %s for GPU %s\n", ib_nic_it->first.c_str(),
         gpu_device_path.c_str());

  S.context = ibv_open_device(dev_list[selected_idx]);
  if (!S.context) {
    perror("Failed to open device");
    exit(1);
  }
  printf("[RDMA] Selected NIC: %s (index %d)\n",
         ibv_get_device_name(dev_list[selected_idx]), selected_idx);

  ibv_free_device_list(dev_list);

  S.pd = ibv_alloc_pd(S.context);
  if (!S.pd) {
    perror("Failed to allocate PD");
    exit(1);
  }
  uint64_t iova = (uintptr_t)gpu_buf;
#ifndef EFA
  S.mr = ibv_reg_mr_iova2(S.pd, gpu_buf, bytes, iova,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_ATOMIC |
                              IBV_ACCESS_RELAXED_ORDERING);
#else
  S.mr = ibv_reg_mr_iova2(S.pd, gpu_buf, bytes, iova,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_RELAXED_ORDERING);
#endif

  if (!S.mr) {
    perror("ibv_reg_mr failed");
    exit(1);
  }

  if (S.rkey != 0) {
    fprintf(stderr, "Warning: rkey already set (%x), overwriting\n", S.rkey);
  }
  S.rkey = S.mr->rkey;
}

ibv_cq* create_per_thread_cq(ProxyCtx& S) {
  int cq_depth = kMaxOutstandingSends * 2;
#ifdef EFA
  struct ibv_cq_init_attr_ex cq_ex_attr = {};
  cq_ex_attr.cqe = cq_depth;
  cq_ex_attr.cq_context = nullptr;
  cq_ex_attr.channel = nullptr;
  cq_ex_attr.comp_vector = 0;
  // cq_ex_attr.wc_flags =
  //     IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;  // Timestamp support.
  cq_ex_attr.comp_mask = 0;
  cq_ex_attr.flags = 0;
  // EFA requires these values for wc_flags and comp_mask.
  // See `efa_create_cq_ex` in rdma-core.
  cq_ex_attr.wc_flags = IBV_WC_STANDARD_FLAGS;

  S.cq = (struct ibv_cq*)ibv_create_cq_ex(S.context, &cq_ex_attr);
#else
  S.cq =
      ibv_create_cq(S.context, /* cqe */ cq_depth, /* user_context */ nullptr,
                    /* channel */ nullptr, /* comp_vector */ 0);
#endif
  if (!S.cq) {
    perror("Failed to create CQ");
    exit(1);
  }
  return S.cq;
}

struct ibv_qp* create_srd_qp_ex(ProxyCtx& S) {
  struct ibv_qp_init_attr_ex qp_attr_ex = {};
  struct efadv_qp_init_attr efa_attr = {};

  qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;

  qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                              IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                              IBV_QP_EX_WITH_SEND_WITH_IMM;

  qp_attr_ex.cap.max_send_wr = kMaxOutstandingSends * 2;
  qp_attr_ex.cap.max_recv_wr = kMaxOutstandingSends * 2;
  qp_attr_ex.cap.max_send_sge = 1;
  qp_attr_ex.cap.max_recv_sge = 1;
  qp_attr_ex.cap.max_inline_data = 0;

  qp_attr_ex.pd = S.pd;
  qp_attr_ex.qp_context = S.context;
  qp_attr_ex.sq_sig_all = 1;

  qp_attr_ex.send_cq = S.cq;
  qp_attr_ex.recv_cq = S.cq;

  qp_attr_ex.qp_type = IBV_QPT_DRIVER;

  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
#define EFA_QP_LOW_LATENCY_SERVICE_LEVEL 8
  efa_attr.sl = EFA_QP_LOW_LATENCY_SERVICE_LEVEL;
  efa_attr.flags = 0;
  // If set, Receive WRs will not be consumed for RDMA write with imm.
  // efa_attr.flags |= EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

  struct ibv_qp* qp = efadv_create_qp_ex(S.context, &qp_attr_ex, &efa_attr,
                                         sizeof(struct efadv_qp_init_attr));

  if (!qp) {
    perror("Failed to create QP");
    exit(1);
  }

  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qkey = QKEY;
  if (ibv_modify_qp(
          qp, &attr,
          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
    perror("Failed to modify QP to INIT");
    exit(1);
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
    perror("Failed to modify QP to RTR");
    exit(1);
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }

  return qp;
}

void create_per_thread_qp(ProxyCtx& S, void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank) {
  if (S.qp) return;  // Already initialized for this thread
  if (S.ack_qp) return;
  if (S.recv_ack_qp) return;
#ifdef EFA
  S.qp = create_srd_qp_ex(S);
  S.ack_qp = create_srd_qp_ex(S);
  S.recv_ack_qp = create_srd_qp_ex(S);
  if (!S.qp || !S.ack_qp || !S.recv_ack_qp) {
    perror("Failed to create QPs for EFA");
    exit(1);
  }
#else
  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = S.cq;
  qp_init_attr.recv_cq = S.cq;
  qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
  qp_init_attr.cap.max_send_wr =
      kMaxOutstandingSends * 2;  // max outstanding sends
  qp_init_attr.cap.max_recv_wr =
      kMaxOutstandingSends * 2;  // max outstanding recvs
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.sq_sig_all = 0;
  S.qp = ibv_create_qp(S.pd, &qp_init_attr);
  if (!S.qp) {
    perror("Failed to create QP");
    exit(1);
  }
  S.ack_qp = ibv_create_qp(S.pd, &qp_init_attr);
  if (!S.ack_qp) {
    perror("Failed to create Ack QP");
    exit(1);
  }

  S.recv_ack_qp = ibv_create_qp(S.pd, &qp_init_attr);
  if (!S.recv_ack_qp) {
    perror("Failed to create Receive Ack QP");
    exit(1);
  }
#endif

  // Query port
  struct ibv_port_attr port_attr;
  if (ibv_query_port(S.context, 1, &port_attr)) {
    perror("Failed to query port");
    exit(1);
  }
  printf("Local LID: 0x%x\n", port_attr.lid);
  // Fill local connection info
  local_info->qp_num = S.qp->qp_num;
  local_info->ack_qp_num = S.ack_qp->qp_num;
  local_info->recv_ack_qp_num = S.recv_ack_qp->qp_num;
  local_info->lid = port_attr.lid;
  local_info->rkey = S.rkey;
  local_info->addr = reinterpret_cast<uintptr_t>(gpu_buffer);
  // local_info->psn = rand() & 0xffffff;      // random psn
  // local_info->ack_psn = rand() & 0xffffff;  // random ack psn
  local_info->psn = 0;
  local_info->ack_psn = 0;
  // printf("[DEBUG] Rank %d: Registering local buffer addr=0x%lx, size=%zu
  // bytes\n",
  //        rank, local_info->addr, size);
  fill_local_gid(S, local_info);
  printf(
      "Local RDMA info: addr=0x%lx, rkey=0x%x, qp_num=%u, psn=%u, "
      "ack_qp_num=%u, recv_ack_qp_num=%u, ack_psn: %u\n",
      local_info->addr, local_info->rkey, local_info->qp_num, local_info->psn,
      local_info->ack_qp_num, local_info->recv_ack_qp_num, local_info->ack_psn);
}

void modify_qp_to_init(ProxyCtx& S) {
#ifdef EFA
  return;
#endif
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));

  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;  // HCA port you use
  attr.pkey_index = 0;
  attr.qkey = QKEY;
#ifndef EFA
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
#endif
  int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY;
#ifndef EFA
  flags |= IBV_QP_ACCESS_FLAGS;
#endif

  if (ibv_modify_qp(S.qp, &attr, flags)) {
    perror("Failed to modify QP to INIT");
    exit(1);
  }

  if (S.ack_qp) {
    int ret = ibv_modify_qp(S.ack_qp, &attr, flags);
    if (ret) {
      perror("Failed to modify Ack QP to INIT");
      fprintf(stderr, "errno: %d\n", errno);
      exit(1);
    }
  }

  if (S.recv_ack_qp) {
    int ret = ibv_modify_qp(S.recv_ack_qp, &attr, flags);
    if (ret) {
      perror("Failed to modify Receive Ack QP to INIT");
      fprintf(stderr, "errno: %d\n", errno);
      exit(1);
    }
  }

  printf("QP modified to INIT state\n");
}

struct ibv_ah* create_ah(ProxyCtx& S, uint8_t* remote_gid) {
  struct ibv_ah_attr ah_attr = {};
  ah_attr.is_global = 1;  // Enable Global Routing Header (GRH)
  ah_attr.port_num = 1;
  ah_attr.grh.sgid_index = 0;  // Local GID index
  memcpy(&ah_attr.grh.dgid, remote_gid, 16);
  ah_attr.grh.flow_label = 0;
  ah_attr.grh.hop_limit = 255;
  ah_attr.grh.traffic_class = 0;

  struct ibv_ah* ah = ibv_create_ah(S.pd, &ah_attr);
  if (ah == nullptr) {
    perror("Failed to create AH");
    exit(1);
  }
  return ah;
}

void modify_qp_to_rtr(ProxyCtx& S, RDMAConnectionInfo* remote) {
#ifdef EFA
  S.dst_qpn = remote->qp_num;
  S.dst_ack_qpn = remote->recv_ack_qp_num;
  S.dst_ah = create_ah(S, remote->gid);
  return;
#endif

  int is_roce = 0;

  struct ibv_port_attr port_attr;
  if (ibv_query_port(S.context, 1, &port_attr)) {
    perror("Failed to query port");
    exit(1);
  }

  if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
    printf("RoCE detected (Ethernet)\n");
    is_roce = 1;
  } else if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    printf("InfiniBand detected\n");
    is_roce = 0;
  } else {
    printf("Unknown link layer: %d\n", port_attr.link_layer);
    exit(1);
  }

  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = port_attr.active_mtu;
  attr.dest_qp_num = remote->qp_num;
  attr.rq_psn = remote->psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  if (is_roce) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.grh.hop_limit = 1;
    // Fill GID from remote_info
    memcpy(&attr.ah_attr.grh.dgid, remote->gid, 16);
    attr.ah_attr.grh.sgid_index = 1;  // Assume GID index 0
  } else {
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remote->lid;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.static_rate = 0;
    memset(&attr.ah_attr.grh, 0, sizeof(attr.ah_attr.grh));  // Safe
  }

  int flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  printf("Remote LID: 0x%x, QPN: %u, PSN: %u\n", remote->lid, remote->qp_num,
         remote->psn);
  printf("Verifying port state:\n");
  printf("  link_layer: %s\n", (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET)
                                   ? "Ethernet (RoCE)"
                                   : "InfiniBand");
  printf("  port_state: %s\n",
         (port_attr.state == IBV_PORT_ACTIVE) ? "ACTIVE" : "NOT ACTIVE");
  printf("  max_mtu: %d\n", port_attr.max_mtu);
  printf("  active_mtu: %d\n", port_attr.active_mtu);
  printf("  lid: 0x%x\n", port_attr.lid);

  int ret = ibv_modify_qp(S.qp, &attr, flags);
  if (ret) {
    perror("Failed to modify QP to RTR");
    fprintf(stderr, "errno: %d\n", errno);
    exit(1);
  }
  printf("QP modified to RTR state\n");

  if (S.ack_qp) {
    attr.dest_qp_num = remote->recv_ack_qp_num;
    attr.rq_psn = remote->ack_psn;
    ret = ibv_modify_qp(S.ack_qp, &attr, flags);
    if (ret) {
      perror("Failed to modify Ack QP to RTR");
      fprintf(stderr, "errno: %d\n", errno);
      exit(1);
    }
  }

  if (S.recv_ack_qp) {
    attr.dest_qp_num = remote->ack_qp_num;
    attr.rq_psn = remote->ack_psn;  // Use the same PSN for receive ack QP
    ret = ibv_modify_qp(S.recv_ack_qp, &attr, flags);
    if (ret) {
      perror("Failed to modify Receive Ack QP to RTR");
      fprintf(stderr, "errno: %d\n", errno);
      exit(1);
    }
  }
  printf("ACK-QP modified to RTR state\n");
}

void modify_qp_to_rts(ProxyCtx& S, RDMAConnectionInfo* local_info) {
#ifdef EFA
  return;
#endif
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;

#ifdef EFA
  attr.sq_psn = 0;
  if (ibv_modify_qp(S.qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }
  if (ibv_modify_qp(S.ack_qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
    perror("Failed to modify Ack QP to RTS");
    exit(1);
  }
  if (ibv_modify_qp(S.recv_ack_qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
    perror("Failed to modify Receive Ack QP to RTS");
    exit(1);
  }
  printf("QP modified to RTS state\n");
  return;
#endif

  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_info->psn;
  attr.max_rd_atomic = 1;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC |
              IBV_QP_ACCESS_FLAGS;

  if (ibv_modify_qp(S.qp, &attr, flags)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }
  printf("QP modified to RTS state\n");

  attr.sq_psn = local_info->ack_psn;
  int ret = ibv_modify_qp(S.ack_qp, &attr, flags);
  if (ret) {
    perror("Failed to modify Ack QP to RTS");
    fprintf(stderr, "errno: %d\n", errno);
    exit(1);
  }

  ret = ibv_modify_qp(S.recv_ack_qp, &attr, flags);
  if (ret) {
    perror("Failed to modify Receive Ack QP to RTS");
    fprintf(stderr, "errno: %d\n", errno);
    exit(1);
  }
  printf("ACK-QP modified to RTS state\n");
}

void post_receive_buffer_for_imm(ProxyCtx& S) {
  std::vector<ibv_recv_wr> wrs(kMaxOutstandingRecvs);
  std::vector<ibv_sge> sges(kMaxOutstandingRecvs);

  for (size_t i = 0; i < kMaxOutstandingRecvs; ++i) {
    int offset = kNumThBlocks > i ? i : (i % kNumThBlocks);

    sges[i] = {.addr = (uintptr_t)S.mr->addr + offset * kObjectSize,
               .length = kObjectSize,
               .lkey = S.mr->lkey};
    wrs[i] = {.wr_id = make_wr_id(S.tag, static_cast<uint32_t>(i)),
              .next = (i + 1 < kMaxOutstandingRecvs) ? &wrs[i + 1] : nullptr,
              .sg_list = &sges[i],
              .num_sge = 1};
  }

  /* Post the whole chain with ONE verbs call */
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(S.qp, &wrs[0], &bad)) {
    perror("ibv_post_recv");
    abort();
  }
}

void post_rdma_async_batched(ProxyCtx& S, void* buf, size_t num_wrs,
                             std::vector<uint64_t> const& wrs_to_post,
                             std::vector<TransferCmd> const& cmds_to_post,
                             std::vector<std::unique_ptr<ProxyCtx>>& ctxs,
                             int my_rank) {
  if (num_wrs == 0) return;
  if (wrs_to_post.size() != num_wrs || cmds_to_post.size() != num_wrs) {
    fprintf(stderr,
            "Size mismatch (num_wrs=%zu, wr_ids=%zu, "
            "cmds=%zu)\n",
            num_wrs, wrs_to_post.size(), cmds_to_post.size());
    std::abort();
  }

  std::unordered_map<int, std::vector<size_t>> dst_rank_wr_ids;
  for (size_t i = 0; i < num_wrs; ++i) {
    if (cmds_to_post[i].dst_rank == static_cast<uint32_t>(my_rank)) {
      // NOTE(MaoZiming): this should not happen.
      continue;
    } else {
      dst_rank_wr_ids[cmds_to_post[i].dst_rank].push_back(i);
    }
  }
  for (auto& [dst_rank, wr_ids] : dst_rank_wr_ids) {
    if (wr_ids.empty()) continue;

    ProxyCtx* ctx = ctxs[dst_rank].get();
    if (!ctx || !ctx->qp || !ctx->mr) {
      fprintf(stderr, "Destination ctx missing fields for dst=%d\n", dst_rank);
      std::abort();
    }
    const size_t k = wr_ids.size();
#ifdef EFA
    struct ibv_qp_ex* qpx = (struct ibv_qp_ex*)ctx->qp;
    ibv_wr_start(qpx);

    for (size_t j = 0; j < k; ++j) {
      size_t i = wr_ids[j];
      auto const& cmd = cmds_to_post[i];
      wr_ids[j] = wrs_to_post[i];
      qpx->wr_id = wrs_to_post[i];
      qpx->comp_mask = 0;
      qpx->wr_flags = (j + 1 == k) ? IBV_SEND_SIGNALED : 0;

      uint64_t remote_addr = ctx->remote_addr + S.dispatch_recv_data_offset +
                             (cmd.req_rptr ? cmd.req_rptr : 0);

      if (j + 1 == k) {
        ibv_wr_rdma_write_imm(qpx, ctx->remote_rkey, remote_addr,
                              htonl(static_cast<uint32_t>(qpx->wr_id)));
      } else {
        ibv_wr_rdma_write(qpx, ctx->remote_rkey, remote_addr);
      }

      uintptr_t laddr = cmd.req_lptr
                            ? cmd.req_lptr
                            : reinterpret_cast<uintptr_t>(buf) + i * cmd.bytes;
      ibv_wr_set_ud_addr(qpx, ctx->dst_ah, ctx->dst_qpn, QKEY);
      ibv_wr_set_sge(qpx, ctx->mr->lkey, laddr,
                     static_cast<uint32_t>(cmd.bytes));
    }
    int ret = ibv_wr_complete(qpx);
    if (ret) {
      fprintf(stderr, "ibv_wr_complete failed (dst=%d): %s (ret=%d)\n",
              dst_rank, strerror(ret), ret);
      std::abort();
    }
    const uint64_t batch_tail_wr = wr_ids.back();
#else
    const size_t k = wr_ids.size();
    std::vector<ibv_sge> sges(k);
    std::vector<ibv_send_wr> wrs(k);
    for (size_t j = 0; j < k; ++j) {
      size_t i = wr_ids[j];
      auto const& cmd = cmds_to_post[i];
      wr_ids[j] = wrs_to_post[i];
      sges[j].addr = cmd.req_lptr
                         ? cmd.req_lptr
                         : reinterpret_cast<uintptr_t>(buf) + i * cmd.bytes;
      sges[j].length = static_cast<uint32_t>(cmd.bytes);
      sges[j].lkey = ctx->mr->lkey;
      std::memset(&wrs[j], 0, sizeof(wrs[j]));
      wrs[j].sg_list = &sges[j];
      wrs[j].num_sge = 1;
      wrs[j].wr_id = wr_ids[j];
      wrs[j].wr.rdma.remote_addr =
          ctx->remote_addr + S.dispatch_recv_data_offset + cmd.req_rptr;
      wrs[j].wr.rdma.rkey = ctx->remote_rkey;
      wrs[j].opcode = IBV_WR_RDMA_WRITE;
      wrs[j].send_flags = 0;
      wrs[j].next = (j + 1 < k) ? &wrs[j + 1] : nullptr;
    }
    const size_t last = k - 1;
    const uint64_t batch_tail_wr = wr_ids[last];
    wrs[last].send_flags |= IBV_SEND_SIGNALED;
    wrs[last].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wrs[last].imm_data = htonl(static_cast<uint32_t>(batch_tail_wr));

    ibv_send_wr* bad = nullptr;
    int ret = ibv_post_send(ctx->qp, &wrs[0], &bad);
    if (ret) {
      fprintf(stderr, "ibv_post_send failed (dst=%d): %s (ret=%d)\n", dst_rank,
              strerror(ret), ret);
      if (bad)
        fprintf(stderr, "Bad WR at %p (wr_id=%lu)\n", (void*)bad, bad->wr_id);
      std::abort();
    }
#endif
    S.posted.fetch_add(k, std::memory_order_relaxed);
    if (S.wr_id_to_wr_ids.find(batch_tail_wr) != S.wr_id_to_wr_ids.end()) {
      fprintf(stderr,
              "Error: tail wr_id %lu already exists in wr_id_to_wr_ids "
              "(dst_rank=%d)\n",
              batch_tail_wr, dst_rank);
      std::abort();
    }
    S.wr_id_to_wr_ids[batch_tail_wr] = std::move(wr_ids);
  }
}

void local_process_completions(ProxyCtx& S,
                               std::unordered_set<uint64_t>& finished_wrs,
                               std::mutex& finished_wrs_mutex, int thread_idx,
                               ibv_wc* wc, int ne,
                               std::vector<ProxyCtx*>& ctx_by_tag) {
  if (ne == 0) return;
  int send_completed = 0;

  for (int i = 0; i < ne; ++i) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      fprintf(stderr,
              "CQE ERROR wr_id=%llu status=%d(%s) opcode=%d byte_len=%u "
              "vendor_err=0x%x qp_num=0x%x\n",
              (unsigned long long)wc[i].wr_id, wc[i].status,
              ibv_wc_status_str(wc[i].status), wc[i].opcode, wc[i].byte_len,
              wc[i].vendor_err, wc[i].qp_num);
      std::abort();
    }

    switch (wc[i].opcode) {
      case IBV_WC_RDMA_WRITE:
      case IBV_WC_SEND: {
        std::lock_guard<std::mutex> lock(finished_wrs_mutex);
        for (auto const& wr_id : S.wr_id_to_wr_ids[wc[i].wr_id]) {
          finished_wrs.insert(wr_id);
          send_completed++;
        }
        S.wr_id_to_wr_ids.erase(wc[i].wr_id);
      } break;
      case IBV_WC_RECV:
        if (wc[i].wc_flags & IBV_WC_WITH_IMM) {
          const uint32_t slot = wr_slot(wc[i].wr_id);
          uint64_t wr_done = static_cast<uint64_t>(ntohl(wc[i].imm_data));

          // Check if this is an atomic operation ACK (special WR ID format)
          bool is_atomic_wr =
              (wr_done & 0xFFFF000000000000ULL) == 0xa70a000000000000ULL;

          if (is_atomic_wr) {
            // Handle atomic operation ACK separately - no need to track in
            // sequence
            printf("Local thread %d received atomic ACK for WR %lu\n",
                   thread_idx, wr_done);
          } else {
            // Handle regular operation ACK with sequential tracking
            if (!S.has_received_ack || wr_done >= S.largest_completed_wr) {
              S.largest_completed_wr = wr_done;
              S.has_received_ack = true;
            } else {
              S.largest_completed_wr =
                  std::max(S.largest_completed_wr, wr_done);
              S.has_received_ack = true;
            }
          }
          const uint32_t tag = wr_tag(wc[i].wr_id);
          ProxyCtx& S_ack = *ctx_by_tag[tag];
          ibv_sge sge = {
              .addr = reinterpret_cast<uintptr_t>(&S_ack.ack_recv_buf[slot]),
              .length = sizeof(uint64_t),
              .lkey = S_ack.ack_recv_mr->lkey,
          };
          ibv_recv_wr rwr = {};
          ibv_recv_wr* bad = nullptr;
          rwr.wr_id = make_wr_id(tag, slot);
          rwr.sg_list = &sge;
          rwr.num_sge = 1;
          if (ibv_post_recv(S_ack.recv_ack_qp, &rwr, &bad)) {
            perror("ibv_post_recv(repost ACK)");
            std::abort();
          }
        } else {
          std::abort();
        }
        break;

      default:
        break;
    }
  }
  S.completed.fetch_add(send_completed, std::memory_order_relaxed);
}

void local_poll_completions(ProxyCtx& S,
                            std::unordered_set<uint64_t>& finished_wrs,
                            std::mutex& finished_wrs_mutex, int thread_idx,
                            std::vector<ProxyCtx*>& ctx_by_tag) {
  struct ibv_wc wc[kMaxOutstandingSends];
  int ne = 0;
#ifdef EFA
  auto cqx = (struct ibv_cq_ex*)(S.cq);
  struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
  auto ret = ibv_start_poll(cqx, &poll_cq_attr);
  if (ret) return;
  while (1) {
    wc[ne].status = cqx->status;
    wc[ne].wr_id = cqx->wr_id;
    wc[ne].opcode = ibv_wc_read_opcode(cqx);
    wc[ne].wc_flags = ibv_wc_read_wc_flags(cqx);
    wc[ne].imm_data = ibv_wc_read_imm_data(cqx);
    wc[ne].byte_len = ibv_wc_read_byte_len(cqx);
    ne++;
    if (ne >= kMaxOutstandingSends) break;
    ret = ibv_next_poll(cqx);
    if (ret) break;
  }
  ibv_end_poll(cqx);
#else
  ne = ibv_poll_cq(S.cq, kMaxOutstandingSends, wc);
#endif
  // printf("Local poll thread %d polled %d completions\n", thread_idx, ne);
  local_process_completions(S, finished_wrs, finished_wrs_mutex, thread_idx, wc,
                            ne, ctx_by_tag);
}

void poll_cq_dual(ProxyCtx& S, std::unordered_set<uint64_t>& finished_wrs,
                  std::mutex& finished_wrs_mutex, int thread_idx,
                  CopyRingBuffer& g_ring, std::vector<ProxyCtx*>& ctx_by_tag) {
  struct ibv_wc wc[kMaxOutstandingSends];  // batch poll
  int ne = 0;
#ifdef EFA
  auto cqx = (struct ibv_cq_ex*)(S.cq);
  struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
  auto ret = ibv_start_poll(cqx, &poll_cq_attr);
  if (ret) return;
  while (1) {
    wc[ne].status = cqx->status;
    wc[ne].wr_id = cqx->wr_id;
    wc[ne].opcode = ibv_wc_read_opcode(cqx);
    wc[ne].wc_flags = ibv_wc_read_wc_flags(cqx);
    wc[ne].imm_data = ibv_wc_read_imm_data(cqx);
    wc[ne].byte_len = ibv_wc_read_byte_len(cqx);
    ne++;
    if (ne >= kMaxOutstandingSends) break;
    ret = ibv_next_poll(cqx);
    if (ret) break;
  }
  ibv_end_poll(cqx);
#else
  ne = ibv_poll_cq(S.cq, kMaxOutstandingSends, wc);
#endif
  printf("Poll CQ Dual thread %d polled %d completions\n", thread_idx, ne);
  local_process_completions(S, finished_wrs, finished_wrs_mutex, thread_idx, wc,
                            ne, ctx_by_tag);
  remote_process_completions(S, thread_idx, g_ring, ne, wc, ctx_by_tag);
}

void remote_process_completions(ProxyCtx& S, int idx, CopyRingBuffer& g_ring,
                                int ne, ibv_wc* wc,
                                std::vector<ProxyCtx*>& ctx_by_tag) {
  if (ne == 0) return;
  std::unordered_map<uint32_t, std::vector<ibv_recv_wr>> per_tag;
  per_tag.reserve(8);

  std::vector<CopyTask> task_vec;
  task_vec.reserve(ne);
  int nDevices;
  cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    std::abort();
  }
  for (int i = 0; i < ne; ++i) {
    ibv_wc const& cqe = wc[i];
    if (cqe.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "RDMA error: %s\n", ibv_wc_status_str(cqe.status));
      std::abort();
    }
    if (cqe.opcode == IBV_WC_SEND) {
      continue;
    }
    if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM || cqe.opcode == IBV_WC_RECV) {
      const uint32_t tag = wr_tag(cqe.wr_id);
      auto& batch = per_tag[tag];
      ibv_recv_wr wr{};
      S.pool_index = (S.pool_index + 1) % (kRemoteBufferSize / kObjectSize - 1);
      wr.wr_id = make_wr_id(wr_tag(cqe.wr_id), S.pool_index);
      wr.sg_list = nullptr;
      wr.num_sge = 0;
      wr.next = nullptr;
      batch.push_back(wr);

      if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        // NOTE: imm_data is network byte order per verbs; convert:
        uint32_t imm = ntohl(cqe.imm_data);
        int destination_gpu = static_cast<int>(imm % nDevices);
        size_t src_offset = static_cast<size_t>(S.pool_index) * kObjectSize;
        // TODO(MaoZiming): Implement the logic to set dst_ptr
        CopyTask task{.wr_id = imm,
                      .dst_dev = destination_gpu,
                      .src_ptr = static_cast<char*>(S.mr->addr) + src_offset,
                      .dst_ptr = nullptr,
                      .bytes = kObjectSize};
        if (task.dst_ptr && task.bytes) {
          task_vec.push_back(task);
        } else {
          ProxyCtx* peer_ctx = ctx_by_tag[tag];
          remote_send_ack(peer_ctx, peer_ctx->ack_qp, task.wr_id, g_ring.ack_mr,
                          g_ring.ack_buf, idx);
        }
      }
    }
  }
  for (auto& [tag, batch] : per_tag) {
    if (batch.empty()) continue;
    for (size_t i = 0; i + 1 < batch.size(); ++i) {
      batch[i].next = &batch[i + 1];
    }
    ibv_recv_wr* first = batch.data();
    ibv_recv_wr* bad = nullptr;
    ProxyCtx& S_ack = *ctx_by_tag[tag];
    int ret = ibv_post_recv(S_ack.qp, first, &bad);
    if (ret) {
      fprintf(stderr, "ibv_post_recv failed for tag=%u: %s\n", tag,
              strerror(ret));
      std::abort();
    }
  }
  if (!task_vec.empty()) {
    while (!g_ring.pushN(task_vec.data(), task_vec.size())) {
      printf("Remote thread %d: Ring buffer full, retrying...\n", idx);
    }
  }
}

void remote_poll_completions(ProxyCtx& S, int idx, CopyRingBuffer& g_ring,
                             std::vector<ProxyCtx*>& ctx_by_tag) {
  struct ibv_wc wc[kMaxOutstandingRecvs];
  int ne = 0;
#ifdef EFA
  auto cqx = (struct ibv_cq_ex*)(S.cq);
  struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
  auto ret = ibv_start_poll(cqx, &poll_cq_attr);
  if (ret) return;
  while (1) {
    wc[ne].status = cqx->status;
    wc[ne].wr_id = cqx->wr_id;
    wc[ne].opcode = ibv_wc_read_opcode(cqx);
    wc[ne].wc_flags = ibv_wc_read_wc_flags(cqx);
    wc[ne].imm_data = ibv_wc_read_imm_data(cqx);
    wc[ne].byte_len = ibv_wc_read_byte_len(cqx);
    ne++;
    if (ne >= kMaxOutstandingRecvs) break;
    ret = ibv_next_poll(cqx);
    if (ret) break;
  }
  ibv_end_poll(cqx);
#else
  ne = ibv_poll_cq(S.cq, kMaxOutstandingRecvs, wc);
#endif
  remote_process_completions(S, idx, g_ring, ne, wc, ctx_by_tag);
}

void remote_reg_ack_buf(ibv_pd* pd, uint64_t* ack_buf, ibv_mr*& ack_mr) {
  if (ack_mr) return;
  ack_mr = ibv_reg_mr(pd, ack_buf, sizeof(uint64_t) * RECEIVER_BATCH_SIZE,
                      IBV_ACCESS_LOCAL_WRITE);  // host-only

  if (!ack_mr) {
    perror("ibv_reg_mr(ack_buf)");
    std::abort();
  }
}

void remote_send_ack(ProxyCtx* ctx, struct ibv_qp* ack_qp, uint64_t& wr_id,
                     ibv_mr* local_ack_mr, uint64_t* ack_buf, int worker_idx) {
  if (!ack_qp || !local_ack_mr) {
    if (!ack_qp) {
      fprintf(stderr, "QP not initialised\n");
      std::abort();
    }
    if (!local_ack_mr) {
      fprintf(stderr, "ACK MR not initialised\n");
      std::abort();
    }
    fprintf(stderr, "ACK resources not initialised\n");
    std::abort();
  }

  *reinterpret_cast<uint64_t*>(ack_buf) = wr_id;
  ibv_sge sge = {
      .addr = reinterpret_cast<uintptr_t>(ack_buf),
      .length = sizeof(uint64_t),
      .lkey = local_ack_mr->lkey,
  };

#ifdef EFA

  auto qpx = (struct ibv_qp_ex*)ack_qp;
  ibv_wr_start(qpx);

  qpx->wr_flags = IBV_SEND_SIGNALED;
  qpx->wr_id = wr_id;

  ibv_wr_send_imm(qpx, htonl(static_cast<uint32_t>(wr_id)));
  ibv_wr_set_ud_addr(qpx, ctx->dst_ah, ctx->dst_ack_qpn, QKEY);
  ibv_wr_set_sge(qpx, sge.lkey, sge.addr, sge.length);

  auto ret = ibv_wr_complete(qpx);
  if (ret) {
    fprintf(stderr, "ibv_wr_complete(SEND_WITH_IMM) failed: %d (%s)\n", ret,
            strerror(ret));
    std::abort();
  }

#else
  ibv_send_wr wr = {};
  ibv_send_wr* bad = nullptr;
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;  // generate a CQE
  wr.imm_data = htonl(static_cast<uint32_t>(wr_id));

  int ret = ibv_post_send(ack_qp, &wr, &bad);

  if (ret) {  // ret is already an errno value
    fprintf(stderr, "ibv_post_send(SEND_WITH_IMM) failed: %d (%s)\n", ret,
            strerror(ret));  // strerror(ret) gives the text
    if (bad) {
      fprintf(stderr,
              "  first bad WR: wr_id=%llu  opcode=%u  addr=0x%llx  lkey=0x%x\n",
              (unsigned long long)bad->wr_id, bad->opcode,
              (unsigned long long)bad->sg_list[0].addr, bad->sg_list[0].lkey);
    }
    std::abort();
  }
#endif
}

void local_post_ack_buf(ProxyCtx& S, int depth) {
  if (!S.pd || !S.recv_ack_qp) {
    fprintf(stderr,
            "local_post_ack_buf: PD/QP not ready (pd=%p, recv_ack_qp=%p)\n",
            (void*)S.pd, (void*)S.recv_ack_qp);
    std::abort();
  }
  S.ack_recv_buf.resize(static_cast<size_t>(depth), 0);
  S.ack_recv_mr = ibv_reg_mr(S.pd, S.ack_recv_buf.data(),
                             S.ack_recv_buf.size() * sizeof(uint64_t),
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!S.ack_recv_mr) {
    perror("ibv_reg_mr(ack_recv)");
    std::abort();
  }
  for (int i = 0; i < depth; ++i) {
    ibv_sge sge = {
        .addr = reinterpret_cast<uintptr_t>(&S.ack_recv_buf[i]),
        .length = sizeof(uint64_t),
        .lkey = S.ack_recv_mr->lkey,
    };
    ibv_recv_wr rwr = {};
    ibv_recv_wr* bad = nullptr;
    rwr.wr_id = make_wr_id(S.tag, static_cast<uint32_t>(i));
    rwr.sg_list = &sge;
    rwr.num_sge = 1;
    if (ibv_post_recv(S.recv_ack_qp, &rwr, &bad)) {
      perror("ibv_post_recv(ack)");
      std::abort();
    }
  }
}

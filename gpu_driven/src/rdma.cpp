#include "rdma.hpp"
#include "common.hpp"
#include "peer_copy_worker.hpp"
#include "rdma_util.hpp"
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
#include <cuda_runtime.h>
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

void exchange_connection_info(int rank, char const* peer_ip, int tid,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote) {
  int sockfd;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  printf("Rank %d exchanging RDMA connection info with peer %s\n", rank,
         peer_ip);
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
  send(sockfd, local, sizeof(*local), 0);
  recv(sockfd, remote, sizeof(*remote), MSG_WAITALL);
  close(sockfd);

  printf(
      "Rank %d exchanged RDMA info: addr=0x%lx, rkey=0x%x, "
      "qp_num=%u, psn=%u\n",
      rank, remote->addr, remote->rkey, remote->qp_num, remote->psn);
}

void per_thread_rdma_init(ProxyCtx& S, void* gpu_buf, size_t bytes, int rank,
                          int block_idx) {
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
  S.mr = ibv_reg_mr(S.pd, gpu_buf, bytes,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_RELAXED_ORDERING);

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
  S.cq =
      ibv_create_cq(S.context, /* cqe */ cq_depth, /* user_context */ nullptr,
                    /* channel */ nullptr, /* comp_vector */ 0);
  if (!S.cq) {
    perror("Failed to create CQ");
    exit(1);
  }
  return S.cq;
}

void create_per_thread_qp(ProxyCtx& S, void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank) {
  if (S.qp) return;  // Already initialized for this thread
  if (S.ack_qp) return;
  if (S.recv_ack_qp) return;
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
  local_info->psn = rand() & 0xffffff;      // random psn
  local_info->ack_psn = rand() & 0xffffff;  // random ack psn
  fill_local_gid(S, local_info);
  printf(
      "Local RDMA info: addr=0x%lx, rkey=0x%x, qp_num=%u, psn=%u, "
      "ack_qp_num=%u, recv_ack_qp_num=%u, ack_psn: %u\n",
      local_info->addr, local_info->rkey, local_info->qp_num, local_info->psn,
      local_info->ack_qp_num, local_info->recv_ack_qp_num, local_info->ack_psn);
}

void modify_qp_to_init(ProxyCtx& S) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));

  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;  // HCA port you use
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

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

void modify_qp_to_rtr(ProxyCtx& S, RDMAConnectionInfo* remote) {
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
    attr.ah_attr.grh.sgid_index = 0;  // Assume GID index 0
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
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_info->psn;
  attr.max_rd_atomic = 1;

  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

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
    wrs[i] = {.wr_id = i,
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

void post_rdma_async_batched(ProxyCtx& S, void* buf, size_t bytes,
                             size_t num_wrs, std::vector<uint64_t> wrs_to_post,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex) {
  struct ibv_sge sge {
    .addr = (uintptr_t)buf /*+ start_offset * bytes*/,
    .length = (uint32_t)(bytes * num_wrs), .lkey = S.mr->lkey
  };
  uint64_t largest_wr = wrs_to_post.back();
  struct ibv_send_wr wr {};
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr = S.remote_addr /*+ start_offset * bytes*/;
  wr.wr.rdma.rkey = S.remote_rkey;
  wr.wr_id = largest_wr;
  wr.imm_data = largest_wr;
  wr.send_flags = IBV_SEND_SIGNALED;

  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(S.qp, &wr, &bad);
  if (ret) {
    fprintf(stderr, "ibv_post_send failed: %s (ret=%d)\n", strerror(ret), ret);
    if (bad) {
      fprintf(stderr, "Bad WR at address: %p\n", bad);
    }
    exit(1);
  }
  S.posted.fetch_add(num_wrs, std::memory_order_relaxed);
  if (S.wr_id_to_wr_ids.find(largest_wr) != S.wr_id_to_wr_ids.end()) {
    fprintf(stderr, "Error: largest_wr %lu already exists in wr_id_to_wr_ids\n",
            largest_wr);
    exit(1);
  }
  S.wr_id_to_wr_ids[largest_wr] = wrs_to_post;
}

void local_process_completions(ProxyCtx& S,
                               std::unordered_set<uint64_t>& finished_wrs,
                               std::mutex& finished_wrs_mutex, int thread_idx,
                               ibv_wc* wc, int ne) {
  if (ne == 0) return;
  int send_completed = 0;

  assert(S.ack_qp->send_cq == S.cq);
  assert(S.ack_qp->recv_cq == S.cq);
  assert(S.recv_ack_qp->send_cq == S.cq);
  assert(S.recv_ack_qp->recv_cq == S.cq);
  for (int i = 0; i < ne; ++i) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      fprintf(stderr, "CQE error wr_id=%llu status=%s\n",
              (unsigned long long)wc[i].wr_id, ibv_wc_status_str(wc[i].status));
      std::abort();
    }

    switch (wc[i].opcode) {
      case IBV_WC_RDMA_WRITE: {
        std::lock_guard<std::mutex> lock(finished_wrs_mutex);
        for (auto const& wr_id : S.wr_id_to_wr_ids[wc[i].wr_id]) {
          finished_wrs.insert(wr_id);
          send_completed++;
        }
        S.wr_id_to_wr_ids.erase(wc[i].wr_id);
      } break;
      case IBV_WC_RECV:
        if (wc[i].wc_flags & IBV_WC_WITH_IMM) {
          uint64_t slot = static_cast<uint64_t>(wc[i].wr_id);
          uint64_t wr_done = static_cast<uint64_t>(wc[i].imm_data);
          if (!S.has_received_ack || wr_done >= S.largest_completed_wr) {
            S.largest_completed_wr = wr_done;
            S.has_received_ack = true;
          } else {
            fprintf(stderr,
                    "Warning: received ACK for WR %lu, but largest completed "
                    "WR is %lu\n",
                    wr_done, S.largest_completed_wr);
            std::abort();
          }
          ibv_sge sge = {
              .addr = reinterpret_cast<uintptr_t>(&S.ack_recv_buf[slot]),
              .length = sizeof(uint64_t),
              .lkey = S.ack_recv_mr->lkey,
          };
          ibv_recv_wr rwr = {};
          ibv_recv_wr* bad = nullptr;
          rwr.wr_id = static_cast<uint64_t>(slot);
          rwr.sg_list = &sge;
          rwr.num_sge = 1;
          if (ibv_post_recv(S.recv_ack_qp, &rwr, &bad)) {
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
                            std::mutex& finished_wrs_mutex, int thread_idx) {
  struct ibv_wc wc[kMaxOutstandingSends];
  int ne = ibv_poll_cq(S.cq, kMaxOutstandingSends, wc);
  local_process_completions(S, finished_wrs, finished_wrs_mutex, thread_idx, wc,
                            ne);
}

void poll_cq_dual(ProxyCtx& S, std::unordered_set<uint64_t>& finished_wrs,
                  std::mutex& finished_wrs_mutex, int thread_idx,
                  CopyRingBuffer& g_ring) {
  struct ibv_wc wc[kMaxOutstandingSends];  // batch poll
  int ne = ibv_poll_cq(S.cq, kMaxOutstandingSends, wc);
  local_process_completions(S, finished_wrs, finished_wrs_mutex, thread_idx, wc,
                            ne);
  remote_process_completions(S, thread_idx, g_ring, ne, wc);
}

void remote_process_completions(ProxyCtx& S, int idx, CopyRingBuffer& g_ring,
                                int ne, ibv_wc* wc) {
  struct ibv_sge sges[kMaxOutstandingRecvs];
  struct ibv_recv_wr wrs[kMaxOutstandingRecvs];
  if (ne == 0) return;
  int num_wr_imm = 0;
  for (int i = 0; i < ne; ++i) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      fprintf(stderr, "RDMA error: %s\n", ibv_wc_status_str(wc[i].status));
      std::abort();
    }
    if (wc[i].opcode == IBV_WC_SEND) {
      S.send_ack_completed++;
      continue;
    }
    if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      S.pool_index = (S.pool_index + 1) % (kRemoteBufferSize / kObjectSize - 1);
      wrs[num_wr_imm] = {.wr_id = S.pool_index,
                         .next = nullptr,
                         .sg_list = nullptr,
                         .num_sge = 0};
      if (num_wr_imm >= 1) wrs[num_wr_imm - 1].next = &wrs[num_wr_imm];
      num_wr_imm++;
    }
  }
  ibv_recv_wr* bad = nullptr;
  if (num_wr_imm > 0) {
    int ret = ibv_post_recv(S.qp, &wrs[0], &bad);
    if (ret) {
      fprintf(stderr, "ibv_post_recv failed: %s\n", strerror(ret));
      std::abort();
    }
  }

#ifdef ENABLE_PROXY_CUDA_MEMCPY
  std::vector<CopyTask> task_vec;
  task_vec.reserve(num_wr_imm);
  for (int i = 0; i < ne; ++i) {
    int src_addr_offset = 0;
    // int destination_gpu;
    uint32_t destination_addr_offset = 0;
    if (wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
      continue;
    }
    int destination_gpu = wc[i].imm_data % NUM_GPUS;
    if (S.per_gpu_device_buf[destination_gpu] == nullptr) {
      fprintf(stderr, "per_gpu_device_buf[%d] is null\n", destination_gpu);
      std::abort();
    }
    if (wc[i].imm_data > kIterations) {
      fprintf(stderr, "Unexpected imm_data: %u, expected <= %d\n",
              wc[i].imm_data, kIterations);
      std::abort();
    }
    CopyTask task{
        .wr_id = wc[i].imm_data,
        .dst_dev = destination_gpu,
        .src_ptr = static_cast<char*>(S.mr->addr) + src_addr_offset,
        .dst_ptr = static_cast<char*>(S.per_gpu_device_buf[destination_gpu]) +
                   destination_addr_offset,
        .bytes = wc[i].byte_len};
    task_vec.push_back(task);
  }
  if (!task_vec.empty()) {
    while (!g_ring.pushN(task_vec.data(), task_vec.size())) { /* Busy spin. */
    }
  }
#endif
}

void remote_poll_completions(ProxyCtx& S, int idx, CopyRingBuffer& g_ring) {
  struct ibv_wc wc[kMaxOutstandingRecvs];

  assert(S.ack_qp->send_cq == S.cq);
  assert(S.qp->send_cq == S.cq);
  assert(S.recv_ack_qp->send_cq == S.cq);
  assert(S.recv_ack_qp->recv_cq == S.cq);
  int ne = ibv_poll_cq(S.cq, kMaxOutstandingRecvs, wc);
  remote_process_completions(S, idx, g_ring, ne, wc);
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

void remote_send_ack(struct ibv_qp* ack_qp, uint64_t& wr_id,
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

  ibv_send_wr wr = {};
  ibv_send_wr* bad = nullptr;
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;  // generate a CQE
  wr.imm_data = static_cast<uint32_t>(wr_id);

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
}

void local_post_ack_buf(ProxyCtx& S, int depth) {
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
    rwr.wr_id = static_cast<uint64_t>(i);
    rwr.sg_list = &sge;
    rwr.num_sge = 1;
    if (ibv_post_recv(S.recv_ack_qp, &rwr, &bad)) {
      perror("ibv_post_recv(ack)");
      std::abort();
    }
  }
}

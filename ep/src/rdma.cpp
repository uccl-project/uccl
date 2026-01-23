#include "rdma.hpp"
#include "common.hpp"
#include "peer_copy.cuh"
#include "peer_copy_worker.hpp"
#include "proxy_ctx.hpp"
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
#include "bench_utils.hpp"
#include "util/util.h"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

void recv_connection_info_as_server(int my_rank, int* actual_peer,
                                    int listen_fd,
                                    RDMAConnectionInfo* remote_array) {
  int sockfd;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  // Already listening when calling uccl::create_listen_socket().
  socklen_t len = sizeof(addr);
  sockfd = accept(listen_fd, (struct sockaddr*)&addr, &len);

  // Exchange info
  uccl::receive_message(sockfd, actual_peer, sizeof(*actual_peer));
  uccl::receive_message(sockfd, &remote_array[*actual_peer],
                        sizeof(remote_array[*actual_peer]));
  close(sockfd);
}

void send_connection_info_as_client(int my_rank, int peer, char const* peer_ip,
                                    int peer_listen_port,
                                    RDMAConnectionInfo* local) {
  int sockfd;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  // Connect
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(peer_listen_port);
  inet_pton(AF_INET, peer_ip, &addr.sin_addr);

  int retry = 0;
  while (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    if (errno == ECONNREFUSED || errno == ENETUNREACH) {
      if (++retry > MAX_RETRIES) {
        fprintf(stderr, "Rank %d: failed to connect to %d after %d retries\n",
                my_rank, peer, retry);
        exit(1);
      }
      usleep(RETRY_DELAY_MS * 1000);  // sleep 200 ms
      continue;
    } else {
      perror("connect failed");
      exit(1);
    }
  }

  // Exchange info
  uccl::send_message(sockfd, &my_rank, sizeof(my_rank));
  uccl::send_message(sockfd, local, sizeof(*local));
  close(sockfd);
}

void per_thread_rdma_init(ProxyCtx& S, void* gpu_buf, size_t bytes, int rank,
                          int thread_idx, int local_rank) {
  if (S.context) return;  // already initialized

  int num_devices = 0;
  struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list) {
    perror("Failed to get IB devices list");
    exit(1);
  }
  int gpu_idx = local_rank;
  cudaSetDevice(gpu_idx);  // Needed.

  // Ranked by GPU idx
  auto gpu_cards = uccl::get_gpu_cards();
  // Ranked by RDMA NIC name (not the ibv_get_device_list order)
  auto ib_nics = uccl::get_rdma_nics();
  // Get GPU pcie path
  auto gpu_device_path = gpu_cards[gpu_idx];
  // Find the RDMA NIC that is closest to the GPU.
  std::vector<std::pair<std::string, uint32_t>> dist;
  dist.reserve(ib_nics.size());

  // Conforming to UCCL_IB_HCA filter.
  char* ib_hca = getenv("UCCL_IB_HCA");
  if (!ib_hca) {
    ib_hca = getenv("NCCL_IB_HCA");
  }
  struct uccl::ib_dev user_ib_ifs[MAX_IB_DEVS];
  bool searchNot = ib_hca && ib_hca[0] == '^';
  if (searchNot) ib_hca++;
  bool searchExact = ib_hca && ib_hca[0] == '=';
  if (searchExact) ib_hca++;
  int num_ib_ifs = uccl::parse_interfaces(ib_hca, user_ib_ifs, MAX_IB_DEVS);

  std::string selected_nic_name;
  for (auto& nic : ib_nics) {
    if (!(uccl::match_if_list(nic.first.c_str(), 1, user_ib_ifs, num_ib_ifs,
                              searchExact) ^
          searchNot)) {
      continue;
    }
    uint32_t d = uccl::safe_pcie_distance(gpu_device_path, nic.second);
    dist.emplace_back(nic.first, d);
  }

  // Find the NIC with the minimum distance.
  if (dist.empty()) {
    fprintf(stderr, "[WARN] no NIC found, defaulting to empty\n");
    selected_nic_name.clear();
  } else {
    // Find the minimum distance
    auto min_it = std::min_element(
        dist.begin(), dist.end(),
        [](auto const& a, auto const& b) { return a.second < b.second; });
    auto min_d = min_it->second;

    // Collect all NICs with equal minimum distance
    std::vector<std::string> candidates;
#ifdef EFA
    int num_efas = 0;
#endif
    for (auto& p : dist) {
#ifdef EFA
      if (strncmp(p.first.c_str(), "rdmap", 5) == 0) num_efas++;
      if (p.second == min_d && strncmp(p.first.c_str(), "rdmap", 5) == 0)
        candidates.push_back(p.first);
#else
      if (!uccl::is_iface_up(p.first)) continue;
      if (p.second == min_d) candidates.push_back(p.first);
#endif
    }

    if (candidates.empty()) {
      fprintf(stderr, "[WARN] no candidate NIC found, defaulting to first\n");
      selected_nic_name = dist.front().first;
    } else {
      // Spread GPUs across equal-distance NICs: use local GPU index modulo
      // For example, pass in `local_rank` or derive gpu_index from device path
      selected_nic_name = candidates[thread_idx % candidates.size()];
#ifdef EFA
      // NOTE(MaoZiming): This is a temporary hack.
      if (num_efas == 32) {
        assert(candidates.size() == 4);
        // On p5, there are 4 NICs with the same distance.
        selected_nic_name = candidates[thread_idx % 4];
        use_ll_sl = true;
      } else if (num_efas == 16) {
        assert(candidates.size() == 4);
        // On p5e/p5en, there are 4 NICs with the same distance.
        // We hardcode the first half Proxies to use the first NIC, and the
        // second half to use the second NIC.
        auto half = (local_rank % 2) * 2;
        // GPU0 uses candidates[0/1], GPU1 uses candidates[2/3], etc.
        selected_nic_name = candidates[thread_idx % 2 + half];
        use_ll_sl = true;
      } else {
        // On p6-b200, there is 2 NICs with the same distance.
        assert(num_efas == 8);
        assert(candidates.size() == 2);
        auto half = (local_rank % 2) * 1;
        selected_nic_name = candidates[thread_idx % 1 + half];
        use_ll_sl = true;
      }
#endif
    }
  }

  int selected_dev_idx = -1;
  for (int i = 0; i < num_devices; i++) {
    if (strcmp(ibv_get_device_name(dev_list[i]), selected_nic_name.c_str()) ==
        0) {
      selected_dev_idx = i;
      break;
    }
  }
  if (selected_dev_idx < 0) {
    fprintf(stderr,
            "[FATAL] Selected RDMA NIC '%s' not found in verbs device list\n",
            selected_nic_name.c_str());
    std::abort();
  }

  S.context = ibv_open_device(dev_list[selected_dev_idx]);
  if (!S.context) {
    perror("Failed to open device");
    exit(1);
  }
  S.numa_node = uccl::get_dev_numa_node(selected_nic_name.c_str());
  printf("[RDMA] Selected NIC %s (index %d) for GPU %d, NUMA node %d\n",
         selected_nic_name.c_str(), selected_dev_idx, gpu_idx, S.numa_node);
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
                              IBV_ACCESS_REMOTE_ATOMIC);
  // | IBV_ACCESS_RELAXED_ORDERING
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
    exit(1);
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

#ifdef EFA
struct ibv_qp* create_srd_qp_ex(ProxyCtx& S) {
  struct ibv_qp_init_attr_ex qp_attr_ex = {};
  struct efadv_qp_init_attr efa_attr = {};

  qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;

  qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                              IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                              IBV_QP_EX_WITH_SEND_WITH_IMM;

  qp_attr_ex.cap.max_send_wr = kMaxOutstandingSends;
  qp_attr_ex.cap.max_recv_wr = kMaxOutstandingSends;
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
  if (use_ll_sl) efa_attr.sl = EFA_QP_LOW_LATENCY_SERVICE_LEVEL;
  efa_attr.flags = 0;
  // If set, Receive WRs will not be consumed for RDMA write with imm.
  efa_attr.flags |= EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

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
#endif

void create_per_thread_qp(ProxyCtx& S, void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank,
                          size_t num_rings, bool use_normal_mode,
                          void* atomic_buffer_ptr) {
  if (S.qp) return;  // Already initialized for this thread
  if (S.ack_qp) return;
  if (S.recv_ack_qp) return;
#ifdef EFA
  S.qp = create_srd_qp_ex(S);
  S.ack_qp = create_srd_qp_ex(S);
  S.recv_ack_qp = create_srd_qp_ex(S);
#else
  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = S.cq;
  qp_init_attr.recv_cq = S.cq;
  qp_init_attr.qp_type = IBV_QPT_RC;                    // Reliable Connection
  qp_init_attr.cap.max_send_wr = kMaxOutstandingSends;  // max outstanding sends
  qp_init_attr.cap.max_recv_wr = kMaxOutstandingSends;  // max outstanding recvs
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

  if (use_normal_mode) {
    size_t const rings_to_create =
        std::min(num_rings, (size_t)kChannelPerProxy);
    S.data_qps_by_channel.resize(rings_to_create);
    for (size_t r = 0; r < rings_to_create; ++r) {
#ifdef EFA
      S.data_qps_by_channel[r] = create_srd_qp_ex(S);
#else
      S.data_qps_by_channel[r] = ibv_create_qp(S.pd, &qp_init_attr);
#endif
      if (!S.data_qps_by_channel[r]) {
        perror("Failed to create data QP");
        exit(1);
      }
    }

    // Advertise per-ring QPNs (zero-fill the rest for determinism)
    local_info->num_rings = static_cast<uint32_t>(rings_to_create);
    for (uint32_t r = 0; r < local_info->num_rings; ++r) {
      local_info->data_qp_num[r] = S.data_qps_by_channel[r]->qp_num;
    }
    for (uint32_t r = local_info->num_rings; r < kChannelPerProxy; ++r) {
      local_info->data_qp_num[r] = 0;
    }
  }

  // Query port
  struct ibv_port_attr port_attr;
  if (ibv_query_port(S.context, 1, &port_attr)) {
    perror("Failed to query port");
    exit(1);
  }
  ncclIbGetGidIndex(S.context, 1, &port_attr, &S.gid_index);
  local_info->qp_num = S.qp->qp_num;
  local_info->ack_qp_num = S.ack_qp->qp_num;
  local_info->recv_ack_qp_num = S.recv_ack_qp->qp_num;
  local_info->lid = port_attr.lid;
  local_info->rkey = S.rkey;
  local_info->addr = reinterpret_cast<uintptr_t>(gpu_buffer);
  local_info->len = size;
  local_info->psn = 0;
  local_info->ack_psn = 0;

  // Populate atomic buffer info if available
  // Use S.atomic_buffer_mr if it exists (even if atomic_buffer_ptr is nullptr
  // for this thread) This ensures all threads exchange the same atomic buffer
  // info
  if (S.atomic_buffer_mr) {
    local_info->atomic_buffer_rkey = S.atomic_buffer_mr->rkey;
    local_info->atomic_buffer_addr =
        reinterpret_cast<uintptr_t>(S.atomic_buffer_mr->addr);
    local_info->atomic_buffer_len = S.atomic_buffer_mr->length;
    fprintf(stderr,
            "[create_per_thread_qp] Populated atomic buffer info: addr=0x%llx, "
            "len=%zu, rkey=0x%x\n",
            (unsigned long long)local_info->atomic_buffer_addr,
            (size_t)local_info->atomic_buffer_len,
            local_info->atomic_buffer_rkey);
  } else {
    // TODO(MaoZiming): Only for non-EFA case.
    assert(false && "Atomic buffer is not registered");
  }

  fill_local_gid(S, local_info);
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
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  if (ibv_modify_qp(S.qp, &attr, flags)) {
    perror("Failed to modify QP to INIT");
    exit(1);
  }

  for (size_t r = 0; r < S.data_qps_by_channel.size(); ++r) {
    if (ibv_modify_qp(S.data_qps_by_channel[r], &attr, flags)) {
      perror("Failed to modify QP to INIT");
      exit(1);
    }
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

void modify_qp_to_rtr(ProxyCtx& S, RDMAConnectionInfo* remote,
                      bool use_normal_mode) {
#ifdef EFA
  S.dst_qpn = remote->qp_num;
  S.dst_ack_qpn = remote->recv_ack_qp_num;
  S.dst_ah = create_ah(S, remote->gid);
#endif

  if (use_normal_mode) {
    S.dst_data_qpn_by_ring.clear();
    uint32_t const remote_rings =
        std::min(remote->num_rings, (uint32_t)kChannelPerProxy);
    S.dst_data_qpn_by_ring.reserve(remote_rings);
    for (uint32_t r = 0; r < remote_rings; ++r) {
      S.dst_data_qpn_by_ring.push_back(remote->data_qp_num[r]);
    }
  }

#ifdef EFA
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
    char const* sl_env = getenv("UCCL_IB_SL");
    attr.ah_attr.sl = sl_env ? atoi(sl_env) : 3;
    attr.ah_attr.src_path_bits = 0;
    char const* tc_env = getenv("UCCL_IB_TC");
    attr.ah_attr.grh.traffic_class = tc_env ? atoi(tc_env) : 104;
    attr.ah_attr.grh.hop_limit = 255;
    // Fill GID from remote_info
    memcpy(&attr.ah_attr.grh.dgid, remote->gid, 16);
    attr.ah_attr.grh.sgid_index = S.gid_index;
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

  for (size_t r = 0; r < S.data_qps_by_channel.size(); ++r) {
    attr.dest_qp_num = remote->data_qp_num[r];
    if (ibv_modify_qp(S.data_qps_by_channel[r], &attr, flags)) {
      perror("Failed to modify QP to RTR");
      exit(1);
    }
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

  for (size_t r = 0; r < S.data_qps_by_channel.size(); ++r) {
    if (ibv_modify_qp(S.data_qps_by_channel[r], &attr, flags)) {
      perror("Failed to modify QP to RTR");
      exit(1);
    }
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

void post_receive_buffer_for_imm_on_qp(ProxyCtx& S, ibv_qp* qp) {
  std::vector<ibv_recv_wr> wrs(kMaxOutstandingRecvs);
  std::vector<ibv_sge> sges(kMaxOutstandingRecvs);
  for (size_t i = 0; i < kMaxOutstandingRecvs; ++i) {
    size_t offset = (i < kNumProxyThs) ? i : (i % kNumProxyThs);
    sges[i] = {(uintptr_t)S.mr->addr + offset * kObjectSize, kObjectSize,
               S.mr->lkey};
    wrs[i] = {.wr_id = make_wr_id(S.tag, (uint32_t)i),
              .next = (i + 1 < kMaxOutstandingRecvs) ? &wrs[i + 1] : nullptr,
              .sg_list = &sges[i],
              .num_sge = 1};
  }
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(qp, &wrs[0], &bad)) {
    perror("ibv_post_recv");
    abort();
  }
}

void post_receive_buffer_for_imm(ProxyCtx& S) {
  post_receive_buffer_for_imm_on_qp(S, S.qp);  // main QP
  for (auto* q : S.data_qps_by_channel) {
    post_receive_buffer_for_imm_on_qp(S, q);  // per-ring QPs
  }
}

// Normal mode implementation
static void post_rdma_async_batched_normal_mode(
    ProxyCtx& S, void* buf, size_t num_wrs,
    std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post,
    std::vector<std::unique_ptr<ProxyCtx>>& ctxs, int my_rank, int thread_idx) {
  if (num_wrs == 0) return;
  if (wrs_to_post.size() != num_wrs || cmds_to_post.size() != num_wrs) {
    fprintf(stderr, "Size mismatch (num_wrs=%zu, wr_ids=%zu, cmds=%zu)\n",
            num_wrs, wrs_to_post.size(), cmds_to_post.size());
    std::abort();
  }

  std::unordered_map<int, std::vector<size_t>> dst_rank_wr_ids;
  for (size_t i = 0; i < num_wrs; ++i) {
    if (cmds_to_post[i].dst_rank == static_cast<uint32_t>(my_rank)) {
      // NOTE(MaoZiming): this should not happen.
      printf("Posting rdma to itself\n");
      std::abort();
      continue;
    } else if (std::abs((int)cmds_to_post[i].dst_rank - (int)my_rank) %
                   MAX_NUM_GPUS !=
               0) {
      // NOTE(MaoZiming): this should not happen.
      printf("Posting rdma to a different rank\n");
      std::abort();
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
    size_t const k = wr_ids.size();
    std::unordered_map<size_t, std::vector<size_t>> ring_to_indices;
    ring_to_indices.reserve(k);
    for (size_t j = 0; j < k; ++j) {
      size_t i = wr_ids[j];
      size_t ring_idx =
          static_cast<size_t>((wrs_to_post[i] >> 32) & 0xFFFFFFFFu);
      ring_to_indices[ring_idx].push_back(i);
    }

    for (auto& [ring_idx_raw, idxs] : ring_to_indices) {
#ifdef EFA
      const size_t local_ring_count = ctx->data_qps_by_channel.size();
      struct ibv_qp_ex* qpx =
          (struct ibv_qp_ex*)(local_ring_count
                                  ? ctx->data_qps_by_channel[ring_idx_raw %
                                                             local_ring_count]
                                  : ctx->ack_qp);

      size_t const remote_ring_count = ctx->dst_data_qpn_by_ring.size();
      uint32_t const dst_qpn =
          remote_ring_count
              ? ctx->dst_data_qpn_by_ring[ring_idx_raw % remote_ring_count]
              : ctx->dst_qpn;

      ibv_wr_start(qpx);
      // No receiver barrier: build a single chain for this ring group
      std::vector<uint64_t> ring_wrids;
      ring_wrids.reserve(idxs.size());

      for (size_t j = 0; j < idxs.size(); ++j) {
        size_t i = idxs[j];
        auto const& cmd = cmds_to_post[i];

        qpx->wr_id = wrs_to_post[i];
        qpx->comp_mask = 0;
        qpx->wr_flags = IBV_SEND_SIGNALED;

        uint64_t remote_addr =
            ctx->remote_addr + (cmd.req_rptr ? cmd.req_rptr : 0);
        uint64_t remote_end = ctx->remote_addr + ctx->remote_len;

        if (remote_addr < ctx->remote_addr ||
            remote_addr + cmd.bytes > remote_end) {
          fprintf(stderr,
                  "[ERROR] Remote write OOB: addr=0x%llx len=%u (base=0x%llx, "
                  "size=%zu), cmd.req_rptr: 0x%llx\n",
                  (unsigned long long)remote_addr, cmd.bytes,
                  (unsigned long long)ctx->remote_addr, (size_t)ctx->remote_len,
                  (unsigned long long)cmd.req_rptr);
          cudaError_t err = cudaDeviceSynchronize();
          if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                    cudaGetErrorString(err));
          }
          std::abort();
        }
        // Optionally send an inline "atomic" via imm, else use imm only on tail
        if (cmd.atomic_offset > 0 && cmd.atomic_val > 0) {
          int v = static_cast<int>(cmd.atomic_val);
          if (v < -kMaxSendAtomicValue || v > kMaxSendAtomicValue) {
            fprintf(stderr, "[EFA] atomic value=%d won't fit in 15 bits\n", v);
            std::abort();
          }
          size_t index =
              static_cast<size_t>(cmd.atomic_offset / sizeof(int64_t));
          // Initialize missing entries lazily
          auto key = ctx->seq_key(dst_rank, index);
          if (ctx->next_seq_per_index.find(key) ==
              ctx->next_seq_per_index.end())
            ctx->next_seq_per_index[key] = 0;

          uint8_t seq = ctx->next_seq_per_index[key];
          ctx->next_seq_per_index[key] =
              (seq + 1) % kReorderingBufferSize;  // 4-bit wrap (0–15)
          uint32_t imm =
              AtomicsImm::PackAtomicWithSeq(v, cmd.atomic_offset, seq, true)
                  .GetImmData();
          AtomicsImm aimm(imm);
          assert(aimm.GetSeq() == seq);

          ibv_wr_rdma_write_imm(qpx, ctx->remote_rkey, remote_addr, htonl(imm));
        } else if (j + 1 == idxs.size()) {
          uint32_t imm =
              WriteImm::Pack(get_is_combine(cmd.cmd_type),
                             get_low_latency(cmd.cmd_type), cmd.expert_idx,
                             (uint32_t)idxs.size(), my_rank)
                  .GetImmData();
          ibv_wr_rdma_write_imm(qpx, ctx->remote_rkey, remote_addr, htonl(imm));
        } else {
          ibv_wr_rdma_write(qpx, ctx->remote_rkey, remote_addr);
        }

        uintptr_t laddr =
            cmd.req_lptr + reinterpret_cast<uintptr_t>(ctx->mr->addr);
        ibv_wr_set_ud_addr(qpx, ctx->dst_ah, dst_qpn, QKEY);
        ibv_wr_set_sge(qpx, ctx->mr->lkey, laddr,
                       static_cast<uint32_t>(cmd.bytes));

        ring_wrids.push_back(wrs_to_post[i]);
      }
      int ret = ibv_wr_complete(qpx);
      if (ret) {
        fprintf(stderr, "ibv_wr_complete failed (dst=%d): %s (ret=%d)\n",
                dst_rank, strerror(ret), ret);
        std::abort();
      }
#elif defined(SOFTWARE_ORDERING)
      {
        size_t const local_ring_count = ctx->data_qps_by_channel.size();
        struct ibv_qp* qp =
            local_ring_count
                ? ctx->data_qps_by_channel[ring_idx_raw % local_ring_count]
                : ctx->ack_qp;

        size_t const kgroup = idxs.size();
        std::vector<ibv_sge> sges(kgroup);
        std::vector<ibv_send_wr> wrs(kgroup);
        std::vector<uint64_t> ring_wrids;
        ring_wrids.reserve(kgroup);

        for (size_t j = 0; j < kgroup; ++j) {
          size_t i = idxs[j];
          auto const& cmd = cmds_to_post[i];
          ring_wrids.push_back(wrs_to_post[i]);

          // Remote address bounds check
          uint64_t remote_addr =
              ctx->remote_addr + (cmd.req_rptr ? cmd.req_rptr : 0);
          uint64_t remote_end = ctx->remote_addr + ctx->remote_len;

          if (remote_addr < ctx->remote_addr ||
              remote_addr + cmd.bytes > remote_end) {
            fprintf(
                stderr,
                "[ERROR] Remote write OOB: addr=0x%llx len=%u (base=0x%llx, "
                "size=%zu), cmd.req_rptr: 0x%llx\n",
                (unsigned long long)remote_addr, cmd.bytes,
                (unsigned long long)ctx->remote_addr, (size_t)ctx->remote_len,
                (unsigned long long)cmd.req_rptr);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
              fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                      cudaGetErrorString(err));
            }
            std::abort();
          }

          // Local SGE
          uintptr_t laddr =
              cmd.req_lptr + reinterpret_cast<uintptr_t>(ctx->mr->addr);
          sges[j] = {
              .addr = laddr,
              .length = static_cast<uint32_t>(cmd.bytes),
              .lkey = ctx->mr->lkey,
          };

          // Build WR
          std::memset(&wrs[j], 0, sizeof(wrs[j]));
          wrs[j].wr_id = wrs_to_post[i];
          wrs[j].sg_list = &sges[j];
          wrs[j].num_sge = 1;
          wrs[j].wr.rdma.remote_addr = remote_addr;
          wrs[j].wr.rdma.rkey = ctx->remote_rkey;
          wrs[j].opcode = IBV_WR_RDMA_WRITE;  // default
          wrs[j].send_flags = (j + 1 == kgroup) ? IBV_SEND_SIGNALED : 0;
          wrs[j].next = (j + 1 < kgroup) ? &wrs[j + 1] : nullptr;

          if (cmd.atomic_offset > 0 && cmd.atomic_val > 0) {
            int v = static_cast<int>(cmd.atomic_val);
            if (v < -kMaxSendAtomicValue || v > kMaxSendAtomicValue) {
              fprintf(stderr, "atomic value=%d won't fit in 15 bits\n", v);
              std::abort();
            }
            size_t index =
                static_cast<size_t>(cmd.atomic_offset / sizeof(int64_t));
            // Initialize missing entries lazily
            auto key = ctx->seq_key(dst_rank, index);
            if (ctx->next_seq_per_index.find(key) ==
                ctx->next_seq_per_index.end())
              ctx->next_seq_per_index[key] = 0;

            uint8_t seq = ctx->next_seq_per_index[key];
            ctx->next_seq_per_index[key] =
                (seq + 1) % kReorderingBufferSize;  // 4-bit wrap (0–15)
            uint32_t imm =
                AtomicsImm::PackAtomicWithSeq(v, cmd.atomic_offset, seq, true)
                    .GetImmData();
            AtomicsImm aimm(imm);
            assert(aimm.GetSeq() == seq);

            wrs[j].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wrs[j].imm_data = htonl(imm);

            assert(aimm.GetValue() == cmd.atomic_val);
            assert(aimm.GetOff() == cmd.atomic_offset);
          } else {
            wrs[j].opcode = IBV_WR_RDMA_WRITE;
          }
        }

        // Post the chain
        ibv_send_wr* bad = nullptr;
        int ret = ibv_post_send(qp, &wrs[0], &bad);
        if (ret) {
          fprintf(stderr, "ibv_post_send failed (dst=%d): %s (ret=%d)\n",
                  dst_rank, strerror(ret), ret);
          if (bad)
            fprintf(stderr, "Bad WR at %p (wr_id=%lu)\n", (void*)bad,
                    bad->wr_id);
          std::abort();
        }

        // Track wr_id mappings for SOFTWARE_ORDERING
        size_t const last = kgroup - 1;
        uint64_t const batch_tail_wr = ring_wrids[last];
        {
          auto [it, inserted] = S.wr_id_to_wr_ids.try_emplace(
              batch_tail_wr, std::move(ring_wrids));
          if (!inserted) {
            fprintf(stderr,
                    "thread_idx: %d, Error: tail wr_id %lu already exists "
                    "(map=%p)\n",
                    thread_idx, batch_tail_wr, (void*)&S.wr_id_to_wr_ids);
            std::abort();
          }
        }
      }
#else
      {
        size_t const local_ring_count = ctx->data_qps_by_channel.size();
        struct ibv_qp* qp =
            local_ring_count
                ? ctx->data_qps_by_channel[ring_idx_raw % local_ring_count]
                : ctx->ack_qp;

        size_t const kgroup = idxs.size();
        std::vector<ibv_sge> sges(kgroup);
        std::vector<ibv_send_wr> wrs(kgroup);
        std::vector<uint64_t> ring_wrids;
        ring_wrids.reserve(kgroup);

        for (size_t j = 0; j < kgroup; ++j) {
          size_t i = idxs[j];
          auto const& cmd = cmds_to_post[i];
          ring_wrids.push_back(wrs_to_post[i]);

          // Remote address bounds check
          uint64_t remote_addr =
              ctx->remote_addr + (cmd.req_rptr ? cmd.req_rptr : 0);
          uint64_t remote_end = ctx->remote_addr + ctx->remote_len;

          if (remote_addr < ctx->remote_addr ||
              remote_addr + cmd.bytes > remote_end) {
            fprintf(
                stderr,
                "[ERROR] Remote write OOB: addr=0x%llx len=%u (base=0x%llx, "
                "size=%zu), cmd.req_rptr: 0x%llx\n",
                (unsigned long long)remote_addr, cmd.bytes,
                (unsigned long long)ctx->remote_addr, (size_t)ctx->remote_len,
                (unsigned long long)cmd.req_rptr);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
              fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                      cudaGetErrorString(err));
            }
            std::abort();
          }

          // Local SGE
          uintptr_t laddr =
              cmd.req_lptr + reinterpret_cast<uintptr_t>(ctx->mr->addr);
          sges[j] = {
              .addr = laddr,
              .length = static_cast<uint32_t>(cmd.bytes),
              .lkey = ctx->mr->lkey,
          };

          // Build WR
          std::memset(&wrs[j], 0, sizeof(wrs[j]));
          wrs[j].wr_id = wrs_to_post[i];
          wrs[j].sg_list = &sges[j];
          wrs[j].num_sge = 1;
          wrs[j].wr.rdma.remote_addr = remote_addr;
          wrs[j].wr.rdma.rkey = ctx->remote_rkey;
          wrs[j].opcode = IBV_WR_RDMA_WRITE;  // default
          wrs[j].send_flags = (j + 1 == kgroup) ? IBV_SEND_SIGNALED : 0;
          wrs[j].next = (j + 1 < kgroup) ? &wrs[j + 1] : nullptr;

          if (cmd.atomic_offset > 0 && cmd.atomic_val > 0) {
            int v = static_cast<int>(cmd.atomic_val);
            if (v < -kMaxSendAtomicValue || v > kMaxSendAtomicValue) {
              fprintf(stderr, "atomic value=%d won't fit in 15 bits\n", v);
              std::abort();
            }
            uint32_t imm =
                AtomicsImm::Pack(true, false, cmd.atomic_val, cmd.atomic_offset,
                                 get_low_latency(cmd.cmd_type))
                    .GetImmData();
            wrs[j].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wrs[j].imm_data = htonl(imm);

            AtomicsImm aimm(imm);
            assert(aimm.GetValue() == cmd.atomic_val);
            assert(aimm.GetOff() == cmd.atomic_offset);
          } else {
            wrs[j].opcode = IBV_WR_RDMA_WRITE;
          }
        }

        // Post the chain
        ibv_send_wr* bad = nullptr;
        int ret = ibv_post_send(qp, &wrs[0], &bad);
        if (ret) {
          fprintf(stderr, "ibv_post_send failed (dst=%d): %s (ret=%d)\n",
                  dst_rank, strerror(ret), ret);
          if (bad)
            fprintf(stderr, "Bad WR at %p (wr_id=%lu)\n", (void*)bad,
                    bad->wr_id);
          std::abort();
        }
        size_t const last = kgroup - 1;
        uint64_t const batch_tail_wr = ring_wrids[last];
        {
          auto [it, inserted] = S.wr_id_to_wr_ids.try_emplace(
              batch_tail_wr, std::move(ring_wrids));
          if (!inserted) {
            fprintf(stderr,
                    "thread_idx: %d, Error: tail wr_id %lu already exists "
                    "(map=%p)\n",
                    thread_idx, batch_tail_wr, (void*)&S.wr_id_to_wr_ids);
            std::abort();
          }
        }
      }
#endif
    }
  }
}

// Fast mode implementation
static void post_rdma_async_batched_fast_mode(
    ProxyCtx& S, void* buf, size_t num_wrs,
    std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post,
    std::vector<std::unique_ptr<ProxyCtx>>& ctxs, int my_rank, int thread_idx) {
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
      printf("Posting rdma to itself\n");
      std::abort();
      continue;
    } else {
      dst_rank_wr_ids[cmds_to_post[i].dst_rank].push_back(i);
    }
  }
  for (auto& [dst_rank, wr_ids] : dst_rank_wr_ids) {
    if (wr_ids.empty()) continue;

    ProxyCtx* ctx = ctxs[dst_rank].get();
    if (!ctx || !ctx->qp || !ctx->mr) {
      fprintf(stderr, "Destination ctx missing fields for dst=%d, ctx=%p\n",
              dst_rank, (void*)ctx);
      std::abort();
    }
    size_t const k = wr_ids.size();
#ifdef EFA
    struct ibv_qp_ex* qpx = (struct ibv_qp_ex*)ctx->qp;
    ibv_wr_start(qpx);

#ifdef USE_RECEIVER_BARRIER
    std::unordered_map<int, std::vector<size_t>> dst_expert_wr_ids;
    for (size_t j = 0; j < k; ++j) {
      size_t i = wr_ids[j];
      int expert_idx = cmds_to_post[i].expert_idx;
      dst_expert_wr_ids[expert_idx].push_back(i);
    }
#endif

#ifdef USE_RECEIVER_BARRIER
    for (auto& [expert_idx, expert_wr_ids] : dst_expert_wr_ids) {
      size_t expert_k = expert_wr_ids.size();
      for (size_t j = 0; j < expert_k; ++j) {
        size_t i = expert_wr_ids[j];
#else
    for (size_t j = 0; j < k; ++j) {
      size_t i = wr_ids[j];
#endif

        auto const& cmd = cmds_to_post[i];
#ifdef USE_RECEIVER_BARRIER
        expert_wr_ids[j] = wrs_to_post[i];
#else
      wr_ids[j] = wrs_to_post[i];
#endif
        qpx->wr_id = wrs_to_post[i];
        qpx->comp_mask = 0;
        qpx->wr_flags = IBV_SEND_SIGNALED;

        uint64_t remote_addr =
            ctx->remote_addr + (cmd.req_rptr ? cmd.req_rptr : 0);
        uint64_t remote_end = ctx->remote_addr + ctx->remote_len;

        if (remote_addr < ctx->remote_addr ||
            remote_addr + cmd.bytes > remote_end) {
          fprintf(stderr,
                  "[ERROR] Remote write OOB: addr=0x%llx len=%u (base=0x%llx, "
                  "size=%zu), cmd.req_rptr: 0x%llx\n",
                  (unsigned long long)remote_addr, cmd.bytes,
                  (unsigned long long)ctx->remote_addr, (size_t)ctx->remote_len,
                  (unsigned long long)cmd.req_rptr);
          cudaError_t err = cudaDeviceSynchronize();
          if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                    cudaGetErrorString(err));
            std::abort();
          }
          std::abort();
        }
#ifdef USE_SENDER_BARRIER
        S.wr_id_to_write_struct[qpx->wr_id] = {cmd.expert_idx, dst_rank,
                                               get_is_combine(cmd.cmd_type),
                                               get_low_latency(cmd.cmd_type)};
#endif
#ifdef USE_RECEIVER_BARRIER
        uint32_t imm = WriteImm::Pack(get_is_combine(cmd.cmd_type),
                                      get_low_latency(cmd.cmd_type),
                                      cmd.expert_idx, 1, my_rank)
                           .GetImmData();
        ibv_wr_rdma_write_imm(qpx, ctx->remote_rkey, remote_addr, htonl(imm));
#else
      if (j + 1 == k) {
        uint32_t imm = WriteImm::Pack(get_is_combine(cmd.cmd_type),
                                      get_low_latency(cmd.cmd_type),
                                      cmd.expert_idx, k, my_rank)
                           .GetImmData();
        ibv_wr_rdma_write_imm(qpx, ctx->remote_rkey, remote_addr, htonl(imm));
      } else {
        ibv_wr_rdma_write(qpx, ctx->remote_rkey, remote_addr);
      }
#endif
        uintptr_t laddr =
            cmd.req_lptr + reinterpret_cast<uintptr_t>(ctx->mr->addr);
        ibv_wr_set_ud_addr(qpx, ctx->dst_ah, ctx->dst_qpn, QKEY);
        ibv_wr_set_sge(qpx, ctx->mr->lkey, laddr,
                       static_cast<uint32_t>(cmd.bytes));
      }

#ifdef USE_RECEIVER_BARRIER
      uint64_t const expert_tail_wr = expert_wr_ids.back();
      {
        auto [it, inserted] = S.wr_id_to_wr_ids.try_emplace(
            expert_tail_wr, std::move(expert_wr_ids));
        if (!inserted) {
          fprintf(stderr,
                  "thread_idx: %d, Error: tail wr_id %lu already exists "
                  "(map=%p)\n",
                  thread_idx, expert_tail_wr, (void*)&S.wr_id_to_wr_ids);
          std::abort();
        }
      }
    }
#else
    uint64_t const tail_wr = wr_ids.back();
    {
      auto [it, inserted] =
          S.wr_id_to_wr_ids.try_emplace(tail_wr, std::move(wr_ids));
      if (!inserted) {
        fprintf(stderr,
                "thread_idx: %d, Error: tail wr_id %lu already exists "
                "(map=%p)\n",
                thread_idx, tail_wr, (void*)&S.wr_id_to_wr_ids);
        std::abort();
      }
    }
#endif

    int ret = ibv_wr_complete(qpx);
    if (ret) {
      fprintf(stderr, "ibv_wr_complete failed (dst=%d): %s (ret=%d)\n",
              dst_rank, strerror(ret), ret);
      std::abort();
    }
#else
    std::vector<ibv_sge> sges(k);
    std::vector<ibv_send_wr> wrs(k);
    for (size_t j = 0; j < k; ++j) {
      size_t i = wr_ids[j];
      auto const& cmd = cmds_to_post[i];
      wr_ids[j] = wrs_to_post[i];
      sges[j].addr = cmd.req_lptr + reinterpret_cast<uintptr_t>(ctx->mr->addr);
      sges[j].length = static_cast<uint32_t>(cmd.bytes);
      sges[j].lkey = ctx->mr->lkey;
      std::memset(&wrs[j], 0, sizeof(wrs[j]));
      wrs[j].sg_list = &sges[j];
      wrs[j].num_sge = 1;
      wrs[j].wr_id = wr_ids[j];

      wrs[j].wr.rdma.remote_addr = ctx->remote_addr + cmd.req_rptr;

      uint64_t remote_end = ctx->remote_addr + ctx->remote_len;
      if (wrs[j].wr.rdma.remote_addr < ctx->remote_addr ||
          wrs[j].wr.rdma.remote_addr + cmd.bytes > remote_end) {
        fprintf(stderr,
                "[ERROR] Remote write OOB: addr=0x%llx len=%u (base=0x%llx, "
                "size=%zu), cmd.req_rptr: 0x%llx\n",
                (unsigned long long)wrs[j].wr.rdma.remote_addr, cmd.bytes,
                (unsigned long long)ctx->remote_addr, (size_t)ctx->remote_len,
                (unsigned long long)cmd.req_rptr);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                  cudaGetErrorString(err));
          std::abort();
        }
        std::abort();
      }

      wrs[j].wr.rdma.rkey = ctx->remote_rkey;
      wrs[j].opcode = IBV_WR_RDMA_WRITE;
      wrs[j].send_flags = 0;
      wrs[j].next = (j + 1 < k) ? &wrs[j + 1] : nullptr;
    }
    size_t const last = k - 1;
    uint64_t const batch_tail_wr = wr_ids[last];
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
    {
      auto [it, inserted] =
          S.wr_id_to_wr_ids.try_emplace(batch_tail_wr, std::move(wr_ids));
      if (!inserted) {
        fprintf(stderr,
                "thread_idx: %d, Error: tail wr_id %lu already exists "
                "(map=%p)\n",
                thread_idx, batch_tail_wr, (void*)&S.wr_id_to_wr_ids);
        std::abort();
      }
    }
#endif
  }
}

// Wrapper that selects implementation based on use_normal_mode
void post_rdma_async_batched(ProxyCtx& S, void* buf, size_t num_wrs,
                             std::vector<uint64_t> const& wrs_to_post,
                             std::vector<TransferCmd> const& cmds_to_post,
                             std::vector<std::unique_ptr<ProxyCtx>>& ctxs,
                             int my_rank, int thread_idx,
                             bool use_normal_mode) {
  if (use_normal_mode) {
    post_rdma_async_batched_normal_mode(
        S, buf, num_wrs, wrs_to_post, cmds_to_post, ctxs, my_rank, thread_idx);
  } else {
    post_rdma_async_batched_fast_mode(S, buf, num_wrs, wrs_to_post,
                                      cmds_to_post, ctxs, my_rank, thread_idx);
  }
}

void local_process_completions(ProxyCtx& S,
                               std::unordered_set<uint64_t>& acked_wrs,
                               int thread_idx, ibv_wc* wc, int ne,
                               std::vector<ProxyCtx*>& ctx_by_tag) {
  if (ne == 0) return;
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
      case IBV_WC_SEND:
      case IBV_WC_RDMA_WRITE: {
        uint64_t wrid = wc[i].wr_id;
        if ((wrid & kAtomicWrTag) == kAtomicWrTag) {
          wrid &= kAtomicMask;
#ifdef EFA
          acked_wrs.insert(wrid);
#else
          auto it = S.wr_id_to_wr_ids.find(wrid);
          if (it != S.wr_id_to_wr_ids.end()) {
            for (uint64_t sub_wr : it->second) {
              acked_wrs.insert(sub_wr);
            }
            S.wr_id_to_wr_ids.erase(it);
          } else {
            printf("Error: Atomic ACK for unknown wr_id %lu\n", wrid);
            std::abort();
          }
#endif
          break;
        }
        if ((wrid & kBarrierWrTag) == kBarrierWrTag) {
          break;
        }
#ifdef USE_SENDER_BARRIER
        {
          auto it = S.wr_id_to_write_struct.find(wrid);
          if (it != S.wr_id_to_write_struct.end()) {
            WriteStruct const& ws = it->second;
            S.wr_id_to_write_struct.erase(it);
            if (ws.is_combine) {
              S.combine_sent_counter.Add(
                  {ws.low_latency_buffer_idx, ws.expert_idx, ws.dst_rank}, 1);
            } else {
              S.dispatch_sent_counter.Add(
                  {ws.low_latency_buffer_idx, ws.expert_idx, ws.dst_rank}, 1);
            }
          } else {
            assert(false && "wr_id not found in write_struct map");
          }
        }
#endif
        {
          uint64_t const wr_done = wc[i].wr_id;
#ifdef EFA
          acked_wrs.insert(wr_done);
#else
          auto it = S.wr_id_to_wr_ids.find(wr_done);
          if (it != S.wr_id_to_wr_ids.end()) {
            for (uint64_t sub_wr : it->second) {
              acked_wrs.insert(sub_wr);
            }
            S.wr_id_to_wr_ids.erase(it);
          } else {
            printf("Error: Write ACK for unknown wr_id %lu\n", wr_done);
            std::abort();
          }
#endif
        }
      } break;
      case IBV_WC_RECV:
        if (wc[i].wc_flags & IBV_WC_WITH_IMM &&
            !((ntohl(wc[i].imm_data) >> 31) & 0x1)) {
          assert(false && "Explicit Ack is deprecated on local proxy");
        }
        break;
      case IBV_WC_FETCH_ADD: {
        uint64_t wrid = wc[i].wr_id;
        auto it = S.wr_id_to_wr_ids.find(wrid);
        if (it != S.wr_id_to_wr_ids.end()) {
          for (uint64_t sub_wr : it->second) {
            acked_wrs.insert(sub_wr);
          }
          S.wr_id_to_wr_ids.erase(it);
        } else {
          fprintf(stderr,
                  "[Atomic] No batch found for wr_id=0x%lx, treating as single "
                  "(map_size=%zu)\n",
                  wrid, S.wr_id_to_wr_ids.size());
        }
      } break;
      default:
        break;
    }
  }
}

int poll_cq_once(ibv_cq* cq, ibv_wc* wc, int max_cqes) {
#ifdef EFA
  auto cqx = reinterpret_cast<ibv_cq_ex*>(cq);
  ibv_poll_cq_attr attr{.comp_mask = 0};
  if (ibv_start_poll(cqx, &attr)) return 0;

  int n = 0;
  while (n < max_cqes) {
    wc[n].status = cqx->status;
    wc[n].wr_id = cqx->wr_id;
    wc[n].opcode = ibv_wc_read_opcode(cqx);
    wc[n].wc_flags = ibv_wc_read_wc_flags(cqx);
    wc[n].imm_data = ibv_wc_read_imm_data(cqx);
    wc[n].byte_len = ibv_wc_read_byte_len(cqx);
    ++n;
    if (ibv_next_poll(cqx)) break;
  }
  ibv_end_poll(cqx);
  return n;
#else
  return ibv_poll_cq(cq, max_cqes, wc);
#endif
}

void local_poll_completions(ProxyCtx& S,
                            std::unordered_set<uint64_t>& acked_wrs,
                            int thread_idx,
                            std::vector<ProxyCtx*>& ctx_by_tag) {
  ibv_wc wc[kMaxOutstandingSends];
  auto poll_one = [&](ibv_cq* cq) {
    int ne = poll_cq_once(cq, wc, kMaxOutstandingSends);
    if (ne > 0) {
      local_process_completions(S, acked_wrs, thread_idx, wc, ne, ctx_by_tag);
    }
  };
  if (S.cq) poll_one(S.cq);
  // for (auto* cq : S.extra_cqs) poll_one(cq);
}

void poll_cq_dual(ProxyCtx& S, std::unordered_set<uint64_t>& acked_wrs,
                  int thread_idx, CopyRingBuffer& g_ring,
                  std::vector<ProxyCtx*>& ctx_by_tag, void* atomic_buffer_ptr,
                  int num_ranks, int num_experts,
                  std::set<PendingUpdate>& pending_atomic_updates, int my_rank,
                  int num_nodes, bool use_normal_mode) {
  ibv_wc wc[kMaxOutstandingSends];
  auto poll_one = [&](ibv_cq* cq) {
    int ne = poll_cq_once(cq, wc, kMaxOutstandingSends);
    if (ne > 0) {
      local_process_completions(S, acked_wrs, thread_idx, wc, ne, ctx_by_tag);
      remote_process_completions(S, thread_idx, g_ring, ne, wc, ctx_by_tag,
                                 atomic_buffer_ptr, num_ranks, num_experts,
                                 pending_atomic_updates, my_rank, num_nodes,
                                 use_normal_mode);
    }
  };
  if (S.cq) poll_one(S.cq);
  // for (auto* cq : S.extra_cqs) poll_one(cq);
}

void apply_pending_updates(ProxyCtx& ctx,
                           std::set<PendingUpdate>& pending_atomic_updates,
                           void* atomic_buffer_ptr, int num_experts,
                           int num_ranks) {
  for (auto it = pending_atomic_updates.begin();
       it != pending_atomic_updates.end();) {
    PendingUpdate const& upd = *it;
    bool is_atomic_ready = false;
    int value = upd.value;
    if (!upd.is_combine) {
      int num_tokens = ctx.dispatch_token_counter.Get(
          {upd.low_latency_buffer_idx, upd.expert_idx, upd.src_rank});
      if ((-value - 1) == num_tokens) {
        is_atomic_ready = true;
        ctx.dispatch_token_counter.Reset(
            {upd.low_latency_buffer_idx, upd.expert_idx, upd.src_rank});
      }
    } else {
      int combine_num_tokens = ctx.combine_token_counter.Get(
          {upd.low_latency_buffer_idx, upd.expert_idx});
      if (value == combine_num_tokens) {
        is_atomic_ready = true;
        ctx.combine_token_counter.Reset(
            {upd.low_latency_buffer_idx, upd.expert_idx});
      }
    }
    if (is_atomic_ready) {
      if (upd.is_combine) value = 1;
      upd.addr->fetch_add(value, std::memory_order_release);
      it = pending_atomic_updates.erase(it);
    } else {
      ++it;
    }
  }
}

ibv_qp* qp_from_qpnum(ProxyCtx& S, uint32_t qpnum) {
  if (S.qp && S.qp->qp_num == qpnum) return S.qp;
  if (S.recv_ack_qp && S.recv_ack_qp->qp_num == qpnum) return S.recv_ack_qp;
  if (S.ack_qp && S.ack_qp->qp_num == qpnum) return S.ack_qp;
  for (auto* q : S.data_qps_by_channel)
    if (q && q->qp_num == qpnum) return q;
  return nullptr;
}

void remote_process_completions_normal_mode(
    ProxyCtx& S, int idx, CopyRingBuffer& g_ring, int ne, ibv_wc* wc,
    std::vector<ProxyCtx*>& ctx_by_tag, void* atomic_buffer_ptr, int num_ranks,
    int num_experts, std::set<PendingUpdate>& pending_atomic_updates,
    int my_rank, int num_nodes) {
  if (ne == 0) return;
  std::unordered_map<uint32_t, std::vector<ibv_recv_wr>> per_tag;
  per_tag.reserve(8);

  for (int i = 0; i < ne; ++i) {
    ibv_wc const& cqe = wc[i];
    if (cqe.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "RDMA error: %s\n", ibv_wc_status_str(cqe.status));
      std::abort();
    }
    if (cqe.opcode == IBV_WC_SEND) {
      continue;
    }
    if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
        ImmType::IsAtomics(ntohl(cqe.imm_data))) {
      AtomicsImm aimm(ntohl(cqe.imm_data));
      int value = aimm.GetValue();
      uint32_t offset = aimm.GetOff();
      size_t index = offset / sizeof(int64_t);
      auto* addr64 =
          reinterpret_cast<std::atomic<int64_t>*>(atomic_buffer_ptr) + index;

      if (value == kMaxSendAtomicValue) value = kLargeAtomicValue;

      if (!aimm.IsReorderable()) {
        addr64->fetch_add(value, std::memory_order_release);
      } else {
#ifndef EFA
        assert(false &&
               "Reorderable atomic operations should not be triggered");
#endif
        struct SeqBuf {
          uint8_t expected = 0;       // next seq expected
          uint16_t present_mask = 0;  // bitmask of buffered seqs
          int vals[kReorderingBufferSize] = {0};
        };

        // Thread-local map to maintain per-index state
        static thread_local std::unordered_map<size_t, SeqBuf> seqbufs;
        auto& sb = seqbufs[index];

        auto commit = [&](int delta) {
          addr64->fetch_add(delta, std::memory_order_release);
        };
        uint8_t seq = aimm.GetSeq();
        if (seq >= kReorderingBufferSize) {
          fprintf(stderr, "Error: seq %u out of range\n", seq);
          std::abort();
        }
        if (seq == sb.expected) {
          commit(value);
          sb.expected = (sb.expected + 1) % kReorderingBufferSize;

          // Drain buffered consecutive entries
          for (int step = 0; step < kReorderingBufferSize; ++step) {
            uint8_t e = sb.expected;
            uint16_t bit = static_cast<uint16_t>(1u << e);
            if (!(sb.present_mask & bit)) break;
            commit(sb.vals[e]);
            sb.present_mask &= static_cast<uint16_t>(~bit);
            sb.expected = (sb.expected + 1) % kReorderingBufferSize;
          }
        } else {
          // Out-of-order arrival — buffer it
          if (seq >= kReorderingBufferSize) {
            fprintf(stderr, "Error: seq %u out of range\n", seq);
            std::abort();
          }
          if (sb.present_mask & (1u << seq)) {
            fprintf(stderr, "Error: duplicate seq %u arrival\n", seq);
            std::abort();
          }
          uint16_t bit = static_cast<uint16_t>(1u << seq);
          if (sb.present_mask & bit) {
            // Duplicate (possible with UD/SRD). Ignore safely.
            // If you prefer strictness, keep the abort here.
            fprintf(stderr, "Error: duplicate seq %u arrival\n", seq);
            std::abort();
          } else {
            sb.present_mask |= bit;
            sb.vals[seq] = value;
          }
        }
      }
    } else if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
               ImmType::IsBarrier(ntohl(cqe.imm_data))) {
      BarrierImm bimm(ntohl(cqe.imm_data));
      bool is_ack = bimm.GetIsAck();
      uint32_t seq = bimm.GetSeq();
      uint16_t src = bimm.GetRank();
      // First node.
      // TODO(MaoZiming): pass node_idx instead.
#ifdef USE_SUBSET_BARRIER
      if (my_rank < MAX_NUM_GPUS) {
#else
      if (my_rank == 0) {
#endif
        if (!is_ack) {
          if (S.barrier_arrived.empty()) {
            assert(S.barrier_arrival_count == 0 &&
                   "Barrier arrival count should be 0");
            S.barrier_arrived.resize((size_t)num_nodes, 0);
          }
          int num_ranks_per_node = num_ranks / num_nodes;
          size_t node_rank = src / num_ranks_per_node;
          if (node_rank < S.barrier_arrived.size() &&
              !S.barrier_arrived[node_rank]) {
            S.barrier_arrived[node_rank] = 1;
            ++S.barrier_arrival_count;
          } else {
            assert(false &&
                   "Duplicate barrier arrival or local_rank out of range");
          }
        } else {
          assert(false && "Rank 0 should not receive barrier ack");
        }
      } else {
        if (is_ack) {
          S.barrier_released = true;
          S.barrier_release_seq = seq;
        } else {
          assert(false && "Non-leader rank should not receive barrier request");
        }
      }
    } else if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
               ImmType::IsWrite(ntohl(cqe.imm_data))) {
    } else if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      fprintf(stderr, "Unexpected CQE opcode: %d\n", cqe.opcode);
      std::abort();
    }

#ifndef EFA
    if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      uint32_t const tag = wr_tag(cqe.wr_id);
      if (tag >= ctx_by_tag.size() || ctx_by_tag[tag] == nullptr) {
        fprintf(stderr, "Invalid tag or uninitialized context for tag=%u\n",
                tag);
        std::abort();
      }
      ProxyCtx& S = *ctx_by_tag[tag];
      ibv_qp* qp = qp_from_qpnum(S, cqe.qp_num);
      if (!qp) {
        fprintf(stderr, "No matching QP for qp_num=0x%x (tag=%u)\n", cqe.qp_num,
                tag);
        std::abort();
      }
      ibv_sge sge = {
          .addr = reinterpret_cast<uintptr_t>(&S.ack_recv_buf[0]),
          .length = sizeof(uint64_t),
          .lkey = S.ack_recv_mr->lkey,
      };
      ibv_recv_wr rwr{};
      S.pool_index = (S.pool_index + 1) % (kRemoteBufferSize / kObjectSize - 1);
      rwr.wr_id = make_wr_id(wr_tag(cqe.wr_id), S.pool_index);
      rwr.sg_list = &sge;
      rwr.num_sge = 1;
      ibv_recv_wr* bad = nullptr;
      int ret = ibv_post_recv(qp, &rwr, &bad);
      if (ret) {
        fprintf(stderr,
                "ibv_post_recv (imm replenish) failed on qp=0x%x: %s (%d)\n",
                qp->qp_num, strerror(ret), ret);
        if (bad)
          fprintf(stderr, "  bad wr_id=%llu\n", (unsigned long long)bad->wr_id);
        std::abort();
      }
    }
#endif
  }
}

void remote_process_completions_fast_mode(
    ProxyCtx& S, int idx, CopyRingBuffer& g_ring, int ne, ibv_wc* wc,
    std::vector<ProxyCtx*>& ctx_by_tag, void* atomic_buffer_ptr, int num_ranks,
    int num_experts, std::set<PendingUpdate>& pending_atomic_updates,
    int my_rank, int num_nodes) {
  if (ne == 0) return;
  std::unordered_map<uint32_t, std::vector<ibv_recv_wr>> per_tag;
  per_tag.reserve(8);

  for (int i = 0; i < ne; ++i) {
    ibv_wc const& cqe = wc[i];
    if (cqe.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "RDMA error: %s\n", ibv_wc_status_str(cqe.status));
      std::abort();
    }
    if (cqe.opcode == IBV_WC_SEND) {
      continue;
    }
    if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
        ImmType::IsAtomics(ntohl(cqe.imm_data))) {
      AtomicsImm aimm(ntohl(cqe.imm_data));
      int value = aimm.GetValue();
      uint32_t offset = aimm.GetOff();
      size_t index = offset / sizeof(int64_t);
#ifdef USE_RECEIVER_BARRIER
      // ep_config.hpp
      bool is_combine = aimm.IsCombine();
      int low_latency_buffer_idx = aimm.GetBufferIdx();
      uint32_t new_offset =
          offset - low_latency_buffer_idx *
                       align<size_t>(num_experts * sizeof(int64_t), 128);
      size_t new_index = new_offset / sizeof(int64_t);
      int src_rank = -1;
      bool is_atomic_ready = false;
      int expert_idx = -1;
      if (!is_combine) {
        expert_idx = new_index / num_ranks;
        src_rank = new_index % num_ranks;
        int num_tokens = S.dispatch_token_counter.Get(
            {low_latency_buffer_idx, expert_idx, src_rank});
        if ((-value - 1) == num_tokens) {
          is_atomic_ready = true;
        }
        if ((-value - 1) < num_tokens) {
          fprintf(stderr,
                  "[Error] Required Dispatch value %d is smaller than received "
                  "counter %d for "
                  "expert_idx %d, src_rank %d\n",
                  -value - 1, num_tokens, expert_idx, src_rank);
        }
        if (is_atomic_ready) {
          S.dispatch_token_counter.Reset(
              {low_latency_buffer_idx, expert_idx, src_rank});
        }

      } else {
        expert_idx = new_index;
        if (expert_idx > num_experts) {
          fprintf(stderr,
                  "Error: expert_idx %d out of range (num_experts=%d)\n",
                  expert_idx, num_experts);
          std::abort();
        }
        int combine_num_tokens =
            S.combine_token_counter.Get({low_latency_buffer_idx, expert_idx});
        if (value == combine_num_tokens) {
          is_atomic_ready = true;
        }
        if (value < combine_num_tokens) {
          fprintf(stderr,
                  "[Error] Required Combine value %d is smaller than received "
                  "counter %d for "
                  "expert_idx %d\n",
                  value, combine_num_tokens, expert_idx);
        }
        if (is_atomic_ready) {
          S.combine_token_counter.Reset({low_latency_buffer_idx, expert_idx});
        }
      }
      auto* addr64 =
          reinterpret_cast<std::atomic<int64_t>*>(atomic_buffer_ptr) + index;
      if (is_atomic_ready) {
        if (is_combine) value = 1;
        addr64->fetch_add(value, std::memory_order_release);
      } else {
        pending_atomic_updates.insert({addr64, value, aimm.GetImmData(),
                                       low_latency_buffer_idx, expert_idx,
                                       is_combine, src_rank});
      }
#else
      auto* addr64 =
          reinterpret_cast<std::atomic<int64_t>*>(atomic_buffer_ptr) + index;
#ifdef USE_SENDER_BARRIER
      if (aimm.IsCombine()) value = 1;
#ifndef EFA
      const uint32_t tag = wr_tag(cqe.wr_id);
      ProxyCtx& S_atomic = *ctx_by_tag[tag];
      ibv_sge sge = {
          .addr = reinterpret_cast<uintptr_t>(S_atomic.mr->addr),
          .length = 1,
          .lkey = S_atomic.mr->lkey,
      };
      ibv_recv_wr rwr = {};
      S.pool_index = (S.pool_index + 1) % (kRemoteBufferSize / kObjectSize - 1);
      rwr.wr_id = make_wr_id(wr_tag(cqe.wr_id), S.pool_index);
      rwr.sg_list = &sge;
      rwr.num_sge = 1;
      ibv_recv_wr* bad = nullptr;
      if (ibv_post_recv(S_atomic.qp, &rwr, &bad)) {
        perror("ibv_post_recv (atomics replenish)");
        std::abort();
      }
      continue;
#endif
#endif
      if (value == kMaxSendAtomicValue) value = kLargeAtomicValue;
      bool is_combine = aimm.IsCombine();
      assert(!is_combine || value >= 0);
      if (is_combine) {
        assert(value >= 0 && "Combine atomic value should be non-negative");
      } else {
        assert(value <= -1 && "Dispatch atomic value should be <= -1");
      }
      if (is_combine) value = 1;
      addr64->fetch_add(value, std::memory_order_release);
#endif
    } else if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
               ImmType::IsBarrier(ntohl(cqe.imm_data))) {
      BarrierImm bimm(ntohl(cqe.imm_data));
      bool is_ack = bimm.GetIsAck();
      uint32_t seq = bimm.GetSeq();
      uint16_t src = bimm.GetRank();
      if (my_rank == 0) {
        if (!is_ack) {
          if (S.barrier_arrived.empty()) {
            assert(S.barrier_arrival_count == 0 &&
                   "Barrier arrival count should be 0");
            S.barrier_arrived.resize((size_t)num_nodes, 0);
          }
          int num_ranks_per_node = num_ranks / num_nodes;
          size_t node_rank = src / num_ranks_per_node;
          if (node_rank < S.barrier_arrived.size() &&
              !S.barrier_arrived[node_rank]) {
            S.barrier_arrived[node_rank] = 1;
            ++S.barrier_arrival_count;
          } else {
            assert(false &&
                   "Duplicate barrier arrival or local_rank out of range");
          }
        } else {
          assert(false && "Rank 0 should not receive barrier ack");
        }
      } else {
        if (is_ack) {
          S.barrier_released = true;
          S.barrier_release_seq = seq;
        } else {
          assert(false && "Non-leader rank should not receive barrier request");
        }
      }
    } else if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
               ImmType::IsWrite(ntohl(cqe.imm_data))) {
#ifdef USE_RECEIVER_BARRIER
      uint32_t imm = ntohl(cqe.imm_data);
      WriteImm wimm(imm);
      bool is_combine = wimm.IsCombine();
      uint32_t buffer_idx = wimm.GetBufferIdx();
      uint32_t expert_idx = wimm.GetExpertIdx();
      uint32_t k = wimm.GetNumTokens();
      uint32_t src_rank = wimm.GetRank();

      if (!is_combine) {
        /* expert_idx here is the local expert index of the receiver. */
        S.dispatch_token_counter.Add({buffer_idx, expert_idx, src_rank}, k);
      } else {
        /* expert_idx here is the global expert index of the sender. */
        assert(expert_idx >= src_rank * (num_experts / num_ranks) &&
               expert_idx < (src_rank + 1) * (num_experts / num_ranks));
        S.combine_token_counter.Add({buffer_idx, expert_idx}, k);
      }
#endif
    } else if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      fprintf(stderr, "Unexpected CQE opcode: %d\n", cqe.opcode);
      std::abort();
    }
#ifndef EFA
    if (cqe.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      uint32_t const tag = wr_tag(cqe.wr_id);
      if (tag >= ctx_by_tag.size() || ctx_by_tag[tag] == nullptr) {
        fprintf(stderr, "Invalid tag or uninitialized context for tag=%u\n",
                tag);
        std::abort();
      }
      ProxyCtx& S = *ctx_by_tag[tag];
      ibv_qp* qp = qp_from_qpnum(S, cqe.qp_num);
      if (!qp) {
        fprintf(stderr, "No matching QP for qp_num=0x%x (tag=%u)\n", cqe.qp_num,
                tag);
        std::abort();
      }
      ibv_sge sge = {
          .addr = reinterpret_cast<uintptr_t>(&S.ack_recv_buf[0]),
          .length = sizeof(uint64_t),
          .lkey = S.ack_recv_mr->lkey,
      };
      ibv_recv_wr rwr{};
      S.pool_index = (S.pool_index + 1) % (kRemoteBufferSize / kObjectSize - 1);
      rwr.wr_id = make_wr_id(wr_tag(cqe.wr_id), S.pool_index);
      rwr.sg_list = &sge;
      rwr.num_sge = 1;
      ibv_recv_wr* bad = nullptr;
      int ret = ibv_post_recv(qp, &rwr, &bad);
      if (ret) {
        fprintf(stderr,
                "ibv_post_recv (imm replenish) failed on qp=0x%x: %s (%d)\n",
                qp->qp_num, strerror(ret), ret);
        if (bad)
          fprintf(stderr, "  bad wr_id=%llu\n", (unsigned long long)bad->wr_id);
        std::abort();
      }
    }
#endif
  }
}

void remote_process_completions(
    ProxyCtx& S, int idx, CopyRingBuffer& g_ring, int ne, ibv_wc* wc,
    std::vector<ProxyCtx*>& ctx_by_tag, void* atomic_buffer_ptr, int num_ranks,
    int num_experts, std::set<PendingUpdate>& pending_atomic_updates,
    int my_rank, int num_nodes, bool use_normal_mode) {
  if (use_normal_mode) {
    remote_process_completions_normal_mode(
        S, idx, g_ring, ne, wc, ctx_by_tag, atomic_buffer_ptr, num_ranks,
        num_experts, pending_atomic_updates, my_rank, num_nodes);
  } else {
    remote_process_completions_fast_mode(
        S, idx, g_ring, ne, wc, ctx_by_tag, atomic_buffer_ptr, num_ranks,
        num_experts, pending_atomic_updates, my_rank, num_nodes);
  }
}

void remote_poll_completions(ProxyCtx& S, int idx, CopyRingBuffer& g_ring,
                             std::vector<ProxyCtx*>& ctx_by_tag,
                             void* atomic_buffer_ptr, int num_ranks,
                             int num_experts,
                             std::set<PendingUpdate>& pending_atomic_updates,
                             int my_rank, int num_nodes, bool use_normal_mode) {
  ibv_wc wc[kMaxOutstandingRecvs];
  auto poll_one = [&](ibv_cq* cq) {
    int ne = poll_cq_once(cq, wc, kMaxOutstandingRecvs);
    if (ne > 0) {
      remote_process_completions(S, idx, g_ring, ne, wc, ctx_by_tag,
                                 atomic_buffer_ptr, num_ranks, num_experts,
                                 pending_atomic_updates, my_rank, num_nodes,
                                 use_normal_mode);
    }
  };
  if (S.cq) poll_one(S.cq);
  // for (auto* cq : S.extra_cqs) poll_one(cq);
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
  assert(false && "ACK is disabled");
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

// Normal mode implementation
static void post_atomic_operations_normal_mode(
    ProxyCtx& S, std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post,
    std::vector<std::unique_ptr<ProxyCtx>>& ctxs, int my_rank, int thread_idx,
    std::unordered_set<uint64_t>& acked_wrs) {
  if (cmds_to_post.size() > ProxyCtx::kMaxAtomicOps) {
    fprintf(stderr, "Too many atomic operations: %zu > %zu\n",
            cmds_to_post.size(), ProxyCtx::kMaxAtomicOps);
    std::abort();
  }

  std::unordered_map<int, std::vector<size_t>> dst_rank_wr_ids;
  dst_rank_wr_ids.reserve(cmds_to_post.size());
  for (size_t i = 0; i < wrs_to_post.size(); ++i) {
    int dst = static_cast<int>(cmds_to_post[i].dst_rank);
    if (dst == my_rank) {
      fprintf(stderr, "Posting atomic to itself\n");
      std::abort();
    }
    dst_rank_wr_ids[dst].push_back(i);
  }

  for (auto& [dst_rank, wr_ids] : dst_rank_wr_ids) {
    if (wr_ids.empty()) continue;

    ProxyCtx* ctx = ctxs[dst_rank].get();
    size_t const k = wr_ids.size();
    // Group by ring index (upper 32 bits in wrs_to_post)
    std::unordered_map<size_t, std::vector<size_t>> ring_to_indices;
    ring_to_indices.reserve(k);
    for (size_t ii = 0; ii < k; ++ii) {
      size_t global_i = wr_ids[ii];
      size_t ring_idx =
          static_cast<size_t>((wrs_to_post[global_i] >> 32) & 0xFFFFFFFFu);
      ring_to_indices[ring_idx].push_back(global_i);
    }

    for (auto& [ring_idx_raw, idxs] : ring_to_indices) {
      size_t const local_ring_count = ctx->data_qps_by_channel.size();
#ifdef EFA
      struct ibv_qp_ex* qpx =
          (struct ibv_qp_ex*)(local_ring_count
                                  ? ctx->data_qps_by_channel[ring_idx_raw %
                                                             local_ring_count]
                                  : ctx->ack_qp);
      size_t const remote_ring_count = ctx->dst_data_qpn_by_ring.size();
      uint32_t const dst_qpn =
          remote_ring_count
              ? ctx->dst_data_qpn_by_ring[ring_idx_raw % remote_ring_count]
              : ctx->dst_qpn;
      ibv_wr_start(qpx);

      // Build the chain
      std::vector<uint64_t> group_wrids;
      group_wrids.reserve(idxs.size());

      for (size_t t = 0; t < idxs.size(); ++t) {
        size_t i = idxs[t];
        auto const& cmd = cmds_to_post[i];
        auto wr_id = wrs_to_post[i];
        group_wrids.push_back(wr_id);

        int v = static_cast<int>(cmd.value);
        if (v > kLargeAtomicValue) {
          // Sender-side saturation to fit imm payload
          v = kMaxSendAtomicValue;
        }
        if (v < -kMaxSendAtomicValue || v > kMaxSendAtomicValue) {
          fprintf(stderr,
                  "[EFA] value=%d (cmd.value: %lu) won't fit in 15 bits; "
                  "use an inline payload scheme instead.\n",
                  v, (unsigned long)cmd.value);
          std::abort();
        }

        uint32_t offset = static_cast<int64_t>(cmd.req_rptr);
        int low_latency_buffer_idx = get_low_latency(cmd.cmd_type);
        if (low_latency_buffer_idx < 0 || low_latency_buffer_idx > 1) {
          fprintf(stderr, "Invalid low_latency_buffer_idx: %d\n",
                  low_latency_buffer_idx);
          std::abort();
        }

        uint32_t imm = AtomicsImm::PackAtomic(v, offset).GetImmData();

        qpx->wr_id = kAtomicWrTag | (wr_id & kAtomicMask);
        qpx->comp_mask = 0;
        qpx->wr_flags = IBV_SEND_SIGNALED;

        ibv_wr_rdma_write_imm(qpx, ctx->remote_rkey, ctx->remote_addr,
                              htonl(imm));
        ibv_wr_set_ud_addr(qpx, ctx->dst_ah, dst_qpn, QKEY);
        ibv_wr_set_sge(qpx, ctx->mr->lkey, (uintptr_t)ctx->mr->addr, 0);
      }

      int ret = ibv_wr_complete(qpx);
      if (ret) {
        fprintf(stderr, "[EFA] post_send failed: %s (ret=%d)\n", strerror(ret),
                ret);
        std::abort();
      }
#else
      struct ibv_qp* qp =
          local_ring_count
              ? ctx->data_qps_by_channel[ring_idx_raw % local_ring_count]
              : ctx->ack_qp;

      size_t const k = idxs.size();
      std::vector<ibv_sge> sge(k);
      std::vector<ibv_send_wr> wr(k);
      std::vector<uint64_t> group_wrids;
      group_wrids.reserve(k);

      for (size_t t = 0; t < k; ++t) {
        size_t i = idxs[t];
        auto const& cmd = cmds_to_post[i];
        uint64_t const wr_id = wrs_to_post[i];
        group_wrids.push_back(wr_id);

        int v = static_cast<int>(cmd.value);
        if (v > kLargeAtomicValue) v = kMaxSendAtomicValue;
        if (v < -kMaxSendAtomicValue || v > kMaxSendAtomicValue) {
          fprintf(stderr,
                  "value=%d (cmd.value=%lu) won't fit in 15 bits for imm; "
                  "use a different scheme.\n",
                  v, (unsigned long)cmd.value);
          std::abort();
        }

        // If your AtomicsImm for non-EFA expects 16-bit offsets, keep the
        // mask:
        uint32_t off16 = static_cast<uint32_t>(cmd.req_rptr) & 0xFFFFu;
        int low_latency_buffer_idx = get_low_latency(cmd.cmd_type);
        if (low_latency_buffer_idx < 0 || low_latency_buffer_idx > 1) {
          fprintf(stderr, "Invalid low_latency_buffer_idx: %d\n",
                  low_latency_buffer_idx);
          std::abort();
        }
        uint32_t imm = AtomicsImm::Pack(
                           /*is_atomic*/ true,
                           /*is_combine*/ get_is_combine(cmd.cmd_type), v,
                           /*offset*/ off16, low_latency_buffer_idx)
                           .GetImmData();

        // Zero-length write-with-imm on RC QP
        sge[t].addr = reinterpret_cast<uintptr_t>(ctx->mr->addr);
        sge[t].length = 0;
        sge[t].lkey = ctx->mr->lkey;

        std::memset(&wr[t], 0, sizeof(wr[t]));
        wr[t].wr_id = kAtomicWrTag | (wr_id & kAtomicMask);
        wr[t].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr[t].send_flags = (t + 1 == k) ? IBV_SEND_SIGNALED : 0;
        wr[t].imm_data = htonl(imm);
        wr[t].sg_list = &sge[t];
        wr[t].num_sge = 1;
        wr[t].wr.rdma.remote_addr = ctx->remote_addr;
        wr[t].wr.rdma.rkey = ctx->remote_rkey;
        wr[t].next = (t + 1 < k) ? &wr[t + 1] : nullptr;
      }

      ibv_send_wr* bad = nullptr;
      int ret = ibv_post_send(qp, &wr[0], &bad);
      if (ret) {
        fprintf(stderr, "[RC] post_send(atomic imm) failed: %s (ret=%d)\n",
                strerror(ret), ret);
        if (bad) {
          fprintf(stderr, "  bad wr_id=0x%llx opcode=%u\n",
                  (unsigned long long)bad->wr_id, bad->opcode);
        }
        std::abort();
      }
      uint64_t const batch_tail_wr = group_wrids.back();
      {
        auto [it, inserted] = S.wr_id_to_wr_ids.try_emplace(
            batch_tail_wr, std::move(group_wrids));
        if (!inserted) {
          fprintf(stderr,
                  "thread_idx: %d, Error: tail wr_id %lu already exists "
                  "(map=%p, "
                  "size=%zu, dst_rank=%d)\n",
                  thread_idx, batch_tail_wr, (void*)&S.wr_id_to_wr_ids,
                  S.wr_id_to_wr_ids.size(), dst_rank);
          std::abort();
        }
      }
#endif
    }
  }
}

// Fast mode implementation
static void post_atomic_operations_fast_mode(
    ProxyCtx& S, std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post,
    std::vector<std::unique_ptr<ProxyCtx>>& ctxs, int my_rank, int thread_idx,
    std::unordered_set<uint64_t>& acked_wrs) {
  if (cmds_to_post.size() > ProxyCtx::kMaxAtomicOps) {
    fprintf(stderr, "Too many atomic operations: %zu > %zu\n",
            cmds_to_post.size(), ProxyCtx::kMaxAtomicOps);
    std::abort();
  }

  std::unordered_map<int, std::vector<size_t>> dst_rank_wr_ids;
  dst_rank_wr_ids.reserve(cmds_to_post.size());
  for (size_t i = 0; i < wrs_to_post.size(); ++i) {
    int dst = static_cast<int>(cmds_to_post[i].dst_rank);
    if (dst == my_rank) {
      fprintf(stderr, "Posting atomic to itself\n");
      std::abort();
    }
    dst_rank_wr_ids[dst].push_back(i);
  }

  for (auto& [dst_rank, wr_ids] : dst_rank_wr_ids) {
    if (wr_ids.empty()) continue;

    ProxyCtx* ctx = ctxs[dst_rank].get();
    size_t const k = wr_ids.size();
#ifdef EFA
    struct ibv_qp_ex* qpx = (struct ibv_qp_ex*)ctx->qp;
    ibv_wr_start(qpx);
    for (size_t i = 0; i < k; ++i) {
      auto const& cmd = cmds_to_post[wr_ids[i]];
      auto wr_id = wrs_to_post[wr_ids[i]];
      wr_ids[i] = wr_id;

      int v = static_cast<int>(cmd.value);
      if (v == kLargeAtomicValue) v = kMaxSendAtomicValue;
      if (v < -kMaxSendAtomicValue || v > kMaxSendAtomicValue) {
        fprintf(stderr,
                "[EFA] value=%d (cmd.value: %lu) won't fit in 15 bits; "
                "use an inline payload scheme instead.\n",
                v, (unsigned long)cmd.value);
        std::abort();
      }
      uint32_t offset = static_cast<int64_t>(cmd.req_rptr);
      int low_latency_buffer_idx = get_low_latency(cmd.cmd_type);
      if (low_latency_buffer_idx < 0 || low_latency_buffer_idx > 1) {
        fprintf(stderr, "Invalid low_latency_buffer_idx: %d\n",
                low_latency_buffer_idx);
        std::abort();
      }
      uint32_t imm = AtomicsImm::Pack(true, get_is_combine(cmd.cmd_type), v,
                                      offset, low_latency_buffer_idx)
                         .GetImmData();

      qpx->wr_id = kAtomicWrTag | (wr_id & kAtomicMask);
      qpx->comp_mask = 0;
      qpx->wr_flags = IBV_SEND_SIGNALED;
      ibv_wr_rdma_write_imm(qpx, ctx->remote_rkey, ctx->remote_addr,
                            htonl(imm));

      ibv_wr_set_ud_addr(qpx, ctx->dst_ah, ctx->dst_qpn, QKEY);
      ibv_wr_set_sge(qpx, ctx->mr->lkey, (uintptr_t)ctx->mr->addr, 0);
    }
    int ret = ibv_wr_complete(qpx);
    if (ret) {
      fprintf(stderr, "[EFA] post_send failed: %s (ret=%d)\n", strerror(ret),
              ret);
      std::abort();
    }
#else
    std::vector<ibv_sge> sge(k);
    std::vector<ibv_send_wr> wr(k);

    for (size_t i = 0; i < k; ++i) {
      auto const& cmd = cmds_to_post[wr_ids[i]];
      uint64_t const wrid = wrs_to_post[wr_ids[i]];
      wr_ids[i] = wrid;

      int v = static_cast<int>(cmd.value);
      if (v == kLargeAtomicValue) v = kMaxSendAtomicValue;
      if (v < -kMaxSendAtomicValue || v > kMaxSendAtomicValue) {
        fprintf(stderr, "value=%d won't fit in 15 bits\n", v);
        std::abort();
      }
      uint32_t const off16 = static_cast<uint32_t>(cmd.req_rptr) & 0xFFFFu;
      int low_latency_buffer_idx = get_low_latency(cmd.cmd_type);
      if (low_latency_buffer_idx < 0 || low_latency_buffer_idx > 1) {
        fprintf(stderr, "Invalid low_latency_buffer_idx: %d\n",
                low_latency_buffer_idx);
        std::abort();
      }
      uint32_t const imm = AtomicsImm::Pack(true, get_is_combine(cmd.cmd_type),
                                            v, off16, low_latency_buffer_idx)
                               .GetImmData();
      sge[i].addr = reinterpret_cast<uintptr_t>(ctx->mr->addr);
      sge[i].length = 0;
      sge[i].lkey = ctx->mr->lkey;

      std::memset(&wr[i], 0, sizeof(wr[i]));
      wr[i].wr_id = kAtomicWrTag | (wrid & kAtomicMask);
      wr[i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      wr[i].send_flags = (i + 1 == k) ? IBV_SEND_SIGNALED : 0;
      wr[i].imm_data = htonl(imm);
      wr[i].sg_list = &sge[i];
      wr[i].num_sge = 1;
      wr[i].wr.rdma.remote_addr = ctx->remote_addr;
      wr[i].wr.rdma.rkey = ctx->remote_rkey;
      wr[i].next = (i + 1 < k) ? &wr[i + 1] : nullptr;
    }
    {
      ibv_send_wr* bad = nullptr;
      int ret = ibv_post_send(ctx->qp, &wr[0], &bad);
      if (ret) {
        fprintf(stderr, "ibv_post_send(atomic) failed: %d (%s)\n", ret,
                strerror(ret));
        if (bad)
          fprintf(stderr, "  bad wr_id=0x%llx\n",
                  (unsigned long long)bad->wr_id);
        std::abort();
      }
    }
#endif
    uint64_t const batch_tail_wr = wr_ids.back();
    {
      auto [it, inserted] =
          S.wr_id_to_wr_ids.try_emplace(batch_tail_wr, std::move(wr_ids));
      if (!inserted) {
        fprintf(stderr,
                "thread_idx: %d, Error: tail wr_id %lu already exists "
                "(map=%p, "
                "size=%zu, dst_rank=%d)\n",
                thread_idx, batch_tail_wr, (void*)&S.wr_id_to_wr_ids,
                S.wr_id_to_wr_ids.size(), dst_rank);
        std::abort();
      }
    }
  }
}

// Fast mode + native RDMA atomics (non-EFA)
//
// This mirrors the batching/dispatch structure of
// `post_atomic_operations_fast_mode` (group by dst_rank, use ctx->qp, single
// completion per dst batch), but emits real RDMA atomics like
// `post_atomic_operations_native_rdma` (FETCH_AND_ADD to the remote atomic
// buffer).
static void post_atomic_operations_fast_mode_native_rdma(
    ProxyCtx& S, std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post,
    std::vector<std::unique_ptr<ProxyCtx>>& ctxs, int my_rank, int thread_idx,
    std::unordered_set<uint64_t>& acked_wrs) {
  (void)thread_idx;
  (void)acked_wrs;
#ifdef EFA
  // EFA path uses write-with-imm atomics; native FETCH_AND_ADD is not used
  // here.
  post_atomic_operations_fast_mode(S, wrs_to_post, cmds_to_post, ctxs, my_rank,
                                   thread_idx, acked_wrs);
#else
  if (cmds_to_post.size() > ProxyCtx::kMaxAtomicOps) {
    fprintf(stderr, "Too many atomic operations: %zu > %zu\n",
            cmds_to_post.size(), ProxyCtx::kMaxAtomicOps);
    std::abort();
  }

  std::unordered_map<int, std::vector<size_t>> dst_rank_wr_ids;
  dst_rank_wr_ids.reserve(cmds_to_post.size());
  for (size_t i = 0; i < wrs_to_post.size(); ++i) {
    int dst = static_cast<int>(cmds_to_post[i].dst_rank);
    if (dst == my_rank) {
      fprintf(stderr, "Posting atomic to itself\n");
      std::abort();
    }
    dst_rank_wr_ids[dst].push_back(i);
  }

  if (!S.atomic_old_values_buf || !S.atomic_old_values_mr) {
    fprintf(stderr,
            "[Native RDMA Fast] Local atomic scratch (old values) not "
            "initialized (buf=%p mr=%p)\n",
            (void*)S.atomic_old_values_buf, (void*)S.atomic_old_values_mr);
    std::abort();
  }
  uintptr_t atomic_buf_addr =
      reinterpret_cast<uintptr_t>(S.atomic_old_values_buf);
  if ((atomic_buf_addr & 0x7) != 0) {
    fprintf(stderr,
            "[Native RDMA Fast] atomic_old_values_buf not 8-byte aligned: "
            "0x%llx\n",
            (unsigned long long)atomic_buf_addr);
    std::abort();
  }

  for (auto& [dst_rank, idxs] : dst_rank_wr_ids) {
    if (idxs.empty()) continue;

    ProxyCtx* ctx = ctxs[dst_rank].get();
    if (!ctx || !ctx->qp) {
      fprintf(stderr,
              "[Native RDMA Fast] Destination ctx missing fields for dst=%d\n",
              dst_rank);
      std::abort();
    }
    if (ctx->remote_atomic_buffer_addr == 0 ||
        ctx->remote_atomic_buffer_rkey == 0) {
      fprintf(stderr,
              "[Native RDMA Fast] Remote atomic buffer not registered "
              "(addr=0x%llx rkey=0x%x) for dst=%d\n",
              (unsigned long long)ctx->remote_atomic_buffer_addr,
              ctx->remote_atomic_buffer_rkey, dst_rank);
      std::abort();
    }

    size_t const k = idxs.size();
    if (S.atomic_old_values_mr->length < k * sizeof(uint64_t)) {
      fprintf(stderr,
              "[Native RDMA Fast] atomic_old_values_mr too small: len=%zu, "
              "need=%zu (k=%zu)\n",
              (size_t)S.atomic_old_values_mr->length, k * sizeof(uint64_t), k);
      std::abort();
    }

    std::vector<ibv_sge> sge(k);
    std::vector<ibv_send_wr> wr(k);
    std::vector<uint64_t> group_wrids;
    group_wrids.reserve(k);

    for (size_t t = 0; t < k; ++t) {
      size_t cmd_i = idxs[t];
      auto const& cmd = cmds_to_post[cmd_i];
      uint64_t const wr_id = wrs_to_post[cmd_i];
      group_wrids.push_back(wr_id);

      int v = static_cast<int>(cmd.value);
      bool const is_combine = get_is_combine(cmd.cmd_type);
      if (is_combine) v = 1;
      if (v == kLargeAtomicValue) v = kMaxSendAtomicValue;
      int64_t v64 = static_cast<int64_t>(static_cast<int32_t>(v));

      uint64_t remote_atomic_addr =
          ctx->remote_atomic_buffer_addr + cmd.req_rptr;
      if ((remote_atomic_addr & 0x7) != 0) {
        fprintf(stderr,
                "[Native RDMA Fast] Remote atomic address must be 8-byte "
                "aligned: 0x%llx\n",
                (unsigned long long)remote_atomic_addr);
        std::abort();
      }
      if (remote_atomic_addr < ctx->remote_atomic_buffer_addr ||
          remote_atomic_addr + sizeof(uint64_t) >
              ctx->remote_atomic_buffer_addr + ctx->remote_atomic_buffer_len) {
        fprintf(stderr,
                "[Native RDMA Fast] Remote atomic address out of bounds: "
                "0x%llx (base=0x%llx, len=%zu)\n",
                (unsigned long long)remote_atomic_addr,
                (unsigned long long)ctx->remote_atomic_buffer_addr,
                (size_t)ctx->remote_atomic_buffer_len);
        std::abort();
      }

      sge[t].addr = atomic_buf_addr + t * sizeof(uint64_t);
      sge[t].length = sizeof(uint64_t);
      sge[t].lkey = S.atomic_old_values_mr->lkey;

      std::memset(&wr[t], 0, sizeof(wr[t]));
      wr[t].wr_id = wr_id;
      wr[t].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
      wr[t].send_flags = (t + 1 == k) ? IBV_SEND_SIGNALED : 0;
      wr[t].sg_list = &sge[t];
      wr[t].num_sge = 1;
      wr[t].wr.atomic.remote_addr = remote_atomic_addr;
      wr[t].wr.atomic.rkey = ctx->remote_atomic_buffer_rkey;
      wr[t].wr.atomic.compare_add = v64;
      wr[t].next = (t + 1 < k) ? &wr[t + 1] : nullptr;
    }

    ibv_send_wr* bad = nullptr;
    int ret = ibv_post_send(ctx->qp, &wr[0], &bad);
    if (ret) {
      fprintf(stderr,
              "[Native RDMA Fast] ibv_post_send(atomic) failed: %d (%s)\n", ret,
              strerror(ret));
      if (bad) {
        fprintf(stderr, "  bad wr_id=0x%llx opcode=%u\n",
                (unsigned long long)bad->wr_id, bad->opcode);
      }
      std::abort();
    }

    uint64_t const batch_tail_wr = group_wrids.back();
    {
      auto [it, inserted] =
          S.wr_id_to_wr_ids.try_emplace(batch_tail_wr, std::move(group_wrids));
      if (!inserted) {
        fprintf(stderr,
                "thread_idx: %d, Error: tail wr_id %lu already exists "
                "(map=%p, size=%zu, dst_rank=%d)\n",
                thread_idx, batch_tail_wr, (void*)&S.wr_id_to_wr_ids,
                S.wr_id_to_wr_ids.size(), dst_rank);
        std::abort();
      }
    }
  }
#endif
}

// Native RDMA implementation (non-EFA)
static void post_atomic_operations_native_rdma(
    ProxyCtx& S, std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post,
    std::vector<std::unique_ptr<ProxyCtx>>& ctxs, int my_rank, int thread_idx,
    std::unordered_set<uint64_t>& acked_wrs) {
  if (cmds_to_post.size() > ProxyCtx::kMaxAtomicOps) {
    fprintf(stderr, "Too many atomic operations: %zu > %zu\n",
            cmds_to_post.size(), ProxyCtx::kMaxAtomicOps);
    std::abort();
  }

  std::unordered_map<int, std::vector<size_t>> dst_rank_wr_ids;
  dst_rank_wr_ids.reserve(cmds_to_post.size());
  for (size_t i = 0; i < wrs_to_post.size(); ++i) {
    int dst = static_cast<int>(cmds_to_post[i].dst_rank);
    if (dst == my_rank) {
      fprintf(stderr, "Posting atomic to itself\n");
      std::abort();
    }
    dst_rank_wr_ids[dst].push_back(i);
  }

  for (auto& [dst_rank, wr_ids] : dst_rank_wr_ids) {
    if (wr_ids.empty()) continue;

    ProxyCtx* ctx = ctxs[dst_rank].get();
    size_t const k = wr_ids.size();
    // Group by ring index (upper 32 bits in wrs_to_post)
    std::unordered_map<size_t, std::vector<size_t>> ring_to_indices;
    ring_to_indices.reserve(k);
    for (size_t ii = 0; ii < k; ++ii) {
      size_t global_i = wr_ids[ii];
      size_t ring_idx =
          static_cast<size_t>((wrs_to_post[global_i] >> 32) & 0xFFFFFFFFu);
      ring_to_indices[ring_idx].push_back(global_i);
    }

    for (auto& [ring_idx_raw, idxs] : ring_to_indices) {
      size_t const local_ring_count = ctx->data_qps_by_channel.size();
      struct ibv_qp* qp =
          local_ring_count
              ? ctx->data_qps_by_channel[ring_idx_raw % local_ring_count]
              : ctx->ack_qp;

      size_t const k = idxs.size();
      std::vector<ibv_sge> sge(k);
      std::vector<ibv_send_wr> wr(k);
      std::vector<uint64_t> group_wrids;
      group_wrids.reserve(k);

      // Verify atomic buffer is properly aligned (8-byte for 64-bit atomics)
      uintptr_t atomic_buf_addr =
          reinterpret_cast<uintptr_t>(S.atomic_old_values_buf);
      if ((atomic_buf_addr & 0x7) != 0) {
        fprintf(
            stderr,
            "[Native RDMA] atomic_old_values_buf not 8-byte aligned: 0x%llx\n",
            (unsigned long long)atomic_buf_addr);
        std::abort();
      }

      uint64_t* atomic_old_values_64 =
          reinterpret_cast<uint64_t*>(S.atomic_old_values_buf);

      for (size_t t = 0; t < k; ++t) {
        size_t i = idxs[t];
        auto const& cmd = cmds_to_post[i];
        uint64_t const wr_id = wrs_to_post[i];
        group_wrids.push_back(wr_id);

        int v = static_cast<int>(cmd.value);
        if (v == kLargeAtomicValue) v = kMaxSendAtomicValue;

        // Convert 32-bit signed int to 64-bit for RDMA atomics
        // IBV_WR_ATOMIC_FETCH_AND_ADD requires 64-bit operations
        int64_t v64 = static_cast<int64_t>(static_cast<int32_t>(v));

        // Calculate remote address - must be 8-byte aligned for 64-bit RDMA
        // atomics cmd.req_rptr is an offset relative to atomic_base_addr (local
        // atomic_buffer_ptr) Use the remote atomic buffer address if available,
        // otherwise fall back to remote_addr
        if ((cmd.req_rptr & 0x7) != 0) {
          fprintf(stderr, "[Native RDMA] req_rptr not 8-byte aligned: 0x%x\n",
                  cmd.req_rptr);
          std::abort();
        }
        uint64_t remote_atomic_addr;
        if (ctx->remote_atomic_buffer_addr != 0) {
          // Use registered atomic buffer
          remote_atomic_addr = ctx->remote_atomic_buffer_addr + cmd.req_rptr;
        } else {
          assert(false && "Atomic buffer is not registered");
        }

        // Verify final address alignment (should always be true if req_rptr is
        // aligned)
        assert((ctx->remote_atomic_buffer_addr & 0x7) == 0 &&
               "Remote atomic buffer address must be 8-byte aligned");
        assert((cmd.req_rptr & 0x7) == 0 && "req_rptr must be 8-byte aligned");
        assert((remote_atomic_addr & 0x7) == 0 &&
               "Remote atomic address must be 8-byte aligned");

        // Verify remote address is within bounds of the atomic buffer
        if (ctx->remote_atomic_buffer_addr == 0) {
          fprintf(stderr,
                  "[Native RDMA] Remote atomic buffer not registered\n");
          std::abort();
        }
        if (remote_atomic_addr < ctx->remote_atomic_buffer_addr ||
            remote_atomic_addr + sizeof(uint64_t) >
                ctx->remote_atomic_buffer_addr +
                    ctx->remote_atomic_buffer_len) {
          fprintf(stderr,
                  "[Native RDMA] Remote atomic address out of bounds: 0x%llx "
                  "(base=0x%llx, len=%zu)\n",
                  (unsigned long long)remote_atomic_addr,
                  (unsigned long long)ctx->remote_atomic_buffer_addr,
                  (size_t)ctx->remote_atomic_buffer_len);
          std::abort();
        }

        // Local SGE: point to local buffer where old value will be stored
        // (64-bit) Use local context S's memory region, not destination ctx
        uintptr_t local_addr =
            reinterpret_cast<uintptr_t>(&atomic_old_values_64[t]);
        sge[t].addr = local_addr;
        sge[t].length = sizeof(uint64_t);
        sge[t].lkey = S.atomic_old_values_mr->lkey;

        std::memset(&wr[t], 0, sizeof(wr[t]));
        wr[t].wr_id = wr_id;
        wr[t].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
        wr[t].send_flags = (t + 1 == k) ? IBV_SEND_SIGNALED : 0;
        wr[t].sg_list = &sge[t];
        wr[t].num_sge = 1;
        wr[t].wr.atomic.remote_addr = remote_atomic_addr;
        // Use remote atomic buffer rkey if available, otherwise use main buffer
        // rkey
        assert(ctx->remote_atomic_buffer_rkey != 0);
        wr[t].wr.atomic.rkey = ctx->remote_atomic_buffer_rkey;
        wr[t].wr.atomic.compare_add = v64;  // 64-bit value to add
        wr[t].next = (t + 1 < k) ? &wr[t + 1] : nullptr;
      }

      ibv_send_wr* bad = nullptr;
      int ret = ibv_post_send(qp, &wr[0], &bad);
      if (ret) {
        fprintf(stderr, "[Native RDMA] ibv_post_send(atomic) failed: %d (%s)\n",
                ret, strerror(ret));
        if (bad) {
          fprintf(stderr, "  bad wr_id=0x%llx opcode=%u\n",
                  (unsigned long long)bad->wr_id, bad->opcode);
        }
        std::abort();
      }
      // The completion will have wr_id = wr[k-1].wr_id (the last signaled WR)
      // Store this exact value to ensure lookup matches
      uint64_t const batch_tail_wr = wr[k - 1].wr_id;
      // Verify this matches group_wrids.back() (should always be true)
      if (batch_tail_wr != group_wrids.back()) {
        fprintf(
            stderr,
            "[Native RDMA] ERROR: batch_tail_wr (0x%lx) != group_wrids.back() "
            "(0x%lx)\n",
            batch_tail_wr, group_wrids.back());
        std::abort();
      }
      {
        auto [it, inserted] = S.wr_id_to_wr_ids.try_emplace(
            batch_tail_wr, std::move(group_wrids));

        // printf("[Native RDMA] batch_tail_wr: 0x%lx, map_size: %zu, dst_rank:
        // %d\n", batch_tail_wr, it->second.size(), dst_rank);
        if (!inserted) {
          fprintf(stderr,
                  "thread_idx: %d, Error: tail wr_id %lu already exists "
                  "(map=%p, "
                  "size=%zu, dst_rank=%d)\n",
                  thread_idx, batch_tail_wr, (void*)&S.wr_id_to_wr_ids,
                  S.wr_id_to_wr_ids.size(), dst_rank);
          std::abort();
        }
      }
    }
  }
}

// Wrapper that selects implementation based on use_normal_mode
void post_atomic_operations(ProxyCtx& S,
                            std::vector<uint64_t> const& wrs_to_post,
                            std::vector<TransferCmd> const& cmds_to_post,
                            std::vector<std::unique_ptr<ProxyCtx>>& ctxs,
                            int my_rank, int thread_idx,
                            std::unordered_set<uint64_t>& acked_wrs,
                            bool use_normal_mode) {
  if (use_normal_mode) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    post_atomic_operations_native_rdma(S, wrs_to_post, cmds_to_post, ctxs,
                                       my_rank, thread_idx, acked_wrs);
#else
    post_atomic_operations_normal_mode(S, wrs_to_post, cmds_to_post, ctxs,
                                       my_rank, thread_idx, acked_wrs);
#endif
  } else {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    post_atomic_operations_fast_mode_native_rdma(
        S, wrs_to_post, cmds_to_post, ctxs, my_rank, thread_idx, acked_wrs);
#else
    post_atomic_operations_fast_mode(S, wrs_to_post, cmds_to_post, ctxs,
                                     my_rank, thread_idx, acked_wrs);
#endif
  }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <infiniband/verbs.h>
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#include <infiniband/efadv.h>
#endif
#include <algorithm>
#include <atomic>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <cerrno>
#endif
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

constexpr size_t MSG_SIZE = 7168;
constexpr int MAX_LATENCY_SAMPLES = 10000;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
constexpr int MAX_posted_WRS = 64;
#else
constexpr int MAX_OUTSTANDING_WRS = 64;
constexpr uint32_t QKEY = 0x11111111;
#endif

struct RDMAConnectionInfo {
  uint32_t qp_num;
  uint32_t rkey;
  uint64_t addr;
  uint64_t len;
  uint8_t gid[16];
};

struct ThreadMetrics {
  std::vector<uint64_t> latency_samples_ns;
  uint64_t write_count = 0;
  uint64_t completion_count = 0;
};

struct RDMAContext {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_mr* mr = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
  ibv_ah* ah = nullptr;
#endif

  void* local_buf = nullptr;
  uint64_t remote_addr = 0;
  uint32_t remote_qpn = 0;
  uint32_t remote_rkey = 0;

  int gid_index = 0;
};

void mop_sleep(uint64_t nanoseconds) {
  auto start = std::chrono::high_resolution_clock::now();
  while (true) {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - start)
            .count();
    if (elapsed >= static_cast<int64_t>(nanoseconds)) break;
  }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
bool is_ipv6(char const* addr) { return strchr(addr, ':') != nullptr; }
#endif

// RDMA info exchange
void exchange_connection_info(int rank, int peer_rank, char const* peer_ip,
                              RDMAConnectionInfo* local_info,
                              RDMAConnectionInfo* remote_info) {
  int base_port = 50000;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  bool use_ipv6 = is_ipv6(peer_ip);
  int af = use_ipv6 ? AF_INET6 : AF_INET;
#endif

  if (rank < peer_rank) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    int listen_sock = socket(af, SOCK_STREAM, 0);
#else
    int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
#endif
    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (use_ipv6) {
      struct sockaddr_in6 addr;
      memset(&addr, 0, sizeof(addr));
      addr.sin6_family = AF_INET6;
      addr.sin6_addr = in6addr_any;
      addr.sin6_port = htons(base_port + rank);

      bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr));
    } else {
      struct sockaddr_in addr;
      memset(&addr, 0, sizeof(addr));
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = INADDR_ANY;
      addr.sin_port = htons(base_port + rank);

      bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr));
    }
#else
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(base_port + rank);

    bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr));
#endif
    listen(listen_sock, 1);

    int conn = accept(listen_sock, nullptr, nullptr);
    send(conn, local_info, sizeof(*local_info), 0);
    recv(conn, remote_info, sizeof(*remote_info), 0);

    close(conn);
    close(listen_sock);
  } else {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    int sock = socket(af, SOCK_STREAM, 0);

    if (use_ipv6) {
      struct sockaddr_in6 addr;
      memset(&addr, 0, sizeof(addr));
      addr.sin6_family = AF_INET6;
      addr.sin6_port = htons(base_port + peer_rank);
      inet_pton(AF_INET6, peer_ip, &addr.sin6_addr);

      for (int retry = 0; retry < 30; retry++) {
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } else {
      struct sockaddr_in addr;
      memset(&addr, 0, sizeof(addr));
      addr.sin_family = AF_INET;
      addr.sin_port = htons(base_port + peer_rank);
      inet_pton(AF_INET, peer_ip, &addr.sin_addr);

      for (int retry = 0; retry < 30; retry++) {
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(base_port + peer_rank);
    inet_pton(AF_INET, peer_ip, &addr.sin_addr);

    for (int retry = 0; retry < 30; retry++) {
      if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#endif

    recv(sock, remote_info, sizeof(*remote_info), 0);
    send(sock, local_info, sizeof(*local_info), 0);

    close(sock);
  }
}

// per-thread rdma ctx init
bool init_rdma_context(RDMAContext* rctx, void* gpu_buf, size_t bytes,
                       int local_rank, int nic_index) {
  int num_devices = 0;
  struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list || num_devices == 0) {
    fprintf(stderr, "Failed to get IB devices\n");
    return false;
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  int selected_dev = -1;
  for (int i = 0; i < num_devices; i++) {
    char const* name = ibv_get_device_name(dev_list[i]);
    if (strcmp(name, "bnxt_re0") == 0) {
      selected_dev = i;
      break;
    }
  }

  if (selected_dev < 0) {
    fprintf(stderr, "bnxt_re0 not found\n");
    ibv_free_device_list(dev_list);
    return false;
  }

  static bool printed = false;
  if (nic_index == 0 && !printed) {
    printf("Using bnxt_re0 + GPU 0 (PCIe co-located)\n");
    printed = true;
  }
#else
  char const* target_nics[2] = {"rdmap85s0", "rdmap86s0"};
  int selected_dev = -1;

  char const* target_nic = target_nics[nic_index % 2];

  for (int i = 0; i < num_devices; i++) {
    char const* name = ibv_get_device_name(dev_list[i]);
    if (strcmp(name, target_nic) == 0) {
      selected_dev = i;
      break;
    }
  }

  if (selected_dev == -1) {
    fprintf(stderr, "Failed to find NIC: %s\n", target_nic);
    ibv_free_device_list(dev_list);
    return false;
  }
#endif

  rctx->ctx = ibv_open_device(dev_list[selected_dev]);
  if (!rctx->ctx) {
    fprintf(stderr, "Failed to open device %d\n", selected_dev);
    ibv_free_device_list(dev_list);
    return false;
  }

  ibv_free_device_list(dev_list);

  rctx->pd = ibv_alloc_pd(rctx->ctx);
  if (!rctx->pd) {
    fprintf(stderr, "Failed to allocate PD\n");
    return false;
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  rctx->mr = ibv_reg_mr_iova2(rctx->pd, gpu_buf, bytes,
                              reinterpret_cast<uint64_t>(gpu_buf),
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
#else
  uint64_t iova = (uintptr_t)gpu_buf;
  rctx->mr = ibv_reg_mr_iova2(rctx->pd, gpu_buf, bytes, iova,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_RELAXED_ORDERING);
#endif

  if (!rctx->mr) {
    fprintf(stderr, "Failed to register MR\n");
    return false;
  }

  rctx->local_buf = gpu_buf;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  rctx->cq = ibv_create_cq(rctx->ctx, MAX_posted_WRS * 2, nullptr, nullptr, 0);
#else
  rctx->cq =
      ibv_create_cq(rctx->ctx, MAX_OUTSTANDING_WRS * 2, nullptr, nullptr, 0);
#endif
  if (!rctx->cq) {
    fprintf(stderr, "Failed to create CQ\n");
    return false;
  }

  return true;
}

bool create_qp(RDMAContext* rctx) {
  struct ibv_qp_init_attr_ex qp_attr = {};
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
  struct efadv_qp_init_attr efa_attr = {};
#endif

  qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  qp_attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
  qp_attr.cap.max_send_wr = MAX_posted_WRS;
  qp_attr.cap.max_recv_wr = MAX_posted_WRS;
#else
  qp_attr.send_ops_flags =
      IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;
  qp_attr.cap.max_send_wr = MAX_OUTSTANDING_WRS;
  qp_attr.cap.max_recv_wr = MAX_OUTSTANDING_WRS;
#endif
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  qp_attr.cap.max_inline_data = 0;
  qp_attr.pd = rctx->pd;
  qp_attr.qp_context = rctx->ctx;
  qp_attr.sq_sig_all = 1;
  qp_attr.send_cq = rctx->cq;
  qp_attr.recv_cq = rctx->cq;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  qp_attr.qp_type = IBV_QPT_RC;
#else
  qp_attr.qp_type = IBV_QPT_DRIVER;

  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  efa_attr.sl = 8;
  efa_attr.flags = 0;
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  rctx->qp = ibv_create_qp_ex(rctx->ctx, &qp_attr);
#else
  rctx->qp =
      efadv_create_qp_ex(rctx->ctx, &qp_attr, &efa_attr, sizeof(efa_attr));
#endif

  if (!rctx->qp) {
    fprintf(stderr, "Failed to create QP\n");
    return false;
  }

  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  if (ibv_modify_qp(rctx->qp, &attr,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS)) {
    fprintf(stderr, "Failed to modify QP to INIT\n");
    return false;
  }

  return true;
}

bool connect_qp(RDMAContext* rctx, uint8_t* remote_gid, uint32_t remote_qpn) {
  struct ibv_qp_attr attr = {};

  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  attr.ah_attr.is_global = 1;
  attr.ah_attr.port_num = 1;
  attr.ah_attr.sl = 0;
  memcpy(attr.ah_attr.grh.dgid.raw, remote_gid, 16);
  attr.ah_attr.grh.flow_label = 0;
  attr.ah_attr.grh.hop_limit = 64;
  attr.ah_attr.grh.sgid_index = rctx->gid_index;
  attr.ah_attr.grh.traffic_class = 0;

  int ret = ibv_modify_qp(rctx->qp, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret) {
    fprintf(stderr, "Failed to modify QP to RTR: errno=%d (%s)\n", errno,
            strerror(errno));
    return false;
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;

  if (ibv_modify_qp(rctx->qp, &attr,
                    IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                        IBV_QP_MAX_QP_RD_ATOMIC)) {
    fprintf(stderr, "Failed to modify QP to RTS\n");
    return false;
  }

  return true;
}
#else
  attr.qkey = QKEY;

  if (ibv_modify_qp(
          rctx->qp, &attr,
          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
    fprintf(stderr, "Failed to modify QP to INIT\n");
    return false;
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(rctx->qp, &attr, IBV_QP_STATE)) {
    fprintf(stderr, "Failed to modify QP to RTR\n");
    return false;
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.rnr_retry = 3;
  if (ibv_modify_qp(rctx->qp, &attr,
                    IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY)) {
    fprintf(stderr, "Failed to modify QP to RTS\n");
    return false;
  }

  return true;
}

bool create_ah(RDMAContext* rctx, uint8_t* remote_gid) {
  struct ibv_ah_attr ah_attr = {};
  ah_attr.port_num = 1;
  ah_attr.is_global = 1;
  memcpy(ah_attr.grh.dgid.raw, remote_gid, 16);

  rctx->ah = ibv_create_ah(rctx->pd, &ah_attr);
  if (!rctx->ah) {
    fprintf(stderr, "Failed to create AH\n");
    return false;
  }

  return true;
}
#endif

int poll_completions(RDMAContext* rctx, int max_polls,
                     std::vector<uint64_t>& completed_wr_ids) {
  ibv_wc wc[32];
  int poll_budget = std::min(32, max_polls);
  int n = ibv_poll_cq(rctx->cq, poll_budget, wc);

  if (n < 0) {
    fprintf(stderr, "Poll CQ error\n");
    std::abort();
  }

  for (int i = 0; i < n; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      fprintf(stderr, "CQE error: %s\n", ibv_wc_status_str(wc[i].status));
      std::abort();
    }
    completed_wr_ids.push_back(wc[i].wr_id);
  }

  return n;
}

bool post_rdma_write(RDMAContext* rctx, uint64_t wr_id, size_t offset) {
  ibv_qp_ex* qpx = ibv_qp_to_qp_ex(rctx->qp);

  ibv_wr_start(qpx);
  qpx->wr_id = wr_id;
  qpx->wr_flags = IBV_SEND_SIGNALED;

  uint64_t remote_addr = rctx->remote_addr + offset;
  ibv_wr_rdma_write(qpx, rctx->remote_rkey, remote_addr);

  uintptr_t local_addr = reinterpret_cast<uintptr_t>(rctx->local_buf) + offset;
  ibv_sge sge = {local_addr, MSG_SIZE, rctx->mr->lkey};
  ibv_wr_set_sge_list(qpx, 1, &sge);
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
  ibv_wr_set_ud_addr(qpx, rctx->ah, rctx->remote_qpn, QKEY);
#endif

  int ret = ibv_wr_complete(qpx);
  if (ret) {
    fprintf(stderr, "ibv_wr_complete failed: %d\n", ret);
    return false;
  }

  return true;
}

void working_thread(int thread_id, RDMAContext* rctx, uint64_t sleep_ns,
                    int test_duration_ms, std::atomic<bool>& stop_flag,
                    ThreadMetrics& metrics) {
  auto test_start = std::chrono::high_resolution_clock::now();
  auto test_duration = std::chrono::milliseconds(test_duration_ms);

  uint64_t wr_id = 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  int posted = 0;
#else
  int outstanding = 0;
#endif

  std::unordered_map<uint64_t, std::chrono::high_resolution_clock::time_point>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
      post_times;
  auto last_post_time = std::chrono::high_resolution_clock::now();
#else
      pending_requests;

  auto last_post_time = test_start;
#endif

  while (!stop_flag) {
    auto now = std::chrono::high_resolution_clock::now();
    if (now - test_start > test_duration) break;

    std::vector<uint64_t> completed_wr_ids;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    int completed = poll_completions(rctx, posted, completed_wr_ids);
    if (completed > 0) {
      auto completion_time = std::chrono::high_resolution_clock::now();
      posted -= completed;
      metrics.completion_count += completed;

      for (uint64_t id : completed_wr_ids) {
        auto it = post_times.find(id);
        if (it != post_times.end()) {
          auto latency_ns =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  completion_time - it->second)
                  .count();

          if (metrics.latency_samples_ns.size() < MAX_LATENCY_SAMPLES) {
            metrics.latency_samples_ns.push_back(latency_ns);
          }
          post_times.erase(it);
        }
      }
    }
#else
    int completed = poll_completions(rctx, outstanding, completed_wr_ids);
    outstanding -= completed;
    metrics.completion_count += completed;

    auto completion_time = std::chrono::high_resolution_clock::now();
    for (uint64_t id : completed_wr_ids) {
      auto it = pending_requests.find(id);
      if (it != pending_requests.end()) {
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              completion_time - it->second)
                              .count();

        if (metrics.latency_samples_ns.size() < MAX_LATENCY_SAMPLES) {
          metrics.latency_samples_ns.push_back(latency_ns);
        }

        pending_requests.erase(it);
      }
    }
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (posted < MAX_posted_WRS) {
      auto post_gap = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          now - last_post_time)
                          .count();

      if (sleep_ns == 0 || post_gap >= static_cast<int64_t>(sleep_ns)) {
        auto post_time = std::chrono::high_resolution_clock::now();
        size_t offset = (wr_id % 8) * MSG_SIZE;

        if (post_rdma_write(rctx, wr_id, offset)) {
          post_times[wr_id] = post_time;
          posted++;
          metrics.write_count++;
          wr_id++;
          last_post_time = post_time;
        }
      }
    }
#else
    auto time_since_last_post =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now -
                                                             last_post_time)
            .count();
    // only allowed to continue posting RDMA WRs under certain  MOPS control
    if (outstanding < MAX_OUTSTANDING_WRS &&
        (sleep_ns == 0 || time_since_last_post >= (int64_t)sleep_ns)) {
      size_t offset = (wr_id % 8) * MSG_SIZE;
      auto post_time = std::chrono::high_resolution_clock::now();

      if (post_rdma_write(rctx, wr_id, offset)) {
        pending_requests[wr_id] = post_time;
        outstanding++;
        metrics.write_count++;
        wr_id++;
        last_post_time = post_time;
      }
    }
#endif
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  while (posted > 0) {
#else
  while (outstanding > 0) {
#endif
    std::vector<uint64_t> completed_wr_ids;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    int completed = poll_completions(rctx, posted, completed_wr_ids);
    if (completed > 0) {
      auto completion_time = std::chrono::high_resolution_clock::now();
      posted -= completed;
      metrics.completion_count += completed;

      for (uint64_t id : completed_wr_ids) {
        auto it = post_times.find(id);
        if (it != post_times.end()) {
          auto latency_ns =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  completion_time - it->second)
                  .count();

          if (metrics.latency_samples_ns.size() < MAX_LATENCY_SAMPLES) {
            metrics.latency_samples_ns.push_back(latency_ns);
          }
          post_times.erase(it);
        }
      }
    }
#else
    int completed = poll_completions(rctx, outstanding, completed_wr_ids);
    outstanding -= completed;
    metrics.completion_count += completed;

    auto completion_time = std::chrono::high_resolution_clock::now();
    for (uint64_t id : completed_wr_ids) {
      auto it = pending_requests.find(id);
      if (it != pending_requests.end()) {
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              completion_time - it->second)
                              .count();

        if (metrics.latency_samples_ns.size() < MAX_LATENCY_SAMPLES) {
          metrics.latency_samples_ns.push_back(latency_ns);
        }

        pending_requests.erase(it);
      }
    }
#endif
  }
}

double calculate_percentile(std::vector<uint64_t>& samples, int percentile) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  size_t idx = (samples.size() * percentile) / 100;
  if (idx >= samples.size()) idx = samples.size() - 1;
  return static_cast<double>(samples[idx]);
}

void run_benchmark(int rank, int peer_rank, char const* peer_ip,
                   float target_mops, int num_threads, int test_duration_ms,
                   int local_rank
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
                   ,
                   bool print_results = true
#endif
) {
  // calculate sleep time for controlling MOPS, only rough approximation
  float ops_per_thread_per_sec = (target_mops * 1e6f) / num_threads;
  float sleep_time_sec = 1.0f / ops_per_thread_per_sec;
  uint64_t sleep_ns = static_cast<uint64_t>(sleep_time_sec * 1e9f);

  size_t buffer_size = 64 * 1024 * 1024;
  void* gpu_buf = nullptr;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  hipError_t hip_err = hipMalloc(&gpu_buf, buffer_size);
  if (hip_err != hipSuccess) {
    fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(hip_err));
    return;
  }
  hipMemset(gpu_buf, 0xAB, buffer_size);
#else
  cudaMalloc(&gpu_buf, buffer_size);
  cudaMemset(gpu_buf, 0xAB, buffer_size);
#endif

  std::vector<RDMAContext> contexts(num_threads);
  std::vector<RDMAConnectionInfo> local_infos(num_threads);
  std::vector<RDMAConnectionInfo> remote_infos(num_threads);

  for (int i = 0; i < num_threads; i++) {
    void* thread_buf =
        static_cast<char*>(gpu_buf) + i * (buffer_size / num_threads);
    if (!init_rdma_context(&contexts[i], thread_buf, buffer_size / num_threads,
                           local_rank, i)) {
      fprintf(stderr, "Failed to init RDMA context %d\n", i);
      return;
    }

    if (!create_qp(&contexts[i])) {
      fprintf(stderr, "Failed to create QP %d\n", i);
      return;
    }

    local_infos[i].qp_num = contexts[i].qp->qp_num;
    local_infos[i].rkey = contexts[i].mr->rkey;
    local_infos[i].addr = reinterpret_cast<uint64_t>(thread_buf);
    local_infos[i].len = buffer_size / num_threads;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    int gid_index = 3;
    contexts[i].gid_index = gid_index;
#else
    int gid_index = 0;
#endif
    ibv_gid gid;
    ibv_query_gid(contexts[i].ctx, 1, gid_index, &gid);
    memcpy(local_infos[i].gid, &gid, 16);
  }

  for (int i = 0; i < num_threads; i++) {
    exchange_connection_info(rank * num_threads + i,
                             peer_rank * num_threads + i, peer_ip,
                             &local_infos[i], &remote_infos[i]);

    contexts[i].remote_addr = remote_infos[i].addr;
    contexts[i].remote_qpn = remote_infos[i].qp_num;
    contexts[i].remote_rkey = remote_infos[i].rkey;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (!connect_qp(&contexts[i], remote_infos[i].gid,
                    remote_infos[i].qp_num)) {
      fprintf(stderr, "Failed to connect QP %d\n", i);
      return;
    }
#else
    if (!create_ah(&contexts[i], remote_infos[i].gid)) {
      fprintf(stderr, "Failed to create AH %d\n", i);
      return;
    }
#endif
  }

  std::vector<ThreadMetrics> metrics(num_threads);
  std::atomic<bool> stop_flag(false);
  std::vector<std::thread> threads;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(working_thread, i, &contexts[i], sleep_ns,
                         test_duration_ms, std::ref(stop_flag),
                         std::ref(metrics[i]));
  }

  std::this_thread::sleep_for(
      std::chrono::milliseconds(test_duration_ms + 500));
  stop_flag = true;

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double duration_sec =
      std::chrono::duration<double>(end_time - start_time).count();

  std::vector<uint64_t> all_samples;
  uint64_t total_writes = 0;

  for (auto& m : metrics) {
    total_writes += m.write_count;
    all_samples.insert(all_samples.end(), m.latency_samples_ns.begin(),
                       m.latency_samples_ns.end());
  }

  double actual_mops = total_writes / duration_sec / 1e6;

  double avg_latency_ns = 0.0;
  if (!all_samples.empty()) {
    uint64_t sum = 0;
    for (auto s : all_samples) sum += s;
    avg_latency_ns = static_cast<double>(sum) / all_samples.size();
  }

  double p99_latency_ns = calculate_percentile(all_samples, 99);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (rank == 0 && print_results) {
#else
  if (rank == 0) {
#endif
    printf("%11.1f | %7d | %11.2f | %16.0f | %16.0f\n", target_mops,
           num_threads, actual_mops, avg_latency_ns, p99_latency_ns);
  }

  for (int i = 0; i < num_threads; i++) {
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
    if (contexts[i].ah) ibv_destroy_ah(contexts[i].ah);
#endif
    if (contexts[i].qp) ibv_destroy_qp(contexts[i].qp);
    if (contexts[i].cq) ibv_destroy_cq(contexts[i].cq);
    if (contexts[i].mr) ibv_dereg_mr(contexts[i].mr);
    if (contexts[i].pd) ibv_dealloc_pd(contexts[i].pd);
    if (contexts[i].ctx) ibv_close_device(contexts[i].ctx);
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  hipFree(gpu_buf);
#else
  cudaFree(gpu_buf);
#endif
}

int main(int argc, char** argv) {
  int rank = 0;
  int world_size = 2;

  char const* rank_env = std::getenv("RANK");
  if (rank_env) rank = std::atoi(rank_env);

  char const* world_size_env = std::getenv("WORLD_SIZE");
  if (world_size_env) world_size = std::atoi(world_size_env);

  if (world_size != 2) {
    if (rank == 0) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
      fprintf(stderr, "more processes needed\n");
#else
      fprintf(stderr, "Error: This benchmark requires 2 processes\n");
#endif
    }
    return 1;
  }

  int local_rank = 0;

  char const* peer_ip = std::getenv("PEER_IP");
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (!peer_ip) exit(1);
  hipSetDevice(0);
#else
  if (!peer_ip) exit(1);

  cudaSetDevice(0);
#endif

  int peer_rank = 1 - rank;

  if (rank == 0) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    printf("RDMA/RoCE Network Latency Benchmark (Broadcom)\n");
    printf("Message Size: %zu bytes (7KB)\n", MSG_SIZE);
    printf("Peer IP: %s\n\n", peer_ip);
#else
    printf("Message Size: %zu bytes (7KB)\n", MSG_SIZE);
#endif
    printf(
        "Target Mops | Threads | Actual Mops | Avg Latency (ns) | P99 Latency "
        "(ns)\n");
    printf(
        "----------------------------------------------------------------------"
        "--\n");
  }

  int num_threads = 32;
  int test_duration_ms = 5000;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (rank == 0) {
    printf("Warming up...\n");
    fflush(stdout);
  }
  run_benchmark(rank, peer_rank, peer_ip, 1.0f, num_threads, 2000, local_rank,
                false);
  hipDeviceSynchronize();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  if (rank == 0) {
    printf("Warmup done.\n\nStarting benchmark:\n");
    printf(
        "Target Mops | Threads | Actual Mops | Avg Latency (ns) | P99 Latency "
        "(ns)\n");
    printf(
        "----------------------------------------------------------------------"
        "--\n");
    fflush(stdout);
  }

  for (float target_mops = 0.5f; target_mops <= 355.0f; target_mops += 0.5f) {
    run_benchmark(rank, peer_rank, peer_ip, target_mops, num_threads,
                  test_duration_ms, local_rank);

    hipDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
#else
  for (float target_mops = 0.5f; target_mops <= 25.0f; target_mops += 0.5f) {
    run_benchmark(rank, peer_rank, peer_ip, target_mops, num_threads,
                  test_duration_ms, local_rank);

    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
#endif

  return 0;
}

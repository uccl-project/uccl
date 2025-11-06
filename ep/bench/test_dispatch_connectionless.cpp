// DeepEP Low-Latency Dispatch Pattern Test
// Simulates MoE token dispatch across nodes with random expert selection
//
// Pattern:
// - Each GPU has 128 tokens, each token is 7KB
// - Each token randomly selects 8 remote GPUs (experts) on OTHER nodes
// - Only tests inter-node RDMA (skips intra-node traffic)
// - Uses shared QP architecture (1 QP per NIC)

#include <arpa/inet.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <unistd.h>

// Configuration
constexpr int NUM_GPUS_PER_NODE = 8;
constexpr int NUM_NICS_PER_GPU = 2;
constexpr size_t TOKEN_SIZE = 7 * 1024;  // 7KB per token (DeepEP FP8 format)
constexpr int NUM_TOKENS_PER_GPU = 128;  // Each GPU has 128 tokens
constexpr int TOPK = 8;  // Each token selects 8 experts (remote GPUs)
constexpr int SLOTS_PER_SRC = 1024;
constexpr int WINDOW_PER_NIC = 256;
constexpr uint32_t QKEY = 0x11111111u;
constexpr int TCP_PORT_BASE = 18516;

#define CUDA_CHECK(cmd)                                             \
  do {                                                              \
    cudaError_t e = (cmd);                                          \
    if (e != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                               \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

// Connection info exchanged via TCP
struct RDMAConnectionInfo {
  uint32_t qp_num;
  uint8_t gid[16];
  uint32_t rkey;
  uint64_t addr;
  size_t len;
};

// Base NIC context with shared QP
struct BaseNicCtx {
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_mr* mr = nullptr;
  ibv_qp* qp = nullptr;
  uint32_t lkey = 0;

  std::atomic<uint64_t> posted{0};
  std::atomic<uint64_t> completed{0};
};

// Per-remote-GPU endpoint
struct PeerEndpoint {
  ibv_ah* ah = nullptr;
  uint32_t remote_qpn = 0;
  uint32_t remote_rkey = 0;
  uint64_t remote_addr = 0;
  size_t remote_len = 0;
};

// Token routing info: which remote GPUs each token should be sent to
struct TokenRoute {
  int remote_rank;
  int remote_gpu;
};

std::vector<std::string> parse_ip_list(std::string const& ip_str) {
  std::vector<std::string> ips;
  std::stringstream ss(ip_str);
  std::string ip;
  while (std::getline(ss, ip, ',')) {
    if (!ip.empty()) ips.push_back(ip);
  }
  return ips;
}

void tcp_barrier(int rank, int world_size, std::vector<std::string> const& ips,
                 int port_offset) {
  int port = TCP_PORT_BASE + port_offset;
  if (rank == 0) {
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(listenfd, (struct sockaddr*)&addr, sizeof(addr));
    listen(listenfd, world_size - 1);

    for (int i = 1; i < world_size; i++) {
      struct sockaddr_in client_addr;
      socklen_t len = sizeof(client_addr);
      int connfd = accept(listenfd, (struct sockaddr*)&client_addr, &len);
      close(connfd);
    }
    close(listenfd);
  } else {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ips[0].c_str(), &addr.sin_addr);

    while (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      usleep(100000);
    }
    close(sockfd);
  }
}

void exchange_connection_info(int my_rank, int my_gpu, int nic_id,
                              int remote_rank, int remote_gpu,
                              std::string const& peer_ip,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote) {
  int gpu_a = my_rank * NUM_GPUS_PER_NODE + my_gpu;
  int gpu_b = remote_rank * NUM_GPUS_PER_NODE + remote_gpu;
  int low_gpu = (gpu_a < gpu_b) ? gpu_a : gpu_b;
  int high_gpu = (gpu_a < gpu_b) ? gpu_b : gpu_a;
  int tid = nic_id * 100000 + low_gpu * 100 + high_gpu;
  int port = TCP_PORT_BASE + 1000 + tid;

  int sockfd;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  if (my_rank < remote_rank ||
      (my_rank == remote_rank && my_gpu < remote_gpu)) {
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    int bind_retries = 10;
    for (int i = 0; i < bind_retries; i++) {
      if (bind(listenfd, (struct sockaddr*)&addr, sizeof(addr)) == 0) break;
      usleep(100000);
    }

    listen(listenfd, 1);
    struct timeval tv = {30, 0};
    setsockopt(listenfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    socklen_t len = sizeof(addr);
    sockfd = accept(listenfd, (struct sockaddr*)&addr, &len);
    close(listenfd);

    send(sockfd, local, sizeof(*local), 0);
    recv(sockfd, remote, sizeof(*remote), MSG_WAITALL);
    close(sockfd);
  } else {
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, peer_ip.c_str(), &addr.sin_addr);

    int max_retries = 300;
    for (int retry = 0; retry < max_retries; retry++) {
      if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == 0) break;
      usleep(100000);
      close(sockfd);
      sockfd = socket(AF_INET, SOCK_STREAM, 0);
    }

    send(sockfd, local, sizeof(*local), 0);
    recv(sockfd, remote, sizeof(*remote), MSG_WAITALL);
    close(sockfd);
  }
}

void fill_local_gid(ibv_context* ctx, uint8_t* gid_out) {
  union ibv_gid gid;
  if (ibv_query_gid(ctx, 1, 0, &gid) != 0) {
    perror("ibv_query_gid");
    exit(EXIT_FAILURE);
  }
  memcpy(gid_out, &gid, 16);
}

std::vector<std::string> get_efa_device_names() {
  std::vector<std::string> names;
  int num_devices = 0;
  ibv_device** dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list) return names;

  for (int i = 0; i < num_devices; i++) {
    char const* name = ibv_get_device_name(dev_list[i]);
    if (strstr(name, "rdmap") != nullptr) {
      names.push_back(name);
    }
  }
  ibv_free_device_list(dev_list);
  return names;
}

ibv_context* open_device_by_name(std::string const& dev_name) {
  int num_devices = 0;
  ibv_device** dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list) return nullptr;

  ibv_context* ctx = nullptr;
  for (int i = 0; i < num_devices; i++) {
    if (strcmp(ibv_get_device_name(dev_list[i]), dev_name.c_str()) == 0) {
      ctx = ibv_open_device(dev_list[i]);
      break;
    }
  }
  ibv_free_device_list(dev_list);
  return ctx;
}

ibv_qp* create_srd_qp_ex(BaseNicCtx& base) {
  ibv_qp_init_attr_ex qp_attr_ex = {};
  efadv_qp_init_attr efa_attr = {};

  qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                              IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                              IBV_QP_EX_WITH_SEND_WITH_IMM;

  qp_attr_ex.cap.max_send_wr = WINDOW_PER_NIC * 2;
  qp_attr_ex.cap.max_recv_wr = WINDOW_PER_NIC * 2;
  qp_attr_ex.cap.max_send_sge = 1;
  qp_attr_ex.cap.max_recv_sge = 1;
  qp_attr_ex.cap.max_inline_data = 0;

  qp_attr_ex.pd = base.pd;
  qp_attr_ex.qp_context = base.context;
  qp_attr_ex.sq_sig_all = 1;

  qp_attr_ex.send_cq = base.cq;
  qp_attr_ex.recv_cq = base.cq;
  qp_attr_ex.qp_type = IBV_QPT_DRIVER;

  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  efa_attr.flags = EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

  ibv_qp* qp = efadv_create_qp_ex(base.context, &qp_attr_ex, &efa_attr,
                                  sizeof(efadv_qp_init_attr));
  if (!qp) return nullptr;

  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qkey = QKEY;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
    ibv_destroy_qp(qp);
    return nullptr;
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
    ibv_destroy_qp(qp);
    return nullptr;
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
    ibv_destroy_qp(qp);
    return nullptr;
  }

  return qp;
}

ibv_ah* create_ah(ibv_pd* pd, uint8_t const* gid_bytes) {
  union ibv_gid gid;
  memcpy(&gid, gid_bytes, 16);

  ibv_ah_attr ah_attr = {};
  ah_attr.is_global = 1;
  ah_attr.grh.dgid = gid;
  ah_attr.grh.sgid_index = 0;
  ah_attr.grh.hop_limit = 255;
  ah_attr.port_num = 1;

  return ibv_create_ah(pd, &ah_attr);
}

std::atomic<int> thread_barrier_count{0};
std::atomic<bool> thread_barrier_sense{false};
std::mutex print_mutex;

void thread_barrier(int num_threads) {
  bool local_sense = !thread_barrier_sense.load();
  int count = thread_barrier_count.fetch_add(1) + 1;

  if (count == num_threads) {
    thread_barrier_count.store(0);
    thread_barrier_sense.store(local_sense);
  } else {
    while (thread_barrier_sense.load() != local_sense) {
      std::this_thread::yield();
    }
  }
}

void run_gpu_thread(int gpu_id, int my_rank, int num_nodes,
                    std::vector<std::string> const& node_ips, int seed) {
  CUDA_CHECK(cudaSetDevice(gpu_id));

  // Generate routing table: each token randomly selects TOPK experts from ALL GPUs
  std::mt19937 rng(seed + my_rank * NUM_GPUS_PER_NODE + gpu_id);

  // Build list of ALL GPUs (experts) across all nodes
  std::vector<std::pair<int, int>> all_gpus;  // (rank, gpu_id)
  for (int r = 0; r < num_nodes; r++) {
    for (int g = 0; g < NUM_GPUS_PER_NODE; g++) {
      all_gpus.push_back({r, g});
    }
  }

  // For each token, randomly select TOPK experts
  // Only add to route if expert is on a different node 
  std::vector<std::vector<TokenRoute>> token_routes(NUM_TOKENS_PER_GPU);
  uint64_t total_selections = 0;
  uint64_t intra_node_selections = 0;

  for (int t = 0; t < NUM_TOKENS_PER_GPU; t++) {
    std::vector<std::pair<int, int>> candidates = all_gpus;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    // Select top-K experts
    for (int k = 0; k < TOPK && k < (int)candidates.size(); k++) {
      total_selections++;

      // Only send via RDMA if expert is on a different node
      if (candidates[k].first != my_rank) {
        TokenRoute route;
        route.remote_rank = candidates[k].first;
        route.remote_gpu = candidates[k].second;
        token_routes[t].push_back(route);
      } else {
        intra_node_selections++;
      }
    }
  }

  // Calculate buffer sizes
  size_t send_size = NUM_TOKENS_PER_GPU * TOKEN_SIZE;
  size_t recv_size = TOKEN_SIZE * SLOTS_PER_SRC * num_nodes * NUM_GPUS_PER_NODE;
  size_t total_size = send_size + recv_size;

  void* gpu_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&gpu_buf, total_size));
  void* gpu_send_buf = gpu_buf;
  void* gpu_recv_buf = (char*)gpu_buf + send_size;
  CUDA_CHECK(cudaMemset(gpu_send_buf, gpu_id + 1, send_size));
  CUDA_CHECK(cudaMemset(gpu_recv_buf, 0, recv_size));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::string> efa_devs = get_efa_device_names();
  if (efa_devs.size() < 2) {
    fprintf(stderr, "[GPU %d] ERROR: Need at least 2 EFA devices, found %zu\n",
            gpu_id, efa_devs.size());
    exit(EXIT_FAILURE);
  }

  int nic0_idx = (2 * gpu_id) % (int)efa_devs.size();
  int nic1_idx = (2 * gpu_id + 1) % (int)efa_devs.size();

  BaseNicCtx base_nics[NUM_NICS_PER_GPU];

  // Initialize NICs
  for (int nic = 0; nic < NUM_NICS_PER_GPU; nic++) {
    int dev_idx = (nic == 0) ? nic0_idx : nic1_idx;
    std::string dev_name = efa_devs[dev_idx];

    base_nics[nic].context = open_device_by_name(dev_name);
    if (!base_nics[nic].context) {
      fprintf(stderr, "[GPU %d] Failed to open device %s\n", gpu_id, dev_name.c_str());
      exit(EXIT_FAILURE);
    }

    base_nics[nic].pd = ibv_alloc_pd(base_nics[nic].context);
    if (!base_nics[nic].pd) exit(EXIT_FAILURE);

    uint64_t iova = (uintptr_t)gpu_buf;
    base_nics[nic].mr = ibv_reg_mr_iova2(
        base_nics[nic].pd, gpu_buf, total_size, iova,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_RELAXED_ORDERING);
    if (!base_nics[nic].mr) exit(EXIT_FAILURE);
    base_nics[nic].lkey = base_nics[nic].mr->lkey;

    ibv_cq_init_attr_ex cq_ex_attr = {};
    cq_ex_attr.cqe = WINDOW_PER_NIC * 4;
    base_nics[nic].cq = (ibv_cq*)ibv_create_cq_ex(base_nics[nic].context, &cq_ex_attr);
    if (!base_nics[nic].cq) exit(EXIT_FAILURE);

    base_nics[nic].qp = create_srd_qp_ex(base_nics[nic]);
    if (!base_nics[nic].qp) exit(EXIT_FAILURE);
  }

  // Setup peer endpoints
  std::vector<std::vector<std::vector<PeerEndpoint>>> peers(
      NUM_NICS_PER_GPU,
      std::vector<std::vector<PeerEndpoint>>(
          num_nodes, std::vector<PeerEndpoint>(NUM_GPUS_PER_NODE)));

  thread_barrier(NUM_GPUS_PER_NODE);
  if (gpu_id == 0) {
    tcp_barrier(my_rank, num_nodes, node_ips, 4000);
  }
  thread_barrier(NUM_GPUS_PER_NODE);

  // Establish connections only to needed remote GPUs
  for (int nic = 0; nic < NUM_NICS_PER_GPU; nic++) {
    for (int remote_rank = 0; remote_rank < num_nodes; remote_rank++) {
      if (remote_rank == my_rank) continue;

      for (int remote_gpu = 0; remote_gpu < NUM_GPUS_PER_NODE; remote_gpu++) {
        RDMAConnectionInfo local_info = {};
        local_info.qp_num = base_nics[nic].qp->qp_num;
        local_info.rkey = base_nics[nic].mr->rkey;
        local_info.addr = (uint64_t)gpu_buf;
        local_info.len = total_size;
        fill_local_gid(base_nics[nic].context, local_info.gid);

        RDMAConnectionInfo remote_info = {};
        exchange_connection_info(my_rank, gpu_id, nic, remote_rank, remote_gpu,
                                 node_ips[remote_rank], &local_info, &remote_info);

        peers[nic][remote_rank][remote_gpu].ah = create_ah(base_nics[nic].pd, remote_info.gid);
        if (!peers[nic][remote_rank][remote_gpu].ah) exit(EXIT_FAILURE);

        peers[nic][remote_rank][remote_gpu].remote_qpn = remote_info.qp_num;
        peers[nic][remote_rank][remote_gpu].remote_rkey = remote_info.rkey;
        peers[nic][remote_rank][remote_gpu].remote_addr = remote_info.addr;
        peers[nic][remote_rank][remote_gpu].remote_len = remote_info.len;
      }
    }
  }

  thread_barrier(NUM_GPUS_PER_NODE);
  if (gpu_id == 0) {
    tcp_barrier(my_rank, num_nodes, node_ips, 5000);
  }
  thread_barrier(NUM_GPUS_PER_NODE);

  auto t_start = std::chrono::high_resolution_clock::now();

  // Calculate total messages to send
  uint64_t total_msgs = 0;
  for (int t = 0; t < NUM_TOKENS_PER_GPU; t++) {
    total_msgs += token_routes[t].size();
  }

  // Track messages sent per token
  std::vector<int> token_progress(NUM_TOKENS_PER_GPU, 0);
  int current_token = 0;

  int inflight_per_nic[NUM_NICS_PER_GPU] = {0};
  uint64_t total_posted = 0;
  uint64_t total_completed = 0;

  while (total_completed < total_msgs) {
    // Post phase: send tokens to their selected remote GPUs
    while (total_posted < total_msgs) {
      // Find next token that needs to be sent
      while (current_token < NUM_TOKENS_PER_GPU &&
             token_progress[current_token] >= (int)token_routes[current_token].size()) {
        current_token++;
      }
      if (current_token >= NUM_TOKENS_PER_GPU) break;

      TokenRoute& route = token_routes[current_token][token_progress[current_token]];

      // Striping: alternate NICs
      int nic = (token_progress[current_token] / 4) % NUM_NICS_PER_GPU;

      if (inflight_per_nic[nic] >= WINDOW_PER_NIC) break;

      // Post RDMA WRITE
      int sender_global_gpu_id = my_rank * NUM_GPUS_PER_NODE + gpu_id;
      uint64_t slot = token_progress[current_token] % SLOTS_PER_SRC;
      uint64_t remote_offset = send_size +
                               sender_global_gpu_id * SLOTS_PER_SRC * TOKEN_SIZE +
                               slot * TOKEN_SIZE;
      uint64_t remote_addr = peers[nic][route.remote_rank][route.remote_gpu].remote_addr +
                             remote_offset;

      ibv_qp_ex* qpx = (ibv_qp_ex*)base_nics[nic].qp;
      ibv_wr_start(qpx);

      qpx->wr_id = ((uint64_t)nic << 56) | ((uint64_t)current_token << 32) |
                   (token_progress[current_token] & 0xFFFF);
      qpx->wr_flags = IBV_SEND_SIGNALED;
      qpx->comp_mask = 0;
      ibv_wr_rdma_write(qpx, peers[nic][route.remote_rank][route.remote_gpu].remote_rkey,
                        remote_addr);
      ibv_wr_set_ud_addr(qpx, peers[nic][route.remote_rank][route.remote_gpu].ah,
                         peers[nic][route.remote_rank][route.remote_gpu].remote_qpn, QKEY);

      // Send the specific token
      uint64_t local_offset = current_token * TOKEN_SIZE;
      ibv_wr_set_sge(qpx, base_nics[nic].lkey, (uintptr_t)gpu_send_buf + local_offset,
                     TOKEN_SIZE);

      int ret = ibv_wr_complete(qpx);
      if (ret) {
        fprintf(stderr, "[GPU %d] ibv_wr_complete failed: %s\n", gpu_id, strerror(ret));
        exit(EXIT_FAILURE);
      }

      token_progress[current_token]++;
      inflight_per_nic[nic]++;
      total_posted++;
      base_nics[nic].posted.fetch_add(1, std::memory_order_relaxed);
    }

    // Poll phase
    for (int nic = 0; nic < NUM_NICS_PER_GPU; nic++) {
      ibv_cq_ex* cqx = (ibv_cq_ex*)base_nics[nic].cq;
      ibv_poll_cq_attr poll_attr = {.comp_mask = 0};
      int ret = ibv_start_poll(cqx, &poll_attr);
      if (ret) continue;

      while (true) {
        if (cqx->status != IBV_WC_SUCCESS) {
          fprintf(stderr, "[GPU %d] CQE error: %s\n", gpu_id,
                  ibv_wc_status_str(cqx->status));
          exit(EXIT_FAILURE);
        }

        uint32_t opcode = ibv_wc_read_opcode(cqx);
        if (opcode == IBV_WC_RDMA_WRITE) {
          inflight_per_nic[nic]--;
          total_completed++;
          base_nics[nic].completed.fetch_add(1, std::memory_order_relaxed);
        }

        ret = ibv_next_poll(cqx);
        if (ret) break;
      }
      ibv_end_poll(cqx);
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  thread_barrier(NUM_GPUS_PER_NODE);
  if (gpu_id == 0) {
    tcp_barrier(my_rank, num_nodes, node_ips, 6000);
  }
  thread_barrier(NUM_GPUS_PER_NODE);

  uint64_t total_bytes = total_msgs * TOKEN_SIZE;
  double bw_gbps = (total_bytes / elapsed) / (1024.0 * 1024.0 * 1024.0);

  {
    std::lock_guard<std::mutex> lock(print_mutex);
    printf("\n=== [Rank %d GPU %d] DeepEP Dispatch Pattern Results ===\n", my_rank, gpu_id);
    printf("  Elapsed: %.3f ms\n", elapsed * 1000.0);
    printf("  Num tokens: %d\n", NUM_TOKENS_PER_GPU);
    printf("  Token size: %zu bytes (7KB)\n", TOKEN_SIZE);
    printf("  Top-K: %d\n", TOPK);
    printf("  Total expert selections: %lu\n", total_selections);
    printf("  Intra-node selections (skipped): %lu (%.1f%%)\n",
           intra_node_selections,
           100.0 * intra_node_selections / total_selections);
    printf("  Inter-node RDMA dispatches: %lu (%.1f%%)\n",
           total_msgs,
           100.0 * total_msgs / total_selections);
    printf("  Total bytes: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("  Bandwidth: %.3f GB/s\n", bw_gbps);
    printf("  Avg latency per token: %.2f us\n", (elapsed * 1e6) / NUM_TOKENS_PER_GPU);
    printf("  NIC0 posted=%lu, completed=%lu\n", base_nics[0].posted.load(),
           base_nics[0].completed.load());
    printf("  NIC1 posted=%lu, completed=%lu\n", base_nics[1].posted.load(),
           base_nics[1].completed.load());
  }

  // Cleanup
  for (int nic = 0; nic < NUM_NICS_PER_GPU; nic++) {
    for (int r = 0; r < num_nodes; r++) {
      if (r == my_rank) continue;
      for (int g = 0; g < NUM_GPUS_PER_NODE; g++) {
        if (peers[nic][r][g].ah) ibv_destroy_ah(peers[nic][r][g].ah);
      }
    }
    if (base_nics[nic].qp) ibv_destroy_qp(base_nics[nic].qp);
    if (base_nics[nic].cq) ibv_destroy_cq(base_nics[nic].cq);
    if (base_nics[nic].mr) ibv_dereg_mr(base_nics[nic].mr);
    if (base_nics[nic].pd) ibv_dealloc_pd(base_nics[nic].pd);
    if (base_nics[nic].context) ibv_close_device(base_nics[nic].context);
  }

  CUDA_CHECK(cudaFree(gpu_buf));
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <my_rank> <node_ips> [seed]\n", argv[0]);
    printf("  Example: %s 0 10.1.1.1,10.1.1.2 42\n", argv[0]);
    return 1;
  }

  int my_rank = atoi(argv[1]);
  std::vector<std::string> node_ips = parse_ip_list(argv[2]);
  int seed = (argc > 3) ? atoi(argv[3]) : 42;

  if (node_ips.size() < 2) {
    fprintf(stderr, "ERROR: Need at least 2 nodes for inter-node RDMA test\n");
    return 1;
  }

  int num_nodes = (int)node_ips.size();

  printf("=== DeepEP Low-Latency Dispatch Pattern Test ===\n");
  printf("Total nodes: %d\n", num_nodes);
  printf("My rank: %d\n", my_rank);
  printf("Config: %d tokens/GPU, %zu bytes/token (7KB), Top-%d routing\n",
         NUM_TOKENS_PER_GPU, TOKEN_SIZE, TOPK);
  printf("Pattern: Each token randomly selects %d experts from ALL %d GPUs\n",
         TOPK, num_nodes * NUM_GPUS_PER_NODE);
  printf("         Intra-node selections are skipped (no RDMA)\n");
  printf("         Only inter-node selections are sent via RDMA\n");
  printf("Random seed: %d\n", seed);

  std::vector<std::thread> threads;
  for (int gpu = 0; gpu < NUM_GPUS_PER_NODE; gpu++) {
    threads.emplace_back(run_gpu_thread, gpu, my_rank, num_nodes, node_ips, seed);
  }

  for (auto& t : threads) {
    t.join();
  }

  return 0;
}

// DeepEP Low-Latency Dispatch Pattern Test - Multi-QP Version
// Based on test_dispatch_connectionless.cpp architecture
// Uses DeepEP's one-QP-per-expert design (matches test_low_latency.py)
//
// Architecture:
// - Multiple GPU threads (one per GPU), similar to test_dispatch_connectionless.cpp
// - Each GPU has 36 local experts, each expert has its own QP
// - Each local expert ONLY connects to corresponding expert on remote nodes
//   Example: Rank0 GPU0 expert5 <-> Rank1 GPU0 expert5, Rank2 GPU0 expert5
// - Total QPs per GPU: 36
// - Total connections per GPU: 36 experts × (N-1) remote nodes
//
// Parameters 
// - 128 tokens per GPU, 7KB per token
// - Top-8 expert selection
// - 36 experts per GPU (288 total experts across 8 GPUs)

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

// Configuration (match DeepEP)
constexpr int NUM_GPUS_PER_NODE = 8;
constexpr int NUM_EXPERTS_PER_GPU = 36;  // 288 experts / 8 GPUs = 36 per GPU
constexpr size_t TOKEN_SIZE = 7 * 1024;  // 7KB per token
constexpr int NUM_TOKENS_PER_GPU = 128;
constexpr int TOPK = 8;
constexpr int SLOTS_PER_SRC = 1024;
constexpr int WINDOW_PER_EXPERT = 16;  // In-flight ops per expert QP
constexpr uint32_t QKEY = 0x11111111u;
constexpr int TCP_PORT_BASE = 18517;

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

// Per-expert QP context
struct ExpertQPCtx {
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_mr* mr = nullptr;
  uint32_t lkey = 0;

  std::atomic<uint64_t> posted{0};
  std::atomic<uint64_t> completed{0};
  std::atomic<int> inflight{0};
};

// Per-remote-expert endpoint
struct PeerEndpoint {
  ibv_ah* ah = nullptr;
  uint32_t remote_qpn = 0;
  uint32_t remote_rkey = 0;
  uint64_t remote_addr = 0;
  size_t remote_len = 0;
};

// Token routing info
struct TokenRoute {
  int remote_rank;
  int remote_gpu;
  int remote_expert;  // Expert ID on remote GPU (0-35)
  int local_expert;   // Which local expert QP to use (0-35)
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

void exchange_connection_info(int my_rank, int my_gpu, int my_expert,
                              int remote_rank, int remote_gpu, int remote_expert,
                              std::string const& peer_ip,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote) {
  int my_global_gpu = my_rank * NUM_GPUS_PER_NODE + my_gpu;
  int remote_global_gpu = remote_rank * NUM_GPUS_PER_NODE + remote_gpu;
  int low_global_gpu = (my_global_gpu < remote_global_gpu) ? my_global_gpu : remote_global_gpu;
  int high_global_gpu = (my_global_gpu < remote_global_gpu) ? remote_global_gpu : my_global_gpu;

  int pair_id = low_global_gpu * 31 - (low_global_gpu * (low_global_gpu - 1)) / 2 + (high_global_gpu - low_global_gpu - 1);
  int port = TCP_PORT_BASE + pair_id * NUM_EXPERTS_PER_GPU + my_expert;

  int sockfd;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  if (my_rank < remote_rank || (my_rank == remote_rank && my_gpu < remote_gpu)) {
    // This side listens
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(listenfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      perror("bind");
      fprintf(stderr, "Failed to bind port %d for expert %d\n", port, my_expert);
      exit(EXIT_FAILURE);
    }

    listen(listenfd, 1);
    struct timeval tv = {30, 0};
    setsockopt(listenfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    socklen_t len = sizeof(addr);
    sockfd = accept(listenfd, (struct sockaddr*)&addr, &len);
    if (sockfd < 0) {
      perror("accept");
      exit(EXIT_FAILURE);
    }
    close(listenfd);

    send(sockfd, local, sizeof(*local), 0);
    recv(sockfd, remote, sizeof(*remote), MSG_WAITALL);
    close(sockfd);
  } else {
    // This side connects
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

ibv_qp* create_srd_qp_ex(ibv_context* context, ibv_pd* pd, ibv_cq* cq) {
  ibv_qp_init_attr_ex qp_attr_ex = {};
  efadv_qp_init_attr efa_attr = {};

  qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                              IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                              IBV_QP_EX_WITH_SEND_WITH_IMM;

  qp_attr_ex.cap.max_send_wr = WINDOW_PER_EXPERT * 2;
  qp_attr_ex.cap.max_recv_wr = WINDOW_PER_EXPERT * 2;
  qp_attr_ex.cap.max_send_sge = 1;
  qp_attr_ex.cap.max_recv_sge = 1;
  qp_attr_ex.cap.max_inline_data = 0;

  qp_attr_ex.pd = pd;
  qp_attr_ex.qp_context = context;
  qp_attr_ex.sq_sig_all = 1;

  qp_attr_ex.send_cq = cq;
  qp_attr_ex.recv_cq = cq;
  qp_attr_ex.qp_type = IBV_QPT_DRIVER;

  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  efa_attr.flags = EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

  ibv_qp* qp = efadv_create_qp_ex(context, &qp_attr_ex, &efa_attr,
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

  // Calculate total experts across all GPUs
  int total_experts = num_nodes * NUM_GPUS_PER_NODE * NUM_EXPERTS_PER_GPU;

  // Generate routing table: each token randomly selects TOPK experts from ALL experts
  std::mt19937 rng(seed + my_rank * NUM_GPUS_PER_NODE + gpu_id);

  // Build list of ALL experts across all nodes
  // Each expert is identified by (rank, gpu, expert_id)
  struct ExpertID {
    int rank;
    int gpu;
    int expert;  // 0-35
  };
  std::vector<ExpertID> all_experts;
  for (int r = 0; r < num_nodes; r++) {
    for (int g = 0; g < NUM_GPUS_PER_NODE; g++) {
      for (int e = 0; e < NUM_EXPERTS_PER_GPU; e++) {
        all_experts.push_back({r, g, e});
      }
    }
  }

  // For each token, randomly select TOPK experts
  std::vector<std::vector<TokenRoute>> token_routes(NUM_TOKENS_PER_GPU);
  uint64_t total_selections = 0;
  uint64_t intra_node_selections = 0;

  for (int t = 0; t < NUM_TOKENS_PER_GPU; t++) {
    std::vector<ExpertID> candidates = all_experts;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    for (int k = 0; k < TOPK && k < (int)candidates.size(); k++) {
      total_selections++;

      // Only send via RDMA if expert is on a different node
      if (candidates[k].rank != my_rank) {
        TokenRoute route;
        route.remote_rank = candidates[k].rank;
        route.remote_gpu = candidates[k].gpu;
        route.remote_expert = candidates[k].expert;
        // DeepEP style: use corresponding local expert for routing
        route.local_expert = route.remote_expert;
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

  // Create QPs for each local expert
  std::vector<std::unique_ptr<ExpertQPCtx>> expert_qps;
  expert_qps.reserve(NUM_EXPERTS_PER_GPU);

  for (int e = 0; e < NUM_EXPERTS_PER_GPU; e++) {
    auto expert_qp = std::make_unique<ExpertQPCtx>();

    // Distribute experts across EFA devices
    int efa_idx = (e * efa_devs.size()) / NUM_EXPERTS_PER_GPU;
    std::string dev_name = efa_devs[efa_idx];

    expert_qp->context = open_device_by_name(dev_name);
    if (!expert_qp->context) {
      fprintf(stderr, "[GPU %d] Failed to open device %s for expert %d\n",
              gpu_id, dev_name.c_str(), e);
      exit(EXIT_FAILURE);
    }

    expert_qp->pd = ibv_alloc_pd(expert_qp->context);
    if (!expert_qp->pd) exit(EXIT_FAILURE);

    uint64_t iova = (uintptr_t)gpu_buf;
    expert_qp->mr = ibv_reg_mr_iova2(
        expert_qp->pd, gpu_buf, total_size, iova,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_RELAXED_ORDERING);
    if (!expert_qp->mr) exit(EXIT_FAILURE);
    expert_qp->lkey = expert_qp->mr->lkey;

    ibv_cq_init_attr_ex cq_ex_attr = {};
    cq_ex_attr.cqe = WINDOW_PER_EXPERT * 4;
    expert_qp->cq = (ibv_cq*)ibv_create_cq_ex(expert_qp->context, &cq_ex_attr);
    if (!expert_qp->cq) exit(EXIT_FAILURE);

    expert_qp->qp = create_srd_qp_ex(expert_qp->context, expert_qp->pd, expert_qp->cq);
    if (!expert_qp->qp) {
      fprintf(stderr, "[GPU %d] Failed to create QP for expert %d\n", gpu_id, e);
      exit(EXIT_FAILURE);
    }

    expert_qps.push_back(std::move(expert_qp));
  }

  // Setup peer endpoints: [local_expert][remote_rank]
  // each local expert only connects to same expert on remote nodes
  std::vector<std::vector<PeerEndpoint>> peers(
      NUM_EXPERTS_PER_GPU, std::vector<PeerEndpoint>(num_nodes));

  thread_barrier(NUM_GPUS_PER_NODE);
  if (gpu_id == 0) {
    tcp_barrier(my_rank, num_nodes, node_ips, 4000);
  }
  thread_barrier(NUM_GPUS_PER_NODE);

  // Establish connections 
  for (int local_expert = 0; local_expert < NUM_EXPERTS_PER_GPU; local_expert++) {
    for (int remote_rank = 0; remote_rank < num_nodes; remote_rank++) {
      if (remote_rank == my_rank) continue;

      // DeepEP: connect to same GPU same expert on remote node
      int remote_gpu = gpu_id;
      int remote_expert = local_expert;

      RDMAConnectionInfo local_info = {};
      local_info.qp_num = expert_qps[local_expert]->qp->qp_num;
      local_info.rkey = expert_qps[local_expert]->mr->rkey;
      local_info.addr = (uint64_t)gpu_buf;
      local_info.len = total_size;
      fill_local_gid(expert_qps[local_expert]->context, local_info.gid);

      RDMAConnectionInfo remote_info = {};
      exchange_connection_info(my_rank, gpu_id, local_expert,
                               remote_rank, remote_gpu, remote_expert,
                               node_ips[remote_rank], &local_info, &remote_info);

      peers[local_expert][remote_rank].ah =
          create_ah(expert_qps[local_expert]->pd, remote_info.gid);
      if (!peers[local_expert][remote_rank].ah) exit(EXIT_FAILURE);

      peers[local_expert][remote_rank].remote_qpn = remote_info.qp_num;
      peers[local_expert][remote_rank].remote_rkey = remote_info.rkey;
      peers[local_expert][remote_rank].remote_addr = remote_info.addr;
      peers[local_expert][remote_rank].remote_len = remote_info.len;
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

  std::vector<int> token_progress(NUM_TOKENS_PER_GPU, 0);
  int current_token = 0;

  uint64_t total_posted = 0;
  uint64_t total_completed = 0;

  while (total_completed < total_msgs) {
    // Post phase
    while (total_posted < total_msgs) {
      while (current_token < NUM_TOKENS_PER_GPU &&
             token_progress[current_token] >= (int)token_routes[current_token].size()) {
        current_token++;
      }
      if (current_token >= NUM_TOKENS_PER_GPU) break;

      TokenRoute& route = token_routes[current_token][token_progress[current_token]];
      int local_expert = route.local_expert;

      if (expert_qps[local_expert]->inflight.load() >= WINDOW_PER_EXPERT) {
        break;
      }

      // Post RDMA WRITE
      int sender_global_gpu_id = my_rank * NUM_GPUS_PER_NODE + gpu_id;
      uint64_t slot = token_progress[current_token] % SLOTS_PER_SRC;
      uint64_t remote_offset = send_size +
                               sender_global_gpu_id * SLOTS_PER_SRC * TOKEN_SIZE +
                               slot * TOKEN_SIZE;
      uint64_t remote_addr = peers[local_expert][route.remote_rank].remote_addr + remote_offset;

      ibv_qp_ex* qpx = (ibv_qp_ex*)expert_qps[local_expert]->qp;
      ibv_wr_start(qpx);

      qpx->wr_id = ((uint64_t)local_expert << 32) | current_token;
      qpx->wr_flags = IBV_SEND_SIGNALED;
      qpx->comp_mask = 0;

      ibv_wr_rdma_write(qpx, peers[local_expert][route.remote_rank].remote_rkey, remote_addr);
      ibv_wr_set_ud_addr(qpx, peers[local_expert][route.remote_rank].ah,
                         peers[local_expert][route.remote_rank].remote_qpn, QKEY);

      uint64_t local_offset = current_token * TOKEN_SIZE;
      ibv_wr_set_sge(qpx, expert_qps[local_expert]->lkey,
                     (uintptr_t)gpu_send_buf + local_offset, TOKEN_SIZE);

      int ret = ibv_wr_complete(qpx);
      if (ret) {
        fprintf(stderr, "[GPU %d] ibv_wr_complete failed: %s\n", gpu_id, strerror(ret));
        exit(EXIT_FAILURE);
      }

      token_progress[current_token]++;
      expert_qps[local_expert]->inflight.fetch_add(1);
      total_posted++;
      expert_qps[local_expert]->posted.fetch_add(1, std::memory_order_relaxed);
    }

    // Poll phase
    for (int e = 0; e < NUM_EXPERTS_PER_GPU; e++) {
      ibv_cq_ex* cqx = (ibv_cq_ex*)expert_qps[e]->cq;
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
          expert_qps[e]->inflight.fetch_sub(1);
          total_completed++;
          expert_qps[e]->completed.fetch_add(1, std::memory_order_relaxed);
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

  // Calculate QP statistics
  uint64_t total_qp_posted = 0;
  for (auto& qp : expert_qps) {
    total_qp_posted += qp->posted.load();
  }

  {
    std::lock_guard<std::mutex> lock(print_mutex);
    printf("\n=== [Rank %d GPU %d] DeepEP Multi-QP Results ===\n", my_rank, gpu_id);
    printf("  Elapsed: %.3f ms\n", elapsed * 1000.0);
    printf("  Num tokens: %d\n", NUM_TOKENS_PER_GPU);
    printf("  Token size: %zu bytes (7KB)\n", TOKEN_SIZE);
    printf("  Top-K: %d\n", TOPK);
    printf("  Local experts: %d (each with own QP)\n", NUM_EXPERTS_PER_GPU);
    printf("  Total selections: %lu\n", total_selections);
    printf("  Intra-node selections (skipped): %lu (%.1f%%)\n",
           intra_node_selections,
           100.0 * intra_node_selections / total_selections);
    printf("  Inter-node RDMA dispatches: %lu (%.1f%%)\n",
           total_msgs,
           100.0 * total_msgs / total_selections);
    printf("  Total bytes: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("  Bandwidth: %.3f GB/s\n", bw_gbps);
    printf("  Avg latency per token: %.2f us\n", (elapsed * 1e6) / NUM_TOKENS_PER_GPU);
    printf("  Connection pattern: Each local expert -> corresponding expert on %d remote nodes\n",
           num_nodes - 1);
  }

  // Cleanup
  for (int e = 0; e < NUM_EXPERTS_PER_GPU; e++) {
    for (int r = 0; r < num_nodes; r++) {
      if (r != my_rank && peers[e][r].ah) {
        ibv_destroy_ah(peers[e][r].ah);
      }
    }
    if (expert_qps[e]->qp) ibv_destroy_qp(expert_qps[e]->qp);
    if (expert_qps[e]->cq) ibv_destroy_cq(expert_qps[e]->cq);
    if (expert_qps[e]->mr) ibv_dereg_mr(expert_qps[e]->mr);
    if (expert_qps[e]->pd) ibv_dealloc_pd(expert_qps[e]->pd);
    if (expert_qps[e]->context) ibv_close_device(expert_qps[e]->context);
  }

  CUDA_CHECK(cudaFree(gpu_buf));
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <my_rank> <node_ips> [seed]\n", argv[0]);
    printf("  Example: %s 0 10.1.1.1,10.1.1.2,10.1.1.3 42\n", argv[0]);
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
  int total_experts = num_nodes * NUM_GPUS_PER_NODE * NUM_EXPERTS_PER_GPU;

  printf("=== DeepEP Multi-QP Dispatch Pattern Test ===\n");
  printf("Architecture: Multi-threaded (based on test_dispatch_connectionless.cpp)\n");
  printf("QP Design: One QP per expert (matches DeepEP test_low_latency.py)\n");
  printf("Total nodes: %d\n", num_nodes);
  printf("My rank: %d\n", my_rank);
  printf("Config: %d tokens/GPU, %zu bytes/token (7KB), Top-%d routing\n",
         NUM_TOKENS_PER_GPU, TOKEN_SIZE, TOPK);
  printf("Experts: %d per GPU, %d total experts\n",
         NUM_EXPERTS_PER_GPU, total_experts);
  printf("QPs per GPU: %d (one per expert)\n", NUM_EXPERTS_PER_GPU);
  printf("Connections per GPU: %d experts × %d remote nodes = %d\n",
         NUM_EXPERTS_PER_GPU, num_nodes - 1, NUM_EXPERTS_PER_GPU * (num_nodes - 1));
  printf("Connection pattern: Local expert N <-> Remote expert N (DeepEP style)\n");
  printf("Random seed: %d\n\n", seed);

  std::vector<std::thread> threads;
  for (int gpu = 0; gpu < NUM_GPUS_PER_NODE; gpu++) {
    threads.emplace_back(run_gpu_thread, gpu, my_rank, num_nodes, node_ips, seed);
  }

  for (auto& t : threads) {
    t.join();
  }

  printf("\n✓ All GPUs completed successfully\n");
  return 0;
}

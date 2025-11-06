// Multi-NIC EFA All-to-All RDMA Test with Shared QP Architecture
// Supports arbitrary number of nodes
//
// Architecture:
// - N nodes × 8 GPUs = N*8 total GPUs
// - Each GPU: 1 CPU thread managing 2 NICs
// - Each NIC: ONE shared QP for all destinations
// - Each (NIC, remote_GPU) pair: dedicated AH (Address Handler)
// - True all-to-all: every local GPU sends to all GPUs on all other nodes

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
#include <sstream>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <unistd.h>

// Configuration
constexpr int NUM_GPUS_PER_NODE = 8;
constexpr int NUM_NICS_PER_GPU = 2;
constexpr size_t MSG_SIZE = 4096 * 1024;  // 512KB per message
constexpr int MSGS_PER_REMOTE_GPU = 1000;
constexpr int SLOTS_PER_SRC = 1024;
constexpr int WINDOW_PER_NIC = 256;  // In-flight ops per NIC
constexpr int SIGNAL_INTERVAL = 32;
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

// Base NIC context - now includes shared QP
struct BaseNicCtx {
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_mr* mr = nullptr;
  ibv_qp* qp = nullptr;  // Shared QP for all destinations
  uint32_t lkey = 0;

  // Statistics
  std::atomic<uint64_t> posted{0};
  std::atomic<uint64_t> completed{0};
};

// Per-remote-GPU endpoint - only AH and remote info
struct PeerEndpoint {
  ibv_ah* ah = nullptr;
  uint32_t remote_qpn = 0;
  uint32_t remote_rkey = 0;
  uint64_t remote_addr = 0;
  size_t remote_len = 0;
};

// Parse "ip1,ip2,ip3,..." into vector
std::vector<std::string> parse_ip_list(std::string const& ip_str) {
  std::vector<std::string> ips;
  std::stringstream ss(ip_str);
  std::string ip;
  while (std::getline(ss, ip, ',')) {
    if (!ip.empty()) ips.push_back(ip);
  }
  return ips;
}

// Simple TCP barrier
void tcp_barrier(int rank, int world_size, std::vector<std::string> const& ips,
                 int port_offset) {
  int port = TCP_PORT_BASE + port_offset;
  if (rank == 0) {
    // Rank 0 waits for all others to connect
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
    // Others connect to rank 0
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

// Exchange connection info between two threads/GPUs
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
    // Lower rank/gpu listens
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd < 0) {
      fprintf(stderr, "[Rank %d GPU %d] socket() failed: %s\n", my_rank, my_gpu,
              strerror(errno));
      exit(EXIT_FAILURE);
    }

    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    // Retry bind in case of transient failures
    int bind_retries = 10;
    int bind_success = 0;
    for (int i = 0; i < bind_retries; i++) {
      if (bind(listenfd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        bind_success = 1;
        break;
      }
      if (errno != EADDRINUSE) {
        // Non-transient error
        break;
      }
      usleep(100000);  // Wait 100ms before retry
    }

    if (!bind_success) {
      fprintf(stderr,
              "[Rank %d GPU %d] bind() failed on port %d after %d retries: %s "
              "(for remote_rank=%d remote_gpu=%d nic=%d)\n",
              my_rank, my_gpu, port, bind_retries, strerror(errno), remote_rank,
              remote_gpu, nic_id);
      close(listenfd);
      exit(EXIT_FAILURE);
    }

    if (listen(listenfd, 1) < 0) {
      fprintf(stderr, "[Rank %d GPU %d] listen() failed: %s\n", my_rank, my_gpu,
              strerror(errno));
      close(listenfd);
      exit(EXIT_FAILURE);
    }

    // Set accept timeout
    struct timeval tv;
    tv.tv_sec = 30;  // 30 second timeout
    tv.tv_usec = 0;
    setsockopt(listenfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    socklen_t len = sizeof(addr);
    sockfd = accept(listenfd, (struct sockaddr*)&addr, &len);
    if (sockfd < 0) {
      fprintf(stderr,
              "[Rank %d GPU %d] accept() timeout waiting for remote_rank=%d "
              "remote_gpu=%d nic=%d on port %d\n",
              my_rank, my_gpu, remote_rank, remote_gpu, nic_id, port);
      close(listenfd);
      exit(EXIT_FAILURE);
    }
    close(listenfd);

    // Exchange: send local, recv remote
    if (send(sockfd, local, sizeof(*local), 0) != sizeof(*local)) {
      fprintf(stderr, "[Rank %d GPU %d] send() failed\n", my_rank, my_gpu);
      close(sockfd);
      exit(EXIT_FAILURE);
    }
    if (recv(sockfd, remote, sizeof(*remote), MSG_WAITALL) != sizeof(*remote)) {
      fprintf(stderr, "[Rank %d GPU %d] recv() failed\n", my_rank, my_gpu);
      close(sockfd);
      exit(EXIT_FAILURE);
    }
    close(sockfd);
  } else {
    // Higher rank/gpu connects
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
      fprintf(stderr, "[Rank %d GPU %d] socket() failed: %s\n", my_rank, my_gpu,
              strerror(errno));
      exit(EXIT_FAILURE);
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, peer_ip.c_str(), &addr.sin_addr);

    // Retry connect with timeout
    int max_retries = 300;  // 30 seconds total (300 * 100ms)
    int retry_count = 0;
    bool connected = false;
    while (!connected) {
      if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        connected = true;
        break;
      }

      if (retry_count >= max_retries) {
        fprintf(stderr,
                "[Rank %d GPU %d] connect() timeout to %s:%d for "
                "remote_rank=%d remote_gpu=%d nic=%d after %d retries\n",
                my_rank, my_gpu, peer_ip.c_str(), port, remote_rank, remote_gpu,
                nic_id, retry_count);
        close(sockfd);
        exit(EXIT_FAILURE);
      }

      usleep(100000);  // 100ms
      retry_count++;

      // Recreate socket for retry
      close(sockfd);
      sockfd = socket(AF_INET, SOCK_STREAM, 0);
      if (sockfd < 0) {
        fprintf(stderr, "[Rank %d GPU %d] socket() recreate failed: %s\n",
                my_rank, my_gpu, strerror(errno));
        exit(EXIT_FAILURE);
      }
    }

    // Exchange: send local, recv remote
    if (send(sockfd, local, sizeof(*local), 0) != sizeof(*local)) {
      fprintf(stderr, "[Rank %d GPU %d] send() failed\n", my_rank, my_gpu);
      close(sockfd);
      exit(EXIT_FAILURE);
    }
    if (recv(sockfd, remote, sizeof(*remote), MSG_WAITALL) != sizeof(*remote)) {
      fprintf(stderr, "[Rank %d GPU %d] recv() failed\n", my_rank, my_gpu);
      close(sockfd);
      exit(EXIT_FAILURE);
    }
    close(sockfd);
  }
}

// Fill local GID
void fill_local_gid(ibv_context* ctx, uint8_t* gid_out) {
  union ibv_gid gid;
  if (ibv_query_gid(ctx, 1, 0, &gid) != 0) {
    perror("ibv_query_gid");
    exit(EXIT_FAILURE);
  }
  memcpy(gid_out, &gid, 16);
}

// Find EFA devices
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

// Open device by name
ibv_context* open_device_by_name(std::string const& dev_name) {
  int num_devices = 0;
  ibv_device** dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list) {
    fprintf(stderr, "ibv_get_device_list failed\n");
    return nullptr;
  }

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

// Create shared SRD QP
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
  if (!qp) {
    perror("efadv_create_qp_ex");
    return nullptr;
  }

  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qkey = QKEY;
  if (ibv_modify_qp(
          qp, &attr,
          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
    perror("ibv_modify_qp INIT");
    ibv_destroy_qp(qp);
    return nullptr;
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
    perror("ibv_modify_qp RTR");
    ibv_destroy_qp(qp);
    return nullptr;
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
    perror("ibv_modify_qp RTS");
    ibv_destroy_qp(qp);
    return nullptr;
  }

  return qp;
}

// Create AH
ibv_ah* create_ah(ibv_pd* pd, uint8_t const* gid_bytes) {
  union ibv_gid gid;
  memcpy(&gid, gid_bytes, 16);

  ibv_ah_attr ah_attr = {};
  ah_attr.is_global = 1;
  ah_attr.grh.dgid = gid;
  ah_attr.grh.sgid_index = 0;
  ah_attr.grh.hop_limit = 255;
  ah_attr.port_num = 1;

  ibv_ah* ah = ibv_create_ah(pd, &ah_attr);
  if (!ah) {
    perror("ibv_create_ah");
    return nullptr;
  }
  return ah;
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
                    std::vector<std::string> const& node_ips) {
  CUDA_CHECK(cudaSetDevice(gpu_id));

  // Calculate total number of remote GPUs across all other nodes
  int num_remote_nodes = num_nodes - 1;
  int total_remote_gpus = num_remote_nodes * NUM_GPUS_PER_NODE;

  size_t send_size = MSG_SIZE;
  // Receive buffer needs space for all possible senders (all GPUs from all
  // nodes)
  size_t recv_size = MSG_SIZE * SLOTS_PER_SRC * num_nodes * NUM_GPUS_PER_NODE;
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

  // Initialize NICs and create shared QPs
  for (int nic = 0; nic < NUM_NICS_PER_GPU; nic++) {
    int dev_idx = (nic == 0) ? nic0_idx : nic1_idx;
    std::string dev_name = efa_devs[dev_idx];

    base_nics[nic].context = open_device_by_name(dev_name);
    if (!base_nics[nic].context) {
      fprintf(stderr, "[GPU %d] Failed to open device %s\n", gpu_id,
              dev_name.c_str());
      exit(EXIT_FAILURE);
    }

    base_nics[nic].pd = ibv_alloc_pd(base_nics[nic].context);
    if (!base_nics[nic].pd) {
      perror("ibv_alloc_pd");
      exit(EXIT_FAILURE);
    }

    // Register entire buffer
    uint64_t iova = (uintptr_t)gpu_buf;
    base_nics[nic].mr =
        ibv_reg_mr_iova2(base_nics[nic].pd, gpu_buf, total_size, iova,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                             IBV_ACCESS_RELAXED_ORDERING);
    if (!base_nics[nic].mr) {
      perror("ibv_reg_mr_iova2");
      exit(EXIT_FAILURE);
    }
    base_nics[nic].lkey = base_nics[nic].mr->lkey;

    // Create CQ with larger size for multiple nodes
    ibv_cq_init_attr_ex cq_ex_attr = {};
    cq_ex_attr.cqe = WINDOW_PER_NIC * 4;
    cq_ex_attr.cq_context = nullptr;
    cq_ex_attr.channel = nullptr;
    cq_ex_attr.comp_vector = 0;
    cq_ex_attr.wc_flags = IBV_WC_STANDARD_FLAGS;
    cq_ex_attr.comp_mask = 0;
    cq_ex_attr.flags = 0;

    base_nics[nic].cq =
        (ibv_cq*)ibv_create_cq_ex(base_nics[nic].context, &cq_ex_attr);
    if (!base_nics[nic].cq) {
      perror("ibv_create_cq_ex");
      exit(EXIT_FAILURE);
    }

    // Create ONE shared QP per NIC
    base_nics[nic].qp = create_srd_qp_ex(base_nics[nic]);
    if (!base_nics[nic].qp) {
      fprintf(stderr, "[GPU %d] Failed to create shared QP for NIC %d\n",
              gpu_id, nic);
      exit(EXIT_FAILURE);
    }
  }

  // Allocate peer endpoints: [NIC][remote_rank][remote_gpu]
  // Now only stores AH and remote info, no QP
  std::vector<std::vector<std::vector<PeerEndpoint>>> peers(
      NUM_NICS_PER_GPU,
      std::vector<std::vector<PeerEndpoint>>(
          num_nodes, std::vector<PeerEndpoint>(NUM_GPUS_PER_NODE)));

  // Global barrier before starting connection establishment
  thread_barrier(NUM_GPUS_PER_NODE);
  if (gpu_id == 0) {
    tcp_barrier(my_rank, num_nodes, node_ips, 4000);
  }
  thread_barrier(NUM_GPUS_PER_NODE);

  // Establish connections to all remote nodes and GPUs
  for (int nic = 0; nic < NUM_NICS_PER_GPU; nic++) {
    for (int remote_rank = 0; remote_rank < num_nodes; remote_rank++) {
      if (remote_rank == my_rank) continue;  // Skip self

      for (int remote_gpu = 0; remote_gpu < NUM_GPUS_PER_NODE; remote_gpu++) {
        // Prepare local connection info (using shared QP)
        RDMAConnectionInfo local_info = {};
        local_info.qp_num = base_nics[nic].qp->qp_num;
        local_info.rkey = base_nics[nic].mr->rkey;
        local_info.addr = (uint64_t)gpu_buf;
        local_info.len = total_size;
        fill_local_gid(base_nics[nic].context, local_info.gid);

        // Exchange connection info with remote GPU
        RDMAConnectionInfo remote_info = {};
        exchange_connection_info(my_rank, gpu_id, nic, remote_rank, remote_gpu,
                                 node_ips[remote_rank], &local_info,
                                 &remote_info);

        // Create AH for this destination
        peers[nic][remote_rank][remote_gpu].ah =
            create_ah(base_nics[nic].pd, remote_info.gid);
        if (!peers[nic][remote_rank][remote_gpu].ah) {
          fprintf(stderr,
                  "[GPU %d] Failed to create AH for nic=%d, remote_rank=%d, "
                  "remote_gpu=%d\n",
                  gpu_id, nic, remote_rank, remote_gpu);
          exit(EXIT_FAILURE);
        }

        // Store remote info
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

  // Total messages = messages to each remote GPU × total remote GPUs
  uint64_t total_msgs = total_remote_gpus * MSGS_PER_REMOTE_GPU;

  // Track posted messages per (remote_rank, remote_gpu) pair
  std::vector<std::vector<uint64_t>> posted_per_remote(
      num_nodes, std::vector<uint64_t>(NUM_GPUS_PER_NODE, 0));

  uint64_t completed_per_nic[NUM_NICS_PER_GPU] = {0};
  int inflight_per_nic[NUM_NICS_PER_GPU] = {0};

  uint64_t total_posted = 0;
  uint64_t total_completed = 0;

  int next_remote_rank = (my_rank + 1) % num_nodes;  // Start with next rank
  int next_remote_gpu = 0;

  while (total_completed < total_msgs) {
    // Post phase: round-robin across all remote (rank, gpu) pairs
    while (total_posted < total_msgs) {
      // Find next remote (rank, gpu) that needs more messages
      int attempts = 0;
      while (attempts < total_remote_gpus) {
        if (next_remote_rank == my_rank) {
          // Skip to next rank
          next_remote_rank = (next_remote_rank + 1) % num_nodes;
          continue;
        }

        if (posted_per_remote[next_remote_rank][next_remote_gpu] <
            (uint64_t)MSGS_PER_REMOTE_GPU) {
          break;
        }

        // Move to next GPU/rank
        next_remote_gpu++;
        if (next_remote_gpu >= NUM_GPUS_PER_NODE) {
          next_remote_gpu = 0;
          next_remote_rank = (next_remote_rank + 1) % num_nodes;
        }
        attempts++;
      }
      if (attempts >= total_remote_gpus) break;  // All done posting

      // Striping: alternate NICs based on message sequence
      int nic =
          (int)(posted_per_remote[next_remote_rank][next_remote_gpu] / 64) %
          NUM_NICS_PER_GPU;

      // Check window
      if (inflight_per_nic[nic] >= WINDOW_PER_NIC) break;

      // Post RDMA WRITE using shared QP
      uint64_t seq = posted_per_remote[next_remote_rank][next_remote_gpu];
      uint64_t slot = seq % SLOTS_PER_SRC;

      // Calculate remote offset based on sender's global GPU ID
      int sender_global_gpu_id = my_rank * NUM_GPUS_PER_NODE + gpu_id;
      uint64_t remote_offset = send_size +
                               sender_global_gpu_id * SLOTS_PER_SRC * MSG_SIZE +
                               slot * MSG_SIZE;
      uint64_t remote_addr =
          peers[nic][next_remote_rank][next_remote_gpu].remote_addr +
          remote_offset;

      // Use shared QP from base_nics[nic], not from peer
      ibv_qp_ex* qpx = (ibv_qp_ex*)base_nics[nic].qp;
      ibv_wr_start(qpx);

      qpx->wr_id = ((uint64_t)nic << 56) | ((uint64_t)next_remote_rank << 48) |
                   ((uint64_t)next_remote_gpu << 32) | (seq & 0xFFFFFFFF);
      qpx->wr_flags = IBV_SEND_SIGNALED;
      qpx->comp_mask = 0;
      ibv_wr_rdma_write(
          qpx, peers[nic][next_remote_rank][next_remote_gpu].remote_rkey,
          remote_addr);
      ibv_wr_set_ud_addr(
          qpx, peers[nic][next_remote_rank][next_remote_gpu].ah,
          peers[nic][next_remote_rank][next_remote_gpu].remote_qpn, QKEY);
      ibv_wr_set_sge(qpx, base_nics[nic].lkey, (uintptr_t)gpu_send_buf,
                     MSG_SIZE);

      int ret = ibv_wr_complete(qpx);
      if (ret) {
        fprintf(stderr, "[GPU %d] ibv_wr_complete failed: %s\n", gpu_id,
                strerror(ret));
        exit(EXIT_FAILURE);
      }

      posted_per_remote[next_remote_rank][next_remote_gpu]++;
      inflight_per_nic[nic]++;
      total_posted++;
      base_nics[nic].posted.fetch_add(1, std::memory_order_relaxed);

      // Move to next GPU/rank
      next_remote_gpu++;
      if (next_remote_gpu >= NUM_GPUS_PER_NODE) {
        next_remote_gpu = 0;
        next_remote_rank = (next_remote_rank + 1) % num_nodes;
      }
    }

    // Poll phase: poll both NICs
    for (int nic = 0; nic < NUM_NICS_PER_GPU; nic++) {
      ibv_cq_ex* cqx = (ibv_cq_ex*)base_nics[nic].cq;
      ibv_poll_cq_attr poll_attr = {.comp_mask = 0};
      int ret = ibv_start_poll(cqx, &poll_attr);
      if (ret) continue;  // No completions

      while (true) {
        if (cqx->status != IBV_WC_SUCCESS) {
          uint64_t nic_id = (cqx->wr_id >> 56) & 0xFF;
          uint64_t remote_rank = (cqx->wr_id >> 48) & 0xFF;
          uint64_t remote_gpu = (cqx->wr_id >> 32) & 0xFFFF;
          uint64_t seq_id = cqx->wr_id & 0xFFFFFFFF;

          fprintf(stderr, "[GPU %d] CQE error: %s\n", gpu_id,
                  ibv_wc_status_str(cqx->status));
          fprintf(stderr,
                  "  nic=%lu, remote_rank=%lu, remote_gpu=%lu, seq=%lu\n",
                  nic_id, remote_rank, remote_gpu, seq_id);
          exit(EXIT_FAILURE);
        }

        uint32_t opcode = ibv_wc_read_opcode(cqx);
        if (opcode == IBV_WC_RDMA_WRITE) {
          inflight_per_nic[nic]--;
          total_completed++;
          completed_per_nic[nic]++;
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

  uint64_t total_bytes = total_msgs * MSG_SIZE;
  double bw_gbps = (total_bytes / elapsed) / (1024.0 * 1024.0 * 1024.0);

  {
    std::lock_guard<std::mutex> lock(print_mutex);
    printf("\n=== [Rank %d GPU %d] Shared QP Results ===\n", my_rank, gpu_id);
    printf("  Elapsed: %.3f s\n", elapsed);
    printf("  Remote nodes: %d\n", num_remote_nodes);
    printf("  Total remote GPUs: %d\n", total_remote_gpus);
    printf("  Total msgs: %lu\n", total_msgs);
    printf("  Total bytes: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("  Bandwidth: %.3f GB/s\n", bw_gbps);
    printf("  NIC0 posted=%lu, completed=%lu\n", base_nics[0].posted.load(),
           base_nics[0].completed.load());
    printf("  NIC1 posted=%lu, completed=%lu\n", base_nics[1].posted.load(),
           base_nics[1].completed.load());
    printf("  QP usage: 2 shared QPs (1 per NIC)\n");
  }

  // Cleanup
  for (int nic = 0; nic < NUM_NICS_PER_GPU; nic++) {
    for (int r = 0; r < num_nodes; r++) {
      if (r == my_rank) continue;
      for (int g = 0; g < NUM_GPUS_PER_NODE; g++) {
        if (peers[nic][r][g].ah) ibv_destroy_ah(peers[nic][r][g].ah);
      }
    }
    // Destroy shared QP
    if (base_nics[nic].qp) ibv_destroy_qp(base_nics[nic].qp);
    if (base_nics[nic].cq) ibv_destroy_cq(base_nics[nic].cq);
    if (base_nics[nic].mr) ibv_dereg_mr(base_nics[nic].mr);
    if (base_nics[nic].pd) ibv_dealloc_pd(base_nics[nic].pd);
    if (base_nics[nic].context) ibv_close_device(base_nics[nic].context);
  }

  CUDA_CHECK(cudaFree(gpu_buf));
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <my_rank> <node_ips>\n", argv[0]);
    printf("  Example (2 nodes): %s 0 10.1.1.1,10.1.1.2\n", argv[0]);
    printf("  Example (4 nodes): %s 0 10.1.1.1,10.1.1.2,10.1.1.3,10.1.1.4\n",
           argv[0]);
    return 1;
  }

  int my_rank = atoi(argv[1]);
  std::vector<std::string> node_ips = parse_ip_list(argv[2]);

  if (node_ips.size() < 2) {
    fprintf(stderr, "ERROR: Need at least 2 nodes\n");
    return 1;
  }

  int num_nodes = (int)node_ips.size();

  if (my_rank < 0 || my_rank >= num_nodes) {
    fprintf(stderr, "ERROR: Invalid rank %d (must be 0-%d)\n", my_rank,
            num_nodes - 1);
    return 1;
  }

  printf("=== Multi-NIC EFA All-to-All Test (Shared QP Architecture) ===\n");
  printf("Total nodes: %d\n", num_nodes);
  printf("My rank: %d\n", my_rank);
  printf("Node IPs: ");
  for (size_t i = 0; i < node_ips.size(); i++) {
    printf("%s%s", node_ips[i].c_str(),
           (i < node_ips.size() - 1) ? ", " : "\n");
  }
  printf(
      "Config: %d GPUs/node, %d NICs/GPU, %d msgs/remote_gpu, msg_size=%zu "
      "bytes\n",
      NUM_GPUS_PER_NODE, NUM_NICS_PER_GPU, MSGS_PER_REMOTE_GPU, MSG_SIZE);
  printf("Architecture: Each NIC uses ONE shared QP for all destinations\n");

  // Launch 8 threads (one per GPU)
  std::vector<std::thread> threads;
  for (int gpu = 0; gpu < NUM_GPUS_PER_NODE; gpu++) {
    threads.emplace_back(run_gpu_thread, gpu, my_rank, num_nodes, node_ips);
  }

  for (auto& t : threads) {
    t.join();
  }

  return 0;
}

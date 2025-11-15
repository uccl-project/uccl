// All-to-All RDMA Benchmark with Random Dispatch
// Each GPU sends 128 x 7KB messages randomly to other GPUs across nodes

#include <arpa/inet.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <cuda_runtime.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

constexpr int NUM_GPUS_PER_NODE = 8;
constexpr int NUM_NICS_PER_GPU = 1;  // 1 for p6, 2 for p5en
constexpr size_t MSG_SIZE = 7168;    // 7KB
constexpr int NUM_MSGS = 128;
constexpr int TOPK = 8;
constexpr int WINDOW_SIZE = 64;
constexpr uint32_t QKEY = 0x11111111u;
constexpr int TCP_PORT_BASE = 18515;

std::mutex barrier_mutex;
std::condition_variable barrier_cv;
std::atomic<int> barrier_count{0};
int barrier_target = 0;

// Global TCP socket state for control plane (exchange + barriers)
int control_listenfd = -1;
std::vector<int> control_socks;

#define CUDA_CHECK(cmd)                                             \
  do {                                                              \
    cudaError_t e = (cmd);                                          \
    if (e != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                               \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

struct RDMAConnectionInfo {
  uint32_t qp_num;
  uint8_t gid[16];
  uint32_t rkey;
  uint64_t addr;
};

struct NicContext {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_mr* mr = nullptr;
  void* gpu_buf = nullptr;
  size_t buf_size = 0;
};

struct PeerEndpoint {
  ibv_qp* qp = nullptr;
  ibv_ah* ah = nullptr;
  uint32_t remote_qpn = 0;
  uint32_t remote_rkey = 0;
  uint64_t remote_addr = 0;
};

// Prepopulated random dispatch table
std::vector<int> generate_dispatch_table(int rank, int world_size) {
  std::vector<int> table(NUM_MSGS * TOPK);
  unsigned int seed = rank * 12345;

  for (int i = 0; i < NUM_MSGS * TOPK; i++) {
    table[i] = rand_r(&seed) % world_size;
  }
  return table;
}

void get_gid(ibv_context* ctx, int port, int index, uint8_t* gid) {
  ibv_gid g;
  if (ibv_query_gid(ctx, port, index, &g)) {
    fprintf(stderr, "Failed to query GID\n");
    exit(1);
  }
  memcpy(gid, g.raw, 16);
}

ibv_qp* create_srd_qp(NicContext* nic) {
  ibv_qp_init_attr_ex qp_attr = {};
  efadv_qp_init_attr efa_attr = {};

  qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  qp_attr.send_ops_flags =
      IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;
  qp_attr.cap.max_send_wr = 2048;
  qp_attr.cap.max_recv_wr = 2048;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  qp_attr.cap.max_inline_data = 0;
  qp_attr.pd = nic->pd;
  qp_attr.qp_context = nic->ctx;
  qp_attr.sq_sig_all = 0;
  qp_attr.send_cq = nic->cq;
  qp_attr.recv_cq = nic->cq;
  qp_attr.qp_type = IBV_QPT_DRIVER;

  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  efa_attr.sl = 8;  // Low latency service level

  ibv_qp* qp =
      efadv_create_qp_ex(nic->ctx, &qp_attr, &efa_attr, sizeof(efa_attr));
  if (!qp) {
    perror("Failed to create QP");
    exit(1);
  }

  ibv_qp_attr attr = {};
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

  attr = {};
  attr.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
    perror("Failed to modify QP to RTR");
    exit(1);
  }

  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.rnr_retry = 3;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }

  return qp;
}

ibv_ah* create_ah(ibv_pd* pd, uint8_t* remote_gid) {
  ibv_ah_attr ah_attr = {};
  ah_attr.port_num = 1;
  ah_attr.is_global = 1;
  memcpy(ah_attr.grh.dgid.raw, remote_gid, 16);

  ibv_ah* ah = ibv_create_ah(pd, &ah_attr);
  if (!ah) {
    perror("Failed to create AH");
    exit(1);
  }
  return ah;
}

void pin_to_numa(int local_rank) {
  // For p5.48xlarge: 8 GPUs, 2 NUMA nodes
  // GPUs 0-3 -> NUMA 0, GPUs 4-7 -> NUMA 1
  int numa_node = local_rank / 4;

  if (numa_available() < 0) {
    fprintf(stderr, "NUMA not available\n");
    return;
  }

  // Set memory policy to bind to specific NUMA node
  struct bitmask* numa_mask = numa_allocate_nodemask();
  numa_bitmask_clearall(numa_mask);
  numa_bitmask_setbit(numa_mask, numa_node);
  numa_bind(numa_mask);  // void return, no error checking
  numa_free_nodemask(numa_mask);

  // Set CPU affinity to CPUs on the same NUMA node
  cpu_set_t cpu_mask;
  CPU_ZERO(&cpu_mask);

  // Get CPUs for this NUMA node
  struct bitmask* cpu_bitmask = numa_allocate_cpumask();
  numa_node_to_cpus(numa_node, cpu_bitmask);

  // Iterate through all possible CPUs and check if they're in the bitmask
  unsigned int max_cpus = numa_num_configured_cpus();
  for (unsigned int i = 0; i < max_cpus; i++) {
    if (numa_bitmask_isbitset(cpu_bitmask, i)) {
      CPU_SET(i, &cpu_mask);
    }
  }

  if (sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask) < 0) {
    fprintf(stderr, "Warning: Failed to set CPU affinity for NUMA node %d\n",
            numa_node);
  }

  numa_free_cpumask(cpu_bitmask);

  printf("Rank pinned to NUMA node %d (local_rank %d)\n", numa_node,
         local_rank);
}

void tcp_control_init(
    int rank, int world_size, char const* master_ip,
    std::vector<RDMAConnectionInfo>& local_info,
    std::vector<std::vector<RDMAConnectionInfo>>& remote_info) {
  remote_info.resize(world_size);
  remote_info[rank] = local_info;

  if (rank == 0) {
    control_listenfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(control_listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT_BASE);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(control_listenfd, (sockaddr*)&addr, sizeof(addr));
    listen(control_listenfd, world_size);

    control_socks.resize(world_size, -1);
    for (int i = 1; i < world_size; i++) {
      int sock = accept(control_listenfd, nullptr, nullptr);
      int peer_rank;
      recv(sock, &peer_rank, sizeof(peer_rank), 0);
      remote_info[peer_rank].resize(NUM_NICS_PER_GPU);
      recv(sock, remote_info[peer_rank].data(),
           sizeof(RDMAConnectionInfo) * NUM_NICS_PER_GPU, 0);
      control_socks[peer_rank] = sock;
    }

    // Broadcast all connection info to all ranks
    for (int i = 1; i < world_size; i++) {
      for (int j = 0; j < world_size; j++) {
        if (j == i) continue;
        send(control_socks[i], remote_info[j].data(),
             sizeof(RDMAConnectionInfo) * NUM_NICS_PER_GPU, 0);
      }
    }
  } else {
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT_BASE);
    inet_pton(AF_INET, master_ip, &addr.sin_addr);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int retry_count = 0;
    while (connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
      if (++retry_count > 100) {
        fprintf(stderr,
                "Rank %d: Failed to connect to master after 100 retries\n",
                rank);
        exit(1);
      }
      close(sock);
      sock = socket(AF_INET, SOCK_STREAM, 0);
      usleep(100000);
    }

    send(sock, &rank, sizeof(rank), 0);
    send(sock, local_info.data(), sizeof(RDMAConnectionInfo) * NUM_NICS_PER_GPU,
         0);

    for (int i = 0; i < world_size; i++) {
      if (i == rank) continue;
      remote_info[i].resize(NUM_NICS_PER_GPU);
      recv(sock, remote_info[i].data(),
           sizeof(RDMAConnectionInfo) * NUM_NICS_PER_GPU, 0);
    }

    control_socks.push_back(sock);
  }
}

void tcp_barrier(int rank, int world_size) {
  if (rank == 0) {
    for (int i = 1; i < world_size; i++) {
      char sync;
      recv(control_socks[i], &sync, 1, 0);
    }
    for (int i = 1; i < world_size; i++) {
      char ack = 1;
      send(control_socks[i], &ack, 1, 0);
    }
  } else {
    char sync = 1;
    send(control_socks[0], &sync, 1, 0);
    char ack;
    recv(control_socks[0], &ack, 1, 0);
  }
}

void tcp_control_cleanup(int rank, int world_size) {
  if (rank == 0) {
    for (int i = 1; i < world_size; i++) {
      if (control_socks[i] >= 0) close(control_socks[i]);
    }
    if (control_listenfd >= 0) close(control_listenfd);
  } else {
    if (!control_socks.empty() && control_socks[0] >= 0) {
      close(control_socks[0]);
    }
  }
}

struct PollResult {
  int send_completions = 0;
  int recv_completions = 0;
};

PollResult poll_cq(ibv_cq* cq, int max_poll) {
  ibv_wc wc[32];
  int poll_budget = std::min(32, max_poll);
  int n = ibv_poll_cq(cq, poll_budget, wc);
  if (n < 0) {
    fprintf(stderr, "Poll CQ error\n");
    exit(1);
  }

  PollResult result;
  for (int i = 0; i < n; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      fprintf(stderr, "WC error: %s\n", ibv_wc_status_str(wc[i].status));
      exit(1);
    }
    if (wc[i].opcode & IBV_WC_RECV) {
      result.recv_completions++;
    } else {
      result.send_completions++;
    }
  }
  return result;
}

void run_benchmark(int rank, int local_rank, int world_size,
                   char const* master_ip) {
  // Pin to NUMA node based on GPU
  pin_to_numa(local_rank);

  CUDA_CHECK(cudaSetDevice(local_rank));

  std::vector<NicContext> nics(NUM_NICS_PER_GPU);

  int num_devices = 0;
  ibv_device** dev_list = ibv_get_device_list(&num_devices);

  // Filter devices with names starting with "rdmap"
  std::vector<ibv_device*> rdmap_devices;
  for (int i = 0; i < num_devices; i++) {
    char const* dev_name = ibv_get_device_name(dev_list[i]);
    if (strncmp(dev_name, "rdmap", 5) == 0) {
      rdmap_devices.push_back(dev_list[i]);
    }
  }

  int nic_base = local_rank * NUM_NICS_PER_GPU;

  if (nic_base + NUM_NICS_PER_GPU > static_cast<int>(rdmap_devices.size())) {
    fprintf(stderr, "Not enough rdmap devices: need %d, found %zu\n",
            nic_base + NUM_NICS_PER_GPU, rdmap_devices.size());
    exit(1);
  }

  for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
    auto& nic = nics[i];
    nic.ctx = ibv_open_device(rdmap_devices[nic_base + i]);
    if (!nic.ctx) {
      fprintf(stderr, "Failed to open device %s (index %d)\n",
              ibv_get_device_name(rdmap_devices[nic_base + i]), nic_base + i);
      exit(1);
    }

    nic.pd = ibv_alloc_pd(nic.ctx);
    nic.cq = ibv_create_cq(nic.ctx, 4096, nullptr, nullptr, 0);

    nic.buf_size = NUM_MSGS * MSG_SIZE;
    CUDA_CHECK(cudaMalloc(&nic.gpu_buf, nic.buf_size));
    CUDA_CHECK(cudaMemset(nic.gpu_buf, rank, nic.buf_size));

    nic.mr = ibv_reg_mr(nic.pd, nic.gpu_buf, nic.buf_size,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_RELAXED_ORDERING);
    if (!nic.mr) {
      perror("Failed to register MR");
      exit(1);
    }
  }
  ibv_free_device_list(dev_list);

  std::vector<RDMAConnectionInfo> local_info(NUM_NICS_PER_GPU);
  std::vector<std::vector<RDMAConnectionInfo>> remote_info;

  std::vector<std::vector<PeerEndpoint>> peers(NUM_NICS_PER_GPU);
  for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
    peers[i].resize(world_size);
  }

  for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
    auto& nic = nics[i];
    ibv_qp* recv_qp = create_srd_qp(&nic);

    auto& info = local_info[i];
    info.qp_num = recv_qp->qp_num;
    get_gid(nic.ctx, 1, 0, info.gid);
    info.rkey = nic.mr->rkey;
    info.addr = (uint64_t)nic.gpu_buf;

    peers[i][rank].qp = recv_qp;
  }

  // Initialize control plane: exchange connection info and keep sockets open
  // for barriers
  tcp_control_init(rank, world_size, master_ip, local_info, remote_info);

  for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
    auto& nic = nics[i];

    for (int r = 0; r < world_size; r++) {
      if (r == rank) continue;

      auto& peer = peers[i][r];
      peer.qp = create_srd_qp(&nic);
      peer.ah = create_ah(nic.pd, remote_info[r][i].gid);
      peer.remote_qpn = remote_info[r][i].qp_num;
      peer.remote_rkey = remote_info[r][i].rkey;
      peer.remote_addr = remote_info[r][i].addr;
    }

    peers[i][rank].ah = create_ah(nic.pd, local_info[i].gid);
  }

  for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
    auto& nic = nics[i];
    auto& recv_qp = peers[i][rank].qp;

    for (int slot = 0; slot < 2048; slot++) {
      int msg_id = slot % NUM_MSGS;
      ibv_sge sge = {(uint64_t)nic.gpu_buf + msg_id * MSG_SIZE, MSG_SIZE,
                     nic.mr->lkey};
      ibv_recv_wr wr = {}, *bad_wr;
      wr.wr_id = slot;
      wr.num_sge = 1;
      wr.sg_list = &sge;

      if (ibv_post_recv(recv_qp, &wr, &bad_wr)) {
        perror("Failed to post recv");
        exit(1);
      }
    }
  }

  auto dispatch_table = generate_dispatch_table(rank, world_size);
  int node_rank = rank / NUM_GPUS_PER_NODE;
  CUDA_CHECK(cudaDeviceSynchronize());

  // Barrier to ensure all ranks have posted their receive buffers
  tcp_barrier(rank, world_size);

  constexpr int NUM_ROUNDS = 50;
  constexpr int WARMUP_ROUNDS = 20;
  std::vector<double> round_times;
  int total_ops = NUM_MSGS * TOPK;

  for (int round = 0; round < NUM_ROUNDS; round++) {
    tcp_barrier(rank, world_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    int send_completed = 0;
    int send_inflight = 0;
    int network_ops = 0;

    std::vector<int> recv_completed(NUM_NICS_PER_GPU, 0);
    std::vector<int> recv_slot(NUM_NICS_PER_GPU, 2048);

    for (int op_idx = 0; op_idx < total_ops; op_idx++) {
      int dst_rank = dispatch_table[op_idx];
      int dst_node = dst_rank / NUM_GPUS_PER_NODE;

      if (dst_node == node_rank) {
        continue;
      }
      network_ops++;

      int msg_id = op_idx / TOPK;
      int nic_idx = op_idx % NUM_NICS_PER_GPU;
      auto& nic = nics[nic_idx];
      auto& peer = peers[nic_idx][dst_rank];

      ibv_qp_ex* qpx = ibv_qp_to_qp_ex(peer.qp);
      ibv_wr_start(qpx);

      qpx->wr_id = op_idx;
      qpx->wr_flags = IBV_SEND_SIGNALED;

      ibv_wr_rdma_write_imm(qpx, peer.remote_rkey,
                            peer.remote_addr + msg_id * MSG_SIZE, op_idx);

      ibv_sge sge = {(uint64_t)nic.gpu_buf + msg_id * MSG_SIZE, MSG_SIZE,
                     nic.mr->lkey};
      ibv_wr_set_sge_list(qpx, 1, &sge);
      ibv_wr_set_ud_addr(qpx, peer.ah, peer.remote_qpn, QKEY);

      if (ibv_wr_complete(qpx)) {
        fprintf(stderr, "Failed to post RDMA write\n");
        exit(1);
      }

      send_inflight++;

      while (send_inflight >= WINDOW_SIZE) {
        for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
          auto& nic = nics[i];
          auto result = poll_cq(nic.cq, send_inflight);
          send_completed += result.send_completions;
          send_inflight -= result.send_completions;
          recv_completed[i] += result.recv_completions;

          // Refill recv queue
          auto& recv_qp = peers[i][rank].qp;
          while (recv_completed[i] > 0) {
            int msg_id = recv_slot[i] % NUM_MSGS;
            ibv_sge sge = {(uint64_t)nic.gpu_buf + msg_id * MSG_SIZE, MSG_SIZE,
                           nic.mr->lkey};
            ibv_recv_wr wr = {}, *bad_wr;
            wr.wr_id = recv_slot[i];
            wr.num_sge = 1;
            wr.sg_list = &sge;

            if (ibv_post_recv(recv_qp, &wr, &bad_wr)) {
              perror("Failed to refill recv");
              exit(1);
            }
            recv_slot[i]++;
            recv_completed[i]--;
          }
        }
      }
    }

    // Drain all remaining completions
    while (send_completed < network_ops) {
      for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
        auto& nic = nics[i];
        auto result = poll_cq(nic.cq, network_ops - send_completed);
        send_completed += result.send_completions;
        recv_completed[i] += result.recv_completions;

        // Refill recv queue
        auto& recv_qp = peers[i][rank].qp;
        while (recv_completed[i] > 0) {
          int msg_id = recv_slot[i] % NUM_MSGS;
          ibv_sge sge = {(uint64_t)nic.gpu_buf + msg_id * MSG_SIZE, MSG_SIZE,
                         nic.mr->lkey};
          ibv_recv_wr wr = {}, *bad_wr;
          wr.wr_id = recv_slot[i];
          wr.num_sge = 1;
          wr.sg_list = &sge;

          if (ibv_post_recv(recv_qp, &wr, &bad_wr)) {
            perror("Failed to refill recv");
            exit(1);
          }
          recv_slot[i]++;
          recv_completed[i]--;
        }
      }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_us =
        std::chrono::duration<double, std::micro>(end - start).count();

    round_times.push_back(elapsed_us);

    if (rank == 0) {
      printf("Round %d: %.2f us\n", round, elapsed_us);
    }

    // Barrier between rounds to ensure all ranks finish together
    tcp_barrier(rank, world_size);
  }

  // Calculate average of last 5 rounds
  double sum_last_5 = 0.0;
  for (int i = WARMUP_ROUNDS; i < NUM_ROUNDS; i++) {
    sum_last_5 += round_times[i];
  }
  double avg_us = sum_last_5 / (NUM_ROUNDS - WARMUP_ROUNDS);

  printf("Rank %d: average of last %d rounds: %.2f us\n", rank,
         NUM_ROUNDS - WARMUP_ROUNDS, avg_us);

  if (rank == 0) {
    double total_data_gb = (NUM_MSGS * TOPK * MSG_SIZE) / 1e9;
    double elapsed_s = avg_us / 1e6;
    printf("Average time: %.2f us\n", avg_us);
    printf("Throughput: %.2f GB/s\n", total_data_gb / elapsed_s);
  }

  // Barrier to ensure all incoming RDMA writes have completed before cleanup
  tcp_barrier(rank, world_size);

  // Cleanup control plane connections
  tcp_control_cleanup(rank, world_size);

  for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
    auto& nic = nics[i];
    for (int r = 0; r < world_size; r++) {
      if (peers[i][r].qp) ibv_destroy_qp(peers[i][r].qp);
      if (peers[i][r].ah) ibv_destroy_ah(peers[i][r].ah);
    }
    if (nic.mr) ibv_dereg_mr(nic.mr);
    if (nic.cq) ibv_destroy_cq(nic.cq);
    if (nic.pd) ibv_dealloc_pd(nic.pd);
    if (nic.ctx) ibv_close_device(nic.ctx);
    if (nic.gpu_buf) cudaFree(nic.gpu_buf);
  }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <rank> <world_size> <master_ip>\n", argv[0]);
    return 1;
  }

  int rank = atoi(argv[1]);
  int world_size = atoi(argv[2]);
  char const* master_ip = argv[3];

  int local_rank = rank % NUM_GPUS_PER_NODE;

  run_benchmark(rank, local_rank, world_size, master_ip);

  return 0;
}

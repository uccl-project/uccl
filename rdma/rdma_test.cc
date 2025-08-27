/**
    Usage:

    server: ./rdma_test
    client: ./rdma_test <server_ip>

    Atomic Tests:
    - Set TEST_ATOMICS=1 to enable atomic operation tests
    - Tests both compare-and-swap and fetch-and-add operations
    - Client performs atomic operations on server's atomic buffer
    - Server displays final atomic values after completion
 */
#include <arpa/inet.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <thread>
#include <tuple>
#include <assert.h>
#include <cuda_runtime.h>  // cuda only for now
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

//////////////////////////////////////////////////////////////////////
#define DEV_INDEX 2           // mlx5_0-7
#define USE_GPU 0             // GPU_0-7
#define NUM_QPS 4             // Number of QPs.
#define MSG_SIZE (1ul << 20)  // Message size.
#define OUTSTNADING_MSG 4     // Number of outstanding messages.
#define ITERATIONS 1000000    // Number of iterations.
#define ROCE_NET false        // true: RoCE, false: IB
#define TEST_ATOMICS 1        // 1: Enable atomic tests, 0: Regular RDMA tests
// #define MANAGED            // Use cudaMallocManaged not cudaMalloc
//////////////////////////////////////////////////////////////////////

#if ROCE_NET
static constexpr uint8_t GID_INDEX = 0;
#else
static constexpr uint8_t GID_INDEX = 3;
#endif
#define PORT_NUM 1
#define QKEY 0x12345
#define TCP_PORT 55555
#define MAX_WR 1024
#define BASE_PSN 0x12345

static volatile uint64_t rx_stats_bytes = 0;
static volatile uint64_t last_rx_stats_bytes = 0;
static volatile uint64_t tx_stats_bytes = 0;
static volatile uint64_t last_tx_stats_bytes = 0;

static bool is_server = false;

static uint32_t next_data_offset = 0;
static uint32_t next_send_qp_idx = 0;

#if TEST_ATOMICS
static volatile uint64_t atomic_ops_completed = 0;
static volatile uint64_t last_atomic_ops_completed = 0;
static uint64_t* atomic_buffer = nullptr;  // Buffer for atomic operations
static uint64_t expected_atomic_value =
    0;                                 // Expected value for compare-and-swap
static uint64_t atomic_increment = 1;  // Increment value for fetch-and-add
#define ATOMIC_BUFFER_SIZE 8           // Number of 64-bit atomic values
#endif

struct metadata {
  uint32_t qpn[NUM_QPS];
  uint32_t rkey;
  union ibv_gid gid;
  uint64_t addr;
  uint16_t lid;
#if TEST_ATOMICS
  uint32_t atomic_rkey;
  uint64_t atomic_addr;
#endif
};

struct rdma_context {
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  struct ibv_cq_ex* cq_ex;
  struct ibv_mr* mr;

  struct ibv_srq* srq;

  char* local_buf;
  uint32_t remote_rkey;
  uint64_t remote_addr;

#if TEST_ATOMICS
  struct ibv_mr* atomic_mr;     // Memory region for atomic operations
  uint64_t* atomic_local_buf;   // Local atomic buffer
  uint64_t remote_atomic_addr;  // Remote atomic buffer address
  uint32_t remote_atomic_rkey;  // Remote atomic rkey
#endif

  struct ibv_qp* qp[NUM_QPS];
  struct metadata remote_meta;

  union ibv_gid gid;

  struct ibv_device_attr dev_attr;
  struct ibv_port_attr port_attr;
};

static bool force_exit = false;

void signal_handler(int signal) { force_exit = true; }

// Exchange QPNs and GIDs via TCP
void exchange_qpns(char const* peer_ip, metadata* local_meta,
                   metadata* remote_meta) {
  int sock;
  struct sockaddr_in addr;
  char mode = peer_ip ? 'c' : 's';

  sock = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt,
             sizeof(opt));  // Avoid port conflicts

  addr.sin_family = AF_INET;
  addr.sin_port = htons(TCP_PORT);
  addr.sin_addr.s_addr = peer_ip ? inet_addr(peer_ip) : INADDR_ANY;

  if (mode == 's') {
    int listensock = sock;
    DCHECK(bind(listensock, (struct sockaddr*)&addr, sizeof(addr)) == 0)
        << "Failed to bind";
    DCHECK(listen(listensock, 10) == 0) << "Failed to listen";
    printf("Server waiting for connection...\n");
    // accept in a non-blocking way
    while (!force_exit) {
      struct sockaddr_in client_addr;
      socklen_t client_len = sizeof(client_addr);
      sock = accept(listensock, (struct sockaddr*)&client_addr, &client_len);
      if (sock != -1) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    printf("Server accepted connection\n");
  } else {
    printf("Client attempting connection...\n");
    int attempts = 5;
    // connect in a non-blocking way
    while (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0 &&
           attempts--) {
      sleep(1);
    }
    if (attempts == 0) {
      perror("Failed to connect after retries");
      exit(1);
    }
    printf("Client connected\n");
  }

  // Set receive timeout to avoid blocking
  struct timeval timeout = {5, 0};  // 5 seconds timeout
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  // Send local metadata
  if (send(sock, local_meta, sizeof(*local_meta), 0) <= 0)
    perror("send() failed");

  // Receive remote metadata
  if (recv(sock, remote_meta, sizeof(*remote_meta), 0) <= 0)
    perror("recv() timeout");

  VLOG(1) << "local addr=" << (uint64_t)local_meta->addr
          << ", local rkey=" << local_meta->rkey
          << ", remote addr=" << (uint64_t)remote_meta->addr
          << ", remote rkey=" << remote_meta->rkey;

  close(sock);
}

void create_qp(struct rdma_context* rdma);
void create_cq(struct rdma_context* rdma);
void create_srq(struct rdma_context* rdma);
void init_srq_wr();
void post_srq(struct rdma_context* rdma, int nb_wr);
uint32_t poll_cq(struct rdma_context* rdma, uint32_t expect_opcode);
void send_message(struct rdma_context* rdma);

#if TEST_ATOMICS
void send_atomic_cmp_swap(struct rdma_context* rdma);
void send_atomic_fetch_add(struct rdma_context* rdma);
void run_atomic_server(struct rdma_context* rdma);
void run_atomic_client(struct rdma_context* rdma);
void init_atomic_buffer(struct rdma_context* rdma);
uint32_t poll_atomic_cq(struct rdma_context* rdma);
#endif

struct ibv_recv_wr wr[MAX_WR];

void init_srq_wr() {
  for (int i = 0; i < MAX_WR; i++) {
    wr[i].next = (i == MAX_WR - 1) ? nullptr : &wr[i + 1];
  }
}

void post_srq(struct rdma_context* rdma, int nb_wr) {
  if (nb_wr == 0) return;
  DCHECK(nb_wr <= MAX_WR) << "nb_wr is greater than MAX_WR";

  auto* srq = rdma->srq;

  struct ibv_recv_wr* cache_wr = wr[nb_wr - 1].next;
  wr[nb_wr - 1].next = nullptr;

  struct ibv_recv_wr* bad_wr;
  DCHECK(ibv_post_srq_recv(srq, wr, &bad_wr) == 0);

  wr[nb_wr - 1].next = cache_wr;
}

void modify_qp_rtr(struct rdma_context* rdma) {
  auto remote_meta = rdma->remote_meta;
  struct ibv_qp_attr attr = {};
  int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
                  IBV_QP_RQ_PSN | IBV_QP_MIN_RNR_TIMER |
                  IBV_QP_MAX_DEST_RD_ATOMIC;

  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = rdma->port_attr.active_mtu;

#if ROCE_NET
  attr.ah_attr.is_global = 1;
  attr.ah_attr.grh.dgid = rdma->remote_meta.gid;
  attr.ah_attr.grh.sgid_index = GID_INDEX;
  attr.ah_attr.grh.hop_limit = 0xff;
  attr.ah_attr.grh.traffic_class = 0;
#else
  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = rdma->remote_meta.lid;
#endif

  attr.ah_attr.sl = 0;
  attr.ah_attr.port_num = PORT_NUM;
  attr.rq_psn = BASE_PSN;
  attr.min_rnr_timer = 12;
  attr.max_dest_rd_atomic = 1;

  for (int i = 0; i < NUM_QPS; i++) {
    attr.dest_qp_num = rdma->remote_meta.qpn[i];
    int ret = ibv_modify_qp(rdma->qp[i], &attr, attr_mask);
    DCHECK(ret == 0) << "Failed to modify QP: " << ret;
  }
}

void modify_qp_rts(struct rdma_context* rdma) {
  struct ibv_qp_attr attr = {};
  int attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                  IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = BASE_PSN;

  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;

  for (int i = 0; i < NUM_QPS; i++) {
    int ret = ibv_modify_qp(rdma->qp[i], &attr, attr_mask);
    DCHECK(ret == 0) << "Failed to modify QP: " << ret;
  }
}

// Initialize RDMA resources
struct rdma_context* init_rdma(char const* server_ip) {
  struct rdma_context* rdma =
      (struct rdma_context*)calloc(1, sizeof(struct rdma_context));

  int nb_devices;

  struct ibv_device** dev_list = ibv_get_device_list(&nb_devices);

  char device_name[32];
  sprintf(device_name, "mlx5_%d", DEV_INDEX);

  int i;
  for (i = 0; i < nb_devices; i++) {
    if (strcmp(ibv_get_device_name(dev_list[i]), device_name) == 0) {
      break;
    }
  }

  DCHECK(i < nb_devices) << "Device " << device_name << " not found";
  auto* open_dev = dev_list[i];

  rdma->ctx = ibv_open_device(open_dev);
  DCHECK(rdma->ctx) << "Failed to open device";

  ibv_free_device_list(dev_list);

  rdma->pd = ibv_alloc_pd(rdma->ctx);
  DCHECK(rdma->pd) << "Failed to allocate pd";

  DCHECK(ibv_query_device(rdma->ctx, &rdma->dev_attr) == 0)
      << "Failed to query device";

  DCHECK(rdma->dev_attr.phys_port_cnt == 1) << "Only one port is supported";

  DCHECK(ibv_query_port(rdma->ctx, 1, &rdma->port_attr) == 0)
      << "Failed to query port";

  DCHECK(rdma->port_attr.state == IBV_PORT_ACTIVE) << "Port is not active";

  DCHECK(rdma->port_attr.link_layer ==
         (ROCE_NET ? IBV_LINK_LAYER_ETHERNET : IBV_LINK_LAYER_INFINIBAND))
      << "Link layer error";

  DCHECK(ibv_query_gid(rdma->ctx, 1, GID_INDEX, &rdma->gid) == 0)
      << "Failed to query gid";

#ifdef USE_GPU
  cudaSetDevice(USE_GPU);
#ifndef MANAGED
  if (cudaMalloc(&rdma->local_buf, OUTSTNADING_MSG * MSG_SIZE) != cudaSuccess) {
#else
  if (cudaMallocManaged(&rdma->local_buf, OUTSTNADING_MSG * MSG_SIZE) !=
      cudaSuccess) {
#endif
    perror("Failed to allocate GPU memory");
    exit(1);
  }

#ifdef MANAGED
  DCHECK(cudaMemAdvise(rdma->local_buf, OUTSTNADING_MSG * MSG_SIZE,
                       cudaMemAdviseSetPreferredLocation,
                       USE_GPU) == cudaSuccess)
      << "Failed to set preferred location";
  DCHECK(cudaMemAdvise(rdma->local_buf, OUTSTNADING_MSG * MSG_SIZE,
                       cudaMemAdviseSetAccessedBy, USE_GPU) == cudaSuccess)
      << "Failed to set accessed by";
  DCHECK(cudaMemPrefetchAsync(rdma->local_buf, OUTSTNADING_MSG * MSG_SIZE,
                              USE_GPU) == cudaSuccess)
      << "Failed to prefetch GPU memory";
  DCHECK(cudaDeviceSynchronize() == cudaSuccess) << "Failed to synchronize";
#endif

  rdma->mr =
      ibv_reg_mr(rdma->pd, rdma->local_buf, OUTSTNADING_MSG * MSG_SIZE,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
#else
  if (posix_memalign((void**)&rdma->local_buf, sysconf(_SC_PAGESIZE),
                     OUTSTNADING_MSG * MSG_SIZE)) {
    perror("Failed to allocate local buffer");
    exit(1);
  }
  rdma->mr =
      ibv_reg_mr(rdma->pd, rdma->local_buf, OUTSTNADING_MSG * MSG_SIZE,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
#endif

  DCHECK(rdma->mr) << "Failed to register memory regions";
  assert((uintptr_t)rdma->mr->addr == (uintptr_t)rdma->local_buf);

  VLOG(1) << "RX MR: addr=" << (uint64_t)rdma->mr->addr
          << ", len=" << rdma->mr->length << ", lkey=" << rdma->mr->lkey
          << ", rkey=" << rdma->mr->rkey;

  create_srq(rdma);

  create_cq(rdma);

  create_qp(rdma);

  metadata local_meta;
  for (int i = 0; i < NUM_QPS; i++) {
    local_meta.qpn[i] = rdma->qp[i]->qp_num;
  }
  local_meta.rkey = rdma->mr->rkey;
  local_meta.addr = (uint64_t)rdma->local_buf;
  local_meta.lid = rdma->port_attr.lid;  // Only for IB
  memcpy(&local_meta.gid, &rdma->gid, sizeof(local_meta.gid));

#if TEST_ATOMICS
  // Initialize atomic buffer for atomic operations
  init_atomic_buffer(rdma);
#endif

#if TEST_ATOMICS
  local_meta.atomic_rkey = rdma->atomic_mr->rkey;
  local_meta.atomic_addr = (uint64_t)rdma->atomic_local_buf;
#endif

  exchange_qpns(server_ip, &local_meta, &rdma->remote_meta);

#if TEST_ATOMICS
  // Store remote atomic buffer information
  rdma->remote_atomic_addr = rdma->remote_meta.atomic_addr;
  rdma->remote_atomic_rkey = rdma->remote_meta.atomic_rkey;

  VLOG(1) << "Remote atomic buffer: addr=" << rdma->remote_atomic_addr
          << ", rkey=" << rdma->remote_atomic_rkey;
#endif

  modify_qp_rtr(rdma);

  modify_qp_rts(rdma);

  init_srq_wr();

  post_srq(rdma, MAX_WR);

  return rdma;
}

void create_cq(struct rdma_context* rdma) {
  struct ibv_cq_init_attr_ex cq_ex_attr;
  cq_ex_attr.cqe = 16384;
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

  rdma->cq_ex = ibv_create_cq_ex(rdma->ctx, &cq_ex_attr);
  DCHECK(rdma->cq_ex) << "Failed to create CQ";
}

void create_srq(struct rdma_context* rdma) {
  struct ibv_srq_init_attr srq_init_attr;
  memset(&srq_init_attr, 0, sizeof(srq_init_attr));
  srq_init_attr.attr.max_sge = 1;
  srq_init_attr.attr.max_wr = MAX_WR;
  srq_init_attr.attr.srq_limit = 0;
  rdma->srq = ibv_create_srq(rdma->pd, &srq_init_attr);
  DCHECK(rdma->srq) << "Failed to create SRQ";
}

// Create and configure a UD QP
void create_qp(struct rdma_context* rdma) {
  struct ibv_qp_init_attr attr;
  memset(&attr, 0, sizeof(attr));

  attr.qp_context = nullptr;
  attr.send_cq = ibv_cq_ex_to_cq(rdma->cq_ex);
  attr.recv_cq = ibv_cq_ex_to_cq(rdma->cq_ex);
  attr.qp_type = IBV_QPT_RC;

  attr.cap.max_send_wr = MAX_WR;
  attr.cap.max_send_sge = 1;
  attr.cap.max_inline_data = 0;
  attr.srq = rdma->srq;

  struct ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = PORT_NUM;
#if TEST_ATOMICS
  qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
#else
  qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
#endif

  for (int i = 0; i < NUM_QPS; i++) {
    rdma->qp[i] = ibv_create_qp(rdma->pd, &attr);
    DCHECK(rdma->qp[i]) << "Failed to create QP";

    DCHECK(ibv_modify_qp(rdma->qp[i], &qp_attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                             IBV_QP_ACCESS_FLAGS) == 0)
        << "Failed to modify QP";
  }
}

uint32_t poll_cq(struct rdma_context* rdma, uint32_t expect_opcode) {
  auto totoal_bytes = 0;
  auto* cq_ex = rdma->cq_ex;
  int post_srq_cnt = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (!force_exit) {
    DCHECK(cq_ex->status == IBV_WC_SUCCESS)
        << "Failed to poll CQ: " << cq_ex->status;
    DCHECK(ibv_wc_read_opcode(cq_ex) == expect_opcode)
        << "Unexpected opcode: " << ibv_wc_read_opcode(cq_ex);

    if (ibv_wc_read_opcode(cq_ex) == IBV_WC_RECV_RDMA_WITH_IMM) {
      totoal_bytes += ibv_wc_read_byte_len(cq_ex);
      post_srq_cnt++;
    } else {
      totoal_bytes++;  // it is actually acked_msgs
    }

    if (ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  post_srq(rdma, post_srq_cnt);

  return totoal_bytes;
}

#if TEST_ATOMICS
uint32_t poll_atomic_cq(struct rdma_context* rdma) {
  auto completed_ops = 0;
  auto* cq_ex = rdma->cq_ex;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (!force_exit) {
    DCHECK(cq_ex->status == IBV_WC_SUCCESS)
        << "Failed to poll CQ: " << cq_ex->status;

    uint32_t opcode = ibv_wc_read_opcode(cq_ex);
    DCHECK(opcode == IBV_WC_COMP_SWAP || opcode == IBV_WC_FETCH_ADD)
        << "Unexpected atomic opcode: " << opcode;

    completed_ops++;

    if (ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return completed_ops;
}

void init_atomic_buffer(struct rdma_context* rdma) {
  // Allocate atomic buffer
#if USE_GPU
  cudaSetDevice(USE_GPU);
#ifndef MANAGED
  if (cudaMalloc((void**)&rdma->atomic_local_buf,
                 ATOMIC_BUFFER_SIZE * sizeof(uint64_t)) != cudaSuccess) {
#else
  if (cudaMallocManaged((void**)&rdma->atomic_local_buf,
                        ATOMIC_BUFFER_SIZE * sizeof(uint64_t)) != cudaSuccess) {
#endif
    perror("Failed to allocate GPU atomic buffer");
    exit(1);
  }

#ifdef MANAGED
  DCHECK(cudaMemAdvise(
             rdma->atomic_local_buf, ATOMIC_BUFFER_SIZE * sizeof(uint64_t),
             cudaMemAdviseSetPreferredLocation, USE_GPU) == cudaSuccess)
      << "Failed to set atomic buffer preferred location";
  DCHECK(cudaMemAdvise(rdma->atomic_local_buf,
                       ATOMIC_BUFFER_SIZE * sizeof(uint64_t),
                       cudaMemAdviseSetAccessedBy, USE_GPU) == cudaSuccess)
      << "Failed to set atomic buffer accessed by";
  DCHECK(cudaMemPrefetchAsync(rdma->atomic_local_buf,
                              ATOMIC_BUFFER_SIZE * sizeof(uint64_t),
                              USE_GPU) == cudaSuccess)
      << "Failed to prefetch GPU atomic buffer";
  DCHECK(cudaDeviceSynchronize() == cudaSuccess)
      << "Failed to synchronize atomic buffer";
#endif

#else
  if (posix_memalign((void**)&rdma->atomic_local_buf, sysconf(_SC_PAGESIZE),
                     ATOMIC_BUFFER_SIZE * sizeof(uint64_t))) {
    perror("Failed to allocate atomic buffer");
    exit(1);
  }
#endif

  // Initialize atomic buffer with known values
#if USE_GPU
  // For GPU memory, we need to use cudaMemcpy
  uint64_t host_init_data[ATOMIC_BUFFER_SIZE];
  for (int i = 0; i < ATOMIC_BUFFER_SIZE; i++) {
    host_init_data[i] = i;
  }
  DCHECK(cudaMemcpy(rdma->atomic_local_buf, host_init_data,
                    ATOMIC_BUFFER_SIZE * sizeof(uint64_t),
                    cudaMemcpyHostToDevice) == cudaSuccess)
      << "Failed to initialize GPU atomic buffer";
  DCHECK(cudaDeviceSynchronize() == cudaSuccess)
      << "Failed to synchronize after atomic buffer initialization";
#else
  // For host memory, direct access is fine
  for (int i = 0; i < ATOMIC_BUFFER_SIZE; i++) {
    rdma->atomic_local_buf[i] = i;
  }
#endif

  // Register atomic memory region
  rdma->atomic_mr = ibv_reg_mr(
      rdma->pd, rdma->atomic_local_buf, ATOMIC_BUFFER_SIZE * sizeof(uint64_t),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  if (!rdma->atomic_mr) {
    perror("Failed to register atomic memory region");
    exit(1);
  }

  VLOG(1) << "Atomic MR: addr=" << (uint64_t)rdma->atomic_mr->addr
          << ", len=" << rdma->atomic_mr->length
          << ", lkey=" << rdma->atomic_mr->lkey
          << ", rkey=" << rdma->atomic_mr->rkey;
}

void send_atomic_cmp_swap(struct rdma_context* rdma) {
  struct ibv_send_wr wr = {};
  struct ibv_sge sge = {};

  // Use first 8 bytes of local buffer for atomic result
  sge.lkey = rdma->atomic_mr->lkey;
  sge.addr = (uint64_t)rdma->atomic_local_buf;
  sge.length = sizeof(uint64_t);

  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr.send_flags = IBV_SEND_SIGNALED;

  // Set up atomic operation parameters
  wr.wr.atomic.remote_addr = rdma->remote_atomic_addr;
  wr.wr.atomic.rkey = rdma->remote_atomic_rkey;
  wr.wr.atomic.compare_add = expected_atomic_value;  // Compare value
  wr.wr.atomic.swap = expected_atomic_value + 1;     // Swap value

  struct ibv_send_wr* bad_wr;
  if (ibv_post_send(rdma->qp[next_send_qp_idx], &wr, &bad_wr) != 0) {
    perror("Failed to post atomic compare-and-swap");
    return;
  }

  next_send_qp_idx = (next_send_qp_idx + 1) % NUM_QPS;
  expected_atomic_value++;
}

void send_atomic_fetch_add(struct rdma_context* rdma) {
  struct ibv_send_wr wr = {};
  struct ibv_sge sge = {};

  // Use second 8 bytes of local buffer for atomic result
  sge.lkey = rdma->atomic_mr->lkey;
  sge.addr = (uint64_t)(rdma->atomic_local_buf + 1);
  sge.length = sizeof(uint64_t);

  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr.send_flags = IBV_SEND_SIGNALED;

  // Set up atomic operation parameters
  wr.wr.atomic.remote_addr = rdma->remote_atomic_addr + sizeof(uint64_t);
  wr.wr.atomic.rkey = rdma->remote_atomic_rkey;
  wr.wr.atomic.compare_add = atomic_increment;  // Add value

  struct ibv_send_wr* bad_wr;
  if (ibv_post_send(rdma->qp[next_send_qp_idx], &wr, &bad_wr) != 0) {
    perror("Failed to post atomic fetch-and-add");
    return;
  }

  next_send_qp_idx = (next_send_qp_idx + 1) % NUM_QPS;
}

void run_atomic_server(struct rdma_context* rdma) {
  uint32_t iterations = 0;
  printf("Atomic server waiting for atomic operations...\n");

  while (!force_exit) {
    // For atomic operations, the server is passive
    // Just wait and let the client perform atomic operations
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    if (iterations++ >= ITERATIONS) {
      break;
    }
  }

  printf("Atomic server completed. Final atomic values:\n");
#if USE_GPU
  // For GPU memory, copy to host before reading
  uint64_t host_final_data[ATOMIC_BUFFER_SIZE];
  DCHECK(cudaMemcpy(host_final_data, rdma->atomic_local_buf,
                    ATOMIC_BUFFER_SIZE * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost) == cudaSuccess)
      << "Failed to read GPU atomic buffer";
  DCHECK(cudaDeviceSynchronize() == cudaSuccess)
      << "Failed to synchronize after reading atomic buffer";
  for (int i = 0; i < ATOMIC_BUFFER_SIZE; i++) {
    printf("  atomic_buf[%d] = %lu\n", i, host_final_data[i]);
  }
#else
  // For host memory, direct access is fine
  for (int i = 0; i < ATOMIC_BUFFER_SIZE; i++) {
    printf("  atomic_buf[%d] = %lu\n", i, rdma->atomic_local_buf[i]);
  }
#endif
}

void run_atomic_client(struct rdma_context* rdma) {
  uint32_t outstanding_ops = 0;
  uint32_t iterations = 0;
  bool use_cmp_swap = true;  // Alternate between operations

  printf("Atomic client starting atomic operations...\n");

  while (!force_exit) {
    // Send atomic operations
    while (outstanding_ops < OUTSTNADING_MSG) {
      if (use_cmp_swap) {
        send_atomic_cmp_swap(rdma);
      } else {
        send_atomic_fetch_add(rdma);
      }

      atomic_ops_completed++;
      outstanding_ops++;
      iterations++;
      use_cmp_swap = !use_cmp_swap;  // Alternate operations
    }

    // Poll for completions
    uint32_t completed = poll_atomic_cq(rdma);
    outstanding_ops -= completed;
  }

  printf("Atomic client completed %u atomic operations\n", iterations);
}
#endif

// Server: Post a receive and poll CQ
void run_server(struct rdma_context* rdma) {
  uint32_t iterations = 0;
  printf("Server waiting for message...\n");

  while (!force_exit) {
    auto received_bytes = poll_cq(rdma, IBV_WC_RECV_RDMA_WITH_IMM);
    rx_stats_bytes += received_bytes;

    if (received_bytes) {
      iterations++;
    }

    if (iterations == ITERATIONS) {
      break;
    }
  }
}

void send_message(struct rdma_context* rdma) {
  auto* data = rdma->local_buf + next_data_offset;
  auto remote_meta = rdma->remote_meta;

  struct ibv_send_wr wr = {};
  struct ibv_sge sge = {};

  wr.wr.rdma.remote_addr = remote_meta.addr + next_data_offset;
  wr.wr.rdma.rkey = remote_meta.rkey;

  sge.lkey = rdma->mr->lkey;
  sge.addr = (uint64_t)data;
  sge.length = MSG_SIZE;

  wr.sg_list = &sge;
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr;
  DCHECK(ibv_post_send(rdma->qp[next_send_qp_idx], &wr, &bad_wr) == 0);

  next_send_qp_idx = (next_send_qp_idx + 1) % NUM_QPS;
  next_data_offset =
      (next_data_offset + MSG_SIZE) % (MSG_SIZE * OUTSTNADING_MSG);
}

// Client: Send message
void run_client(struct rdma_context* rdma) {
  uint32_t outstanding_msg = 0;
  uint32_t iterations = 0;

  while (!force_exit) {
    while (outstanding_msg < OUTSTNADING_MSG && iterations < ITERATIONS) {
      send_message(rdma);
      tx_stats_bytes += MSG_SIZE;
      outstanding_msg++;
      iterations++;
    }

    outstanding_msg -= poll_cq(rdma, IBV_WC_RDMA_WRITE);

    if (outstanding_msg == 0 && iterations == ITERATIONS) {
      break;
    }
  }
}

#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static bool GdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  return KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
         KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
         KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

void check_gdr_support() {
  if (!GdrSupportInitOnce()) {
    printf("nvidia_peermem is not available for GPUDirect RDMA\n");
    return;
  }

  int driverVersion;
  cudaDriverGetVersion(&driverVersion);
  DCHECK(driverVersion >= 11030);
  for (int i = 0; i < 8; i++) {
    int cudaDev, attr = 0;
    cudaSetDevice(i);
    cudaGetDevice(&cudaDev);
    cudaDeviceGetAttribute(&attr, cudaDevAttrGPUDirectRDMASupported, cudaDev);
    printf("GPU%d GDR support: %d\n", i, attr);
  }
}

// TO RUN:
// [client]: ./rdma_test
// [server]: ./rdma_test 10.0.100.114

int main(int argc, char* argv[]) {
  signal(SIGINT, signal_handler);

  check_gdr_support();

  is_server = (argc == 1);

  struct rdma_context* rdma = init_rdma(is_server ? nullptr : argv[1]);

  // launch a stats thread, print stats every 1 second
  std::thread stats_thread([&]() {
    while (!force_exit) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
#if TEST_ATOMICS
      // Atomic operations stats
      auto atomic_ops_per_sec =
          atomic_ops_completed - last_atomic_ops_completed;
      printf("[mlx5_%d] Atomic Operations: %lu ops/sec, Total: %lu\n",
             DEV_INDEX, atomic_ops_per_sec, atomic_ops_completed);
      last_atomic_ops_completed = atomic_ops_completed;
#else
      // Regular RDMA stats
      // unit: Gbps
      auto rx_stats_bytes_per_sec =
          8 * (rx_stats_bytes - last_rx_stats_bytes) / 1e9;
      auto tx_stats_bytes_per_sec =
          8 * (tx_stats_bytes - last_tx_stats_bytes) / 1e9;
#ifdef USE_GPU
      printf("[GPU%d <---> mlx5_%d] RX: %.2f Gbps, TX: %.2f Gbps\n", USE_GPU,
             DEV_INDEX, rx_stats_bytes_per_sec, tx_stats_bytes_per_sec);
#else
      printf("[Host  <---> mlx5_%d]  RX: %.2f Gbps, TX: %.2f Gbps\n", DEV_INDEX,
             rx_stats_bytes_per_sec, tx_stats_bytes_per_sec);
#endif
      last_rx_stats_bytes = rx_stats_bytes;
      last_tx_stats_bytes = tx_stats_bytes;
#endif
    }
  });

#if TEST_ATOMICS
  if (!is_server)
    run_atomic_client(rdma);
  else
    run_atomic_server(rdma);
#else
  if (!is_server)
    run_client(rdma);
  else
    run_server(rdma);
#endif

  return 0;
}

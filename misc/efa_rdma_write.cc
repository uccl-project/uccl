#include <arpa/inet.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <tuple>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#define GID_INDEX 0
#define PORT_NUM 1
#define QKEY 0x12345
#define MTU 8928
#define MSG_SIZE 1024000
#define TCP_PORT 12345
// #define USE_GPU

struct rdma_context {
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  struct ibv_cq_ex* cq_ex;
  struct ibv_qp* qp;
  struct ibv_mr* mr;
  struct ibv_ah* ah;
  char* local_buf;
  uint32_t remote_rkey;
  uint64_t remote_addr;
};

// Retrieve GID based on gid_index
void get_gid(struct rdma_context* rdma, int gid_index, union ibv_gid* gid) {
  if (ibv_query_gid(rdma->ctx, PORT_NUM, gid_index, gid)) {
    perror("Failed to query GID");
    exit(1);
  }
  printf("GID[%d]: %s\n", gid_index, inet_ntoa(*(struct in_addr*)&gid->raw[8]));
}

// Create AH using specific GID index
struct ibv_ah* create_ah(struct rdma_context* rdma, int gid_index,
                         union ibv_gid remote_gid) {
  struct ibv_ah_attr ah_attr = {0};

  ah_attr.port_num = PORT_NUM;
  ah_attr.is_global = 1;          // Enable Global Routing Header (GRH)
  ah_attr.grh.dgid = remote_gid;  // Destination GID

  // ah_attr.grh.sgid_index = gid_index;  // Use selected GID index
  // ah_attr.grh.flow_label = 0;
  // ah_attr.grh.hop_limit = 255;
  // ah_attr.grh.traffic_class = 0;

  struct ibv_ah* ah = ibv_create_ah(rdma->pd, &ah_attr);
  if (!ah) {
    perror("Failed to create AH");
    exit(1);
  }
  return ah;
}

struct metadata {
  uint32_t qpn;
  union ibv_gid gid;
  uint32_t rkey;
  uint64_t addr;
};

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
    printf("Server waiting for connection...\n");
    bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(sock, 10);
    sock = accept(sock, NULL, NULL);  // Blocks if no client
    printf("Server accepted connection\n");
  } else {
    printf("Client attempting connection...\n");
    int attempts = 5;
    while (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0 &&
           attempts--) {
      perror("Connect failed, retrying...");
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

  printf(
      "local addr=0x%lx, local rkey=0x%x, remote addr=0x%lx, remote "
      "rkey=0x%x\n",
      local_meta->addr, local_meta->rkey, remote_meta->addr, remote_meta->rkey);

  close(sock);
  printf("QPNs and GIDs exchanged\n");
}

struct ibv_qp* create_qp(struct rdma_context* rdma);

// Initialize RDMA resources
struct rdma_context* init_rdma(int gid_index) {
  struct rdma_context* rdma =
      (struct rdma_context*)calloc(1, sizeof(struct rdma_context));

  struct ibv_device** dev_list = ibv_get_device_list(NULL);
  rdma->ctx = ibv_open_device(dev_list[gid_index]);
  printf("Using device: %s\n", ibv_get_device_name(dev_list[gid_index]));
  ibv_free_device_list(dev_list);
  if (!rdma->ctx) {
    perror("Failed to open device");
    exit(1);
  }

  rdma->pd = ibv_alloc_pd(rdma->ctx);

  struct ibv_cq_init_attr_ex init_attr_ex = {
      .cqe = 1024,
      .cq_context = NULL,
      .channel = NULL,
      .comp_vector = 0,
      /* EFA requires these values for wc_flags and comp_mask.
       * See `efa_create_cq_ex` in rdma-core.
       */
      .wc_flags = IBV_WC_STANDARD_FLAGS,
      .comp_mask = 0,
  };

  rdma->cq_ex = ibv_create_cq_ex(rdma->ctx, &init_attr_ex);
  if (!rdma->pd || !rdma->cq_ex) {
    perror("Failed to allocate PD or CQ");
    exit(1);
  }

#ifdef USE_GPU
  if (cudaMalloc(&rdma->local_buf, MSG_SIZE) != cudaSuccess) {
    perror("Failed to allocate GPU memory");
    exit(1);
  }
  rdma->mr = ibv_reg_mr(rdma->pd, rdma->local_buf, MSG_SIZE,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ);
#else
  if (posix_memalign((void**)&rdma->local_buf, sysconf(_SC_PAGESIZE),
                     MSG_SIZE)) {
    perror("Failed to allocate local buffer");
    exit(1);
  }
  rdma->mr = ibv_reg_mr(rdma->pd, rdma->local_buf, MSG_SIZE,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ);
#endif

  if (!rdma->mr) {
    perror("Failed to register memory regions");
    exit(1);
  }
  assert((uintptr_t)rdma->mr->addr == (uintptr_t)rdma->local_buf);

  printf("RX MR: addr=%p, len=%zu, lkey=0x%x, rkey=0x%x\n", rdma->mr->addr,
         rdma->mr->length, rdma->mr->lkey, rdma->mr->rkey);

  rdma->qp = create_qp(rdma);
  return rdma;
}

// Create and configure a UD QP
struct ibv_qp* create_qp(struct rdma_context* rdma) {
  struct ibv_qp_init_attr_ex qp_attr_ex = {0};
  struct efadv_qp_init_attr efa_attr = {0};

  qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                              IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                              IBV_QP_EX_WITH_RDMA_READ;

  qp_attr_ex.cap.max_send_wr = 256;
  qp_attr_ex.cap.max_recv_wr = 256;
  qp_attr_ex.cap.max_send_sge = 1;
  qp_attr_ex.cap.max_recv_sge = 1;
  qp_attr_ex.cap.max_inline_data = 0;

  qp_attr_ex.pd = rdma->pd;
  qp_attr_ex.qp_context = rdma->ctx;
  qp_attr_ex.sq_sig_all = 1;

  qp_attr_ex.send_cq = ibv_cq_ex_to_cq(rdma->cq_ex);
  qp_attr_ex.recv_cq = ibv_cq_ex_to_cq(rdma->cq_ex);

  qp_attr_ex.qp_type = IBV_QPT_DRIVER;

  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
#define EFA_QP_LOW_LATENCY_SERVICE_LEVEL 8
  efa_attr.sl = EFA_QP_LOW_LATENCY_SERVICE_LEVEL;

  struct ibv_qp* qp = efadv_create_qp_ex(rdma->ctx, &qp_attr_ex, &efa_attr,
                                         sizeof(struct efadv_qp_init_attr));

  if (!qp) {
    perror("Failed to create QP");
    exit(1);
  }

  struct ibv_qp_attr attr = {};
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = PORT_NUM;
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
#define EFA_RDM_DEFAULT_RNR_RETRY (3)
  attr.rnr_retry = EFA_RDM_DEFAULT_RNR_RETRY;  // Set RNR retry count
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }

  return qp;
}

// From
// https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_cq.c#L424
int poll_cq(struct rdma_context* rdma, int expect_opcode) {
  int poll_count = 0;

  /* Initialize an empty ibv_poll_cq_attr struct for ibv_start_poll.
   * EFA expects .comp_mask = 0, or otherwise returns EINVAL.
   */
  struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
  ssize_t err = ibv_start_poll(rdma->cq_ex, &poll_cq_attr);
  bool should_end_poll = !err;
  while (!err) {
    if (rdma->cq_ex->status != IBV_WC_SUCCESS) {
      printf("poll_cq: %s\n", ibv_wc_status_str(rdma->cq_ex->status));
    }
    auto wr_id = rdma->cq_ex->wr_id;
    auto length = ibv_wc_read_byte_len(rdma->cq_ex);
    printf("poll_cq: wr_id=%lx, length=%d\n", wr_id, length);
    int opcode = ibv_wc_read_opcode(rdma->cq_ex);
    // assert(opcode == expect_opcode);

    if (opcode != expect_opcode) {
      printf("Unexpected opcode: %d\n", opcode);
      break;
    }

    if (expect_opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      auto recvd_imm = ibv_wc_read_imm_data(rdma->cq_ex);
      printf("Received immediate data: %x\n", recvd_imm);
    }

    /*
     * ibv_next_poll MUST be call after the current WC is fully processed,
     * which prevents later calls on ibv_cq_ex from reading the wrong WC.
     */
    poll_count++;
    err = ibv_next_poll(rdma->cq_ex);
  }

  if (should_end_poll) ibv_end_poll(rdma->cq_ex);

  return poll_count;
}

// Server: Post a receive and poll CQ
void run_server(struct rdma_context* rdma, int gid_index) {
  metadata local_meta, remote_meta;
  local_meta.qpn = rdma->qp->qp_num;
  local_meta.rkey = rdma->mr->rkey;
  local_meta.addr = (uint64_t)rdma->local_buf;
  get_gid(rdma, gid_index, &local_meta.gid);

  exchange_qpns(NULL, &local_meta, &remote_meta);

  // This is the key to enabling RDMA read/write over SRD.
  rdma->ah = create_ah(rdma, gid_index, remote_meta.gid);

#ifdef USE_GPU
  // prepare message
  char* h_data = (char*)malloc(MSG_SIZE);
  strcpy(h_data, "World Hello!");
  cudaMemcpy(rdma->local_buf, h_data, MSG_SIZE, cudaMemcpyHostToDevice);
#else
  strcpy(rdma->local_buf, "World Hello!");
#endif

  // Post first receive buffer
  struct ibv_sge sge[1] = {
      {(uintptr_t)rdma->local_buf, MSG_SIZE, rdma->mr->lkey}};

  struct ibv_recv_wr wr = {0}, *bad_wr;
  wr.wr_id = 1;
  wr.num_sge = 1;
  wr.sg_list = sge;
  // This will let sender-side hang
  // wr.num_sge = 0;
  // wr.sg_list = nullptr;
  wr.next = NULL;

  if (ibv_post_recv(rdma->qp, &wr, &bad_wr)) {
    perror("Failed to post recv");
    exit(1);
  }

  // Post second receive buffer
  wr.wr_id = 2;
  wr.num_sge = 1;
  wr.sg_list = sge;
  wr.next = NULL;

  if (ibv_post_recv(rdma->qp, &wr, &bad_wr)) {
    perror("Failed to post recv");
    exit(1);
  }

  printf("Server waiting for message...\n");
  auto total_pull_count = 0;
  while (total_pull_count < 2) {
    total_pull_count += poll_cq(rdma, IBV_WC_RECV_RDMA_WITH_IMM);
  }

#ifdef USE_GPU
  // Only the first message is attached a hdr.
  char* h_data = (char*)malloc(MSG_SIZE);
  cudaMemcpy(h_data, rdma->local_buf, MSG_SIZE, cudaMemcpyDeviceToHost);
  printf("Server received: %s\n", h_data);
  free(h_data);
#else
  printf("Server received: %s\n", rdma->local_buf);
#endif
}

// Client: Send message
void run_client(struct rdma_context* rdma, char const* server_ip,
                int gid_index) {
  metadata local_meta, remote_meta;
  local_meta.qpn = rdma->qp->qp_num;
  local_meta.rkey = rdma->mr->rkey;
  local_meta.addr = (uint64_t)rdma->local_buf;
  get_gid(rdma, gid_index, &local_meta.gid);

  exchange_qpns(server_ip, &local_meta, &remote_meta);

  sleep(1);  // Wait for server to post receive

  rdma->ah = create_ah(rdma, gid_index, remote_meta.gid);

#ifdef USE_GPU
  // prepare message
  char* h_data = (char*)malloc(MSG_SIZE);
  strcpy(h_data, "Hello World!");
  cudaMemcpy(rdma->local_buf, h_data, MSG_SIZE, cudaMemcpyHostToDevice);
#else
  strcpy(rdma->local_buf, "Hello World!");
#endif

  auto* qpx = ibv_qp_to_qp_ex(rdma->qp);
  ibv_wr_start(qpx);

  // sending a large message
  qpx->wr_id = 1;
  qpx->comp_mask = 0;
  qpx->wr_flags = IBV_SEND_SIGNALED;
  ibv_wr_rdma_write_imm(qpx, remote_meta.rkey, remote_meta.addr, 0x1);
  // ibv_wr_rdma_write(qpx, remote_meta.rkey, remote_meta.addr);
  // ibv_wr_rdma_read(qpx, remote_meta.rkey, remote_meta.addr);

  struct ibv_sge sge[1] = {
      {(uintptr_t)rdma->local_buf, MSG_SIZE, rdma->mr->lkey}};

  ibv_wr_set_sge_list(qpx, 1, sge);
  ibv_wr_set_ud_addr(qpx, rdma->ah, remote_meta.qpn, QKEY);

  if (ibv_wr_complete(qpx)) {
    printf("ibv_wr_complete failed\n");
  }
  printf("Client: Large message sent: wr_id=%lx, length=%d\n", qpx->wr_id,
         MSG_SIZE);

  // sending a small message
  qpx->wr_id = 2;
  qpx->comp_mask = 0;
  qpx->wr_flags = IBV_SEND_SIGNALED;
  ibv_wr_rdma_write_imm(qpx, remote_meta.rkey, remote_meta.addr, 0x2);
  sge[0].addr = (uintptr_t)rdma->local_buf;
  sge[0].length = 8;
  sge[0].lkey = rdma->mr->lkey;
  ibv_wr_set_sge_list(qpx, 1, sge);
  ibv_wr_set_ud_addr(qpx, rdma->ah, remote_meta.qpn, QKEY);
  if (ibv_wr_complete(qpx)) {
    printf("ibv_wr_complete failed\n");
  }
  printf("Client: Small message sent: wr_id=%lx, length=%d\n", qpx->wr_id, 8);

  struct ibv_wc wc;
  printf("Client poll message completion...\n");
  auto totol_pull_count = 0;
  while (totol_pull_count < 2) {
    totol_pull_count += poll_cq(rdma, IBV_WC_RDMA_WRITE);
    // totol_pull_count += poll_cq(rdma, IBV_WC_RDMA_READ);
  }

#ifdef USE_GPU
  memset(h_data, 0, MSG_SIZE);
  cudaMemcpy(h_data, rdma->local_buf, MSG_SIZE, cudaMemcpyDeviceToHost);
  printf("Client sent: %s\n", h_data);
  free(h_data);
#else
  printf("Client sent: %s\n", rdma->local_buf);
#endif
}

int main(int argc, char* argv[]) {
  struct rdma_context* rdma = init_rdma(GID_INDEX);

  if (argc == 2)
    run_client(rdma, argv[1], GID_INDEX);
  else
    run_server(rdma, GID_INDEX);

  return 0;
}
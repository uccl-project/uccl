#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEVICE_NAME "rdmap16s27"  // Change to your RDMA device
#define GID_INDEX 0
#define PORT_NUM 1
#define QKEY 0x12345
#define BUFFER_SIZE 1024
#define TCP_PORT 12345    // Port for exchanging QPNs & GIDs
#define UD_ADDITION (40)  // Extra space for Global Routing Header
#define SHARED_QPS 256    // Maximum QPs on EFA NIC

struct rdma_context {
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  struct ibv_qp* qp[SHARED_QPS];
  struct ibv_mr* mr;
  struct ibv_ah* ah;
  char* buffer;
  uint32_t remote_qpn[SHARED_QPS];
  union ibv_gid remote_gid;
};

size_t align_size(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

// Retrieve GID based on gid_index
void get_gid(struct rdma_context* rdma, int gid_index, union ibv_gid* gid) {
  if (ibv_query_gid(rdma->ctx, PORT_NUM, gid_index, gid)) {
    perror("Failed to query GID");
    exit(1);
  }
  printf("GID[%d]: %s\n", gid_index, inet_ntoa(*(struct in_addr*)&gid->raw[8]));
}

// Exchange QPNs and GIDs via TCP
void exchange_qp_info(char const* peer_ip, uint32_t* local_qpn,
                      union ibv_gid local_gid, uint32_t* remote_qpn,
                      union ibv_gid* remote_gid) {
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

  send(sock, local_qpn, sizeof(uint32_t) * SHARED_QPS, 0);
  send(sock, &local_gid, sizeof(union ibv_gid), 0);
  recv(sock, remote_qpn, sizeof(uint32_t) * SHARED_QPS, 0);
  recv(sock, remote_gid, sizeof(union ibv_gid), 0);

  close(sock);
  printf("QPNs and GIDs exchanged\n");
}

// Create an Address Handle (AH) for UD mode
struct ibv_ah* create_ah(struct rdma_context* rdma, union ibv_gid remote_gid) {
  struct ibv_ah_attr ah_attr = {};

  ah_attr.is_global = 1;
  ah_attr.port_num = PORT_NUM;
  ah_attr.grh.sgid_index = GID_INDEX;
  ah_attr.grh.dgid = remote_gid;
  ah_attr.grh.flow_label = 0;
  ah_attr.grh.hop_limit = 255;
  ah_attr.grh.traffic_class = 0;

  struct ibv_ah* ah = ibv_create_ah(rdma->pd, &ah_attr);
  if (!ah) {
    perror("Failed to create AH");
    exit(1);
  }
  return ah;
}

// Initialize RDMA resources
struct rdma_context* init_rdma() {
  struct rdma_context* rdma =
      (struct rdma_context*)calloc(1, sizeof(struct rdma_context));

  struct ibv_device** dev_list = ibv_get_device_list(NULL);
  rdma->ctx = ibv_open_device(dev_list[0]);
  ibv_free_device_list(dev_list);
  if (!rdma->ctx) {
    perror("Failed to open device");
    exit(1);
  }

  rdma->pd = ibv_alloc_pd(rdma->ctx);
  rdma->cq = ibv_create_cq(rdma->ctx, 256, NULL, NULL, 0);
  if (!rdma->pd || !rdma->cq) {
    perror("Failed to allocate PD or CQ");
    exit(1);
  }

  // Allocate and register receive buffer
  size_t aligned_size = align_size(4096, BUFFER_SIZE + UD_ADDITION);
  rdma->buffer = (char*)aligned_alloc(4096, aligned_size);
  rdma->mr =
      ibv_reg_mr(rdma->pd, rdma->buffer, aligned_size, IBV_ACCESS_LOCAL_WRITE);
  if (!rdma->mr) {
    perror("Failed to register memory region");
    exit(1);
  }

  // Create 2 UD QPs sharing the same CQ
  for (int i = 0; i < SHARED_QPS; i++) {
    struct ibv_qp_init_attr qp_attr = {};
    qp_attr.qp_type = IBV_QPT_UD;
    qp_attr.send_cq = rdma->cq;
    qp_attr.recv_cq = rdma->cq;
    qp_attr.cap.max_send_wr = 128;
    qp_attr.cap.max_recv_wr = 128;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;

    rdma->qp[i] = ibv_create_qp(rdma->pd, &qp_attr);
    if (!rdma->qp[i]) {
      perror("Failed to create QP");
      exit(1);
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = PORT_NUM;
    attr.qkey = QKEY;
    if (ibv_modify_qp(
            rdma->qp[i], &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
      perror("Failed to modify QP to INIT");
      exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(rdma->qp[i], &attr, IBV_QP_STATE)) {
      perror("Failed to modify QP to RTR");
      exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0x12345;  // Set initial Send Queue PSN
    if (ibv_modify_qp(rdma->qp[i], &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
      perror("Failed to modify QP to RTS");
      exit(1);
    }
  }
  return rdma;
}

struct ibv_wc cq_wc[SHARED_QPS] = {};

void poll_cq_ntimes(struct rdma_context* rdma, int n) {
  int polled = 0;
  do {
    auto ret = ibv_poll_cq(rdma->cq, 32, cq_wc);
    if (ret < 0) {
      perror("Failed to poll CQ");
      exit(1);
    }
    polled += ret;
  } while (polled < n);
  return;
}

struct ibv_send_wr wr[SHARED_QPS] = {};
struct ibv_send_wr* bad_wr[SHARED_QPS] = {};
struct ibv_sge sge[SHARED_QPS] = {};
uint32_t local_qpn[SHARED_QPS] = {};

// Client: Send messages using both QPs
void run_client(struct rdma_context* rdma, char const* server_ip) {
  union ibv_gid local_gid;
  for (int i = 0; i < SHARED_QPS; i++) local_qpn[i] = rdma->qp[i]->qp_num;
  get_gid(rdma, GID_INDEX, &local_gid);

  exchange_qp_info(server_ip, local_qpn, local_gid, rdma->remote_qpn,
                   &rdma->remote_gid);

  rdma->ah = create_ah(rdma, rdma->remote_gid);

  strcpy(rdma->buffer, "Hello World!");

  for (int i = 0; i < SHARED_QPS; i++) {
    wr[i].opcode = IBV_WR_SEND;
    wr[i].num_sge = 1;
    sge[i] = {(uintptr_t)rdma->buffer, BUFFER_SIZE, rdma->mr->lkey};
    wr[i].sg_list = &sge[i];
    wr[i].wr.ud.ah = rdma->ah;
    wr[i].wr.ud.remote_qpn = rdma->remote_qpn[i];
    wr[i].wr.ud.remote_qkey = QKEY;
    wr[i].send_flags = IBV_SEND_SIGNALED;

    if (ibv_post_send(rdma->qp[i], &wr[i], &bad_wr[i])) {
      perror("Failed to post send");
      exit(1);
    }
  }
  printf("Client: Message sent!\n");

  poll_cq_ntimes(rdma, SHARED_QPS);
  printf("Client poll message completion...\n");

  printf("Client sent: %s\n", rdma->buffer);
}

struct ibv_recv_wr recv_wr[SHARED_QPS] = {};
struct ibv_recv_wr* bad_recv_wr[SHARED_QPS];
struct ibv_sge recv_sge[SHARED_QPS] = {};

// Server: Receive messages
void run_server(struct rdma_context* rdma) {
  union ibv_gid local_gid;
  for (int i = 0; i < SHARED_QPS; i++) local_qpn[i] = rdma->qp[i]->qp_num;
  get_gid(rdma, GID_INDEX, &local_gid);

  exchange_qp_info(NULL, local_qpn, local_gid, rdma->remote_qpn,
                   &rdma->remote_gid);

  for (int i = 0; i < SHARED_QPS; i++) {
    recv_wr[i].num_sge = 1;
    recv_sge[i] = {(uintptr_t)rdma->buffer, BUFFER_SIZE + UD_ADDITION,
                   rdma->mr->lkey};
    recv_wr[i].sg_list = &recv_sge[i];

    if (ibv_post_recv(rdma->qp[i], &recv_wr[i], &bad_recv_wr[i])) {
      perror("Failed to post send");
      exit(1);
    }
  }
  printf("Server waiting for message...\n");
  poll_cq_ntimes(rdma, SHARED_QPS);

  printf("Server received: %s\n", rdma->buffer + UD_ADDITION);
}

void free_resources(struct rdma_context* rdma) {
  ibv_dereg_mr(rdma->mr);
  for (int i = 0; i < SHARED_QPS; i++) {
    ibv_destroy_qp(rdma->qp[i]);
  }
  ibv_destroy_cq(rdma->cq);
  ibv_dealloc_pd(rdma->pd);
  ibv_close_device(rdma->ctx);
}

int main(int argc, char* argv[]) {
  struct rdma_context* rdma = init_rdma();
  if (argc == 2)
    run_client(rdma, argv[1]);
  else
    run_server(rdma);
  free_resources(rdma);
  return 0;
}

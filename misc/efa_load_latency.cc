#include "util/util.h"
#include <arpa/inet.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define DEVICE_NAME "rdmap16s27"  // Change to your RDMA device
#define GID_INDEX 0
#define PORT_NUM 1
#define QKEY 0x12345
#define MTU 8928
#define PKT_HERDER_SIZE 64
#define BUFFER_SIZE1 (MTU - PKT_HERDER_SIZE)
#define BUFFER_SIZE2 (PKT_HERDER_SIZE)
#define TCP_PORT 12345  // Port for exchanging QPNs & GIDs
#define USE_GDR 1
#define USE_SRD 0
#define ITERATION 100000
#define WARMUP 1000
#define MAX_INFLIGHT 32
#define MAX_NUM_BUFFERS (MAX_INFLIGHT * 2)
#define MAX_POLL_CQ 16

#if USE_SRD == 0
#define UD_ADDITION (40)
#else
#define UD_ADDITION (0)
#endif

struct rdma_context {
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  struct ibv_qp* qp;
  struct ibv_mr *mr1, *mr2;
  struct ibv_ah* ah;
  char *buf1, *buf2;
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

// Create and configure a UD QP
struct ibv_qp* create_qp(struct rdma_context* rdma) {
  struct ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = rdma->cq;
  qp_attr.recv_cq = rdma->cq;
  qp_attr.cap.max_send_wr = MAX_INFLIGHT * 2;
  qp_attr.cap.max_recv_wr = MAX_INFLIGHT * 2;
  qp_attr.cap.max_send_sge = 2;
  qp_attr.cap.max_recv_sge = 2;

#if USE_SRD == 0
  qp_attr.qp_type = IBV_QPT_UD;
  struct ibv_qp* qp = ibv_create_qp(rdma->pd, &qp_attr);
#else
  qp_attr.qp_type = IBV_QPT_DRIVER;
  struct ibv_qp* qp = qp =
      efadv_create_driver_qp(rdma->pd, &qp_attr, EFADV_QP_DRIVER_TYPE_SRD);
#endif

  if (!qp) {
    perror("Failed to create QP");
    exit(1);
  }

  struct ibv_qp_attr attr = {};
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
  attr.sq_psn = 0x12345;  // Set initial Send Queue PSN
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }

  return qp;
}

// Create AH using specific GID index
struct ibv_ah* create_ah(struct rdma_context* rdma, int gid_index,
                         union ibv_gid remote_gid) {
  struct ibv_ah_attr ah_attr = {};

  ah_attr.is_global = 1;  // Enable Global Routing Header (GRH)
  ah_attr.port_num = PORT_NUM;
  ah_attr.grh.sgid_index = gid_index;  // Use selected GID index
  ah_attr.grh.dgid = remote_gid;       // Destination GID
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

// Exchange QPNs and GIDs via TCP
void exchange_qpns(char const* peer_ip, uint32_t* local_qpn,
                   uint32_t* remote_qpn, union ibv_gid* local_gid,
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

  // Send local QPN and GID
  if (send(sock, local_qpn, sizeof(*local_qpn), 0) <= 0)
    perror("send() failed");
  if (send(sock, local_gid, sizeof(*local_gid), 0) <= 0)
    perror("send() failed");

  // Receive remote QPN and GID
  if (recv(sock, remote_qpn, sizeof(*remote_qpn), 0) <= 0)
    perror("recv() timeout");
  if (recv(sock, remote_gid, sizeof(*remote_gid), 0) <= 0)
    perror("recv() timeout");

  close(sock);
  printf("QPNs and GIDs exchanged\n");
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
  rdma->cq = ibv_create_cq(rdma->ctx, 1024, NULL, NULL, 0);
  if (!rdma->pd || !rdma->cq) {
    perror("Failed to allocate PD or CQ");
    exit(1);
  }

// Register memory regions
#if USE_GDR == 0
    rdma->buf1 = (char *)aligned_alloc(
        4096, align_size(4096, (BUFFER_SIZE1 + UD_ADDITION) * MAX_NUM_BUFFERS);
#else
  if (cudaMalloc(&rdma->buf1, (BUFFER_SIZE1 + UD_ADDITION) * MAX_NUM_BUFFERS) !=
      cudaSuccess) {
    perror("Failed to allocate GPU memory");
    exit(1);
  }
#endif
    rdma->buf2 = (char *)aligned_alloc(
        4096, align_size(4096, BUFFER_SIZE2 * MAX_NUM_BUFFERS));

    rdma->mr1 = ibv_reg_mr(rdma->pd, rdma->buf1, (BUFFER_SIZE1 + UD_ADDITION) * MAX_NUM_BUFFERS,
                           IBV_ACCESS_LOCAL_WRITE);
    rdma->mr2 = ibv_reg_mr(rdma->pd, rdma->buf2, BUFFER_SIZE2 * MAX_NUM_BUFFERS,
                           IBV_ACCESS_LOCAL_WRITE);
    if (!rdma->mr1 || !rdma->mr2) {
    perror("Failed to register memory regions");
    exit(1);
    }

    rdma->qp = create_qp(rdma);
    return rdma;
}

void rdma_send(struct rdma_context* rdma, uint32_t remote_qpn, int buf_idx) {
  struct ibv_sge send_sge[2] = {
      {(uintptr_t)rdma->buf1 + (BUFFER_SIZE1 + UD_ADDITION) * buf_idx +
           UD_ADDITION,
       BUFFER_SIZE1, rdma->mr1->lkey},
      {(uintptr_t)rdma->buf2 + BUFFER_SIZE2 * buf_idx, BUFFER_SIZE2,
       rdma->mr2->lkey}};
  struct ibv_send_wr send_wr = {0}, *bad_send_wr;

  send_wr.wr_id = buf_idx;
  send_wr.opcode = IBV_WR_SEND;
  send_wr.num_sge = 2;
  send_wr.sg_list = send_sge;
  send_wr.wr.ud.ah = rdma->ah;
  send_wr.wr.ud.remote_qpn = remote_qpn;
  send_wr.wr.ud.remote_qkey = QKEY;
  send_wr.send_flags = IBV_SEND_SIGNALED;

  if (ibv_post_send(rdma->qp, &send_wr, &bad_send_wr)) {
    perror("Server: Failed to post send");
    exit(1);
  }
}

void rdma_post_recv(struct rdma_context* rdma, int buf_idx) {
  // Must consider the incomding packet header size + packet size.
  struct ibv_sge new_sge[2] = {
      {(uintptr_t)rdma->buf1 + (BUFFER_SIZE1 + UD_ADDITION) * buf_idx,
       BUFFER_SIZE1 + UD_ADDITION, rdma->mr1->lkey},
      {(uintptr_t)rdma->buf2 + BUFFER_SIZE2 * buf_idx, BUFFER_SIZE2,
       rdma->mr2->lkey}};
  struct ibv_recv_wr new_recv_wr = {0}, *bad_new_wr;

  new_recv_wr.wr_id = buf_idx;
  new_recv_wr.num_sge = 2;
  new_recv_wr.sg_list = new_sge;

  if (ibv_post_recv(rdma->qp, &new_recv_wr, &bad_new_wr)) {
    perror("Server: Failed to repost recv");
    exit(1);
  }
}

// Server: Post a receive and poll CQ
// Server: For each received message, forward it back to the client.
// Server: Prepost MAX_INFLIGHT receives, then process completions in batches
// of 16.
void run_server(struct rdma_context* rdma, int gid_index) {
  uint32_t remote_qpn;
  union ibv_gid local_gid, remote_gid;

  // Exchange QP numbers and GIDs (server passes NULL as its IP)
  get_gid(rdma, gid_index, &local_gid);
  exchange_qpns(NULL, &rdma->qp->qp_num, &remote_qpn, &local_gid, &remote_gid);

  // Create an address handle for replying back to the client.
  rdma->ah = create_ah(rdma, gid_index, remote_gid);

  // Prepost MAX_INFLIGHT receive work requests.
  for (int i = 0; i < MAX_INFLIGHT; i++) {
    rdma_post_recv(rdma, i);
  }

  int total_messages = 0;
  struct ibv_wc wc[MAX_POLL_CQ];

  while (total_messages < ITERATION) {
    int ne = ibv_poll_cq(rdma->cq, MAX_POLL_CQ, wc);
    if (ne < 0) {
      perror("Server: ibv_poll_cq error");
      exit(1);
    }
    for (int i = 0; i < ne; i++) {
      // Check the completion status.
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr, "Server: Completion error: %s\n",
                ibv_wc_status_str(wc[i].status));
        exit(1);
      }

      if (wc[i].opcode == IBV_WC_RECV) {
        auto buf_idx = wc[i].wr_id;
        // Received a message from the client.
        // Post a send to forward the message back.
        rdma_send(rdma, remote_qpn, buf_idx);

        if (total_messages % 10000 == 0)
          printf("Server: Message %d recv!\n", total_messages);

        total_messages++;  // Count one message forwarded.
      } else if (wc[i].opcode == IBV_WC_SEND) {
        // Send completion for a forwarded message.
        auto buf_idx = wc[i].wr_id;
        // Repost a receive to keep the pool at MAX_INFLIGHT.
        rdma_post_recv(rdma, buf_idx);
      } else {
        fprintf(stderr, "Server: Unexpected completion opcode: %d\n",
                wc[i].opcode);
      }
    }
  }

  // Only the first message is attached a hdr.
#if USE_GDR == 0
  printf("Server received: %s | %s\n", rdma->buf1 + UD_ADDITION, rdma->buf2);
#else
  char* h_data = (char*)malloc(BUFFER_SIZE1);
  cudaMemcpy(h_data, rdma->buf1 + UD_ADDITION, BUFFER_SIZE1,
             cudaMemcpyDeviceToHost);
  printf("Server received: %s | %s\n", h_data, rdma->buf2);
  free(h_data);
#endif
}

// Revised Client: Merged polling loop that both throttles sends (using current
// time) and polls the CQ in batches of up to 16 completions.
void run_client(struct rdma_context* rdma, char const* server_ip, int gid_index,
                double target_rate) {
  uint32_t remote_qpn;
  union ibv_gid local_gid, remote_gid;

  get_gid(rdma, gid_index, &local_gid);
  exchange_qpns(server_ip, &rdma->qp->qp_num, &remote_qpn, &local_gid,
                &remote_gid);
  rdma->ah = create_ah(rdma, gid_index, remote_gid);

  sleep(1);  // Wait for server to post receive

  // prepare message
  char* h_data = (char*)malloc(BUFFER_SIZE1);
  strcpy(h_data, "Hello");
  for (int i = 0; i < MAX_NUM_BUFFERS; i++) {
#if USE_GDR == 0
    strcpy(rdma->buf1 + (BUFFER_SIZE1 + UD_ADDITION) * i + UD_ADDITION,
           "Hello");
#else
    cudaMemcpy(rdma->buf1 + (BUFFER_SIZE1 + UD_ADDITION) * i + UD_ADDITION,
               h_data, BUFFER_SIZE1, cudaMemcpyHostToDevice);
#endif
    strcpy(rdma->buf2 + BUFFER_SIZE2 * i, "World");
  }

  // Allocate arrays for timestamping and latency measurements.
  std::vector<uint64_t> send_timestamps;
  std::vector<uint64_t> latencies;
  send_timestamps.resize(ITERATION);
  latencies.resize(ITERATION);
  memset(send_timestamps.data(), 0, sizeof(uint64_t) * ITERATION);
  memset(latencies.data(), 0, sizeof(uint64_t) * ITERATION);

  int inflight = 0;
  int sent_count = 0;
  int recv_count = 0;
  struct timespec ts, start, now, finish;

  // Record the start time.
  clock_gettime(CLOCK_MONOTONIC, &start);
  double start_sec = start.tv_sec + start.tv_nsec / 1e9;

  // Prepost MAX_INFLIGHT receive work requests.
  for (int i = 0; i < MAX_INFLIGHT; i++) {
    rdma_post_recv(rdma, i);
  }

  // Main loop: either post sends if it's time or poll for completions.
  while (sent_count < ITERATION || inflight > 0) {
    // Update current time.
    clock_gettime(CLOCK_MONOTONIC, &now);
    double now_sec = now.tv_sec + now.tv_nsec / 1e9;
    // Expected send time for the next message.
    double expected_time = start_sec + ((double)sent_count / target_rate);

    // If we can send a new message and the time has arrived...
    if (inflight < MAX_INFLIGHT && sent_count < ITERATION &&
        now_sec >= expected_time) {
      // Record the send timestamp.
      clock_gettime(CLOCK_MONOTONIC, &ts);
      send_timestamps[sent_count] = ts.tv_sec * 1000000000LL + ts.tv_nsec;

      // Sending buffers use the second half of the buffer space.
      auto buf_idx = sent_count % MAX_INFLIGHT + MAX_INFLIGHT;
      // avoiding overwritting the `world` string.
      *(uint64_t*)(rdma->buf2 + BUFFER_SIZE2 * buf_idx + 32) = sent_count;
      rdma_send(rdma, remote_qpn, buf_idx);

      if (sent_count % 10000 == 0)
        printf("Client: Message %d sent!\n", sent_count);

      inflight++;
      sent_count++;
    }

    // Poll the CQ (non-blocking) for up to 16 completions.
    struct ibv_wc wc[MAX_POLL_CQ];
    int ne = ibv_poll_cq(rdma->cq, MAX_POLL_CQ, wc);
    if (ne < 0) {
      perror("Client: ibv_poll_cq error");
      exit(1);
    }
    for (int i = 0; i < ne; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr, "Client: Completion error: %s\n",
                ibv_wc_status_str(wc[i].status));
        exit(1);
      }
      // Process RECV completions (replies from server).
      if (wc[i].opcode == IBV_WC_RECV) {
        auto buf_idx = wc[i].wr_id;
        auto iter = *(uint64_t*)(rdma->buf2 + BUFFER_SIZE2 * buf_idx + 32);

        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t recv_time = ts.tv_sec * 1000000000LL + ts.tv_nsec;
        latencies[iter] = recv_time - send_timestamps[iter];
        inflight--;

        rdma_post_recv(rdma, buf_idx);
        recv_count++;
      }
      // Optionally, handle SEND completions if needed.
    }
  }

  // Record finish time and compute overall elapsed time.
  clock_gettime(CLOCK_MONOTONIC, &finish);
  double finish_sec = finish.tv_sec + finish.tv_nsec / 1e9;
  double elapsed = finish_sec - start_sec;
  double measured_rate = ITERATION / elapsed;
  printf("Target message rate: %.2f msg/s\n", target_rate);
  printf("Measured message rate: %.2f msg/s\n", measured_rate);

  latencies.erase(latencies.begin(), latencies.begin() + WARMUP);
  auto min_ = *std::min_element(latencies.begin(), latencies.end());
  auto max_ = *std::max_element(latencies.begin(), latencies.end());
  auto sum_ = std::accumulate(latencies.begin(), latencies.end(), 0ull);
  auto avg_ = (double)sum_ / (ITERATION - WARMUP);
  auto median = uccl::Percentile(latencies, 5);
  auto tail90 = uccl::Percentile(latencies, 90);
  auto tail99 = uccl::Percentile(latencies, 99);
  printf(
      "Latency (us): min=%.2f, max=%.2f, med=%.2f, 90th=%.2f, 99th=%.2f, "
      "avg=%.2f\n",
      min_ / 1e3, max_ / 1e3, median / 1e3, tail90 / 1e3, tail99 / 1e3,
      avg_ / 1e3);
}

int main(int argc, char* argv[]) {
  struct rdma_context* rdma = init_rdma();

  if (argc == 3) {
    double target_rate = atof(argv[2]);
    run_client(rdma, argv[1], GID_INDEX, target_rate);
  } else
    run_server(rdma, GID_INDEX);

  return 0;
}

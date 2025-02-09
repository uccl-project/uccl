#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define DEVICE_NAME "rdmap16s27"  // Change to your RDMA device
#define GID_INDEX 0
#define PORT_NUM 1
#define QKEY 0x12345
#define BUFFER_SIZE 1024
#define TCP_PORT 12345  // Port for exchanging QPNs & GIDs
#define UD_ADDITION (40)
#define USE_GDR 1

struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr1, *mr2;
    struct ibv_ah *ah;
    char *buf1, *buf2;
};

// Retrieve GID based on gid_index
void get_gid(struct rdma_context *rdma, int gid_index, union ibv_gid *gid) {
    if (ibv_query_gid(rdma->ctx, PORT_NUM, gid_index, gid)) {
        perror("Failed to query GID");
        exit(1);
    }
    printf("GID[%d]: %s\n", gid_index,
           inet_ntoa(*(struct in_addr *)&gid->raw[8]));
}

// Create and configure a UD QP
struct ibv_qp *create_qp(struct rdma_context *rdma) {
    struct ibv_qp_init_attr qp_attr = {};
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_type = IBV_QPT_UD;
    qp_attr.send_cq = rdma->cq;
    qp_attr.recv_cq = rdma->cq;
    qp_attr.cap.max_send_wr = 256;
    qp_attr.cap.max_recv_wr = 256;
    qp_attr.cap.max_send_sge = 2;
    qp_attr.cap.max_recv_sge = 2;

    struct ibv_qp *qp = ibv_create_qp(rdma->pd, &qp_attr);
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
    attr.sq_psn = 0x12345;  // ✅ Set initial Send Queue PSN

    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        perror("Failed to modify QP to RTS");
        exit(1);
    }

    return qp;
}

// Create AH using specific GID index
struct ibv_ah *create_ah(struct rdma_context *rdma, int gid_index,
                         union ibv_gid remote_gid) {
    struct ibv_ah_attr ah_attr = {};
    memset(&ah_attr, 0, sizeof(ah_attr));

    ah_attr.is_global = 1;  // ✅ Enable Global Routing Header (GRH)
    ah_attr.port_num = PORT_NUM;
    ah_attr.grh.sgid_index = gid_index;  // ✅ Use selected GID index
    ah_attr.grh.dgid = remote_gid;       // ✅ Destination GID
    ah_attr.grh.flow_label = 0;
    ah_attr.grh.hop_limit = 255;
    ah_attr.grh.traffic_class = 0;

    struct ibv_ah *ah = ibv_create_ah(rdma->pd, &ah_attr);
    if (!ah) {
        perror("Failed to create AH");
        exit(1);
    }
    return ah;
}

// Exchange QPNs and GIDs via TCP
void exchange_qpns(const char *peer_ip, uint32_t *local_qpn,
                   uint32_t *remote_qpn, union ibv_gid *local_gid,
                   union ibv_gid *remote_gid) {
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
        bind(sock, (struct sockaddr *)&addr, sizeof(addr));
        listen(sock, 10);
        sock = accept(sock, NULL, NULL);  // Blocks if no client
        printf("Server accepted connection\n");
    } else {
        printf("Client attempting connection...\n");
        int attempts = 5;
        while (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0 &&
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
struct rdma_context *init_rdma() {
    struct rdma_context *rdma =
        (struct rdma_context *)calloc(1, sizeof(struct rdma_context));

    struct ibv_device **dev_list = ibv_get_device_list(NULL);
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
    rdma->buf1 = (char *)aligned_alloc(4096, BUFFER_SIZE + UD_ADDITION);
#else
    if (cudaMalloc(&rdma->buf1, BUFFER_SIZE) != cudaSuccess) {
        perror("Failed to allocate GPU memory");
        exit(1);
    }
#endif
    rdma->buf2 = (char *)aligned_alloc(4096, BUFFER_SIZE + UD_ADDITION);
    rdma->mr1 = ibv_reg_mr(rdma->pd, rdma->buf1, BUFFER_SIZE + UD_ADDITION,
                           IBV_ACCESS_LOCAL_WRITE);
    rdma->mr2 = ibv_reg_mr(rdma->pd, rdma->buf2, BUFFER_SIZE + UD_ADDITION,
                           IBV_ACCESS_LOCAL_WRITE);
    if (!rdma->mr1 || !rdma->mr2) {
        perror("Failed to register memory regions");
        exit(1);
    }

    rdma->qp = create_qp(rdma);
    return rdma;
}

// Server: Post a receive and poll CQ
void run_server(struct rdma_context *rdma, int gid_index) {
    uint32_t remote_qpn;
    union ibv_gid local_gid, remote_gid;

    get_gid(rdma, gid_index, &local_gid);
    exchange_qpns(NULL, &rdma->qp->qp_num, &remote_qpn, &local_gid,
                  &remote_gid);

    // Post receive buffer
    struct ibv_sge sge[2] = {
        {(uintptr_t)rdma->buf1, BUFFER_SIZE + UD_ADDITION, rdma->mr1->lkey},
        {(uintptr_t)rdma->buf2, BUFFER_SIZE + UD_ADDITION, rdma->mr2->lkey}};

    struct ibv_recv_wr wr = {}, *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.num_sge = 2;
    wr.sg_list = sge;

    if (ibv_post_recv(rdma->qp, &wr, &bad_wr)) {
        perror("Failed to post recv");
        exit(1);
    }

    struct ibv_wc wc;
    printf("Server waiting for message...\n");
    while (ibv_poll_cq(rdma->cq, 1, &wc) < 1);
    // Only the first message is attached a hdr.
#if USE_GDR == 0
    printf("Server received: %s | %s\n", rdma->buf1 + UD_ADDITION, rdma->buf2);
#else
    char *h_data = (char *)malloc(BUFFER_SIZE);
    cudaMemcpy(h_data, rdma->buf1, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    printf("Server received: %s | %s\n", h_data + UD_ADDITION, rdma->buf2);
    free(h_data);
#endif
}

// Client: Send message
void run_client(struct rdma_context *rdma, const char *server_ip,
                int gid_index) {
    uint32_t remote_qpn;
    union ibv_gid local_gid, remote_gid;

    get_gid(rdma, gid_index, &local_gid);
    exchange_qpns(server_ip, &rdma->qp->qp_num, &remote_qpn, &local_gid,
                  &remote_gid);
    rdma->ah = create_ah(rdma, gid_index, remote_gid);

    // prepare message
#if USE_GDR == 0
    strcpy(rdma->buf1, "Hello");
#else
    char *h_data = (char *)malloc(BUFFER_SIZE);
    strcpy(h_data, "Hello");
    cudaMemcpy(rdma->buf1, h_data, BUFFER_SIZE, cudaMemcpyHostToDevice);
#endif
    strcpy(rdma->buf2, "World");

    struct ibv_sge sge[2] = {
        {(uintptr_t)rdma->buf1, BUFFER_SIZE, rdma->mr1->lkey},
        {(uintptr_t)rdma->buf2, BUFFER_SIZE, rdma->mr2->lkey}};

    struct ibv_send_wr wr = {};
    memset(&wr, 0, sizeof(wr));
    struct ibv_send_wr *bad_wr = NULL;
    wr.wr_id = 1;
    wr.opcode = IBV_WR_SEND;
    wr.num_sge = 2;
    wr.sg_list = sge;
    wr.wr.ud.ah = rdma->ah;
    wr.wr.ud.remote_qpn = remote_qpn;
    wr.wr.ud.remote_qkey = QKEY;
    wr.send_flags = IBV_SEND_SIGNALED;

    sleep(1);  // Wait for server to post receive

    if (ibv_post_send(rdma->qp, &wr, &bad_wr)) {
        perror("Failed to post send");
        exit(1);
    }

    printf("Client: Message sent!\n");

    struct ibv_wc wc;
    printf("Client poll message completion...\n");
    while (ibv_poll_cq(rdma->cq, 1, &wc) < 1);

#if USE_GDR == 0
    printf("Client sent: %s | %s\n", rdma->buf1, rdma->buf2);
#else
    memset(h_data, 0, BUFFER_SIZE);
    cudaMemcpy(h_data, rdma->buf1, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    printf("Client sent: %s | %s\n", h_data, rdma->buf2);
    free(h_data);
#endif
}

int main(int argc, char *argv[]) {
    struct rdma_context *rdma = init_rdma();

    if (argc == 2)
        run_client(rdma, argv[1], GID_INDEX);
    else
        run_server(rdma, GID_INDEX);

    return 0;
}

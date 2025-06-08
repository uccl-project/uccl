#include "rdma.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

// Define globals
struct ibv_context* context = nullptr;
struct ibv_pd* pd = nullptr;
struct ibv_qp* qp = nullptr;
struct ibv_mr* mr = nullptr;
uint32_t rkey = 0;
uintptr_t remote_addr = 0;
uint32_t remote_rkey = 0;

constexpr int TCP_PORT = 18515;

void exchange_connection_info(int rank, const char* peer_ip, RDMAConnectionInfo* local, RDMAConnectionInfo* remote) {
    int sockfd;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));

    if (rank == 0) {
        // Listen
        int listenfd = socket(AF_INET, SOCK_STREAM, 0);
        addr.sin_family = AF_INET;
        addr.sin_port = htons(TCP_PORT);
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(listenfd, (struct sockaddr*)&addr, sizeof(addr));
        listen(listenfd, 1);

        socklen_t len = sizeof(addr);
        sockfd = accept(listenfd, (struct sockaddr*)&addr, &len);
        close(listenfd);
    } else {
        // Connect
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        addr.sin_family = AF_INET;
        addr.sin_port = htons(TCP_PORT);
        inet_pton(AF_INET, peer_ip, &addr.sin_addr);
        while (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            sleep(1);  // retry
        }
    }

    // Exchange info
    send(sockfd, local, sizeof(*local), 0);
    recv(sockfd, remote, sizeof(*remote), MSG_WAITALL);

    close(sockfd);
}

void setup_rdma(void* gpu_buffer, size_t size) {
    // 1. Get device list and open the first one
    struct ibv_device** dev_list = ibv_get_device_list(NULL);
    if (!dev_list) {
        perror("Failed to get IB devices list");
        exit(1);
    }

    context = ibv_open_device(dev_list[0]);
    if (!context) {
        perror("Failed to open device");
        exit(1);
    }
    ibv_free_device_list(dev_list);

    // 2. Allocate a Protection Domain
    pd = ibv_alloc_pd(context);
    if (!pd) {
        perror("Failed to allocate PD");
        exit(1);
    }

    // 3. Register the GPU memory
    mr = ibv_reg_mr(pd, gpu_buffer, size,
                    IBV_ACCESS_LOCAL_WRITE |
                    IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_READ |
                    IBV_ACCESS_ZERO_BASED);

    if (!mr) {
        perror("ibv_reg_mr (GPUDirect) failed");
        exit(1);
    }
    rkey = mr->rkey;
}

// Dummy `rdma_write_stub` — real post
void rdma_write_stub(int dst_rank, void* local_dev_ptr, size_t bytes) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr   = reinterpret_cast<uintptr_t>(local_dev_ptr);  // GPU memory address
    sge.length = bytes;
    sge.lkey   = mr->lkey;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id      = 0;
    wr.opcode     = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;

    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey        = remote_rkey;

    struct ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
        perror("ibv_post_send failed");
        exit(1);
    }
}

#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
bool GdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  return KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
         KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
         KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

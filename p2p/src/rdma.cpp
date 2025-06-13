#include "rdma.hpp"
#include "common.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

// Define globals
struct ibv_context* context = nullptr;
struct ibv_pd* pd = nullptr;
struct ibv_mr* mr = nullptr;
uint32_t rkey = 0;

// Define thread_local structs
thread_local struct ibv_qp* qp = nullptr;
thread_local uintptr_t remote_addr = 0;
thread_local uint32_t remote_rkey = 0;

constexpr int TCP_PORT = 18515;
static std::atomic<uint64_t> g_posted = 0;     // WRs posted
static std::atomic<uint64_t> g_completed = 0;  // CQEs seen
std::atomic<bool> g_progress_run{true};

void exchange_connection_info(int rank, char const* peer_ip, int tid,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote) {
  int sockfd;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  printf("Rank %d exchanging RDMA connection info with peer %s\n", rank,
         peer_ip);
  if (rank == 0) {
    // Listen
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT + tid);
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
    addr.sin_port = htons(TCP_PORT + tid);
    inet_pton(AF_INET, peer_ip, &addr.sin_addr);
    while (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      sleep(1);  // retry
    }
  }

  // Exchange info
  send(sockfd, local, sizeof(*local), 0);
  recv(sockfd, remote, sizeof(*remote), MSG_WAITALL);
  close(sockfd);

  printf(
      "Rank %d exchanged RDMA info: addr=0x%lx, rkey=0x%x, "
      "qp_num=%u, psn=%u\n",
      rank, remote->addr, remote->rkey, remote->qp_num, remote->psn);
}

void global_rdma_init(void* gpu_buf, size_t bytes, RDMAConnectionInfo* local,
                      int rank) {
  static std::once_flag flag;
  std::call_once(flag, [&] {
    setup_rdma(gpu_buf, bytes, local, rank);  // your existing function
  });
}

ibv_cq* create_per_thread_cq() {
  ibv_cq* cq;
  int cq_depth = kMaxOutstandingSends * 2;
  cq = ibv_create_cq(context, /* cqe */ cq_depth, /* user_context */ nullptr,
                     /* channel */ nullptr, /* comp_vector */ 0);
  if (!cq) {
    perror("Failed to create CQ");
    exit(1);
  }
  return cq;
}

void create_per_thread_qp(void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank,
                          ibv_cq* cq) {
  if (qp) return;  // Already initialized for this thread
  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = cq;
  qp_init_attr.recv_cq = cq;
  qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
  qp_init_attr.cap.max_send_wr =
      kMaxOutstandingSends * 2;      // max outstanding sends
  qp_init_attr.cap.max_recv_wr = 1;  // max outstanding recvs
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.sq_sig_all = 0;

  qp = ibv_create_qp(pd, &qp_init_attr);
  if (!qp) {
    perror("Failed to create QP");
    exit(1);
  }

  // Query port
  struct ibv_port_attr port_attr;
  if (ibv_query_port(context, 1, &port_attr)) {
    perror("Failed to query port");
    exit(1);
  }
  printf("Local LID: 0x%x\n", port_attr.lid);
  // Fill local connection info
  local_info->qp_num = qp->qp_num;
  local_info->lid = port_attr.lid;
  local_info->rkey = rkey;
  local_info->addr = reinterpret_cast<uintptr_t>(gpu_buffer);
  local_info->psn = rand() & 0xffffff;  // random psn
  memset(local_info->gid, 0, 16);
  printf("Local RDMA info: addr=0x%lx, rkey=0x%x, qp_num=%u, psn=%u\n",
         local_info->addr, local_info->rkey, local_info->qp_num,
         local_info->psn);
}

void setup_rdma(void* gpu_buffer, size_t size, RDMAConnectionInfo* local_info,
                int rank) {
  if (qp) return;

  srand(time(NULL) + getpid() + rank * 1000);
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
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_ZERO_BASED);

  if (!mr) {
    perror("ibv_reg_mr (GPUDirect) failed");
    exit(1);
  }
  if (rkey != 0) {
    perror("rkey already set, this should not happen");
    exit(1);
  }
  rkey = mr->rkey;
}

void modify_qp_to_init() {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));

  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;  // HCA port you use
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  if (ibv_modify_qp(qp, &attr, flags)) {
    perror("Failed to modify QP to INIT");
    exit(1);
  }

  printf("QP modified to INIT state\n");
}

void modify_qp_to_rtr(RDMAConnectionInfo* remote) {
  int is_roce = 0;

  struct ibv_port_attr port_attr;
  if (ibv_query_port(context, 1, &port_attr)) {
    perror("Failed to query port");
    exit(1);
  }

  if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
    printf("RoCE detected (Ethernet)\n");
    is_roce = 1;
  } else if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    printf("InfiniBand detected\n");
    is_roce = 0;
  } else {
    printf("Unknown link layer: %d\n", port_attr.link_layer);
    exit(1);
  }

  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = port_attr.active_mtu;
  attr.dest_qp_num = remote->qp_num;
  attr.rq_psn = remote->psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  if (is_roce) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.grh.hop_limit = 1;
    // Fill GID from remote_info
    memcpy(&attr.ah_attr.grh.dgid, remote->gid, 16);
    attr.ah_attr.grh.sgid_index = 0;  // Assume GID index 0
  } else {
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remote->lid;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.static_rate = 0;
    memset(&attr.ah_attr.grh, 0, sizeof(attr.ah_attr.grh));  // Safe
  }

  int flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  printf("Remote LID: 0x%x, QPN: %u, PSN: %u\n", remote->lid, remote->qp_num,
         remote->psn);
  printf("Verifying port state:\n");
  printf("  link_layer: %s\n", (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET)
                                   ? "Ethernet (RoCE)"
                                   : "InfiniBand");
  printf("  port_state: %s\n",
         (port_attr.state == IBV_PORT_ACTIVE) ? "ACTIVE" : "NOT ACTIVE");
  printf("  max_mtu: %d\n", port_attr.max_mtu);
  printf("  active_mtu: %d\n", port_attr.active_mtu);
  printf("  lid: 0x%x\n", port_attr.lid);

  int ret = ibv_modify_qp(qp, &attr, flags);
  if (ret) {
    perror("Failed to modify QP to RTR");
    fprintf(stderr, "errno: %d\n", errno);
    exit(1);
  }

  printf("QP modified to RTR state\n");
}

void modify_qp_to_rts(RDMAConnectionInfo* local_info) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_info->psn;
  attr.max_rd_atomic = 1;

  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

  if (ibv_modify_qp(qp, &attr, flags)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }

  printf("QP modified to RTS state\n");
}

void post_rdma_async_chained(void* buf, size_t bytes, size_t num_wrs,
                             std::vector<uint64_t> wrs_to_post, ibv_cq* cq,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex) {
  // while (g_posted.load() - g_completed.load() > kMaxOutstandingSends) {
  //   poll_completions(cq, finished_wrs, finished_wrs_mutex);
  // }

  std::vector<struct ibv_sge> sges(num_wrs);
  std::vector<struct ibv_send_wr> wrs(num_wrs);
  if (num_wrs != wrs_to_post.size()) {
    fprintf(stderr,
            "Error: num_wrs (%ld) does not match wrs_to_post size (%zu)\n",
            num_wrs, wrs_to_post.size());
    exit(1);
  }

  for (size_t i = 0; i < num_wrs; ++i) {
    int offset = kNumThBlocks > i ? i : (i % kNumThBlocks);
    sges[i].addr = (uintptr_t)buf + offset * bytes;
    sges[i].length = (uint32_t)bytes;
    sges[i].lkey = mr->lkey;

    wrs[i].opcode = IBV_WR_RDMA_WRITE;
    wrs[i].sg_list = &sges[i];
    wrs[i].num_sge = 1;
    wrs[i].wr.rdma.remote_addr = remote_addr + offset * bytes;
    wrs[i].wr.rdma.rkey = remote_rkey;

    // printf("Posting WR %zu: addr=0x%lx, rkey=0x%x, remote_addr=0x%lx\n",
    //        wrs_to_post[i], sges[i].addr, mr->rkey,
    //        wrs[i].wr.rdma.remote_addr);
    wrs[i].wr_id = wrs_to_post[i];

    if ((i + 1) % kSignalledEvery == 0)
      wrs[i].send_flags = IBV_SEND_SIGNALED;  // generate a CQE
    else
      wrs[i].send_flags = 0;  // no CQE → cheaper

    if (i < num_wrs - 1) {
      wrs[i].next = &wrs[i + 1];  // chain to next WR
    } else {
      wrs[i].next = nullptr;  // last WR in the chain
    }
  }
  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(qp, &wrs[0], &bad);
  if (ret) {
    fprintf(stderr, "ibv_post_send failed: %s (ret=%d)\n", strerror(ret), ret);
    if (bad) {
      fprintf(stderr, "Bad WR at address: %p\n", bad);
    }
    exit(1);
  }
  g_posted.fetch_add(num_wrs, std::memory_order_relaxed);
}

void post_rdma_async(void* buf, size_t bytes, uint64_t wr_id, ibv_cq* cq,
                     std::unordered_set<uint64_t>& finished_wrs,
                     std::mutex& finished_wrs_mutex) {
  /* Make it a closed loop to limit the maximum outstanding sends. */
  while (g_posted.load() - g_completed.load() > kMaxOutstandingSends) {
    poll_completions(cq, finished_wrs, finished_wrs_mutex);
  }

  struct ibv_sge sge {
    .addr = (uintptr_t)buf, .length = (uint32_t)bytes, .lkey = mr->lkey
  };

  struct ibv_send_wr wr {};
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;
  wr.wr_id = wr_id;

  if (wr_id % kSignalledEvery == 0)
    wr.send_flags = IBV_SEND_SIGNALED;  // generate a CQE
  else
    wr.send_flags = 0;

  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(qp, &wr, &bad);
  if (ret) {
    fprintf(stderr, "ibv_post_send failed: %s (ret=%d)\n", strerror(ret), ret);
    if (bad) {
      fprintf(stderr, "Bad WR at address: %p\n", bad);
    }
    // Optionally query QP state here for more info
    exit(1);
  }

  g_posted.fetch_add(1, std::memory_order_relaxed);
}

void rdma_write_stub(void* local_dev_ptr, size_t bytes) {
  struct ibv_qp_attr qattr;
  struct ibv_qp_init_attr qinit;
  ibv_query_qp(qp, &qattr, IBV_QP_STATE, &qinit);

  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = reinterpret_cast<uintptr_t>(local_dev_ptr);  // GPU memory address
  sge.length = bytes;
  sge.lkey = mr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;

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

void poll_completions(ibv_cq* cq, std::unordered_set<uint64_t>& finished_wrs,
                      std::mutex& finished_wrs_mutex) {
  struct ibv_wc wc[kMaxOutstandingSends];  // batch poll
  int ne = ibv_poll_cq(cq, kMaxOutstandingSends, wc);
  if (ne == 0) return;
  for (int i = 0; i < ne; ++i) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      fprintf(stderr, "CQE error wr_id=%llu status=%s\n",
              (unsigned long long)wc[i].wr_id, ibv_wc_status_str(wc[i].status));
      std::abort();
    }
    {
      std::lock_guard<std::mutex> lock(finished_wrs_mutex);
      finished_wrs.insert(wc[i].wr_id);
    }
  }
  g_completed.fetch_add(ne, std::memory_order_relaxed);
}

void per_thread_polling(int thread_idx, struct ibv_cq* per_thread_cq,
                        std::unordered_set<uint64_t>* per_thread_finished_wrs,
                        std::mutex* per_thread_finished_wrs_mutex) {
  pin_thread_to_cpu(thread_idx);
  printf("Progress thread started on CPU %d\n", sched_getcpu());

  while (per_thread_cq == nullptr && g_progress_run.load()) _mm_pause();
  printf("Progress thread %d: cq=%p\n", thread_idx, per_thread_cq);

  while (g_progress_run.load(std::memory_order_acquire)) {
    poll_completions(per_thread_cq, *per_thread_finished_wrs,
                     *per_thread_finished_wrs_mutex);
  }
}

bool check_cq_completion() {
  uint64_t posted = g_posted.load(std::memory_order_acquire);
  uint64_t completed = g_completed.load(std::memory_order_acquire);
  printf("check_cq_completion: g_completed: %ld, g_posted: %ld, total: %d\n",
         completed, posted, kIterations * kNumThBlocks);
  return completed * kSignalledEvery == posted &&
         kIterations * kNumThBlocks == completed;
}
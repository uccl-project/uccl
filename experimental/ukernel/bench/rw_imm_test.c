/*
 * Standalone RDMA write-with-immediate validation program.
 *
 * Usage:
 *   server: ./rw_imm_test --role server [--dev mlx5_0] [--port 18515] [--gpu]
 *   client: ./rw_imm_test --role client --ip <server_ip> [--dev mlx5_0] \
 *           [--port 18515] [--size 1048576] [--sge] [--gpu] [--gpu-id n]
 *
 * The receiver posts one recv WQE on the data QP and the sender performs
 * IBV_WR_RDMA_WRITE_WITH_IMM.  After the receiver observes the immediate
 * completion it checks whether the payload is already visible in the
 * remote buffer, and again after a short delay.  This isolates the
 * write-with-imm data-visibility behavior from the ukernel stack.
 *
 * With --gpu, the test uses CUDA device memory as both the RDMA source
 * and destination, which is the path relevant to ukernel.
 *
 * Build:
 *   gcc -O2 -o rw_imm_test rw_imm_test.c -libverbs -lcudart -lcuda
 */
#include <arpa/inet.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <getopt.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define DEFAULT_PORT 18515
#define DEFAULT_SIZE (1024 * 1024)
#define PATTERN 0x5A
#define MAX_QPS 8

struct peer_msg {
  uint16_t lid;
  uint16_t reserved;
  uint32_t qpn[MAX_QPS];
  uint8_t num_qps;
  uint8_t reserved2[3];
  uint32_t psn;
  uint64_t mr_addr;
  uint32_t rkey;
  uint32_t size;
  uint8_t gid[16];
};

struct ctx {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_qp *qps[MAX_QPS];
  int num_qps;
  struct ibv_mr *mr;
  struct ibv_mr *dummy_mr;
  char *buf;        /* host buffer (host mode) or host staging/check */
  char *dummy;      /* optional recv SGE scratch */
  void *gpu_buf;    /* GPU buffer (GPU mode) */
  size_t size;
  int use_sge;
  int use_gpu;
  int gpu_id;
  int split;
  int iters;
  int recv_pool;
  int threaded;
  atomic_int *ready;
};

static const char *g_dev_name = NULL;

static void *data_ptr(struct ctx *c);

static void die(const char *msg) {
  perror(msg);
  exit(1);
}

static void die_ibv(const char *msg, int err) {
  fprintf(stderr, "%s: %s\n", msg, strerror(err));
  exit(1);
}

static int create_qp(struct ctx *c, int port, int num_qps) {
  struct ibv_device **devs;
  struct ibv_device *dev = NULL;
  int ndev = 0;
  const char *dev_name = g_dev_name ? g_dev_name : getenv("UK_RDMA_DEV");
  devs = ibv_get_device_list(&ndev);
  if (!devs) return -1;
  if (dev_name) {
    for (int i = 0; i < ndev; i++) {
      if (strcmp(ibv_get_device_name(devs[i]), dev_name) == 0) {
        dev = devs[i];
        break;
      }
    }
  } else if (ndev > 0) {
    dev = devs[0];
  }
  if (!dev) {
    fprintf(stderr, "no RDMA device found\n");
    return -1;
  }
  c->ctx = ibv_open_device(dev);
  ibv_free_device_list(devs);
  if (!c->ctx) return -1;

  c->pd = ibv_alloc_pd(c->ctx);
  if (!c->pd) return -1;

  c->cq = ibv_create_cq(c->ctx, 4096, NULL, NULL, 0);
  if (!c->cq) return -1;

  if (num_qps < 1) num_qps = 1;
  if (num_qps > MAX_QPS) num_qps = MAX_QPS;
  c->num_qps = num_qps;
  for (int i = 0; i < num_qps; i++) {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = c->cq;
    attr.recv_cq = c->cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = 256;
    attr.cap.max_recv_wr = 256;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = 0;
    c->qps[i] = ibv_create_qp(c->pd, &attr);
    if (!c->qps[i]) return -1;
  }
  (void)port;
  return 0;
}

static int modify_qp_to_rts(struct ctx *c, struct peer_msg *peer) {
  int n = c->num_qps;
  for (int i = 0; i < n; i++) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(c->qps[i], &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS)) {
      return -1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = peer->qpn[i];
    attr.rq_psn = peer->psn;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 12;
    if (peer->lid != 0) {
      attr.ah_attr.is_global = 0;
      attr.ah_attr.dlid = peer->lid;
      attr.ah_attr.sl = 0;
      attr.ah_attr.src_path_bits = 0;
      attr.ah_attr.port_num = 1;
    } else {
      attr.ah_attr.is_global = 1;
      attr.ah_attr.port_num = 1;
      memcpy(&attr.ah_attr.grh.dgid, peer->gid, 16);
      attr.ah_attr.grh.sgid_index = 0;
      attr.ah_attr.grh.hop_limit = 64;
    }
    if (ibv_modify_qp(c->qps[i], &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
      return -1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 16;
    if (ibv_modify_qp(c->qps[i], &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC)) {
      return -1;
    }
  }
  return 0;
}

static int post_recv_one(struct ctx *c, int qp_idx) {
  struct ibv_recv_wr wr;
  struct ibv_sge sge;
  struct ibv_recv_wr *bad = NULL;
  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));
  if (c->use_sge) {
    sge.addr = (uintptr_t)c->dummy;
    sge.length = (uint32_t)c->size;
    sge.lkey = c->dummy_mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
  }
  return ibv_post_recv(c->qps[qp_idx], &wr, &bad);
}

static int post_recv_pool(struct ctx *c, int n) {
  for (int q = 0; q < c->num_qps; q++)
    for (int k = 0; k < n; k++)
      if (post_recv_one(c, q)) return -1;
  return 0;
}

static int post_write_imm_at(struct ctx *c, struct peer_msg *peer,
                             size_t off, int qp_idx) {
  struct ibv_sge sge, sge2;
  struct ibv_send_wr wr, wr2;
  struct ibv_send_wr *bad = NULL;
  memset(&sge, 0, sizeof(sge));
  memset(&sge2, 0, sizeof(sge2));
  memset(&wr, 0, sizeof(wr));
  memset(&wr2, 0, sizeof(wr2));

  if (c->split) {
    /* Mimic ukernel: first plain RDMA write, then write-with-imm. */
    uint32_t half = (uint32_t)(c->size / 2);
    sge.addr = (uintptr_t)data_ptr(c) + off;
    sge.length = half;
    sge.lkey = c->mr->lkey;
    sge2.addr = (uintptr_t)data_ptr(c) + off + half;
    sge2.length = half;
    sge2.lkey = c->mr->lkey;

    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = 0;
    wr.wr.rdma.remote_addr = peer->mr_addr + off;
    wr.wr.rdma.rkey = peer->rkey;

    wr2.wr_id = 2;
    wr2.sg_list = &sge2;
    wr2.num_sge = 1;
    wr2.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr2.send_flags = IBV_SEND_SIGNALED;
    wr2.imm_data = htonl(0x12345678);
    wr2.wr.rdma.remote_addr = peer->mr_addr + off + half;
    wr2.wr.rdma.rkey = peer->rkey;
    wr.next = &wr2;
    return ibv_post_send(c->qps[qp_idx], &wr, &bad);
  }

  sge.addr = (uintptr_t)data_ptr(c) + off;
  sge.length = (uint32_t)c->size;
  sge.lkey = c->mr->lkey;

  wr.wr_id = 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(0x12345678);
  wr.wr.rdma.remote_addr = peer->mr_addr + off;
  wr.wr.rdma.rkey = peer->rkey;
  return ibv_post_send(c->qps[qp_idx], &wr, &bad);
}

static int wait_send_completion(struct ctx *c) {
  struct ibv_wc wc;
  int n;
  do {
    n = ibv_poll_cq(c->cq, 1, &wc);
  } while (n == 0);
  if (n < 0) return -1;
  if (wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "send WC status %d\n", wc.status);
    return -1;
  }
  return 0;
}

static int wait_recv_imm(struct ctx *c, int *qp_idx) {
  struct ibv_wc wc;
  int n;
  do {
    n = ibv_poll_cq(c->cq, 1, &wc);
  } while (n == 0);
  if (n < 0) return -1;
  if (wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
    fprintf(stderr, "recv WC status=%d opcode=%d\n", wc.status, wc.opcode);
    return -1;
  }
  *qp_idx = -1;
  for (int i = 0; i < c->num_qps; i++) {
    if (c->qps[i]->qp_num == wc.qp_num) {
      *qp_idx = i;
      break;
    }
  }
  printf("receiver got IMM=0x%08x qp_num=%u qp_idx=%d\n", ntohl(wc.imm_data),
         wc.qp_num, *qp_idx);
  return 0;
}

static void *poll_thread_fn(void *arg) {
  struct ctx *c = (struct ctx *)arg;
  for (int i = 0; i < c->iters; i++) {
    int qp_idx = 0;
    if (wait_recv_imm(c, &qp_idx)) return (void *)1;
    /* Match ukernel ordering: publish the completion to the reader
     * BEFORE reposting the recv WQE. */
    atomic_store_explicit(&c->ready[i], 1, memory_order_release);
    if (qp_idx >= 0) (void)post_recv_one(c, qp_idx);
  }
  return NULL;
}

static int check_buffer(char *buf, size_t size, int print) {
  size_t bad = 0;
  for (size_t i = 0; i < size; i++) {
    if ((unsigned char)buf[i] != PATTERN) bad++;
  }
  if (print) {
    printf("  buffer check: bad=%zu first=%02x %02x %02x %02x\n", bad,
           (unsigned char)buf[0], (unsigned char)buf[1],
           (unsigned char)buf[2], (unsigned char)buf[3]);
  }
  return bad == 0;
}

static void *data_ptr(struct ctx *c) {
  return c->use_gpu ? c->gpu_buf : (void *)c->buf;
}

static void fill_data(struct ctx *c) {
  size_t total = c->size * (size_t)c->iters;
  if (c->use_gpu) {
    cudaError_t e = cudaMemset(c->gpu_buf, PATTERN, total);
    if (e != cudaSuccess) {
      fprintf(stderr, "cudaMemset: %s\n", cudaGetErrorString(e));
      exit(1);
    }
    cudaDeviceSynchronize();
  } else {
    memset(c->buf, PATTERN, total);
  }
}

static int read_and_check(struct ctx *c, int print) {
  if (c->use_gpu) {
    cudaError_t e = cudaMemcpy(c->buf, c->gpu_buf, c->size,
                               cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy D2H: %s\n", cudaGetErrorString(e));
      exit(1);
    }
    cudaDeviceSynchronize();
  }
  return check_buffer(c->buf, c->size, print);
}

static int read_and_check_at(struct ctx *c, size_t off, int print) {
  if (c->use_gpu) {
    cudaError_t e = cudaMemcpy(c->buf + off, (char *)c->gpu_buf + off,
                               c->size, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy D2H at %zu: %s\n", off,
              cudaGetErrorString(e));
      exit(1);
    }
    cudaDeviceSynchronize();
  }
  return check_buffer(c->buf + off, c->size, print);
}

static void cuda_check(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    exit(1);
  }
}

static int register_gpu_mr(struct ctx *c) {
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
               IBV_ACCESS_REMOTE_READ;
  size_t total = c->size * (size_t)c->iters;
  c->mr = ibv_reg_mr(c->pd, c->gpu_buf, total, access);
  if (c->mr) return 0;

  /* Fallback: DMA-BUF registration for GPU memory. */
  int dmabuf_fd = -1;
  CUresult cu_ret = cuMemGetHandleForAddressRange(
      &dmabuf_fd, (CUdeviceptr)c->gpu_buf, total,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (cu_ret == CUDA_SUCCESS && dmabuf_fd >= 0) {
    c->mr = ibv_reg_dmabuf_mr(c->pd, 0, total,
                              (uint64_t)c->gpu_buf, dmabuf_fd, access);
  }
  if (!c->mr) {
    fprintf(stderr, "register_gpu_mr: ibv_reg_mr and DMA-BUF both failed\n");
    return -1;
  }
  return 0;
}

static int tcp_connect(const char *ip, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) die("socket");
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) die("inet_pton");
  if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) die("connect");
  return fd;
}

static int tcp_listen(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) die("socket");
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) die("bind");
  if (listen(fd, 1) < 0) die("listen");
  int cfd = accept(fd, NULL, NULL);
  if (cfd < 0) die("accept");
  close(fd);
  return cfd;
}

static void send_msg(int fd, struct peer_msg *m) {
  char *p = (char *)m;
  size_t left = sizeof(*m);
  while (left) {
    ssize_t n = send(fd, p, left, 0);
    if (n <= 0) die("send");
    p += n;
    left -= (size_t)n;
  }
}

static void recv_msg(int fd, struct peer_msg *m) {
  char *p = (char *)m;
  size_t left = sizeof(*m);
  while (left) {
    ssize_t n = recv(fd, p, left, 0);
    if (n <= 0) die("recv");
    p += n;
    left -= (size_t)n;
  }
}

static void fill_peer_msg(struct ctx *c, struct peer_msg *m) {
  struct ibv_port_attr pattr;
  memset(m, 0, sizeof(*m));
  ibv_query_port(c->ctx, 1, &pattr);
  m->lid = pattr.lid;
  m->num_qps = (uint8_t)c->num_qps;
  for (int i = 0; i < c->num_qps; i++) m->qpn[i] = c->qps[i]->qp_num;
  m->psn = 0;
  m->mr_addr = (uintptr_t)data_ptr(c);
  m->rkey = c->mr->rkey;
  m->size = (uint32_t)c->size;
  if (ibv_query_gid(c->ctx, 1, 0, (union ibv_gid *)m->gid)) {
    memset(m->gid, 0, 16);
  }
}

int main(int argc, char **argv) {
  const char *role = NULL;
  const char *ip = NULL;
  int port = DEFAULT_PORT;
  size_t size = DEFAULT_SIZE;
  int use_sge = 0;
  int use_gpu = 0;
  int gpu_id = 0;
  int split = 0;
  int iters = 1;
  int recv_pool = 1;
  int qps = 1;
  int threaded = 0;
  int pin_qp = -1;

  static struct option opts[] = {
      {"role", required_argument, 0, 'r'},
      {"ip", required_argument, 0, 'i'},
      {"port", required_argument, 0, 'p'},
      {"size", required_argument, 0, 's'},
      {"sge", no_argument, 0, 'g'},
      {"dev", required_argument, 0, 'd'},
      {"gpu", no_argument, 0, 'u'},
      {"gpu-id", required_argument, 0, 'n'},
      {"split", no_argument, 0, 'S'},
      {"iters", required_argument, 0, 'I'},
      {"recv-pool", required_argument, 0, 'R'},
      {"qps", required_argument, 0, 'Q'},
      {"threaded", no_argument, 0, 'T'},
      {"pin-qp", required_argument, 0, 'P'},
      {0, 0, 0, 0},
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "r:i:p:s:gd:un:SI:R:Q:TP:", opts, NULL)) !=
         -1) {
    switch (opt) {
      case 'r': role = optarg; break;
      case 'i': ip = optarg; break;
      case 'p': port = atoi(optarg); break;
      case 's': size = (size_t)atoll(optarg); break;
      case 'g': use_sge = 1; break;
      case 'd': g_dev_name = optarg; break;
      case 'u': use_gpu = 1; break;
      case 'n': gpu_id = atoi(optarg); break;
      case 'S': split = 1; break;
      case 'I': iters = atoi(optarg); break;
      case 'R': recv_pool = atoi(optarg); break;
      case 'Q': qps = atoi(optarg); break;
      case 'T': threaded = 1; break;
      case 'P': pin_qp = atoi(optarg); break;
      default: return 2;
    }
  }
  if (!role || (strcmp(role, "server") && strcmp(role, "client"))) {
    fprintf(stderr,
            "usage: %s --role server|client [--ip ip] [--port p] [--size n] "
            "[--sge] [--gpu] [--gpu-id n] [--dev d] [--iters n] "
            "[--recv-pool n] [--qps n] [--threaded] [--pin-qp n]\n",
            argv[0]);
    return 2;
  }
  if (strcmp(role, "client") == 0 && !ip) {
    fprintf(stderr, "client requires --ip\n");
    return 2;
  }

  struct ctx c;
  memset(&c, 0, sizeof(c));
  c.size = size;
  c.use_sge = use_sge;
  c.iters = iters;
  c.recv_pool = recv_pool;
  c.threaded = threaded;
  if (create_qp(&c, port, qps)) die_ibv("create_qp", errno);

  size_t total = size * (size_t)iters;
  c.use_gpu = use_gpu;
  c.gpu_id = gpu_id;
  c.split = split;
  c.buf = aligned_alloc(4096, total);
  if (!c.buf) die("aligned_alloc");
  memset(c.buf, 0, total);

  if (use_gpu) {
    cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice");
    cuda_check(cudaMalloc(&c.gpu_buf, total), "cudaMalloc");
    cuda_check(cudaMemset(c.gpu_buf, 0, total), "cudaMemset");
    cudaDeviceSynchronize();
    if (register_gpu_mr(&c)) return 1;
    printf("using GPU %d buffer %p total=%zu\n", gpu_id, c.gpu_buf, total);
  } else {
    c.mr = ibv_reg_mr(c.pd, c.buf, total,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_READ);
    if (!c.mr) die_ibv("reg_mr", errno);
  }

  if (use_sge) {
    c.dummy = aligned_alloc(4096, size);
    if (!c.dummy) die("aligned_alloc dummy");
    memset(c.dummy, 0, size);
    c.dummy_mr = ibv_reg_mr(c.pd, c.dummy, size, IBV_ACCESS_LOCAL_WRITE);
    if (!c.dummy_mr) die_ibv("reg_dummy_mr", errno);
  }

  int fd;
  struct peer_msg my_msg, peer_msg;
  fill_peer_msg(&c, &my_msg);

  if (strcmp(role, "server") == 0) {
    printf("server listening on port %d\n", port);
    fd = tcp_listen(port);
    recv_msg(fd, &peer_msg);
    send_msg(fd, &my_msg);
  } else {
    printf("client connecting to %s:%d\n", ip, port);
    fd = tcp_connect(ip, port);
    send_msg(fd, &my_msg);
    recv_msg(fd, &peer_msg);
  }
  printf("%s: peer lid=%u num_qps=%d remote=0x%llx rkey=%u\n", role,
         peer_msg.lid, (int)peer_msg.num_qps,
         (unsigned long long)peer_msg.mr_addr, peer_msg.rkey);

  if (modify_qp_to_rts(&c, &peer_msg)) die_ibv("modify_qp_to_rts", errno);
  printf("%s: QP RTS\n", role);

  int rc = 0;
  if (strcmp(role, "server") == 0) {
    /* Server is the receiver: post a recv pool and wait for each IMM. */
    if (post_recv_pool(&c, c.recv_pool)) die_ibv("post_recv_pool", errno);
    printf("server: waiting for %d write-with-imm...\n", c.iters);

    if (c.threaded) {
      /* ukernel-like handoff: poll thread receives IMM and marks ready;
       * this (reader) thread dequeues the marker and reads GPU data. */
      c.ready = calloc((size_t)c.iters, sizeof(atomic_int));
      if (!c.ready) die("calloc ready");
      pthread_t pt;
      if (pthread_create(&pt, NULL, poll_thread_fn, &c)) die("pthread_create");
      for (int i = 0; i < c.iters; i++) {
        while (atomic_load_explicit(&c.ready[i], memory_order_acquire) == 0)
          ;
        size_t off = (size_t)i * c.size;
        int ok_immediate = read_and_check_at(&c, off, 1);
        printf("server: iter %d data visible immediately: %s\n", i,
               ok_immediate ? "YES" : "NO");
        if (!ok_immediate) {
          usleep(100000);
          int ok_delayed = read_and_check_at(&c, off, 1);
          printf("server: iter %d data visible after 100ms: %s\n", i,
                 ok_delayed ? "YES" : "NO");
          if (!ok_delayed) rc = 1;
        }
      }
      pthread_join(pt, NULL);
      free(c.ready);
      c.ready = NULL;
    } else {
      for (int i = 0; i < c.iters; i++) {
        int qp_idx = 0;
        if (wait_recv_imm(&c, &qp_idx)) return 1;
        if (qp_idx >= 0) (void)post_recv_one(&c, qp_idx);
        size_t off = (size_t)i * c.size;
        int ok_immediate = read_and_check_at(&c, off, 1);
        printf("server: iter %d data visible immediately: %s\n", i,
               ok_immediate ? "YES" : "NO");
        if (!ok_immediate) {
          usleep(100000);
          int ok_delayed = read_and_check_at(&c, off, 1);
          printf("server: iter %d data visible after 100ms: %s\n", i,
                 ok_delayed ? "YES" : "NO");
          if (!ok_delayed) rc = 1;
        }
      }
    }
  } else {
    /* Client is the sender. */
    fill_data(&c);
    printf("client: posting %d RDMA write-with-imm, size=%zu each\n",
           c.iters, c.size);
    for (int i = 0; i < c.iters; i++) {
      size_t off = (size_t)i * c.size;
      int qp_idx = (pin_qp >= 0) ? pin_qp : (i % c.num_qps);
      if (post_write_imm_at(&c, &peer_msg, off, qp_idx))
        die_ibv("post_write_imm", errno);
      if (wait_send_completion(&c)) return 1;
    }
    printf("client: all send completions OK\n");
    rc = 0;
  }

  /* Small pause so the peer can finish printing. */
  usleep(200000);
  close(fd);
  if (c.mr) ibv_dereg_mr(c.mr);
  if (c.dummy_mr) ibv_dereg_mr(c.dummy_mr);
  for (int i = 0; i < c.num_qps; i++)
    if (c.qps[i]) ibv_destroy_qp(c.qps[i]);
  if (c.cq) ibv_destroy_cq(c.cq);
  if (c.pd) ibv_dealloc_pd(c.pd);
  if (c.ctx) ibv_close_device(c.ctx);
  if (c.use_gpu && c.gpu_buf) cudaFree(c.gpu_buf);
  free(c.buf);
  free(c.dummy);
  return rc;
}

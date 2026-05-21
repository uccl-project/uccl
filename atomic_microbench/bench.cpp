// Microbenchmark comparing latency of:
//   (1) RDMA WRITE only
//   (2) RDMA WRITE + hardware atomic FETCH_AND_ADD (where supported)
//   (3) RDMA WRITE + UEP-emulated atomic via WRITE_WITH_IMM
//
// UEP-emulated atomic: a single RDMA_WRITE_WITH_IMM whose 32-bit immediate
// encodes a signed 15-bit value and a 13-bit aligned offset. The responder
// applies the atomic add locally upon consuming the RECV CQE (mirrors
// ep/src/rdma.cpp's AtomicsImm path in ep/include/rdma.hpp).
//
// Two-process bench: one server, one client. RC QP, RoCEv2 over a chosen GID.

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

// ---------- AtomicsImm (mirrors ep/include/rdma.hpp layout) -----------------
struct AtomicsImm {
  static constexpr int kV_BITS = 15;
  static constexpr uint32_t kV_MASK = (1u << kV_BITS) - 1;  // 0x7FFF
  static constexpr uint32_t kOFF_MASK = 0x1FFFu;            // 13 bits
  static constexpr int kV_SHIFT = 13;
  static constexpr int kREORDERABLE = 28;
  static constexpr int kBUFFER_IDX = 29;
  static constexpr int kIS_COMBINE = 30;
  static constexpr int kIS_ATOMICS = 31;

  static uint32_t PackAtomic(int v15, uint16_t off_aligned_bytes) {
    uint32_t vfield = static_cast<uint32_t>(v15) & kV_MASK;
    uint32_t imm = 0;
    imm |= (1u << kIS_ATOMICS);
    imm |= (vfield << kV_SHIFT);
    imm |= (off_aligned_bytes & kOFF_MASK);
    return imm;
  }
  static int GetValue(uint32_t imm) {
    int32_t x = static_cast<int32_t>(imm);
    return (x << 4) >> (4 + kV_SHIFT);
  }
  static uint16_t GetOff(uint32_t imm) { return imm & kOFF_MASK; }
};

// ---------- OOB exchange ---------------------------------------------------
struct QpInfo {
  uint32_t qpn;
  uint32_t psn;
  uint64_t addr;        // remote buffer for WRITE/WRITE_WITH_IMM
  uint32_t rkey;
  uint64_t atomic_addr; // remote 64-bit counter for HW atomic
  uint32_t atomic_rkey;
  uint8_t gid[16];
  uint16_t lid;
};

static int tcp_listen_and_accept(int port) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_ANY);
  a.sin_port = htons(port);
  if (bind(s, (sockaddr*)&a, sizeof(a)) < 0) { perror("bind"); exit(1); }
  if (listen(s, 1) < 0) { perror("listen"); exit(1); }
  int c = accept(s, nullptr, nullptr);
  close(s);
  return c;
}
static int tcp_connect(const char* host, int port) {
  int s = -1;
  for (int i = 0; i < 50; ++i) {
    s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, host, &a.sin_addr);
    if (connect(s, (sockaddr*)&a, sizeof(a)) == 0) return s;
    close(s);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  perror("connect");
  exit(1);
}
static void xchg(int sock, QpInfo* local, QpInfo* remote) {
  if (send(sock, local, sizeof(*local), 0) != (ssize_t)sizeof(*local)) exit(1);
  if (recv(sock, remote, sizeof(*remote), MSG_WAITALL) != (ssize_t)sizeof(*remote)) exit(1);
}

// ---------- helpers --------------------------------------------------------
struct Ctx {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* send_cq = nullptr;
  ibv_cq* recv_cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_mr* mr = nullptr;        // main buffer
  ibv_mr* amr = nullptr;       // 8B atomic counter buffer
  uint8_t* buf = nullptr;
  uint64_t* acnt = nullptr;
  size_t buf_bytes = 0;
  uint8_t port_num = 1;
  int gid_idx = 3;             // RoCE v2 by default
};

static ibv_context* open_dev(const char* name) {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  ibv_context* ctx = nullptr;
  for (int i = 0; i < n; ++i) {
    if (!name || strcmp(ibv_get_device_name(list[i]), name) == 0) {
      ctx = ibv_open_device(list[i]);
      break;
    }
  }
  ibv_free_device_list(list);
  if (!ctx) { fprintf(stderr, "open dev %s failed\n", name ? name : "(any)"); exit(1); }
  return ctx;
}

static void make_qp(Ctx& c) {
  ibv_cq* scq = ibv_create_cq(c.ctx, 1024, nullptr, nullptr, 0);
  ibv_cq* rcq = ibv_create_cq(c.ctx, 1024, nullptr, nullptr, 0);
  c.send_cq = scq; c.recv_cq = rcq;
  ibv_qp_init_attr ia{};
  ia.send_cq = scq;
  ia.recv_cq = rcq;
  ia.qp_type = IBV_QPT_RC;
  ia.sq_sig_all = 0;
  ia.cap.max_send_wr = 256;
  ia.cap.max_recv_wr = 4096;
  ia.cap.max_send_sge = 2;
  ia.cap.max_recv_sge = 2;
  ia.cap.max_inline_data = 64;
  c.qp = ibv_create_qp(c.pd, &ia);
  if (!c.qp) { perror("create_qp"); exit(1); }
}

static void to_init(Ctx& c) {
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = c.port_num;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(c.qp, &a, flags)) { perror("modify INIT"); exit(1); }
}
static void to_rtr(Ctx& c, const QpInfo& rem) {
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = IBV_MTU_1024;
  a.dest_qp_num = rem.qpn;
  a.rq_psn = rem.psn;
  a.max_dest_rd_atomic = 1;
  a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;
  a.ah_attr.port_num = c.port_num;
  a.ah_attr.grh.hop_limit = 64;
  a.ah_attr.grh.sgid_index = c.gid_idx;
  memcpy(&a.ah_attr.grh.dgid, rem.gid, 16);
  int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  if (ibv_modify_qp(c.qp, &a, flags)) { perror("modify RTR"); exit(1); }
}
static void to_rts(Ctx& c, uint32_t my_psn) {
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.sq_psn = my_psn;
  a.max_rd_atomic = 1;
  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  if (ibv_modify_qp(c.qp, &a, flags)) { perror("modify RTS"); exit(1); }
}

static void post_recv(Ctx& c, uint64_t wr_id) {
  ibv_sge sge{};  // ignored for RDMA_WRITE_WITH_IMM (no payload landing here)
  sge.addr = (uintptr_t)c.buf;
  sge.length = (uint32_t)c.buf_bytes;
  sge.lkey = c.mr->lkey;
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  if (ibv_post_recv(c.qp, &wr, &bad)) { perror("post_recv"); exit(1); }
}

static inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void poll_one(ibv_cq* cq, const char* tag) {
  ibv_wc wc;
  for (;;) {
    int n = ibv_poll_cq(cq, 1, &wc);
    if (n < 0) { fprintf(stderr, "poll_cq err\n"); exit(1); }
    if (n == 0) continue;
    if (wc.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "[%s] WC failed: status=%s(%d) op=%d wr_id=%llu\n", tag,
              ibv_wc_status_str(wc.status), wc.status, wc.opcode,
              (unsigned long long)wc.wr_id);
      exit(1);
    }
    return;
  }
}

// ---------- the three operations -------------------------------------------
static void post_write(Ctx& c, const QpInfo& rem, size_t bytes, bool signaled) {
  ibv_sge sge{};
  sge.addr = (uintptr_t)c.buf;
  sge.length = (uint32_t)bytes;
  sge.lkey = c.mr->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.wr.rdma.remote_addr = rem.addr;
  wr.wr.rdma.rkey = rem.rkey;
  if (ibv_post_send(c.qp, &wr, &bad)) { perror("post_send write"); exit(1); }
}

// Chain: WRITE then ATOMIC_FETCH_AND_ADD. Only last is signaled.
static void post_write_then_hw_atomic(Ctx& c, const QpInfo& rem, size_t bytes) {
  ibv_sge sge_w{};
  sge_w.addr = (uintptr_t)c.buf;
  sge_w.length = (uint32_t)bytes;
  sge_w.lkey = c.mr->lkey;
  ibv_sge sge_a{};
  sge_a.addr = (uintptr_t)c.acnt;  // local landing for fetched value
  sge_a.length = 8;
  sge_a.lkey = c.amr->lkey;

  ibv_send_wr w_write{}, w_atom{}, *bad = nullptr;
  w_write.wr_id = 1;
  w_write.sg_list = &sge_w;
  w_write.num_sge = 1;
  w_write.opcode = IBV_WR_RDMA_WRITE;
  w_write.send_flags = 0;  // unsignaled
  w_write.wr.rdma.remote_addr = rem.addr;
  w_write.wr.rdma.rkey = rem.rkey;
  w_write.next = &w_atom;

  w_atom.wr_id = 2;
  w_atom.sg_list = &sge_a;
  w_atom.num_sge = 1;
  w_atom.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  w_atom.send_flags = IBV_SEND_SIGNALED;
  w_atom.wr.atomic.remote_addr = rem.atomic_addr;
  w_atom.wr.atomic.rkey = rem.atomic_rkey;
  w_atom.wr.atomic.compare_add = 1;
  w_atom.next = nullptr;

  if (ibv_post_send(c.qp, &w_write, &bad)) {
    perror("post_send write+atomic");
    fprintf(stderr, "  (HW atomic chain may not be supported on this NIC)\n");
    exit(1);
  }
}

static void post_write_with_imm(Ctx& c, const QpInfo& rem, size_t bytes,
                                int v15, uint16_t off_aligned) {
  ibv_sge sge{};
  sge.addr = (uintptr_t)c.buf;
  sge.length = (uint32_t)bytes;
  sge.lkey = c.mr->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  uint32_t imm = AtomicsImm::PackAtomic(v15, off_aligned);
  wr.imm_data = htonl(imm);
  wr.wr.rdma.remote_addr = rem.addr;
  wr.wr.rdma.rkey = rem.rkey;
  if (ibv_post_send(c.qp, &wr, &bad)) { perror("post_send write_imm"); exit(1); }
}

// Receiver-side handling for UEP atomic: pull RECV CQE, decode imm, apply add.
static void uep_recv_apply(Ctx& c) {
  ibv_wc wc;
  for (;;) {
    int n = ibv_poll_cq(c.recv_cq, 1, &wc);
    if (n < 0) { fprintf(stderr, "poll_cq recv err\n"); exit(1); }
    if (n == 0) continue;
    if (wc.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "recv WC failed: %s\n", ibv_wc_status_str(wc.status));
      exit(1);
    }
    if (wc.wc_flags & IBV_WC_WITH_IMM) {
      uint32_t imm = ntohl(wc.imm_data);
      int v = AtomicsImm::GetValue(imm);
      uint16_t off = AtomicsImm::GetOff(imm);
      auto* p = reinterpret_cast<std::atomic<int32_t>*>(c.buf + off);
      p->fetch_add(v, std::memory_order_relaxed);
    }
    // Replenish the consumed RECV so we never run out.
    post_recv(c, wc.wr_id);
    return;
  }
}

// ---------- bench loop -----------------------------------------------------
struct Stats { double avg_us, min_us, p50_us, p90_us, p99_us, p999_us, max_us, stddev_us; };
static Stats summarize(std::vector<uint64_t>& ns) {
  std::sort(ns.begin(), ns.end());
  double sum = 0;
  for (auto x : ns) sum += x;
  double mean = sum / ns.size();
  double var = 0;
  for (auto x : ns) { double d = (double)x - mean; var += d * d; }
  var /= ns.size();
  Stats s{};
  s.avg_us = mean / 1000.0;
  s.min_us = ns.front() / 1000.0;
  s.max_us = ns.back() / 1000.0;
  s.p50_us = ns[ns.size() / 2] / 1000.0;
  s.p90_us = ns[(ns.size() * 90) / 100] / 1000.0;
  s.p99_us = ns[(ns.size() * 99) / 100] / 1000.0;
  s.p999_us = ns[(ns.size() * 999) / 1000] / 1000.0;
  s.stddev_us = std::sqrt(var) / 1000.0;
  return s;
}

enum class Mode { WRITE, WRITE_HW_ATOMIC, WRITE_UEP_ATOMIC };

static Stats run_client(Ctx& c, const QpInfo& rem, Mode mode, size_t bytes,
                        int iters, const char* label) {
  // warmup
  for (int i = 0; i < 128; ++i) {
    if (mode == Mode::WRITE)            post_write(c, rem, bytes, true);
    else if (mode == Mode::WRITE_HW_ATOMIC) post_write_then_hw_atomic(c, rem, bytes);
    else                                 post_write_with_imm(c, rem, bytes, 1, 0);
    poll_one(c.send_cq, label);
  }

  std::vector<uint64_t> lats; lats.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    uint64_t t0 = now_ns();
    if (mode == Mode::WRITE)            post_write(c, rem, bytes, true);
    else if (mode == Mode::WRITE_HW_ATOMIC) post_write_then_hw_atomic(c, rem, bytes);
    else                                 post_write_with_imm(c, rem, bytes, 1, 0);
    poll_one(c.send_cq, label);
    lats.push_back(now_ns() - t0);
  }
  auto s = summarize(lats);
  printf("  %-26s bytes=%-5zu iters=%-7d  avg=%.2f us  min=%.2f  p50=%.2f  p90=%.2f  p99=%.2f  p99.9=%.2f  max=%.2f  std=%.2f\n",
         label, bytes, iters, s.avg_us, s.min_us, s.p50_us, s.p90_us, s.p99_us,
         s.p999_us, s.max_us, s.stddev_us);
  return s;
}

static void run_server_uep(Ctx& c, int iters) {
  int total = iters + 64;  // include warmup
  for (int i = 0; i < total; ++i) uep_recv_apply(c);
}

// ---------- main -----------------------------------------------------------
struct Args {
  bool server = false;
  const char* peer = nullptr;
  int port = 18515;
  const char* dev = "mlx5_0";
  int gid_idx = 3;
  int iters = 5000;
  int repeats = 1;
  const char* csv_path = nullptr;
  std::vector<size_t> sizes = {8, 64, 256, 1024, 4096};
};

static void usage() {
  fprintf(stderr,
          "Usage:\n"
          "  bench -s [-d dev] [-i gid_idx] [-p port] [-n iters]\n"
          "  bench -c <peer-ip> [-d dev] [-i gid_idx] [-p port] [-n iters]\n");
}

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "-s") a.server = true;
    else if (k == "-c") a.peer = argv[++i];
    else if (k == "-d") a.dev = argv[++i];
    else if (k == "-i") a.gid_idx = atoi(argv[++i]);
    else if (k == "-p") a.port = atoi(argv[++i]);
    else if (k == "-n") a.iters = atoi(argv[++i]);
    else if (k == "-r") a.repeats = atoi(argv[++i]);
    else if (k == "-o") a.csv_path = argv[++i];
    else { usage(); return 1; }
  }
  if (!a.server && !a.peer) { usage(); return 1; }

  Ctx c;
  c.gid_idx = a.gid_idx;
  c.ctx = open_dev(a.dev);
  c.pd = ibv_alloc_pd(c.ctx);
  c.buf_bytes = 1 << 16;
  if (posix_memalign((void**)&c.buf, 4096, c.buf_bytes)) { perror("alloc"); return 1; }
  memset(c.buf, 0, c.buf_bytes);
  c.mr = ibv_reg_mr(c.pd, c.buf, c.buf_bytes,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_READ);
  if (!c.mr) { perror("reg_mr"); return 1; }
  if (posix_memalign((void**)&c.acnt, 8, 64)) { perror("alloc a"); return 1; }
  *c.acnt = 0;
  c.amr = ibv_reg_mr(c.pd, c.acnt, 64,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  if (!c.amr) { perror("reg_mr atomic"); return 1; }

  // capability print
  ibv_device_attr da;
  ibv_query_device(c.ctx, &da);
  fprintf(stderr, "[info] dev=%s atomic_cap=%d (0=NONE,1=HCA,2=GLOB) gid_idx=%d\n",
          a.dev, da.atomic_cap, a.gid_idx);

  make_qp(c);
  to_init(c);

  // pre-post recvs for UEP path (server side)
  if (a.server) {
    for (int i = 0; i < 1024; ++i) post_recv(c, i);
  }

  QpInfo local{}, remote{};
  local.qpn = c.qp->qp_num;
  local.psn = 0x1234;
  local.addr = (uintptr_t)c.buf;
  local.rkey = c.mr->rkey;
  local.atomic_addr = (uintptr_t)c.acnt;
  local.atomic_rkey = c.amr->rkey;
  ibv_gid gid; ibv_query_gid(c.ctx, c.port_num, c.gid_idx, &gid);
  memcpy(local.gid, &gid, 16);
  local.lid = 0;

  int sock = a.server ? tcp_listen_and_accept(a.port) : tcp_connect(a.peer, a.port);
  xchg(sock, &local, &remote);

  to_rtr(c, remote);
  to_rts(c, local.psn);

  // Tiny TCP sync helper so client/server step through the same modes.
  auto sync = [&](const char* tag) {
    char m = a.server ? 'S' : 'C', r;
    send(sock, &m, 1, 0);
    recv(sock, &r, 1, MSG_WAITALL);
  };

  bool hw_atomic_ok = (da.atomic_cap != IBV_ATOMIC_NONE);
  if (a.server) {
    fprintf(stderr, "[server] ready; HW atomics %s\n",
            hw_atomic_ok ? "ENABLED" : "DISABLED");
  } else {
    fprintf(stderr, "[client] HW atomics %s on local NIC\n",
            hw_atomic_ok ? "ENABLED" : "DISABLED");
  }
  sync("ready");

  FILE* csv = nullptr;
  if (!a.server && a.csv_path) {
    csv = fopen(a.csv_path, "w");
    if (!csv) { perror("open csv"); return 1; }
    fprintf(csv, "timestamp,host_local,host_remote,dev,gid_idx,trial,mode,bytes,iters,"
                 "avg_us,min_us,p50_us,p90_us,p99_us,p999_us,max_us,stddev_us\n");
    fflush(csv);
  }
  char hostbuf[128]; gethostname(hostbuf, sizeof(hostbuf));
  std::string local_host = hostbuf;
  std::string remote_host = a.server ? "(client)" : a.peer;

  if (!a.server) printf("\n=== latency (client-observed; signaled per op) ===\n");

  auto phase_run = [&](Mode mode, size_t bytes, const char* label,
                       const char* tag, const char* csv_mode, int trial) {
    if (a.server) {
      if (mode == Mode::WRITE_UEP_ATOMIC) run_server_uep(c, a.iters);
      sync(tag);
      return;
    }
    Stats s = run_client(c, remote, mode, bytes, a.iters, label);
    if (csv) {
      char ts[64];
      time_t t = time(nullptr); strftime(ts, sizeof(ts), "%FT%T", localtime(&t));
      fprintf(csv, "%s,%s,%s,%s,%d,%d,%s,%zu,%d,"
                   "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
              ts, local_host.c_str(), remote_host.c_str(), a.dev, a.gid_idx,
              trial, csv_mode, bytes, a.iters,
              s.avg_us, s.min_us, s.p50_us, s.p90_us, s.p99_us, s.p999_us,
              s.max_us, s.stddev_us);
      fflush(csv);
    }
    sync(tag);
  };

  for (int trial = 0; trial < a.repeats; ++trial) {
    if (!a.server) printf("\n--- trial %d/%d ---\n", trial + 1, a.repeats);
    for (size_t bytes : a.sizes) {
      phase_run(Mode::WRITE, bytes, "WRITE only", "w", "write", trial);
      if (hw_atomic_ok)
        phase_run(Mode::WRITE_HW_ATOMIC, bytes, "WRITE + HW atomic FA", "h",
                  "write_hw_atomic", trial);
      else if (!a.server)
        printf("  %-26s SKIPPED (NIC has no atomic_cap)\n",
               "WRITE + HW atomic FA");
      phase_run(Mode::WRITE_UEP_ATOMIC, bytes, "WRITE + UEP atomic (imm)", "u",
                "write_uep_atomic", trial);
      if (!a.server) printf("  ----\n");
    }
  }
  if (csv) fclose(csv);
  if (a.server) fprintf(stderr, "[server] done\n");

  close(sock);
  return 0;
}

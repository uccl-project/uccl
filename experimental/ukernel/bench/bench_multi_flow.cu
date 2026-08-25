// bench_multi_flow.cu — single-process multi-flow RDMA delivery bench.
//
// One sender process issues N concurrent write-with-imm PutSignals to N
// remote peers per round and waits for ALL completions (the collective
// round barrier). N receiver processes each accept one flow and wait for
// the signal. Purpose: compare single-process multi-flow delivery
// against N separate p2p processes (bench_transport), isolating where
// the cross-node alltoall's flow-count degradation lives.

#include "gpu_rt.h"
#include "transport.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace UKernel::Transport;

static long garg(int argc, char** argv, const char* name, long def) {
  std::string n(name);
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == n && i + 1 < argc) return std::atol(argv[i + 1]);
    if (a.size() > n.size() && a.compare(0, n.size(), n) == 0 &&
        a[n.size()] == '=')
      return std::atol(a.c_str() + n.size() + 1);
  }
  return def;
}

static std::shared_ptr<Communicator> make_comm(int gpu, int rank, int nranks,
                                               std::string const& ip, int port,
                                               int local_id) {
  auto cfg = std::make_shared<CommunicatorConfig>();
  cfg->exchanger_ip = ip;
  cfg->exchanger_port = port;
  cfg->local_id = local_id;
  cfg->preferred_transport = PreferredTransport::Rdma;
  return std::make_shared<Communicator>(gpu, rank, nranks, cfg);
}

static bool wait_one(Communicator& comm, unsigned rid) {
  if (rid == 0) return true;
  while (true) {
    CompletionResult results[16];
    size_t n = comm.try_complete_put(results, 16);
    for (size_t i = 0; i < n; ++i)
      if (results[i].rid == rid) return !results[i].failed;

    SignalCompletion events[16];
    size_t m = comm.try_complete_sig_wait(events, 16);
    for (size_t i = 0; i < m; ++i)
      if (events[i].rid == rid) return !events[i].failed;

    size_t k = comm.try_complete_sig_send(results, 16);
    for (size_t i = 0; i < k; ++i)
      if (results[i].rid == rid) return !results[i].failed;

    if (n == 0 && m == 0 && k == 0) std::this_thread::yield();
  }
}

static bool wait_all(Communicator& comm, std::vector<unsigned>& reqs) {
  if (reqs.empty()) return true;
  std::unordered_set<unsigned> pending(reqs.begin(), reqs.end());
  pending.erase(0);
  while (!pending.empty()) {
    CompletionResult results[16];
    size_t n = comm.try_complete_put(results, 16);
    for (size_t i = 0; i < n; ++i) {
      if (results[i].failed) return false;
      pending.erase(results[i].rid);
    }
    SignalCompletion events[16];
    size_t m = comm.try_complete_sig_wait(events, 16);
    for (size_t i = 0; i < m; ++i) {
      if (events[i].failed) return false;
      pending.erase(events[i].rid);
    }
    size_t k = comm.try_complete_sig_send(results, 16);
    for (size_t i = 0; i < k; ++i) {
      if (results[i].failed) return false;
      pending.erase(results[i].rid);
    }
    if (n == 0 && m == 0 && k == 0) std::this_thread::yield();
  }
  reqs.clear();
  return true;
}

// Wait until at least one rid in `reqs` completes; erase the completed
// ones. Returns false on any failed completion.
static bool wait_any(Communicator& comm, std::vector<unsigned>& reqs) {
  if (reqs.empty()) return true;
  bool any = false;
  while (!any) {
    CompletionResult results[16];
    size_t n = comm.try_complete_put(results, 16);
    for (size_t i = 0; i < n; ++i) {
      if (results[i].failed) return false;
      auto it = std::find(reqs.begin(), reqs.end(), results[i].rid);
      if (it != reqs.end()) {
        reqs.erase(it);
        any = true;
      }
    }
    SignalCompletion events[16];
    size_t m = comm.try_complete_sig_wait(events, 16);
    for (size_t i = 0; i < m; ++i) {
      if (events[i].failed) return false;
      auto it = std::find(reqs.begin(), reqs.end(), events[i].rid);
      if (it != reqs.end()) {
        reqs.erase(it);
        any = true;
      }
    }
    size_t k = comm.try_complete_sig_send(results, 16);
    for (size_t i = 0; i < k; ++i) {
      if (results[i].failed) return false;
      auto it = std::find(reqs.begin(), reqs.end(), results[i].rid);
      if (it != reqs.end()) {
        reqs.erase(it);
        any = true;
      }
    }
    if (n == 0 && m == 0 && k == 0) std::this_thread::yield();
  }
  return true;
}

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  int rank = static_cast<int>(garg(argc, argv, "--rank", 0));
  int nranks = static_cast<int>(garg(argc, argv, "--nranks", 7));
  int gpu = static_cast<int>(garg(argc, argv, "--gpu", 0));
  int local_id = static_cast<int>(garg(argc, argv, "--local-id", gpu));
  size_t msg_size = static_cast<size_t>(garg(argc, argv, "--msg-size", 1 << 24));
  int iters = static_cast<int>(garg(argc, argv, "--iters", 20));
  int warmup = static_cast<int>(garg(argc, argv, "--warmup", 5));
  int msgs_per_flow = static_cast<int>(garg(argc, argv, "--msgs-per-flow", 1));
  size_t window = static_cast<size_t>(garg(argc, argv, "--window", 0));
  int echo = static_cast<int>(garg(argc, argv, "--echo", 0));
  std::string role = "sender";
  std::string ip = "0.0.0.0";
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto take = [&](std::string const& name, std::string& out) {
      if (a == name && i + 1 < argc) {
        out = argv[i + 1];
        ++i;
        return true;
      }
      if (a.size() > name.size() && a.compare(0, name.size(), name) == 0 &&
          a[name.size()] == '=') {
        out = a.substr(name.size() + 1);
        return true;
      }
      return false;
    };
    take("--role", role);
    take("--ip", ip);
  }
  int port = static_cast<int>(garg(argc, argv, "--port", 7100));

  GPU_RT_CHECK(gpuSetDevice(gpu));
  std::string conn_ip = (rank == 0) ? "0.0.0.0" : ip;
  auto comm = make_comm(gpu, rank, nranks, conn_ip, port, local_id);

  // Connect all pairs to rank 0 (the flow sender).
  for (int p = 0; p < nranks; ++p) {
    if (p == rank) continue;
    if (rank < p) {
      if (!comm->connect(p) || !comm->accept(p)) {
        fprintf(stderr, "[r%d] connect/accept to %d failed\n", rank, p);
        return 1;
      }
    } else {
      if (!comm->accept(p) || !comm->connect(p)) {
        fprintf(stderr, "[r%d] accept/connect from %d failed\n", rank, p);
        return 1;
      }
    }
  }

  if (role == "sender") {
    int nflows = nranks - 1;
    void* recv_buf = nullptr;
    GPU_RT_CHECK(gpuMalloc(&recv_buf, msg_size));
    if (!comm->reg_mr(1, recv_buf, msg_size, true)) {
      fprintf(stderr, "[r%d] reg_mr recv failed\n", rank);
      return 1;
    }
    for (int f = 0; f < nflows; ++f) {
      bool ok = comm->resolve_remote_buffer(f + 1, 1, 30000);
      fprintf(stderr, "[r%d] resolve peer %d -> %d\n", rank, f + 1, (int)ok);
      if (!ok) {
        fprintf(stderr, "[r%d] resolve remote buffer from %d failed\n", rank,
                f + 1);
        return 1;
      }
    }
    std::vector<void*> bufs(static_cast<size_t>(nflows));
    for (int f = 0; f < nflows; ++f) {
      GPU_RT_CHECK(gpuMalloc(&bufs[static_cast<size_t>(f)], msg_size));
      GPU_RT_CHECK(gpuMemset(bufs[static_cast<size_t>(f)], 0, msg_size));
      uint32_t bid = 100 + static_cast<uint32_t>(f);
      if (!comm->reg_mr(bid, bufs[static_cast<size_t>(f)], msg_size, false)) {
        fprintf(stderr, "[r%d] reg_mr send failed\n", rank);
        return 1;
      }
    }

    std::vector<double> times;
    size_t const total_msgs =
        static_cast<size_t>(nflows) * static_cast<size_t>(msgs_per_flow);
    auto round = [&](uint64_t tag_base) {
      std::vector<unsigned> inflight;
      size_t posted = 0;
      auto post_one = [&](size_t idx) -> bool {
        int peer = static_cast<int>(idx % static_cast<size_t>(nflows)) + 1;
        unsigned send_bid = 100 + static_cast<uint32_t>(idx % static_cast<size_t>(nflows));
        unsigned rid = comm->alloc_rid();
        bool ok = comm->send_put_signal_async_with_rid(
            peer, send_bid, 0, 1, 0, msg_size, PeerTransportKind::Rdma,
            tag_base + static_cast<uint64_t>(idx), rid);
        if (!ok) {
          fprintf(stderr, "[r%d] put peer %d failed\n", rank, peer);
          return false;
        }
        inflight.push_back(rid);
        return true;
      };
      size_t const cap =
          (window == 0) ? total_msgs : std::min(window, total_msgs);
      while (posted < total_msgs) {
        while (inflight.size() < cap && posted < total_msgs) {
          if (!post_one(posted++)) return false;
        }
        if (!wait_any(*comm, inflight)) return false;
      }
      if (!wait_all(*comm, inflight)) return false;
      if (echo) {
        // Wait for every receiver's echo put (same idx->tag mapping as
        // the outbound messages, offset by 0x4000).
        std::vector<unsigned> echo_reqs;
        for (size_t idx = 0; idx < total_msgs; ++idx) {
          unsigned rid = comm->alloc_rid();
          if (!comm->wait_signal_async_with_rid(
                  static_cast<int>(idx % static_cast<size_t>(nflows)) + 1,
                  tag_base + static_cast<uint64_t>(idx) + 0x4000ull,
                  PeerTransportKind::Rdma, rid, 1, true))
            return false;
          echo_reqs.push_back(rid);
        }
        if (!wait_all(*comm, echo_reqs)) return false;
      }
      return true;
    };

    for (int i = 0; i < warmup; ++i)
      if (!round(0x1000ull + static_cast<uint64_t>(i) * 1000)) return 1;
    for (int i = 0; i < iters; ++i) {
      double t0 = now_s();
      if (!round(0x2000ull + static_cast<uint64_t>(i) * 1000)) return 1;
      times.push_back(now_s() - t0);
    }
    double avg = 0;
    for (double t : times) avg += t;
    avg /= times.size();
    double total_moved =
        static_cast<double>(total_msgs) * msg_size * 2;  // send+recv
    fprintf(stderr,
            "[sender] flows=%d msgs/flow=%d msg=%.1fMB win=%zu iters=%zu "
            "avg=%.1fus aggregate=%.2fGB/s\n",
            nflows, msgs_per_flow, msg_size / 1e6, window, times.size(),
            avg * 1e6,
            total_moved / avg / 1e9);
  } else {
    uint32_t bid = 1;
    void* recv_buf = nullptr;
    GPU_RT_CHECK(gpuMalloc(&recv_buf, msg_size));
    if (!comm->reg_mr(bid, recv_buf, msg_size, true)) {
      fprintf(stderr, "[r%d] reg_mr recv failed\n", rank);
      return 1;
    }
    // Echo: register a send buffer back to rank 0 and resolve rank 0's
    // recv buffer (bid 1 on rank 0 as well).
    void* echo_buf = nullptr;
    if (echo) {
      GPU_RT_CHECK(gpuMalloc(&echo_buf, msg_size));
      if (!comm->reg_mr(100 + static_cast<uint32_t>(rank - 1), echo_buf,
                        msg_size, false)) {
        fprintf(stderr, "[r%d] reg_mr echo failed\n", rank);
        return 1;
      }
      if (!comm->resolve_remote_buffer(0, 1, 30000)) {
        fprintf(stderr, "[r%d] resolve echo dst failed\n", rank);
        return 1;
      }
    }
    size_t const total_msgs =
        static_cast<size_t>(msgs_per_flow);
    for (int i = 0; i < warmup + iters; ++i) {
      std::vector<unsigned> recv_reqs;
      for (size_t k = 0; k < total_msgs; ++k) {
        uint64_t tag =
            ((i < warmup) ? 0x1000ull : 0x2000ull) +
            static_cast<uint64_t>(std::max(0, i - warmup)) * 1000 +
            static_cast<uint64_t>(rank - 1) * total_msgs +
            static_cast<uint64_t>(k);
        unsigned rid = comm->wait_signal_async_with_rid(
            0, tag, PeerTransportKind::Rdma, comm->alloc_rid(), 1, true);
        if (rid == 0) {
          fprintf(stderr, "[r%d] wait failed\n", rank);
          return 1;
        }
        recv_reqs.push_back(rid);
      }
      if (!wait_all(*comm, recv_reqs)) {
        fprintf(stderr, "[r%d] recv wait_all failed\n", rank);
        return 1;
      }
      if (echo) {
        for (size_t k = 0; k < total_msgs; ++k) {
          unsigned rid = comm->alloc_rid();
          if (!comm->send_put_signal_async_with_rid(
                  0, 100 + static_cast<uint32_t>(rank - 1), 0, 1, 0, msg_size,
                  PeerTransportKind::Rdma,
                  ((i < warmup) ? 0x1000ull : 0x2000ull) +
                      static_cast<uint64_t>(std::max(0, i - warmup)) * 1000 +
                      static_cast<uint64_t>(rank - 1) * total_msgs +
                      static_cast<uint64_t>(k) + 0x4000ull,
                  rid)) {
            fprintf(stderr, "[r%d] echo put failed\n", rank);
            return 1;
          }
          if (!wait_one(*comm, rid)) {
            fprintf(stderr, "[r%d] echo wait failed\n", rank);
            return 1;
          }
        }
      }
    }
    fprintf(stderr, "[receiver r%d] done\n", rank);
  }
  return 0;
}

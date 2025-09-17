#include "proxy.hpp"
#include "ep_util.hpp"
#include <arpa/inet.h>  // for htonl, ntohl
#include <chrono>
#include <thread>

double Proxy::avg_rdma_write_us() const {
  if (kIterations == 0) return 0.0;
  return total_rdma_write_durations_.count() / static_cast<double>(kIterations);
}

double Proxy::avg_wr_latency_us() const {
  if (completion_count_ == 0) return 0.0;
  return static_cast<double>(wr_time_total_us_) /
         static_cast<double>(completion_count_);
}

uint64_t Proxy::completed_wr() const { return completion_count_; }
void Proxy::pin_thread() {
  if (cfg_.pin_thread) {
    // TODO(MaoZiming): improves pinning.
    pin_thread_to_cpu(cfg_.block_idx + 1);
    int cpu = sched_getcpu();
    if (cpu == -1) {
      perror("sched_getcpu");
    } else {
      printf("Local CPU thread pinned to core %d\n", cpu);
    }
  }
}

void Proxy::set_peers_meta(std::vector<PeerMeta> const& peers) {
  peers_.clear();
  peers_.reserve(peers.size());
  for (auto const& p : peers) {
    peers_.push_back(p);
    if (cfg_.block_idx == 0)
      printf("PeerMeta(rank=%d, ptr=0x%lx, nbytes=%zu, ip=%s)\n", p.rank,
             static_cast<unsigned long>(p.ptr), p.nbytes, p.ip.c_str());
  }
  ctxs_for_all_ranks_.clear();
  ctxs_for_all_ranks_.resize(peers.size());
  for (size_t i = 0; i < peers.size(); ++i) {
    ctxs_for_all_ranks_[i] = std::make_unique<ProxyCtx>();
  }
}

static inline int pair_tid_block(int a, int b, int N, int block_idx) {
  int lo = std::min(a, b), hi = std::max(a, b);
  return block_idx * (N * N) + lo * N + hi;
}

void Proxy::init_common() {
  int const my_rank = cfg_.rank;

  per_thread_rdma_init(ctx_, cfg_.gpu_buffer, cfg_.total_size, my_rank,
                       cfg_.block_idx, cfg_.local_rank);
  pin_thread();
  if (!ctx_.cq) ctx_.cq = create_per_thread_cq(ctx_);
  if (ctxs_for_all_ranks_.empty()) {
    fprintf(stderr,
            "Error: peers metadata not set before init_common (peers_.size() "
            "=%zu)\n",
            peers_.size());
    std::abort();
  }

  // Allocate GPU buffer for atomic old values (within the main GPU buffer)
  // Use a small section at the end of the GPU buffer
  size_t atomic_buf_size = ProxyCtx::kMaxAtomicOps * sizeof(uint32_t);
  if (cfg_.total_size < atomic_buf_size) {
    fprintf(stderr, "GPU buffer too small for atomic operations buffer\n");
    std::abort();
  }
  // Place atomic buffer at the end of the GPU buffer
  ctx_.atomic_old_values_buf =
      reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(cfg_.gpu_buffer) +
                                  cfg_.total_size - atomic_buf_size);

  // printf("[PROXY_INIT] Atomic buffer at %p, size %zu bytes\n",
  //        ctx_.atomic_old_values_buf, atomic_buf_size);

  int num_ranks = ctxs_for_all_ranks_.size();
  local_infos_.assign(num_ranks, RDMAConnectionInfo{});
  remote_infos_.assign(num_ranks, RDMAConnectionInfo{});

  uint32_t next_tag = 1;
  ctx_by_tag_.clear();
  ctx_by_tag_.resize(ctxs_for_all_ranks_.size() + 1, nullptr);
  // Per peer QP initialization
  for (int peer = 0; peer < num_ranks; ++peer) {
    // if (peer == my_rank) continue;
    // // Skip rdma connection for intra-node.
    // if (peers_[peer].ip == peers_[my_rank].ip) continue;
    auto& c = *ctxs_for_all_ranks_[peer];

    c.tag = next_tag++;
    if (c.tag >= ctx_by_tag_.size()) ctx_by_tag_.resize(c.tag + 1, nullptr);
    ctx_by_tag_[c.tag] = &c;

    c.context = ctx_.context;
    c.pd = ctx_.pd;
    c.mr = ctx_.mr;
    c.rkey = ctx_.rkey;
    // NOTE(MaoZiming): each context can share the same cq, pd, mr.
    // but the qp must be different.
    c.cq = ctx_.cq;
    create_per_thread_qp(c, cfg_.gpu_buffer, cfg_.total_size,
                         &local_infos_[peer], my_rank);
    modify_qp_to_init(c);
  }

  usleep(50 * 1000);

  // Out-of-band exchange per pair.
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) continue;
    // Skip rdma connection for intra-node.
    if (peers_[peer].ip == peers_[my_rank].ip) continue;

    bool const i_listen = (my_rank < peer);
    int const tid = pair_tid_block(my_rank, peer, num_ranks, cfg_.block_idx);
    char const* ip = peers_[peer].ip.c_str();

    int virt_rank = i_listen ? 0 : 1;
    if (peers_[cfg_.rank].ptr != local_infos_[my_rank].addr) {
      fprintf(stderr,
              "Rank %d block %d: Warning: local addr mismatch: got 0x%lx, "
              "expected 0x%lx\n",
              my_rank, cfg_.block_idx, local_infos_[my_rank].addr,
              peers_[cfg_.rank].ptr);
      std::abort();
    }
    exchange_connection_info(virt_rank, ip, tid, &local_infos_[peer],
                             &remote_infos_[peer]);
    if (remote_infos_[peer].addr != peers_[peer].ptr) {
      fprintf(stderr,
              "Rank %d block %d: Warning: remote addr mismatch for peer %d: "
              "got 0x%lx, expected 0x%lx\n",
              my_rank, cfg_.block_idx, peer, remote_infos_[peer].addr,
              peers_[peer].ptr);
      std::abort();
    }
  }

  // Bring each per-peer QP to RTR/RTS
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) continue;
    // Skip rdma connection for intra-node.
    if (peers_[peer].ip == peers_[my_rank].ip) continue;
    auto& c = *ctxs_for_all_ranks_[peer];

    // qp is different from each rank.
    modify_qp_to_rtr(c, &remote_infos_[peer]);
    modify_qp_to_rts(c, &local_infos_[peer]);

    c.remote_addr = remote_infos_[peer].addr;
    c.remote_rkey = remote_infos_[peer].rkey;
    c.remote_len = remote_infos_[peer].len;

    printf("Peer %d remote addr=%p rkey=%u len=%lu\n", peer,
           (void*)c.remote_addr, c.remote_rkey, c.remote_len);

    if (FILE* f = fopen("/tmp/uccl_debug.txt", "a")) {
      fprintf(
          f,
          "[PROXY_INIT] me=%d peer=%d: remote_addr=0x%lx local_buffer=0x%lx\n",
          my_rank, peer, c.remote_addr, (uintptr_t)cfg_.gpu_buffer);
      fclose(f);
    }
  }

  usleep(50 * 1000);
}

void Proxy::init_sender() {
  init_common();
  assert(cfg_.rank == 0);
  auto& ctx_ptr = ctxs_for_all_ranks_[1];
  local_post_ack_buf(*ctx_ptr, kSenderAckQueueDepth);
}

void Proxy::init_remote() {
  init_common();
  assert(cfg_.rank == 1);
  auto& ctx_ptr = ctxs_for_all_ranks_[0];
  remote_reg_ack_buf(ctx_ptr->pd, ring.ack_buf, ring.ack_mr);
  ring.ack_qp = ctx_ptr->ack_qp;
  post_receive_buffer_for_imm(*ctx_ptr);
}

void Proxy::run_sender() {
  printf("CPU sender thread for block %d started\n", cfg_.block_idx + 1);
  init_sender();
  size_t seen = 0;
  uint64_t my_tail = 0;
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    local_poll_completions(ctx_, finished_wrs_, acked_wrs_, finished_wrs_mutex_,
                           cfg_.block_idx, ctx_by_tag_);
    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);
  }
}

void Proxy::run_remote() {
  printf("Remote CPU thread for block %d started\n", cfg_.block_idx + 1);
  init_remote();
  printf("Finished\n");
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    remote_poll_completions(ctx_, cfg_.block_idx, ring, ctx_by_tag_,
                            atomic_buffer_ptr_);
  }
}

void Proxy::run_dual() {
  printf("Dual (single-thread) proxy for block %d starting\n",
         cfg_.block_idx + 1);
  init_common();
  for (int peer = 0; peer < (int)ctxs_for_all_ranks_.size(); ++peer) {
    if (peer == cfg_.rank) continue;
    if (peers_[peer].ip == peers_[cfg_.rank].ip) continue;
    auto& ctx_ptr = ctxs_for_all_ranks_[peer];
    if (!ctx_ptr) continue;
    local_post_ack_buf(*ctx_ptr, kSenderAckQueueDepth);
    remote_reg_ack_buf(ctx_ptr->pd, ring.ack_buf, ring.ack_mr);
    ring.ack_qp = ctx_ptr->ack_qp;
    post_receive_buffer_for_imm(*ctx_ptr);
  }
  uint64_t my_tail = 0;
  size_t seen = 0;
  // printf("run_dual initialization complete\n");
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    poll_cq_dual(ctx_, finished_wrs_, acked_wrs_, finished_wrs_mutex_,
                 cfg_.block_idx, ring, ctx_by_tag_, atomic_buffer_ptr_);
    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);
  }
}

void Proxy::notify_gpu_completion(uint64_t& my_tail) {
  if (finished_wrs_.empty()) return;
  if (acked_wrs_.empty()) return;

  std::lock_guard<std::mutex> lock(finished_wrs_mutex_);
  int check_i = 0;
  int actually_completed = 0;

  // Copy to iterate safely while erasing.
  std::vector<uint64_t> finished_copy(finished_wrs_.begin(),
                                      finished_wrs_.end());
  std::sort(finished_copy.begin(), finished_copy.end());
  for (auto wr_id : finished_copy) {
    if (acked_wrs_.find(wr_id) == acked_wrs_.end()) break;
    finished_wrs_.erase(wr_id);
    acked_wrs_.erase(wr_id);
    // Clear ring entry (contiguity assumed)
    cfg_.rb->volatile_store_cmd(my_tail + check_i, 0);
    check_i++;

    auto it = wr_id_to_start_time_.find(wr_id);
    if (it == wr_id_to_start_time_.end()) {
      fprintf(stderr, "Error: WR ID %lu not found in wr_id_to_start_time\n",
              wr_id);
      std::abort();
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - it->second);
    wr_time_total_us_ += duration.count();
    completion_count_++;
    actually_completed++;
  }
  if (!actually_completed) return;
  my_tail += actually_completed;
  cfg_.rb->cpu_volatile_store_tail(my_tail);
}

void Proxy::post_gpu_command(uint64_t& my_tail, size_t& seen) {
  // Force load head from DRAM
  uint64_t cur_head = cfg_.rb->volatile_head();
  if (cur_head == my_tail) {
    cpu_relax();
    return;
  }
  size_t batch_size = cur_head - seen;
  if (batch_size == 0) return;

  std::vector<uint64_t> wrs_to_post;
  wrs_to_post.reserve(batch_size);
  std::vector<TransferCmd> cmds_to_post;
  cmds_to_post.reserve(batch_size);

  for (size_t i = seen; i < cur_head; ++i) {
    uint64_t cmd = cfg_.rb->volatile_load_cmd(i);
    // NOTE(MaoZiming): Non-blocking. prevent local and remote both while loop.
    if (cmd == 0) break;

    TransferCmd& cmd_entry = cfg_.rb->load_cmd_entry(i);
    wrs_to_post.push_back(i);
    cmds_to_post.push_back(cmd_entry);
    wr_id_to_start_time_[i] = std::chrono::high_resolution_clock::now();
    seen = i + 1;
  }
  if (!wrs_to_post.empty()) {
    auto start = std::chrono::high_resolution_clock::now();
    post_gpu_commands_mixed(wrs_to_post, cmds_to_post);
    auto end = std::chrono::high_resolution_clock::now();
    total_rdma_write_durations_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  }
}

void Proxy::run_local() {
  pin_thread();
  uint64_t my_tail = 0;
  printf("Local CPU thread for block %d started\n", cfg_.block_idx + 1);
  // for (int seen = 0; seen < kIterations; ++seen) {
  int seen = 0;
  while (true) {
    if (!ctx_.progress_run.load(std::memory_order_acquire)) {
      printf("Local block %d stopping early at seen=%d\n", cfg_.block_idx + 1,
             seen);
      return;
    }
    // Prefer volatile read to defeat CPU cache stale head.
    // If your ring already has volatile_head(), use it; otherwise keep
    // rb->head.

    while (cfg_.rb->volatile_head() == my_tail) {
#ifdef DEBUG_PRINT
      if (cfg_.block_idx == 0) {
        printf("Local block %d waiting: tail=%lu head=%lu\n", cfg_.block_idx,
               my_tail, cfg_.rb->head);
      }
#endif
      cpu_relax();
      if (!ctx_.progress_run.load(std::memory_order_acquire)) {
        printf("Local block %d stopping early at seen=%d\n", cfg_.block_idx + 1,
               seen);
        return;
      }
    }

    uint64_t const idx = my_tail & kQueueMask;
    uint64_t cmd;
    auto last_print = std::chrono::steady_clock::now();
    size_t spin_count = 0;
    do {
      cmd = cfg_.rb->volatile_load_cmd(idx);
      cpu_relax();

      auto now = std::chrono::steady_clock::now();
      if (now - last_print > std::chrono::seconds(10)) {
        printf(
            "Still waiting at block %d, seen=%d, spin_count=%zu, my_tail=%lu, "
            "cmd: %lu\n",
            cfg_.block_idx + 1, seen, spin_count, my_tail, cmd);
        last_print = now;
        spin_count++;
      }

      if (!ctx_.progress_run.load(std::memory_order_acquire)) {
        printf("Local block %d stopping early at seen=%d\n", cfg_.block_idx + 1,
               seen);
        return;
      }
    } while (cmd == 0);

#ifdef DEBUG_PRINT
    printf("Local block %d, seen=%d head=%lu tail=%lu consuming cmd=%llu\n",
           cfg_.block_idx, seen, cfg_.rb->head, my_tail,
           static_cast<unsigned long long>(cmd));
#endif

    /*
        const uint64_t expected_cmd =
            (static_cast<uint64_t>(cfg_.block_idx) << 32) | (seen + 1);
        if (cmd != expected_cmd) {
          fprintf(stderr, "Error[Local]: block %d expected %llu got %llu\n",
                  cfg_.block_idx, static_cast<unsigned long long>(expected_cmd),
                  static_cast<unsigned long long>(cmd));
          std::abort();
        }
    */
    std::atomic_thread_fence(std::memory_order_acquire);
    if (cmd == 1) {
      TransferCmd& cmd_entry = cfg_.rb->buf[idx];
      printf("Received command 1: block %d, seen=%d, value: %d\n",
             cfg_.block_idx + 1, seen, cmd_entry.value);
    }

    cfg_.rb->volatile_store_cmd(idx, 0);
    ++my_tail;
    // cfg_.rb->tail = my_tail;
    cfg_.rb->cpu_volatile_store_tail(my_tail);
    seen++;
  }

  printf("Local block %d finished %d commands, tail=%lu\n", cfg_.block_idx,
         kIterations, my_tail);
}

void Proxy::post_gpu_commands_mixed(
    std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post) {
  // Separate atomic operations from regular RDMA writes
  std::vector<uint64_t> rdma_wrs, atomic_wrs;
  std::vector<TransferCmd> rdma_cmds, atomic_cmds;

  for (size_t i = 0; i < cmds_to_post.size(); ++i) {
    if (cmds_to_post[i].is_atomic) {
      atomic_wrs.push_back(wrs_to_post[i]);
      atomic_cmds.push_back(cmds_to_post[i]);
    } else {
      rdma_wrs.push_back(wrs_to_post[i]);
      rdma_cmds.push_back(cmds_to_post[i]);
    }
  }
  // Handle regular RDMA writes
  if (!rdma_wrs.empty()) {
    post_rdma_async_batched(ctx_, cfg_.gpu_buffer, rdma_wrs.size(), rdma_wrs,
                            rdma_cmds, ctxs_for_all_ranks_, cfg_.rank,
                            cfg_.block_idx, finished_wrs_, finished_wrs_mutex_);
  }
  if (!atomic_wrs.empty()) {
    post_atomic_operations(ctx_, atomic_wrs, atomic_cmds, ctxs_for_all_ranks_,
                           cfg_.rank, cfg_.block_idx, finished_wrs_,
                           finished_wrs_mutex_, acked_wrs_);
  }
}

void Proxy::destroy(bool free_gpu_buffer) {
  ctx_.progress_run.store(false, std::memory_order_release);
  cudaDeviceSynchronize();

  for (auto& ctx_ptr : ctxs_for_all_ranks_) {
    if (!ctx_ptr) continue;
    qp_to_error(ctx_ptr->qp);
    qp_to_error(ctx_ptr->ack_qp);
  }

  drain_cq(ctx_.cq);

  for (auto& ctx_ptr : ctxs_for_all_ranks_) {
    if (!ctx_ptr) continue;
    if (ctx_ptr->qp) {
      ibv_destroy_qp(ctx_ptr->qp);
      ctx_ptr->qp = nullptr;
    }
    if (ctx_ptr->ack_qp) {
      ibv_destroy_qp(ctx_ptr->ack_qp);
      ctx_ptr->ack_qp = nullptr;
    }
  }
  ring.ack_qp = nullptr;

  drain_cq(ctx_.cq);

  if (ctx_.cq) {
    ibv_destroy_cq(ctx_.cq);
    ctx_.cq = nullptr;
  }

  auto dereg = [&](ibv_mr*& mr) {
    if (!mr) return;
    for (int i = 0; i < 5; ++i) {
      int ret = ibv_dereg_mr(mr);
      if (ret == 0) {
        mr = nullptr;
        return;
      }
      fprintf(stderr, "[destroy] ibv_dereg_mr ret=%d (%s), attempt=%d\n", ret,
              strerror(ret), i + 1);
      std::this_thread::sleep_for(std::chrono::milliseconds(2 << i));
    }
  };
  dereg(ring.ack_mr);
  dereg(ctx_.mr);

  if (free_gpu_buffer && cfg_.gpu_buffer) {
    cudaError_t e = cudaFree(cfg_.gpu_buffer);
    if (e != cudaSuccess)
      fprintf(stderr, "[destroy] cudaFree failed: %s\n", cudaGetErrorString(e));
    else
      cfg_.gpu_buffer = nullptr;
  }

  if (ctx_.pd) {
    ibv_dealloc_pd(ctx_.pd);
    ctx_.pd = nullptr;
  }
  if (ctx_.context) {
    ibv_close_device(ctx_.context);
    ctx_.context = nullptr;
  }

  finished_wrs_.clear();
  acked_wrs_.clear();
  wr_id_to_start_time_.clear();
  ctxs_for_all_ranks_.clear();
  ctx_by_tag_.clear();
  local_infos_.clear();
  remote_infos_.clear();
}
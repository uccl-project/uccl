#include "proxy.hpp"
#include "bench_utils.hpp"
#include "d2h_queue_host.hpp"
#include "ep_util.hpp"
#include <arpa/inet.h>  // for htonl, ntohl
#include <chrono>
#include <cstdlib>
#include <thread>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

double Proxy::avg_rdma_write_us() const {
  if (kIterations == 0) return 0.0;
  return total_rdma_write_durations_.count() / static_cast<double>(kIterations);
}

double Proxy::avg_wr_latency_us() const {
  if (completion_count_ <= kWarmupOps) return 0.0;
  return static_cast<double>(wr_time_total_us_) /
         static_cast<double>(completion_count_ - kWarmupOps);
}

uint64_t Proxy::completed_wr() const { return completion_count_; }

void Proxy::pin_thread_to_cpu_wrapper() {
  if (cfg_.pin_thread) {
    // TODO(MaoZiming): improves pinning.
    pin_thread_to_cpu(cfg_.thread_idx + cfg_.local_rank * kNumThBlocks);
    int cpu = sched_getcpu();
    if (cpu == -1) {
      perror("sched_getcpu");
    } else {
      printf(
          "Local CPU thread pinned to core %d, thread_idx: %d, "
          "local_rank: %d\n",
          cpu, cfg_.thread_idx, cfg_.local_rank);
    }
  }
}

void Proxy::pin_thread_to_numa_wrapper() {
  if (cfg_.pin_thread) {
    assert(ctx_.numa_node != -1);
    pin_thread_unique(ctx_.numa_node, cfg_.local_rank, cfg_.thread_idx,
                      kNumThBlocks);

    // Get the actual CPU this thread is running on
    int cpu = sched_getcpu();

    // Get the affinity mask (optional but useful)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    printf(
        "Local CPU thread pinned to NUMA node %d, thread_idx: %d, local_rank: "
        "%d, "
        "running on CPU %d.\n",
        ctx_.numa_node, cfg_.thread_idx, cfg_.local_rank, cpu);
  }
}

void Proxy::set_peers_meta(std::vector<PeerMeta> const& peers) {
  peers_.clear();
  peers_.reserve(peers.size());
  for (auto const& p : peers) {
    peers_.push_back(p);
  }
  ctxs_for_all_ranks_.clear();
  ctxs_for_all_ranks_.resize(peers.size());
  for (size_t i = 0; i < peers.size(); ++i) {
    ctxs_for_all_ranks_[i] = std::make_unique<ProxyCtx>();
  }
}

void Proxy::set_bench_d2h_channel_addrs(std::vector<uintptr_t> const& addrs) {
#ifndef USE_MSCCLPP_FIFO_BACKEND
  ring_tails_.clear();
  ring_seen_.clear();

  ring_tails_.resize(addrs.size(), 0);
  ring_seen_.resize(addrs.size(), 0);
#endif
  cfg_.d2h_queues.clear();
  cfg_.d2h_queues.reserve(addrs.size());

  for (auto addr : addrs) {
    d2hq::HostD2HHandle h{};
    d2hq::init_from_addr(h, addr);  // unified initialization
    cfg_.d2h_queues.push_back(h);
  }
}

static inline int pair_tid_block(int a, int b, int N, int thread_idx) {
  int lo = std::min(a, b), hi = std::max(a, b);
  return thread_idx * (N * N) + lo * N + hi;
}

void Proxy::init_common() {
  int const my_rank = cfg_.rank;

  per_thread_rdma_init(ctx_, cfg_.gpu_buffer, cfg_.total_size, my_rank,
                       cfg_.thread_idx, cfg_.local_rank);
  pin_thread_to_numa_wrapper();
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

    if (peer == my_rank) continue;
    // Skip rdma connection for intra-node.
    if (peers_[peer].ip == peers_[my_rank].ip) continue;
    if (cfg_.use_normal_mode && std::abs(peer - my_rank) % MAX_NUM_GPUS != 0)
      continue;
    create_per_thread_qp(c, cfg_.gpu_buffer, cfg_.total_size,
                         &local_infos_[peer], my_rank, cfg_.d2h_queues.size(),
                         cfg_.use_normal_mode);
    modify_qp_to_init(c);
  }

  usleep(50 * 1000);

  // Out-of-band exchange per pair.
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) continue;
    // Skip rdma connection for intra-node.
    if (peers_[peer].ip == peers_[my_rank].ip) continue;
    if (cfg_.use_normal_mode && std::abs(peer - my_rank) % MAX_NUM_GPUS != 0)
      continue;
    bool const i_listen = (my_rank < peer);
    int const tid = pair_tid_block(my_rank, peer, num_ranks, cfg_.thread_idx);
    char const* ip = peers_[peer].ip.c_str();

    int virt_rank = i_listen ? 0 : 1;
    exchange_connection_info(virt_rank, ip, tid, &local_infos_[peer],
                             &remote_infos_[peer]);
    if (remote_infos_[peer].addr != peers_[peer].ptr) {
      fprintf(stderr,
              "Rank %d thread %d: Warning: remote addr mismatch for peer %d: "
              "got 0x%lx, expected 0x%lx\n",
              my_rank, cfg_.thread_idx, peer, remote_infos_[peer].addr,
              peers_[peer].ptr);
      std::abort();
    }
  }

  // Bring each per-peer QP to RTR/RTS
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) continue;
    // Skip rdma connection for intra-node.
    if (peers_[peer].ip == peers_[my_rank].ip) continue;
    if (cfg_.use_normal_mode && std::abs(peer - my_rank) % MAX_NUM_GPUS != 0)
      continue;
    auto& c = *ctxs_for_all_ranks_[peer];

    // qp is different from each rank.
    modify_qp_to_rtr(c, &remote_infos_[peer], cfg_.use_normal_mode);
    modify_qp_to_rts(c, &local_infos_[peer]);

    c.remote_addr = remote_infos_[peer].addr;
    c.remote_rkey = remote_infos_[peer].rkey;
    c.remote_len = remote_infos_[peer].len;
    if (FILE* f = fopen("/tmp/uccl_debug.txt", "a")) {
      fprintf(
          f,
          "[PROXY_INIT] me=%d peer=%d: remote_addr=0x%lx local_buffer=0x%lx\n",
          my_rank, peer, c.remote_addr, (uintptr_t)cfg_.gpu_buffer);
      fclose(f);
    }
  }
  usleep(50 * 1000);
  if (cfg_.use_normal_mode) {
    // if (cfg_.thread_idx != 0) {
    //   return;
    // }
    // Discover local ranks (same IP as me)
    std::string const my_ip = peers_[cfg_.rank].ip;
    std::vector<int> local_ranks;
    local_ranks.reserve(ctxs_for_all_ranks_.size());
    int leader_rank = cfg_.rank;
    for (int r = 0; r < (int)peers_.size(); ++r) {
      if (peers_[r].ip == my_ip) {
        local_ranks.push_back(r);
        if (r < leader_rank) leader_rank = r;
      }
    }
    ctx_.num_local_ranks = (int)local_ranks.size();
    ctx_.node_leader_rank = leader_rank;
    ctx_.local_rank = cfg_.local_rank;
    ctx_.thread_idx = cfg_.thread_idx;

    if (ctx_.num_local_ranks > UCCL_MAX_LOCAL_RANKS) {
      fprintf(stderr, "num_local_ranks=%d exceeds UCCL_MAX_LOCAL_RANKS=%d\n",
              ctx_.num_local_ranks, (int)UCCL_MAX_LOCAL_RANKS);
      std::abort();
    }
  }

#ifdef USE_MSCCLPP_FIFO_BACKEND
  fifo_seq_.assign(cfg_.d2h_queues.size(), 0);
  fifo_pending_.assign(cfg_.d2h_queues.size(), std::deque<uint64_t>{});
#endif
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
  local_post_ack_buf(*ctx_ptr, kSenderAckQueueDepth);
  remote_reg_ack_buf(ctx_ptr->pd, ring.ack_buf, ring.ack_mr);
  ring.ack_qp = ctx_ptr->ack_qp;
#ifndef EFA
  post_receive_buffer_for_imm(*ctx_ptr);
#endif
}

void Proxy::run_sender() {
  printf("CPU sender thread %d started\n", cfg_.thread_idx);
  init_sender();
  size_t seen = 0;
  uint64_t my_tail = 0;
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    local_poll_completions(ctx_, acked_wrs_, cfg_.thread_idx, ctx_by_tag_);
    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);
  }
}

void Proxy::run_remote() {
  printf("Remote CPU thread %d started\n", cfg_.thread_idx);
  init_remote();
  std::set<PendingUpdate> pending_atomic_updates;
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    remote_poll_completions(ctx_, cfg_.thread_idx, ring, ctx_by_tag_,
                            atomic_buffer_ptr_, cfg_.num_ranks,
                            cfg_.num_experts, pending_atomic_updates, cfg_.rank,
                            cfg_.num_nodes, cfg_.use_normal_mode);
#ifdef USE_RECEIVER_BARRIER
    if (!cfg_.use_normal_mode) {
      apply_pending_updates(ctx_, pending_atomic_updates, atomic_buffer_ptr_,
                            cfg_.num_experts, cfg_.num_ranks);
    }
#endif
  }
}

void Proxy::run_dual() {
  init_common();
  for (int peer = 0; peer < (int)ctxs_for_all_ranks_.size(); ++peer) {
    if (peer == cfg_.rank) continue;
    if (peers_[peer].ip == peers_[cfg_.rank].ip) continue;
    if (cfg_.use_normal_mode && std::abs(peer - cfg_.rank) % MAX_NUM_GPUS != 0)
      continue;
    auto& ctx_ptr = ctxs_for_all_ranks_[peer];
    if (!ctx_ptr) continue;
    local_post_ack_buf(*ctx_ptr, kSenderAckQueueDepth);
    remote_reg_ack_buf(ctx_ptr->pd, ring.ack_buf, ring.ack_mr);
    ring.ack_qp = ctx_ptr->ack_qp;
#ifndef EFA
    post_receive_buffer_for_imm(*ctx_ptr);
#endif
  }
  uint64_t my_tail = 0;
  size_t seen = 0;
  std::set<PendingUpdate> pending_atomic_updates;
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    poll_cq_dual(ctx_, acked_wrs_, cfg_.thread_idx, ring, ctx_by_tag_,
                 atomic_buffer_ptr_, cfg_.num_ranks, cfg_.num_experts,
                 pending_atomic_updates, cfg_.rank, cfg_.num_nodes,
                 cfg_.use_normal_mode);
    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);
#ifdef USE_RECEIVER_BARRIER
    if (!cfg_.use_normal_mode) {
      apply_pending_updates(ctx_, pending_atomic_updates, atomic_buffer_ptr_,
                            cfg_.num_experts, cfg_.num_ranks);
    }
#endif

#ifdef USE_SENDER_BARRIER
    if (!cfg_.use_normal_mode) {
      auto postponed_wr_ids = postponed_wr_ids_;
      auto postponed_atomics = postponed_atomics_;
      postponed_wr_ids_.clear();
      postponed_atomics_.clear();
      assert(postponed_wr_ids.size() == postponed_atomics.size());
      assert(postponed_wr_ids_.size() == 0);
      post_gpu_commands_mixed(postponed_wr_ids, postponed_atomics);
    }
#endif

    if (cfg_.use_normal_mode) {
      barrier_check();
    }
  }
}

void Proxy::notify_gpu_completion(uint64_t& my_tail) {
  if (acked_wrs_.empty()) return;

    // Mark all acked command slots in each ring's bitmask
#ifdef USE_MSCCLPP_FIFO_BACKEND
  // FIFO path: pop in order using the pending deque and the completion set.
  for (size_t rb_idx = 0; rb_idx < cfg_.d2h_queues.size(); ++rb_idx) {
    auto* fifo = cfg_.d2h_queues[rb_idx].fifo;
    if (!fifo) continue;
    auto& pend = fifo_pending_[rb_idx];
    while (!pend.empty()) {
      uint64_t front_wr = pend.front();
      if (acked_wrs_.find(front_wr) == acked_wrs_.end()) break;
      acked_wrs_.erase(front_wr);  // consume this completion
      pend.pop_front();            // retire pending entry

      if (ctx_.quiet_wr != -1 && front_wr == (uint64_t)ctx_.quiet_wr) {
        ctx_.quiet_inflight = false;
        ctx_.quiet_wr = -1;
        fifo->pop();
      }

      if (ctx_.barrier_wr != -1 && front_wr == (uint64_t)ctx_.barrier_wr) {
        ctx_.barrier_inflight = false;
        ctx_.barrier_wr = -1;
        fifo->pop();
      }
    }
  }
#else
  for (auto wr_id : acked_wrs_) {
    size_t const rb_idx = (wr_id >> 32) & 0xFFFFFFFF;
    size_t const cmd_idx = wr_id & 0xFFFFFFFF;

    if (rb_idx >= cfg_.d2h_queues.size()) {
      fprintf(stderr, "Invalid rb_idx %zu in acked_wrs_\n", rb_idx);
      continue;
    }

    d2hq::HostD2HHandle* h = &cfg_.d2h_queues[rb_idx];
    h->volatile_clear_cmd_type(cmd_idx);
    h->mark_acked(cmd_idx % h->capacity());
  }
  acked_wrs_.clear();

  // Advance tails for each ring buffer based on contiguous acked bits
  for (size_t rb_idx = 0; rb_idx < cfg_.d2h_queues.size(); ++rb_idx) {
    d2hq::HostD2HHandle* h = &cfg_.d2h_queues[rb_idx];
    ring_tails_[rb_idx] = h->advance_tail_from_mask();
  }
#endif
}

void Proxy::post_gpu_command(uint64_t& my_tail, size_t& seen) {
  // Multi-ring buffer processing: collect commands from all ring buffers
  std::vector<uint64_t> wrs_to_post;
  std::vector<TransferCmd> cmds_to_post;
  bool found_work = false;
  // Process each ring buffer (similar to test_multi_ring_throughput.cu)
  for (size_t rb_idx = 0; rb_idx < cfg_.d2h_queues.size(); rb_idx++) {
    d2hq::HostD2HHandle* h = &cfg_.d2h_queues[rb_idx];
#ifdef USE_MSCCLPP_FIFO_BACKEND
    assert(h && "h is empty!\n");
    assert(h->fifo && "h->fifo is empty!\n");
    // FIFO path: one trigger == one command. Do NOT pop yet.
    auto* fifo = h->fifo;
    if (!fifo) continue;
    // Available budget for this FIFO.
    size_t pending = fifo_pending_[rb_idx].size();
    size_t kMaxInflight =
        cfg_.use_normal_mode ? kMaxInflightNormal : kMaxInflightLowLatency;
    size_t budget = (kMaxInflight > pending) ? (kMaxInflight - pending) : 0;
    for (size_t take = 0; take < budget; ++take) {
      auto trig = fifo->poll();
      if (trig.fst == 0) break;
      TransferCmd cmd = d2hq::decode_from_trigger(trig);

      /* For some reason, this is important for correctness */
      /* It cannot be if (ctx_.barrier_inflight) */
      if (get_base_cmd(cmd.cmd_type) == CmdType::BARRIER &&
          ctx_.barrier_inflight) {
        break;
      }

      if (get_base_cmd(cmd.cmd_type) == CmdType::QUIET && ctx_.quiet_inflight) {
        break;
      }

      uint64_t unique_wr_id = (static_cast<uint64_t>(rb_idx) << 32) |
                              (fifo_seq_[rb_idx]++ & 0xFFFFFFFFULL);
      wrs_to_post.push_back(unique_wr_id);
      cmds_to_post.push_back(cmd);
      fifo_pending_[rb_idx].push_back(unique_wr_id);
      found_work = true;

      if (get_base_cmd(cmd.cmd_type) == CmdType::BARRIER ||
          get_base_cmd(cmd.cmd_type) == CmdType::QUIET) {
        if (get_base_cmd(cmd.cmd_type) == CmdType::BARRIER) {
          assert(!ctx_.barrier_inflight);
          assert(ctx_.barrier_wr == -1);
          ctx_.barrier_inflight = true;
        } else if (get_base_cmd(cmd.cmd_type) == CmdType::QUIET) {
          assert(!ctx_.quiet_inflight);
          assert(ctx_.quiet_wr == -1);
          ctx_.quiet_inflight = true;
        }
        break;
      } else {
        // Other operations just pop.
        fifo->pop();
      }
    }
#else
    uint64_t& ring_tail = ring_tails_[rb_idx];
    size_t& ring_seen = ring_seen_[rb_idx];

    // Force load head from DRAM (like original code)
    uint64_t cur_head = h->volatile_head();
    if (cur_head == ring_tail) {
      continue;  // No new work in this ring
    }

    // Batch processing for this ring buffer
    size_t batch_size = cur_head - ring_seen;
    if (batch_size == 0) continue;

    // Reserve space for new commands
    size_t current_size = wrs_to_post.size();
    wrs_to_post.reserve(current_size + batch_size);
    cmds_to_post.reserve(current_size + batch_size);

    // Collect batch of commands from this ring buffer
    for (size_t i = ring_seen; i < cur_head; ++i) {
      CmdType cmd = h->volatile_load_cmd_type(i);
      // NOTE(MaoZiming): Non-blocking. prevent local and remote both while
      // loop.
      if (cmd == CmdType::EMPTY) break;

      TransferCmd& cmd_entry = h->load_cmd_entry(i);
      if (cmd_entry.cmd_type == CmdType::EMPTY) break;

      if (get_base_cmd(cmd_entry.cmd_type) == CmdType::WRITE ||
          get_base_cmd(cmd_entry.cmd_type) == CmdType::ATOMIC) {
        if (static_cast<int>(cmd_entry.dst_rank) == cfg_.rank) {
          fprintf(stderr,
                  "[ERROR] Local command!, cmd.dst_rank: %d, cfg_.rank: %d, "
                  "cmd_entry.cmd_type: %d, cur_head: %lu, i: %lu\n",
                  cmd_entry.dst_rank, cfg_.rank,
                  static_cast<int>(cmd_entry.cmd_type), cur_head, i);
          cudaDeviceSynchronize();
          std::abort();
        }
        if (peers_[cmd_entry.dst_rank].ip == peers_[cfg_.rank].ip) {
          fprintf(
              stderr,
              "[ERROR] Intra-node command!, cmd.dst_rank: %d, cfg_.rank: %d, "
              "cmd_entry.cmd_type: %d, cur_head: %lu, i: %lu\n",
              cmd_entry.dst_rank, cfg_.rank,
              static_cast<int>(cmd_entry.cmd_type), cur_head, i);
          cudaDeviceSynchronize();
          std::abort();
        }
      }

      // Use a unique ID combining ring buffer index and command index
      uint64_t unique_wr_id = (rb_idx << 32) | i;
      wrs_to_post.push_back(unique_wr_id);
      cmds_to_post.push_back(cmd_entry);
#ifdef MEASURE_PER_VERB_LATENCY
      wr_id_to_start_time_[unique_wr_id] =
          std::chrono::high_resolution_clock::now();
#endif
      ring_seen = i + 1;
      found_work = true;
    }
#endif
  }

  // If no work found across all ring buffers, relax CPU
  if (!found_work) {
    cpu_relax();
    return;
  }

  // Process all collected commands in batch
  if (!wrs_to_post.empty()) {
    auto start = std::chrono::high_resolution_clock::now();
    post_gpu_commands_mixed(wrs_to_post, cmds_to_post);
    auto end = std::chrono::high_resolution_clock::now();
    total_rdma_write_durations_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  }
}

void Proxy::run_local() {
  pin_thread_to_cpu_wrapper();
  printf("Local CPU thread %d started with %zu ring buffers\n", cfg_.thread_idx,
         cfg_.d2h_queues.size());

  if (cfg_.d2h_queues.empty()) {
    printf("Error: No ring buffers available for local mode\n");
    return;
  }

  int total_seen = 0;
  while (true) {
    if (!ctx_.progress_run.load(std::memory_order_acquire)) {
      printf("Local thread %d stopping early at total_seen=%d\n",
             cfg_.thread_idx, total_seen);
      return;
    }

    bool found_work = false;

    // Multi-ring buffer polling (consistent with other modes)
    for (size_t rb_idx = 0; rb_idx < cfg_.d2h_queues.size(); rb_idx++) {
      d2hq::HostD2HHandle* h = &cfg_.d2h_queues[rb_idx];
#ifdef USE_MSCCLPP_FIFO_BACKEND
      auto* fifo = h->fifo;
      if (!fifo) continue;
      auto trig = fifo->poll();
      if (trig.fst != 0) {
        d2hq::decode_from_trigger(trig);
        fifo->pop();
        total_seen++;
        found_work = true;
      }
#else
      uint64_t& ring_tail = ring_tails_[rb_idx];

      // Check for new work in this ring buffer
      uint64_t cur_head = h->volatile_head();
      if (cur_head == ring_tail) {
        continue;  // No new work in this ring
      }

      // Process commands from this ring buffer
      while (ring_tail < cur_head) {
        uint64_t const idx = ring_tail & kQueueMask;
        CmdType cmd;
        auto last_print = std::chrono::steady_clock::now();
        size_t spin_count = 0;
        do {
          cmd = h->volatile_load_cmd_type(idx);
          cpu_relax();

          auto now = std::chrono::steady_clock::now();
          if (now - last_print > std::chrono::seconds(10)) {
            printf(
                "Still waiting at thread %d, ring %zu, total_seen=%d, "
                "spin_count=%zu, ring_tail=%lu, cmd: %d\n",
                cfg_.thread_idx, rb_idx, total_seen, spin_count, ring_tail,
                static_cast<int>(cmd));
            last_print = now;
            spin_count++;
          }

          if (!ctx_.progress_run.load(std::memory_order_acquire)) {
            printf("Local thread %d stopping early at total_seen=%d\n",
                   cfg_.thread_idx, total_seen);
            return;
          }
        } while (cmd == CmdType::EMPTY);

#ifdef DEBUG_PRINT
        printf(
            "Local thread %d, ring %zu, total_seen=%d head=%lu tail=%lu "
            "consuming cmd=%llu\n",
            cfg_.thread_idx, rb_idx, total_seen, h->head, ring_tail,
            static_cast<unsigned long long>(cmd));
#endif

        std::atomic_thread_fence(std::memory_order_acquire);

        // Mark command as processed
        h->volatile_clear_cmd_type(idx);
        ring_tail++;
        h->cpu_volatile_store_tail(ring_tail);
        total_seen++;
        found_work = true;

        // Break to check other ring buffers and progress_run flag
        break;
      }
#endif
    }

    // If no work found across all ring buffers, relax CPU
    if (!found_work) {
      cpu_relax();
    }
  }

  printf("Local thread %d finished %d commands across %zu ring buffers\n",
         cfg_.thread_idx, total_seen, cfg_.d2h_queues.size());
}

void Proxy::post_gpu_commands_mixed(
    std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post) {
  // Separate atomic operations from regular RDMA writes
  std::vector<uint64_t> rdma_wrs, atomic_wrs;
  std::vector<TransferCmd> rdma_cmds, atomic_cmds;

  std::vector<uint64_t> quiet_wrs, barrier_wrs;
  std::vector<TransferCmd> quiet_cmds, barrier_cmds;

  for (size_t i = 0; i < cmds_to_post.size(); ++i) {
    if (get_base_cmd(cmds_to_post[i].cmd_type) == CmdType::ATOMIC) {
#ifdef USE_SENDER_BARRIER
      if (!cfg_.use_normal_mode) {
        int value = cmds_to_post[i].value;
        uint32_t offset = static_cast<int64_t>(cmds_to_post[i].req_rptr);
        uint32_t new_offset =
            offset - cmds_to_post[i].low_latency_buffer_idx *
                         align<size_t>(cfg_.num_experts * sizeof(int), 128);
        size_t new_index = new_offset / sizeof(int);
        int expected_value;
        int expert_idx;

        if (cmds_to_post[i].is_combine) {
          expert_idx = new_index;
          expected_value = ctx_.combine_sent_counter.Get(
              {cmds_to_post[i].low_latency_buffer_idx, expert_idx,
               cmds_to_post[i].dst_rank});
        } else {
          expert_idx = new_index / cfg_.num_ranks;
          expected_value = ctx_.dispatch_sent_counter.Get(
              {cmds_to_post[i].low_latency_buffer_idx, expert_idx,
               cmds_to_post[i].dst_rank});
          value = -value - 1;
        }
        if (value != expected_value) {
          postponed_atomics_.push_back(cmds_to_post[i]);
          postponed_wr_ids_.push_back(wrs_to_post[i]);
          assert(postponed_atomics_.size() == postponed_wr_ids_.size());
          continue;
        }
      }
#endif
      atomic_wrs.push_back(wrs_to_post[i]);
      atomic_cmds.push_back(cmds_to_post[i]);

#ifdef USE_SENDER_BARRIER
      if (!cfg_.use_normal_mode) {
        uint32_t offset = static_cast<int64_t>(cmds_to_post[i].req_rptr);
        uint32_t new_offset =
            offset - cmds_to_post[i].low_latency_buffer_idx *
                         align<size_t>(cfg_.num_experts * sizeof(int), 128);
        size_t new_index = new_offset / sizeof(int);
        int expert_idx;
        if (cmds_to_post[i].is_combine) {
          expert_idx = new_index;
          ctx_.combine_sent_counter.Reset(
              {cmds_to_post[i].low_latency_buffer_idx, expert_idx,
               cmds_to_post[i].dst_rank});
        } else {
          expert_idx = new_index / cfg_.num_ranks;
          ctx_.dispatch_sent_counter.Reset(
              {cmds_to_post[i].low_latency_buffer_idx, expert_idx,
               cmds_to_post[i].dst_rank});
        }
      }
#endif
    } else if (get_base_cmd(cmds_to_post[i].cmd_type) == CmdType::WRITE) {
      rdma_wrs.push_back(wrs_to_post[i]);
      rdma_cmds.push_back(cmds_to_post[i]);
    } else if (get_base_cmd(cmds_to_post[i].cmd_type) == CmdType::QUIET) {
      quiet_cmds.push_back(cmds_to_post[i]);
      quiet_wrs.push_back(wrs_to_post[i]);
    } else if (get_base_cmd(cmds_to_post[i].cmd_type) == CmdType::BARRIER) {
      barrier_cmds.push_back(cmds_to_post[i]);
      barrier_wrs.push_back(wrs_to_post[i]);
    } else {
      fprintf(stderr, "Error: Unknown command type %d\n",
              static_cast<int>(cmds_to_post[i].cmd_type));
      std::abort();
    }
  }
  if (rdma_wrs.size() + atomic_wrs.size() + barrier_cmds.size() +
          quiet_cmds.size() ==
      0) {
    return;
  }
  // Handle regular RDMA writes
  if (!rdma_wrs.empty()) {
    post_rdma_async_batched(ctx_, cfg_.gpu_buffer, rdma_wrs.size(), rdma_wrs,
                            rdma_cmds, ctxs_for_all_ranks_, cfg_.rank,
                            cfg_.thread_idx, cfg_.use_normal_mode);
  }
  if (!atomic_wrs.empty()) {
    post_atomic_operations(ctx_, atomic_wrs, atomic_cmds, ctxs_for_all_ranks_,
                           cfg_.rank, cfg_.thread_idx, acked_wrs_,
                           cfg_.use_normal_mode);
  }
  if (!barrier_cmds.empty()) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    assert(barrier_wrs.size() == 1);
    assert(ctx_.barrier_wr == -1);
#endif
    send_barrier(barrier_wrs[0]);
  }
  if (!quiet_cmds.empty()) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    assert(quiet_wrs.size() == 1);
    assert(ctx_.quiet_wr == -1);
#endif
    ctx_.quiet_wr = quiet_wrs[0];
    quiet(quiet_wrs, quiet_cmds);
  }
}

void Proxy::quiet_cq() {
  auto outstanding_batches = [&]() -> size_t {
    size_t sum = 0;
    for (auto& ctx_ptr : ctxs_for_all_ranks_) {
      if (!ctx_ptr) continue;
      sum += ctx_ptr->wr_id_to_wr_ids.size();
    }
    return sum;
  };
  constexpr int kConsecutiveEmptyToExit = 3;
  int empty_iters = 0;
  ibv_wc wc[kMaxOutstandingSends];
  using clock = std::chrono::steady_clock;
  auto last_log = clock::now();
  std::set<PendingUpdate> pending_atomic_updates;
  for (;;) {
    int ne = poll_cq_once(ctx_.cq, wc, kMaxOutstandingSends);
    if (ne > 0) {
      empty_iters = 0;
      local_process_completions(ctx_, acked_wrs_, cfg_.thread_idx, wc, ne,
                                ctx_by_tag_);
      remote_process_completions(
          ctx_, cfg_.thread_idx, ring, ne, wc, ctx_by_tag_, atomic_buffer_ptr_,
          cfg_.num_ranks, cfg_.num_experts, pending_atomic_updates, cfg_.rank,
          cfg_.num_nodes, cfg_.use_normal_mode);
#ifdef USE_RECEIVER_BARRIER
      if (!cfg_.use_normal_mode) {
        apply_pending_updates(ctx_, pending_atomic_updates, atomic_buffer_ptr_,
                              cfg_.num_experts, cfg_.num_ranks);
      }
#endif
    } else {
      ++empty_iters;
    }
    if (outstanding_batches() == 0 && empty_iters >= kConsecutiveEmptyToExit) {
      break;
    }
    auto now = clock::now();
    if (now - last_log > std::chrono::milliseconds(1000)) {
      fprintf(stderr, "[quiet] polling... outstanding=%zu\n",
              outstanding_batches());
      last_log = now;
    }
  }
}

void Proxy::quiet(std::vector<uint64_t> wrs, std::vector<TransferCmd> cmds) {
  assert(cmds.size() == 1 && "quiet size must be 1");
  quiet_cq();
  acked_wrs_.insert(wrs[0]);
}

void Proxy::destroy(bool free_gpu_buffer) {
  for (auto& ctx_ptr : ctxs_for_all_ranks_) {
    if (!ctx_ptr) continue;
    qp_to_error(ctx_ptr->qp);
    qp_to_error(ctx_ptr->ack_qp);
    for (auto* q : ctx_ptr->data_qps_by_channel) {
      if (q) qp_to_error(q);
    }
  }

  for (auto& ctx_ptr : ctxs_for_all_ranks_) {
    if (!ctx_ptr) continue;
  }
  if (ctx_.cq) drain_cq(ctx_.cq);

  for (auto& ctx_ptr : ctxs_for_all_ranks_) {
    if (!ctx_ptr) continue;
    if (ctx_ptr->qp) {
      ibv_destroy_qp(ctx_ptr->qp);
      ctx_ptr->qp = nullptr;
    }
    for (auto*& q : ctx_ptr->data_qps_by_channel) {
      if (q) {
        ibv_destroy_qp(q);
        q = nullptr;
      }
    }
    ctx_ptr->data_qps_by_channel.clear();
    if (ctx_ptr->ack_qp) {
      ibv_destroy_qp(ctx_ptr->ack_qp);
      ctx_ptr->ack_qp = nullptr;
    }
  }
  ring.ack_qp = nullptr;

  for (auto& ctx_ptr : ctxs_for_all_ranks_) {
    if (!ctx_ptr) continue;
  }
  if (ctx_.cq) drain_cq(ctx_.cq);

  for (auto& ctx_ptr : ctxs_for_all_ranks_) {
    if (!ctx_ptr) continue;
  }
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
  acked_wrs_.clear();
  wr_id_to_start_time_.clear();
  ctxs_for_all_ranks_.clear();
  ctx_by_tag_.clear();
  local_infos_.clear();
  remote_infos_.clear();
}

void Proxy::post_barrier_msg(int dst_rank, bool ack, uint64_t seq) {
  ProxyCtx* ctx = ctxs_for_all_ranks_[dst_rank].get();
  if (!ctx || !ctx->qp || !ctx->mr) {
    fprintf(stderr, "barrier_msg: bad ctx for dst=%d\n", dst_rank);
    std::abort();
  }
  uint32_t imm = BarrierImm::Pack(ack, (uint32_t)seq, (uint8_t)cfg_.rank);
#ifdef EFA
  auto* qpx = (struct ibv_qp_ex*)ctx->qp;
  int barrier_seq = 0;
  ibv_wr_start(qpx);
  qpx->wr_id = kBarrierWrTag | (barrier_seq & kBarrierMask);
  qpx->comp_mask = 0;
  qpx->wr_flags = IBV_SEND_SIGNALED;
  // zero-length RDMA write with imm
  ibv_wr_rdma_write_imm(qpx, ctx->remote_rkey, ctx->remote_addr, htonl(imm));
  ibv_wr_set_ud_addr(qpx, ctx->dst_ah, ctx->dst_qpn, QKEY);
  ibv_wr_set_sge(qpx, ctx->mr->lkey, (uintptr_t)ctx->mr->addr, 0);
  int ret = ibv_wr_complete(qpx);
  if (ret) {
    fprintf(stderr, "post_barrier_msg (EFA) failed: %s (%d)\n", strerror(ret),
            ret);
    std::abort();
  }
#else
  ibv_sge sge{};
  sge.addr = (uintptr_t)ctx->mr->addr;
  sge.length = 0;
  sge.lkey = ctx->mr->lkey;

  ibv_send_wr wr{};
  int barrier_seq = 0;
  wr.wr_id = kBarrierWrTag | (barrier_seq & kBarrierMask);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(imm);
  wr.wr.rdma.remote_addr = ctx->remote_addr;
  wr.wr.rdma.rkey = ctx->remote_rkey;

  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(ctx->qp, &wr, &bad);
  if (ret) {
    fprintf(stderr, "post_barrier_msg failed: %s (%d)\n", strerror(ret), ret);
    if (bad)
      fprintf(stderr, "  bad wr_id=%llu\n", (unsigned long long)bad->wr_id);
    std::abort();
  }
#endif
}

void Proxy::send_barrier(uint64_t wr) {
#ifndef USE_MSCCLPP_FIFO_BACKEND
  assert(!ctx_.barrier_inflight && "only one barrier at a time");
  ctx_.barrier_inflight = true;
#endif
  assert(ctx_.barrier_wr == -1 && "barrier_wr should be 0");
  ctx_.barrier_wr = wr;
  ctx_.barrier_seq = ctx_.barrier_seq + 1;

  if (cfg_.rank == ctx_.node_leader_rank) {
    if (ctx_.barrier_arrived.size() != static_cast<size_t>(cfg_.num_nodes)) {
      ctx_.barrier_arrived.assign(cfg_.num_nodes, 0);
      ctx_.barrier_arrival_count = 0;
    }
  }
}

void Proxy::barrier_check() {
  if (!ctx_.barrier_inflight) return;

  uint64_t const seq = ctx_.barrier_seq;
  // Node leader aggregates local arrivals
  static thread_local uint64_t last_sent_seq = 0;
  if (last_sent_seq != seq) {
    last_sent_seq = seq;
    if (cfg_.node_idx == 0) {
      // Global leader: mark self-arrival; remote arrivals will come via
      // your existing CQ handler.
      if (ctx_.barrier_arrived.empty()) {
        ctx_.barrier_arrived.assign(ctxs_for_all_ranks_.size(), 0);
        ctx_.barrier_arrival_count = 0;
      }
      if (!ctx_.barrier_arrived[0]) {
        ctx_.barrier_arrived[0] = 1;
        ++ctx_.barrier_arrival_count;
      }
    } else {
      int rank = cfg_.rank - cfg_.node_idx * MAX_NUM_GPUS;
      if (rank < 0 || rank >= MAX_NUM_GPUS) {
        printf("rank: %d, node_idx: %d invalid for barrier\n", cfg_.rank,
               cfg_.node_idx);
      }
      assert(rank >= 0 && rank < MAX_NUM_GPUS);
      post_barrier_msg(/*dst=*/rank,
                       /*ack=*/false, seq);
    }
  }

  if (cfg_.node_idx == 0) {
    if (ctx_.barrier_arrival_count == cfg_.num_nodes) {
      std::unordered_map<std::string, int> leader_for_ip;
      for (int r = 0; r < (int)peers_.size(); ++r) {
        if (r >= MAX_NUM_GPUS && (r - cfg_.rank) % MAX_NUM_GPUS == 0) {
          leader_for_ip[peers_[r].ip] = r;
        }
      }
      for (auto const& kv : leader_for_ip) {
        std::string const& ip = kv.first;
        int leader_r = kv.second;
        if (ip == peers_[0].ip) continue;
        post_barrier_msg(leader_r, true, seq);
      }
      ctx_.barrier_arrived.clear();
      ctx_.barrier_arrival_count = 0;

      acked_wrs_.insert(ctx_.barrier_wr);
#ifndef USE_MSCCLPP_FIFO_BACKEND
      ctx_.barrier_inflight = false;
      ctx_.barrier_wr = -1;
#endif
      return;
    }
  }

  // When global release comes back (CQ handler should set these):
  if (ctx_.barrier_released && ctx_.barrier_release_seq == seq) {
    // Reset local mask for next barrier and consume the global release
    ctx_.barrier_released = false;

    // Complete WR
    acked_wrs_.insert(ctx_.barrier_wr);
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ctx_.barrier_inflight = false;
    ctx_.barrier_wr = -1;
#endif
  }
}
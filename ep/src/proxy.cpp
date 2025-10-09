#include "proxy.hpp"
#include "bench_utils.hpp"
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

static std::string shm_name_for_barrier(std::string const& ip, int thread_idx) {
  return "/uccl_barrier_" + ip + "_th" + std::to_string(thread_idx);
}

LocalBarrier* map_local_barrier_shm(std::string const& name, bool* out_owner) {
  *out_owner = false;
  size_t const kSize = sizeof(LocalBarrier);
  mode_t const kMode = 0600;
  int fd = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, kMode);
  if (fd >= 0) {
    *out_owner = true;
    if (ftruncate(fd, kSize) != 0) {
      perror("ftruncate(LocalBarrier)");
      close(fd);
      shm_unlink(name.c_str());
      return nullptr;
    }
  } else {
    if (errno != EEXIST) {
      perror("shm_open");
      return nullptr;
    }
    fd = shm_open(name.c_str(), O_RDWR, kMode);
    if (fd < 0) {
      perror("shm_open(existing)");
      return nullptr;
    }
    struct stat st {};
    int tries = 1000;
    while (tries-- > 0) {
      if (fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) >= kSize)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (tries < 0) {
      fprintf(stderr,
              "map_local_barrier_shm: existing shm '%s' never sized to %zu\n",
              name.c_str(), kSize);
      close(fd);
      return nullptr;
    }
  }
  void* p = mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  int saved_errno = errno;
  close(fd);
  if (p == MAP_FAILED) {
    errno = saved_errno;
    perror("mmap(LocalBarrier)");
    return nullptr;
  }
  return reinterpret_cast<LocalBarrier*>(p);
}

void unmap_local_barrier_shm(std::string const& name, LocalBarrier* lb,
                             bool owner) {
  if (lb) munmap(lb, sizeof(LocalBarrier));
  if (owner) shm_unlink(name.c_str());
}

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
    pin_thread_to_numa(ctx_.numa_node);
    printf(
        "Local CPU thread pinned to NUMA node %d, thread_idx: %d, "
        "local_rank: %d\n",
        ctx_.numa_node, cfg_.thread_idx, cfg_.local_rank);
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

void Proxy::set_bench_ring_addrs(std::vector<uintptr_t> const& addrs) {
  ring_tails_.clear();
  ring_seen_.clear();
  cfg_.ring_buffers.clear();

  ring_tails_.resize(addrs.size(), 0);
  ring_seen_.resize(addrs.size(), 0);
  cfg_.ring_buffers.reserve(addrs.size());
  for (auto addr : addrs) {
    cfg_.ring_buffers.push_back(reinterpret_cast<DeviceToHostCmdBuffer*>(addr));
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
#ifdef USE_NORMAL_MODE
    if (std::abs(peer - my_rank) % MAX_NUM_GPUS != 0) continue;
#endif
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
#ifdef USE_NORMAL_MODE
    if (std::abs(peer - my_rank) % MAX_NUM_GPUS != 0) continue;
#endif
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
#ifdef USE_NORMAL_MODE
    if (std::abs(peer - my_rank) % MAX_NUM_GPUS != 0) continue;
#endif
    auto& c = *ctxs_for_all_ranks_[peer];

    // qp is different from each rank.
    modify_qp_to_rtr(c, &remote_infos_[peer]);
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
#ifdef USE_NORMAL_MODE
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
  // if (cfg_.thread_idx != 0) return;
  const std::string shm_name = shm_name_for_barrier(my_ip, cfg_.thread_idx);
  ctx_.lb = map_local_barrier_shm(shm_name, &ctx_.lb_owner);
  if (!ctx_.lb) {
    fprintf(stderr, "Failed to map local barrier shm: %s\n", shm_name.c_str());
    std::abort();
  }
  if (ctx_.lb_owner) {
    ctx_.lb->full_mask = (ctx_.num_local_ranks >= 64)
                             ? ~0ULL
                             : ((1ULL << ctx_.num_local_ranks) - 1ULL);
    for (int i = 0; i < ctx_.num_local_ranks; ++i) {
      ctx_.lb->arrive_seq[i].store(0, std::memory_order_relaxed);
      ctx_.lb->release_seq[i].store(0, std::memory_order_relaxed);
    }
  } else {
    while (ctx_.lb->full_mask == 0ULL) cpu_relax();
  }
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
    local_poll_completions(ctx_, finished_wrs_, acked_wrs_, cfg_.thread_idx,
                           ctx_by_tag_);
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
                            cfg_.num_nodes);
    apply_pending_updates(ctx_, pending_atomic_updates, atomic_buffer_ptr_,
                          cfg_.num_experts, cfg_.num_ranks);
  }
}

void Proxy::run_dual() {
  init_common();
  for (int peer = 0; peer < (int)ctxs_for_all_ranks_.size(); ++peer) {
    if (peer == cfg_.rank) continue;
    if (peers_[peer].ip == peers_[cfg_.rank].ip) continue;
#ifdef USE_NORMAL_MODE
    if (std::abs(peer - cfg_.rank) % MAX_NUM_GPUS != 0) continue;
#endif
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
  // printf("run_dual initialization complete\n");
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    poll_cq_dual(ctx_, finished_wrs_, acked_wrs_, cfg_.thread_idx, ring,
                 ctx_by_tag_, atomic_buffer_ptr_, cfg_.num_ranks,
                 cfg_.num_experts, pending_atomic_updates, cfg_.rank,
                 cfg_.num_nodes);
    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);
#ifdef USE_RECEIVER_BARRIER
    apply_pending_updates(ctx_, pending_atomic_updates, atomic_buffer_ptr_,
                          cfg_.num_experts, cfg_.num_ranks);
#endif

#ifdef USE_SENDER_BARRIER
    auto postponed_wr_ids = postponed_wr_ids_;
    auto postponed_atomics = postponed_atomics_;
    postponed_wr_ids_.clear();
    postponed_atomics_.clear();
    assert(postponed_wr_ids.size() == postponed_atomics.size());
    assert(postponed_wr_ids_.size() == 0);
    post_gpu_commands_mixed(postponed_wr_ids, postponed_atomics);
#endif

#ifdef USE_NORMAL_MODE
    barrier_check();
#endif
  }
}

void Proxy::notify_gpu_completion(uint64_t& my_tail) {
  if (finished_wrs_.empty()) return;
  if (acked_wrs_.empty()) return;

  // Group completed work requests by ring buffer
  std::map<size_t, std::vector<std::pair<uint64_t, uint64_t>>>
      completed_by_ring;

  // Copy to iterate safely while erasing.
  std::vector<uint64_t> finished_copy(finished_wrs_.begin(),
                                      finished_wrs_.end());
  std::sort(finished_copy.begin(), finished_copy.end());

  for (auto wr_id : finished_copy) {
    if (acked_wrs_.find(wr_id) == acked_wrs_.end()) break;
    // Decode ring buffer index and command index from unique_wr_id
    size_t rb_idx = (wr_id >> 32) & 0xFFFFFFFF;
    uint64_t cmd_idx = wr_id & 0xFFFFFFFF;

    completed_by_ring[rb_idx].push_back({wr_id, cmd_idx});

    finished_wrs_.erase(wr_id);
    acked_wrs_.erase(wr_id);

#ifdef MEASURE_PER_VERB_LATENCY
    auto it = wr_id_to_start_time_.find(wr_id);
    if (it == wr_id_to_start_time_.end()) {
      fprintf(stderr, "Error: WR ID %lu not found in wr_id_to_start_time\n",
              wr_id);
      std::abort();
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - it->second);
    if (completion_count_ > kWarmupOps) {
      wr_time_total_us_ += duration.count();
    }
    completion_count_++;
#endif
  }

  // Process completions for each ring buffer
  for (auto& [rb_idx, completions] : completed_by_ring) {
    if (rb_idx >= cfg_.ring_buffers.size()) {
      fprintf(stderr, "Error: Invalid ring buffer index %zu\n", rb_idx);
      continue;
    }

    auto* ring_buffer = cfg_.ring_buffers[rb_idx];
    uint64_t& ring_tail = ring_tails_[rb_idx];

    // Clear ring entries and update tail
    for (auto& [wr_id, cmd_idx] : completions) {
      ring_buffer->volatile_store_cmd(cmd_idx, 0);
    }

    // Update tail for this ring buffer
    ring_tail += completions.size();
    ring_buffer->cpu_volatile_store_tail(ring_tail);
  }
}

void Proxy::post_gpu_command(uint64_t& my_tail, size_t& seen) {
  // Multi-ring buffer processing: collect commands from all ring buffers
  std::vector<uint64_t> wrs_to_post;
  std::vector<TransferCmd> cmds_to_post;
  bool found_work = false;

  // Process each ring buffer (similar to test_multi_ring_throughput.cu)
  for (size_t rb_idx = 0; rb_idx < cfg_.ring_buffers.size(); rb_idx++) {
    auto* ring_buffer = cfg_.ring_buffers[rb_idx];
    uint64_t& ring_tail = ring_tails_[rb_idx];
    size_t& ring_seen = ring_seen_[rb_idx];

    // Force load head from DRAM (like original code)
    uint64_t cur_head = ring_buffer->volatile_head();
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
      uint64_t cmd = ring_buffer->volatile_load_cmd(i);
      // NOTE(MaoZiming): Non-blocking. prevent local and remote both while
      // loop.
      if (cmd == 0) break;

      TransferCmd& cmd_entry = ring_buffer->load_cmd_entry(i);
      if (cmd_entry.cmd_type == CmdType::EMPTY) break;

      if (cmd_entry.cmd_type == CmdType::WRITE ||
          cmd_entry.cmd_type == CmdType::ATOMIC) {
        if (static_cast<int>(cmd_entry.dst_rank) == cfg_.rank) {
          fprintf(stderr,
                  "[ERROR] Local command!, cmd.dst_rank: %d, cfg_.rank: %d, "
                  "cmd_entry.cmd_type: %d, cur_head: %lu, i: %lu, cmd: %lu\n",
                  cmd_entry.dst_rank, cfg_.rank,
                  static_cast<int>(cmd_entry.cmd_type), cur_head, i, cmd);
          cudaDeviceSynchronize();
          std::abort();
        }
        if (peers_[cmd_entry.dst_rank].ip == peers_[cfg_.rank].ip) {
          fprintf(
              stderr,
              "[ERROR] Intra-node command!, cmd.dst_rank: %d, cfg_.rank: %d, "
              "cmd_entry.cmd_type: %d, cur_head: %lu, i: %lu, cmd: %lu\n",
              cmd_entry.dst_rank, cfg_.rank,
              static_cast<int>(cmd_entry.cmd_type), cur_head, i, cmd);
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
         cfg_.ring_buffers.size());

  if (cfg_.ring_buffers.empty()) {
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
    for (size_t rb_idx = 0; rb_idx < cfg_.ring_buffers.size(); rb_idx++) {
      auto* ring_buffer = cfg_.ring_buffers[rb_idx];
      uint64_t& ring_tail = ring_tails_[rb_idx];

      // Check for new work in this ring buffer
      uint64_t cur_head = ring_buffer->volatile_head();
      if (cur_head == ring_tail) {
        continue;  // No new work in this ring
      }

      // Process commands from this ring buffer
      while (ring_tail < cur_head) {
        uint64_t const idx = ring_tail & kQueueMask;
        uint64_t cmd;

        auto last_print = std::chrono::steady_clock::now();
        size_t spin_count = 0;
        do {
          cmd = ring_buffer->volatile_load_cmd(idx);
          cpu_relax();

          auto now = std::chrono::steady_clock::now();
          if (now - last_print > std::chrono::seconds(10)) {
            printf(
                "Still waiting at thread %d, ring %zu, total_seen=%d, "
                "spin_count=%zu, ring_tail=%lu, cmd: %lu\n",
                cfg_.thread_idx, rb_idx, total_seen, spin_count, ring_tail,
                cmd);
            last_print = now;
            spin_count++;
          }

          if (!ctx_.progress_run.load(std::memory_order_acquire)) {
            printf("Local thread %d stopping early at total_seen=%d\n",
                   cfg_.thread_idx, total_seen);
            return;
          }
        } while (cmd == 0);

#ifdef DEBUG_PRINT
        printf(
            "Local thread %d, ring %zu, total_seen=%d head=%lu tail=%lu "
            "consuming cmd=%llu\n",
            cfg_.thread_idx, rb_idx, total_seen, ring_buffer->head, ring_tail,
            static_cast<unsigned long long>(cmd));
#endif

        std::atomic_thread_fence(std::memory_order_acquire);
        if (cmd == 1) {
          TransferCmd& cmd_entry = ring_buffer->buf[idx];
          printf(
              "Received command 1: thread %d, ring %zu, total_seen=%d, value: "
              "%d\n",
              cfg_.thread_idx, rb_idx, total_seen, cmd_entry.value);
        }

        // Mark command as processed
        ring_buffer->volatile_store_cmd(idx, 0);
        ring_tail++;
        ring_buffer->cpu_volatile_store_tail(ring_tail);
        total_seen++;
        found_work = true;

        // Break to check other ring buffers and progress_run flag
        break;
      }
    }

    // If no work found across all ring buffers, relax CPU
    if (!found_work) {
      cpu_relax();
    }
  }

  printf("Local thread %d finished %d commands across %zu ring buffers\n",
         cfg_.thread_idx, total_seen, cfg_.ring_buffers.size());
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
    if (cmds_to_post[i].cmd_type == CmdType::ATOMIC) {
#ifdef USE_SENDER_BARRIER
      int value = cmds_to_post[i].value;
      uint32_t offset = static_cast<int64_t>(cmds_to_post[i].req_rptr);
      uint32_t new_offset =
          offset - cmds_to_post[i].low_latency_buffer_idx *
                       align<size_t>(cfg_.num_experts * sizeof(int), 128);
      size_t new_index = new_offset / sizeof(int);
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
#endif
      atomic_wrs.push_back(wrs_to_post[i]);
      atomic_cmds.push_back(cmds_to_post[i]);

#ifdef USE_SENDER_BARRIER
      if (cmds_to_post[i].is_combine) {
        ctx_.combine_sent_counter.Reset({cmds_to_post[i].low_latency_buffer_idx,
                                         expert_idx, cmds_to_post[i].dst_rank});
      } else {
        ctx_.dispatch_sent_counter.Reset(
            {cmds_to_post[i].low_latency_buffer_idx, expert_idx,
             cmds_to_post[i].dst_rank});
      }
#endif
    } else if (cmds_to_post[i].cmd_type == CmdType::WRITE) {
      rdma_wrs.push_back(wrs_to_post[i]);
      rdma_cmds.push_back(cmds_to_post[i]);
    } else if (cmds_to_post[i].cmd_type == CmdType::QUIET) {
      quiet_cmds.push_back(cmds_to_post[i]);
      quiet_wrs.push_back(wrs_to_post[i]);
    } else if (cmds_to_post[i].cmd_type == CmdType::BARRIER) {
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
  // if (rdma_wrs.size() + atomic_wrs.size() + barrier_cmds.size() +
  //       quiet_cmds.size() > 10) {
  //   printf(
  //       "[post_gpu_commands_mixed] thread %d: Posting %zu RDMA writes, %zu "
  //       "atomics, %zu barriers, %zu quiets\n",
  //       cfg_.thread_idx, rdma_wrs.size(), atomic_wrs.size(),
  //       barrier_cmds.size(), quiet_cmds.size());
  // }

  // Handle regular RDMA writes
  if (!barrier_cmds.empty()) {
    // barrier(barrier_wrs, barrier_cmds);
    assert(barrier_wrs.size() == 1);
    send_barrier(barrier_wrs[0]);
  }
  if (!quiet_cmds.empty()) {
    assert(quiet_wrs.size() == 1);
    quiet(quiet_wrs, quiet_cmds);
  }

  if (!rdma_wrs.empty()) {
    post_rdma_async_batched(ctx_, cfg_.gpu_buffer, rdma_wrs.size(), rdma_wrs,
                            rdma_cmds, ctxs_for_all_ranks_, cfg_.rank,
                            cfg_.thread_idx, finished_wrs_);
  }
  if (!atomic_wrs.empty()) {
    post_atomic_operations(ctx_, atomic_wrs, atomic_cmds, ctxs_for_all_ranks_,
                           cfg_.rank, cfg_.thread_idx, finished_wrs_,
                           acked_wrs_);
  }
}

static inline int poll_cq_once(ibv_cq* cq, ibv_wc* wc, int max_cqes) {
#ifdef EFA
  auto cqx = reinterpret_cast<ibv_cq_ex*>(cq);
  ibv_poll_cq_attr attr{.comp_mask = 0};
  if (ibv_start_poll(cqx, &attr)) return 0;

  int n = 0;
  while (n < max_cqes) {
    wc[n].status = cqx->status;
    wc[n].wr_id = cqx->wr_id;
    wc[n].opcode = ibv_wc_read_opcode(cqx);
    wc[n].wc_flags = ibv_wc_read_wc_flags(cqx);
    wc[n].imm_data = ibv_wc_read_imm_data(cqx);
    wc[n].byte_len = ibv_wc_read_byte_len(cqx);
    ++n;
    if (ibv_next_poll(cqx)) break;
  }
  ibv_end_poll(cqx);
  return n;
#else
  return ibv_poll_cq(cq, max_cqes, wc);
#endif
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
      local_process_completions(ctx_, finished_wrs_, acked_wrs_,
                                cfg_.thread_idx, wc, ne, ctx_by_tag_);
      remote_process_completions(
          ctx_, cfg_.thread_idx, ring, ne, wc, ctx_by_tag_, atomic_buffer_ptr_,
          cfg_.num_ranks, cfg_.num_experts, pending_atomic_updates, cfg_.rank,
          cfg_.num_nodes);
#ifdef USE_RECEIVER_BARRIER
      apply_pending_updates(ctx_, pending_atomic_updates, atomic_buffer_ptr_,
                            cfg_.num_experts, cfg_.num_ranks);
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
  finished_wrs_.insert(wrs[0]);
  acked_wrs_.insert(wrs[0]);
}

void Proxy::destroy(bool free_gpu_buffer) {
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

  // Unmap local barrier shm
  // if (cfg_.thread_idx == 0) {
  std::string const my_ip =
      (cfg_.rank < (int)peers_.size()) ? peers_[cfg_.rank].ip : "";
  std::string const shm_name = shm_name_for_barrier(my_ip, cfg_.thread_idx);
  unmap_local_barrier_shm(shm_name, ctx_.lb, ctx_.lb_owner);
  ctx_.lb = nullptr;
  ctx_.lb_owner = false;
  // }

  finished_wrs_.clear();
  acked_wrs_.clear();
  wr_id_to_start_time_.clear();
  ctxs_for_all_ranks_.clear();
  ctx_by_tag_.clear();
  local_infos_.clear();
  remote_infos_.clear();
}

void Proxy::post_barrier_msg(int dst_rank, bool ack, uint64_t seq) {
  // assert(cfg_.thread_idx == 0 && "barrier must be posted on thread 0");
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
  wr.wr_id = 0;
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
  // assert(cfg_.thread_idx == 0 && "barrier must be placed on thread 0");
  assert(!ctx_.barrier_inflight && "only one barrier at a time");
  assert(ctx_.barrier_wr == 0 && "barrier_wr should be 0");
  ctx_.barrier_inflight = true;
  ctx_.barrier_wr = wr;
  ctx_.barrier_seq = ctx_.barrier_seq + 1;
  finished_wrs_.insert(wr);

  if (cfg_.rank == ctx_.node_leader_rank) {
    if (ctx_.barrier_arrived.size() != static_cast<size_t>(cfg_.num_nodes)) {
      ctx_.barrier_arrived.assign(cfg_.num_nodes, 0);
      ctx_.barrier_arrival_count = 0;
    }
  }

  // Mark local arrival
  auto* lb = ctx_.lb;
  lb->seq.store(ctx_.barrier_seq, std::memory_order_release);
  lb->arrive_seq[ctx_.local_rank].store((uint32_t)ctx_.barrier_seq,
                                        std::memory_order_release);

  uint64_t bit = (1ULL << (uint64_t)ctx_.local_rank);
  lb->arrived_mask.fetch_or(bit, std::memory_order_acq_rel);
}

void Proxy::barrier_check() {
  // if (cfg_.thread_idx != 0) return;  // first thread handle barrier.
  if (!ctx_.barrier_inflight) return;

  auto* lb = ctx_.lb;
  uint64_t const seq = ctx_.barrier_seq;

  // Node leader aggregates local arrivals
  if (cfg_.rank == ctx_.node_leader_rank) {
    bool all_local_arrived = true;
    for (int lr = 0; lr < ctx_.num_local_ranks; ++lr) {
      uint32_t seen = lb->arrive_seq[lr].load(std::memory_order_acquire);
      if ((uint32_t)seen != seq) {
        all_local_arrived = false;
        break;
      }
    }

    if (all_local_arrived) {
      static thread_local uint64_t last_sent_seq = 0;
      if (last_sent_seq != seq) {
        last_sent_seq = seq;
        if (cfg_.rank == 0) {
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
          post_barrier_msg(/*dst=*/0, /*ack=*/false, seq);
        }
      }

      if (cfg_.rank == 0) {
        if (ctx_.barrier_arrival_count == cfg_.num_nodes) {
          std::unordered_map<std::string, int> leader_for_ip;
          for (int r = 0; r < (int)peers_.size(); ++r) {
            auto it = leader_for_ip.find(peers_[r].ip);
            if (it == leader_for_ip.end() || r < it->second) {
              assert(r % MAX_NUM_GPUS == 0);
              leader_for_ip[peers_[r].ip] = r;
            }
          }
          for (auto const& kv : leader_for_ip) {
            std::string const& ip = kv.first;
            int leader_r = kv.second;
            if (ip == peers_[0].ip) continue;
            post_barrier_msg(leader_r, true, seq);
          }

          for (int lr = 0; lr < ctx_.num_local_ranks; ++lr) {
            lb->release_seq[lr].store(seq, std::memory_order_release);
          }
          ctx_.barrier_arrived.clear();
          ctx_.barrier_arrival_count = 0;

          acked_wrs_.insert(ctx_.barrier_wr);
          ctx_.barrier_inflight = false;
          ctx_.barrier_wr = 0;
          return;
        }
      }

      // When global release comes back (CQ handler should set these):
      if (ctx_.barrier_released && ctx_.barrier_release_seq == seq) {
        // Fan-out to local ranks via shared memory
        for (int lr = 0; lr < ctx_.num_local_ranks; ++lr) {
          lb->release_seq[lr].store(seq, std::memory_order_release);
        }
        // Reset local mask for next barrier and consume the global release
        ctx_.barrier_released = false;

        // Complete WR
        acked_wrs_.insert(ctx_.barrier_wr);
        ctx_.barrier_inflight = false;
        ctx_.barrier_wr = 0;
      }
    }
    return;
  } else {
    assert(!ctx_.barrier_released &&
           "This can only be set by local leader thread.");
  }

  // Followers: wait until leader sets our release_seq
  if (lb->release_seq[ctx_.local_rank].load(std::memory_order_acquire) == seq) {
    acked_wrs_.insert(ctx_.barrier_wr);
    ctx_.barrier_inflight = false;
    ctx_.barrier_wr = 0;
  }
}
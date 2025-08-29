#include "proxy.hpp"
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
    pin_thread_to_cpu(cfg_.block_idx + 1);
    int cpu = sched_getcpu();
    if (cpu == -1) {
      perror("sched_getcpu");
    } else {
      printf("Local CPU thread pinned to core %d\n", cpu);
    }
  }
}

void Proxy::init_common() {
  per_thread_rdma_init(ctx_, cfg_.gpu_buffer, cfg_.total_size, cfg_.rank,
                       cfg_.block_idx);
  pin_thread();

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

  printf("[PROXY_INIT] Atomic buffer at %p, size %zu bytes\n",
         ctx_.atomic_old_values_buf, atomic_buf_size);

  // CQ + QP creation
  ctx_.cq = create_per_thread_cq(ctx_);
  create_per_thread_qp(ctx_, cfg_.gpu_buffer, cfg_.total_size, &local_info_,
                       cfg_.rank);

  modify_qp_to_init(ctx_);
  exchange_connection_info(cfg_.rank, cfg_.peer_ip, cfg_.block_idx,
                           &local_info_, &remote_info_);
  modify_qp_to_rtr(ctx_, &remote_info_);
  modify_qp_to_rts(ctx_, &local_info_);

  ctx_.remote_addr = remote_info_.addr;
  printf("Remote address: %p, RKey: %u\n", (void*)ctx_.remote_addr,
         remote_info_.rkey);
  // Add to debug file for core issue tracking
  FILE* debug_file = fopen("/tmp/uccl_debug.txt", "a");
  if (debug_file) {
    fprintf(debug_file,
            "[PROXY_INIT] Block %d: remote_addr=0x%lx, local_buffer=0x%lx\n",
            cfg_.block_idx, ctx_.remote_addr, (uintptr_t)cfg_.gpu_buffer);
    fclose(debug_file);
  }
  ctx_.remote_rkey = remote_info_.rkey;
}

void Proxy::init_sender() {
  init_common();
  // sender ACK receive ring (your existing code)
  local_post_ack_buf(ctx_, kSenderAckQueueDepth);
}

void Proxy::init_remote() {
  init_common();
  // Remote side ensures ack sender resources (legacy globals)
  remote_reg_ack_buf(ctx_.pd, ring.ack_buf, ring.ack_mr);
  ring.ack_qp = ctx_.ack_qp;
  post_receive_buffer_for_imm(ctx_);
}

void Proxy::run_sender() {
  printf("CPU sender thread for block %d started\n", cfg_.block_idx + 1);
  init_sender();
  size_t seen = 0;
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    local_poll_completions(ctx_, finished_wrs_, finished_wrs_mutex_,
                           cfg_.block_idx);
    notify_gpu_completion(cfg_.rb->tail);
    post_gpu_command(cfg_.rb->tail, seen);
  }
}

void Proxy::run_remote() {
  printf("Remote CPU thread for block %d started\n", cfg_.block_idx + 1);
  init_remote();
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    remote_poll_completions(ctx_, cfg_.block_idx, ring);
  }
}

void Proxy::run_dual() {
  printf("Dual (single-thread) proxy for block %d starting\n",
         cfg_.block_idx + 1);
  init_remote();
  // == sender-only bits:
  local_post_ack_buf(ctx_, kSenderAckQueueDepth);

  uint64_t my_tail = 0;
  size_t seen = 0;
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    poll_cq_dual(ctx_, finished_wrs_, finished_wrs_mutex_, cfg_.block_idx,
                 ring);
    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);
  }
}

void Proxy::notify_gpu_completion(uint64_t& my_tail) {
#ifdef ASSUME_WR_IN_ORDER
  if (finished_wrs_.empty()) return;

  std::lock_guard<std::mutex> lock(finished_wrs_mutex_);
  int check_i = 0;
  int actually_completed = 0;

  // Copy to iterate safely while erasing.
  std::unordered_set<uint64_t> finished_copy(finished_wrs_.begin(),
                                             finished_wrs_.end());
  for (auto wr_id : finished_copy) {
#ifdef SYNCHRONOUS_COMPLETION
    // These are your existing global conditions.
    if (!(ctx_.has_received_ack && ctx_.largest_completed_wr >= wr_id)) {
      continue;
    }
    finished_wrs_.erase(wr_id);
#else
    finished_wrs_.erase(wr_id);
#endif
    // Clear ring entry (contiguity assumed)
    cfg_.rb->buf[(my_tail + check_i) & kQueueMask].cmd = 0;
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

  my_tail += actually_completed;
#ifndef SYNCHRONOUS_COMPLETION
  finished_wrs_.clear();
#endif
  // cfg_.rb->tail = my_tail;
  cfg_.rb->cpu_volatile_store_tail(my_tail);
#else
  printf("ASSUME_WR_IN_ORDER is not defined. This is not supported.\n");
  std::abort();
#endif
}

void Proxy::post_gpu_command(uint64_t& my_tail, size_t& seen) {
  // Force load head from DRAM
  uint64_t cur_head = cfg_.rb->volatile_head();
  if (cur_head == my_tail) {
    cpu_relax();
    return;
  }
  size_t batch_size = cur_head - seen;
  std::vector<uint64_t> wrs_to_post;
  wrs_to_post.reserve(batch_size);
  std::vector<TransferCmd> cmds_to_post;
  cmds_to_post.reserve(batch_size);

  for (size_t i = seen; i < cur_head; ++i) {
    uint64_t cmd;
    auto last_print = std::chrono::steady_clock::now();
    size_t spin_count = 0;
    do {
      cmd = cfg_.rb->volatile_load_cmd(i);
      cpu_relax();  // avoid hammering cacheline

      auto now = std::chrono::steady_clock::now();
      if (now - last_print > std::chrono::seconds(10)) {
        printf(
            "Still waiting at block %d, seen=%ld, spin_count=%zu, my_tail=%lu, "
            "cmd: %lu\n",
            cfg_.block_idx + 1, seen, spin_count, my_tail, cmd);
        last_print = now;
        spin_count++;
      }

      if (!ctx_.progress_run.load(std::memory_order_acquire)) {
        printf("Local block %d stopping early at seen=%ld\n",
               cfg_.block_idx + 1, seen);
        return;
      }
    } while (cmd == 0);

    TransferCmd& cmd_entry = cfg_.rb->buf[i];
    wrs_to_post.push_back(i);
    cmds_to_post.push_back(cmd_entry);
    wr_id_to_start_time_[i] = std::chrono::high_resolution_clock::now();
  }
  seen = cur_head;

  if (wrs_to_post.size() != batch_size) {
    fprintf(stderr, "Error: wrs_to_post size %zu != batch_size %zu\n",
            wrs_to_post.size(), batch_size);
    std::abort();
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

    const uint64_t idx = my_tail & kQueueMask;
    uint64_t cmd;
    auto last_print = std::chrono::steady_clock::now();
    size_t spin_count = 0;
    do {
      cmd = cfg_.rb->volatile_load_cmd(idx);
      cpu_relax();  // avoid hammering cacheline

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

    cfg_.rb->buf[idx].cmd = 0;
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
    if (cmds_to_post[i].cmd == 1) {
      // Atomic operation (cmd.cmd == 1)
      atomic_wrs.push_back(wrs_to_post[i]);
      atomic_cmds.push_back(cmds_to_post[i]);
    } else {
      // Regular RDMA write
      rdma_wrs.push_back(wrs_to_post[i]);
      rdma_cmds.push_back(cmds_to_post[i]);
    }
  }

  // Handle regular RDMA writes
  if (!rdma_wrs.empty()) {
    post_rdma_async_batched(ctx_, cfg_.gpu_buffer, kObjectSize, rdma_wrs.size(),
                            rdma_wrs, finished_wrs_, finished_wrs_mutex_,
                            rdma_cmds);
  }

  // Handle atomic operations
  if (!atomic_wrs.empty()) {
    post_atomic_operations(atomic_wrs, atomic_cmds);
  }
}

void Proxy::post_atomic_operations(
    std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post) {
  // Use RDMA hardware atomic operations directly on remote GPU memory

  if (cmds_to_post.size() > ProxyCtx::kMaxAtomicOps) {
    fprintf(stderr, "Too many atomic operations: %zu > %zu\n",
            cmds_to_post.size(), ProxyCtx::kMaxAtomicOps);
    std::abort();
  }

  std::vector<ibv_send_wr> wrs(cmds_to_post.size());
  std::vector<ibv_sge> sges(cmds_to_post.size());

  for (size_t i = 0; i < cmds_to_post.size(); ++i) {
    auto const& cmd = cmds_to_post[i];

    // SGE for receiving the old value, using GPU memory!
    // Place atomic buffers safely within MR bounds, starting from the end
    uint64_t mr_start = reinterpret_cast<uintptr_t>(ctx_.mr->addr);
    uint64_t mr_end = mr_start + ctx_.mr->length;

    // TODO:yihan: modify tthe size later. Reserve 16KB at the end for atomic
    // operations (sufficient for all threads, )
    uint64_t atomic_region_start = mr_end - 16384;   // 16KB for atomic ops
    uint64_t thread_offset = cfg_.block_idx * 1024;  // 1KB per thread (reduced)
    uint64_t sge_addr = atomic_region_start + thread_offset +
                        i * sizeof(uint64_t);  // 64-bit aligned

    sges[i] = {
        .addr = sge_addr,
        .length =
            sizeof(uint64_t),  // 8-byte atomic operations (like rdma_test.cc)
        .lkey = ctx_.mr->lkey  // Now this matches - our own MR's lkey
    };

    // Verify atomic buffer is within bounds
    if (sge_addr < mr_start || sge_addr + sizeof(uint64_t) > mr_end) {
      printf("[ERROR] SGE addr 0x%lx exceeds MR bounds - ABORTING\n", sge_addr);
      std::abort();
    }

    std::memset(&wrs[i], 0, sizeof(wrs[i]));
    wrs[i].wr_id = wrs_to_post[i];
    wrs[i].sg_list = &sges[i];
    wrs[i].num_sge = 1;

    // Use RDMA atomic fetch_and_add
    wrs[i].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;

    int64_t signed_offset = static_cast<int64_t>(cmd.req_rptr);
    bool is_combine_op = (cmd.value == 1);

    uint64_t target_addr;
    if (is_combine_op) {
      // Combine operations: use direct offset from remote base
      target_addr = ctx_.remote_addr + signed_offset;
    } else {
      // Dispatch operations: use offset from dispatch_recv_data_offset
      target_addr =
          ctx_.remote_addr + ctx_.dispatch_recv_data_offset + signed_offset;
    }

    // Validate target address
    uint64_t remote_mr_end = ctx_.remote_addr + ctx_.mr->length;
    if (target_addr < ctx_.remote_addr ||
        target_addr + sizeof(uint64_t) > remote_mr_end) {
      printf("[ERROR] Atomic target 0x%lx outside MR bounds - ABORTING\n",
             target_addr);
      std::abort();
    }

    if (target_addr % sizeof(uint64_t) != 0) {
      printf("[ERROR] Atomic target 0x%lx not 8-byte aligned - ABORTING\n",
             target_addr);
      std::abort();
    }

    wrs[i].wr.atomic.remote_addr = target_addr;
    wrs[i].wr.atomic.rkey = ctx_.remote_rkey;
    // For 64-bit atomic operations with signed values
    // RDMA fetch_and_add supports signed addition - negative values are part of
    // the protocol! Negative values indicate token counts: -num_tokens_sent - 1
    // The receiver expects negative values to detect completion and recover
    // token count

    // Convert signed int to unsigned 64-bit while preserving the bit pattern
    // This allows negative values to be correctly added by RDMA hardware
    uint64_t increment_value =
        static_cast<uint64_t>(static_cast<int64_t>(cmd.value));

    wrs[i].wr.atomic.compare_add = increment_value;
    wrs[i].send_flags = IBV_SEND_SIGNALED;
    wrs[i].next = (i + 1 < cmds_to_post.size()) ? &wrs[i + 1] : nullptr;

    // Key info: what we're sending where
    printf("[ATOMIC_SEND] → 0x%lx: %d\n", wrs[i].wr.atomic.remote_addr,
           cmd.value);
  }

  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(ctx_.qp, &wrs[0], &bad);
  if (ret) {
    fprintf(stderr, "RDMA atomic operations failed: %s (ret=%d)\n",
            strerror(ret), ret);
    if (bad)
      fprintf(stderr, "Bad atomic WR at %p (wr_id=%lu)\n", (void*)bad,
              bad->wr_id);
    std::abort();
  }
}
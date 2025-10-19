#include "uccl_fifo_proxy.hpp"
#include "common.hpp"
#include "proxy_ctx.hpp"
#include "rdma.hpp"  // For local_poll_completions, remote_poll_completions, apply_pending_updates
#include "ring_buffer.cuh"
#include <chrono>
#include <iostream>
#include <set>
#include <unordered_set>

FifoProxy::FifoProxy(int thread_idx, uintptr_t gpu_buffer_addr,
                     size_t total_size, int rank, int node_idx, int local_rank,
                     std::string const& peer_ip)
    : thread_idx(thread_idx),
      fifo_(nullptr),
      stop_flag_(false),
      gpu_buffer_addr_(gpu_buffer_addr),
      total_size_(total_size),
      rank_(rank),
      node_idx_(node_idx),
      local_rank_(local_rank),
      peer_ip_(peer_ip) {}

FifoProxy::~FifoProxy() { stop(); }

void FifoProxy::set_fifo(mscclpp::Fifo* fifo) { fifo_ = fifo; }

void FifoProxy::set_peers_meta(
    std::vector<std::tuple<int, uintptr_t, size_t, std::string>> const& meta) {
  peers_meta_ = meta;

  // Convert to PeerMeta format for Proxy
  std::vector<PeerMeta> peer_metas;
  for (auto const& [rank, ptr, nbytes, ip] : meta) {
    peer_metas.push_back({rank, ptr, nbytes, ip});
  }

  // Create Proxy::Config
  // Note: we don't pass ring_buffers since we're using FIFO
  Proxy::Config cfg;
  cfg.thread_idx = thread_idx;
  cfg.gpu_buffer = reinterpret_cast<void*>(gpu_buffer_addr_);
  cfg.total_size = total_size_;
  cfg.rank = rank_;
  cfg.node_idx = node_idx_;
  cfg.local_rank = local_rank_;
  cfg.peer_ip = peer_ip_.empty() ? nullptr : peer_ip_.c_str();
  cfg.pin_thread = true;

  // Set RDMA parameters (for 2-node benchmarking)
  cfg.num_experts = 0;
  cfg.num_ranks = 2;
  cfg.num_nodes = 2;

  // Create the underlying Proxy
  proxy_ = std::make_unique<Proxy>(cfg);
  proxy_->set_peers_meta(peer_metas);
}

void FifoProxy::start_sender() {
  if (!fifo_ || !proxy_) {
    throw std::runtime_error("FIFO or Proxy not set before starting");
  }
  stop_flag_.store(false, std::memory_order_release);
  proxy_->set_progress_run(true);
  thread_ = std::make_unique<std::thread>([this]() { run_sender(); });
}

void FifoProxy::start_remote() {
  if (!fifo_ || !proxy_) {
    throw std::runtime_error("FIFO or Proxy not set before starting");
  }
  stop_flag_.store(false, std::memory_order_release);
  proxy_->set_progress_run(true);
  thread_ = std::make_unique<std::thread>([this]() { run_remote(); });
}

void FifoProxy::stop() {
  stop_flag_.store(true, std::memory_order_release);
  if (proxy_) {
    proxy_->set_progress_run(false);
  }
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
  thread_.reset();
}

void FifoProxy::run_sender() {
  std::cout << "[FifoProxy " << thread_idx << "] Sender started (FIFO mode)"
            << std::endl;

  // Initialize RDMA like the original Proxy::init_sender()
  proxy_->init_sender();

  // Main loop: mimic Proxy::run_sender() - poll FIFO, post, poll completions
  uint64_t fifo_head_seen = 0;   // Number of triggers we've read from FIFO
  uint64_t fifo_tail_acked = 0;  // Number of triggers we've popped from FIFO

  while (!stop_flag_.load(std::memory_order_acquire) &&
         proxy_->ctx_.progress_run.load(std::memory_order_acquire)) {
    // Poll completions (like original proxy)
    local_poll_completions(proxy_->ctx_, proxy_->finished_wrs_,
                           proxy_->acked_wrs_, thread_idx, proxy_->ctx_by_tag_);

    // printf("Finished wrs: %zu, Acked wrs: %zu, fifo_tail_acked: %lu,
    // fifo_head_seen: %lu\n", proxy_->finished_wrs_.size(),
    // proxy_->acked_wrs_.size(), fifo_tail_acked, fifo_head_seen);

    // Process completed work requests (similar to notify_gpu_completion)
    while (fifo_tail_acked < fifo_head_seen &&
           proxy_->finished_wrs_.count(fifo_tail_acked) > 0 &&
           proxy_->acked_wrs_.count(fifo_tail_acked) > 0) {
#ifdef MEASURE_PER_VERB_LATENCY
      // Track latency
      auto it = proxy_->wr_id_to_start_time_.find(fifo_tail_acked);
      if (it != proxy_->wr_id_to_start_time_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - it->second);
        if (proxy_->completion_count_ > kWarmupOps) {
          proxy_->wr_time_total_us_ += duration.count();
        }
        proxy_->completion_count_++;
        proxy_->wr_id_to_start_time_.erase(it);
      }
#endif

      // Remove from tracking sets
      proxy_->finished_wrs_.erase(fifo_tail_acked);
      proxy_->acked_wrs_.erase(fifo_tail_acked);

      // Pop from FIFO
      fifo_->pop();
      fifo_tail_acked++;
    }

    if (fifo_head_seen - fifo_tail_acked > 128) {
      // printf("Fifo head seen: %lu, Fifo tail acked: %lu, diff: %lu\n",
      // fifo_head_seen, fifo_tail_acked, fifo_head_seen - fifo_tail_acked);
      continue;
    }

    // Poll FIFO for next command
    mscclpp::ProxyTrigger trigger = fifo_->poll();

    // If fst is 0, FIFO is empty - continue polling
    if (trigger.fst == 0) {
      cpu_relax();
      continue;
    }

    // Convert trigger to TransferCmd
    TransferCmd cmd;
    cmd.cmd_type = CmdType::WRITE;
    cmd.cmd = trigger.fst;
    cmd.dst_rank = 1;
    cmd.dst_gpu = 0;
    cmd.src_ptr = reinterpret_cast<void*>(gpu_buffer_addr_);
    cmd.bytes = kObjectSize;
    cmd.message_idx = trigger.snd & 0xFFFFFFFF;
    cmd.warp_id = 0;
    cmd.lane_id = 0;
    cmd.expert_idx = 0;
    cmd.req_rptr = 0;
    cmd.req_lptr = 0;
    cmd.is_atomic = false;
    cmd.value = 0;
    cmd.is_combine = false;
    cmd.low_latency_buffer_idx = 0;

    // Post immediately (no batching)
    std::vector<uint64_t> wrs_to_post{fifo_head_seen};
    std::vector<TransferCmd> cmds_to_post{cmd};

#ifdef MEASURE_PER_VERB_LATENCY
    // Record timestamp for latency measurement (like original proxy)
    proxy_->wr_id_to_start_time_[fifo_head_seen] =
        std::chrono::high_resolution_clock::now();
#endif

    proxy_->post_gpu_commands_mixed(wrs_to_post, cmds_to_post);
    fifo_head_seen++;
  }

  // Wait for all remaining completions
  while (fifo_tail_acked < fifo_head_seen) {
    local_poll_completions(proxy_->ctx_, proxy_->finished_wrs_,
                           proxy_->acked_wrs_, thread_idx, proxy_->ctx_by_tag_);

    while (fifo_tail_acked < fifo_head_seen &&
           proxy_->finished_wrs_.count(fifo_tail_acked) > 0 &&
           proxy_->acked_wrs_.count(fifo_tail_acked) > 0) {
#ifdef MEASURE_PER_VERB_LATENCY
      auto it = proxy_->wr_id_to_start_time_.find(fifo_tail_acked);
      if (it != proxy_->wr_id_to_start_time_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - it->second);
        if (proxy_->completion_count_ > kWarmupOps) {
          proxy_->wr_time_total_us_ += duration.count();
        }
        proxy_->completion_count_++;
        proxy_->wr_id_to_start_time_.erase(it);
      }
#endif

      proxy_->finished_wrs_.erase(fifo_tail_acked);
      proxy_->acked_wrs_.erase(fifo_tail_acked);
      fifo_->pop();
      fifo_tail_acked++;
    }
  }

  std::cout << "[FifoProxy " << thread_idx << "] Sender stopped, processed "
            << fifo_tail_acked << " commands" << std::endl;
}

void FifoProxy::run_remote() {
  std::cout << "[FifoProxy " << thread_idx << "] Remote started (FIFO mode)"
            << std::endl;

  // Initialize RDMA like the original Proxy::init_remote()
  proxy_->init_remote();

  // Main loop: poll RDMA completions like the original run_remote()
  // The remote side doesn't read commands from FIFO, it receives RDMA
  // operations
  std::set<PendingUpdate> pending_atomic_updates;

  while (!stop_flag_.load(std::memory_order_acquire) &&
         proxy_->ctx_.progress_run.load(std::memory_order_acquire)) {
    remote_poll_completions(proxy_->ctx_, thread_idx, proxy_->ring,
                            proxy_->ctx_by_tag_, proxy_->atomic_buffer_ptr_,
                            2,  // num_ranks (simplified for 2-node case)
                            0,  // num_experts
                            pending_atomic_updates, rank_, 2);
    apply_pending_updates(proxy_->ctx_, pending_atomic_updates,
                          proxy_->atomic_buffer_ptr_, 0,
                          2);  // num_experts=0, num_ranks=2
  }

  std::cout << "[FifoProxy " << thread_idx << "] Remote stopped" << std::endl;
}

double FifoProxy::avg_wr_latency_us() const {
  if (proxy_) {
    return proxy_->avg_wr_latency_us();
  }
  return 0.0;
}

uint64_t FifoProxy::processed_count() const {
  if (proxy_) {
    return proxy_->completed_wr();
  }
  return 0;
}

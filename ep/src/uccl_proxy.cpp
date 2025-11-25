#include "uccl_proxy.hpp"
#include "common.hpp"
#include "proxy_ctx.hpp"
#include "rdma.hpp"
#include "ring_buffer.cuh"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <set>
#include <stdexcept>
#include <unordered_set>

UcclProxy::UcclProxy(int thread_idx, uintptr_t gpu_buffer_addr,
                     size_t total_size, int rank, int node_idx, int local_rank,
                     int num_experts, int num_ranks, int num_nodes,
                     bool use_normal_mode, bool is_intranode)
    : thread_{},
      mode_{Mode::None},
      running_{false},
      is_intranode_{is_intranode} {
  // EP 8 of internode_ll also need atomic_buffer_ptr

  Proxy::Config cfg{};
  thread_idx_ = thread_idx;
  gpu_buffer_addr_ = reinterpret_cast<void*>(gpu_buffer_addr);

  cfg.d2h_queues.reserve(kChannelPerProxy);
  d2h_queues.reserve(kChannelPerProxy);
  for (size_t i = 0; i < kChannelPerProxy; ++i) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    auto fifo = std::make_unique<mscclpp::Fifo>(kQueueSize);
    uintptr_t addr = reinterpret_cast<uintptr_t>(fifo.get());
    fifos.push_back(std::move(fifo));
#else
    uintptr_t addr = alloc_cmd_ring();
#endif
    d2hq::init_from_addr(d2h_queues[i], addr);
    cfg.d2h_queues.push_back(d2h_queues[i]);
    d2h_channel_addrs_.push_back(addr);
  }

  cfg.thread_idx = thread_idx;
  cfg.gpu_buffer = reinterpret_cast<void*>(gpu_buffer_addr);
  cfg.total_size = total_size;
  cfg.rank = rank;
  cfg.node_idx = node_idx;
  cfg.local_rank = local_rank;
  cfg.num_experts = num_experts;
  cfg.num_ranks = num_ranks;
  cfg.num_nodes = num_nodes;
  cfg.use_normal_mode = use_normal_mode;
  cfg.is_intranode = is_intranode;
  proxy_ = std::make_unique<Proxy>(cfg);
  local_rank_ = local_rank;
  node_idx_ = node_idx;

  if (thread_idx == 0) {
#ifdef USE_GRACE_HOPPER
    cudaMallocManaged(&atomic_buffer_ptr_, kAtomicBufferSize);
#else
    cudaHostAlloc(&atomic_buffer_ptr_, kAtomicBufferSize,
                  cudaHostAllocMapped | cudaHostAllocWriteCombined);
#endif
    cudaMemset(atomic_buffer_ptr_, 0, kAtomicBufferSize);
    proxy_->set_atomic_buffer_ptr(atomic_buffer_ptr_);
  }
}

UcclProxy::~UcclProxy() {
  try {
    stop();
  } catch (...) {
  }

  // Free all allocated ring buffers
#ifndef USE_MSCCLPP_FIFO_BACKEND
  for (auto d2h_channel_addr : d2h_channel_addrs_) {
    free_cmd_ring(d2h_channel_addr);
  }
#endif
  d2h_channel_addrs_.clear();
}

std::vector<uint64_t> UcclProxy::get_d2h_channel_addrs() const {
  std::vector<uint64_t> addrs;
  addrs.reserve(d2h_channel_addrs_.size());
  for (auto addr : d2h_channel_addrs_) {
    addrs.push_back(static_cast<uint64_t>(addr));
  }
  return addrs;
}

void UcclProxy::set_peers_meta(std::vector<PeerMeta> const& peers) {
  peers_ = peers;
  proxy_->set_peers_meta(peers);
}

void UcclProxy::start_sender() { start(Mode::Sender); }
void UcclProxy::start_remote() { start(Mode::Remote); }
void UcclProxy::start_local() { start(Mode::Local); }
void UcclProxy::start_dual() { start(Mode::Dual); }

void UcclProxy::stop() {
  if (!running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Proxy already stopped");
  }
  proxy_->set_progress_run(false);
  if (thread_.joinable()) thread_.join();
  running_.store(false, std::memory_order_release);
  // Because proxies share the gpu_buffer, only destroy gpu_buffer for the first
  // proxy.
  proxy_->destroy(thread_idx_ == 0);
}

void UcclProxy::start(Mode m) {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Proxy already running");
  }
  mode_ = m;
  proxy_->set_progress_run(true);
  running_.store(true, std::memory_order_release);

  thread_ = std::thread([this]() {
    if (is_intranode_) {
      std::printf("UcclProxy: no peer IP set, running in local mode\n");
      proxy_->run_local();
      return;
    }
    switch (mode_) {
      case Mode::Sender:
        proxy_->run_sender();
        break;
      case Mode::Remote:
        proxy_->run_remote();
        break;
      case Mode::Local:
        proxy_->run_local();
        break;
      case Mode::Dual:
        proxy_->run_dual();
        break;
      default:
        break;
    }
  });
}

// ============================================================================
// FifoProxy Implementation
// ============================================================================

FifoProxy::FifoProxy(int thread_idx, uintptr_t gpu_buffer_addr,
                     size_t total_size, int rank, int node_idx, int local_rank,
                     bool is_intranode)
    : thread_idx(thread_idx),
      fifo_(nullptr),
      stop_flag_(false),
      gpu_buffer_addr_(gpu_buffer_addr),
      total_size_(total_size),
      rank_(rank),
      node_idx_(node_idx),
      local_rank_(local_rank),
      is_intranode_(is_intranode) {}

FifoProxy::~FifoProxy() { stop(); }

void FifoProxy::set_fifo(mscclpp::Fifo* fifo) { fifo_ = fifo; }

void FifoProxy::set_peers_meta(std::vector<PeerMeta> const& meta) {
  // Create Proxy::Config
  // Note: we don't pass ring_buffers since we're using FIFO
  Proxy::Config cfg;
  cfg.thread_idx = thread_idx;
  assert(gpu_buffer_addr_ != 0 && "GPU buffer address is null");
  assert(total_size_ != 0 && "Total size is 0");
  cfg.gpu_buffer = reinterpret_cast<void*>(gpu_buffer_addr_);
  cfg.total_size = total_size_;
  cfg.rank = rank_;
  cfg.node_idx = node_idx_;
  cfg.local_rank = local_rank_;
  cfg.is_intranode = is_intranode_;
  cfg.pin_thread = true;

  // Set RDMA parameters (for 2-node benchmarking)
  cfg.num_experts = 0;
  cfg.num_ranks = 2;
  cfg.num_nodes = 2;

  // Create the underlying Proxy
  proxy_ = std::make_unique<Proxy>(cfg);
  proxy_->set_peers_meta(meta);
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
    local_poll_completions(proxy_->ctx_, proxy_->acked_wrs_, thread_idx,
                           proxy_->ctx_by_tag_);

    // Process completed work requests (similar to notify_gpu_completion)
    while (fifo_tail_acked < fifo_head_seen &&
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
    cmd.cmd_type = make_cmd_type(CmdType::WRITE, false, 0);
    cmd.dst_rank = 1;
    cmd.bytes = kObjectSize;
    cmd.expert_idx = 0;
    cmd.req_rptr = 0;
    cmd.req_lptr = 0;
    cmd.value = 0;

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
    local_poll_completions(proxy_->ctx_, proxy_->acked_wrs_, thread_idx,
                           proxy_->ctx_by_tag_);

    while (fifo_tail_acked < fifo_head_seen &&
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
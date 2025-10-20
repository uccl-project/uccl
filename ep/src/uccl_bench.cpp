#include "uccl_bench.hpp"
#include "bench_kernel.cuh"     // For kernel launchers
#include "proxy.hpp"          // Proxy, make_cfg
#include <stdexcept>

Bench::Bench()
    : running_{false}, have_t0_{false}, have_t1_{false}, done_evt_(nullptr) {
  init_env(env_);
  GPU_RT_CHECK(cudaEventCreateWithFlags(&done_evt_, cudaEventDisableTiming));
}

Bench::~Bench() {
  try {
    join_proxies();
  } catch (...) {
  }
  if (done_evt_) {
    cudaEventDestroy(done_evt_);
    done_evt_ = nullptr;
  }
  destroy_env(env_);
}

EnvInfo Bench::env_info() const {
  EnvInfo e;
  e.blocks = env_.blocks;
  e.queue_size = kQueueSize;
  e.threads_per_block = kNumThPerBlock;
  e.iterations = kIterations;
  e.stream_addr = reinterpret_cast<uintptr_t>(env_.stream);
  e.rbs_addr = reinterpret_cast<uintptr_t>(env_.rbs);
  return e;
}

int Bench::blocks() const { return env_.blocks; }
int Bench::num_proxies() const { return env_.blocks; }
bool Bench::is_running() const {
  return running_.load(std::memory_order_acquire);
}

uintptr_t Bench::ring_addr(int i) const {
  if (i < 0 || i >= env_.blocks) throw std::out_of_range("ring index");
  return reinterpret_cast<uintptr_t>(&env_.rbs[i]);
}

void Bench::timing_start() {
  t0_ = std::chrono::high_resolution_clock::now();
  have_t0_ = true;
}
void Bench::timing_stop() {
  t1_ = std::chrono::high_resolution_clock::now();
  have_t1_ = true;
}

void Bench::start_local_proxies(int rank, std::string const& peer_ip) {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Proxies already running");
  }
  threads_.reserve(env_.blocks);
  for (int i = 0; i < env_.blocks; ++i) {
    threads_.emplace_back([this, i, rank, peer_ip]() {
      Proxy p{
          make_cfg(env_, i, rank, peer_ip.empty() ? nullptr : peer_ip.c_str())};
      p.run_local();
    });
  }
  running_.store(true, std::memory_order_release);
}

void Bench::launch_gpu_issue_batched_commands() {
  timing_start();
  const size_t shmem_bytes = kQueueSize * 2 * sizeof(unsigned long long);
  auto st = launch_gpu_issue_batched_commands_shim(
      env_.blocks, kNumThPerBlock, shmem_bytes, env_.stream, env_.rbs);
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string("kernel launch failed: ") +
                             cudaGetErrorString(st));
  }
  GPU_RT_CHECK(cudaEventRecord(done_evt_, env_.stream));
}

void Bench::sync_stream() {
  auto st = cudaStreamSynchronize(env_.stream);
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                             cudaGetErrorString(st));
  }
  timing_stop();
}

void Bench::sync_stream_interruptible(
    int poll_ms, long long timeout_ms,
    std::function<bool()> const& should_abort) {
  auto start = std::chrono::steady_clock::now();
  while (true) {
    cudaError_t st = cudaEventQuery(done_evt_);
    if (st == cudaSuccess) break;
    if (st != cudaErrorNotReady) {
      (void)cudaGetLastError();
      throw std::runtime_error(std::string("cudaEventQuery failed: ") +
                               cudaGetErrorString(st));
    }

    if (should_abort && should_abort()) {
      throw std::runtime_error("aborted");
    }

    if (timeout_ms >= 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
      if (elapsed.count() >= timeout_ms) {
        throw std::runtime_error("Stream sync timed out");
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
  }
  timing_stop();
}

void Bench::join_proxies() {
  for (auto& t : threads_)
    if (t.joinable()) t.join();
  threads_.clear();
  running_.store(false, std::memory_order_release);
}

void Bench::print_block_latencies() { ::print_block_latencies(env_); }

Stats Bench::compute_stats() const {
  if (!have_t0_ || !have_t1_) {
    throw std::runtime_error(
        "compute_stats: missing t0/t1. Call launch_* then sync_stream() "
        "first.");
  }
  return ::compute_stats(env_, t0_, t1_);
}

void Bench::print_summary(Stats const& s) const { ::print_summary(env_, s); }
void Bench::print_summary_last() const {
  ::print_summary(env_, compute_stats());
}

double Bench::last_elapsed_ms() const {
  if (!have_t0_ || !have_t1_) return 0.0;
  return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
}

// ============================================================================
// BenchFifo Implementation
// ============================================================================

BenchFifo::BenchFifo()
    : running_{false}, have_t0_{false}, have_t1_{false}, done_evt_(nullptr) {
  init_env_fifo(env_);
  GPU_RT_CHECK(cudaEventCreateWithFlags(&done_evt_, cudaEventDisableTiming));
}

BenchFifo::~BenchFifo() {
  try {
    join_proxies();
  } catch (...) {
  }
  if (done_evt_) {
    cudaEventDestroy(done_evt_);
    done_evt_ = nullptr;
  }
  destroy_env_fifo(env_);
}

EnvInfo BenchFifo::env_info() const {
  EnvInfo e;
  e.blocks = env_.blocks;
  e.queue_size = kQueueSize;  // Default FIFO size
  e.threads_per_block = kNumThPerBlock;
  e.iterations = kIterations;
  e.stream_addr = reinterpret_cast<uintptr_t>(env_.stream);
  e.rbs_addr = reinterpret_cast<uintptr_t>(env_.d_fifo_handles);
  return e;
}

int BenchFifo::blocks() const { return env_.blocks; }
int BenchFifo::num_proxies() const { return env_.blocks; }
bool BenchFifo::is_running() const {
  return running_.load(std::memory_order_acquire);
}

mscclpp::Fifo* BenchFifo::get_fifo(int i) const { return ::get_fifo(env_, i); }

void BenchFifo::timing_start() {
  t0_ = std::chrono::high_resolution_clock::now();
  have_t0_ = true;
}

void BenchFifo::timing_stop() {
  t1_ = std::chrono::high_resolution_clock::now();
  have_t1_ = true;
}

void BenchFifo::launch_gpu_issue_batched_commands() {
  timing_start();
  // Shared memory for circular buffer, sized to kQueueSize (not kIterations!)
  const size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
  auto st = launch_gpu_issue_batched_commands_fifo(
      env_.blocks, kNumThPerBlock, shmem_bytes, env_.stream, env_.d_fifo_handles
#ifdef MEASURE_PER_OP_LATENCY
      ,
      env_.cycle_start, env_.cycle_end, env_.cycle_accum, env_.op_count
#endif
  );
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string("kernel launch failed: ") +
                             cudaGetErrorString(st));
  }
  GPU_RT_CHECK(cudaEventRecord(done_evt_, env_.stream));
}

void BenchFifo::sync_stream() {
  auto st = cudaStreamSynchronize(env_.stream);
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                             cudaGetErrorString(st));
  }
  timing_stop();
}

void BenchFifo::sync_stream_interruptible(
    int poll_ms, long long timeout_ms,
    std::function<bool()> const& should_abort) {
  auto start = std::chrono::steady_clock::now();
  while (true) {
    cudaError_t st = cudaEventQuery(done_evt_);
    if (st == cudaSuccess) break;
    if (st != cudaErrorNotReady) {
      (void)cudaGetLastError();
      throw std::runtime_error(std::string("cudaEventQuery failed: ") +
                               cudaGetErrorString(st));
    }

    if (should_abort && should_abort()) {
      throw std::runtime_error("aborted");
    }

    if (timeout_ms >= 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
      if (elapsed.count() >= timeout_ms) {
        throw std::runtime_error("Stream sync timed out");
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
  }
  timing_stop();
}

void BenchFifo::join_proxies() {
  for (auto& t : threads_)
    if (t.joinable()) t.join();
  threads_.clear();
  running_.store(false, std::memory_order_release);
}

void BenchFifo::print_block_latencies() { ::print_block_latencies_fifo(env_); }

Stats BenchFifo::compute_stats() const {
  if (!have_t0_ || !have_t1_) {
    throw std::runtime_error(
        "compute_stats: missing t0/t1. Call launch_* then sync_stream() "
        "first.");
  }
  return ::compute_stats_fifo(env_, t0_, t1_);
}

void BenchFifo::print_summary(Stats const& s) const {
  ::print_summary_fifo(env_, s);
}
void BenchFifo::print_summary_last() const {
  ::print_summary_fifo(env_, compute_stats());
}

double BenchFifo::last_elapsed_ms() const {
  if (!have_t0_ || !have_t1_) return 0.0;
  return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
}
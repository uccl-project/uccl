#include "cpu_executor.hpp"
#include "timeline.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <immintrin.h>
#include <sched.h>
namespace uccl_lite::cpu_router {
struct CpuTaskState {
  std::atomic<uint64_t> completed{0};
  std::mutex mutex;
  std::condition_variable credit;
  std::exception_ptr error;
  uint64_t failed = 0;
};
bool CpuCompletion::ready() const {
  return !state_ ||
         state_->completed.load(std::memory_order_acquire) >= ticket_;
}
void CpuCompletion::wait() const {
  if (!state_) return;
  std::unique_lock<std::mutex> lock(state_->mutex);
  state_->credit.wait(lock, [this] { return ready(); });
  if (state_->failed && ticket_ >= state_->failed)
    std::rethrow_exception(state_->error);
}
struct CpuExecutor::Impl {
  struct Task {
    Callback run = nullptr;
    void* context = nullptr;
    std::vector<MemorySpan> sources;
    size_t count = 0;
    MemorySpan destination;
  };
  struct Lane {
    std::shared_ptr<CpuTaskState> state = std::make_shared<CpuTaskState>();
    std::vector<Task> queue;
    std::atomic<uint64_t> tail{0};
    std::atomic<bool> sleeping{false};
    std::condition_variable available;
    std::thread thread;
  };
  std::vector<std::unique_ptr<Lane>> lanes;
  std::atomic<bool> stopping{false};
  size_t depth;
  Impl(std::vector<int> const& cpus, size_t d, uint32_t spinMicros) : depth(d) {
    if (cpus.empty() || !depth)
      throw std::invalid_argument("empty CPU executor");
    for (int cpu : cpus) {
      if (cpu < 0 || cpu >= CPU_SETSIZE)
        throw std::invalid_argument("invalid executor CPU");
      auto l = std::make_unique<Lane>();
      l->queue.resize(depth);
      lanes.push_back(std::move(l));
    }
    for (size_t i = 0; i < lanes.size(); ++i) {
      lanes[i]->thread = std::thread([this, i, cpu = cpus[i], spinMicros] {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
        auto& lane = *lanes[i];
        uint64_t head = 0;
        while (true) {
          if (head) {
            auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::microseconds(spinMicros);
            while (lane.tail.load(std::memory_order_acquire) == head &&
                   !stopping.load() &&
                   std::chrono::steady_clock::now() < deadline)
              _mm_pause();
          }
          Task task;
          {
            std::unique_lock<std::mutex> lock(lane.state->mutex);
            lane.sleeping.store(true, std::memory_order_relaxed);
            lane.available.wait(lock, [&] {
              return stopping.load() ||
                     lane.tail.load(std::memory_order_acquire) > head;
            });
            lane.sleeping.store(false, std::memory_order_relaxed);
            if (stopping.load() && lane.tail.load() == head) return;
            task = lane.queue[head % depth];
          }
          TimelineRange taskTrace("cpu.executor.task", head + 1);
          std::exception_ptr error;
          try {
            if (task.run)
              task.run(task.context, int(i));
            else
              CpuExecutor::sumBf16(task.sources.data(), task.count,
                                   task.destination);
          } catch (...) {
            error = std::current_exception();
          }
          {
            std::lock_guard<std::mutex> lock(lane.state->mutex);
            if (error && !lane.state->error) {
              lane.state->error = error;
              lane.state->failed = head + 1;
            }
            lane.state->completed.store(++head, std::memory_order_release);
          }
          lane.state->credit.notify_all();
        }
      });
    }
  }
  void stop() {
    stopping.store(true);
    for (auto& l : lanes) {
      std::lock_guard<std::mutex> lock(l->state->mutex);
      l->available.notify_all();
      l->state->credit.notify_all();
    }
    for (auto& l : lanes)
      if (l->thread.joinable()) l->thread.join();
  }
  ~Impl() { stop(); }
};
CpuExecutor::CpuExecutor(std::vector<int> cpus, size_t depth,
                         uint32_t spinMicros)
    : impl_(std::make_unique<Impl>(cpus, depth, spinMicros)) {}
CpuExecutor::~CpuExecutor() = default;
void CpuExecutor::stop() { impl_->stop(); }
CpuCompletion CpuExecutor::enqueue(size_t index, Callback fn, void* context) {
  if (index >= impl_->lanes.size() || !fn)
    throw std::invalid_argument("invalid CPU task");
  auto& lane = *impl_->lanes[index];
  std::unique_lock<std::mutex> lock(lane.state->mutex);
  uint64_t tail = lane.tail.load();
  lane.state->credit.wait(lock, [&] {
    tail = lane.tail.load(std::memory_order_relaxed);
    return tail - lane.state->completed.load(std::memory_order_acquire) <
               impl_->depth ||
           impl_->stopping.load();
  });
  if (impl_->stopping.load()) throw std::runtime_error("CPU executor stopped");
  lane.queue[tail % impl_->depth] = {fn, context};
  lane.tail.store(tail + 1, std::memory_order_release);
  bool wake = lane.sleeping.load(std::memory_order_relaxed);
  lock.unlock();
  if (wake) lane.available.notify_one();
  CpuCompletion done;
  done.state_ = lane.state;
  done.ticket_ = tail + 1;
  return done;
}
void CpuExecutor::enqueueBatch(size_t const* indices, size_t count, Callback fn,
                               void* context) {
  if (!fn || (count && !indices))
    throw std::invalid_argument("invalid CPU batch");
  for (size_t i = 0; i < count; ++i)
    if (indices[i] >= impl_->lanes.size() ||
        (i && indices[i] <= indices[i - 1]))
      throw std::invalid_argument("CPU batch lanes must be increasing");
  for (size_t i = 0; i < count; ++i) {
    auto& lane = *impl_->lanes[indices[i]];
    std::unique_lock<std::mutex> lock(lane.state->mutex);
    auto tail = lane.tail.load(std::memory_order_relaxed);
    lane.state->credit.wait(lock, [&] {
      tail = lane.tail.load(std::memory_order_relaxed);
      return tail - lane.state->completed.load(std::memory_order_acquire) <
                 impl_->depth ||
             impl_->stopping.load();
    });
    if (impl_->stopping.load())
      throw std::runtime_error("CPU executor stopped");
    lane.queue[tail % impl_->depth] = {fn, context};
    lane.tail.store(tail + 1, std::memory_order_release);
  }
  for (size_t i = 0; i < count; ++i) {
    auto& lane = *impl_->lanes[indices[i]];
    if (lane.sleeping.load(std::memory_order_relaxed))
      lane.available.notify_one();
  }
}
CpuCompletion CpuExecutor::enqueueSum(size_t index, MemorySpan const* sources,
                                      size_t count, MemorySpan destination) {
  if (index >= impl_->lanes.size() || (count && !sources))
    throw std::invalid_argument("invalid reduction task");
  auto& lane = *impl_->lanes[index];
  std::unique_lock<std::mutex> lock(lane.state->mutex);
  auto tail = lane.tail.load();
  lane.state->credit.wait(lock, [&] {
    tail = lane.tail.load(std::memory_order_relaxed);
    return tail - lane.state->completed.load(std::memory_order_acquire) <
               impl_->depth ||
           impl_->stopping.load();
  });
  if (impl_->stopping.load()) throw std::runtime_error("CPU executor stopped");
  Impl::Task task;
  task.count = count;
  task.sources.resize(count);
  task.destination = destination;
  for (size_t i = 0; i < count; ++i) task.sources[i] = sources[i];
  lane.queue[tail % impl_->depth] = std::move(task);
  lane.tail.store(tail + 1, std::memory_order_release);
  bool wake = lane.sleeping.load(std::memory_order_relaxed);
  lock.unlock();
  if (wake) lane.available.notify_one();
  CpuCompletion done;
  done.state_ = lane.state;
  done.ticket_ = tail + 1;
  return done;
}
CpuCompletion CpuExecutor::enqueueMerge(size_t lane, MemorySpan local,
                                        MemorySpan remote,
                                        MemorySpan destination) {
  MemorySpan inputs[] = {local, remote};
  return enqueueSum(lane, inputs, 2, destination);
}
static float decode(uint16_t v) {
  uint32_t bits = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
static uint16_t encode(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  if ((b & 0x7fffffffU) > 0x7f800000U)
    b |= 0x00400000U;
  else
    b += 0x7fffU + ((b >> 16) & 1U);
  return b >> 16;
}
__attribute__((target("avx512f,avx512bw"))) static __m512 load(
    uint16_t const* p) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(
          _mm256_loadu_si256(reinterpret_cast<__m256i const*>(p))),
      16));
}
__attribute__((target("avx512f,avx512bw"))) static void store(uint16_t* p,
                                                              __m512 v) {
  __m512i b = _mm512_castps_si512(v);
  auto bias = _mm512_add_epi32(
      _mm512_set1_epi32(0x7fff),
      _mm512_and_si512(_mm512_srli_epi32(b, 16), _mm512_set1_epi32(1)));
  auto rounded = _mm512_add_epi32(b, bias);
  auto nan =
      _mm512_cmp_epu32_mask(_mm512_and_si512(b, _mm512_set1_epi32(0x7fffffff)),
                            _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
  rounded = _mm512_mask_mov_epi32(
      rounded, nan, _mm512_or_si512(b, _mm512_set1_epi32(0x00400000)));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p),
                      _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16)));
}
// Preserve the generic vector loop for every overlapping span, including
// descriptor storage, because the unrolled path reads ahead of its stores.
static bool disjoint(void const *a, size_t aBytes, void const *b,
                     size_t bBytes) {
  uintptr_t x = reinterpret_cast<uintptr_t>(a);
  uintptr_t y = reinterpret_cast<uintptr_t>(b);
  return x <= y ? y - x >= aBytes : x - y >= bBytes;
}
__attribute__((target("avx512f,avx512bw"))) static size_t
sumTwoWide(MemorySpan const *src, uint16_t *out, size_t n) {
  auto const *a = static_cast<uint16_t const *>(src[0].address);
  auto const *b = static_cast<uint16_t const *>(src[1].address);
  size_t i = 0;
  for (; n - i >= 64; i += 64) {
    __m512 v0 = _mm512_add_ps(_mm512_setzero_ps(), load(a + i));
    __m512 v1 = _mm512_add_ps(_mm512_setzero_ps(), load(a + i + 16));
    __m512 v2 = _mm512_add_ps(_mm512_setzero_ps(), load(a + i + 32));
    __m512 v3 = _mm512_add_ps(_mm512_setzero_ps(), load(a + i + 48));
    store(out + i, _mm512_add_ps(v0, load(b + i)));
    store(out + i + 16, _mm512_add_ps(v1, load(b + i + 16)));
    store(out + i + 32, _mm512_add_ps(v2, load(b + i + 32)));
    store(out + i + 48, _mm512_add_ps(v3, load(b + i + 48)));
  }
  for (; n - i >= 16; i += 16)
    store(out + i,
          _mm512_add_ps(_mm512_add_ps(_mm512_setzero_ps(), load(a + i)),
                        load(b + i)));
  return i;
}
__attribute__((target("avx512f,avx512bw"))) static size_t sumWide(
    MemorySpan const* src, size_t count, uint16_t* out, size_t n) {
  if (count == 2 && n >= 64 && disjoint(out, n * 2, src[0].address, n * 2) &&
      disjoint(out, n * 2, src[1].address, n * 2) &&
      disjoint(out, n * 2, src, 2 * sizeof(MemorySpan)))
    return sumTwoWide(src, out, n);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_setzero_ps();
    for (size_t s = 0; s < count; ++s)
      v = _mm512_add_ps(v,
                        load(static_cast<uint16_t const*>(src[s].address) + i));
    store(out + i, v);
  }
  return i;
}
__attribute__((target("avx2"))) static size_t sumAvx2(MemorySpan const* src,
                                                      size_t count,
                                                      uint16_t* out, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_setzero_ps();
    for (size_t s = 0; s < count; ++s) {
      auto packed = _mm_loadu_si128(reinterpret_cast<__m128i const*>(
          static_cast<uint16_t const*>(src[s].address) + i));
      v = _mm256_add_ps(v, _mm256_castsi256_ps(_mm256_slli_epi32(
                               _mm256_cvtepu16_epi32(packed), 16)));
    }
    auto bits = _mm256_castps_si256(v);
    auto bias = _mm256_add_epi32(
        _mm256_set1_epi32(0x7fff),
        _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
    auto nan = _mm256_cmpgt_epi32(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)),
        _mm256_set1_epi32(0x7f800000));
    auto rounded = _mm256_blendv_epi8(
        _mm256_add_epi32(bits, bias),
        _mm256_or_si256(bits, _mm256_set1_epi32(0x00400000)), nan);
    auto hi = _mm256_srli_epi32(rounded, 16);
    auto packed = _mm_packus_epi32(_mm256_castsi256_si128(hi),
                                   _mm256_extracti128_si256(hi, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), packed);
  }
  return i;
}
void CpuExecutor::sumBf16(MemorySpan const* src, size_t count, MemorySpan dst) {
  if (dst.kind == MemoryKind::Gpu || dst.bytes % 2 || !dst.address)
    throw std::invalid_argument("invalid BF16 output");
  for (size_t s = 0; s < count; ++s)
    if (src[s].kind == MemoryKind::Gpu || src[s].bytes < dst.bytes ||
        !src[s].address)
      throw std::invalid_argument("invalid BF16 source");
  auto* out = static_cast<uint16_t*>(dst.address);
  size_t n = dst.bytes / 2, i = 0;
  static bool const wide =
      __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
  i = wide ? sumWide(src, count, out, n) : sumAvx2(src, count, out, n);
  for (; i < n; ++i) {
    float v = 0;
    for (size_t s = 0; s < count; ++s)
      v += decode(static_cast<uint16_t const*>(src[s].address)[i]);
    out[i] = encode(v);
  }
}
void CpuExecutor::mergeBf16(MemorySpan a, MemorySpan b, MemorySpan out) {
  MemorySpan inputs[] = {a, b};
  sumBf16(inputs, 2, out);
}
}  // namespace uccl_lite::cpu_router

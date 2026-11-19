#pragma once
#include "host_packet_transport.h"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace uccl_lite::multinode {
struct HostPacketView {
  char* control[2];
};
__device__ inline uint64_t* packetFlag(HostPacketView v, int group, int slot,
                                       int field) {
  return reinterpret_cast<uint64_t*>(
      v.control[group] + (slot * SharedHostPackets::flagCount + field) * 64);
}
__device__ inline uint64_t packetLoad(uint64_t const* p) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(p)
               : "memory");
  return value;
}
__device__ inline void packetStore(uint64_t* p, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(value)
               : "memory");
}
__global__ void hostPacketWait(HostPacketView v, int rank, int slot,
                               uint64_t epoch, int phase) {
  if (threadIdx.x) return;
  int group = rank % 4 / 2;
  for (;;) {
    if (packetLoad(packetFlag(v, group, slot, 13))) asm volatile("trap;");
    bool ready = true;
    if (phase == 1) {
      for (int g = 0; g < 2; ++g)
        for (int r = 0; r < 2; ++r)
          ready &= packetLoad(packetFlag(v, g, slot, r)) == epoch;
    } else if (phase == 2) {
      for (int r = 0; r < 2; ++r)
        ready &= packetLoad(packetFlag(v, group, slot, 10 + r)) == epoch;
    } else {
      for (int r = 0; r < 4; ++r)
        ready &= packetLoad(packetFlag(v, group, slot, 2 + r)) >= epoch - 1;
      for (int r = 0; r < 2; ++r)
        ready &= packetLoad(packetFlag(v, group, slot, 6 + r)) >= epoch - 1;
      ready &= packetLoad(packetFlag(v, group, slot, 12)) >= epoch - 1;
    }
    if (ready) return;
    __nanosleep(64);
  }
}
__global__ void hostPacketPublish(HostPacketView v, int rank, int slot,
                                  uint64_t epoch, bool consume) {
  if (threadIdx.x) return;
  int group = rank % 4 / 2;
  if (consume) {
    for (int g = 0; g < 2; ++g)
      packetStore(packetFlag(v, g, slot, 2 + rank % 4), epoch);
    packetStore(packetFlag(v, group, slot, 6 + rank % 2), epoch);
  } else
    packetStore(packetFlag(v, group, slot, rank % 2), epoch);
}

// Explicit preparation; one GPU per process. The worker performs only host
// atomics and IB operations. All CUDA submissions stay on the caller thread.
class AsyncHostGather {
  struct Piece {
    int slot;
    size_t length;
    uint64_t epoch;
  };
  struct Work {
    std::vector<Piece> pieces;
  };
  int rank_;
  size_t chunk_;
  SharedHostPackets packets_;
  HostPacketTransport transport_;
  HostPacketView view_{};
  std::array<cudaStream_t, 3> lanes_{};
  std::array<cudaEvent_t, 3> joined_{};
  cudaEvent_t start_{}, done_{};
  std::array<uint64_t, 3> epochs_{};
  bool pending_ = false, stop_ = false, active_ = false;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Work> queue_;
  std::exception_ptr failure_;
  std::thread worker_;
  static void check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
  }
  void progress(Work const& work) {
    std::array<size_t, 3> next{0, 1, 2};
    std::array<int, 3> phase{};
    size_t complete = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (complete < work.pieces.size()) {
      bool advanced = false;
      for (int slot = 0; slot < 3; ++slot) {
        if (next[slot] >= work.pieces.size()) continue;
        auto const& p = work.pieces[next[slot]];
        if (phase[slot] == 0 && transport_.canReuse(slot, p.epoch)) {
          transport_.grant(slot, p.epoch);
          phase[slot] = 1;
          advanced = true;
        }
        if (phase[slot] == 1 && transport_.canSend(slot, p.epoch)) {
          transport_.send(slot, p.length, p.epoch);
          transport_.flush();
          phase[slot] = 2;
          advanced = true;
        }
        if (phase[slot] == 2 && transport_.canReuse(slot, p.epoch + 1)) {
          next[slot] += 3;
          phase[slot] = 0;
          ++complete;
          advanced = true;
        }
      }
      if (advanced)
        deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
      else {
        if (std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error("host gather progress timeout");
        std::this_thread::yield();
      }
    }
  }
  void worker() {
    try {
      for (;;) {
        Work work;
        {
          std::unique_lock lock(mutex_);
          changed_.wait(lock, [&] { return stop_ || !queue_.empty(); });
          if (stop_ && queue_.empty()) return;
          work = std::move(queue_.front());
          queue_.pop_front();
          active_ = true;
        }
        progress(work);
        {
          std::lock_guard lock(mutex_);
          active_ = false;
        }
        changed_.notify_all();
      }
    } catch (...) {
      for (int slot = 0; slot < 3; ++slot)
        __atomic_store_n(packets_.flag(rank_ % 4 / 2, slot, 13), uint64_t(1),
                         __ATOMIC_RELEASE);
      {
        std::lock_guard lock(mutex_);
        failure_ = std::current_exception();
        active_ = false;
      }
      changed_.notify_all();
    }
  }

 public:
  AsyncHostGather(mscclpp::Communicator& comm, int rank,
                  size_t chunk = 256UL << 10)
      : rank_(rank),
        chunk_(chunk),
        packets_(comm, rank, chunk),
        transport_(comm, packets_, rank) {
    for (int g = 0; g < 2; ++g) view_.control[g] = packets_.mapped(g);
    check(cudaEventCreateWithFlags(&start_, cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    for (int i = 0; i < 3; ++i) {
      check(cudaStreamCreateWithFlags(&lanes_[i], cudaStreamNonBlocking));
      check(cudaEventCreateWithFlags(&joined_[i], cudaEventDisableTiming));
    }
    if (!(rank % 2)) worker_ = std::thread([this] { worker(); });
  }
  ~AsyncHostGather() {
    synchronize();
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    changed_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (int i = 0; i < 3; ++i) {
      cudaEventDestroy(joined_[i]);
      cudaStreamDestroy(lanes_[i]);
    }
    cudaEventDestroy(done_);
    cudaEventDestroy(start_);
  }
  AsyncHostGather(AsyncHostGather const&) = delete;
  AsyncHostGather& operator=(AsyncHostGather const&) = delete;
  void synchronize() {
    if (pending_) check(cudaEventSynchronize(done_));
    if (worker_.joinable()) {
      std::unique_lock lock(mutex_);
      changed_.wait(lock,
                    [&] { return failure_ || (queue_.empty() && !active_); });
      if (failure_) std::rethrow_exception(failure_);
    }
  }
  void enqueue(void const* input, void* output, size_t shardBytes,
               cudaStream_t stream) {
    if (!input || !output || !shardBytes || shardBytes % 16 ||
        shardBytes > (1UL << 27))
      throw std::invalid_argument("host gather extent/pointers");
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("host gather caller capture unsupported");
    if (pending_) check(cudaStreamWaitEvent(stream, done_, 0));
    Work work;
    for (size_t offset = 0, p = 0; offset < shardBytes; offset += chunk_, ++p) {
      int slot = p % 3;
      if (epochs_[slot] == UINT64_MAX)
        throw std::overflow_error("host packet epoch exhausted");
      work.pieces.push_back(
          {slot, std::min(chunk_, shardBytes - offset), ++epochs_[slot]});
    }
    if (!(rank_ % 2)) {
      std::lock_guard lock(mutex_);
      if (failure_) std::rethrow_exception(failure_);
      queue_.push_back(work);
      changed_.notify_all();
    }
    auto* out = static_cast<char*>(output);
    auto const* in = static_cast<char const*>(input);
    if (out + rank_ * shardBytes != in)
      check(cudaMemcpyAsync(out + rank_ * shardBytes, in, shardBytes,
                            cudaMemcpyDeviceToDevice, stream));
    check(cudaEventRecord(start_, stream));
    for (auto lane : lanes_) check(cudaStreamWaitEvent(lane, start_, 0));
    for (size_t p = 0, offset = 0; p < work.pieces.size();
         offset += chunk_, ++p) {
      auto const& piece = work.pieces[p];
      auto lane = lanes_[piece.slot];
      hostPacketWait<<<1, 32, 0, lane>>>(view_, rank_, piece.slot, piece.epoch,
                                         false);
      check(cudaGetLastError());
      check(cudaMemcpyAsync(packets_.row(rank_, piece.slot, piece.length),
                            in + offset, piece.length, cudaMemcpyDeviceToHost,
                            lane));
      hostPacketPublish<<<1, 32, 0, lane>>>(view_, rank_, piece.slot,
                                            piece.epoch, false);
      for (int phase = 1; phase <= 2; ++phase) {
        hostPacketWait<<<1, 32, 0, lane>>>(view_, rank_, piece.slot,
                                           piece.epoch, phase);
        check(cudaGetLastError());
        int first = (phase == 1 ? rank_ / 4 : (rank_ / 4) ^ 1) * 4;
        int last = first + 4;
        while (first < last) {
          if (first == rank_) {
            ++first;
            continue;
          }
          int end = phase == 1 ? (first / 2 + 1) * 2 : last;
          if (rank_ > first && rank_ < end) end = rank_;
          check(cudaMemcpy2DAsync(
              out + size_t(first) * shardBytes + offset, shardBytes,
              packets_.row(first, piece.slot, piece.length), piece.length,
              piece.length, end - first, cudaMemcpyHostToDevice, lane));
          first = end;
        }
      }
      hostPacketPublish<<<1, 32, 0, lane>>>(view_, rank_, piece.slot,
                                            piece.epoch, true);
      check(cudaGetLastError());
    }
    for (int i = 0; i < 3; ++i) {
      check(cudaEventRecord(joined_[i], lanes_[i]));
      check(cudaStreamWaitEvent(stream, joined_[i], 0));
    }
    check(cudaEventRecord(done_, stream));
    pending_ = true;
  }
};
}  // namespace uccl_lite::multinode

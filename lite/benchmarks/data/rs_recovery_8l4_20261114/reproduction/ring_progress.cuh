#pragma once
#include "compact_transport.cuh"

namespace uccl_lite::compact_experiment {
struct RingFlags {
  Word progress;
};
struct RingBinding {
  __nv_bfloat16 const* input;
  __nv_bfloat16* output;
  __nv_bfloat16* host[2][kRanks];
  RingFlags* flags[kRanks];
  int rank, ranks;
};
__device__ inline uint32_t ringAdd(uint32_t a, uint32_t b) {
  auto lo =
      __float2bfloat16_rn(__uint_as_float(a << 16) + __uint_as_float(b << 16));
  auto hi = __float2bfloat16_rn(__uint_as_float(a & 0xffff0000U) +
                                __uint_as_float(b & 0xffff0000U));
  return uint32_t(__bfloat16_as_ushort(lo)) |
         (uint32_t(__bfloat16_as_ushort(hi)) << 16);
}
// A host-mediated ring: only adjacent ranks exchange partials. Each hop adds
// one local BF16 contribution and rounds to BF16. Reader credits protect both
// banks across changing extents and asynchronously queued calls.
__global__ void hostRingScatter(RingBinding b, size_t shard, uint32_t base) {
  auto grid = cooperative_groups::this_grid();
  int prev = (b.rank + b.ranks - 1) % b.ranks, next = (b.rank + 1) % b.ranks;
  // Previous-command completion releases both banks at our sole reader.
  if (grid.thread_rank() == 0 && base)
    while (acquire(&b.flags[next]->progress.value) < base) __nanosleep(64);
  grid.sync();
  copy(b.host[0][b.rank], b.input + size_t(prev) * shard, shard,
       grid.thread_rank(), grid.size());
  __threadfence_system();
  grid.sync();
  if (grid.thread_rank() == 0)
    release(&b.flags[b.rank]->progress.value, base + 1);
  for (int step = 0; step < b.ranks - 1; ++step) {
    int slot = step & 1, outputSlot = slot ^ 1;
    uint32_t seq = base + step + 1;
    bool final = step == b.ranks - 2;
    if (grid.thread_rank() == 0) {
      while (acquire(&b.flags[prev]->progress.value) < seq) __nanosleep(64);
      // Publishing stage step also acknowledges reading the preceding bank.
      // The spare bank for step zero was freed before this command started.
      if (!final && step)
        while (acquire(&b.flags[next]->progress.value) < seq) __nanosleep(64);
    }
    grid.sync();
    int chunk = (b.rank + b.ranks - step - 2) % b.ranks;
    auto const* local = b.input + size_t(chunk) * shard;
    auto const* remote = b.host[slot][prev];
    auto* out = final ? b.output : b.host[outputSlot][b.rank];
    if (shard % 8 == 0 && !((reinterpret_cast<uintptr_t>(b.input) |
                             reinterpret_cast<uintptr_t>(b.output)) &
                            15)) {
      for (size_t i = grid.thread_rank(); i < shard / 8; i += grid.size()) {
        auto a = load4(remote + i * 8), v = load4(local + i * 8);
        reinterpret_cast<uint4*>(out)[i] =
            make_uint4(ringAdd(a.x, v.x), ringAdd(a.y, v.y), ringAdd(a.z, v.z),
                       ringAdd(a.w, v.w));
      }
    } else {
      for (size_t i = grid.thread_rank(); i < shard; i += grid.size())
        out[i] = __float2bfloat16_rn(__bfloat162float(load1(remote + i)) +
                                     __bfloat162float(load1(local + i)));
    }
    __threadfence_system();
    grid.sync();
    if (grid.thread_rank() == 0) {
      release(&b.flags[b.rank]->progress.value, seq + 1);
    }
  }
}
class HostRing {
 public:
  HostRing(std::vector<PacketPort> ports, size_t fullBytes, int blocks)
      : ports_(std::move(ports)), capacity_(fullBytes), blocks_(blocks) {
    if (ports_.size() < 2 || ports_.size() > kRanks || !fullBytes ||
        fullBytes % (ports_.size() * 2) || blocks < 1 || blocks > 128)
      throw std::invalid_argument("invalid host ring geometry");
    size_t stride = ((fullBytes / ports_.size() + 15) / 16) * 16;
    for (size_t r = 0; r < ports_.size(); ++r) {
      check(cudaSetDevice(ports_[r].device));
      int node = gpuNumaNode(ports_[r].device);
      memory_[r] = RegisteredMemory::pinned(2 * stride, node);
      flags_[r] = RegisteredMemory::pinned(sizeof(RingFlags), node);
      std::memset(flags_[r].data(), 0, sizeof(RingFlags));
      int active, sms;
      check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, hostRingScatter, 256, 0));
      check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                   ports_[r].device));
      if (blocks_ > active * sms)
        throw std::invalid_argument("ring grid is not resident");
    }
    for (size_t r = 0; r < ports_.size(); ++r) {
      auto& b = bindings_[r];
      b.rank = int(r);
      b.ranks = int(ports_.size());
      for (size_t s = 0; s < ports_.size(); ++s) {
        auto* p =
            static_cast<char*>(memory_[s].span().gpuAddress(ports_[r].device));
        b.host[0][s] = reinterpret_cast<__nv_bfloat16*>(p);
        b.host[1][s] = reinterpret_cast<__nv_bfloat16*>(p + stride);
        b.flags[s] = static_cast<RingFlags*>(
            flags_[s].span().gpuAddress(ports_[r].device));
      }
    }
  }
  HostRing(HostRing const&) = delete;
  HostRing& operator=(HostRing const&) = delete;
  ~HostRing() {
    for (auto const& p : ports_) {
      cudaSetDevice(p.device);
      cudaStreamSynchronize(p.stream);
    }
  }
  void scatter(size_t count, std::vector<PacketPort> const& ports) {
    if (ports.size() != ports_.size() || !count || count % ports_.size() ||
        count > capacity_ / 2)
      throw std::invalid_argument("invalid ring extent");
    uint32_t steps = uint32_t(ports.size());
    if (base_ > UINT32_MAX - steps)
      throw std::overflow_error("ring sequence exhausted");
    for (size_t r = 0; r < ports.size(); ++r) {
      if (ports[r].device != ports_[r].device ||
          ports[r].stream != ports_[r].stream ||
          ports[r].input == ports[r].output)
        throw std::invalid_argument(
            "ring needs fixed streams and out-of-place buffers");
      check(cudaSetDevice(ports[r].device));
      cudaStreamCaptureStatus capture;
      check(cudaStreamIsCapturing(ports[r].stream, &capture));
      if (capture != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("prototype ring is eager-only");
    }
    size_t shard = count / ports.size();
    for (size_t r = 0; r < ports.size(); ++r) {
      check(cudaSetDevice(ports[r].device));
      auto b = bindings_[r];
      b.input = static_cast<__nv_bfloat16 const*>(ports[r].input);
      b.output = static_cast<__nv_bfloat16*>(ports[r].output);
      void* args[] = {&b, &shard, &base_};
      check(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(hostRingScatter), dim3(blocks_), dim3(256),
          args, 0, ports[r].stream));
    }
    base_ += steps;
  }

 private:
  std::vector<PacketPort> ports_;
  size_t capacity_;
  int blocks_;
  uint32_t base_ = 0;
  std::array<RegisteredMemory, kRanks> memory_, flags_;
  std::array<RingBinding, kRanks> bindings_{};
};
}  // namespace uccl_lite::compact_experiment

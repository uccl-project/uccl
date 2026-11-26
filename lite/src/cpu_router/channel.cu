#include "channel.hpp"
#include "channel_device.cuh"
#include <stdexcept>
namespace uccl_lite::cpu_router {
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void copySpan(void* dst, void const* src, size_t bytes) {
  DeviceChannel::put(dst, src, bytes, blockIdx.x * blockDim.x + threadIdx.x,
                     gridDim.x * blockDim.x);
}
struct ChannelCompletionState {
  cudaEvent_t event = nullptr;
  int device;
  MemorySpan source, destination;
  bool recorded = false;
  ~ChannelCompletionState() {
    if (event) {
      cudaSetDevice(device);
      cudaEventSynchronize(event);
      cudaEventDestroy(event);
    }
  }
};
Channel::Channel(ChannelOptions o) : options_(o) {
  if (o.blocks < 1 || o.threads < 1)
    throw std::invalid_argument("invalid channel resources");
  check(cudaSetDevice(o.device));
  for (size_t i = 0; i < o.completionSlots; ++i) {
    auto state = std::make_shared<ChannelCompletionState>();
    state->device = o.device;
    check(cudaEventCreateWithFlags(&state->event, cudaEventDisableTiming));
    pending_.push_back(std::move(state));
  }
}
Channel::~Channel() = default;
bool ChannelCompletion::ready() const {
  if (!state_) return true;
  check(cudaSetDevice(state_->device));
  auto e = cudaEventQuery(state_->event);
  if (e == cudaErrorNotReady) return false;
  check(e);
  return true;
}
void ChannelCompletion::wait() const {
  if (state_) {
    check(cudaSetDevice(state_->device));
    check(cudaEventSynchronize(state_->event));
  }
}
void ChannelCompletion::waitOn(int device, cudaStream_t stream) const {
  if (state_) {
    check(cudaSetDevice(device));
    check(cudaStreamWaitEvent(stream, state_->event, 0));
  }
}
void Channel::putAsync(MemorySpan dst, MemorySpan src,
                       cudaStream_t stream) const {
  if (dst.bytes < src.bytes || !src.address || !dst.address)
    throw std::invalid_argument("invalid channel spans");
  check(cudaSetDevice(options_.device));
  if (options_.backend == TransportBackend::Dma) {
    if (dst.kind == MemoryKind::Gpu && src.kind == MemoryKind::Gpu &&
        dst.device != src.device) {
      src.gpuAddress(dst.device);
      check(cudaSetDevice(options_.device));
      check(cudaMemcpyPeerAsync(dst.address, dst.device, src.address,
                                src.device, src.bytes, stream));
    } else
      check(cudaMemcpyAsync(dst.address, src.address, src.bytes,
                            cudaMemcpyDefault, stream));
  } else {
    void* d = dst.gpuAddress(options_.device);
    void const* s = src.gpuAddress(options_.device);
    copySpan<<<options_.blocks, options_.threads, 0, stream>>>(d, s, src.bytes);
    check(cudaGetLastError());
  }
}
void Channel::getAsync(MemorySpan d, MemorySpan s, cudaStream_t stream) const {
  putAsync(d, s, stream);
}
ChannelCompletion Channel::put(MemorySpan d, MemorySpan s,
                               cudaStream_t stream) {
  check(cudaSetDevice(options_.device));
  std::shared_ptr<ChannelCompletionState> state;
  for (auto& slot : pending_) {
    if (slot.use_count() != 1) continue;
    auto status = slot->recorded ? cudaEventQuery(slot->event) : cudaSuccess;
    if (status == cudaErrorNotReady) continue;
    check(status);
    state = slot;
    break;
  }
  if (!state) throw std::runtime_error("channel completion slots exhausted");
  state->source = s;
  state->destination = d;
  putAsync(d, s, stream);
  check(cudaEventRecord(state->event, stream));
  state->recorded = true;
  ChannelCompletion done;
  done.state_ = std::move(state);
  return done;
}
ChannelCompletion Channel::get(MemorySpan d, MemorySpan s,
                               cudaStream_t stream) {
  return put(d, s, stream);
}
}  // namespace uccl_lite::cpu_router

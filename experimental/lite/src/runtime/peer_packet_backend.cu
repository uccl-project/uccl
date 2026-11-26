#include "uccl_lite/bootstrap/control.hpp"
#include "uccl_lite/collectives/peer_packet.hpp"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include "uccl_lite/topology.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

namespace uccl_lite::runtime {
namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
struct RankOrder {
  int rank[4];
};
// Every execution shard maps bijectively to one logical shard. Integer loads
// preserve AG payload bits, including NaNs; source and destination are
// disjoint.
__global__ void permute(uint32_t const* input, uint32_t* output, size_t count,
                        RankOrder order, bool toExecution) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < 4 * count;
       i += size_t(gridDim.x) * blockDim.x) {
    size_t logical = size_t(order.rank[i / count]) * count + i % count;
    if (toExecution)
      output[i] = input[logical];
    else
      output[logical] = input[i];
  }
}
class Adapter : public CollectiveBackend {
  BackendContext context_;
  std::unique_ptr<PeerPacketCollectives> executor_;
  RankOrder order_{};
  bool remap_ = false, recorded_ = false;
  uint32_t *input_ = nullptr, *output_ = nullptr;
  cudaEvent_t done_ = nullptr;
  cudaStream_t last_ = nullptr;

 public:
  explicit Adapter(BackendContext const& c) : context_(c) {}
  ~Adapter() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(context_.device);
    if (recorded_) cudaEventSynchronize(done_);
    executor_.reset();
    if (input_) cudaFree(input_);
    if (output_) cudaFree(output_);
    if (done_) cudaEventDestroy(done_);
    if (previous >= 0 && previous != context_.device) cudaSetDevice(previous);
  }
  void prepare() override {
    if (initialized()) return;
    if (context_.ranks != 4 || context_.rank < 0 || context_.rank >= 4)
      throw std::invalid_argument(
          "peer packet requires four participating ranks");
    bootstrap::HostControl control("127.0.0.1", context_.id.data(),
                                   context_.id.size(), context_.rank, 4,
                                   120000);
    struct Wire {
      int valid = 1, device = -1;
      size_t capacity = 0;
      std::array<int, 4> devices{}, order{};
      std::array<int, 2> nodes{};
      std::array<std::array<char, 16>, 4> uuids{};
    } local;
    int executionRank = -1;
    try {
      if (context_.localDevices.size() != 4 ||
          context_.localDevices[context_.rank] != context_.device ||
          !context_.capacity || context_.capacity > (1UL << 30))
        throw std::invalid_argument("peer packet participant metadata invalid");
      check(cudaGetDevice(&local.device));
      if (local.device != context_.device)
        throw std::invalid_argument("peer packet current device mismatch");
      check(cudaDeviceSynchronize());
      auto topology = inspectTopology(context_.localDevices);
      Workload probe{Collective::AllReduce, ScalarType::Float32, 128, false, 4};
      if (!peerPacketEligible(topology, probe))
        throw std::invalid_argument(
            "peer packet topology/capability unsupported");
      auto order = pairedNumaRankOrder(topology);
      std::copy(order.begin(), order.end(), local.order.begin());
      std::copy(order.begin(), order.end(), order_.rank);
      std::copy(context_.localDevices.begin(), context_.localDevices.end(),
                local.devices.begin());
      local.nodes = {topology.numaNodes[order[0]],
                     topology.numaNodes[order[2]]};
      local.capacity = context_.capacity;
      for (int i = 0; i < 4; ++i) {
        if (order[i] == context_.rank) executionRank = i;
        remap_ |= order[i] != i;
        cudaDeviceProp property{};
        check(cudaGetDeviceProperties(&property, local.devices[i]));
        std::memcpy(local.uuids[i].data(), property.uuid.bytes, 16);
      }
    } catch (std::exception const& error) {
      local.valid = 0;
      std::fprintf(stderr, "peer packet placement rank=%d: %s\n", context_.rank,
                   error.what());
    }
    std::array<Wire, 4> votes;
    control.allGather(&local, votes.data(), sizeof(Wire));
    for (int rank = 0; rank < 4; ++rank) {
      auto const& v = votes[rank];
      if (!v.valid || !local.valid || v.device != local.devices[rank] ||
          v.devices != local.devices || v.order != local.order ||
          v.nodes != local.nodes || v.uuids != local.uuids ||
          v.capacity != local.capacity)
        throw std::invalid_argument("peer packet placement consensus failed");
    }
    int ready = 1;
    try {
      if (remap_) {
        check(cudaMalloc(&input_, context_.capacity));
        check(cudaMalloc(&output_, context_.capacity));
        check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
      }
    } catch (...) {
      ready = 0;
    }
    std::array<int, 4> allocations;
    control.allGather(&ready, allocations.data(), sizeof(int));
    if (std::find(allocations.begin(), allocations.end(), 0) !=
        allocations.end())
      throw std::runtime_error("peer packet permutation allocation failed");
    executor_ = std::make_unique<PeerPacketCollectives>(
        context_.id, executionRank, context_.device, context_.rail,
        context_.capacity, local.nodes);
    executor_->prepare();
    std::fprintf(stderr,
                 "UCCL Lite peer packet placement: rank=%d execution_rank=%d "
                 "order=%d,%d,%d,%d numa=%d,%d remap=%d\n",
                 context_.rank, executionRank, order_.rank[0], order_.rank[1],
                 order_.rank[2], order_.rank[3], local.nodes[0], local.nodes[1],
                 int(remap_));
  }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    if (!initialized()) throw std::logic_error("peer packet is not prepared");
    int operation = int(op), device;
    size_t scale = operation == 3 ? 4 : 16;
    if (operation < 1 || operation > 3 || !input || !output || !count ||
        count > context_.capacity / scale || (count * scale) % 128 ||
        (reinterpret_cast<uintptr_t>(input) % 4) ||
        (reinterpret_cast<uintptr_t>(output) % 4))
      throw std::invalid_argument("peer packet extent/alignment unsupported");
    check(cudaGetDevice(&device));
    if (device != context_.device)
      throw std::invalid_argument("peer packet current device mismatch");
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("peer packet external capture unsupported");
    if (!remap_) {
      executor_->enqueue(op, input, output, count, stream);
      return;
    }
    if (recorded_ && stream != last_)
      check(cudaStreamWaitEvent(stream, done_, 0));
    int blocks = int(std::min<size_t>((4 * count + 255) / 256, 256));
    if (op == NetworkOperation::ReduceScatter) {
      permute<<<blocks, 256, 0, stream>>>(static_cast<uint32_t const*>(input),
                                          input_, count, order_, true);
      check(cudaGetLastError());
      input = input_;
    }
    executor_->enqueue(op, input,
                       op == NetworkOperation::AllGather ? output_ : output,
                       count, stream);
    if (op == NetworkOperation::AllGather) {
      permute<<<blocks, 256, 0, stream>>>(
          output_, static_cast<uint32_t*>(output), count, order_, false);
      check(cudaGetLastError());
    }
    check(cudaEventRecord(done_, stream));
    recorded_ = true;
    last_ = stream;
  }
  void synchronize() override {
    if (recorded_) check(cudaEventSynchronize(done_));
    if (executor_) executor_->synchronize();
  }
  bool initialized() const override {
    return executor_ && executor_->initialized();
  }
  bool idle() const override {
    return (!recorded_ || cudaEventQuery(done_) == cudaSuccess) &&
           (!executor_ || executor_->idle());
  }
};
}  // namespace
BackendRegistration peerPacketBackend() {
  return {{"peer-packet",
           "UCCL_LITE_MSCCLPP_LOCAL_STRICT_V1",
           {4},
           ScalarType::Float32,
           true,
           true},
          [] { return PeerPacketCollectives::createUniqueId(); },
          [](BackendContext const& c) { return std::make_unique<Adapter>(c); }};
}
}  // namespace uccl_lite::runtime

#include "uccl_lite/collective_call.hpp"
#include "uccl_lite/collectives/device_primitives.cuh"
#include "uccl_lite/collectives/peer_collectives.hpp"
#include "uccl_lite/hardware_calibration.hpp"
#include "uccl_lite/peer_calibration.hpp"
#include "uccl_lite/peer_profile.hpp"
#include "uccl_lite/runtime/build_identity.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace uccl_lite {
namespace {
std::string const& peerExecutorIdentity() {
  static std::string id = runtime::artifactIdentity(
      reinterpret_cast<uintptr_t>(&peerExecutorIdentity));
  return id;
}
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct CurrentDevice {
  int device;
  CurrentDevice() { check(cudaGetDevice(&device)); }
  ~CurrentDevice() { cudaSetDevice(device); }
};
using Inputs = collective::PeerInputs<__nv_bfloat16, kPeerRankCapacity>;
struct Outputs {
  __nv_bfloat16* values[kPeerRankCapacity];
};

// Generalization of peer_ag_rs.cu::pullGather; destination owns its output.
__global__ void peerGather(Inputs inputs, __nv_bfloat16* output, size_t count) {
  int source = blockIdx.y;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    collective::copyElement(inputs.values[source],
                            output + size_t(source) * count, i);
}

// Generalization of peer_ag_rs.cu::pullScatter. AR uses offset zero and a full
// tensor; RS uses the destination shard. Preserve source order and round once.
__global__ void peerReduce(Inputs inputs, __nv_bfloat16* output, size_t count,
                           size_t offset, int ranks) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    output[i] = collective::orderedSum(inputs, offset + i, ranks);
  }
}

// BF16 adaptation of peer_ag_rs.cu::pullVecGather. Each thread moves a
// naturally aligned 128-bit payload, preserving every BF16 bit pattern.
__global__ void peerGatherVector(Inputs inputs, __nv_bfloat16* output,
                                 size_t vectors) {
  int source = blockIdx.y;
  auto const* input = reinterpret_cast<uint4 const*>(inputs.values[source]);
  auto* destination =
      reinterpret_cast<uint4*>(output) + size_t(source) * vectors;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors;
       i += size_t(gridDim.x) * blockDim.x)
    collective::copyVector(input, destination, i);
}

// Load eight BF16 lanes per peer transaction instruction. Arithmetic remains
// FP32 in ascending source order with one final RN BF16 conversion per lane.
__global__ void peerReduceVector(Inputs inputs, __nv_bfloat16* output,
                                 size_t vectors, size_t offset, int ranks) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors;
       i += size_t(gridDim.x) * blockDim.x) {
    reinterpret_cast<uint4*>(output)[i] =
        collective::orderedSumBf16Vector(inputs, i, offset, ranks);
  }
}

// Central reduce/distribute adaptation for small BF16 collectives. One GPU
// owns all output elements; ordinary peer loads/stores need no device polling.
// This is not the native FP32 two-NUMA A2 tree implementation.
__global__ void peerCoordinator(Inputs inputs, Outputs outputs, size_t count,
                                int ranks, Collective op) {
  size_t total = op == Collective::AllReduce ? count : count * ranks;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += size_t(gridDim.x) * blockDim.x) {
    __nv_bfloat16 value;
    if (op == Collective::AllGather) {
      value = inputs.values[i / count][i % count];
    } else {
      value = collective::orderedSum(inputs, i, ranks);
    }
    if (op == Collective::ReduceScatter)
      outputs.values[i / count][i % count] = value;
    else
      for (int destination = 0; destination < ranks; ++destination)
        outputs.values[destination][i] = value;
  }
}

// The vector count is per contribution/shard (full tensor for AR), so every
// source/output ownership boundary is aligned, not only the total byte count.
__global__ void peerCoordinatorVector(Inputs inputs, Outputs outputs,
                                      size_t vectors, int ranks,
                                      Collective op) {
  size_t total = op == Collective::AllReduce ? vectors : vectors * ranks;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += size_t(gridDim.x) * blockDim.x) {
    uint4 value;
    if (op == Collective::AllGather) {
      value = reinterpret_cast<uint4 const*>(
          inputs.values[i / vectors])[i % vectors];
    } else {
      value = collective::orderedSumBf16Vector(inputs, i, 0, ranks);
    }
    if (op == Collective::ReduceScatter)
      reinterpret_cast<uint4*>(outputs.values[i / vectors])[i % vectors] =
          value;
    else
      for (int destination = 0; destination < ranks; ++destination)
        reinterpret_cast<uint4*>(outputs.values[destination])[i] = value;
  }
}

}  // namespace

// Eager CUDA-event bridge. Every producer is recorded before any reader waits.
// Every reader is recorded before any caller waits. Consequently repeated calls
// may reuse event objects without a circular wait or an early input overwrite.
class PeerCollectives::Impl {
 public:
  explicit Impl(Topology const& topology, PeerTuning tuning)
      : topology_(topology),
        tuning_(std::move(tuning)),
        devices_(topology.devices) {
    if (!PeerCollectives::supported(topology))
      throw std::invalid_argument(
          "direct-peer path requires all-pairs CUDA P2P");
    if (tuning_.planCache.empty())
      if (auto path = std::getenv("UCCL_LITE_PEER_PLAN_CACHE"))
        tuning_.planCache = path;
    if (tuning_.hardwareProfile.empty()) {
      if (auto path = std::getenv("UCCL_LITE_HARDWARE_PROFILE"))
        tuning_.hardwareProfile = path;
    }
    if (!tuning_.hardwareProfile.empty())
      applyHardwareCalibration(
          topology_, loadHardwareProfile(tuning_.hardwareProfile).calibration);
    CurrentDevice restore;
    try {
      for (size_t r = 0; r < devices_.size(); ++r) {
        check(cudaSetDevice(devices_[r]));
        for (size_t s = 0; s < devices_.size(); ++s) {
          if (r == s) continue;
          int available = 0;
          check(cudaDeviceCanAccessPeer(&available, devices_[r], devices_[s]));
          if (!available) throw std::invalid_argument("peer topology changed");
          auto status = cudaDeviceEnablePeerAccess(devices_[s], 0);
          if (status == cudaErrorPeerAccessAlreadyEnabled)
            cudaGetLastError();
          else
            check(status);
        }
        check(cudaStreamCreateWithFlags(&streams_[r], cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&ready_[r], cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&done_[r], cudaEventDisableTiming));
        check(
            cudaEventCreateWithFlags(&transferred_[r], cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&reduced_[r], cudaEventDisableTiming));
      }
    } catch (...) {
      release();
      throw;
    }
  }
  ~Impl() { release(); }
  using Key = std::tuple<Collective, ScalarType, size_t, bool, unsigned>;
  Key key(CollectiveCall const& call) const {
    if (call.ports.size() != devices_.size())
      throw std::invalid_argument("peer call rank count changed");
    for (size_t r = 0; r < devices_.size(); ++r)
      if (call.ports[r].device != devices_[r] || !call.ports[r].input ||
          !call.ports[r].output)
        throw std::invalid_argument(
            "peer call device order or buffers changed");
    auto w = call.workload();
    return {w.collective, w.dtype, w.bytes, w.inPlace, w.alignment};
  }
  PeerSelection const& plan(CollectiveCall const& call) const {
    auto it = plans_.find(key(call));
    if (it == plans_.end())
      throw std::logic_error("prepare exact peer workload before enqueue");
    return it->second;
  }
  void prepare(CollectiveCall const& call) {
    auto k = key(call);
    CurrentDevice restore;
    for (auto const& port : call.ports) {
      check(cudaSetDevice(port.device));
      cudaStreamCaptureStatus capture;
      check(cudaStreamIsCapturing(port.stream, &capture));
      if (capture != cudaStreamCaptureStatusNone)
        throw std::invalid_argument(
            "peer preparation is outside graph capture");
    }
    if (plans_.count(k)) return;
    auto w = call.workload();
    PeerSelection selected;
    if (tuning_.automaticCalibration && !tuning_.validationFamily) {
      std::optional<PeerSelection> cached;
      if (!tuning_.planCache.empty())
        cached = matchPeerPlanRecord(topology_, w, peerExecutorIdentity(),
                                     loadPeerPlanRecords(tuning_.planCache));
      selected = cached ? std::move(*cached) : calibrate(call);
      if (!cached && !tuning_.planCache.empty())
        updatePeerPlanRecord(
            tuning_.planCache,
            {topology_.identity, peerExecutorIdentity(),
             selected.calibrationIdentity, w, selected.localTimings, true});
    } else
      selected = selectPeerPlan(topology_, w, tuning_);
    if (selected.family == PeerFamily::Bulk &&
        call.collective != Collective::AllGather)
      reserveScratch(call);
    plans_.emplace(k, std::move(selected));
  }
  PeerSelection calibrate(CollectiveCall const& call) {
    auto w = call.workload();
    // Admission precedes probes, allocations, and timing. A malformed call
    // cannot trigger calibration or bypass the implementation's capabilities.
    auto admitted = peerCandidates(topology_, w, tuning_);
    auto initial = choosePlan(topology_, w, false, admitted);
    if (admitted.empty() || !initial.candidates[0].eligible)
      throw std::invalid_argument(
          "unsupported automatic peer calibration call");
    if (!hardwareCalibrated_) {
      bool complete = topology_.gpuNodes.size() == devices_.size() &&
                      (devices_.size() == 1 || !topology_.paths.empty());
      for (auto node : topology_.gpuNodes)
        complete &=
            node < topology_.hardware.size() &&
            topology_.hardware[node].memory.evidence == Evidence::Measured;
      for (auto const& path : topology_.paths)
        complete &= path.performance.evidence == Evidence::Measured;
      if (!complete)
        applyHardwareCalibration(topology_,
                                 probeGpuResources(topology_).calibration);
      hardwareCalibrated_ = true;
    }
    PeerTuning local = tuning_;
    // Data-only estimates order trials. These zeros are not claimed as measured
    // launch overhead and cannot admit a candidate without exact verification.
    local.overheadUs = {{0, 0, 0}};
    auto candidates = peerCandidates(topology_, w, local);
    auto costs = choosePlan(topology_, w, false, candidates);
    std::vector<size_t> order;
    for (size_t i = 0; i < candidates.size(); ++i)
      if (costs.candidates[i].eligible) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return costs.candidates[a].microseconds <
             costs.candidates[b].microseconds;
    });
    std::vector<PeerFamily> families;
    for (auto i : order) {
      auto family = static_cast<PeerFamily>(i);
      if (family == PeerFamily::Bulk &&
          call.collective != Collective::AllGather)
        reserveScratch(call);
      families.push_back(family);
    }
    auto qualification = qualifyPeerPlans(
        topology_, w, families,
        [this](CollectiveCall const& trial, PeerFamily family) {
          enqueueFamily(trial, family);
        });
    // Fit the nonnegative unmodeled cost (launches, control, model error) for
    // this exact workload. Never
    // extrapolate it to another size, alignment, operation, or topology.
    for (auto const& trial : qualification.trials) {
      auto samples = trial.microseconds;
      std::sort(samples.begin(), samples.end());
      auto index = static_cast<size_t>(trial.family);
      if (costs.candidates[index].estimated)
        local.overheadUs[index] =
            std::max(0.0, samples[samples.size() / 2] -
                              costs.candidates[index].microseconds);
    }
    candidates = peerCandidates(topology_, w, local);
    size_t winner = static_cast<size_t>(qualification.family);
    PeerSelection selected{
        qualification.family, std::move(candidates[winner].plan),
        costs.candidates[winner].estimated,
        "verified exact local timing; 3 repetitions, 3 calls/repetition, host "
        "enqueue-to-completion scope"};
    selected.qualified = true;
    selected.calibrationIdentity = topology_.calibrationIdentity;
    for (auto const& trial : qualification.trials)
      selected.localTimings[static_cast<size_t>(trial.family)] =
          trial.microseconds;
    return selected;
  }
  void reserveScratch(CollectiveCall const& call) {
    CurrentDevice restore;
    size_t bytes = scratchBytesFor(call);
    if (bytes <= scratchBytes_) return;
    // Grow only at the explicit idle preparation boundary. Keep old storage
    // intact until every new allocation succeeds, then replace it atomically
    // with respect to this caller-serialized object.
    for (size_t r = 0; r < devices_.size(); ++r) {
      check(cudaSetDevice(devices_[r]));
      check(cudaStreamSynchronize(streams_[r]));
    }
    std::array<__nv_bfloat16*, kPeerRankCapacity> next{};
    try {
      for (size_t r = 0; r < devices_.size(); ++r) {
        check(cudaSetDevice(devices_[r]));
        check(cudaMalloc(&next[r], bytes));
      }
    } catch (...) {
      for (size_t r = 0; r < devices_.size(); ++r) {
        cudaSetDevice(devices_[r]);
        if (next[r]) cudaFree(next[r]);
      }
      throw;
    }
    auto previous = scratch_;
    scratch_ = next;
    scratchBytes_ = bytes;
    for (size_t r = 0; r < devices_.size(); ++r) {
      check(cudaSetDevice(devices_[r]));
      if (previous[r]) check(cudaFree(previous[r]));
    }
  }
  void enqueue(CollectiveCall const& call) {
    enqueueFamily(call, plan(call).family);
  }
  void enqueueFamily(CollectiveCall const& call, PeerFamily family) {
    CurrentDevice restore;
    int ranks = int(devices_.size());
    if (ranks == 1) {
      // A one-rank collective is an identity operation, including NaN payloads
      // and signed zero. Chain completion across changing caller streams.
      auto const& port = call.ports.front();
      check(cudaSetDevice(port.device));
      check(cudaStreamWaitEvent(port.stream, done_[0], 0));
      check(cudaMemcpyAsync(port.output, port.input, call.count * 2,
                            cudaMemcpyDeviceToDevice, port.stream));
      check(cudaEventRecord(done_[0], port.stream));
      return;
    }
    bool staged =
        family == PeerFamily::Bulk && call.collective != Collective::AllGather;
    bool copyGather =
        family == PeerFamily::Bulk && call.collective == Collective::AllGather;
    bool callerSmall = ranks == 2 && family == PeerFamily::Coordinator;
    if (staged && scratchBytesFor(call) > scratchBytes_)
      throw std::logic_error(
          "prepare peer scratch before a larger staged collective");
    Inputs inputs{};
    for (int r = 0; r < ranks; ++r) {
      inputs.values[r] = static_cast<__nv_bfloat16 const*>(call.ports[r].input);
      if (callerSmall && r == 0)
        continue;  // The coordinator runs behind this caller's producer.
      check(cudaSetDevice(devices_[r]));
      check(cudaEventRecord(ready_[r], call.ports[r].stream));
    }
    size_t count = staged ? reductionCount(call) : call.count;
    bool vectorized = call.workload().alignment >= 16 && count % 8 == 0;
    size_t units = vectorized ? count / 8 : count;
    unsigned blocks = unsigned(std::min(size_t(65535), (units + 255) / 256));
    if (staged) {
      enqueueStaged(call, inputs, vectorized, units, blocks);
      callerCoordinatorPending_ = false;
      return;
    }
    if (family == PeerFamily::Coordinator) {
      enqueueSmall(call, inputs, vectorized, units);
      return;
    }
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices_[r]));
      // A preceding small call may have run outside private stream0. Bridge
      // it before this generic family can replace done0 with a newer record.
      if (r == 0 && callerCoordinatorPending_)
        check(cudaStreamWaitEvent(streams_[0], done_[0], 0));
      for (int s = 0; s < ranks; ++s)
        check(cudaStreamWaitEvent(streams_[r], ready_[s], 0));
      auto output = static_cast<__nv_bfloat16*>(call.ports[r].output);
      if (copyGather) {
        // Source-owned peer-copy AllGather shares the same
        // producer/output readiness and all-source completion bridge.
        for (int phase = 0; phase < ranks; ++phase) {
          int destination = (r + phase) % ranks;
          auto* target =
              static_cast<__nv_bfloat16*>(call.ports[destination].output) +
              size_t(r) * call.count;
          if (destination == r)
            check(cudaMemcpyAsync(target, inputs.values[r], call.count * 2,
                                  cudaMemcpyDeviceToDevice, streams_[r]));
          else
            check(cudaMemcpyPeerAsync(target, devices_[destination],
                                      inputs.values[r], devices_[r],
                                      call.count * 2, streams_[r]));
        }
      } else if (call.collective == Collective::AllGather) {
        if (vectorized)
          peerGatherVector<<<dim3(blocks, ranks), 256, 0, streams_[r]>>>(
              inputs, output, units);
        else
          peerGather<<<dim3(blocks, ranks), 256, 0, streams_[r]>>>(
              inputs, output, call.count);
      } else {
        size_t offset = call.collective == Collective::ReduceScatter
                            ? size_t(r) * call.count
                            : 0;
        if (vectorized)
          peerReduceVector<<<blocks, 256, 0, streams_[r]>>>(
              inputs, output, units, offset, ranks);
        else
          peerReduce<<<blocks, 256, 0, streams_[r]>>>(
              inputs, output, call.count, offset, ranks);
      }
      check(cudaGetLastError());
      check(cudaEventRecord(done_[r], streams_[r]));
    }
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices_[r]));
      for (int s = 0; s < ranks; ++s)
        check(cudaStreamWaitEvent(call.ports[r].stream, done_[s], 0));
    }
    callerCoordinatorPending_ = false;
  }

 private:
  void enqueueSmall(CollectiveCall const& call, Inputs inputs, bool vectorized,
                    size_t units) {
    int ranks = int(devices_.size());
    Outputs outputs{};
    for (int r = 0; r < ranks; ++r)
      outputs.values[r] = static_cast<__nv_bfloat16*>(call.ports[r].output);
    bool caller = ranks == 2;
    cudaStream_t stream = caller ? call.ports[0].stream : streams_[0];
    check(cudaSetDevice(devices_[0]));
    if (caller) check(cudaStreamWaitEvent(stream, done_[0], 0));
    for (int r = caller ? 1 : 0; r < ranks; ++r)
      check(cudaStreamWaitEvent(stream, ready_[r], 0));
    size_t total =
        call.collective == Collective::AllReduce ? units : units * ranks;
    unsigned blocks = unsigned(std::min(size_t(65535), (total + 255) / 256));
    if (vectorized)
      peerCoordinatorVector<<<blocks, 256, 0, stream>>>(inputs, outputs, units,
                                                        ranks, call.collective);
    else
      peerCoordinator<<<blocks, 256, 0, stream>>>(inputs, outputs, call.count,
                                                  ranks, call.collective);
    check(cudaGetLastError());
    // Kernel completion flushes remote output writes before dependent callers.
    // Prior stream order or the explicit done0 wait covers previous stream0
    // readers; untouched done[r>0] preserve the other staged-reader credits.
    check(cudaEventRecord(done_[0], stream));
    if (caller) callerCoordinatorPending_ = true;
    for (int r = caller ? 1 : 0; r < ranks; ++r) {
      check(cudaSetDevice(devices_[r]));
      check(cudaStreamWaitEvent(call.ports[r].stream, done_[0], 0));
    }
  }
  size_t reductionCount(CollectiveCall const& call) const {
    return call.collective == Collective::AllReduce
               ? call.count / devices_.size()
               : call.count;
  }
  size_t scratchBytesFor(CollectiveCall const& call) const {
    size_t factor =
        call.collective == Collective::AllReduce ? 1 : devices_.size();
    if (call.count > std::numeric_limits<size_t>::max() / factor / 2)
      throw std::invalid_argument("peer scratch extent overflow");
    return call.count * factor * 2;
  }
  void enqueueStaged(CollectiveCall const& call, Inputs inputs, bool vectorized,
                     size_t units, unsigned blocks) {
    int ranks = int(devices_.size());
    size_t count = reductionCount(call);
    bool gather = call.collective == Collective::AllReduce;
    bool directSelf = ranks == 2;
    // Source-owned bulk transfers adapt peer_ag_rs.cu::launchPeer. Scratch is
    // destination-local and each source owns one disjoint contribution slice.
    for (int source = 0; source < ranks; ++source) {
      check(cudaSetDevice(devices_[source]));
      for (int r = 0; r < ranks; ++r)
        check(cudaStreamWaitEvent(streams_[source], ready_[r], 0));
      // Caller stream sets may change between host-serialized calls. Protect
      // private scratch with internal credits rather than relying on callers'
      // previous done waits. All waits bind to the previous done generation:
      // no current done event is recorded until every source copy is queued.
      // Newly created, never-recorded events represent already-complete work.
      for (int r = 0; r < ranks; ++r)
        check(cudaStreamWaitEvent(streams_[source], done_[r], 0));
      for (int phase = 0; phase < ranks; ++phase) {
        int destination = (source + phase) % ranks;
        auto* target = scratch_[destination] + size_t(source) * count;
        auto const* origin =
            inputs.values[source] + size_t(destination) * count;
        if (source == destination) {
          if (!directSelf)
            check(cudaMemcpyAsync(target, origin, count * 2,
                                  cudaMemcpyDeviceToDevice, streams_[source]));
        } else
          check(cudaMemcpyPeerAsync(target, devices_[destination], origin,
                                    devices_[source], count * 2,
                                    streams_[source]));
      }
      check(cudaEventRecord(transferred_[source], streams_[source]));
    }
    for (int destination = 0; destination < ranks; ++destination) {
      check(cudaSetDevice(devices_[destination]));
      for (int source = 0; source < ranks; ++source)
        check(cudaStreamWaitEvent(streams_[destination], transferred_[source],
                                  0));
      Inputs local{};
      for (int source = 0; source < ranks; ++source)
        local.values[source] =
            directSelf && source == destination
                ? inputs.values[destination] + size_t(destination) * count
                : scratch_[destination] + size_t(source) * count;
      auto* output =
          static_cast<__nv_bfloat16*>(call.ports[destination].output) +
          (gather ? size_t(destination) * count : 0);
      if (vectorized)
        peerReduceVector<<<blocks, 256, 0, streams_[destination]>>>(
            local, output, units, 0, ranks);
      else
        peerReduce<<<blocks, 256, 0, streams_[destination]>>>(local, output,
                                                              count, 0, ranks);
      check(cudaGetLastError());
      check(cudaEventRecord(gather ? reduced_[destination] : done_[destination],
                            streams_[destination]));
    }
    if (gather) {
      // Record every reduction before any fan-out wait. Finished BF16 shards
      // are immutable; each source alone writes its shard to other outputs.
      for (int source = 0; source < ranks; ++source) {
        check(cudaSetDevice(devices_[source]));
        for (int r = 0; r < ranks; ++r)
          check(cudaStreamWaitEvent(streams_[source], reduced_[r], 0));
        auto const* origin =
            static_cast<__nv_bfloat16 const*>(call.ports[source].output) +
            size_t(source) * count;
        for (int phase = 1; phase < ranks; ++phase) {
          int destination = (source + phase) % ranks;
          auto* target =
              static_cast<__nv_bfloat16*>(call.ports[destination].output) +
              size_t(source) * count;
          check(cudaMemcpyPeerAsync(target, devices_[destination], origin,
                                    devices_[source], count * 2,
                                    streams_[source]));
        }
        check(cudaEventRecord(done_[source], streams_[source]));
      }
    }
    // Public completion covers all reductions and source input readers.
    // Scratch reuse is protected separately by the internal prior-done waits.
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices_[r]));
      for (int s = 0; s < ranks; ++s)
        check(cudaStreamWaitEvent(call.ports[r].stream, done_[s], 0));
    }
  }
  void release() noexcept {
    int old = 0;
    cudaGetDevice(&old);
    for (size_t r = 0; r < devices_.size(); ++r) {
      cudaSetDevice(devices_[r]);
      if (streams_[r]) cudaStreamSynchronize(streams_[r]);
    }
    // The latest done0 transitively covers every caller-stream coordinator,
    // including after an outgoing generic or staged family replaced it.
    if (done_[0]) {
      cudaSetDevice(devices_[0]);
      cudaEventSynchronize(done_[0]);
    }
    for (size_t r = 0; r < devices_.size(); ++r) {
      cudaSetDevice(devices_[r]);
      if (ready_[r]) cudaEventDestroy(ready_[r]);
      if (done_[r]) cudaEventDestroy(done_[r]);
      if (transferred_[r]) cudaEventDestroy(transferred_[r]);
      if (reduced_[r]) cudaEventDestroy(reduced_[r]);
      if (scratch_[r]) cudaFree(scratch_[r]);
      if (streams_[r]) cudaStreamDestroy(streams_[r]);
    }
    cudaSetDevice(old);
  }
  bool hardwareCalibrated_ = false;
  Topology topology_;
  PeerTuning tuning_;
  std::map<Key, PeerSelection> plans_;
  std::vector<int> devices_;
  std::array<cudaStream_t, kPeerRankCapacity> streams_{};
  std::array<cudaEvent_t, kPeerRankCapacity> ready_{}, done_{}, transferred_{},
      reduced_{};
  std::array<__nv_bfloat16*, kPeerRankCapacity> scratch_{};
  size_t scratchBytes_ = 0;
  bool callerCoordinatorPending_ = false;
};

PeerCollectives::PeerCollectives(Topology const& topology, PeerTuning tuning)
    : impl_(std::make_unique<Impl>(topology, std::move(tuning))) {}
PeerCollectives::~PeerCollectives() = default;
void PeerCollectives::prepare(CollectiveCall const& call) {
  impl_->prepare(call);
}
void PeerCollectives::enqueue(CollectiveCall const& call) {
  impl_->enqueue(call);
}
PeerSelection const& PeerCollectives::plan(CollectiveCall const& call) const {
  return impl_->plan(call);
}
bool PeerCollectives::supported(Topology const& topology) {
  size_t n = topology.devices.size();
  if (!n || n > kPeerRankCapacity || topology.peerAccess.size() != n * n)
    return false;
  if (std::set<int>(topology.devices.begin(), topology.devices.end()).size() !=
          n ||
      std::any_of(topology.devices.begin(), topology.devices.end(),
                  [](int device) { return device < 0; }))
    return false;
  for (size_t r = 0; r < n; ++r)
    for (size_t s = 0; s < n; ++s)
      if (r != s && !topology.peerAccess[r * n + s]) return false;
  return true;
}
}  // namespace uccl_lite

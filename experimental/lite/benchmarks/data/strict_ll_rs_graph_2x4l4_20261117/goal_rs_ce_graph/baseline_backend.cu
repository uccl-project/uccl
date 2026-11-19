#include "algorithm_transport.h"
#include "uccl_lite/collectives.h"
#include <linux/mempolicy.h>
#include <mscclpp/algorithm.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
void checked(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
using namespace multinode_kernels;

// Client IO is fused with the retained GPU algorithm. Alternate scratch banks
// protect readers across successive calls without a separate entry handshake.
__global__ void stagedCollective(LocalDirectChans local, PortChan ib, int rank,
                                 char* buffer, Layout layout,
                                 NetworkOperation op, uint32_t const* input,
                                 uint32_t* output, size_t inElements,
                                 size_t outElements) {
  int localRank = rank % 4;
  // Two banks keep a faster peer's next call away from current readers.
  // Reaching call i+2 requires the local and IB handshakes of call i+1,
  // which cannot occur until every participating rank has completed call i.
  bool ag = op == NetworkOperation::AllGather;
  bool rs = op == NetworkOperation::ReduceScatter;
  size_t elements = inElements > outElements ? inElements : outElements;
  if (ag)
    hierarchicalAllGatherSerialBody<true>(
        local, ib, reinterpret_cast<float*>(buffer), elements, layout, rank,
        localRank, input, output);
  else if (rs)
    hierarchicalReduceScatterSerialBody<true>(
        local, ib, reinterpret_cast<float*>(buffer), elements, layout, rank,
        localRank, input, output);
  else {
    constexpr int position[4] = {0, 1, 3, 2};
    hierarchicalAllReduceSerialBody<true>(
        local, ib, reinterpret_cast<float*>(buffer), elements, layout,
        localRank, position[localRank], input, output);
  }
}

// A large call reuses one registered bank. It must wait for the preceding
// collective's readers and output copies on every rank before overwriting it.
__global__ void reuseLarge(LocalDirectChans local, PortChan ib, int rank) {
  if (threadIdx.x == 0) {
    for (int p = 0; p < 4; ++p)
      if (p != rank) local.c[p].signal();
    for (int p = 0; p < 4; ++p)
      if (p != rank) local.c[p].wait();
    ib.signal();
    ib.wait();
    ib.flush();
  }
}

// Keep client staging on the caller's CUDA stream without CUDA's internal
// peer-copy scheduling. The buffers can have different per-rank alignment.
__global__ void copyClient(uint32_t* output, uint32_t const* input,
                           size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  if (((reinterpret_cast<uintptr_t>(input) |
        reinterpret_cast<uintptr_t>(output)) &
       15) == 0) {
    for (size_t i = tid; i < count / 4; i += stride)
      reinterpret_cast<uint4*>(output)[i] =
          reinterpret_cast<uint4 const*>(input)[i];
    for (size_t i = count / 4 * 4 + tid; i < count; i += stride)
      output[i] = input[i];
  } else {
    for (size_t i = tid; i < count; i += stride) output[i] = input[i];
  }
}

__global__ void graphGatherStart(PortChan ib, char* buffer, Layout layout,
                                 int rank, size_t shard) {
  auto const* src = reinterpret_cast<uint4 const*>(buffer + layout.ibSend);
  auto* dst = reinterpret_cast<uint4*>(buffer + layout.data + rank * shard);
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < shard / sizeof(uint4); i += gridDim.x * blockDim.x)
    dst[i] = src[i];
  // ibSend was populated by the preceding graph node, not by this kernel.
  if (tid == 0)
    ib.putWithSignal(layout.data + rank * shard, layout.ibSend, shard);
}
__global__ void graphPeerReady(MemChan send, MemChan recv) {
  if (threadIdx.x == 0) {
    send.signal();
    recv.wait();
  }
}
__global__ void graphIbReady(PortChan ib, bool flush = true) {
  if (threadIdx.x == 0) {
    ib.wait();
    if (flush) ib.flush();
  }
}
__global__ void graphIbSend(PortChan ib, size_t dst, size_t src, size_t bytes) {
  if (threadIdx.x == 0) ib.putWithSignal(dst, src, bytes);
}
__global__ void graphAdd(float* dst, float const* a, float const* b,
                         size_t count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  addOut(dst, a, b, count, tid, gridDim.x * blockDim.x);
}

struct GraphInput {
  float const* pointer;
};
__global__ void setGraphInput(GraphInput* slot, void const* pointer) {
  if (threadIdx.x == 0) slot->pointer = reinterpret_cast<float const*>(pointer);
}
__global__ void graphAddClient(float* dst, GraphInput const* slot,
                               size_t offset, float const* peer, size_t count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto const* own = slot->pointer + offset;
  for (size_t i = tid; i < count / 4; i += stride) {
    float4 x = loadClientFloatQuad(reinterpret_cast<uint32_t const*>(own), i);
    float4 y = reinterpret_cast<float4 const*>(peer)[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    reinterpret_cast<float4*>(dst)[i] = x;
  }
  for (size_t i = count / 4 * 4 + tid; i < count; i += stride)
    dst[i] = own[i] + peer[i];
}
struct InputNode {
  cudaGraphNode_t node;
  void* destination;
  size_t offset, bytes;
};
struct ReduceGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  std::vector<InputNode> inputs;
};

struct Context {
  int rank, device, rail;
  size_t capacity;
  size_t bankBytes = 0;
  uint64_t sequence = 0;
  char* buffer = nullptr;
  cudaEvent_t done = nullptr;
  cudaStream_t lastStream = nullptr;
  bool pending = false;
  Layout layout{}, largeLayout{};
  size_t smallBase = 0;
  int blocks = 96;
  int directRsBlocks = 96;
  std::unique_ptr<GpuTransport> transport;
  cudaStream_t graphBuilder = nullptr;
  std::map<size_t, cudaGraphExec_t> gatherGraphs;
  std::map<size_t, ReduceGraph> reduceGraphs;
  GraphInput* graphInput = nullptr;
  std::shared_ptr<mscclpp::Communicator> owner;
  Context(int r, int d, int net, size_t bytes)
      : rank(r), device(d), rail(net), capacity(bytes) {}
  ~Context() {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (pending) cudaEventSynchronize(done);
    for (auto const& entry : gatherGraphs) cudaGraphExecDestroy(entry.second);
    for (auto const& entry : reduceGraphs) {
      cudaGraphExecDestroy(entry.second.executable);
      cudaGraphDestroy(entry.second.graph);
    }
    if (graphInput) cudaFree(graphInput);
    if (graphBuilder) cudaStreamDestroy(graphBuilder);
    transport.reset();
    if (buffer) cudaFree(buffer);
    if (done) cudaEventDestroy(done);
    cudaSetDevice(previous);
  }
  void prepare(std::shared_ptr<mscclpp::Communicator> comm) {
    if (transport) return;
    if (buffer) throw std::logic_error("failed preparation cannot be retried");
    checked(cudaDeviceSynchronize());
    // MSCCL++ allocation helpers can bind the caller. Preserve its affinity.
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity))
      throw std::runtime_error("cannot read caller affinity");
    int memoryMode = 0;
    std::array<unsigned long, 16> memoryNodes{};
    if (syscall(SYS_get_mempolicy, &memoryMode, memoryNodes.data(),
                sizeof(memoryNodes) * 8, nullptr, 0))
      throw std::runtime_error("cannot read caller NUMA policy");
    struct Restore {
      cpu_set_t const& mask;
      int mode;
      std::array<unsigned long, 16> const& nodes;
      ~Restore() {
        int base = mode & ~MPOL_MODE_FLAGS;
        syscall(
            SYS_set_mempolicy, mode,
            base == MPOL_DEFAULT || base == MPOL_LOCAL ? nullptr : nodes.data(),
            sizeof(nodes) * 8);
        sched_setaffinity(0, sizeof(mask), &mask);
      }
    } restore{affinity, memoryMode, memoryNodes};
    size_t smallCapacity = std::min(capacity, size_t(512) << 10);
    size_t slot = smallCapacity / 4 + 4096;
    layout = {0, smallCapacity, 2 * smallCapacity, 2 * smallCapacity + slot, 0};
    bankBytes = layout.ibScratch + slot;
    if (capacity > smallCapacity) {
      size_t largeSlot = capacity / 4 + 4096;
      largeLayout = {0, capacity, 2 * capacity, 2 * capacity + largeSlot,
                     2 * capacity + 2 * largeSlot};
      smallBase = largeLayout.immutableInput + capacity;
      cudaDeviceProp property{};
      checked(cudaGetDeviceProperties(&property, device));
      int resident = 0;
      auto limit = [&](auto kernel) {
        checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel,
                                                              512, 0));
        blocks = std::min(blocks, resident * property.multiProcessorCount);
      };
      limit(hierarchicalAllReduce<DirectRingChan>);
      limit(hierarchicalAllGatherRing<DirectRingChan>);
      limit(hierarchicalReduceScatterRing<DirectRingChan>);
      limit(hierarchicalReduceScatterDirectRing<DirectRingChan>);
      checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &resident, hierarchicalReduceScatterDirectRing<ClientRingChan>, 512,
          0));
      directRsBlocks =
          std::min(blocks, resident * property.multiProcessorCount);
      if (blocks < 1 || directRsBlocks < 1)
        throw std::runtime_error("large collective cannot reside");
    }
    size_t bytes = smallBase + 2 * bankBytes;
    if (bytes >= (1ULL << 32))
      throw std::invalid_argument("32-bit channel offset overflow");
    checked(cudaMalloc(&buffer, bytes));
    checked(cudaMemset(buffer, 0, bytes));
    checked(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    transport = std::make_unique<GpuTransport>(
        *comm, rank, rank % 4, rail, buffer, bytes, true, false, false, true);
    mscclpp::gpuStreamPool()->clear();
    owner = comm;
    checked(cudaMalloc(&graphInput, sizeof(GraphInput)));
    checked(cudaStreamCreateWithFlags(&graphBuilder, cudaStreamNonBlocking));
  }
  cudaGraphExec_t gatherGraph(size_t bytes) {
    auto found = gatherGraphs.find(bytes);
    if (found != gatherGraphs.end()) return found->second;
    auto& t = *transport;
    size_t shard = bytes / 8;
    constexpr int positions[4] = {0, 1, 3, 2};
    constexpr int order[4] = {0, 1, 3, 2};
    int localRank = rank % 4, pos = positions[localRank];
    MemChan send = t.localDirect.c[ringSuccessor(localRank)];
    MemChan recv = t.localDirect.c[ringPredecessor(localRank)];
    cudaGraph_t graph = nullptr;
    checked(
        cudaStreamBeginCapture(graphBuilder, cudaStreamCaptureModeThreadLocal));
    graphGatherStart<<<256, 256, 0, graphBuilder>>>(t.ib, buffer, largeLayout,
                                                    rank, shard);
    for (int phase = 0; phase < 2; ++phase) {
      if (phase) graphIbReady<<<1, 32, 0, graphBuilder>>>(t.ib);
      int nodeBase = (rank / 4 * 4) ^ (phase * 4);
      for (int step = 0; step < 3; ++step) {
        size_t offset = largeLayout.data +
                        size_t(nodeBase + order[(pos - step + 4) % 4]) * shard;
        checked(cudaMemcpyAsync(reinterpret_cast<char*>(send.dst_) + offset,
                                buffer + offset, shard,
                                cudaMemcpyDeviceToDevice, graphBuilder));
        graphPeerReady<<<1, 32, 0, graphBuilder>>>(send, recv);
      }
    }
    checked(cudaGetLastError());
    checked(cudaStreamEndCapture(graphBuilder, &graph));
    cudaGraphExec_t executable = nullptr;
    auto result = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    checked(result);
    gatherGraphs.emplace(bytes, executable);
    // Extent-only keys are identical across matching collective sequences.
    // Finish cold graph setup on all ranks before any GPU wait can time out.
    owner->bootstrap()->barrier();
    return executable;
  }
  ReduceGraph& reduceGraph(size_t bytes) {
    auto found = reduceGraphs.find(bytes);
    if (found != reduceGraphs.end()) return found->second;
    auto& t = *transport;
    constexpr int positions[4] = {0, 1, 3, 2};
    int localRank = rank % 4, pos = positions[localRank];
    MemChan send = t.localDirect.c[ringSuccessor(localRank)];
    MemChan recv = t.localDirect.c[ringPredecessor(localRank)];
    size_t count = bytes / 4;
    size_t chunk = ((count + 3) / 4 + 31) & ~size_t(31);
    size_t ownBase = size_t((pos + 1) % 4) * chunk;
    size_t ownLength = std::min(chunk, count - ownBase);
    int pieces = int(std::max<size_t>(
        2, std::min<size_t>(32, (bytes / 4 + (2UL << 20) - 1) / (2UL << 20))));
    int distance = std::min(2, pieces);
    size_t piece = ((ownLength + pieces - 1) / pieces + 31) & ~size_t(31);
    auto* data = reinterpret_cast<float*>(buffer + largeLayout.data);
    auto* scratch = reinterpret_cast<float*>(buffer + largeLayout.localScratch);
    auto* partial = reinterpret_cast<float*>(buffer + largeLayout.ibSend);
    auto* remote = reinterpret_cast<float*>(buffer + largeLayout.ibScratch);
    ReduceGraph resultGraph;
    cudaGraph_t graph = nullptr;
    checked(
        cudaStreamBeginCapture(graphBuilder, cudaStreamCaptureModeThreadLocal));
    for (int p = 0; p < pieces + distance; ++p) {
      if (p < pieces) {
        size_t off = size_t(p) * piece;
        size_t length = off >= ownLength ? 0 : std::min(piece, ownLength - off);
        for (int step = 0; step < 3; ++step) {
          int cSend = (pos - step + 4) % 4;
          int cRecv = (cSend + 3) % 4;
          size_t sendOff = size_t(cSend) * chunk + off;
          size_t recvOff = size_t(cRecv) * chunk + off;
          size_t sendLength =
              sendOff >= count ? 0 : std::min(length, count - sendOff);
          size_t recvLength =
              recvOff >= count ? 0 : std::min(length, count - recvOff);
          size_t dst =
              largeLayout.localScratch + (size_t(step) * chunk + off) * 4;
          checked(cudaMemcpyAsync(reinterpret_cast<char*>(send.dst_) + dst,
                                  data + sendOff, sendLength * 4,
                                  cudaMemcpyDeviceToDevice, graphBuilder));
          if (step == 0) {
            cudaStreamCaptureStatus status;
            cudaGraphNode_t const* dependencies = nullptr;
            size_t dependencyCount = 0;
            checked(cudaStreamGetCaptureInfo(graphBuilder, &status, nullptr,
                                             nullptr, &dependencies,
                                             &dependencyCount));
            if (status != cudaStreamCaptureStatusActive || dependencyCount != 1)
              throw std::runtime_error(
                  "unexpected first-hop capture dependencies");
            resultGraph.inputs.push_back(
                {dependencies[0], reinterpret_cast<char*>(send.dst_) + dst,
                 sendOff * 4, sendLength * 4});
          }
          graphPeerReady<<<1, 32, 0, graphBuilder>>>(send, recv);
          graphAddClient<<<256, 256, 0, graphBuilder>>>(
              step == 2 ? partial + off : data + recvOff, graphInput, recvOff,
              scratch + size_t(step) * chunk + off, recvLength);
        }
        graphIbSend<<<1, 32, 0, graphBuilder>>>(
            t.ib, largeLayout.ibScratch + off * 4, largeLayout.ibSend + off * 4,
            length * 4);
      }
      int completed = p - distance;
      if (completed >= 0 && completed < pieces) {
        size_t off = size_t(completed) * piece;
        size_t length = off >= ownLength ? 0 : std::min(piece, ownLength - off);
        graphIbReady<<<1, 32, 0, graphBuilder>>>(t.ib, completed == pieces - 1);
        graphAdd<<<256, 256, 0, graphBuilder>>>(
            data + ownBase + off, partial + off, remote + off, length);
        for (int step = 0; step < 3; ++step) {
          int cSend = (pos + 1 - step + 4) % 4;
          size_t sendOff = size_t(cSend) * chunk + off;
          size_t sendLength =
              sendOff >= count ? 0 : std::min(length, count - sendOff);
          checked(cudaMemcpyAsync(reinterpret_cast<char*>(send.dst_) +
                                      largeLayout.data + sendOff * 4,
                                  data + sendOff, sendLength * 4,
                                  cudaMemcpyDeviceToDevice, graphBuilder));
          graphPeerReady<<<1, 32, 0, graphBuilder>>>(send, recv);
        }
      }
    }
    checked(cudaGetLastError());
    checked(cudaStreamEndCapture(graphBuilder, &graph));
    cudaGraphExec_t executable = nullptr;
    auto result = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (result != cudaSuccess) {
      cudaGraphDestroy(graph);
      checked(result);
    }
    resultGraph.graph = graph;
    resultGraph.executable = executable;
    auto inserted = reduceGraphs.emplace(bytes, std::move(resultGraph));
    owner->bootstrap()->barrier();
    return inserted.first->second;
  }
  mscclpp::CommResult run(NetworkOperation op, void const* input, void* output,
                          size_t inputBytes, size_t outputBytes,
                          cudaStream_t stream) {
    if (!transport) throw std::logic_error("prepare before enqueue");
    if (pending && stream != lastStream)
      checked(cudaStreamWaitEvent(stream, done, 0));
    auto& t = *transport;
    size_t bytes = std::max(inputBytes, outputBytes);
    if ((op == NetworkOperation::AllGather ||
         op == NetworkOperation::AllReduce) &&
        bytes >= (16UL << 20)) {
      cudaGraphExec_t executable;
      if (op == NetworkOperation::AllGather)
        executable = gatherGraph(bytes);
      else {
        auto& graph = reduceGraph(bytes);
        // CUDA documents that node updates affect only future launches.
        // Refresh each call, including when a freed allocation's VA is reused.
        for (auto const& node : graph.inputs)
          checked(cudaGraphExecMemcpyNodeSetParams1D(
              graph.executable, node.node, node.destination,
              reinterpret_cast<char const*>(input) + node.offset, node.bytes,
              cudaMemcpyDeviceToDevice));
        executable = graph.executable;
      }
      reuseLarge<<<1, 32, 0, stream>>>(t.localDirect, t.ib, rank % 4);
      checked(cudaGetLastError());
      if (op == NetworkOperation::AllGather)
        checked(cudaMemcpyAsync(buffer + largeLayout.ibSend, input, inputBytes,
                                cudaMemcpyDeviceToDevice, stream));
      else {
        setGraphInput<<<1, 1, 0, stream>>>(graphInput, input);
        checked(cudaGetLastError());
      }
      checked(cudaGraphLaunch(executable, stream));
      checked(cudaMemcpyAsync(output, buffer + largeLayout.data, outputBytes,
                              cudaMemcpyDeviceToDevice, stream));
      checked(cudaEventRecord(done, stream));
      lastStream = stream;
      pending = true;
      return mscclpp::CommResult::CommSuccess;
    }
    if (bytes > (512UL << 10)) {
      reuseLarge<<<1, 32, 0, stream>>>(t.localDirect, t.ib, rank % 4);
      checked(cudaGetLastError());
      bool ag = op == NetworkOperation::AllGather;
      bool rs = op == NetworkOperation::ReduceScatter;
      bool packed = rs && bytes <= (4UL << 20);
      size_t inputOffset = ag       ? largeLayout.ibSend
                           : packed ? largeLayout.immutableInput
                                    : largeLayout.data;
      if (!rs) {
        copyClient<<<256, 256, 0, stream>>>(
            reinterpret_cast<uint32_t*>(buffer + inputOffset),
            reinterpret_cast<uint32_t const*>(input), inputBytes / 4);
        checked(cudaGetLastError());
      }
      constexpr int positions[4] = {0, 1, 3, 2};
      int pos = positions[rank % 4];
      DirectRingChan send{t.localDirect.c[ringSuccessor(rank % 4)]};
      DirectRingChan recv{t.localDirect.c[ringPredecessor(rank % 4)]};
      size_t count = bytes / 4;
      auto* base = reinterpret_cast<float*>(buffer);
      if (ag)
        hierarchicalAllGatherRing<DirectRingChan><<<blocks, 512, 0, stream>>>(
            send, recv, t.ib, base, count, largeLayout, rank, rank % 4, pos);
      else if (rs && packed) {
        size_t shard = count / 8;
        packReduceScatterRing<<<dim3(std::min<size_t>(128, (shard + 255) / 256),
                                     8),
                                256, 0, stream>>>(
            reinterpret_cast<float const*>(input),
            reinterpret_cast<float*>(buffer + largeLayout.localScratch), shard);
        checked(cudaGetLastError());
        hierarchicalReduceScatterRing<DirectRingChan>
            <<<blocks, 512, 0, stream>>>(send, recv, t.ib, base, count,
                                         largeLayout, rank, pos, true, 1,
                                         reinterpret_cast<float*>(output));
      } else if (rs) {
        int pieces = bytes <= (16UL << 20)
                         ? 1
                         : int(std::max<size_t>(
                               2, (bytes / 8 + (8UL << 20) - 1) / (8UL << 20)));
        ClientRingChan clientSend{
            send.channel, reinterpret_cast<uint32_t const*>(input), inputBytes};
        ClientRingChan clientRecv{recv.channel, nullptr, 0};
        hierarchicalReduceScatterDirectRing<ClientRingChan>
            <<<directRsBlocks, 512, 0, stream>>>(
                clientSend, clientRecv, t.ib, base, count, largeLayout, rank,
                pos, pieces, reinterpret_cast<float*>(output));
      } else {
        int pieces =
            bytes >= (4UL << 20)
                ? int(std::max<size_t>(
                      2, std::min<size_t>(
                             32, (bytes / 4 + (2UL << 20) - 1) / (2UL << 20))))
                : 1;
        hierarchicalAllReduce<DirectRingChan>
            <<<std::min(blocks, 8), 512, 0, stream>>>(
                send, recv, t.localAg, t.ib, base, count, largeLayout, rank % 4,
                pos, pieces, std::min(2, pieces), false);
      }
      checked(cudaGetLastError());
      size_t outputOffset =
          rs ? largeLayout.ibSend + size_t(rank / 4) * bytes / 8
             : largeLayout.data;
      if (!rs) {
        copyClient<<<256, 256, 0, stream>>>(
            reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t const*>(buffer + outputOffset),
            outputBytes / 4);
        checked(cudaGetLastError());
      }
      checked(cudaEventRecord(done, stream));
      lastStream = stream;
      pending = true;
      return mscclpp::CommResult::CommSuccess;
    }
    Layout bank = layout;
    bank.data += smallBase;
    bank.localScratch += smallBase;
    bank.ibSend += smallBase;
    bank.ibScratch += smallBase;
    if (sequence++ & 1) {
      bank.data += bankBytes;
      bank.localScratch += bankBytes;
      bank.ibSend += bankBytes;
      bank.ibScratch += bankBytes;
    }
    stagedCollective<<<1, 256, 0, stream>>>(
        t.localDirect, t.ib, rank, buffer, bank, op,
        reinterpret_cast<uint32_t const*>(input),
        reinterpret_cast<uint32_t*>(output), inputBytes / 4, outputBytes / 4);
    checked(cudaGetLastError());
    checked(cudaEventRecord(done, stream));
    lastStream = stream;
    pending = true;
    return mscclpp::CommResult::CommSuccess;
  }
};
}  // namespace

class NetworkCollectives::Impl {
 public:
  NetworkId id;
  std::shared_ptr<mscclpp::Communicator> comm;
  std::shared_ptr<Context> context;
  mscclpp::AlgorithmCollection algorithms;
  Impl(NetworkId uid, int rank, int device, int rail, size_t capacity)
      : id(uid),
        context(std::make_shared<Context>(rank, device, rail, capacity)) {
    if (rank < 0 || rank >= 8 || device < 0 || rail < 0 || rail >= 8 ||
        !capacity || capacity > (1UL << 30) || capacity % 128)
      throw std::invalid_argument("unsupported network configuration");
    for (int i = 1; i <= 3; ++i) {
      auto op = static_cast<NetworkOperation>(i);
      std::string name = i == 1   ? "allgather"
                         : i == 2 ? "reducescatter"
                                  : "allreduce";
      auto ctx = context;
      auto algorithm = std::make_shared<mscclpp::NativeAlgorithm>(
          "uccl_lite_gpu_hierarchy", name,
          [ctx](auto comm) { ctx->prepare(comm); },
          [ctx, op](auto, void const* input, void* output, size_t in,
                    size_t out, auto dtype, auto reduction, cudaStream_t stream,
                    int, int, auto const&, auto) {
            if (dtype != mscclpp::DataType::FLOAT32 ||
                (op != NetworkOperation::AllGather &&
                 reduction != mscclpp::ReduceOp::SUM))
              return mscclpp::CommResult::CommInvalidArgument;
            return ctx->run(op, input, output, in, out, stream);
          },
          [ctx](auto, auto, auto, size_t, size_t,
                auto) -> std::shared_ptr<void> { return ctx; },
          [capacity](auto, auto, size_t, size_t, auto, bool) {
            // Internal registered storage is stable even if client pointers,
            // sizes or in-place layouts differ between ranks and calls.
            return mscclpp::AlgorithmCtxKey{nullptr, nullptr, capacity,
                                            capacity, 0};
          });
      algorithms.registerAlgorithm(name, algorithm->name(), algorithm);
    }
    algorithms.setSelectors(
        [](auto const& map, auto const& request) {
          return map.at(request.collective).at("uccl_lite_gpu_hierarchy");
        },
        [](auto const&, auto const&) -> std::shared_ptr<mscclpp::Algorithm> {
          return nullptr;
        });
  }
};

NetworkId NetworkCollectives::createUniqueId() {
  return mscclpp::TcpBootstrap::createUniqueId();
}
NetworkCollectives::NetworkCollectives(NetworkId id, int rank, int device,
                                       int rail, size_t capacity)
    : impl_(std::make_unique<Impl>(id, rank, device, rail, capacity)) {}
NetworkCollectives::~NetworkCollectives() = default;
bool NetworkCollectives::initialized() const {
  return bool(impl_->context->transport);
}
bool NetworkCollectives::idle() const {
  auto const& context = *impl_->context;
  return !context.pending || cudaEventQuery(context.done) == cudaSuccess;
}
void NetworkCollectives::prepare() {
  auto& s = *impl_;
  if (initialized()) return;
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(s.context->rank, 8);
  bootstrap->initialize(s.id, 120);
  struct Metadata {
    size_t capacity;
    int device, rail;
    char host[256];
    char uuid[16];
  };
  std::array<Metadata, 8> metadata{};
  auto& local = metadata[s.context->rank];
  local.capacity = s.context->capacity;
  local.device = s.context->device;
  local.rail = s.context->rail;
  cudaDeviceProp property{};
  checked(cudaGetDeviceProperties(&property, local.device));
  std::memcpy(local.uuid, property.uuid.bytes, sizeof(local.uuid));
  if (gethostname(local.host, sizeof(local.host) - 1))
    throw std::runtime_error("host identity unavailable");
  bootstrap->allGather(metadata.data(), sizeof(Metadata));
  for (int r = 0; r < 8; ++r) {
    if (metadata[r].capacity != local.capacity || metadata[r].device != r % 4 ||
        metadata[r].rail != (r % 4) / 2 ||
        std::strcmp(metadata[r].host, metadata[r / 4 * 4].host) ||
        ((std::strcmp(metadata[r].host, metadata[0].host) == 0) != (r < 4)))
      throw std::invalid_argument("two-host placement/capacity mismatch");
    for (int p = r / 4 * 4; p < r; ++p)
      if (!std::memcmp(metadata[r].uuid, metadata[p].uuid, 16))
        throw std::invalid_argument("duplicate local GPU");
  }
  s.comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  s.context->prepare(s.comm);
}
void NetworkCollectives::enqueue(NetworkOperation op, void const* input,
                                 void* output, size_t count,
                                 cudaStream_t stream) {
  auto& s = *impl_;
  auto& c = *s.context;
  if (op != NetworkOperation::AllGather &&
      op != NetworkOperation::ReduceScatter &&
      op != NetworkOperation::AllReduce)
    throw std::invalid_argument("invalid network operation");
  if (!initialized()) throw std::logic_error("prepare before enqueue");
  int device;
  checked(cudaGetDevice(&device));
  size_t scale = op == NetworkOperation::AllReduce ? 4 : 32;
  if (device != c.device || !count || count > c.capacity / scale ||
      (count * scale) % 128 || !input || !output ||
      (reinterpret_cast<uintptr_t>(input) & 3) ||
      (reinterpret_cast<uintptr_t>(output) & 3))
    throw std::invalid_argument("unsupported network call");
  cudaStreamCaptureStatus capture;
  checked(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("network capture is unsupported");
  size_t bytes = count * scale;
  size_t in = op == NetworkOperation::AllGather ? bytes / 8 : bytes;
  size_t out = op == NetworkOperation::ReduceScatter ? bytes / 8 : bytes;
  std::string name = op == NetworkOperation::AllGather       ? "allgather"
                     : op == NetworkOperation::ReduceScatter ? "reducescatter"
                                                             : "allreduce";
  mscclpp::CollectiveRequest request{
      8,      4,      c.rank,
      input,  output, op == NetworkOperation::AllGather ? in : bytes,
      stream, name,   mscclpp::DataType::FLOAT32,
      {}};
  auto algorithm = s.algorithms.selectAlgorithm(request);
  auto result = algorithm->execute(s.comm, input, output, in, out,
                                   mscclpp::DataType::FLOAT32,
                                   mscclpp::ReduceOp::SUM, stream, nullptr);
  if (result != mscclpp::CommResult::CommSuccess)
    throw std::runtime_error("MSCCL++ algorithm execution failed");
}
}  // namespace uccl_lite

#include "uccl_lite/collectives/peer_staged.hpp"
#include "uccl_lite/peer_tree.hpp"
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
using Channel = mscclpp::BaseMemoryChannelDeviceHandle;
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
// Distinct kernel boundaries make publication follow completion of every
// block. Each peer has a separate MSCCL++ semaphore with exactly one writer.
// MSCCL++ signal() uses a remote system atomic in the installed API, which
// PCIe peer access alone does not guarantee. Publish a monotonically increasing
// sequence with a release store instead. Each outbound counter is local and
// owned by one thread; stream chaining serializes successive barriers. The
// paired MSCCL++ wait increments its expected sequence and loads with acquire.
__global__ void barrier(Channel* channels, uint64_t* sent, int peers) {
  if (threadIdx.x < peers) {
    auto sequence = ++sent[threadIdx.x];
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :
                 : "l"(channels[threadIdx.x].semaphore_.remoteInboundToken),
                   "l"(sequence)
                 : "memory");
  }
  __syncthreads();
  if (threadIdx.x < peers) channels[threadIdx.x].wait(-1);
}
template <class Bits>
__global__ void gather(void const* const* sources, Bits* output, size_t count,
                       int ranks) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count * ranks; i += size_t(gridDim.x) * blockDim.x)
    output[i] = static_cast<Bits const*>(sources[i / count])[i % count];
}
template <class T>
__device__ float decode(T value) {
  return value;
}
template <>
__device__ float decode(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <class T>
__device__ T encode(float value) {
  return value;
}
template <>
__device__ __nv_bfloat16 encode(float value) {
  return __float2bfloat16_rn(value);
}
template <class T>
__global__ void reduce(void const* const* sources, T* output, size_t count,
                       size_t offset, int ranks) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    float sum = 0;
    for (int source = 0; source < ranks; ++source)
      sum += decode(static_cast<T const*>(sources[source])[offset + i]);
    output[i] = encode<T>(sum);
  }
}
template <class Bits>
__global__ void treeCopy(Bits const* input, Bits* output, size_t count) {
  // A later semaphore kernel is not the producer of these stores. Each data
  // writer publishes its own writes at system scope before kernel completion;
  // volatile peer accesses avoid retaining payloads across forwarding rounds.
  Bits const volatile* source = input;
  Bits volatile* destination = output;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    destination[i] = source[i];
  __threadfence_system();
}
template <class T>
__global__ void treeAccumulate(T const* input, float* sum, size_t count) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    sum[i] += decode(input[i]);
}
template <class T>
__global__ void treeFinish(float const* sum, T* output, size_t count) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = encode<T>(sum[i]);
}
}  // namespace

class PeerStagedCollectives::Impl {
 public:
  NetworkId id;
  int rank, device, ranks;
  size_t capacity;
  ScalarType dtype;
  void* staging = nullptr;
  float* accumulator = nullptr;
  size_t relayOffset = 0;
  std::vector<int> parents, neighbors;
  std::vector<PeerTreeRoute> routes;
  void** deviceSources = nullptr;
  Channel* deviceChannels = nullptr;
  uint64_t* sent = nullptr;
  cudaEvent_t done = nullptr;
  cudaStream_t lastStream = nullptr;
  bool ready = false, recorded = false, failed = false;
  std::vector<cudaGraphExec_t> graphs;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::unique_ptr<mscclpp::Communicator> comm;
  std::vector<mscclpp::RegisteredMemory> memories;
  std::vector<mscclpp::Connection> connections;
  std::vector<mscclpp::BaseMemoryChannel> channels;

  Impl(NetworkId id, int rank, int device, int ranks, size_t capacity,
       ScalarType dtype, std::vector<int> tree = {})
      : id(id),
        rank(rank),
        device(device),
        ranks(ranks),
        capacity(capacity),
        dtype(dtype),
        parents(std::move(tree)) {
    if (ranks < 2 || ranks > 32 || rank < 0 || rank >= ranks || !capacity ||
        capacity > (1UL << 30) ||
        (dtype != ScalarType::BFloat16 && dtype != ScalarType::Float32))
      throw std::invalid_argument("invalid peer-staged configuration");
    if (!parents.empty()) {
      if (parents.size() != size_t(ranks))
        throw std::invalid_argument("peer tree rank extent mismatch");
      for (int source = 0; source < ranks; ++source)
        routes.push_back(peerTreeRoute(parents, source));
      relayOffset = (capacity + 255) & ~size_t(255);
    }
    for (int peer = 0; peer < ranks; ++peer)
      if (peer != rank &&
          (parents.empty() || parents[peer] == rank || parents[rank] == peer))
        neighbors.push_back(peer);
  }
  ~Impl() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (recorded) cudaEventSynchronize(done);
    for (auto graph : graphs)
      if (graph) cudaGraphExecDestroy(graph);
    // Completion includes the reader-retirement barrier, so no peer can
    // still reference this rank's staging data on a successful teardown.
    if (deviceChannels) cudaFree(deviceChannels);
    if (sent) cudaFree(sent);
    if (deviceSources) cudaFree(deviceSources);
    channels.clear();
    connections.clear();
    memories.clear();
    comm.reset();
    bootstrap.reset();
    if (staging) cudaFree(staging);
    if (accumulator) cudaFree(accumulator);
    if (done) cudaEventDestroy(done);
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
  }
};
NetworkId PeerStagedCollectives::createUniqueId() {
  return mscclpp::TcpBootstrap::createUniqueId();
}
PeerStagedCollectives::PeerStagedCollectives(NetworkId id, int rank, int device,
                                             int ranks, size_t capacity)
    : PeerStagedCollectives(id, rank, device, ranks, capacity,
                            ScalarType::BFloat16) {}
PeerStagedCollectives::PeerStagedCollectives(NetworkId id, int rank, int device,
                                             int ranks, size_t capacity,
                                             ScalarType dtype)
    : impl_(std::make_unique<Impl>(id, rank, device, ranks, capacity, dtype)) {}
PeerStagedCollectives::~PeerStagedCollectives() = default;
PeerStagedCollectives::PeerStagedCollectives(NetworkId id, int rank, int device,
                                             int ranks, size_t capacity,
                                             ScalarType dtype,
                                             std::vector<int> parents)
    : impl_(std::make_unique<Impl>(id, rank, device, ranks, capacity, dtype,
                                   std::move(parents))) {}
void PeerStagedCollectives::prepare() {
  auto& s = *impl_;
  if (s.ready) return;
  if (s.failed)
    throw std::logic_error("failed peer preparation requires a new executor");
  s.failed = true;
  s.bootstrap = std::make_shared<mscclpp::TcpBootstrap>(s.rank, s.ranks);
  s.bootstrap->initialize(s.id, 120);
  struct Metadata {
    size_t capacity = 0;
    int dtype = 0, tree = 0, device = -1, ready = 1;
    std::array<int, 32> parents;
    char host[256]{}, uuid[16]{};
  } local;
  local.capacity = s.capacity;
  local.dtype = int(s.dtype);
  local.tree = !s.parents.empty();
  local.device = s.device;
  local.parents.fill(-1);
  std::copy(s.parents.begin(), s.parents.end(), local.parents.begin());
  try {
    check(cudaSetDevice(s.device));
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, s.device));
    std::memcpy(local.uuid, property.uuid.bytes, sizeof(local.uuid));
    if (gethostname(local.host, sizeof(local.host) - 1))
      throw std::runtime_error("peer host identity unavailable");
  } catch (...) {
    local.ready = 0;
  }
  std::vector<Metadata> metadata(s.ranks);
  metadata[s.rank] = local;
  s.bootstrap->allGather(metadata.data(), sizeof(Metadata));
  for (auto const& m : metadata)
    if (!m.ready || m.capacity != local.capacity || m.dtype != local.dtype ||
        m.tree != local.tree || m.parents != local.parents ||
        std::strcmp(m.host, local.host))
      throw std::invalid_argument(
          "peer configuration/route/host consensus failed");
  int ready = 1;
  try {
    if (!s.parents.empty())
      for (int peer : s.neighbors) {
        auto const& m = metadata[peer];
        cudaDeviceProp property{};
        int access = 0;
        check(cudaGetDeviceProperties(&property, m.device));
        check(cudaDeviceCanAccessPeer(&access, s.device, m.device));
        if (!access || std::memcmp(property.uuid.bytes, m.uuid, sizeof(m.uuid)))
          throw std::invalid_argument(
              "required tree edge or CUDA view unavailable");
      }
  } catch (...) {
    ready = 0;
  }
  auto agreed = [&](int value, char const* message) {
    std::vector<int> votes(s.ranks);
    votes[s.rank] = value;
    s.bootstrap->allGather(votes.data(), sizeof(int));
    if (std::find(votes.begin(), votes.end(), 0) != votes.end())
      throw std::runtime_error(message);
  };
  agreed(ready, "peer tree edge capability consensus failed");
  try {
    check(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
    check(cudaMalloc(&s.staging, s.capacity + s.relayOffset));
    if (!s.parents.empty())
      check(cudaMalloc(&s.accumulator,
                       s.capacity / scalarBytes(s.dtype) * sizeof(float)));
  } catch (...) {
    ready = 0;
  }
  agreed(ready, "peer staging allocation failed collectively");
  s.comm = std::make_unique<mscclpp::Communicator>(s.bootstrap);
  s.memories.resize(s.ranks);
  s.connections.resize(s.ranks);
  s.memories[s.rank] = s.comm->registerMemory(
      s.staging, s.capacity + s.relayOffset, mscclpp::Transport::CudaIpc);
  std::vector<std::shared_future<mscclpp::Connection>> connections(s.ranks);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> memories(s.ranks);
  for (int peer : s.neighbors) {
    connections[peer] = s.comm->connect(mscclpp::Transport::CudaIpc, peer);
    s.comm->sendMemory(s.memories[s.rank], peer);
    memories[peer] = s.comm->recvMemory(peer);
  }
  std::vector<std::shared_future<mscclpp::Semaphore>> semaphores(s.ranks);
  for (int peer : s.neighbors) {
    s.connections[peer] = connections[peer].get();
    s.memories[peer] = memories[peer].get();
    semaphores[peer] = s.comm->buildSemaphore(s.connections[peer], peer);
  }
  std::vector<void*> sources(s.ranks);
  sources[s.rank] = s.staging;
  std::vector<Channel> handles;
  for (int peer : s.neighbors) {
    sources[peer] = s.memories[peer].data();
    s.channels.emplace_back(semaphores[peer].get());
    handles.push_back(s.channels.back().deviceHandle());
  }
  check(cudaMalloc(&s.deviceSources, sources.size() * sizeof(void*)));
  check(cudaMalloc(&s.deviceChannels, handles.size() * sizeof(Channel)));
  check(cudaMalloc(&s.sent, handles.size() * sizeof(uint64_t)));
  check(cudaMemset(s.sent, 0, handles.size() * sizeof(uint64_t)));
  check(cudaMemcpy(s.deviceSources, sources.data(),
                   sources.size() * sizeof(void*), cudaMemcpyHostToDevice));
  check(cudaMemcpy(s.deviceChannels, handles.data(),
                   handles.size() * sizeof(Channel), cudaMemcpyHostToDevice));
  check(cudaDeviceSynchronize());
  s.bootstrap->barrier();
  s.ready = true;
  s.failed = false;
}
void PeerStagedCollectives::enqueue(NetworkOperation op, void const* input,
                                    void* output, size_t count,
                                    cudaStream_t stream) {
  PeerGraphCall call{op, input, output, count};
  validate(call);
  chain(stream);
  submit(call, stream);
  record(stream);
}
void PeerStagedCollectives::validate(PeerGraphCall const& call) const {
  auto [op, input, output, count] = call;
  auto& s = *impl_;
  if (!s.ready) throw std::logic_error("peer-staged is not prepared");
  if (op != NetworkOperation::AllGather &&
      op != NetworkOperation::ReduceScatter &&
      op != NetworkOperation::AllReduce)
    throw std::invalid_argument("invalid peer-staged collective");
  size_t scalar = scalarBytes(s.dtype);
  size_t scale = scalar * (op == NetworkOperation::AllReduce ? 1 : s.ranks);
  if (!input || !output || !count || count > s.capacity / scale ||
      (reinterpret_cast<uintptr_t>(input) % scalar) ||
      (reinterpret_cast<uintptr_t>(output) % scalar))
    throw std::invalid_argument("invalid peer-staged extent or alignment");
  int device;
  check(cudaGetDevice(&device));
  if (device != s.device)
    throw std::invalid_argument("peer-staged current device mismatch");
}
void PeerStagedCollectives::chain(cudaStream_t stream) {
  auto& s = *impl_;
  cudaStreamCaptureStatus capture;
  check(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument(
        "use collective prepareGraph; external peer-staged capture is "
        "unsupported");
  if (s.recorded && stream != s.lastStream)
    check(cudaStreamWaitEvent(stream, s.done, 0));
}
void PeerStagedCollectives::submit(PeerGraphCall const& call,
                                   cudaStream_t stream) {
  auto [op, input, output, count] = call;
  auto& s = *impl_;
  size_t scalar = scalarBytes(s.dtype);
  size_t inputBytes =
      count * scalar * (op == NetworkOperation::ReduceScatter ? s.ranks : 1);
  check(cudaMemcpyAsync(s.staging, input, inputBytes, cudaMemcpyDeviceToDevice,
                        stream));
  if (!s.parents.empty()) {
    auto* relay = static_cast<unsigned char*>(s.staging) + s.relayOffset;
    auto copy = [&](void const* source, void* destination, size_t bytes) {
      size_t elements = bytes / scalar;
      int blocks = int(std::min<size_t>((elements + 255) / 256, 256));
      if (s.dtype == ScalarType::Float32)
        treeCopy<<<blocks, 256, 0, stream>>>(
            static_cast<uint32_t const*>(source),
            static_cast<uint32_t*>(destination), elements);
      else
        treeCopy<<<blocks, 256, 0, stream>>>(
            static_cast<unsigned short const*>(source),
            static_cast<unsigned short*>(destination), elements);
      check(cudaGetLastError());
    };
    auto sync = [&] {
      barrier<<<1, 32, 0, stream>>>(s.deviceChannels, s.sent,
                                    int(s.neighbors.size()));
      check(cudaGetLastError());
    };
    if (op != NetworkOperation::AllGather)
      check(cudaMemsetAsync(s.accumulator, 0, count * sizeof(float), stream));
    sync();
    int blocks = int(std::min<size_t>((count + 255) / 256, 256));
    for (int source = 0; source < s.ranks; ++source) {
      auto const& route = s.routes[source];
      for (int depth = 1; depth <= route.height; ++depth) {
        if (route.depth[s.rank] == depth) {
          int parent = route.parent[s.rank];
          auto* address =
              static_cast<unsigned char const*>(s.memories[parent].data());
          copy(address + (parent == source ? 0 : s.relayOffset), relay,
               inputBytes);
        }
        // A child copies its parent's completed prior level before publishing
        // retirement. Only neighboring ranks share memory/semaphore edges.
        sync();
      }
      auto* data =
          source == s.rank ? static_cast<unsigned char*>(s.staging) : relay;
      if (op == NetworkOperation::AllGather) {
        copy(data,
             static_cast<unsigned char*>(output) +
                 size_t(source) * count * scalar,
             count * scalar);
      } else {
        size_t offset = op == NetworkOperation::ReduceScatter
                            ? size_t(s.rank) * count * scalar
                            : 0;
        if (s.dtype == ScalarType::Float32)
          treeAccumulate<<<blocks, 256, 0, stream>>>(
              reinterpret_cast<float const*>(data + offset), s.accumulator,
              count);
        else
          treeAccumulate<<<blocks, 256, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16 const*>(data + offset),
              s.accumulator, count);
        check(cudaGetLastError());
      }
      // Protect relay reuse by the next source, and input reuse by the next
      // operation. Accumulation follows logical source order, independent of
      // the physical tree, and BF16 is rounded only after the final source.
      sync();
    }
    if (op != NetworkOperation::AllGather) {
      if (s.dtype == ScalarType::Float32)
        treeFinish<<<blocks, 256, 0, stream>>>(
            s.accumulator, static_cast<float*>(output), count);
      else
        treeFinish<<<blocks, 256, 0, stream>>>(
            s.accumulator, static_cast<__nv_bfloat16*>(output), count);
      check(cudaGetLastError());
    }
    return;
  }
  barrier<<<1, 32, 0, stream>>>(s.deviceChannels, s.sent, s.ranks - 1);
  check(cudaGetLastError());
  size_t outputCount =
      count * (op == NetworkOperation::AllGather ? s.ranks : 1);
  int blocks = int(std::min<size_t>((outputCount + 255) / 256, 256));
  size_t offset =
      op == NetworkOperation::ReduceScatter ? size_t(s.rank) * count : 0;
  if (op == NetworkOperation::AllGather) {
    if (s.dtype == ScalarType::Float32)
      gather<<<blocks, 256, 0, stream>>>(
          s.deviceSources, static_cast<uint32_t*>(output), count, s.ranks);
    else
      gather<<<blocks, 256, 0, stream>>>(s.deviceSources,
                                         static_cast<unsigned short*>(output),
                                         count, s.ranks);
  } else {
    if (s.dtype == ScalarType::Float32)
      reduce<<<blocks, 256, 0, stream>>>(
          s.deviceSources, static_cast<float*>(output), count, offset, s.ranks);
    else
      reduce<<<blocks, 256, 0, stream>>>(s.deviceSources,
                                         static_cast<__nv_bfloat16*>(output),
                                         count, offset, s.ranks);
  }
  check(cudaGetLastError());
  // Retire all readers before any rank overwrites staging or destroys the
  // communicator. The next local copy is ordered after this barrier.
  barrier<<<1, 32, 0, stream>>>(s.deviceChannels, s.sent, s.ranks - 1);
  check(cudaGetLastError());
}
void PeerStagedCollectives::record(cudaStream_t stream) {
  auto& s = *impl_;
  check(cudaEventRecord(s.done, stream));
  s.recorded = true;
  s.lastStream = stream;
}
size_t PeerStagedCollectives::prepareGraph(
    std::vector<PeerGraphCall> const& calls) {
  auto& s = *impl_;
  if (!s.ready) throw std::logic_error("peer-staged is not prepared");
  struct Metadata {
    size_t calls, index;
    int dtype, ready;
  } local{calls.size(), s.graphs.size(), int(s.dtype), 1};
  try {
    if (calls.empty() || calls.size() > 1024 || !idle())
      throw std::invalid_argument("invalid graph extent or active peer work");
    for (auto const& call : calls) validate(call);
    check(cudaDeviceSynchronize());
    s.graphs.reserve(s.graphs.size() + 1);
  } catch (std::exception const& error) {
    local.ready = 0;
    std::fprintf(stderr, "peer graph prepare rank=%d: %s\n", s.rank,
                 error.what());
  }
  std::vector<Metadata> metadata(s.ranks);
  metadata[s.rank] = local;
  s.bootstrap->allGather(metadata.data(), sizeof(Metadata));
  for (auto const& entry : metadata)
    if (!entry.ready || entry.calls != local.calls ||
        entry.index != local.index || entry.dtype != local.dtype)
      throw std::invalid_argument("peer graph preparation consensus failed");
  std::vector<uint64_t> operations(size_t(s.ranks) * calls.size() * 2);
  size_t stride = calls.size() * 2;
  for (size_t i = 0; i < calls.size(); ++i) {
    operations[size_t(s.rank) * stride + 2 * i] = uint64_t(calls[i].operation);
    operations[size_t(s.rank) * stride + 2 * i + 1] = calls[i].count;
  }
  s.bootstrap->allGather(operations.data(), int(stride * sizeof(uint64_t)));
  for (int rank = 1; rank < s.ranks; ++rank)
    if (!std::equal(operations.begin(), operations.begin() + stride,
                    operations.begin() + size_t(rank) * stride))
      throw std::invalid_argument(
          "peer graph operation/count sequence mismatch");
  cudaStream_t builder = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  int ready = 1;
  try {
    check(cudaStreamCreateWithFlags(&builder, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(builder, cudaStreamCaptureModeThreadLocal));
    for (auto const& call : calls) submit(call, builder);
    check(cudaStreamEndCapture(builder, &graph));
    check(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  } catch (std::exception const& error) {
    ready = 0;
    std::fprintf(stderr, "peer graph capture rank=%d: %s\n", s.rank,
                 error.what());
  }
  if (builder) {
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(builder, &capture) == cudaSuccess &&
        capture != cudaStreamCaptureStatusNone) {
      cudaGraph_t abandoned = nullptr;
      cudaStreamEndCapture(builder, &abandoned);
      if (abandoned) cudaGraphDestroy(abandoned);
    }
    cudaStreamDestroy(builder);
  }
  if (graph) cudaGraphDestroy(graph);
  std::vector<int> votes(s.ranks);
  votes[s.rank] = ready;
  s.bootstrap->allGather(votes.data(), sizeof(int));
  if (std::find(votes.begin(), votes.end(), 0) != votes.end()) {
    if (executable) cudaGraphExecDestroy(executable);
    throw std::runtime_error("peer graph construction failed collectively");
  }
  s.graphs.push_back(executable);
  return s.graphs.size() - 1;
}
void PeerStagedCollectives::launchGraph(size_t graph, cudaStream_t stream) {
  auto& s = *impl_;
  if (graph >= s.graphs.size() || !s.graphs[graph])
    throw std::invalid_argument("peer graph handle is absent or released");
  int device;
  check(cudaGetDevice(&device));
  if (device != s.device)
    throw std::invalid_argument("peer graph current device mismatch");
  chain(stream);
  check(cudaGraphLaunch(s.graphs[graph], stream));
  record(stream);
}
void PeerStagedCollectives::releaseGraph(size_t graph) {
  auto& s = *impl_;
  if (graph >= s.graphs.size() || !s.graphs[graph])
    throw std::invalid_argument("peer graph handle is absent or released");
  if (!idle())
    throw std::logic_error("peer graph release requires idle execution");
  check(cudaGraphExecDestroy(s.graphs[graph]));
  s.graphs[graph] = nullptr;
}
void PeerStagedCollectives::synchronize() {
  if (impl_->recorded) check(cudaEventSynchronize(impl_->done));
}
bool PeerStagedCollectives::initialized() const { return impl_->ready; }
bool PeerStagedCollectives::idle() const {
  return !impl_->recorded || cudaEventQuery(impl_->done) == cudaSuccess;
}
}  // namespace uccl_lite

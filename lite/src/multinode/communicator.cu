#include "gpu_transport.h"
#include "uccl_lite/collectives.h"
#include <linux/mempolicy.h>
#include <limits>
#include <type_traits>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace uccl_lite {
namespace {
void checkCuda(cudaError_t result, int line) {
  if (result != cudaSuccess)
    throw std::runtime_error("network CUDA line " + std::to_string(line) +
                             ": " + cudaGetErrorString(result));
}
#define check(result) checkCuda(result, __LINE__)
// MSCCL++ FIFO allocation binds the invoking thread, not just its workers.
// Keep that placement during its allocations, then restore the application.
struct CallerPlacement {
  cpu_set_t cpus{};
  int mode = 0;
  std::array<unsigned long, 16> nodes{};
  bool restored = false;
  CallerPlacement() {
    if (sched_getaffinity(0, sizeof(cpus), &cpus) ||
        syscall(SYS_get_mempolicy, &mode, nodes.data(),
                nodes.size() * sizeof(unsigned long) * 8, nullptr, 0))
      throw std::runtime_error("cannot snapshot caller CPU/NUMA placement");
  }
  bool restore() {
    auto mask = (mode & ~MPOL_MODE_FLAGS) == MPOL_DEFAULT ||
                        (mode & ~MPOL_MODE_FLAGS) == MPOL_LOCAL
                    ? nullptr
                    : nodes.data();
    bool memory = syscall(SYS_set_mempolicy, mode, mask,
                          nodes.size() * sizeof(unsigned long) * 8) == 0;
    bool affinity = sched_setaffinity(0, sizeof(cpus), &cpus) == 0;
    restored = memory && affinity;
    return restored;
  }
  ~CallerPlacement() {
    if (!restored) restore();
  }
};
struct CurrentDevice {
  int previous, target;
  CurrentDevice(int device) : target(device) {
    check(cudaGetDevice(&previous));
    if (previous != target) check(cudaSetDevice(target));
  }
  ~CurrentDevice() {
    if (previous != target) cudaSetDevice(previous);
  }
};
// Reusing the registered scratch requires every rank's preceding output copy
// to have completed, including when collective families/sizes change. Local
// arrival followed by a cross-host arrival covers all eight ranks. Semaphore
// counters are per communicator; this touches no process-global sync state.
__global__ void reusable(multinode_detail::LocalDirectChans local,
                         multinode_detail::PortChan ib, int localRank) {
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < 4; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < 4; ++peer)
      if (peer != localRank) local.c[peer].wait();
    ib.signal();
    ib.wait();
    ib.flush();
  }
}
}  // namespace
class MultinodeCommunicator::Impl {
 public:
  MultinodeId id;
  int rank, localRank, device, rail;
  size_t capacity;
  size_t bankBytes = 0;
  uint64_t sequence = 0;
  bool doubleBuffered = false;
  char* buffer = nullptr;
  cudaEvent_t done = nullptr;
  cudaEvent_t externalDone = nullptr;
  bool orderingChosen = false;
  bool pending = false;
  bool uncertain = false;
  multinode_detail::Layout layout{};
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::unique_ptr<mscclpp::Communicator> comm;
  std::unique_ptr<multinode_detail::GpuTransport> transport;
  Impl(MultinodeId uid, int r, int lr, int d, int net, size_t bytes,
       bool buffered)
      : id(uid),
        rank(r),
        localRank(lr),
        device(d),
        rail(net),
        capacity(bytes),
        doubleBuffered(buffered) {
    if (rank < 0 || rank >= 8 || localRank != rank % 4 || device < 0 ||
        rail < 0 || rail >= 8 || !capacity || capacity % 128 ||
        capacity > (1UL << 30))
      throw std::invalid_argument(
          "invalid two-host/four-rank network configuration");
  }
  ~Impl() {
    if (!buffer && !done) return;
    CurrentDevice restore(device);
    if (uncertain)
      cudaDeviceSynchronize();
    else if (pending)
      cudaEventSynchronize(externalDone ? externalDone : done);
    transport.reset();
    comm.reset();
    bootstrap.reset();
    if (done) cudaEventDestroy(done);
    if (buffer) cudaFree(buffer);
  }
  size_t validate(CollectiveCall const& call, bool pointers) const {
    if (call.dtype != ScalarType::Float32 || !call.count ||
        call.ports.size() != 1 ||
        (call.collective != Collective::AllReduce &&
         call.collective != Collective::AllGather &&
         call.collective != Collective::ReduceScatter))
      throw std::invalid_argument(
          "network backend requires positive FP32 AR/AG/RS");
    bool ar = call.collective == Collective::AllReduce;
    size_t expansion = ar ? 1 : 8;
    if (call.count % (ar ? 32 : 4) || call.count > capacity / 4 / expansion)
      throw std::invalid_argument("unsupported network extent/alignment");
    auto const& port = call.ports[0];
    if (port.device != device || !port.input || !port.output ||
        (reinterpret_cast<uintptr_t>(port.input) & 3) ||
        (reinterpret_cast<uintptr_t>(port.output) & 3))
      throw std::invalid_argument(
          "network port must reside on the configured GPU");
    cudaStreamCaptureStatus status;
    check(cudaStreamIsCapturing(port.stream, &status));
    if (status != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("network graph capture is unsupported");
    if (pointers) {
      for (auto address : {port.input, static_cast<void const*>(port.output)}) {
        cudaPointerAttributes attributes{};
        check(cudaPointerGetAttributes(&attributes, address));
        if (attributes.type != cudaMemoryTypeDevice ||
            attributes.device != device)
          throw std::invalid_argument(
              "network buffers must reside on their rank GPU");
      }
    }
    size_t bytes = call.count * 4 * expansion;
    size_t in = call.collective == Collective::AllGather ? bytes / 8 : bytes;
    size_t out =
        call.collective == Collective::ReduceScatter ? bytes / 8 : bytes;
    auto a = reinterpret_cast<uintptr_t>(port.input);
    auto b = reinterpret_cast<uintptr_t>(port.output);
    if (a > UINTPTR_MAX - in || b > UINTPTR_MAX - out)
      throw std::invalid_argument("network pointer range overflow");
    bool alias = call.collective == Collective::AllGather ? a == b + rank * in
                 : call.collective == Collective::ReduceScatter
                     ? b == a + rank * out
                     : a == b;
    if ((call.inPlace && !alias) ||
        (!call.inPlace && a < b + out && b < a + in))
      throw std::invalid_argument("network in-place flag/layout mismatch");
    return bytes;
  }
};
MultinodeId MultinodeCommunicator::createUniqueId() {
  return mscclpp::TcpBootstrap::createUniqueId();
}
MultinodeCommunicator::MultinodeCommunicator(MultinodeId id, int rank,
                                             int localRank, int device,
                                             int rail, size_t maxBytes)
    : MultinodeCommunicator(id, rank, localRank, device, rail, maxBytes,
                            false) {}
MultinodeCommunicator::MultinodeCommunicator(MultinodeId id, int rank,
                                             int localRank, int device,
                                             int rail, size_t maxBytes,
                                             bool doubleBuffered)
    : impl_(std::make_unique<Impl>(id, rank, localRank, device, rail, maxBytes,
                                   doubleBuffered)) {}
MultinodeCommunicator::~MultinodeCommunicator() = default;
bool MultinodeCommunicator::initialized() const {
  return bool(impl_->transport);
}
void MultinodeCommunicator::prepare(CollectiveCall const& call) {
  auto& s = *impl_;
  CurrentDevice restore(s.device);
  s.validate(call, true);
  check(cudaDeviceSynchronize());
  if (s.transport) return;
  if (s.buffer)
    throw std::logic_error("failed network preparation cannot be retried");
  CallerPlacement placement;
  if (s.rail >= mscclpp::getIBDeviceCount())
    throw std::invalid_argument("IB rail unavailable");
  // Match the retained benchmark's registered layout. All collectives share
  // the same allocation so families can change without re-registering memory.
  size_t ibBytes = s.capacity / 4 + 4096;
  s.layout = {0, s.capacity, 2 * s.capacity, 2 * s.capacity + ibBytes, 0};
  s.bankBytes = s.layout.ibScratch + ibBytes;
  size_t registeredBytes = s.bankBytes * (s.doubleBuffered ? 2 : 1);
  if (registeredBytes >= (1ULL << 32))
    throw std::invalid_argument(
        "network registration exceeds 32-bit port offsets");
  check(cudaMalloc(&s.buffer, registeredBytes));
  check(cudaMemset(s.buffer, 0, registeredBytes));
  check(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
  s.bootstrap = std::make_shared<mscclpp::TcpBootstrap>(s.rank, 8);
  s.bootstrap->initialize(s.id, 120);
  struct Metadata {
    size_t capacity;
    int device, localRank, rail, doubleBuffered;
    char host[256];
    char uuid[16];
  };
  std::array<Metadata, 8> metadata{};
  metadata[s.rank] = {
      s.capacity, s.device, s.localRank, s.rail, int(s.doubleBuffered), {}, {}};
  cudaDeviceProp property{};
  check(cudaGetDeviceProperties(&property, s.device));
  std::memcpy(metadata[s.rank].uuid, property.uuid.bytes, 16);
  if (gethostname(metadata[s.rank].host, 255))
    throw std::runtime_error("network host identity unavailable");
  s.bootstrap->allGather(metadata.data(), sizeof(Metadata));
  for (int r = 0; r < 8; ++r) {
    if (metadata[r].capacity != s.capacity ||
        metadata[r].doubleBuffered != int(s.doubleBuffered) ||
        metadata[r].localRank != r % 4 ||
        metadata[r].rail != metadata[r ^ 4].rail ||
        (std::strcmp(metadata[r].host, metadata[0].host) == 0) != (r < 4))
      throw std::invalid_argument(
          "network rank/host/capacity mapping mismatch");
    for (int p = r / 4 * 4; p < r; ++p)
      if (!std::memcmp(metadata[r].uuid, metadata[p].uuid, 16))
        throw std::invalid_argument(
            "network ranks must use distinct local GPUs");
    if (std::strcmp(metadata[r].host, metadata[r / 4 * 4].host))
      throw std::invalid_argument("network ranks must be consecutive by host");
  }
  s.comm = std::make_unique<mscclpp::Communicator>(s.bootstrap);
  s.transport = std::make_unique<multinode_detail::GpuTransport>(
      *s.comm, s.rank, s.localRank, s.rail, s.buffer, registeredBytes, true,
      false, false, true);
  // Allocation helpers return their synchronized temporary streams to this
  // pool. Release the idle cache at the explicit preparation boundary, so
  // those streams cannot outlive a caller's final cudaDeviceReset (nccl-tests).
  mscclpp::gpuStreamPool()->clear();
  if (!placement.restore())
    throw std::runtime_error("cannot restore caller CPU/NUMA placement");
}
void MultinodeCommunicator::enqueue(CollectiveCall const& call) {
  enqueueImpl(call, nullptr);
}
void MultinodeCommunicator::enqueueOrdered(CollectiveCall const& call,
                                           cudaEvent_t completion) {
  if (!completion)
    throw std::invalid_argument(
        "ordered network enqueue requires a completion event");
  enqueueImpl(call, completion);
}
void MultinodeCommunicator::enqueueImpl(CollectiveCall const& call,
                                        cudaEvent_t completion) {
  using namespace multinode_detail;
  auto& s = *impl_;
  CurrentDevice restore(s.device);
  size_t bytes = s.validate(call, false);
  if (!s.transport)
    throw std::logic_error("prepare network communicator before enqueue");
  if (s.orderingChosen && s.externalDone != completion)
    throw std::logic_error("network ordering mode/completion event changed");
  s.orderingChosen = true;
  s.externalDone = completion;
  auto const& p = call.ports[0];
  if (!completion && s.pending) check(cudaStreamWaitEvent(p.stream, s.done, 0));
  s.uncertain = true;
  if (!s.doubleBuffered) {
    reusable<<<1, 32, 0, p.stream>>>(s.transport->localDirect, s.transport->ib,
                                     s.localRank);
    check(cudaGetLastError());
  }
  Layout layout = s.layout;
  if (s.doubleBuffered && (s.sequence & 1)) {
    layout.data += s.bankBytes;
    layout.localScratch += s.bankBytes;
    layout.ibSend += s.bankBytes;
    layout.ibScratch += s.bankBytes;
  }
  bool ag = call.collective == Collective::AllGather;
  bool rs = call.collective == Collective::ReduceScatter;
  if (!completion)
    check(cudaMemcpyAsync(s.buffer + (ag ? layout.ibSend : layout.data),
                          p.input, ag ? bytes / 8 : bytes,
                          cudaMemcpyDeviceToDevice, p.stream));
  size_t elements = bytes / 4;
  auto& t = *s.transport;
  auto launch = [&](auto direct) {
    constexpr bool ClientIO = decltype(direct)::value;
    if (ag)
      hierarchicalAllGatherSerial<ClientIO><<<1, 256, 0, p.stream>>>(
          t.localDirect, t.ib, reinterpret_cast<float*>(s.buffer), elements,
          layout, s.rank, s.localRank,
          reinterpret_cast<uint32_t const*>(p.input),
          reinterpret_cast<uint32_t*>(p.output));
    else if (rs)
      hierarchicalReduceScatterSerial<ClientIO><<<1, 256, 0, p.stream>>>(
          t.localDirect, t.ib, reinterpret_cast<float*>(s.buffer), elements,
          layout, s.rank, s.localRank,
          reinterpret_cast<uint32_t const*>(p.input),
          reinterpret_cast<uint32_t*>(p.output));
    else {
      constexpr int position[4] = {0, 1, 3, 2};
      hierarchicalAllReduceSerial<ClientIO><<<1, 256, 0, p.stream>>>(
          t.localDirect, t.ib, reinterpret_cast<float*>(s.buffer), elements,
          layout, s.localRank, position[s.localRank],
          reinterpret_cast<uint32_t const*>(p.input),
          reinterpret_cast<uint32_t*>(p.output));
    }
  };
  if (completion)
    launch(std::true_type{});
  else
    launch(std::false_type{});
  check(cudaGetLastError());
  size_t offset =
      rs ? layout.ibSend + size_t(s.rank / 4) * bytes / 8 : layout.data;
  if (!completion)
    check(cudaMemcpyAsync(p.output, s.buffer + offset, rs ? bytes / 8 : bytes,
                          cudaMemcpyDeviceToDevice, p.stream));
  if (!completion) check(cudaEventRecord(s.done, p.stream));
  ++s.sequence;
  s.pending = true;
  s.uncertain = false;
}
}  // namespace uccl_lite

#include "uccl_lite/collectives/direct_exchange.hpp"
#include "uccl_lite/network_plan.hpp"
#include "uccl_lite/runtime/build_identity.hpp"
#include <linux/mempolicy.h>
#include <mscclpp/core.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/utils.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cuda_bf16.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Placement {
  cpu_set_t cpus{};
  int mode = 0;
  std::array<unsigned long, 16> nodes{};
  bool active = true;
  Placement() {
    if (sched_getaffinity(0, sizeof(cpus), &cpus) ||
        syscall(SYS_get_mempolicy, &mode, nodes.data(), sizeof(nodes) * 8,
                nullptr, 0))
      throw std::runtime_error("cannot capture direct exchange placement");
  }
  void restore() {
    if (!active) return;
    int base = mode & ~MPOL_MODE_FLAGS;
    if (syscall(
            SYS_set_mempolicy, mode,
            base == MPOL_DEFAULT || base == MPOL_LOCAL ? nullptr : nodes.data(),
            sizeof(nodes) * 8) ||
        sched_setaffinity(0, sizeof(cpus), &cpus))
      throw std::runtime_error("cannot restore direct exchange placement");
    active = false;
  }
  ~Placement() {
    try {
      restore();
    } catch (...) {
    }
  }
};
std::string uuid(cudaUUID_t const& id) {
  std::ostringstream s;
  for (auto byte : id.bytes)
    s << std::hex << std::setw(2) << std::setfill('0')
      << unsigned(uint8_t(byte));
  return s.str();
}
template <class T>
__device__ float read(T x) {
  return float(x);
}
template <>
__device__ float read(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <class T>
__device__ T rounded(float x) {
  return x;
}
template <>
__device__ __nv_bfloat16 rounded(float x) {
  return __float2bfloat16_rn(x);
}
// Exactly one thread owns each output. All source slots have completed their
// transport writes before launch. Accumulate in logical source order, not
// arrival order or physical rank order; BF16 is rounded only at the output.
template <class T>
__global__ void reduceSources(T const* slots, T* output, size_t count,
                              size_t stride, int ranks) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    float sum = 0;
    for (int source = 0; source < ranks; ++source)
      sum = __fadd_rn(sum, read(slots[size_t(source) * stride + i]));
    output[i] = rounded<T>(sum);
  }
}
mscclpp::Transport ib(int rail) {
  return static_cast<mscclpp::Transport>(int(mscclpp::Transport::IB0) + rail);
}
}  // namespace
class DirectExchangeCollectives::Impl {
 public:
  NetworkId id;
  int rank, ranks, device;
  size_t capacity, chunk;
  ScalarType dtype;
  bool ready = false, receivingIb = false;
  int ordering = 0;
  char* arena = nullptr;
  std::shared_ptr<mscclpp::TcpBootstrap> control;
  std::unique_ptr<mscclpp::Communicator> comm;
  std::array<mscclpp::RegisteredMemory, 2> local;
  std::vector<mscclpp::RegisteredMemory> remote;
  std::vector<mscclpp::Connection> connections;
  std::vector<DirectExchangeRoute> routes;
  std::vector<bool> ipc;
  std::array<std::vector<DirectStep>, 3> steps;
  std::string identity;
  Topology topology;
  std::vector<std::string> gpuIds, nicIds;
  std::vector<int> peerDevices;
  std::vector<double> nicRates;
  Impl(NetworkId id, int rank, int ranks, int device, size_t capacity,
       ScalarType dtype, size_t chunk)
      : id(id),
        rank(rank),
        ranks(ranks),
        device(device),
        capacity(capacity),
        chunk(chunk),
        dtype(dtype) {
    size_t scalar = scalarBytes(dtype);
    if (rank < 0 || rank >= ranks || ranks < 2 || ranks > 32 || device < 0 ||
        (dtype != ScalarType::Float32 && dtype != ScalarType::BFloat16) ||
        capacity < scalar || capacity > (1UL << 30) || chunk < scalar ||
        chunk > (1UL << 20) || chunk % scalar)
      throw std::invalid_argument("invalid direct exchange context");
    this->chunk = std::min(chunk, capacity / scalar * scalar);
  }
  ~Impl() {
    // A discarded, unprepared candidate must not initialize a CUDA context or
    // touch its (as yet unvalidated) device. GPU resources imply arena exists.
    if (!arena) return;
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    connections.clear();
    remote.clear();
    local = {};
    comm.reset();
    if (arena) cudaFree(arena);
    control.reset();
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
  }
  template <class T>
  std::vector<T> gather(T const& own) {
    std::vector<T> result(ranks);
    result[rank] = own;
    control->allGather(result.data(), sizeof(T));
    return result;
  }
  template <class F>
  void collective(F f) {
    DirectStatus status;
    try {
      f();
    } catch (std::exception const& error) {
      std::snprintf(status.message, sizeof(status.message), "%s", error.what());
    }
    for (auto const& peer : gather(status))
      if (*peer.message) throw std::runtime_error(peer.message);
  }
  void entry(NetworkOperation op, void const* input, void* output, size_t count,
             cudaStream_t stream) {
    auto& s = *this;
    DirectCallSignature call{count, int(op)};
    s.collective([&] {
      s.current();
      cudaStreamCaptureStatus capture;
      check(cudaStreamIsCapturing(stream, &capture));
      size_t scalar = scalarBytes(s.dtype),
             scale = scalar * (int(op) == 3 ? 1 : s.ranks);
      if (capture != cudaStreamCaptureStatusNone || int(op) < 1 ||
          int(op) > 3 || !count || count > s.capacity / scale || !input ||
          !output || (reinterpret_cast<uintptr_t>(input) % scalar) ||
          (reinterpret_cast<uintptr_t>(output) % scalar))
        throw std::invalid_argument("unsupported direct exchange call");
      auto a = reinterpret_cast<uintptr_t>(input),
           b = reinterpret_cast<uintptr_t>(output);
      size_t in = count * scalar * (int(op) == 2 ? s.ranks : 1);
      size_t out = count * scalar * (int(op) == 1 ? s.ranks : 1);
      bool overlap = a <= b ? b - a < in : a - b < out;
      bool legal = int(op) == 1   ? a == b + size_t(s.rank) * count * scalar
                   : int(op) == 2 ? b == a + size_t(s.rank) * count * scalar
                                  : a == b;
      if (overlap && !legal)
        throw std::invalid_argument("unsupported partial alias");
    });
    for (auto p : s.gather(call))
      if (p.count != count || p.op != int(op))
        throw std::invalid_argument("direct exchange call mismatch");
    s.collective([&] { check(cudaStreamSynchronize(stream)); });
  }
  std::vector<std::string> strings(std::string const& text) {
    auto sizes = gather(text.size());
    size_t width = *std::max_element(sizes.begin(), sizes.end());
    if (!width || width > (1UL << 20))
      throw std::runtime_error("direct model metadata extent invalid");
    std::vector<char> bytes(width * ranks);
    std::memcpy(bytes.data() + width * rank, text.data(), text.size());
    control->allGather(bytes.data(), int(width));
    std::vector<std::string> result;
    for (int r = 0; r < ranks; ++r)
      result.emplace_back(bytes.data() + r * width, sizes[r]);
    return result;
  }
  void current() const {
    int actual;
    check(cudaGetDevice(&actual));
    if (actual != device)
      throw std::invalid_argument("direct exchange device changed");
  }
};
NetworkId DirectExchangeCollectives::createUniqueId() {
  NetworkId result;
  auto id = mscclpp::TcpBootstrap::createUniqueId();
  std::memcpy(result.data(), &id, result.size());
  return result;
}
bool DirectExchangeCollectives::supportsCurrentDevice(ScalarType dtype) {
  if (dtype != ScalarType::Float32 && dtype != ScalarType::BFloat16)
    return false;
  int device, rdma, ordering, flush;
  check(cudaGetDevice(&device));
  check(
      cudaDeviceGetAttribute(&rdma, cudaDevAttrGPUDirectRDMASupported, device));
  check(cudaDeviceGetAttribute(&ordering,
                               cudaDevAttrGPUDirectRDMAWritesOrdering, device));
  check(cudaDeviceGetAttribute(
      &flush, cudaDevAttrGPUDirectRDMAFlushWritesOptions, device));
  if (!rdma || (ordering < cudaGPUDirectRDMAWritesOrderingOwner &&
                !(flush & cudaFlushGPUDirectRDMAWritesOptionHost)))
    return false;
  cudaFuncAttributes attributes{};
  auto status =
      dtype == ScalarType::Float32
          ? cudaFuncGetAttributes(&attributes, reduceSources<float>)
          : cudaFuncGetAttributes(&attributes, reduceSources<__nv_bfloat16>);
  if (status != cudaSuccess) {
    if (status != cudaErrorInvalidDeviceFunction &&
        status != cudaErrorNoKernelImageForDevice &&
        status != cudaErrorUnsupportedPtxVersion)
      check(status);
    auto pending = cudaGetLastError();
    if (pending != cudaSuccess && pending != status) check(pending);
    return false;
  }
  auto t = inspectTopology({device});
  auto const& configured = mscclpp::env()->hcaDevices;
  int count =
      configured.empty()
          ? mscclpp::getIBDeviceCount()
          : int(std::count(configured.begin(), configured.end(), ',')) + 1;
  std::vector<RailDevice> rails;
  for (int r = 0; r < std::min(8, count); ++r) {
    auto name = mscclpp::getIBDeviceName(ib(r));
    if (!name.empty()) rails.push_back({name, r});
  }
  return !rails.empty() &&
         !networkRailCandidates(t, rails, true).front().empty();
}
DirectExchangeCollectives::DirectExchangeCollectives(NetworkId id, int rank,
                                                     int ranks, int device,
                                                     size_t capacity,
                                                     ScalarType dtype,
                                                     size_t chunk)
    : impl_(std::make_unique<Impl>(id, rank, ranks, device, capacity, dtype,
                                   chunk)) {}
DirectExchangeCollectives::~DirectExchangeCollectives() = default;
void DirectExchangeCollectives::prepare() {
  auto& s = *impl_;
  if (s.control)
    throw std::logic_error("direct exchange prepare is single-use");
  s.control = std::make_shared<mscclpp::TcpBootstrap>(s.rank, s.ranks);
  s.control->initialize(s.id, 120);
  std::unique_ptr<Placement> placement;
  s.collective([&] { placement = std::make_unique<Placement>(); });
  struct Participant {
    int rank = -1, device = -1, pid = -1, dtype = -1;
    size_t capacity = 0, chunk = 0;
    char host[256]{}, boot[64]{}, gpu[33]{};
  } own;
  std::map<std::string, int> visible;
  s.collective([&] {
    s.current();
    cudaFuncAttributes attributes{};
    if (s.dtype == ScalarType::Float32)
      check(cudaFuncGetAttributes(&attributes, reduceSources<float>));
    else
      check(cudaFuncGetAttributes(&attributes, reduceSources<__nv_bfloat16>));
    own.rank = s.rank;
    own.device = s.device;
    own.pid = getpid();
    own.dtype = int(s.dtype);
    own.capacity = s.capacity;
    own.chunk = s.chunk;
    if (gethostname(own.host, sizeof(own.host) - 1))
      throw std::runtime_error("direct exchange hostname unavailable");
    std::string boot;
    std::ifstream("/proc/sys/kernel/random/boot_id") >> boot;
    if (boot.empty() || boot.size() >= sizeof(own.boot))
      throw std::runtime_error("direct exchange host identity unavailable");
    std::memcpy(own.boot, boot.c_str(), boot.size() + 1);
    int count;
    check(cudaGetDeviceCount(&count));
    for (int device = 0; device < count; ++device) {
      cudaDeviceProp property{};
      check(cudaGetDeviceProperties(&property, device));
      auto key = uuid(property.uuid);
      visible.emplace(key, device);
      if (device == s.device) std::memcpy(own.gpu, key.c_str(), key.size() + 1);
    }
  });
  auto peers = s.gather(own);
  s.peerDevices.assign(s.ranks, -1);
  for (auto const& peer : peers)
    s.gpuIds.push_back(std::string(peer.boot) + '/' + peer.gpu);
  auto sameHost = [&](int a, int b) {
    return !std::strcmp(peers[a].boot, peers[b].boot);
  };
  std::vector<int> localDevices;
  uint32_t mask = 0;
  s.collective([&] {
    for (int r = 0; r < s.ranks; ++r) {
      auto const& p = peers[r];
      if (p.rank != r || p.dtype != own.dtype || p.capacity != own.capacity ||
          p.chunk != own.chunk || !*p.gpu || !*p.boot)
        throw std::invalid_argument(
            "direct exchange participant/config mismatch");
      for (int q = 0; q < r; ++q)
        if (sameHost(r, q) &&
            (p.pid == peers[q].pid || !std::strcmp(p.gpu, peers[q].gpu)))
          throw std::invalid_argument(
              "direct exchange requires unique process/GPU ownership");
      auto d = visible.find(p.gpu);
      if (sameHost(s.rank, r) && d != visible.end()) {
        localDevices.push_back(d->second);
        s.peerDevices[r] = d->second;
        int access = 0;
        if (r != s.rank)
          check(cudaDeviceCanAccessPeer(&access, s.device, d->second));
        if (access) mask |= uint32_t(1) << r;
      }
    }
  });
  auto masks = s.gather(mask);
  s.ipc.resize(size_t(s.ranks) * s.ranks, false);
  for (int source = 0; source < s.ranks; ++source)
    for (int target = 0; target < s.ranks; ++target)
      if (source != target)
        s.ipc[size_t(source) * s.ranks + target] =
            sameHost(source, target) &&
            (masks[source] & (uint32_t(1) << target)) &&
            (masks[target] & (uint32_t(1) << source));
  for (int op = 0; op < 3; ++op)
    s.steps[op] = directExchangeSteps(Collective(op), s.ranks, s.rank);
  for (int r = 0; r < s.ranks; ++r) {
    if (r == s.rank) continue;
    bool ipc = sameHost(s.rank, r) && (mask & (uint32_t(1) << r)) &&
               (masks[r] & (uint32_t(1) << s.rank));
    s.routes.push_back({r, ipc});
    s.receivingIb |= !ipc;
  }
  struct Nic {
    int rail = -1;
    char name[64]{};
    bool measured = false;
    int port = -1;
    double bandwidth = 0;
  } nic;
  std::string topology;
  s.collective([&] {
    auto t = inspectTopology(localDevices);
    topology = t.identity;
    s.topology = t;
    if (!s.receivingIb) return;
    int rdma, flush;
    check(cudaDeviceGetAttribute(&rdma, cudaDevAttrGPUDirectRDMASupported,
                                 s.device));
    check(cudaDeviceGetAttribute(
        &s.ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, s.device));
    check(cudaDeviceGetAttribute(
        &flush, cudaDevAttrGPUDirectRDMAFlushWritesOptions, s.device));
    if (!rdma || (s.ordering < cudaGPUDirectRDMAWritesOrderingOwner &&
                  !(flush & cudaFlushGPUDirectRDMAWritesOptionHost)))
      throw std::invalid_argument(
          "direct exchange GPU RDMA/visibility unsupported");
    auto const& configured = mscclpp::env()->hcaDevices;
    int count =
        configured.empty()
            ? mscclpp::getIBDeviceCount()
            : int(std::count(configured.begin(), configured.end(), ',')) + 1;
    std::vector<RailDevice> rails;
    for (int r = 0; r < std::min(8, count); ++r) {
      auto name = mscclpp::getIBDeviceName(ib(r));
      if (!name.empty()) rails.push_back({name, r});
    }
    auto position =
        std::find(localDevices.begin(), localDevices.end(), s.device) -
        localDevices.begin();
    RailAssignment selected;
    try {
      selected = planNetworkRails(t, rails).at(position);
    } catch (std::invalid_argument const&) {
      // Explicit unmeasured transport prior; not automatic algorithm admission.
      // Unknown PCIe/NUMA rates remain unknown, rather than fabricated values.
      auto candidates = networkRailCandidates(t, rails, true).at(position);
      if (candidates.empty())
        throw std::invalid_argument("no active direct exchange NIC");
      selected = candidates.front().assignment;
    }
    nic.rail = selected.transportIndex;
    std::snprintf(nic.name, sizeof(nic.name), "%s", selected.nic.c_str());
    for (auto const& port : t.networkPorts)
      if (port.name == selected.nic) {
        nic.measured = port.transmit.evidence == Evidence::Measured;
        nic.port = port.port;
        nic.bandwidth = port.transmit.bandwidthGBs;
      }
  });
  auto nics = s.gather(nic);
  for (int r = 0; r < s.ranks; ++r) {
    s.nicIds.push_back(std::string(peers[r].boot) + '/' + nics[r].name + ':' +
                       std::to_string(nics[r].port));
    s.nicRates.push_back(nics[r].bandwidth);
  }
  s.collective([&] {
    std::ostringstream identity;
    identity << "direct-exchange-v1;source-ordered-fp32;dtype=" << int(s.dtype)
             << ";chunk=" << s.chunk << ";capacity=" << s.capacity
             << ";artifact="
             << runtime::artifactIdentity(
                    reinterpret_cast<uintptr_t>(&createUniqueId));
    for (int r = 0; r < s.ranks; ++r)
      identity << ';' << peers[r].boot << ':' << peers[r].gpu << ':'
               << nics[r].name << ':' << nics[r].rail << ':'
               << nics[r].measured;
    identity << ";local=" << topology;
    s.identity = identity.str();
    check(cudaMalloc(&s.arena, (s.ranks + 1) * s.chunk));
    s.comm = std::make_unique<mscclpp::Communicator>(s.control);
    bool ipc = std::any_of(s.routes.begin(), s.routes.end(),
                           [](auto r) { return r.cudaIpc; });
    if (ipc)
      s.local[0] = s.comm->registerMemory(s.arena, (s.ranks + 1) * s.chunk,
                                          mscclpp::Transport::CudaIpc);
    if (s.receivingIb)
      s.local[1] = s.comm->registerMemory(s.arena, (s.ranks + 1) * s.chunk,
                                          ib(nic.rail));
  });
  s.remote.resize(s.ranks);
  s.connections.resize(s.ranks);
  s.collective([&] {
    std::vector<std::shared_future<mscclpp::Connection>> connections;
    std::vector<std::shared_future<mscclpp::RegisteredMemory>> memories;
    for (auto route : s.routes) {
      int type = route.cudaIpc ? 0 : 1;
      auto transport =
          route.cudaIpc ? mscclpp::Transport::CudaIpc : ib(nic.rail);
      connections.push_back(s.comm->connect(transport, route.peer, 710));
      s.comm->sendMemory(s.local[type], route.peer, 711);
      memories.push_back(s.comm->recvMemory(route.peer, 711));
    }
    for (size_t i = 0; i < s.routes.size(); ++i) {
      int r = s.routes[i].peer;
      s.connections[r] = connections[i].get();
      s.remote[r] = memories[i].get();
    }
    auto probe = cudaGetLastError();
    if (probe != cudaSuccess && probe != cudaErrorInvalidResourceHandle &&
        probe != cudaErrorPeerAccessAlreadyEnabled)
      check(probe);
    check(cudaDeviceSynchronize());
  });
  s.collective([&] { placement->restore(); });
  s.control->barrier();
  s.ready = true;
}
void DirectExchangeCollectives::enqueue(NetworkOperation op, void const* input,
                                        void* output, size_t count,
                                        cudaStream_t stream) {
  auto& s = *impl_;
  if (!s.ready) throw std::logic_error("direct exchange not prepared");
  s.entry(op, input, output, count, stream);
  size_t scalar = scalarBytes(s.dtype), stride = s.chunk / scalar;
  for (size_t begin = 0; begin < count; begin += stride) {
    size_t n = std::min(stride, count - begin), bytes = n * scalar;
    s.collective([&] {
      auto copyInput = [&](int target) {
        size_t offset =
            (begin + (int(op) == 2 ? size_t(target) * count : 0)) * scalar;
        check(cudaMemcpyAsync(s.arena, static_cast<char const*>(input) + offset,
                              bytes, cudaMemcpyDeviceToDevice, stream));
        check(cudaStreamSynchronize(stream));
      };
      auto collective =
          op == NetworkOperation::AllReduce   ? Collective::AllReduce
          : op == NetworkOperation::AllGather ? Collective::AllGather
                                              : Collective::ReduceScatter;
      for (auto step : s.steps[size_t(collective)]) {
        int target = step.target;
        if (step.kind == DirectStepKind::StageInput) {
          copyInput(target);
          continue;
        }
        size_t offset = (1 + size_t(s.rank)) * s.chunk;
        if (target == s.rank) {
          check(cudaMemcpyAsync(s.arena + offset, s.arena, bytes,
                                cudaMemcpyDeviceToDevice, stream));
          check(cudaStreamSynchronize(stream));
        } else {
          bool ipc = s.ipc[size_t(s.rank) * s.ranks + target];
          s.connections[target].write(s.remote[target], offset,
                                      s.local[ipc ? 0 : 1], 0, bytes);
          s.connections[target].flush(30000000);
        }
      }
    });
    // collective() has completed every sender's writes before receiver launch.
    s.collective([&] {
      if (s.receivingIb && s.ordering < cudaGPUDirectRDMAWritesOrderingOwner)
        check(cudaDeviceFlushGPUDirectRDMAWrites(
            cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
            cudaFlushGPUDirectRDMAWritesToOwner));
      if (int(op) == 1) {
        for (int source = 0; source < s.ranks; ++source)
          check(cudaMemcpyAsync(static_cast<char*>(output) +
                                    (size_t(source) * count + begin) * scalar,
                                s.arena + (1 + size_t(source)) * s.chunk, bytes,
                                cudaMemcpyDeviceToDevice, stream));
      } else {
        int blocks = int(std::min<size_t>(65535, (n + 255) / 256));
        if (s.dtype == ScalarType::Float32)
          reduceSources<<<blocks, 256, 0, stream>>>(
              reinterpret_cast<float*>(s.arena + s.chunk),
              static_cast<float*>(output) + begin, n, stride, s.ranks);
        else
          reduceSources<<<blocks, 256, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(s.arena + s.chunk),
              static_cast<__nv_bfloat16*>(output) + begin, n, stride, s.ranks);
        check(cudaGetLastError());
      }
      check(cudaStreamSynchronize(stream));
    });
    // The second agreement protects slot reuse against a faster next
    // chunk/call.
  }
}
void DirectExchangeCollectives::synchronize() {
  auto& s = *impl_;
  if (!s.ready) return;
  // Enqueue completes its own streams and transport before returning. Do not
  // synchronize unrelated application streams here.
  s.current();
}
bool DirectExchangeCollectives::initialized() const { return impl_->ready; }
std::vector<DirectExchangeRoute> const& DirectExchangeCollectives::routes()
    const {
  return impl_->routes;
}
std::string const& DirectExchangeCollectives::identity() const {
  return impl_->identity;
}
DirectTraffic DirectExchangeCollectives::traffic(NetworkOperation op,
                                                 size_t count) const {
  auto const& s = *impl_;
  auto kind = op == NetworkOperation::AllReduce   ? Collective::AllReduce
              : op == NetworkOperation::AllGather ? Collective::AllGather
                                                  : Collective::ReduceScatter;
  size_t scale =
      scalarBytes(s.dtype) * (kind == Collective::AllReduce ? 1 : s.ranks);
  if (!s.ready || int(op) < 1 || int(op) > 3 || !count ||
      count > s.capacity / scale)
    throw std::invalid_argument("invalid direct traffic query");
  return directExchangeTraffic(
      {kind, s.dtype, count * scale, false, unsigned(scalarBytes(s.dtype))},
      s.ranks, s.chunk, s.ipc);
}
std::shared_ptr<DirectModelCalibration const>
DirectExchangeCollectives::calibrateModel() {
  auto& s = *impl_;
  if (!s.ready)
    throw std::logic_error("direct model calibration requires prepare");
  s.current();
  auto model = std::make_shared<DirectModelCalibration>();
  model->dtype = s.dtype;
  model->chunkBytes = s.chunk;
  model->configurationIdentities = s.strings(s.identity);
  auto& t = model->topology;
  t.identity = "direct-calibrated-v1";
  t.peerAccess = s.ipc;
  for (int r = 0; r < s.ranks; ++r) {
    t.devices.push_back(r);
    t.gpuNodes.push_back(r);
    t.hardware.push_back({s.gpuIds[r], HardwareKind::Gpu, r, -1, {}});
  }
  model->paths.resize(size_t(s.ranks) * s.ranks);
  model->operations.inputCopy.resize(s.ranks);
  model->operations.selfCopy.resize(s.ranks);
  model->operations.gather.resize(s.ranks);
  model->operations.reduction.resize(s.ranks);
  for (int r = 0; r < s.ranks; ++r) {
    auto suffix = ":" + std::to_string(r);
    model->operations.inputCopy[r] = "direct-input" + suffix;
    model->operations.selfCopy[r] = "direct-self" + suffix;
    model->operations.gather[r] = "direct-gather" + suffix;
    model->operations.reduction[r] = "direct-reduce" + suffix;
  }
  size_t scalar = scalarBytes(s.dtype), prefix = 64 + scalar;
  struct Scratch {
    char* buffer = nullptr;
    cudaStream_t stream = nullptr;
    ~Scratch() {
      if (stream) cudaStreamSynchronize(stream);
      if (buffer) cudaFree(buffer);
      if (stream) cudaStreamDestroy(stream);
    }
  } scratch;
  size_t arenaBytes = (s.ranks + 1) * s.chunk;
  size_t scratchBytes = size_t(s.ranks) * s.chunk + prefix + 64;
  s.collective([&] {
    check(cudaMalloc(&scratch.buffer, scratchBytes));
    check(cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking));
  });
  auto median = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };
  auto pattern = [](int rank, int seed) {
    unsigned char x = (rank * 37 + seed * 17 + 3) % 127;
    return x == 0x5a ? ++x : x;
  };
  auto fillValue = [&](std::vector<unsigned char>& bytes, size_t offset,
                       size_t payload, float value) {
    uint32_t bits = 0;
    if (scalar == 4)
      std::memcpy(&bits, &value, 4);
    else
      bits = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    for (size_t i = 0; i < payload; i += scalar)
      std::memcpy(bytes.data() + offset + i, &bits, scalar);
  };
  // Modes: input copy, self copy, gathered output, reduction, directed write;
  // negative modes measure entry, one status agreement and owner visibility.
  auto sample = [&](int mode, size_t payload, int source, int destination) {
    std::vector<std::vector<double>> observations(s.ranks);
    for (int repetition = 0; repetition < 3; ++repetition)
      for (int iteration = 0; iteration < 5; ++iteration) {
        int seed = repetition * 5 + iteration;
        std::vector<unsigned char> arena, output, actual;
        if (mode >= 0)
          s.collective([&] {
            arena.assign(arenaBytes, 0x5a);
            output.assign(scratchBytes, 0x5a);
            if (mode == 0)
              std::fill_n(output.data() + prefix, payload,
                          pattern(s.rank, seed));
            if (mode == 1 || mode == 4)
              std::fill_n(arena.data(), payload, pattern(s.rank, seed));
            if (mode == 2 || mode == 3)
              for (int r = 0; r < s.ranks; ++r) {
                if (mode == 2)
                  std::fill_n(arena.data() + (r + 1) * s.chunk, payload,
                              pattern(r, seed));
                else
                  fillValue(arena, (r + 1) * s.chunk, payload,
                            float(r % 3 - 1 + seed % 3 - 1));
              }
            check(cudaMemcpyAsync(s.arena, arena.data(), arenaBytes,
                                  cudaMemcpyHostToDevice, scratch.stream));
            check(cudaMemcpyAsync(scratch.buffer, output.data(), scratchBytes,
                                  cudaMemcpyHostToDevice, scratch.stream));
            check(cudaStreamSynchronize(scratch.stream));
          });
        s.control->barrier();
        double us = 0;
        s.collective([&] {
          auto begin = std::chrono::steady_clock::now();
          if (mode == -1)
            s.entry(NetworkOperation::AllReduce, s.arena,
                    scratch.buffer + prefix, 1, scratch.stream);
          else if (mode == -2)
            s.collective([] {});
          else if (mode == -3) {
            if (s.receivingIb &&
                s.ordering < cudaGPUDirectRDMAWritesOrderingOwner)
              check(cudaDeviceFlushGPUDirectRDMAWrites(
                  cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                  cudaFlushGPUDirectRDMAWritesToOwner));
          } else if (mode == 0) {
            check(cudaMemcpyAsync(s.arena, scratch.buffer + prefix, payload,
                                  cudaMemcpyDeviceToDevice, scratch.stream));
            check(cudaStreamSynchronize(scratch.stream));
          } else if (mode == 1) {
            check(cudaMemcpyAsync(s.arena + (s.rank + 1) * s.chunk, s.arena,
                                  payload, cudaMemcpyDeviceToDevice,
                                  scratch.stream));
            check(cudaStreamSynchronize(scratch.stream));
          } else if (mode == 2) {
            for (int r = 0; r < s.ranks; ++r)
              check(cudaMemcpyAsync(scratch.buffer + prefix + r * payload,
                                    s.arena + (r + 1) * s.chunk, payload,
                                    cudaMemcpyDeviceToDevice, scratch.stream));
            check(cudaStreamSynchronize(scratch.stream));
          } else if (mode == 3) {
            size_t count = payload / scalar;
            int blocks = int(std::min<size_t>(65535, (count + 255) / 256));
            if (s.dtype == ScalarType::Float32)
              reduceSources<<<blocks, 256, 0, scratch.stream>>>(
                  reinterpret_cast<float*>(s.arena + s.chunk),
                  reinterpret_cast<float*>(scratch.buffer + prefix), count,
                  s.chunk / scalar, s.ranks);
            else
              reduceSources<<<blocks, 256, 0, scratch.stream>>>(
                  reinterpret_cast<__nv_bfloat16*>(s.arena + s.chunk),
                  reinterpret_cast<__nv_bfloat16*>(scratch.buffer + prefix),
                  count, s.chunk / scalar, s.ranks);
            check(cudaGetLastError());
            check(cudaStreamSynchronize(scratch.stream));
          } else if (s.rank == source) {
            bool ipc = s.ipc[size_t(source) * s.ranks + destination];
            s.connections[destination].write(s.remote[destination],
                                             (source + 1) * s.chunk,
                                             s.local[ipc ? 0 : 1], 0, payload);
            s.connections[destination].flush(30000000);
          }
          us = std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - begin)
                   .count();
          if (mode == -3 &&
              (!s.receivingIb ||
               s.ordering >= cudaGPUDirectRDMAWritesOrderingOwner))
            us = 0;
        });
        auto durations = s.gather(us);
        if (iteration >= 2)
          for (int r = 0; r < s.ranks; ++r)
            if (mode != 4 || r == source)
              observations[r].push_back(durations[r]);
        if (mode >= 0)
          s.collective([&] {
            if (mode == 4 && s.rank == destination &&
                !s.ipc[size_t(source) * s.ranks + destination] &&
                s.ordering < cudaGPUDirectRDMAWritesOrderingOwner)
              check(cudaDeviceFlushGPUDirectRDMAWrites(
                  cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                  cudaFlushGPUDirectRDMAWritesToOwner));
            if (mode == 0)
              std::fill_n(arena.data(), payload, pattern(s.rank, seed));
            if (mode == 1)
              std::fill_n(arena.data() + (s.rank + 1) * s.chunk, payload,
                          pattern(s.rank, seed));
            if (mode == 2)
              for (int r = 0; r < s.ranks; ++r)
                std::fill_n(output.data() + prefix + r * payload, payload,
                            pattern(r, seed));
            if (mode == 3) {
              float sum = 0;
              for (int r = 0; r < s.ranks; ++r)
                sum += float(r % 3 - 1 + seed % 3 - 1);
              fillValue(output, prefix, payload, sum);
            }
            if (mode == 4 && s.rank == destination)
              std::fill_n(arena.data() + (source + 1) * s.chunk, payload,
                          pattern(source, seed));
            actual.resize(arenaBytes);
            check(cudaMemcpy(actual.data(), s.arena, arenaBytes,
                             cudaMemcpyDeviceToHost));
            if (actual != arena)
              throw std::runtime_error(
                  "direct model probe corrupted source/receive slots or "
                  "guards");
            actual.resize(scratchBytes);
            check(cudaMemcpy(actual.data(), scratch.buffer, scratchBytes,
                             cudaMemcpyDeviceToHost));
            if (actual != output)
              throw std::runtime_error(
                  "direct model probe output/guard mismatch");
          });
      }
    return observations;
  };
  std::map<std::string, Performance> resources;
  auto fit = [&](std::string name, size_t factor,
                 std::vector<double> const& small,
                 std::vector<double> const& large) {
    DirectModelObservation a{
        name,   "host-completion;3-repetitions;2-warm;3-timed;private-checked",
        {},     scalar,
        factor, small};
    if (name.rfind("direct-flow:", 0) == 0)
      a.scope =
          "host-post-write-through-flush;receiver-visibility-outside;3-"
          "repetitions;2-warm;3-timed;private-checked";
    else
      a.scope =
          "host-dispatch-through-stream-completion;scalar-aligned;3-"
          "repetitions;2-warm;3-timed;private-checked";
    auto b = a;
    b.bytes = s.chunk;
    b.microseconds = large;
    Performance p;
    double x = median(small), y = median(large);
    if (std::isfinite(x) && std::isfinite(y) && x > 0 && y > 0) {
      // Service envelopes, not a forced monotonic bandwidth fit. Dispatch can
      // make a smaller transfer slower. The estimator interpolates durations
      // only inside these observations and never extrapolates them.
      p.bandwidthGBs =
          std::max(factor * double(scalar) / x, factor * double(s.chunk) / y) /
          1000;
      p.latencyUs = 0;
      p.evidence = Evidence::Measured;
    } else
      a.rejection = b.rejection = "invalid measured operation duration";
    model->observations.push_back(std::move(a));
    model->observations.push_back(std::move(b));
    resources[name] = p;
    return p;
  };
  for (int mode = 0; mode < 4; ++mode) {
    auto small = sample(mode, scalar, -1, -1),
         large = sample(mode, s.chunk, -1, -1);
    for (int r = 0; r < s.ranks; ++r) {
      std::string name = mode == 0   ? model->operations.inputCopy[r]
                         : mode == 1 ? model->operations.selfCopy[r]
                         : mode == 2 ? model->operations.gather[r]
                                     : model->operations.reduction[r];
      size_t factor = mode < 2 ? 2 : mode == 2 ? 2 * s.ranks : s.ranks + 1;
      auto p = fit(name, factor, small[r], large[r]);
      if (p.evidence == Evidence::Measured &&
          p.bandwidthGBs > t.hardware[r].memory.bandwidthGBs)
        t.hardware[r].memory = {p.bandwidthGBs, 0, Evidence::Measured};
    }
  }
  for (int source = 0; source < s.ranks; ++source)
    for (int destination = 0; destination < s.ranks; ++destination)
      if (source != destination) {
        auto small = sample(4, scalar, source, destination),
             large = sample(4, s.chunk, source, destination);
        auto name = "direct-flow:" + std::to_string(source) + ':' +
                    std::to_string(destination);
        fit(name, 1, small[source], large[source]);
        auto& path = model->paths[size_t(source) * s.ranks + destination];
        path.cudaIpc = s.ipc[size_t(source) * s.ranks + destination];
        path.resources.push_back({name, false});
      }
  model->overheads = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int mode : {-1, -2, -3}) {
    auto values = sample(mode, scalar, -1, -1);
    double cost = 0;
    for (int r = 0; r < s.ranks; ++r) {
      cost = std::max(cost, median(values[r]));
      model->observations.push_back(
          {"direct-control:" + std::to_string(mode) + ':' + std::to_string(r),
           "host-completion;all-ready;3-repetitions;2-warm;3-timed",
           {},
           0,
           0,
           values[r]});
    }
    if (mode == -1) model->overheads.entryUs = cost;
    if (mode == -2)
      model->overheads.publicationUs = model->overheads.retirementUs = cost;
    if (mode == -3) model->overheads.visibilityUs = cost;
  }
  // Directed physical cuts, scoped by actual host identity. Unknown shared
  // cuts remain explicit omissions; measured end-to-end paths are not labeled
  // measurements of each bridge/interconnect.
  using Bounds = std::vector<std::pair<std::string, Performance>>;
  auto const& local = s.topology;
  std::string hostScope =
      s.gpuIds[s.rank].substr(0, s.gpuIds[s.rank].find('/')) + '/';
  auto pathBounds = [&](size_t source, size_t destination) {
    Bounds result;
    std::vector<size_t> prev(local.hardware.size(), local.links.size());
    std::vector<bool> seen(local.hardware.size(), false);
    std::queue<size_t> queue;
    queue.push(source);
    seen[source] = true;
    while (!queue.empty() && !seen[destination]) {
      auto node = queue.front();
      queue.pop();
      for (size_t i = 0; i < local.links.size(); ++i) {
        auto const& edge = local.links[i];
        if (edge.source != node || edge.destination >= seen.size() ||
            seen[edge.destination])
          continue;
        auto kind = local.hardware[edge.destination].kind;
        if (edge.destination != destination &&
            (kind == HardwareKind::Gpu || kind == HardwareKind::Nic))
          continue;
        seen[edge.destination] = true;
        prev[edge.destination] = i;
        queue.push(edge.destination);
      }
    }
    if (!seen[destination]) {
      result.push_back({hostScope + "missing-physical-path", {}});
      return result;
    }
    for (auto node = destination; node != source;) {
      auto const& edge = local.links[prev[node]];
      result.push_back({hostScope + edge.resource, edge.performance});
      node = edge.source;
    }
    return result;
  };
  auto gpuNode = [&](int device) {
    auto pos = std::find(local.devices.begin(), local.devices.end(), device);
    if (pos == local.devices.end())
      throw std::runtime_error("direct physical GPU mapping missing");
    return local.gpuNodes[size_t(pos - local.devices.begin())];
  };
  std::vector<Bounds> localBounds(size_t(s.ranks) + 2);
  s.collective([&] {
    auto own = gpuNode(s.device);
    if (s.receivingIb)
      for (auto const& nic : local.networkPorts)
        if (hostScope + nic.name + ':' + std::to_string(nic.port) ==
            s.nicIds[s.rank]) {
          localBounds[0] = pathBounds(own, nic.node);
          localBounds[1] = pathBounds(nic.node, own);
        }
    for (int p = 0; p < s.ranks; ++p)
      if (s.ipc[size_t(s.rank) * s.ranks + p])
        localBounds[p + 2] = pathBounds(own, gpuNode(s.peerDevices[p]));
  });
  std::ostringstream encoded;
  encoded << std::setprecision(17);
  for (auto const& list : localBounds) {
    encoded << list.size() << ' ';
    for (auto const& item : list)
      encoded << std::quoted(item.first) << ' ' << item.second.bandwidthGBs
              << ' ' << int(item.second.evidence) << ' ';
  }
  auto encodedBounds = s.strings(encoded.str());
  std::vector<std::vector<Bounds>> bounds(s.ranks,
                                          std::vector<Bounds>(s.ranks + 2));
  for (int r = 0; r < s.ranks; ++r) {
    std::istringstream in(encodedBounds[r]);
    for (auto& list : bounds[r]) {
      size_t count;
      if (!(in >> count) || count > 4096)
        throw std::runtime_error("invalid direct physical-bound metadata");
      for (size_t i = 0; i < count; ++i) {
        std::string name;
        double rate;
        int evidence;
        if (!(in >> std::quoted(name) >> rate >> evidence))
          throw std::runtime_error("truncated physical-bound metadata");
        list.push_back({name, {rate, 0, Evidence(evidence)}});
      }
    }
  }
  auto bound = [&](DirectPathModel& path, std::string const& name,
                   Performance p) {
    if (p.evidence == Evidence::Unknown || !std::isfinite(p.bandwidthGBs) ||
        p.bandwidthGBs <= 0) {
      model->unmodeledResources.push_back(name);
      return;
    }
    auto prior = resources.find(name);
    if (prior != resources.end() &&
        (prior->second.bandwidthGBs != p.bandwidthGBs ||
         prior->second.evidence != p.evidence))
      throw std::runtime_error(
          "inconsistent direct physical resource capacity");
    resources[name] = p;
    if (std::none_of(path.resources.begin(), path.resources.end(),
                     [&](auto const& x) { return x.resource == name; }))
      path.resources.push_back({name, true});
  };
  for (int r = 0; r < s.ranks; ++r)
    for (int p = 0; p < s.ranks; ++p)
      if (r != p) {
        auto& path = model->paths[size_t(r) * s.ranks + p];
        if (path.cudaIpc)
          for (auto const& b : bounds[r][p + 2]) bound(path, b.first, b.second);
        else {
          for (auto const& b : bounds[r][0]) bound(path, b.first, b.second);
          for (auto const& b : bounds[p][1]) bound(path, b.first, b.second);
          bound(path, "nic:" + s.nicIds[r] + ":tx",
                {s.nicRates[r], 0, Evidence::Advertised});
          bound(path, "nic:" + s.nicIds[p] + ":rx",
                {s.nicRates[p], 0, Evidence::Advertised});
        }
      }
  for (auto const& r : resources) t.links.push_back({0, 0, r.first, r.second});
  std::sort(model->unmodeledResources.begin(), model->unmodeledResources.end());
  model->unmodeledResources.erase(std::unique(model->unmodeledResources.begin(),
                                              model->unmodeledResources.end()),
                                  model->unmodeledResources.end());
  std::ostringstream identity;
  identity << std::setprecision(17)
           << "direct-operational-calibration-v2;scalar-aligned;size-tables;";
  for (auto const& id : model->configurationIdentities)
    identity << id.size() << ':' << id;
  for (auto const& text : encodedBounds) identity << text.size() << ':' << text;
  for (auto const& o : model->observations) {
    identity << o.resource << ':' << o.scope << ':' << o.bytes << ':'
             << o.trafficFactor << ':' << o.rejection;
    for (auto x : o.microseconds) identity << ':' << x;
  }
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : identity.str()) hash = (hash ^ c) * 1099511628211ULL;
  std::ostringstream key;
  key << "direct-operational-v2:" << std::hex << hash;
  model->identity = key.str();
  t.identity = model->identity;
  s.control->barrier();
  return model;
}
}  // namespace uccl_lite

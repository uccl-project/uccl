#include "uccl_lite/collective_dispatch.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <elf.h>
#include <link.h>
#include <sched.h>

namespace uccl_lite {
namespace {
std::string const& routerBuildId() {
  // The containing ELF includes the CUDA fatbin for both static and shared
  // builds. Its in-memory build ID distinguishes compiler/architecture/flag
  // changes without CUDA initialization or filesystem races after relinking.
  static std::string const identity = [] {
    struct Image {
      uintptr_t address;
      std::array<unsigned char, 64> bytes{};
      size_t count = 0;
    } image{reinterpret_cast<uintptr_t>(&routerBuildId)};
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* context) {
          auto& image = *static_cast<Image*>(context);
          uintptr_t base = info->dlpi_addr;
          bool contains = false;
          for (int i = 0; i < info->dlpi_phnum; ++i) {
            auto const& p = info->dlpi_phdr[i];
            uintptr_t segment = base + p.p_vaddr;
            contains |= p.p_type == PT_LOAD && image.address >= segment &&
                        image.address - segment < p.p_memsz;
          }
          if (!contains) return 0;
          for (int i = 0; i < info->dlpi_phnum; ++i) {
            auto const& p = info->dlpi_phdr[i];
            if (p.p_type != PT_NOTE) continue;
            auto cursor =
                reinterpret_cast<unsigned char const*>(base + p.p_vaddr);
            auto end = cursor + p.p_filesz;
            while (size_t(end - cursor) >= sizeof(ElfW(Nhdr))) {
              auto const& note = *reinterpret_cast<ElfW(Nhdr) const*>(cursor);
              cursor += sizeof(ElfW(Nhdr));
              size_t nameBytes = (size_t(note.n_namesz) + 3) & ~size_t(3);
              size_t descBytes = (size_t(note.n_descsz) + 3) & ~size_t(3);
              if (nameBytes > size_t(end - cursor)) break;
              auto name = cursor;
              cursor += nameBytes;
              if (descBytes > size_t(end - cursor)) break;
              if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
                  std::memcmp(name, "GNU", 4) == 0 &&
                  note.n_descsz <= image.bytes.size()) {
                image.count = note.n_descsz;
                std::memcpy(image.bytes.data(), cursor, image.count);
                return 1;
              }
              cursor += descBytes;
            }
          }
          return 1;
        },
        &image);
    std::string result;
    char const* digits = "0123456789abcdef";
    for (size_t i = 0; i < image.count; ++i) {
      result += digits[image.bytes[i] >> 4];
      result += digits[image.bytes[i] & 15];
    }
    return result;
  }();
  return identity;
}
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::string read(std::string const& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), {});
}
std::vector<int> cpuList(int node, cpu_set_t const& allowed) {
  std::istringstream input(read("/sys/devices/system/node/node" +
                                std::to_string(node) + "/cpulist"));
  std::string range;
  std::vector<int> result;
  while (std::getline(input, range, ',')) {
    auto dash = range.find('-');
    int begin = std::stoi(range), end = dash == std::string::npos
                                            ? begin
                                            : std::stoi(range.substr(dash + 1));
    if (begin < 0 || end >= CPU_SETSIZE || end < begin)
      throw std::runtime_error("invalid host CPU list");
    for (int cpu = begin; cpu <= end; ++cpu)
      if (CPU_ISSET(cpu, &allowed)) result.push_back(cpu);
  }
  return result;
}
cpu_router::PacketProgram program(CollectiveCall const& call) {
  int ranks = int(call.ports.size());
  switch (call.collective) {
    case Collective::AllReduce:
      return cpu_router::PacketProgram::allReduce(call.count, ranks);
    case Collective::AllGather:
      return cpu_router::PacketProgram::allGather(call.count, ranks);
    case Collective::ReduceScatter:
      return cpu_router::PacketProgram::reduceScatter(call.count, ranks);
  }
  throw std::invalid_argument("unsupported collective");
}
}  // namespace

Topology inspectTopology(std::vector<int> const& devices) {
  if (devices.empty() || devices.size() > 32 ||
      std::set<int>(devices.begin(), devices.end()).size() != devices.size())
    throw std::invalid_argument("expected 1..32 distinct local devices");
  Topology t;
  t.devices = devices;
  t.avx2 = __builtin_cpu_supports("avx2");
  check(cudaRuntimeGetVersion(&t.cudaRuntimeVersion));
  std::smatch match;
  std::string driver = read("/proc/driver/nvidia/version");
  if (std::regex_search(driver, match,
                        std::regex(R"(\b[0-9]{3}\.[0-9]+\.[0-9]+\b)")))
    t.driverVersion = match.str();
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof(allowed), &allowed))
    throw std::runtime_error("cannot query CPU affinity");
  std::ostringstream identity;
  identity << "uccl-lite-router-v1;cuda=" << t.cudaRuntimeVersion
           << ";driver=" << t.driverVersion;
  identity << ";avx2=" << bool(__builtin_cpu_supports("avx2"))
           << ";avx512f=" << bool(__builtin_cpu_supports("avx512f"))
           << ";avx512bw=" << bool(__builtin_cpu_supports("avx512bw"));
  std::istringstream cpuInfo(read("/proc/cpuinfo"));
  std::string line;
  while (std::getline(cpuInfo, line))
    if (line.rfind("model name", 0) == 0) {
      identity << ";cpu=" << line;
      break;
    }
  for (int device : devices) {
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, device));
    int node = cpu_router::gpuNumaNode(device);
    t.numaNodes.push_back(node);
    t.gpuNames.emplace_back(property.name);
    t.mappedHost.push_back(property.canMapHostMemory);
    if (!t.permittedCpuIds.count(node))
      t.permittedCpuIds[node] = cpuList(node, allowed);
    t.permittedCpusPerNode[node] = int(t.permittedCpuIds[node].size());
    char pci[32];
    check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
    identity << ";gpu=" << device << ':' << pci << ':' << node << ':'
             << property.name << ':';
    for (unsigned char byte : property.uuid.bytes)
      identity << std::hex << std::setw(2) << std::setfill('0')
               << unsigned(byte);
    identity << std::dec;
    for (int peer : devices) {
      int accessible = 0;
      if (device != peer)
        check(cudaDeviceCanAccessPeer(&accessible, device, peer));
      t.peerAccess.push_back(accessible != 0);
      identity << accessible;
    }
  }
  for (auto const& entry : t.permittedCpuIds) {
    identity << ";node=" << entry.first << ":cpus=";
    for (int cpu : entry.second) identity << cpu << ',';
  }
  t.identity = identity.str();
  return t;
}

Workload CollectiveCall::workload() const {
  size_t scalar = dtype == ScalarType::Float32 ? 4 : 2;
  size_t expansion = collective == Collective::AllReduce ? 1 : ports.size();
  if (!count || !expansion ||
      count > std::numeric_limits<size_t>::max() / scalar / expansion)
    throw std::invalid_argument("empty or overflowing collective extent");
  uintptr_t addresses = 0;
  bool complete = !ports.empty();
  for (auto const& port : ports) {
    addresses |= reinterpret_cast<uintptr_t>(port.input) |
                 reinterpret_cast<uintptr_t>(port.output);
    complete &= port.input && port.output;
  }
  unsigned alignment = !complete ? 0
                       : !addresses
                           ? 256
                           : 1U << std::min(8, __builtin_ctzll(addresses));
  return {collective, dtype, count * scalar * expansion, inPlace, alignment};
}

CollectiveDispatcher::CollectiveDispatcher(Topology topology,
                                           BackendPolicy policy,
                                           Fallback fallback,
                                           cpu_router::CpuRouterOptions options)
    : topology_(std::move(topology)),
      policy_(std::move(policy)),
      fallback_(std::move(fallback)),
      options_(std::move(options)) {
  if (!fallback_)
    throw std::invalid_argument("an existing-backend callback is required");
  if (options_.backend != cpu_router::TransportBackend::GpuKernel)
    throw std::invalid_argument(
        "adaptive CPU-router dispatch supports the SM transport only");
  if (options_.reduction != cpu_router::ReductionPlacement::Cpu &&
      options_.reduction != cpu_router::ReductionPlacement::Gpu)
    throw std::invalid_argument("invalid reduction placement");
  if (options_.interleaveScatter && options_.concurrentProducer &&
      options_.reduction == cpu_router::ReductionPlacement::Cpu)
    throw std::invalid_argument(
        "interleaved scatter requires standalone SM transport");
  if (options_.hostRingReduceScatter &&
      (options_.reduction != cpu_router::ReductionPlacement::Gpu ||
       !options_.gpuProgress || options_.interleaveScatter ||
       options_.concurrentProducer || options_.transportBlocks > 128 ||
       options_.singleCtaBytes || options_.transportVectorBytes != 16))
    throw std::invalid_argument("invalid standalone host-ring options");
  if (options_.gpuProgress &&
      options_.reduction != cpu_router::ReductionPlacement::Gpu)
    throw std::invalid_argument("GPU progress requires GPU reduction");
  if (options_.smallAllReducePacketBytes &&
      (options_.smallAllReducePacketBytes < options_.minPacketBytes ||
       options_.smallAllReducePacketBytes > options_.packetBytes ||
       options_.smallAllReducePacketBytes % 16))
    throw std::invalid_argument("invalid small AllReduce packet size");
  if (options_.largeAllGatherPacketBytes &&
      (options_.largeAllGatherPacketBytes < options_.packetBytes ||
       options_.largeAllGatherPacketBytes > (32UL << 20) ||
       options_.largeAllGatherPacketBytes % 16))
    throw std::invalid_argument("invalid large AllGather packet size");
  if (options_.smallReduceScatterPacketBytes &&
      (options_.smallReduceScatterPacketBytes < options_.minPacketBytes ||
       options_.smallReduceScatterPacketBytes > options_.packetBytes ||
       options_.smallReduceScatterPacketBytes % 16))
    throw std::invalid_argument("invalid small ReduceScatter packet size");
  if ((options_.smallAllReduceWorkersPerNode == 0) !=
          (options_.smallAllReduceTransportBlocks == 0) ||
      options_.smallAllReduceWorkersPerNode < 0 ||
      options_.smallAllReduceWorkersPerNode > options_.workersPerNode ||
      (options_.smallAllReduceTransportBlocks != 0 &&
       options_.smallAllReduceTransportBlocks < 2) ||
      options_.smallAllReduceTransportBlocks > cpu_router::kMaxTransportBlocks)
    throw std::invalid_argument("invalid small AllReduce worker/grid choice");
  if (options_.smallReduceScatterWorkersPerNode < 0 ||
      options_.smallReduceScatterWorkersPerNode > 63)
    throw std::invalid_argument("invalid small ReduceScatter worker limit");
  if (options_.gpuLaunchWorkersPerNode < 0 ||
      options_.gpuLaunchWorkersPerNode > cpu_router::kMaxRanks)
    throw std::invalid_argument("invalid GPU launch worker limit");
  if (options_.smallReduceScatterGpuProgressMaxBytes &&
      (options_.smallReduceScatterGpuProgressMaxBytes < 128 ||
       options_.smallReduceScatterGpuProgressMaxBytes > (64UL << 10) ||
       options_.smallReduceScatterGpuProgressMaxBytes % 16))
    throw std::invalid_argument(
        "invalid small ReduceScatter GPU progress limit");
  for (size_t limit : {options_.smallAllReduceGpuProgressMaxBytes,
                       options_.smallAllGatherGpuProgressMaxBytes})
    if (limit && (limit < 16 || limit > (1UL << 20) || limit % 16))
      throw std::invalid_argument("invalid small AR/AG GPU progress limit");
  std::ostringstream key;
  key << ";transport-revision=5";
  auto const& buildId = routerBuildId();
  key << ";router-build-id=" << (buildId.empty() ? "unavailable" : buildId);
  if (buildId.empty() && policy_.mode == RouterMode::Auto)
    policy_.measurements
        .clear();  // Unknown software retains the caller backend.
  // maxTensorBytes is a validation bound, not packet geometry or an allocation
  // size.
  key << ";options=" << options_.packetBytes << ',' << options_.minPacketBytes
      << ";small-ar-packet-v1=" << options_.smallAllReducePacketBytes << ','
      << ";small-rs-packet-v1=" << options_.smallReduceScatterPacketBytes
      << ";small-rs-gpu-progress-v1=" << options_.smallReduceScatterGpuProgress
      << ',' << options_.transportBlocks << ',' << options_.transportThreads
      << ',' << options_.transportGrainBytes << ',' << options_.workersPerNode
      << ',' << options_.scatterWorkersPerNode << ',' << options_.grainElements
      << ',' << options_.replayCpu << ',' << options_.inlineReplay << ','
      << options_.concurrentProducer;
  // Keep the legacy CPU identity stable, but never reuse its measurements for
  // a different placement. GPU mode needs its own exact local calibration.
  if (options_.reduction != cpu_router::ReductionPlacement::Cpu)
    key << ";reduction=gpu-v1";
  if (options_.interleaveScatter)
    key << (options_.reduction == cpu_router::ReductionPlacement::Gpu
                ? ";scatter=interleaved-v2"
                : ";scatter=cpu-interleaved-v1");
  if (options_.smallReduceScatterWorkersPerNode)
    key << ";small-rs-256k-1m-workers-v1="
        << options_.smallReduceScatterWorkersPerNode;
  if (options_.batchCpuJobs) key << ";batch-cpu-jobs-v2";
  if (options_.directScatterResults) key << ";rs-direct-results-v1";
  if (options_.gpuLaunchWorkersPerNode)
    key << ";gpu-launch-workers-per-node-v1="
        << options_.gpuLaunchWorkersPerNode;
  if (options_.gpuProgress) key << ";progress=gpu-v1";
  if (options_.hostRingReduceScatter) key << ";rs-host-ring=bf16-cta-v1";
  if (options_.agPerSourceProgress) key << ";ag-progress=per-source-v1";
  if (options_.singleNumaReduction) key << ";single-numa-reduction=v1";
  if (options_.singleCtaBytes)
    key << ";single-cta-bytes-v1=" << options_.singleCtaBytes;
  if (options_.transportVectorBytes != 16)
    key << ";transport-vector-bytes=" << options_.transportVectorBytes;
  if (options_.smallAllReduceWorkersPerNode)
    key << ";small-ar-128-256k-workers-v1="
        << options_.smallAllReduceWorkersPerNode << ','
        << options_.smallAllReduceTransportBlocks;
  if (options_.smallReduceScatterGpuProgress64K)
    key << ";small-rs-64k-progress-v1";
  if (options_.smallReduceScatterGpuProgressMaxBytes)
    key << ";small-rs-progress-max-bytes-v1="
        << options_.smallReduceScatterGpuProgressMaxBytes;
  if (options_.smallAllReduceGpuProgressMaxBytes)
    key << ";small-ar-progress-max-bytes-v1="
        << options_.smallAllReduceGpuProgressMaxBytes;
  if (options_.smallAllGatherGpuProgressMaxBytes)
    key << ";small-ag-per-source-progress-max-bytes-v1="
        << options_.smallAllGatherGpuProgressMaxBytes;
  if (options_.largeAllGatherPacketBytes)
    key << ";large-ag-packet-v1=" << options_.largeAllGatherPacketBytes;
  for (int cpu : options_.workerCpus) key << ";worker=" << cpu;
  for (int cpu : options_.gpuLaunchAllowedCpus)
    key << ";gpu-launch-allowed-cpu=" << cpu;
  for (int node : options_.numaNodes) key << ";numa=" << node;
  topology_.identity += key.str();
  for (char const* knob : {"UCCL_LITE_ROUTER_PROFILE", "UCCL_LITE_TIMELINE",
                           "UCCL_LITE_GPU_TIMELINE_PATH"}) {
    bool enabled = std::getenv(knob) != nullptr;
    topology_.identity += std::string(";") + knob + "=" + (enabled ? "1" : "0");
  }
  // Use only the caller's currently permitted affinity, not the whole host
  // pool.
  if (options_.workerCpus.empty()) {
    std::vector<int> nodes;
    for (int node : topology_.numaNodes)
      if (std::find(nodes.begin(), nodes.end(), node) == nodes.end())
        nodes.push_back(node);
    for (int node : nodes)
      options_.workersPerNode = std::min(
          options_.workersPerNode,
          std::max(1, int(topology_.permittedCpuIds[node].size()) - 1));
    for (int node : nodes) {
      auto const& cpus = topology_.permittedCpuIds[node];
      if (cpus.size() < size_t(options_.workersPerNode))
        continue;  // Capability check will reject this topology.
      options_.workerCpus.insert(options_.workerCpus.end(), cpus.begin(),
                                 cpus.begin() + options_.workersPerNode);
    }
    if (options_.replayCpu < 0)
      for (int node : nodes)
        for (int cpu : topology_.permittedCpuIds[node])
          if (std::find(options_.workerCpus.begin(), options_.workerCpus.end(),
                        cpu) == options_.workerCpus.end())
            options_.replayCpu = cpu;
  }
}
CollectiveDispatcher::~CollectiveDispatcher() = default;

void CollectiveDispatcher::validatePorts(CollectiveCall const& call) const {
  if (call.ports.size() != topology_.devices.size())
    throw std::invalid_argument("port count changed");
  auto workload = call.workload();
  size_t scalar = call.dtype == ScalarType::Float32 ? 4 : 2;
  size_t contribution = call.count * scalar;
  size_t inputBytes = call.collective == Collective::ReduceScatter
                          ? workload.bytes
                          : contribution;
  size_t outputBytes =
      call.collective == Collective::AllGather ? workload.bytes : contribution;
  for (size_t rank = 0; rank < call.ports.size(); ++rank) {
    auto const& port = call.ports[rank];
    if (port.device != topology_.devices[rank] || !port.input || !port.output)
      throw std::invalid_argument(
          "invalid collective ports or changed device order");
    auto input = reinterpret_cast<uintptr_t>(port.input),
         output = reinterpret_cast<uintptr_t>(port.output);
    if (input > UINTPTR_MAX - inputBytes || output > UINTPTR_MAX - outputBytes)
      throw std::invalid_argument("collective pointer range overflow");
    if (call.inPlace) {
      bool aliases = call.collective == Collective::AllGather
                         ? input == output + rank * contribution
                     : call.collective == Collective::ReduceScatter
                         ? output == input + rank * contribution
                         : input == output;
      if (!aliases)
        throw std::invalid_argument(
            "in-place flag does not match buffer layout");
    } else if (input < output + outputBytes && output < input + inputBytes) {
      throw std::invalid_argument("out-of-place buffers overlap");
    }
  }
}
BackendDecision CollectiveDispatcher::decision(
    CollectiveCall const& call) const {
  validatePorts(call);
  Workload w = call.workload();
  if (w.bytes > options_.maxTensorBytes) {
    if (policy_.mode == RouterMode::Force)
      throw std::invalid_argument("CPU-router capacity exceeded");
    return {false, "CPU-router capacity exceeded; keep existing backend"};
  }
  if (options_.hostRingReduceScatter) {
    bool supported = call.collective == Collective::ReduceScatter &&
                     call.dtype == ScalarType::BFloat16 && !call.inPlace &&
                     call.ports.size() >= 2 && call.count % 8 == 0 &&
                     call.count <= options_.packetBytes / 2;
    for (auto const& p : call.ports)
      supported = supported && !((reinterpret_cast<uintptr_t>(p.input) |
                                  reinterpret_cast<uintptr_t>(p.output)) &
                                 15);
    if (!supported) {
      if (policy_.mode == RouterMode::Force)
        throw std::invalid_argument("unsupported host-ring workload");
      return {false, "unsupported host-ring workload; keep existing backend"};
    }
  }
  return policy_.select(topology_, w);
}
size_t CollectiveDispatcher::packetBytesFor(CollectiveCall const& call) const {
  if (options_.hostRingReduceScatter) return options_.packetBytes;
  if (call.collective == Collective::AllGather &&
      call.workload().bytes >= (4UL << 20) &&
      options_.largeAllGatherPacketBytes)
    return options_.largeAllGatherPacketBytes;
  if (call.collective == Collective::AllReduce &&
      call.workload().bytes <= (4UL << 20) &&
      options_.smallAllReducePacketBytes)
    return std::min(options_.packetBytes, options_.smallAllReducePacketBytes);
  if (call.collective == Collective::ReduceScatter &&
      call.workload().bytes >= (256UL << 10) &&
      call.workload().bytes <= (4UL << 20) &&
      options_.smallReduceScatterPacketBytes)
    return std::min(options_.packetBytes,
                    options_.smallReduceScatterPacketBytes);
  return options_.packetBytes;
}
int CollectiveDispatcher::workersPerNodeFor(CollectiveCall const& call) const {
  return options_.smallAllReduceWorkersPerNode &&
                 call.collective == Collective::AllReduce &&
                 call.workload().bytes >= (128UL << 10) &&
                 call.workload().bytes <= (256UL << 10)
             ? options_.smallAllReduceWorkersPerNode
             : options_.workersPerNode;
}
int CollectiveDispatcher::transportBlocksFor(CollectiveCall const& call) const {
  return options_.smallAllReduceTransportBlocks &&
                 call.collective == Collective::AllReduce &&
                 call.workload().bytes >= (128UL << 10) &&
                 call.workload().bytes <= (256UL << 10)
             ? options_.smallAllReduceTransportBlocks
             : options_.transportBlocks;
}
CollectiveDispatcher::SmallGpuProgress
CollectiveDispatcher::smallGpuProgressFor(CollectiveCall const& call) const {
  if (options_.reduction != cpu_router::ReductionPlacement::Cpu)
    return SmallGpuProgress::None;
  size_t bytes = call.workload().bytes;
  if (call.collective == Collective::AllReduce)
    return options_.smallAllReduceGpuProgressMaxBytes &&
                   bytes <= options_.smallAllReduceGpuProgressMaxBytes
               ? SmallGpuProgress::Flat
               : SmallGpuProgress::None;
  if (call.collective == Collective::AllGather)
    return options_.smallAllGatherGpuProgressMaxBytes &&
                   bytes <= options_.smallAllGatherGpuProgressMaxBytes
               ? SmallGpuProgress::PerSource
               : SmallGpuProgress::None;
  return options_.smallReduceScatterGpuProgress &&
                 !options_.interleaveScatter &&
                 options_.reduction == cpu_router::ReductionPlacement::Cpu &&
                 call.collective == Collective::ReduceScatter &&
                 (call.workload().bytes == (128UL << 10) ||
                  (options_.smallReduceScatterGpuProgressMaxBytes &&
                   call.workload().bytes <=
                       options_.smallReduceScatterGpuProgressMaxBytes) ||
                  (options_.smallReduceScatterGpuProgress64K &&
                   call.workload().bytes == (64UL << 10)))
             ? SmallGpuProgress::Flat
             : SmallGpuProgress::None;
}
BackendDecision CollectiveDispatcher::prepare(CollectiveCall const& call) {
  auto selected = decision(call);
  size_t packetBytes = packetBytesFor(call);
  int workersPerNode = workersPerNodeFor(call);
  int transportBlocks = transportBlocksFor(call);
  auto smallGpu = smallGpuProgressFor(call);
  if (selected.cpuRouter && (!router_ || routerPacketBytes_ != packetBytes ||
                             routerWorkersPerNode_ != workersPerNode ||
                             routerTransportBlocks_ != transportBlocks ||
                             routerSmallGpuProgress_ != smallGpu)) {
    for (auto const& port : call.ports) {
      check(cudaSetDevice(port.device));
      cudaStreamCaptureStatus status;
      check(cudaStreamIsCapturing(port.stream, &status));
      if (status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("prepare CPU-router before graph capture");
      check(cudaStreamSynchronize(port.stream));
    }
    router_.reset();
    auto activeOptions = options_;
    activeOptions.packetBytes = packetBytes;
    activeOptions.workersPerNode = workersPerNode;
    activeOptions.transportBlocks = transportBlocks;
    if (workersPerNode != options_.workersPerNode &&
        !options_.workerCpus.empty()) {
      std::vector<int> selectedCpus;
      size_t groups = options_.workerCpus.size() / options_.workersPerNode;
      for (size_t group = 0; group < groups; ++group)
        selectedCpus.insert(
            selectedCpus.end(),
            options_.workerCpus.begin() + group * options_.workersPerNode,
            options_.workerCpus.begin() + group * options_.workersPerNode +
                workersPerNode);
      activeOptions.workerCpus = std::move(selectedCpus);
    }
    if (smallGpu != SmallGpuProgress::None) {
      activeOptions.reduction = cpu_router::ReductionPlacement::Gpu;
      activeOptions.gpuProgress = true;
      activeOptions.agPerSourceProgress =
          smallGpu == SmallGpuProgress::PerSource;
    }
    if (activeOptions.gpuLaunchWorkersPerNode) {
      std::vector<int> permitted;
      for (auto const& group : topology_.permittedCpuIds)
        permitted.insert(permitted.end(), group.second.begin(),
                         group.second.end());
      if (activeOptions.gpuLaunchAllowedCpus.empty())
        activeOptions.gpuLaunchAllowedCpus = std::move(permitted);
      else
        for (int cpu : activeOptions.gpuLaunchAllowedCpus)
          if (std::find(permitted.begin(), permitted.end(), cpu) ==
              permitted.end())
            throw std::invalid_argument(
                "GPU launch CPU outside inspected allowance");
    }
    auto router = std::make_unique<cpu_router::CpuRouter>(activeOptions);
    router->prepare(call.ports);
    router_ = std::move(router);
    routerPacketBytes_ = packetBytes;
    routerWorkersPerNode_ = workersPerNode;
    routerTransportBlocks_ = transportBlocks;
    routerSmallGpuProgress_ = smallGpu;
  }
  return selected;
}
BackendDecision CollectiveDispatcher::enqueue(CollectiveCall const& call) {
  auto selected = decision(call);
  if (!selected.cpuRouter) {
    fallback_(call);
    return selected;
  }
  if (!router_)
    throw std::logic_error("call prepare before CPU-router enqueue");
  size_t packetBytes = packetBytesFor(call);
  if (routerPacketBytes_ != packetBytes)
    throw std::logic_error("prepare CPU-router for this packet size");
  if (routerWorkersPerNode_ != workersPerNodeFor(call) ||
      routerTransportBlocks_ != transportBlocksFor(call))
    throw std::logic_error("prepare CPU-router for this worker/grid choice");
  if (routerSmallGpuProgress_ != smallGpuProgressFor(call))
    throw std::logic_error("prepare CPU-router for this reduction mode");
  // PacketPipeline checks every stream before publishing any CPU or GPU work.
  // This dispatcher never begins core capture, so that check also rejects
  // external capture here. Avoid a second CUDA pass over the same streams.
  router_->enqueue(program(call), call.ports);
  return selected;
}
}  // namespace uccl_lite

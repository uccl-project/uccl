#include "discovery.h"
#include "uccl_lite/topology.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <sched.h>

namespace uccl_lite {
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
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
int gpuNumaNode(int device) {
  char pci[32];
  check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
  std::string id(pci);
  std::transform(id.begin(), id.end(), id.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  if (id.size() > 12) id = id.substr(id.size() - 12);
  auto value = read("/sys/bus/pci/devices/" + id + "/numa_node");
  return value.empty() ? -1 : std::stoi(value);
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
    int node = gpuNumaNode(device);
    t.numaNodes.push_back(node);
    t.gpuNames.emplace_back(property.name);
    t.mappedHost.push_back(property.canMapHostMemory);
    if (node >= 0 && !t.permittedCpuIds.count(node))
      t.permittedCpuIds[node] = cpuList(node, allowed);
    t.permittedCpusPerNode[node] = int(t.permittedCpuIds[node].size());
    char pci[32];
    check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
    std::string pciId(pci);
    std::transform(pciId.begin(), pciId.end(), pciId.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (pciId.size() > 12) pciId = pciId.substr(pciId.size() - 12);
    t.gpuPciIds.push_back(pciId);
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
  discoverHardware(t);
  for (auto const& node : t.hardware)
    if (node.kind == HardwareKind::Numa && node.numa >= 0 &&
        !t.permittedCpuIds.count(node.numa)) {
      t.permittedCpuIds[node.numa] = cpuList(node.numa, allowed);
      t.permittedCpusPerNode[node.numa] =
          int(t.permittedCpuIds[node.numa].size());
    }
  for (auto const& entry : t.permittedCpuIds) {
    identity << ";node=" << entry.first << ":cpus=";
    for (int cpu : entry.second) identity << cpu << ',';
  }
  identity << hardwareIdentity(t);
  t.identity = identity.str();
  t.hardwareIdentity = t.identity;
  return t;
}

}  // namespace uccl_lite

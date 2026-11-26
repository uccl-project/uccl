#include "discovery.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace uccl_lite {
namespace {
namespace fs = std::filesystem;
std::string read(fs::path const& path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream), {}};
}
double number(fs::path const& path) {
  std::istringstream stream(read(path));
  double value = 0;
  stream >> value;
  return std::isfinite(value) && value > 0 ? value : 0;
}
int numa(fs::path const& path) {
  std::istringstream stream(read(path / "numa_node"));
  int value = -1;
  stream >> value;
  return value;
}
std::vector<fs::path> children(fs::path const& path) {
  std::vector<fs::path> result;
  std::error_code error;
  for (fs::directory_iterator it(path, error), end; !error && it != end;
       it.increment(error))
    result.push_back(it->path());
  std::sort(result.begin(), result.end());
  return result;
}
bool pciAddress(std::string const& name) {
  if (name.size() != 12 || name[4] != ':' || name[7] != ':' || name[10] != '.')
    return false;
  for (size_t i = 0; i < name.size(); ++i)
    if (i != 4 && i != 7 && i != 10 && !std::isxdigit(name[i])) return false;
  return true;
}
// Maximum capability is stable across idle power states. This is explicitly
// advertised bandwidth, not an estimate of useful concurrent transfer rates.
Performance pcie(fs::path const& path) {
  double speed = number(path / "max_link_speed");
  double width = number(path / "max_link_width");
  double efficiency = speed <= 5 ? .8 : 128.0 / 130;
  // PCIe 6+ FLIT overhead needs a different model; leave it unknown for now.
  if (!speed || !width || speed > 32) return {};
  return {speed * width * efficiency / 8, -1, Evidence::Advertised};
}
class Builder {
 public:
  Topology& t;
  fs::path sysfs;
  std::map<std::string, size_t> indices;
  std::set<std::string> linked;
  size_t node(std::string const& id, HardwareKind kind, int domain = -1) {
    auto [it, added] = indices.emplace(id, t.hardware.size());
    if (added) t.hardware.push_back({id, kind, -1, domain, {}});
    return it->second;
  }
  void attach(size_t child, size_t parent, Performance performance) {
    std::string resource = t.hardware[child].id + "->" + t.hardware[parent].id;
    if (!linked.insert(resource).second) return;
    t.links.push_back({child, parent, resource, performance});
    t.links.push_back({parent, child, resource + ":reverse", performance});
  }
  size_t pci(std::string const& id, HardwareKind kind) {
    fs::path entry = sysfs / "bus/pci/devices" / id;
    int domain = numa(entry);
    size_t leaf = node("pci:" + id, kind, domain), child = leaf;
    std::error_code error;
    auto physical = fs::canonical(entry, error);
    if (error) return leaf;  // Missing topology is not fabricated connectivity.
    auto link = pcie(physical);
    for (auto p = physical.parent_path(); p != p.root_path() && !p.empty();
         p = p.parent_path()) {
      auto name = p.filename().string();
      if (!pciAddress(name)) continue;
      auto parent = node("pci:" + name, HardwareKind::PciBridge, numa(p));
      attach(child, parent, link);
      child = parent;
      link = pcie(p);
    }
    if (domain >= 0) {
      // The root complex to memory link has no sysfs throughput measurement.
      auto host =
          node("numa:" + std::to_string(domain), HardwareKind::Numa, domain);
      attach(child, host, link);
    }
    return leaf;
  }
};
}  // namespace

void discoverHardware(Topology& t, std::filesystem::path const& sysfs) {
  if (t.devices.size() != t.gpuPciIds.size())
    throw std::invalid_argument("GPU PCI identities must follow rank order");
  t.calibrationIdentity.clear();
  t.resourceCalibrationIdentities.clear();
  t.hardware.clear();
  t.links.clear();
  t.paths.clear();
  t.networkPorts.clear();
  t.gpuNodes.clear();
  Builder graph{t, sysfs, {}, {}};
  for (size_t rank = 0; rank < t.devices.size(); ++rank) {
    auto index = graph.pci(t.gpuPciIds[rank], HardwareKind::Gpu);
    t.hardware[index].device = t.devices[rank];
    t.gpuNodes.push_back(index);
  }
  std::set<int> online;
  std::istringstream onlineText(read(sysfs / "devices/system/node/online"));
  std::string range;
  while (std::getline(onlineText, range, ',')) {
    auto dash = range.find('-');
    int begin = std::stoi(range), end = dash == std::string::npos
                                            ? begin
                                            : std::stoi(range.substr(dash + 1));
    if (begin < 0 || end < begin || end > 65535)
      throw std::invalid_argument("invalid online NUMA list");
    for (int node = begin; node <= end; ++node) online.insert(node);
  }
  // NUMA domains are hardware nodes even when they have no selected GPU.
  for (auto const& entry : children(sysfs / "devices/system/node")) {
    auto name = entry.filename().string();
    if (name.rfind("node", 0) != 0 || name.size() == 4 ||
        !std::all_of(name.begin() + 4, name.end(),
                     [](unsigned char c) { return std::isdigit(c); }))
      continue;
    int node = std::stoi(name.substr(4));
    if (online.empty() || online.count(node))
      graph.node("numa:" + std::to_string(node), HardwareKind::Numa, node);
  }
  size_t n = t.devices.size();
  if (t.peerAccess.size() == n * n)
    for (size_t source = 0; source < n; ++source)
      for (size_t dest = 0; dest < n; ++dest) {
        if (source == dest) continue;
        // Copy streams run on the source; remote-read kernels run on the
        // destination. Directional CUDA peer capability must match each path.
        for (auto kind :
             {PathKind::PeerCopy, PathKind::PeerRead, PathKind::PeerWrite}) {
          bool accessible = kind != PathKind::PeerRead
                                ? t.peerAccess[source * n + dest]
                                : t.peerAccess[dest * n + source];
          if (!accessible) continue;
          std::string prefix = kind == PathKind::PeerCopy   ? "peer-copy:"
                               : kind == PathKind::PeerRead ? "peer-read:"
                                                            : "peer-write:";
          t.paths.push_back(
              {t.gpuNodes[source],
               t.gpuNodes[dest],
               kind,
               prefix + t.gpuPciIds[source] + "->" + t.gpuPciIds[dest],
               {}});
        }
      }
  for (auto const& adapter : children(sysfs / "class/infiniband")) {
    std::error_code error;
    auto device = fs::canonical(adapter / "device", error);
    if (error || !pciAddress(device.filename().string())) continue;
    auto id = device.filename().string();
    auto index = graph.pci(id, HardwareKind::Nic);
    for (auto const& port : children(adapter / "ports")) {
      std::istringstream name(port.filename().string());
      int portId = 0;
      if (!(name >> portId) || portId <= 0) continue;
      auto state = read(port / "state");
      bool active = state.find("ACTIVE") != std::string::npos;
      double rate = number(port / "rate") / 8;  // sysfs reports Gb/sec.
      t.networkPorts.push_back(
          {adapter.filename().string(),
           id,
           portId,
           numa(device),
           index,
           active,
           {rate, -1, rate > 0 ? Evidence::Advertised : Evidence::Unknown}});
    }
  }
  std::vector<std::pair<int, size_t>> nodes;
  for (size_t i = 0; i < t.hardware.size(); ++i)
    if (t.hardware[i].kind == HardwareKind::Numa)
      nodes.push_back({t.hardware[i].numa, i});
  for (auto const& [source, index] : nodes) {
    std::istringstream distances(
        read(sysfs / "devices/system/node" / ("node" + std::to_string(source)) /
             "distance"));
    // Linux distance columns follow online node IDs in ascending order.
    for (int destination : online) {
      auto found = std::find_if(
          nodes.begin(), nodes.end(),
          [&](auto const& item) { return item.first == destination; });
      int distance = 0;
      if (!(distances >> distance)) break;
      if (source != destination && distance > 0 && found != nodes.end())
        t.links.push_back({index,
                           found->second,
                           "numa:" + std::to_string(source) +
                               "->numa:" + std::to_string(destination),
                           {},
                           distance});
    }
  }
}

std::string hardwareIdentity(Topology const& t) {
  std::ostringstream text;
  text << ";hardware=v1" << std::setprecision(17);
  for (auto const& node : t.hardware)
    text << ";hw=" << node.id << ':' << int(node.kind) << ':' << node.numa;
  for (auto const& link : t.links)
    text << ";link=" << link.resource << ':' << link.performance.bandwidthGBs
         << ':' << int(link.performance.evidence) << ':' << link.numaDistance;
  for (auto const& path : t.paths)
    text << ";path=" << path.resource << ":" << int(path.kind);
  for (auto const& port : t.networkPorts)
    text << ";nic=" << port.name << ':' << port.port << ':' << port.active
         << ':' << port.transmit.bandwidthGBs;
  return text.str();
}
}  // namespace uccl_lite

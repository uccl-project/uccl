#include "../../src/topology/discovery.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
void write(fs::path const& path, std::string const& value) {
  fs::create_directories(path.parent_path());
  std::ofstream(path) << value;
}
struct Fixture {
  fs::path root;
  Fixture() {
    char path[] = "build/topology-fixture-XXXXXX";
    auto name = mkdtemp(path);
    if (!name) throw std::runtime_error("cannot create fixture");
    root = fs::absolute(name);
  }
  ~Fixture() { fs::remove_all(root); }
  fs::path pci(std::string const& path, int numa, int lanes) {
    auto device = root / "devices" / path;
    write(device / "numa_node", std::to_string(numa));
    write(device / "max_link_speed", "16.0 GT/s PCIe");
    write(device / "max_link_width", std::to_string(lanes));
    fs::create_directories(root / "bus/pci/devices");
    fs::create_directory_symlink(device,
                                 root / "bus/pci/devices" / device.filename());
    return device;
  }
};
}  // namespace

int main() {
  try {
    using namespace uccl_lite;
    Fixture f;
    f.pci("pci0000:00/0000:00:01.0", 3, 8);
    f.pci("pci0000:00/0000:00:01.0/0000:01:00.0", 3, 16);
    f.pci("pci0000:00/0000:00:01.0/0000:02:00.0", 3, 16);
    auto nic = f.pci("pci0000:80/0000:80:00.0", 7, 16);
    auto ib = f.root / "class/infiniband/mlx5_test";
    fs::create_directories(ib);
    fs::create_directory_symlink(nic, ib / "device");
    write(ib / "ports/1/rate", "200 Gb/sec (4X HDR)");
    write(ib / "ports/1/state", "4: ACTIVE");
    write(f.root / "devices/system/node/online", "0,3,7");
    write(f.root / "devices/system/node/node0/distance", "10 20 30");
    write(f.root / "devices/system/node/node3/distance", "20 10 21");
    write(f.root / "devices/system/node/node7/distance", "30 21 10");
    Topology t;
    t.devices = {9, 2};
    t.gpuPciIds = {"0000:02:00.0", "0000:01:00.0"};
    discoverHardware(t, f.root);
    require(t.hardware[t.gpuNodes[0]].id == "pci:0000:02:00.0",
            "rank mapping must preserve caller order");
    require(t.hardware[t.gpuNodes[0]].numa == 3, "non-contiguous NUMA ID");
    int shared = 0;
    for (auto const& link : t.links)
      if (link.resource == "pci:0000:00:01.0->numa:3") {
        ++shared;
        require(std::abs(link.performance.bandwidthGBs -
                         16 * 8 * 128.0 / 130 / 8) < 1e-6,
                "shared uplink width must limit advertised bandwidth");
        require(link.performance.evidence == Evidence::Advertised,
                "discovery cannot claim measured throughput");
      }
    require(shared == 1, "shared uplink must be represented once");
    require(t.networkPorts.size() == 1 && t.networkPorts[0].active &&
                t.networkPorts[0].transmit.bandwidthGBs == 25 &&
                t.networkPorts[0].numa == 7,
            "NIC topology, active state and bit-to-byte conversion");
    for (auto const& node : t.hardware)
      require(node.memory.evidence == Evidence::Unknown,
              "memory rate is unknown");
    bool numaLink = false;
    for (auto const& link : t.links)
      if (link.resource == "numa:3->numa:7") {
        numaLink = true;
        require(
            link.numaDistance == 21 &&
                link.performance.evidence == Evidence::Unknown,
            "NUMA distance must not be interpreted as latency or bandwidth");
      }
    require(numaLink, "non-contiguous NUMA distance columns were not mapped");
    auto before = hardwareIdentity(t);
    write(ib / "ports/1/state", "1: DOWN");
    discoverHardware(t, f.root);
    require(!t.networkPorts[0].active && hardwareIdentity(t) != before,
            "NIC state changes invalidate topology identity");
    t.devices = {4};
    t.gpuPciIds = {"0000:ff:00.0"};
    discoverHardware(t, f.root);
    require(t.hardware[t.gpuNodes[0]].numa == -1,
            "missing NUMA remains unknown");
    for (auto const& link : t.links)
      require(link.source != t.gpuNodes[0] && link.destination != t.gpuNodes[0],
              "missing PCI path must not create connectivity");
    std::cout << "Topology discovery: rank mapping, shared links, NIC "
                 "identity, and unknown capabilities passed\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

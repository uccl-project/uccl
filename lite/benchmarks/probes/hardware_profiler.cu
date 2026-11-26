// Local topology and bandwidth probe. All timed GPU copies run concurrently;
// CPU buffers are first-touched by workers pinned to their reported NUMA node.
#include <cuda_runtime.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/utsname.h>

namespace {
using Clock = std::chrono::steady_clock;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
std::string read(std::string const& path) {
  std::ifstream input(path);
  std::string value;
  std::getline(input, value);
  return value;
}
std::string cpuModel() {
  std::ifstream input("/proc/cpuinfo");
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto colon = line.find(':');
      return colon == std::string::npos ? line : line.substr(colon + 2);
    }
  }
  return "unknown";
}
std::vector<int> parseList(std::string const& value) {
  std::vector<int> result;
  std::stringstream input(value);
  std::string part;
  while (std::getline(input, part, ',')) {
    auto dash = part.find('-');
    int first = std::stoi(part);
    int last = dash == std::string::npos ? first : std::stoi(part.substr(dash + 1));
    if (first < 0 || last < first || last >= CPU_SETSIZE)
      throw std::invalid_argument("invalid CPU list");
    for (int cpu = first; cpu <= last; ++cpu) result.push_back(cpu);
  }
  return result;
}
std::vector<int> parseDevices(std::string const& value) {
  auto devices = parseList(value);
  if (devices.empty() || std::set<int>(devices.begin(), devices.end()).size() !=
                             devices.size())
    throw std::invalid_argument("devices must be distinct");
  return devices;
}
void pin(int cpu) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask))
    throw std::runtime_error("cannot pin CPU worker");
}
struct Config {
  std::vector<int> devices;
  std::vector<int> peerOrder;
  size_t gpuBytes = 16UL << 20, cpuBytes = 8UL << 20;
  int iterations = 10, warmups = 2, repeats = 3, threadsPerNode = 4;
};
Config parse(int argc, char** argv) {
  Config cfg;
  int available = 0;
  check(cudaGetDeviceCount(&available));
  for (int d = 0; d < available; ++d) cfg.devices.push_back(d);
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "Usage: uccl_lite_hardware_profiler [--devices 0,1] "
                   "[--peer-order 0,1,3,2] "
                   "[--gpu-bytes N] [--cpu-bytes N] [--iterations N] "
                   "[--warmups N] [--repeats N] [--threads-per-node N]\n";
      std::exit(0);
    }
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    std::string value = argv[++i];
    if (arg == "--devices") cfg.devices = parseDevices(value);
    else if (arg == "--peer-order") cfg.peerOrder = parseDevices(value);
    else if (arg == "--gpu-bytes") cfg.gpuBytes = std::stoull(value);
    else if (arg == "--cpu-bytes") cfg.cpuBytes = std::stoull(value);
    else if (arg == "--iterations") cfg.iterations = std::stoi(value);
    else if (arg == "--warmups") cfg.warmups = std::stoi(value);
    else if (arg == "--repeats") cfg.repeats = std::stoi(value);
    else if (arg == "--threads-per-node") cfg.threadsPerNode = std::stoi(value);
    else throw std::invalid_argument("unknown option: " + arg);
  }
  if (cfg.devices.empty() || cfg.gpuBytes < 4096 || cfg.cpuBytes < 4096 ||
      cfg.cpuBytes % 2 || cfg.iterations < 1 || cfg.warmups < 0 ||
      cfg.repeats < 1 || cfg.threadsPerNode < 1)
    throw std::invalid_argument("invalid profile dimensions");
  for (int d : cfg.devices)
    if (d >= available) throw std::invalid_argument("device unavailable");
  if (cfg.peerOrder.empty()) cfg.peerOrder = cfg.devices;
  if (std::set<int>(cfg.peerOrder.begin(), cfg.peerOrder.end()) !=
      std::set<int>(cfg.devices.begin(), cfg.devices.end()))
    throw std::invalid_argument("peer order must contain each selected GPU once");
  return cfg;
}
std::vector<int> allowedCpus(int node, cpu_set_t const& allowed) {
  auto all = parseList(read("/sys/devices/system/node/node" +
                            std::to_string(node) + "/cpulist"));
  std::vector<int> result;
  for (int cpu : all)
    if (CPU_ISSET(cpu, &allowed)) result.push_back(cpu);
  return result;
}
std::vector<int> numaNodes() {
  std::vector<int> nodes;
  for (auto const& entry : std::filesystem::directory_iterator(
           "/sys/devices/system/node")) {
    std::string name = entry.path().filename().string();
    if (name.rfind("node", 0) == 0 && name.size() > 4 &&
        std::all_of(name.begin() + 4, name.end(), ::isdigit))
      nodes.push_back(std::stoi(name.substr(4)));
  }
  std::sort(nodes.begin(), nodes.end());
  return nodes;
}
void topology(Config const& cfg) {
  utsname host{};
  if (uname(&host)) throw std::runtime_error("cannot query host name");
  int cudaVersion = 0, driverVersion = 0;
  check(cudaRuntimeGetVersion(&cudaVersion));
  check(cudaDriverGetVersion(&driverVersion));
  std::cout << "# UCCL Lite hardware profiler: CUDA pinned host buffers; "
               "decimal GB/s\n";
  std::cout << "# host=" << host.nodename << " cuda_runtime=" << cudaVersion
            << " cuda_driver_api=" << driverVersion << " cpu=\""
            << cpuModel() << "\"\n";
  std::cout << "# GPU topology: PCI path, current link, NUMA, and CUDA peer "
               "capability (1=accessible)\n";
  for (int d : cfg.devices) {
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, d));
    char pci[32];
    check(cudaDeviceGetPCIBusId(pci, sizeof(pci), d));
    std::string busId = pci;
    std::transform(busId.begin(), busId.end(), busId.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });
    auto sysfs = std::string("/sys/bus/pci/devices/") + busId;
    auto path = std::filesystem::weakly_canonical(sysfs).string();
    std::cout << "topology gpu=" << d << " name=\"" << property.name
              << "\" pci=" << busId << " numa=" << read(sysfs + "/numa_node")
              << " link_speed=\"" << read(sysfs + "/current_link_speed")
              << "\" link_width=\"" << read(sysfs + "/current_link_width")
              << "\" max_link_speed=\"" << read(sysfs + "/max_link_speed")
              << "\" max_link_width=\"" << read(sysfs + "/max_link_width")
              << "\" sysfs=" << path << " peers=";
    for (int peer : cfg.devices) {
      int access = 0;
      if (peer != d) check(cudaDeviceCanAccessPeer(&access, d, peer));
      std::cout << access;
    }
    std::cout << '\n';
  }
}
struct Range {
  double minimum, median, maximum;
};
Range summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return {samples.front(), samples[samples.size() / 2], samples.back()};
}
void report(std::string const& name, size_t bytes, int participants,
            int iterations, std::vector<double> const& samples) {
  auto s = summarize(samples);
  std::cout << std::fixed << std::setprecision(3) << "bandwidth metric=" << name
            << " bytes_per_participant=" << bytes
            << " participants=" << participants << " iterations=" << iterations
            << " min_gbs=" << s.minimum << " median_gbs=" << s.median
            << " max_gbs=" << s.maximum << '\n';
}
struct GpuBuffers {
  int device;
  void* host = nullptr;
  void* gpu = nullptr;
  void* peerDestination = nullptr;
  cudaStream_t stream = nullptr;
  explicit GpuBuffers(int d, size_t bytes) : device(d) {
    check(cudaSetDevice(d));
    check(cudaHostAlloc(&host, bytes, cudaHostAllocPortable));
    std::memset(host, 1, bytes);
    check(cudaMalloc(&gpu, bytes));
    check(cudaMalloc(&peerDestination, bytes));
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  ~GpuBuffers() {
    cudaSetDevice(device);
    if (stream) cudaStreamDestroy(stream);
    if (gpu) cudaFree(gpu);
    if (peerDestination) cudaFree(peerDestination);
    if (host) cudaFreeHost(host);
  }
};
void gpuCopy(Config const& cfg, std::vector<GpuBuffers*> const& buffers,
             cudaMemcpyKind direction) {
  std::vector<double> samples;
  for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<int> done{0};
    std::atomic<bool> release{false};
    std::vector<std::exception_ptr> failures(buffers.size());
    std::vector<std::thread> workers;
    for (size_t i = 0; i < buffers.size(); ++i) {
      workers.emplace_back([&, i] {
        bool published = false;
        try {
          auto& b = *buffers[i];
          check(cudaSetDevice(b.device));
          auto dst = direction == cudaMemcpyHostToDevice ? b.gpu : b.host;
          auto src = direction == cudaMemcpyHostToDevice ? b.host : b.gpu;
          for (int n = 0; n < cfg.warmups; ++n)
            check(cudaMemcpyAsync(dst, src, cfg.gpuBytes, direction, b.stream));
          check(cudaStreamSynchronize(b.stream));
          ready.fetch_add(1);
          published = true;
          while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
          for (int n = 0; n < cfg.iterations; ++n)
            check(cudaMemcpyAsync(dst, src, cfg.gpuBytes, direction, b.stream));
          check(cudaStreamSynchronize(b.stream));
        } catch (...) {
          failures[i] = std::current_exception();
          if (!published) ready.fetch_add(1);
        }
        done.fetch_add(1);
        while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
      });
    }
    while (ready.load() != int(buffers.size())) std::this_thread::yield();
    auto start = Clock::now();
    go.store(true, std::memory_order_release);
    while (done.load() != int(buffers.size())) std::this_thread::yield();
    auto elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    release.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    for (auto const& failure : failures)
      if (failure) std::rethrow_exception(failure);
    samples.push_back(double(cfg.gpuBytes) * cfg.iterations * buffers.size() /
                      elapsed / 1e9);
  }
  report(direction == cudaMemcpyHostToDevice ? "pcie_h2d_concurrent"
                                            : "pcie_d2h_concurrent",
         cfg.gpuBytes, int(buffers.size()), cfg.iterations, samples);
  for (auto* b : buffers) {
    check(cudaSetDevice(b->device));
    uint8_t first = 0, last = 0;
    if (direction == cudaMemcpyHostToDevice) {
      check(cudaMemcpy(&first, b->gpu, 1, cudaMemcpyDeviceToHost));
      check(cudaMemcpy(&last, static_cast<uint8_t*>(b->gpu) + cfg.gpuBytes - 1,
                       1, cudaMemcpyDeviceToHost));
    } else {
      first = *static_cast<uint8_t*>(b->host);
      last = *(static_cast<uint8_t*>(b->host) + cfg.gpuBytes - 1);
    }
    if (first != 1 || last != 1)
      throw std::runtime_error("GPU copy output check failed");
  }
}
void checkPeerOutput(GpuBuffers const& destination, size_t bytes) {
  check(cudaSetDevice(destination.device));
  uint8_t first = 0, last = 0;
  check(cudaMemcpy(&first, destination.peerDestination, 1,
                   cudaMemcpyDeviceToHost));
  check(cudaMemcpy(
      &last,
      static_cast<uint8_t*>(destination.peerDestination) + bytes - 1, 1,
      cudaMemcpyDeviceToHost));
  if (first != 1 || last != 1)
    throw std::runtime_error("GPU peer-copy output check failed");
}
void peerBandwidth(Config const& cfg,
                   std::vector<GpuBuffers*> const& buffers) {
  std::map<int, GpuBuffers*> byDevice;
  for (auto* buffer : buffers) byDevice[buffer->device] = buffer;
  for (auto* source : buffers)
    for (auto* destination : buffers) {
      if (source == destination) continue;
      int accessible = 0;
      check(cudaDeviceCanAccessPeer(&accessible, source->device,
                                    destination->device));
      if (!accessible) continue;
      check(cudaSetDevice(source->device));
      auto enabled = cudaDeviceEnablePeerAccess(destination->device, 0);
      if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
      else check(enabled);
      std::vector<double> samples;
      for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
        check(cudaSetDevice(source->device));
        for (int n = 0; n < cfg.warmups; ++n)
          check(cudaMemcpyPeerAsync(destination->peerDestination,
                                    destination->device, source->gpu,
                                    source->device, cfg.gpuBytes,
                                    source->stream));
        check(cudaStreamSynchronize(source->stream));
        auto start = Clock::now();
        for (int n = 0; n < cfg.iterations; ++n)
          check(cudaMemcpyPeerAsync(destination->peerDestination,
                                    destination->device, source->gpu,
                                    source->device, cfg.gpuBytes,
                                    source->stream));
        check(cudaStreamSynchronize(source->stream));
        auto elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        samples.push_back(double(cfg.gpuBytes) * cfg.iterations / elapsed / 1e9);
      }
      report("pcie_p2p_" + std::to_string(source->device) + "to" +
                 std::to_string(destination->device),
             cfg.gpuBytes, 1, cfg.iterations, samples);
      checkPeerOutput(*destination, cfg.gpuBytes);
    }
  if (cfg.peerOrder.size() < 2) return;
  std::ostringstream order;
  for (size_t i = 0; i < cfg.peerOrder.size(); ++i) {
    int source = cfg.peerOrder[i];
    int destination = cfg.peerOrder[(i + 1) % cfg.peerOrder.size()];
    int accessible = 0;
    check(cudaDeviceCanAccessPeer(&accessible, source, destination));
    if (!accessible) {
      std::cout << "# concurrent peer ring skipped: " << source << "->"
                << destination << " lacks peer access\n";
      return;
    }
    if (i) order << ',';
    order << source;
  }
  std::vector<double> samples;
  for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
    std::atomic<int> ready{0}, done{0};
    std::atomic<bool> go{false}, release{false};
    std::vector<std::exception_ptr> failures(cfg.peerOrder.size());
    std::vector<std::thread> workers;
    for (size_t i = 0; i < cfg.peerOrder.size(); ++i) {
      workers.emplace_back([&, i] {
        bool published = false;
        try {
          auto* source = byDevice.at(cfg.peerOrder[i]);
          auto* destination =
              byDevice.at(cfg.peerOrder[(i + 1) % cfg.peerOrder.size()]);
          check(cudaSetDevice(source->device));
          for (int n = 0; n < cfg.warmups; ++n)
            check(cudaMemcpyPeerAsync(destination->peerDestination,
                                      destination->device, source->gpu,
                                      source->device, cfg.gpuBytes,
                                      source->stream));
          check(cudaStreamSynchronize(source->stream));
          ready.fetch_add(1);
          published = true;
          while (!go.load(std::memory_order_acquire))
            std::this_thread::yield();
          for (int n = 0; n < cfg.iterations; ++n)
            check(cudaMemcpyPeerAsync(destination->peerDestination,
                                      destination->device, source->gpu,
                                      source->device, cfg.gpuBytes,
                                      source->stream));
          check(cudaStreamSynchronize(source->stream));
        } catch (...) {
          failures[i] = std::current_exception();
          if (!published) ready.fetch_add(1);
        }
        done.fetch_add(1);
        while (!release.load(std::memory_order_acquire))
          std::this_thread::yield();
      });
    }
    while (ready.load() != int(workers.size())) std::this_thread::yield();
    auto start = Clock::now();
    go.store(true, std::memory_order_release);
    while (done.load() != int(workers.size())) std::this_thread::yield();
    auto elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    release.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    for (auto const& failure : failures)
      if (failure) std::rethrow_exception(failure);
    samples.push_back(double(cfg.gpuBytes) * cfg.iterations *
                      cfg.peerOrder.size() / elapsed / 1e9);
  }
  report("pcie_p2p_ring_" + order.str(), cfg.gpuBytes,
         int(cfg.peerOrder.size()), cfg.iterations, samples);
  for (int device : cfg.peerOrder)
    checkPeerOutput(*byDevice.at(device), cfg.gpuBytes);
}
float bf16(uint16_t x) {
  uint32_t bits = uint32_t(x) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return uint16_t(bits >> 16);
}
enum class CpuOp { Copy, Reduce };
void cpuBandwidth(Config const& cfg, std::vector<int> const& cpus, int node,
                  CpuOp op) {
  if (cpus.empty()) return;
  std::vector<double> samples;
  for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<int> done{0};
    std::atomic<bool> release{false};
    std::vector<std::exception_ptr> failures(cpus.size());
    std::vector<uint64_t> checksums(cpus.size());
    std::vector<std::thread> workers;
    for (size_t t = 0; t < cpus.size(); ++t) {
      workers.emplace_back([&, t] {
        bool published = false;
        try {
          pin(cpus[t]);
          if (op == CpuOp::Copy) {
            std::vector<uint8_t> source(cfg.cpuBytes, uint8_t(t + 1));
            std::vector<uint8_t> output(cfg.cpuBytes, 0);
            for (int n = 0; n < cfg.warmups; ++n)
              std::memcpy(output.data(), source.data(), cfg.cpuBytes);
            ready.fetch_add(1);
            published = true;
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int n = 0; n < cfg.iterations; ++n)
              std::memcpy(output.data(), source.data(), cfg.cpuBytes);
            checksums[t] = (uint64_t(output[t % cfg.cpuBytes]) << 8) |
                           output.back();
          } else {
            size_t elements = cfg.cpuBytes / 2;
            std::vector<uint16_t> source(4 * elements);
            std::vector<uint16_t> output(elements);
            for (size_t i = 0; i < source.size(); ++i)
              source[i] = uint16_t(0x3f80 + ((i / elements) << 7));
            auto reduce = [&] {
              for (size_t i = 0; i < elements; ++i) {
                float sum = 0;
                for (int s = 0; s < 4; ++s)
                  sum += bf16(source[s * elements + i]);
                output[i] = toBf16(sum);
              }
            };
            for (int n = 0; n < cfg.warmups; ++n) reduce();
            ready.fetch_add(1);
            published = true;
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int n = 0; n < cfg.iterations; ++n) reduce();
            checksums[t] = (uint64_t(output[t % elements]) << 16) |
                           output.back();
          }
        } catch (...) {
          failures[t] = std::current_exception();
          if (!published) ready.fetch_add(1);
        }
        done.fetch_add(1);
        while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
      });
    }
    while (ready.load() != int(cpus.size())) std::this_thread::yield();
    auto start = Clock::now();
    go.store(true, std::memory_order_release);
    while (done.load() != int(cpus.size())) std::this_thread::yield();
    auto elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    release.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    for (auto const& failure : failures)
      if (failure) std::rethrow_exception(failure);
    for (size_t t = 0; t < cpus.size(); ++t) {
      uint64_t expected = op == CpuOp::Copy
                              ? (uint64_t(t + 1) << 8) | uint64_t(t + 1)
                              : (uint64_t(toBf16(15.0f)) << 16) |
                                    toBf16(15.0f);
      if (checksums[t] != expected)
        throw std::runtime_error("CPU output check failed");
    }
    // Copy counts one read plus one write; BF16 reduction counts four source
    // reads plus one result write. These are useful bytes, not DRAM counters.
    int streams = op == CpuOp::Copy ? 2 : 5;
    samples.push_back(double(streams) * cfg.cpuBytes * cfg.iterations *
                      cpus.size() / elapsed / 1e9);
  }
  report(std::string(op == CpuOp::Copy ? "cpu_copy" : "cpu_bf16_reduce4") +
             "_node" + std::to_string(node),
         cfg.cpuBytes, int(cpus.size()), cfg.iterations, samples);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    auto cfg = parse(argc, argv);
    topology(cfg);
    std::cout << "# Timed scope: concurrent worker submission and completion; "
                 "allocation, first touch, warmup, and checks excluded\n";
    std::cout << "# repeats=" << cfg.repeats << " warmups=" << cfg.warmups
              << " iterations=" << cfg.iterations << '\n';
    std::vector<std::unique_ptr<GpuBuffers>> owned;
    std::vector<GpuBuffers*> buffers;
    for (int d : cfg.devices) {
      owned.push_back(std::make_unique<GpuBuffers>(d, cfg.gpuBytes));
      buffers.push_back(owned.back().get());
    }
    gpuCopy(cfg, buffers, cudaMemcpyHostToDevice);
    gpuCopy(cfg, buffers, cudaMemcpyDeviceToHost);
    peerBandwidth(cfg, buffers);
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
      throw std::runtime_error("cannot query CPU affinity");
    for (int node : numaNodes()) {
      auto cpus = allowedCpus(node, allowed);
      if (cpus.empty()) continue;
      cpus.resize(std::min(cpus.size(), size_t(cfg.threadsPerNode)));
      cpuBandwidth(cfg, cpus, node, CpuOp::Copy);
      cpuBandwidth(cfg, cpus, node, CpuOp::Reduce);
    }
  } catch (std::exception const& error) {
    std::cerr << "hardware profiler: " << error.what() << '\n';
    return 1;
  }
}

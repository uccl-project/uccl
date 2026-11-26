#include "uccl_lite/hardware_calibration.hpp"
#include "uccl_lite/runtime/build_identity.hpp"
#include <gnu/libc-version.h>
#include <linux/mempolicy.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
void validate(HostProbeOptions const& o) {
  if (o.bytesPerWorker < (1UL << 20) || o.bytesPerWorker % 8192 ||
      o.bytesPerWorker > (1UL << 30) || o.workersPerNode < 1 ||
      o.workersPerNode > 64 || o.warmups < 0 || o.iterations < 1 ||
      o.repetitions < 3)
    throw std::invalid_argument("invalid host probe configuration");
}
struct Memory {
  void* data = MAP_FAILED;
  size_t bytes;
  Memory(size_t size, int node) : bytes(size) {
    if (node < 0 || node >= 1024)
      throw std::invalid_argument("invalid memory NUMA node");
    data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) throw std::runtime_error("host probe mmap failed");
    std::vector<unsigned long> mask(
        size_t(node) / (8 * sizeof(unsigned long)) + 1, 0);
    mask[size_t(node) / (8 * sizeof(unsigned long))] |=
        1UL << (node % (8 * sizeof(unsigned long)));
    if (syscall(SYS_mbind, data, bytes, MPOL_BIND, mask.data(),
                mask.size() * 8 * sizeof(unsigned long), 0)) {
      munmap(data, bytes);
      data = MAP_FAILED;
      throw std::runtime_error("host probe cannot enforce NUMA placement");
    }
  }
  ~Memory() {
    if (data != MAP_FAILED) munmap(data, bytes);
  }
  void verifyPlacement(int node) const {
    for (auto address :
         {data, static_cast<void*>(static_cast<char*>(data) + bytes - 1)}) {
      int actual = -1;
      if (syscall(SYS_get_mempolicy, &actual, nullptr, 0, address,
                  MPOL_F_NODE | MPOL_F_ADDR) ||
          actual != node)
        throw std::runtime_error(
            "host probe NUMA placement verification failed");
    }
  }
};
struct Workers {
  std::mutex mutex;
  std::condition_variable cv;
  int ready = 0, done = 0;
  bool go = false, verify = false, cancel = false;
  std::vector<std::thread> threads;
  ~Workers() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      cancel = true;
    }
    cv.notify_all();
    for (auto& thread : threads)
      if (thread.joinable()) thread.join();
  }
};
std::vector<double> measure(HostCopyPath const& path, size_t bytes,
                            HostProbeOptions const& options) {
  std::vector<double> times;
  for (int repetition = 0; repetition < options.repetitions; ++repetition) {
    std::vector<std::exception_ptr> errors(path.cpus.size());
    Workers workers;
    for (size_t i = 0; i < path.cpus.size(); ++i)
      workers.threads.emplace_back([&, i] {
        bool ready = false;
        try {
          cpu_set_t mask;
          CPU_ZERO(&mask);
          CPU_SET(path.cpus[i], &mask);
          if (sched_setaffinity(0, sizeof(mask), &mask))
            throw std::runtime_error("host worker pin failed");
          Memory source(bytes, path.sourceNode),
              destination(bytes, path.destinationNode);
          std::memset(source.data, int(i + 1), bytes);
          std::memset(destination.data, 0, bytes);
          source.verifyPlacement(path.sourceNode);
          destination.verifyPlacement(path.destinationNode);
          auto copy = [&] {
            std::memcpy(destination.data, source.data, bytes);
            std::atomic_signal_fence(std::memory_order_seq_cst);
          };
          for (int j = 0; j < options.warmups; ++j) copy();
          {
            std::unique_lock<std::mutex> lock(workers.mutex);
            ++workers.ready;
            ready = true;
            workers.cv.notify_all();
            workers.cv.wait(lock, [&] { return workers.go || workers.cancel; });
            if (workers.cancel) return;
          }
          for (int j = 0; j < options.iterations; ++j) copy();
          {
            std::unique_lock<std::mutex> lock(workers.mutex);
            ++workers.done;
            workers.cv.notify_all();
            workers.cv.wait(lock,
                            [&] { return workers.verify || workers.cancel; });
            if (workers.cancel) return;
          }
          if (std::memcmp(source.data, destination.data, bytes))
            throw std::runtime_error("host copy full-buffer mismatch");
          auto* input = static_cast<unsigned char const*>(source.data);
          if (!std::all_of(input, input + bytes,
                           [i](unsigned char value) { return value == i + 1; }))
            throw std::runtime_error("host copy modified its input");
        } catch (...) {
          std::lock_guard<std::mutex> lock(workers.mutex);
          errors[i] = std::current_exception();
          if (!ready) ++workers.ready;
          workers.cancel = true;
          workers.cv.notify_all();
        }
      });
    std::unique_lock<std::mutex> lock(workers.mutex);
    workers.cv.wait(lock, [&] {
      return workers.ready == int(path.cpus.size()) || workers.cancel;
    });
    if (!workers.cancel) {
      auto begin = std::chrono::steady_clock::now();
      workers.go = true;
      workers.cv.notify_all();
      workers.cv.wait(lock, [&] {
        return workers.done == int(path.cpus.size()) || workers.cancel;
      });
      times.push_back(std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - begin)
                          .count() /
                      options.iterations);
      workers.verify = true;
      workers.cv.notify_all();
    }
    lock.unlock();
    for (auto& thread : workers.threads) thread.join();
    for (auto const& error : errors)
      if (error) std::rethrow_exception(error);
  }
  return times;
}
}  // namespace
std::vector<HostCopyPath> registerHostCopyPaths(Topology& t,
                                                HostProbeOptions const& o) {
  validate(o);
  std::map<int, size_t> nodes;
  for (size_t i = 0; i < t.hardware.size(); ++i)
    if (t.hardware[i].kind == HardwareKind::Numa &&
        t.permittedCpuIds.count(t.hardware[i].numa) &&
        !t.permittedCpuIds.at(t.hardware[i].numa).empty())
      nodes[t.hardware[i].numa] = i;
  std::vector<HostCopyPath> result;
  auto registered = t.paths;
  for (auto const& [source, s] : nodes)
    for (auto const& [dest, d] : nodes) {
      auto cpus = t.permittedCpuIds.at(dest);
      cpus.resize(std::min(cpus.size(), size_t(o.workersPerNode)));
      if (std::set<int>(cpus.begin(), cpus.end()).size() != cpus.size())
        throw std::invalid_argument("duplicate host worker CPU");
      std::ostringstream key;
      key << "host-copy-payload:numa:" << source << "->numa:" << dest
          << ":cpus=";
      for (auto cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE)
          throw std::invalid_argument("invalid host worker CPU");
        key << cpu << ',';
      }
      auto name = key.str();
      if (std::none_of(
              registered.begin(), registered.end(),
              [&](HardwarePath const& p) { return p.resource == name; }))
        registered.push_back({s, d, PathKind::HostCopy, name, {}});
      result.push_back({name, source, dest, cpus});
    }
  if (result.empty())
    throw std::invalid_argument("no CPU-addressable NUMA domains in topology");
  t.paths = std::move(registered);
  return result;
}
HardwareProfile probeHostResources(Topology& t, HostProbeOptions const& o) {
  auto paths = registerHostCopyPaths(t, o);
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof(allowed), &allowed))
    throw std::runtime_error("cannot query probe CPU affinity");
  for (auto const& path : paths)
    for (int cpu : path.cpus)
      if (!CPU_ISSET(cpu, &allowed))
        throw std::invalid_argument(
            "host probe placement no longer matches caller affinity");
  HardwareProfile result;
  result.calibration.topologyIdentity =
      t.hardwareIdentity.empty() ? t.identity : t.hardwareIdentity;
  std::ostringstream identity;
  identity
      << "uccl-lite-host-copy-probe-v2;accounting=payload;build="
      << runtime::artifactIdentity(
             reinterpret_cast<uintptr_t>(&probeHostResources))
      << ";libc=" << gnu_get_libc_version() << ";memcpy-build="
      << runtime::artifactIdentity(reinterpret_cast<uintptr_t>(&std::memcpy))
      << ";glibc-tunables="
      << (std::getenv("GLIBC_TUNABLES") ? std::getenv("GLIBC_TUNABLES") : "")
      << ";bytes=" << o.bytesPerWorker
      << ";workers-per-node=" << o.workersPerNode << ";warmups=" << o.warmups
      << ";iterations=" << o.iterations << ";repetitions=" << o.repetitions
      << ";scope=host-common-interval;placement=mbind;libc-memcpy;no-baseline";
  result.calibration.identity = identity.str();
  for (auto const& path : paths) {
    auto small = measure(path, o.bytesPerWorker / 2, o);
    auto large = measure(path, o.bytesPerWorker, o);
    auto median = [](std::vector<double> x) {
      std::sort(x.begin(), x.end());
      size_t n = x.size();
      return n % 2 ? x[n / 2] : (x[n / 2 - 1] + x[n / 2]) / 2;
    };
    double a = median(small), b = median(large);
    if (!(b > a && a > 0))
      throw std::runtime_error("nonpositive host-copy slope");
    // Fit nonnegative fixed cost and positive time/byte. If the unconstrained
    // intercept is negative, refit the slope through the origin rather than
    // silently changing only the intercept. All host-copy paths count payload
    // bytes, consistently with end-to-end GPU/NIC paths.
    double x = double(o.bytesPerWorker / 2) * path.cpus.size();
    double y = double(o.bytesPerWorker) * path.cpus.size();
    double slope = (b - a) / (y - x);
    double latency = a - slope * x;
    if (latency < 0) {
      latency = 0;
      slope = (x * a + y * b) / (x * x + y * y);
    }
    double bandwidth = 1 / slope / 1000;
    result.calibration.resources.push_back(
        {path.resource,
         {bandwidth, latency, Evidence::Measured},
         o.repetitions});
    std::string scope =
        "host common interval, worker wakeup/copy/completion; "
        "allocation/warmup/verification excluded; workers on destination NUMA";
    result.observations.push_back({path.resource, "host-memcpy", scope,
                                   o.bytesPerWorker / 2, o.warmups,
                                   o.iterations, std::move(small)});
    result.observations.push_back({path.resource, "host-memcpy", scope,
                                   o.bytesPerWorker, o.warmups, o.iterations,
                                   std::move(large)});
  }
  return result;
}
}  // namespace uccl_lite

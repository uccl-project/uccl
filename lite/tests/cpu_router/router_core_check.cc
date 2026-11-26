#include "router_core.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <sched.h>

using namespace uccl_lite::cpu_router;
static MemorySpan host(void* p, size_t bytes) {
  return {p, bytes, MemoryKind::NumaHost, -1, 0, {}};
}
static void require(bool condition) {
  if (!condition) throw std::runtime_error("router core check failed");
}
template <class F>
static void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  require(rejected);
}
int main() {
  try {
    std::array<uint32_t, 3> a{3, 1, 3}, b{1, 7, 9};
    std::array<uint32_t, 8> out{};
    MemorySpan sources[] = {host(a.data(), sizeof(a)),
                            host(b.data(), sizeof(b))};
    auto output = host(out.data(), sizeof(out));
    require(RouterCore::process(sources, 2, output, PackBytes{}) == 24);
    require(out[0] == 3 && out[3] == 1 && out[5] == 9);
    require(RouterCore::process(sources, 2, output, DeduplicateRecords{4}) ==
            16);
    require(out[0] == 3 && out[1] == 1 && out[2] == 7 && out[3] == 9);
    require(RouterCore::process(sources, 2, host(out.data(), 16),
                                DeduplicateRecords{4}) == 16);
    require(RouterCore::process(nullptr, 0, output, PackBytes{}) == 0);
    require(RouterCore::process(nullptr, 0, output, DeduplicateRecords{4}) ==
            0);
    rejects([&] { RouterCore::process(nullptr, 1, output); });
    rejects([&] {
      RouterCore::process(sources, 2, host(out.data(), 4), PackBytes{});
    });
    rejects([&] {
      RouterCore::process(sources, 2, host(out.data(), 4),
                          DeduplicateRecords{4});
    });
    rejects([&] {
      RouterCore::process(sources, 2, output, DeduplicateRecords{0});
    });
    rejects([&] {
      RouterCore::process(sources, 2, output, DeduplicateRecords{5});
    });
    rejects([&] { RouterCore::process(sources, 2, sources[0], PackBytes{}); });
    rejects([&] {
      RouterCore::process(sources, 2, sources[0], DeduplicateRecords{4});
    });
    auto gpu = output;
    gpu.kind = MemoryKind::Gpu;
    rejects([&] { RouterCore::process(sources, 2, gpu, PackBytes{}); });
    rejects([&] { RouterCore::forward(gpu, sources[0]); });
    rejects([&] { RouterCore::forward(host(out.data(), 4), sources[0]); });
    RouterCore::forward(output, sources[0]);
    require(out[0] == 3 && out[1] == 1 && out[2] == 3);
    RouterCore::forward(host(out.data() + 1, 12), host(out.data(), 12));
    require(out[1] == 3 && out[2] == 1 && out[3] == 3);
    // Custom process can change packet size; only the returned prefix is
    // routed.
    auto custom = [](MemorySpan const* src, size_t count, MemorySpan dst) {
      auto* p = static_cast<uint32_t*>(dst.address);
      for (size_t i = 0; i < count; ++i)
        p[i] = *static_cast<const uint32_t*>(src[i].address) * 2;
      return count * sizeof(uint32_t);
    };
    require(RouterCore::process(sources, 2, output, custom) == 8);
    require(out[0] == 6 && out[1] == 2);
    std::array<uint16_t, 17> x{}, y{}, z{};
    x.fill(0x3f80);
    y.fill(0x4000);
    MemorySpan bf16[] = {host(x.data(), sizeof(x)), host(y.data(), sizeof(y))};
    RouterCore::process(bf16, 2, host(z.data(), sizeof(z)));
    for (auto v : z) require(v == 0x4040);
    CpuExecutor cpu({sched_getcpu()});
    z.fill(0);
    RouterCore::process(cpu, 0, bf16, 2, host(z.data(), sizeof(z))).wait();
    for (auto v : z) require(v == 0x4040);
    struct Context {
      MemorySpan* src;
      MemorySpan dst;
    } context{sources, output};
    RouterCore::process(
        cpu, 0,
        [](void* opaque, int) {
          auto& c = *static_cast<Context*>(opaque);
          RouterCore::process(c.src, 2, c.dst, DeduplicateRecords{4});
        },
        &context)
        .wait();
    require(out[0] == 3 && out[3] == 9);
    std::array<uint32_t, 3> readers{};
    std::array<int, 3> destinations{2, 0, 1};
    RouterCore::multicast(42U, destinations, [&](int dst, uint32_t packet) {
      readers[dst] = packet;
    });
    for (auto v : readers) require(v == 42);
    // A small bounded queue forces reuse while batches interleave with normal
    // submissions. Test both spinning workers and workers woken from sleep.
    cpu_set_t permitted;
    CPU_ZERO(&permitted);
    require(sched_getaffinity(0, sizeof(permitted), &permitted) == 0);
    std::vector<int> cpus;
    for (int id = 0; id < CPU_SETSIZE && cpus.size() < 2; ++id)
      if (CPU_ISSET(id, &permitted)) cpus.push_back(id);
    CpuExecutor batches(cpus, 2);
    std::array<std::atomic<int>, 2> counters{};
    std::array<size_t, 2> lanes{0, 1};
    auto increment = [](void* opaque, int lane) {
      auto* counters = static_cast<std::atomic<int>*>(opaque);
      counters[lane].fetch_add(1, std::memory_order_relaxed);
    };
    for (int round = 0; round < 4; ++round) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      for (int batch = 0; batch < 33; ++batch)
        batches.enqueueBatch(lanes.data(), cpus.size(), increment,
                             counters.data());
      for (size_t lane = 0; lane < cpus.size(); ++lane) {
        batches.enqueue(lane, increment, counters.data()).wait();
        require(counters[lane].load() == (round + 1) * 34);
      }
    }
    rejects(
        [&] { batches.enqueueBatch(nullptr, 1, increment, counters.data()); });
    size_t invalid[] = {0, cpus.size()};
    rejects(
        [&] { batches.enqueueBatch(invalid, 2, increment, counters.data()); });
    size_t duplicate[] = {0, 0};
    rejects([&] {
      batches.enqueueBatch(duplicate, 2, increment, counters.data());
    });
    for (size_t lane = 0; lane < cpus.size(); ++lane) {
      batches.enqueue(lane, increment, counters.data()).wait();
      require(counters[lane].load() == 137);
    }
    // Both producers may wait for credit on the same lane. Re-read its tail
    // after each wake so a producer cannot overwrite another producer's slot.
    std::thread batchProducer([&] {
      for (int i = 0; i < 129; ++i)
        batches.enqueueBatch(lanes.data(), cpus.size(), increment,
                             counters.data());
    });
    std::thread singleProducer([&] {
      for (int i = 0; i < 129; ++i)
        batches.enqueue(0, increment, counters.data());
    });
    batchProducer.join();
    singleProducer.join();
    for (size_t lane = 0; lane < cpus.size(); ++lane) {
      batches.enqueue(lane, increment, counters.data()).wait();
      require(counters[lane].load() == 267 + (lane == 0 ? 129 : 0));
    }
    batches.stop();
    bool stopped = false;
    try {
      batches.enqueueBatch(lanes.data(), cpus.size(), increment,
                           counters.data());
    } catch (std::runtime_error const&) {
      stopped = true;
    }
    require(stopped);
    std::puts("Router forward/process/multicast checks passed");
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}

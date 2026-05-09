#pragma once
#include <nanobind/nanobind.h>
#include <map>
#include <utility>
#include <vector>

namespace uccl {
namespace nb = nanobind;

// Keyed by (device_index, low_latency_mode). Allows a single process to
// register two independent sets of UcclProxy instances on the same device --
// one for high-throughput mode (DeepEP "normal" / use_normal_mode=True) and
// one for low-latency mode -- so that two coexisting Buffer instances don't
// collide on shared resources (per-device thread pool, /dev/shm barriers,
// etc.).
using ProxyRegistryKey = std::pair<int, bool>;
extern std::map<ProxyRegistryKey, std::vector<nb::object>> g_proxies_by_dev;

std::map<ProxyRegistryKey, std::vector<nb::object>>& proxies_by_dev();
}  // namespace uccl
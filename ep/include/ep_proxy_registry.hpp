#pragma once
#include <nanobind/nanobind.h>
#include <unordered_map>
#include <vector>

namespace uccl {
extern std::unordered_map<int, std::vector<nanobind::object>> g_proxies_by_dev;

std::unordered_map<int, std::vector<nanobind::object>>& proxies_by_dev();
}  // namespace uccl
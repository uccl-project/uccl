#pragma once
#include <nanobind/nanobind.h>
#include <unordered_map>
#include <vector>

namespace uccl {
namespace nb = nanobind;

extern std::unordered_map<int, std::vector<nb::object>> g_proxies_by_dev;

std::unordered_map<int, std::vector<nb::object>>& proxies_by_dev();
}  // namespace uccl
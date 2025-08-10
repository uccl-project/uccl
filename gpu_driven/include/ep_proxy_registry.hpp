#pragma once
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>

namespace uccl {
namespace py = pybind11;

extern std::unordered_map<int, std::vector<py::object>> g_proxies_by_dev;

std::unordered_map<int, std::vector<py::object>>& proxies_by_dev();
}  // namespace uccl
#pragma once

#include "kernel_runtime.hpp"
#include <filesystem>
#include <memory>
#include <unordered_map>

namespace deep_ep::jit {

class KernelRuntimeCache {
  std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;

 public:
  KernelRuntimeCache() = default;

  void clear() { cache.clear(); }

  std::shared_ptr<KernelRuntime> get(std::filesystem::path const& dir_path) {
    // Hit the runtime cache
    if (auto const iterator = cache.find(dir_path); iterator != cache.end())
      return iterator->second;

    if (KernelRuntime::check_validity(dir_path))
      return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);
    return nullptr;
  }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

}  // namespace deep_ep::jit

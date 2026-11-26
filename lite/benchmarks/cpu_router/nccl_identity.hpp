#pragma once

// Benchmark adapter only. The optional router library has no NCCL dependency.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace uccl_lite::benchmark {
inline std::string ncclFallbackIdentity(int version,
                                        std::vector<std::string> records) {
  if (version <= 0) throw std::invalid_argument("invalid NCCL version");
  std::string identity = "nccl:" + std::to_string(version);
  if (records.empty()) return identity;
  std::sort(records.begin(), records.end());
  // Stable configuration cache key, not an authentication/security primitive.
  // Length-prefix every record; environment values cannot inject separators.
  uint64_t hash = 14695981039346656037ULL;
  auto append = [&](unsigned char byte) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  };
  for (auto const& record : records) {
    uint64_t size = record.size();
    for (int i = 0; i < 8; ++i) append((size >> (8 * i)) & 255);
    for (unsigned char byte : record) append(byte);
  }
  std::ostringstream key;
  key << identity << ";config-v1=" << std::hex << std::setw(16)
      << std::setfill('0') << hash;
  return key.str();
}

inline std::vector<std::string> ncclConfiguration(char* const* environment) {
  std::vector<std::string> records;
  for (auto e = environment; e && *e; ++e) {
    std::string setting(*e);
    if (setting.rfind("NCCL_", 0) == 0) records.push_back("env:" + setting);
  }
  auto file = [&](char const* role, std::filesystem::path const& path) {
    if (!std::filesystem::exists(path)) return;
    if (path == "/dev/null") return;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) > (1UL << 20))
      throw std::invalid_argument(
          "cannot fingerprint this NCCL configuration file");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read NCCL configuration file");
    std::string contents{std::istreambuf_iterator<char>(input), {}};
    records.push_back(std::string(role) + ':' + path.string() + ':' + contents);
  };
  if (auto overridePath = std::getenv("NCCL_CONF_FILE"))
    file("user-file", overridePath);
  else if (auto userHome = std::getenv("HOME"))
    file("user-file", std::filesystem::path(userHome) / ".nccl.conf");
  file("system-file", "/etc/nccl.conf");
  return records;
}
}  // namespace uccl_lite::benchmark

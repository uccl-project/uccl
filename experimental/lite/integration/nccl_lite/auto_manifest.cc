#include "auto_manifest.hpp"
#include <charconv>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
namespace uccl_lite::integration {
namespace {
void require(bool ok, char const* message) {
  if (!ok) throw std::invalid_argument(message);
}
template <class T>
bool number(std::istream& in, T& value) {
  std::string token;
  if (!(in >> token) || token.empty() || token.front() < '0' ||
      token.front() > '9')
    return false;
  auto parsed =
      std::from_chars(token.data(), token.data() + token.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size();
}
size_t scalar(ncclDataType_t type) { return type == ncclFloat32 ? 4 : 2; }
void layout(ucclLiteRankLayout l, size_t bytes) {
  require(l.inPlace <= 1 && l.alignment >= bytes && l.alignment <= 256 &&
              !(l.alignment & (l.alignment - 1)),
          "invalid automatic layout");
}
}  // namespace
AutoManifest loadAutoManifest(std::string const& file) {
  struct stat info {};
  require(!stat(file.c_str(), &info) && S_ISREG(info.st_mode) &&
              info.st_size > 0 && info.st_size <= (1 << 20),
          "automatic manifest must be a bounded regular file");
  AutoManifest result;
  result.contents.resize(size_t(info.st_size));
  std::ifstream input(file, std::ios::binary);
  input.read(result.contents.data(), result.contents.size());
  require(size_t(input.gcount()) == result.contents.size(),
          "automatic manifest changed during read");
  char extra;
  require(!input.get(extra) && input.eof(),
          "automatic manifest read extent changed");
  std::istringstream in(result.contents);
  std::string magic, key, type;
  size_t count;
  require(bool(in >> magic) && magic == "uccl-lite-auto-prepare-v1",
          "unknown automatic manifest format");
  require(bool(in >> key >> type) && key == "dtype", "missing automatic dtype");
  if (type == "float32")
    result.dtype = ncclFloat32;
  else if (type == "bfloat16")
    result.dtype = ncclBfloat16;
  else if (type == "float16")
    result.dtype = ncclFloat16;
  else
    throw std::invalid_argument("unsupported automatic dtype");
  require(bool(in >> key) && number(in, result.capacity) && key == "capacity" &&
              result.capacity > 0 && result.capacity <= (1UL << 30),
          "invalid automatic capacity");
  require(bool(in >> key) && number(in, result.flags) && key == "flags" &&
              !(result.flags & ~63u),
          "invalid automatic flags");
  require(bool(in >> key) && number(in, count) && key == "requests" &&
              count > 0 && count <= 1024,
          "invalid automatic request count");
  for (size_t i = 0; i < count; ++i) {
    AutoRequest r;
    std::string op, mode;
    require(bool(in >> op) && number(in, r.count) && bool(in >> mode) &&
                r.count > 0,
            "truncated automatic request");
    if (op == "allreduce")
      r.operation = ucclLiteAllReduce;
    else if (op == "allgather")
      r.operation = ucclLiteAllGather;
    else if (op == "reducescatter")
      r.operation = ucclLiteReduceScatter;
    else
      throw std::invalid_argument("unsupported automatic operation");
    if (mode == "out" || mode == "in") {
      ucclLiteRankLayout l{unsigned(mode == "in"), 0};
      require(number(in, l.alignment), "truncated automatic alignment");
      layout(l, scalar(result.dtype));
      r.layouts.push_back(l);
    } else if (mode == "ranks") {
      r.uniform = false;
      size_t ranks;
      require(number(in, ranks) && ranks > 0 && ranks <= 32,
              "invalid explicit layout extent");
      for (size_t rank = 0; rank < ranks; ++rank) {
        ucclLiteRankLayout l{};
        require(number(in, l.inPlace) && number(in, l.alignment),
                "truncated explicit layout");
        layout(l, scalar(result.dtype));
        r.layouts.push_back(l);
      }
    } else
      throw std::invalid_argument("unknown automatic layout mode");
    result.requests.push_back(std::move(r));
  }
  in >> std::ws;
  require(in.eof(), "trailing automatic manifest data");
  return result;
}
std::vector<AutoRequest> materializeAutoManifest(AutoManifest const& m,
                                                 int ranks) {
  require(ranks > 0 && ranks <= 32, "automatic rank extent unsupported");
  std::vector<AutoRequest> result;
  std::set<std::string> keys;
  size_t bytes = scalar(m.dtype);
  for (auto r : m.requests) {
    if (!r.uniform && r.layouts.size() != size_t(ranks)) continue;
    if (r.uniform) r.layouts.resize(ranks, r.layouts.front());
    size_t scale = bytes * (r.operation == ucclLiteAllReduce ? 1 : ranks);
    if (r.count > m.capacity / scale) continue;
    bool valid = true;
    std::ostringstream key;
    key << int(r.operation) << ':' << r.count;
    for (int rank = 0; rank < ranks; ++rank) {
      auto l = r.layouts[rank];
      valid &= !(l.inPlace && r.operation != ucclLiteAllReduce &&
                 (size_t(rank) * r.count * bytes) % l.alignment);
      key << ':' << l.inPlace << ':' << l.alignment;
    }
    if (!valid) continue;
    require(keys.insert(key.str()).second,
            "duplicate automatic request signature");
    r.uniform = false;
    result.push_back(std::move(r));
  }
  return result;
}
}  // namespace uccl_lite::integration

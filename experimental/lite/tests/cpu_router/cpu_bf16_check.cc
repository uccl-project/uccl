#include "cpu_executor.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace uccl_lite::cpu_router;
static float decode(uint16_t x) {
  uint32_t b = uint32_t(x) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
static uint16_t encode(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  if ((b & 0x7fffffffU) > 0x7f800000U)
    b |= 0x00400000U;
  else
    b += 0x7fffU + ((b >> 16) & 1U);
  return b >> 16;
}
int main() {
  size_t checked = 0;
  for (size_t sources : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(4),
                         size_t(8), size_t(32)}) {
    for (size_t n :
         {size_t(1), size_t(15), size_t(16), size_t(17), size_t(63), size_t(64),
          size_t(65), size_t(127), size_t(128), size_t(129), size_t(65539)}) {
      std::vector<std::vector<uint16_t>> inputs(
          sources, std::vector<uint16_t>(n + 2, 0x5678));
      std::vector<MemorySpan> spans(sources);
      std::vector<uint16_t> expected(n), out(n + 2, 0x1234);
      for (size_t s = 0; s < sources; ++s) {
        for (size_t i = 0; i < n; ++i) inputs[s][i + 1] = uint16_t(i + s * 193);
        spans[s] = {inputs[s].data() + 1, n * 2, MemoryKind::NumaHost};
      }
      for (size_t i = 0; i < n; ++i) {
        float v = 0;
        for (size_t s = 0; s < sources; ++s) v += decode(inputs[s][i + 1]);
        expected[i] = encode(v);
      }
      for (bool alias : {false, true}) {
        if (alias && !sources) continue;
        auto* dst = alias ? inputs[0].data() + 1 : out.data() + 1;
        CpuExecutor::sumBf16(spans.data(), sources,
                             {dst, n * 2, MemoryKind::NumaHost});
        for (size_t i = 0; i < n; ++i) {
          bool nan = std::isnan(decode(expected[i]));
          if (nan ? !std::isnan(decode(dst[i])) : dst[i] != expected[i])
            throw std::runtime_error("BF16 mismatch");
          ++checked;
        }
        auto const& guarded = alias ? inputs[0] : out;
        uint16_t guard = alias ? 0x5678 : 0x1234;
        if (guarded.front() != guard || guarded.back() != guard)
          throw std::runtime_error("BF16 output guard overwritten");
      }
    }
  }
  printf(
      "CPU BF16 scalar oracle: %zu checks passed (all encodings, tails, alias, "
      "0/1/2/3/4/8/32 inputs; 16/64-element boundaries and guards)\n",
      checked);
}

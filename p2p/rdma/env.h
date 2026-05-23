#pragma once

// Environment-variable driven runtime knobs for the p2p engine.
//
// Each setting follows the pattern:
//   - `get<Name>FromEnv()` parses the env var once and returns the value.
//   - `k<Name>` is a process-singleton cached reference initialized lazily on
//     first access. Use `k<Name>` everywhere; the `get…` form is only there
//     to express the parsing logic.
//
// Add new knobs here (not in define.h) to keep config localized.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// ── Compression knobs ──────────────────────────────────────────────────────

// Break-even between dietgpu compress/decompress kernel cost and saved bytes
// on the wire depends on NIC line rate and compressibility. On a ~50 GB/s
// link with FP32 ones-data (~25% savings) break-even is ~300 MB; with BF16
// (~50% savings) it's ~150 MB. Default to 16 MB to let BF16/FP16 data
// benefit from compression on moderately compressible payloads while keeping
// tiny-message regressions out.
//
// Override with UCCL_P2P_MIN_COMPRESS_BYTES (decimal bytes).
inline size_t getMinCompressBytesFromEnv() {
  char const* env = std::getenv("UCCL_P2P_MIN_COMPRESS_BYTES");
  if (!env) return 16ull * 1024 * 1024;
  char* end = nullptr;
  long long v = std::strtoll(env, &end, 10);
  if (end == env || v <= 0) return 16ull * 1024 * 1024;
  return static_cast<size_t>(v);
}
inline size_t const& kMinCompressBytes = []() -> size_t const& {
  static size_t v = getMinCompressBytesFromEnv();
  return v;
}();

// compress_buffer / decompress_buffer footprint. Each side allocates one
// such buffer in GPU memory. To support an N-byte compressed write we need
// kCompressBufferSize ≥ N × max_ratio (≈0.75 for FP32, 0.5 for BF16/FP16),
// so 2 GB by default supports 2 GB FP32 / 4 GB BF16 single in-flight, or
// proportionally smaller messages with more concurrent in-flight writes.
//
// Override with UCCL_P2P_COMPRESS_BUFFER_BYTES.
inline size_t getCompressBufferBytesFromEnv() {
  char const* env = std::getenv("UCCL_P2P_COMPRESS_BUFFER_BYTES");
  if (!env) return 2ull * 1024 * 1024 * 1024;
  char* end = nullptr;
  long long v = std::strtoll(env, &end, 10);
  if (end == env || v <= 0) return 2ull * 1024 * 1024 * 1024;
  return static_cast<size_t>(v);
}
inline size_t const& kCompressBufferSize = []() -> size_t const& {
  static size_t v = getCompressBufferBytesFromEnv();
  return v;
}();

// ── Compression strategy ───────────────────────────────────────────────────

enum class CompressStrategy {
  kNone,         // no compression
  kSplitOnly,    // only split, no encode
  kSplitEncode,  // split + encode (default)
};

inline CompressStrategy getCompressStrategyFromEnv() {
  char const* env = std::getenv("UCCL_P2P_COMPRESS_STRATEGY");

  // default strategy
  if (!env || env[0] == '\0') {
    return CompressStrategy::kNone;
  }

  std::string s(env);
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);

  // ---- accepted values ----
  if (s == "none" || s == "off" || s == "0") {
    return CompressStrategy::kNone;
  }

  if (s == "split" || s == "split_only") {
    return CompressStrategy::kSplitOnly;
  }

  if (s == "encode" || s == "split_encode" || s == "full" || s == "1") {
    return CompressStrategy::kSplitEncode;
  }

  // ---- fallback ----
  // unknown value -> default
  return CompressStrategy::kSplitEncode;
}

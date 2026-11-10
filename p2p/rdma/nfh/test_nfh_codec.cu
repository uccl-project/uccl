// Round-trip and throughput checks for the NFH codec.
//   ./test_nfh_codec                       correctness cases on synthetic data
//   ./test_nfh_codec --input-file w.bin    adds timing on a raw BF16 file
#include "nfh_codec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x)                                                     \
  do {                                                            \
    cudaError_t e_ = (x);                                         \
    if (e_ != cudaSuccess) {                                      \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e_));                       \
      std::exit(2);                                               \
    }                                                             \
  } while (0)
#define NK(x)                                                    \
  do {                                                           \
    int r_ = (x);                                                \
    if (r_ != NFH_OK) {                                          \
      std::fprintf(stderr, "NFH %s:%d %s\n", __FILE__, __LINE__, \
                   nfh_error_string(r_));                        \
      std::exit(3);                                              \
    }                                                            \
  } while (0)

// Setup copies and memsets below run on the legacy default stream, which does
// not order with the non-blocking streams the codec runs on (and a pageable
// cudaMemcpy to the device may return before its DMA lands). Wait for them.
static void setup_done() { CK(cudaDeviceSynchronize()); }

static uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

struct Ctx {
  nfh_encoder* enc = nullptr;
  nfh_decoder* dec = nullptr;
  cudaStream_t s = nullptr, s2 = nullptr;
  int* status = nullptr;  // mapped host memory
  int failures = 0;
};

struct Result {
  size_t bytes = 0;
  bool exact = false;
  int status = 0;
};

// in_off / out_off shift the device pointers by whole BF16 values (alignment
// cases).
static Result roundtrip(Ctx& c, char const* name,
                        std::vector<uint16_t> const& h, int in_off = 0,
                        int out_off = 0, int expect_stored = -1) {
  size_t n = h.size();
  uint16_t *din, *dout;
  uint8_t* dc;
  CK(cudaMalloc(&din, (n + 16) * 2));
  CK(cudaMalloc(&dout, (n + 16) * 2));
  CK(cudaMalloc(&dc, 2 * n + 64));
  CK(cudaMemcpy(din + in_off, h.data(), n * 2, cudaMemcpyHostToDevice));
  CK(cudaMemset(dout, 0xA5, (n + 16) * 2));
  setup_done();
  Result r;
  NK(nfh_split(c.enc, din + in_off, n, dc, c.s));
  NK(nfh_encode(c.enc, din + in_off, n, dc, c.s, &r.bytes));
  *c.status = 0;
  NK(nfh_decode(c.dec, dc, r.bytes, dout + out_off, n, c.s, c.status));
  CK(cudaStreamSynchronize(c.s));
  r.status = *c.status;
  std::vector<uint16_t> back(n);
  CK(cudaMemcpy(back.data(), dout + out_off, n * 2, cudaMemcpyDeviceToHost));
  r.exact = n == 0 || std::memcmp(back.data(), h.data(), n * 2) == 0;
  bool stored = r.bytes == 2 * n;
  bool ok = r.exact && r.status == 0 && r.bytes <= 2 * n &&
            (expect_stored < 0 || stored == (expect_stored == 1));
  std::printf(
      "{\"type\":\"case\",\"name\":\"%s\",\"n\":%zu,\"bytes\":%zu,\"bits_per_"
      "w\":%.4f,\"stored\":%d,"
      "\"exact\":%d,\"status\":%d,\"pass\":%d}\n",
      name, n, r.bytes, n ? r.bytes * 8.0 / n : 0.0, (int)stored, (int)r.exact,
      r.status, (int)ok);
  if (!ok) c.failures++;
  CK(cudaFree(din));
  CK(cudaFree(dout));
  CK(cudaFree(dc));
  return r;
}

static std::vector<uint16_t> gauss(size_t n, float sd, uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.f, sd);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = to_bf16(d(g));
  return v;
}

static void corrupt_case(Ctx& c) {
  size_t n = 16384 * 40 + 512;
  auto h = gauss(n, 0.02f, 7);
  uint16_t *din, *dout;
  uint8_t* dc;
  CK(cudaMalloc(&din, n * 2));
  CK(cudaMalloc(&dout, n * 2));
  CK(cudaMalloc(&dc, 2 * n));
  CK(cudaMemcpy(din, h.data(), n * 2, cudaMemcpyHostToDevice));
  setup_done();
  size_t bytes = 0;
  NK(nfh_split(c.enc, din, n, dc, c.s));
  NK(nfh_encode(c.enc, din, n, dc, c.s, &bytes));
  int st[3];
  // (a) truncated length, (b) wrong magic, (c) a code length out of range
  *c.status = 0;
  NK(nfh_decode(c.dec, dc, bytes - 4, dout, n, c.s, c.status));
  CK(cudaStreamSynchronize(c.s));
  st[0] = *c.status;
  size_t H = (n + 15) & ~size_t(15);
  uint32_t bad = 0xdeadbeef, good;
  CK(cudaMemcpy(&good, dc + H, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(dc + H, &bad, 4, cudaMemcpyHostToDevice));
  setup_done();
  *c.status = 0;
  NK(nfh_decode(c.dec, dc, bytes, dout, n, c.s, c.status));
  CK(cudaStreamSynchronize(c.s));
  st[1] = *c.status;
  CK(cudaMemcpy(dc + H, &good, 4, cudaMemcpyHostToDevice));
  uint8_t l0;
  CK(cudaMemcpy(&l0, dc + H + 32, 1, cudaMemcpyDeviceToHost));
  uint8_t bl = 15;
  CK(cudaMemcpy(dc + H + 32, &bl, 1, cudaMemcpyHostToDevice));
  setup_done();
  *c.status = 0;
  NK(nfh_decode(c.dec, dc, bytes, dout, n, c.s, c.status));
  CK(cudaStreamSynchronize(c.s));
  st[2] = *c.status;
  CK(cudaMemcpy(dc + H + 32, &l0, 1, cudaMemcpyHostToDevice));
  setup_done();
  // A clean decode after the failures must still be exact (scheduler state
  // intact).
  *c.status = 0;
  NK(nfh_decode(c.dec, dc, bytes, dout, n, c.s, c.status));
  CK(cudaStreamSynchronize(c.s));
  std::vector<uint16_t> back(n);
  CK(cudaMemcpy(back.data(), dout, n * 2, cudaMemcpyDeviceToHost));
  bool exact = std::memcmp(back.data(), h.data(), n * 2) == 0;
  bool ok = st[0] == NFH_ERR_FORMAT && st[1] == NFH_ERR_FORMAT &&
            st[2] == NFH_ERR_FORMAT && exact && *c.status == 0;
  std::printf(
      "{\"type\":\"case\",\"name\":\"corrupt\",\"status\":[%d,%d,%d],\"clean_"
      "after\":%d,\"pass\":%d}\n",
      st[0], st[1], st[2], (int)exact, (int)ok);
  if (!ok) c.failures++;
  CK(cudaFree(din));
  CK(cudaFree(dout));
  CK(cudaFree(dc));
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static void timing(Ctx& c, std::vector<uint16_t> const& all, size_t n,
                   int iters, char const* tag, int chunk = 0) {
  nfh_set_chunk_override(chunk);
  uint16_t *din, *dout;
  uint8_t* dc;
  void* flush;
  size_t fb = 256ull << 20;
  CK(cudaMalloc(&din, n * 2));
  CK(cudaMalloc(&dout, n * 2));
  CK(cudaMalloc(&dc, 2 * n));
  CK(cudaMalloc(&flush, fb));
  CK(cudaMemcpy(din, all.data(), n * 2, cudaMemcpyHostToDevice));
  setup_done();
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  std::vector<double> te, ts, td, tm;
  size_t bytes = 0;
  for (int i = 0; i < iters + 2; ++i) {
    CK(cudaMemsetAsync(flush, i, fb, c.s));
    CK(cudaStreamSynchronize(c.s));
    auto t0 = std::chrono::steady_clock::now();
    // as the P2P backend does: split on the high-priority c.s (it gates the
    // first post), the rest of the encoder concurrently on c.s2
    NK(nfh_split(c.enc, din, n, dc, c.s));
    NK(nfh_encode_begin(c.enc, din, n, dc, c.s2));
    CK(cudaStreamSynchronize(c.s));
    auto ta = std::chrono::steady_clock::now();
    NK(nfh_encode_end(c.enc, c.s2, &bytes));
    auto t1 = std::chrono::steady_clock::now();
    if (i >= 2) {
      te.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      ts.push_back(std::chrono::duration<double, std::micro>(ta - t0).count());
    }
  }
  *c.status = 0;
  for (int i = 0; i < iters + 2; ++i) {
    CK(cudaMemsetAsync(flush, i, fb, c.s));
    CK(cudaEventRecord(a, c.s));
    NK(nfh_decode(c.dec, dc, bytes, dout, n, c.s, c.status));
    CK(cudaEventRecord(b, c.s));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    if (i >= 2) td.push_back(ms * 1e3);
  }
  for (int i = 0; i < iters + 2; ++i) {
    CK(cudaMemsetAsync(flush, i, fb, c.s));
    CK(cudaEventRecord(a, c.s));
    CK(cudaMemcpyAsync(dout, din, n * 2, cudaMemcpyDeviceToDevice, c.s));
    CK(cudaEventRecord(b, c.s));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    if (i >= 2) tm.push_back(ms * 1e3);
  }
  NK(nfh_decode(c.dec, dc, bytes, dout, n, c.s, c.status));
  CK(cudaStreamSynchronize(c.s));
  std::vector<uint16_t> back(n);
  CK(cudaMemcpy(back.data(), dout, n * 2, cudaMemcpyDeviceToHost));
  bool exact =
      std::memcmp(back.data(), all.data(), n * 2) == 0 && *c.status == 0;
  double e = median(te), sp = median(ts), d = median(td), m = median(tm);
  nfh_set_chunk_override(0);
  std::printf(
      "{\"type\":\"time\",\"tag\":\"%s\",\"chunk\":%d,\"n\":%zu,\"MB\":%.1f,"
      "\"bits_per_w\":%.4f,\"split_us\":%.1f,\"enc_us\":%.1f,"
      "\"enc_GBps_in\":%.1f,\"dec_us\":%.1f,\"dec_GBps_out\":%.1f,\"memcpy_"
      "us\":%.1f,\"exact\":%d}\n",
      tag, chunk, n, n * 2 / 1e6, bytes * 8.0 / n, sp, e, n * 2 / e / 1e3, d,
      n * 2 / d / 1e3, m, (int)exact);
  if (!exact) c.failures++;
  CK(cudaFree(din));
  CK(cudaFree(dout));
  CK(cudaFree(dc));
  CK(cudaFree(flush));
}

int main(int argc, char** argv) {
  std::string path;
  int iters = 20;
  bool sweep = false;
  int only_mb = 0;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--input-file" && i + 1 < argc)
      path = argv[++i];
    else if (s == "--iters" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (s == "--chunk-sweep")
      sweep = true;
    else if (s == "--only-mb" && i + 1 < argc)
      only_mb = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: %s [--input-file f] [--iters k]\n", argv[0]);
      return 1;
    }
  }
  Ctx c;
  NK(nfh_encoder_create(&c.enc));
  NK(nfh_decoder_create(&c.dec));
  int prio_lo = 0, prio_hi = 0;
  CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
  CK(cudaStreamCreateWithPriority(&c.s, cudaStreamNonBlocking, prio_hi));
  CK(cudaStreamCreateWithFlags(&c.s2, cudaStreamNonBlocking));
  CK(cudaHostAlloc(&c.status, sizeof(int), cudaHostAllocMapped));

  if (only_mb) {  // profiling: one timing case only
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    std::vector<uint16_t> all(raw.size() / 2);
    std::memcpy(all.data(), raw.data(), all.size() * 2);
    size_t n = std::min(all.size(), (size_t)only_mb * 1024 * 1024 / 2);
    timing(c, all, n, iters, "only");
    return c.failures ? 1 : 0;
  }
  for (int ck :
       {2048, 4096,
        8192}) {  // every chunk size round-trips, incl. tails and segments
    nfh_set_chunk_override(ck);
    std::string nm = "gauss_chunk" + std::to_string(ck);
    roundtrip(c, nm.c_str(),
              gauss(16384 * 4096 + 16384 * 5 + 1000, 0.02f, 20 + ck), 0, 0, 0);
  }
  nfh_set_chunk_override(0);
  roundtrip(c, "gauss_tail", gauss(8u * 1024 * 1024 + 12345, 0.02f, 1), 0, 0,
            0);
  roundtrip(c, "gauss_exact_chunks", gauss(16384 * 64, 0.02f, 2), 0, 0, 0);
  roundtrip(c, "gauss_multi_segment",
            gauss(16384 * 4096 * 2 + 16384 * 3 + 40, 0.02f, 3), 0, 0, 0);
  {
    std::mt19937 g(4);
    std::vector<uint16_t> v(4u << 20);
    for (auto& x : v) x = (uint16_t)g();
    roundtrip(c, "random_bits", v, 0, 0, 1);
  }
  {
    // Values the sampled histogram never sees: every 32768-value block has its
    // last 1/16 scaled far outside the window; the sample reads the first 2048.
    auto v = gauss(16384 * 512, 0.02f, 5);
    for (size_t i = 0; i < v.size(); ++i)
      if ((i & 32767) >= 32768 - 2048) {
        uint32_t e = (v[i] >> 7) & 0xFF;
        if (e > 0 && e < 200)
          v[i] = (uint16_t)((v[i] & 0x807F) | ((e + 40) << 7));
      }
    roundtrip(c, "unsampled_band", v, 0, 0, 0);
  }
  for (size_t n : {1, 7, 8, 100, 16383, 16384, 16385, 16384 * 2 + 8}) {
    std::string nm = "small_" + std::to_string(n);
    roundtrip(c, nm.c_str(), gauss(n, 0.02f, (uint32_t)n));
  }
  roundtrip(c, "misaligned_in", gauss(16384 * 20, 0.02f, 8), 1, 0, 1);
  roundtrip(c, "misaligned_out", gauss(16384 * 20 + 24, 0.02f, 9), 0, 1, 0);
  roundtrip(c, "zeros", std::vector<uint16_t>(1u << 20, 0), 0, 0, 0);
  {
    auto v = gauss(16384 * 64, 0.02f, 10);
    for (size_t i = 0; i < v.size(); i += 97)
      v[i] = (i & 1) ? 0x7F80 : 0xFFC1;  // inf / nan bit patterns
    roundtrip(c, "specials", v, 0, 0, 0);
  }
  corrupt_case(c);

  if (!path.empty()) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    std::vector<uint16_t> all(raw.size() / 2);
    std::memcpy(all.data(), raw.data(), all.size() * 2);
    for (size_t mb : {16, 64, 256}) {
      size_t n = mb * 1024 * 1024 / 2;
      if (n > all.size()) continue;
      std::string tg = "file_" + std::to_string(mb) + "MiB";
      timing(c, all, n, iters, tg.c_str());
      if (sweep)
        for (int ck : {2048, 4096, 8192, 16384})
          timing(c, all, n, iters, tg.c_str(), ck);
    }
    timing(c, all, all.size(), iters, "file_full");
  }
  std::printf("{\"type\":\"summary\",\"failures\":%d}\n", c.failures);
  nfh_encoder_destroy(c.enc);
  nfh_decoder_destroy(c.dec);
  return c.failures ? 1 : 0;
}

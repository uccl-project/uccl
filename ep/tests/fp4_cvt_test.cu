// Checks the inline FP4 (E2M1) helpers in ep/include/fp4_utils.cuh.
//
//   make fp4_cvt_test                               # sm_120a: hardware cvt +
//   soft make fp4_cvt_test FP4_CUDAFLAGS="-gencode
//   arch=compute_120,code=sm_120"  # soft
//   ./fp4_cvt_test
//
// 1. Encoder: the soft encoder must match an independent host reference
//    (nearest grid point, ties to the even code, saturate to +-6) and, when
//    the hardware `cvt.rn.satfinite.e2m1x2.f32` is compiled in, the hardware
//    result bit for bit.  NaN has no E2M1 encoding and is reported apart.
// 2. Block quantizers: the device NVFP4/MXFP4 scale and data bytes must match
//    a host reference that follows the same FP32 operation order.
// 3. Reports the dequantization error on Gaussian rows (lossy by design).
#include "fp4_utils.cuh"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <cuda_bf16.h>

#define CHECK(call)                                     \
  do {                                                  \
    cudaError_t e = (call);                             \
    if (e != cudaSuccess) {                             \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                   \
      exit(1);                                          \
    }                                                   \
  } while (0)

using namespace uccl::fp4;

__global__ void pack_kernel(float const* in, uint32_t* hw, uint32_t* soft,
                            int* hw_compiled, int n8) {
  int const t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t == 0) {
#ifdef UCCL_EP_FP4_HW_CVT
    *hw_compiled = 1;
#else
    *hw_compiled = 0;
#endif
  }
  if (t >= n8) return;
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = in[t * 8 + j];
  hw[t] = pack8_e2m1(v);
  soft[t] = pack8_e2m1_soft(v);
}

// One thread per block of `kB` elements; same FP32 op order as dispatch.
template <int kMode>
__global__ void quant_kernel(nv_bfloat16 const* x, int hidden, int rows,
                             uint8_t* data, uint8_t* scales, float* gs_out) {
  constexpr int kB = Traits<kMode>::kBlock;
  int const r = blockIdx.x;
  if (r >= rows) return;
  nv_bfloat16 const* xr = x + static_cast<size_t>(r) * hidden;
  float gs = 1.0f;
  if constexpr (kMode == kNVFP4) {
    __shared__ float red[1024];
    float a = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
      a = fmaxf(a, fabsf(__bfloat162float(xr[i])));
    red[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
      float m = 0.0f;
      for (int i = 0; i < blockDim.x; ++i) m = fmaxf(m, red[i]);
      red[0] = m;
    }
    __syncthreads();
    gs = nv_token_scale(red[0]);
    if (threadIdx.x == 0) gs_out[r] = gs;
  }
  for (int b = threadIdx.x; b < hidden / kB; b += blockDim.x) {
    float f[kB];
    float amax = 0.0f;
    for (int j = 0; j < kB; ++j) {
      f[j] = __bfloat162float(xr[b * kB + j]);
      amax = fmaxf(amax, fabsf(f[j]));
    }
    float inv;
    uint8_t s = kMode == kNVFP4 ? nv_block_scale(amax, gs, inv)
                                : mx_block_scale(amax, inv);
    scales[static_cast<size_t>(r) * (hidden / kB) + b] = s;
    for (int k = 0; k < kB; k += 8) {
      float v[8];
      for (int j = 0; j < 8; ++j) v[j] = f[k + j] * inv;
      uint32_t w = pack8_e2m1(v);
      memcpy(data + static_cast<size_t>(r) * hidden / 2 + (b * kB + k) / 2, &w,
             4);
    }
  }
}

// ---------------- host reference ----------------
static float const kGrid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static uint32_t ref_e2m1(float x) {
  uint32_t sign;
  memcpy(&sign, &x, 4);
  sign = (sign >> 31) << 3;
  double a = std::fabs(static_cast<double>(x));
  if (a >= 6.0) return sign | 7u;
  int best = 0;
  for (int c = 1; c < 8; ++c) {
    double dc = std::fabs(a - kGrid[c]), db = std::fabs(a - kGrid[best]);
    if (dc < db or (dc == db and (c % 2 == 0))) best = c;
  }
  return sign | static_cast<uint32_t>(best);
}
static uint32_t ref_pack8(float const* v) {
  uint32_t r = 0;
  for (int j = 0; j < 8; ++j) r |= ref_e2m1(v[j]) << (4 * j);
  return r;
}
static float ref_decode(uint32_t nib) {
  float m = kGrid[nib & 7u];
  return (nib & 8u) ? -m : m;
}
static float h_e4m3(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
  return __half2float(*reinterpret_cast<__half*>(&h));
}
static int ref_mx_exp(float amax) {
  if (!(amax > 0.0f)) return -127;
  int e = -127;
  while (e < 127 and std::ldexp(6.0, e) < static_cast<double>(amax)) ++e;
  return e;
}

static bool is_nan_word(float const* v) {
  for (int j = 0; j < 8; ++j)
    if (std::isnan(v[j])) return true;
  return false;
}

int main() {
  // ---- 1. encoder ----
  std::vector<float> vals;
  float const ties[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
  for (float g : kGrid) vals.push_back(g);
  for (float t : ties) {
    vals.push_back(t);
    vals.push_back(std::nextafter(t, 0.0f));
    vals.push_back(std::nextafter(t, 100.0f));
  }
  float const special[] = {
      0.0f,   6.0f,   std::nextafter(6.0f, 100.0f),      7.0f, 1e30f, INFINITY,
      1e-40f, 1e-45f, std::numeric_limits<float>::min(), 0.1f, 0.3f,  5.999f};
  for (float s : special) vals.push_back(s);
  size_t const base = vals.size();
  for (size_t i = 0; i < base; ++i) vals.push_back(-vals[i]);  // incl. -0
  std::mt19937 rng(20261125);
  std::uniform_real_distribution<float> u(-8.0f, 8.0f);
  std::uniform_real_distribution<float> lg(-30.0f, 10.0f);
  for (int i = 0; i < (1 << 20); ++i) vals.push_back(u(rng));
  for (int i = 0; i < (1 << 18); ++i)
    vals.push_back((i & 1 ? -1.f : 1.f) * std::exp2(lg(rng)));
  // Asymmetric neighbours pin the nibble order: rotate the special list.
  // NaN words are checked hw-vs-soft only; E2M1 has no NaN.
  std::vector<float> nan_vals = {NAN, 1.0f, -NAN, 3.0f, 0.5f, NAN, 6.0f, -2.0f};
  size_t const n_nan_start = vals.size();
  for (float v : nan_vals) vals.push_back(v);
  while (vals.size() % 8) vals.push_back(0.5f);
  int const n8 = static_cast<int>(vals.size() / 8);

  float* d_in;
  uint32_t *d_hw, *d_soft;
  int* d_hwc;
  CHECK(cudaMalloc(&d_in, vals.size() * 4));
  CHECK(cudaMalloc(&d_hw, n8 * 4));
  CHECK(cudaMalloc(&d_soft, n8 * 4));
  CHECK(cudaMalloc(&d_hwc, 4));
  CHECK(cudaMemcpy(d_in, vals.data(), vals.size() * 4, cudaMemcpyHostToDevice));
  pack_kernel<<<(n8 + 255) / 256, 256>>>(d_in, d_hw, d_soft, d_hwc, n8);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> hw(n8), soft(n8);
  int hwc = 0;
  CHECK(cudaMemcpy(hw.data(), d_hw, n8 * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(soft.data(), d_soft, n8 * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&hwc, d_hwc, 4, cudaMemcpyDeviceToHost));

  long bad_ref = 0, bad_hw = 0, words = 0, nan_words = 0, nan_diff = 0;
  for (int t = 0; t < n8; ++t) {
    float const* v = vals.data() + t * 8;
    if (is_nan_word(v)) {
      ++nan_words;
      if (hwc and hw[t] != soft[t]) {
        ++nan_diff;
        printf("NaN word %d: hw %08x soft %08x\n", t, hw[t], soft[t]);
      }
      continue;
    }
    ++words;
    uint32_t const r = ref_pack8(v);
    if (soft[t] != r) {
      if (bad_ref < 10)
        printf("SOFT!=REF word %d v0=%a soft %08x ref %08x\n", t, v[0], soft[t],
               r);
      ++bad_ref;
    }
    if (hwc and hw[t] != soft[t]) {
      if (bad_hw < 10)
        printf("HW!=SOFT word %d v0=%a hw %08x soft %08x\n", t, v[0], hw[t],
               soft[t]);
      ++bad_hw;
    }
  }
  (void)n_nan_start;
  printf(
      "ENCODER hw_cvt_compiled=%d words=%ld soft_vs_ref_mismatch=%ld "
      "hw_vs_soft_mismatch=%s nan_words=%ld nan_hw_vs_soft_diff=%s\n",
      hwc, words, bad_ref, hwc ? std::to_string(bad_hw).c_str() : "n/a",
      nan_words, hwc ? std::to_string(nan_diff).c_str() : "n/a");

  // ---- 2./3. block quantizers ----
  int const hidden = 7168, rows = 256;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<nv_bfloat16> hx(static_cast<size_t>(hidden) * rows);
  std::vector<float> fx(hx.size());
  for (size_t i = 0; i < hx.size(); ++i) {
    // Per-row magnitude spread plus a few outliers.
    float s = std::exp2(static_cast<float>((i / hidden) % 24) - 12.0f);
    float v = nd(rng) * s * ((i % 997) == 0 ? 20.0f : 1.0f);
    hx[i] = __float2bfloat16(v);
    fx[i] = __bfloat162float(hx[i]);
  }
  nv_bfloat16* d_x;
  CHECK(cudaMalloc(&d_x, hx.size() * 2));
  CHECK(cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  uint8_t *d_data, *d_sc;
  float* d_gs;
  CHECK(cudaMalloc(&d_data, hx.size() / 2));
  CHECK(cudaMalloc(&d_sc, hx.size() / 16));
  CHECK(cudaMalloc(&d_gs, rows * 4));

  bool ok = (bad_ref == 0) and (bad_hw == 0);
  for (int mode : {1, 2}) {
    int const B = mode == 1 ? 16 : 32, nb = hidden / B;
    if (mode == 1)
      quant_kernel<kNVFP4>
          <<<rows, 256>>>(d_x, hidden, rows, d_data, d_sc, d_gs);
    else
      quant_kernel<kMXFP4>
          <<<rows, 256>>>(d_x, hidden, rows, d_data, d_sc, d_gs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> data(hx.size() / 2),
        sc(static_cast<size_t>(rows) * nb);
    std::vector<float> gs(rows, 1.0f);
    CHECK(cudaMemcpy(data.data(), d_data, data.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sc.data(), d_sc, sc.size(), cudaMemcpyDeviceToHost));
    if (mode == 1)
      CHECK(cudaMemcpy(gs.data(), d_gs, rows * 4, cudaMemcpyDeviceToHost));

    long bad_s = 0, bad_d = 0, bad_g = 0;
    double err2 = 0.0, ref2 = 0.0;
    for (int r = 0; r < rows; ++r) {
      float const* xr = fx.data() + static_cast<size_t>(r) * hidden;
      float g = 1.0f;
      if (mode == 1) {
        float tamax = 0.0f;
        for (int i = 0; i < hidden; ++i)
          tamax = std::fmax(tamax, std::fabs(xr[i]));
        g = tamax > 0.0f ? tamax / (6.0f * 448.0f) : 1.0f;
        if (std::memcmp(&g, &gs[r], 4) != 0) ++bad_g;
      }
      for (int b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < B; ++j)
          amax = std::fmax(amax, std::fabs(xr[b * B + j]));
        uint8_t s;
        float inv, dec;
        if (mode == 1) {
          float const q = amax / (6.0f * g);
          s = static_cast<uint8_t>(
              __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3));
          float const eff = h_e4m3(s) * g;
          inv = eff > 0.0f ? 1.0f / eff : 0.0f;
          dec = eff;
        } else {
          int const e = ref_mx_exp(amax);
          s = static_cast<uint8_t>(e + 127);
          inv = std::ldexp(1.0f, -e);
          dec = std::ldexp(1.0f, e);
        }
        if (sc[static_cast<size_t>(r) * nb + b] != s) ++bad_s;
        for (int k = 0; k < B; k += 8) {
          float v[8];
          for (int j = 0; j < 8; ++j) v[j] = xr[b * B + k + j] * inv;
          uint32_t const w = ref_pack8(v);
          uint32_t got;
          memcpy(&got,
                 data.data() + static_cast<size_t>(r) * hidden / 2 +
                     (b * B + k) / 2,
                 4);
          if (got != w) ++bad_d;
          for (int j = 0; j < 8; ++j) {
            // Decode what the device produced (device-side decode is the
            // receiver's job; use the host table here).
            float const y = ref_decode(got >> (4 * j)) * dec;
            double const d = static_cast<double>(y) - xr[b * B + k + j];
            err2 += d * d;
            ref2 += static_cast<double>(xr[b * B + k + j]) * xr[b * B + k + j];
          }
        }
      }
    }
    printf(
        "%s rows=%d hidden=%d scale_mismatch=%ld data_word_mismatch=%ld "
        "token_scale_mismatch=%ld rel_rms_err=%.4f\n",
        mode == 1 ? "NVFP4" : "MXFP4", rows, hidden, bad_s, bad_d, bad_g,
        std::sqrt(err2 / ref2));
    ok = ok and bad_s == 0 and bad_d == 0 and bad_g == 0;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

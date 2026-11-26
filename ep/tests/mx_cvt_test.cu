// Checks ep/include/mx_cvt.cuh against CUDA's own conversions
// (__nv_cvt_float_to_e8m0 / _fp8 / _fp4, evaluated on the host).
//
//   make mx_cvt_test                                    # sm_120a: hardware
//   path make mx_cvt_test FP4_CUDAFLAGS="-gencode arch=compute_120,code=sm_120"
//   make mx_cvt_test FP4_CUDAFLAGS="-gencode arch=compute_120a,code=sm_120a
//   -DUCCL_EP_FP4_SOFT_CVT"
//   ./mx_cvt_test
//
// For every input pair it checks the compiled path (hardware or software,
// whichever the target selects) and the software encoders called directly.
// Mismatches are split into negative / subnormal / other inputs.  NaN
// inputs are reported apart (no E2M1 encoding).  Then the UE8M0 decoders are
// checked on all 256 codes.
#include "mx_cvt.cuh"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#define CHECK(call)                                     \
  do {                                                  \
    cudaError_t e = (call);                             \
    if (e != cudaSuccess) {                             \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                   \
      exit(1);                                          \
    }                                                   \
  } while (0)

using namespace uccl::mx;

enum { kRp, kRz, kE4, kE2, kRpSoft, kRzSoft, kE2Soft, kOut };

__global__ void flags_kernel(int* f) {
#ifdef UCCL_EP_FP4_HW_CVT
  f[0] = 1;
#else
  f[0] = 0;
#endif
#ifdef UCCL_EP_E4M3_HW_CVT
  f[1] = 1;
#else
  f[1] = 0;
#endif
#ifdef __CUDA_ARCH__
  f[2] = __CUDA_ARCH__;
#endif
#ifdef __CUDA_ARCH_FAMILY_SPECIFIC__
  f[3] = __CUDA_ARCH_FAMILY_SPECIFIC__;
#else
  f[3] = 0;
#endif
#ifdef __CUDA_ARCH_FEAT_SM120_ALL
  f[4] = 1;
#else
  f[4] = 0;
#endif
}

__global__ void pair_kernel(float const* x, int np, uint32_t* out) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= np) return;
  float const lo = x[2 * i], hi = x[2 * i + 1];
  uint32_t* o = out + static_cast<size_t>(i) * kOut;
  o[kRp] = ue8m0x2<ScaleRound::kUp>(lo, hi);
  o[kRz] = ue8m0x2<ScaleRound::kZero>(lo, hi);
  o[kE4] = e4m3x2_rn(lo, hi);
  o[kE2] = e2m1x2_rn(lo, hi);
  o[kRpSoft] =
      ue8m0_soft<ScaleRound::kUp>(lo) | (ue8m0_soft<ScaleRound::kUp>(hi) << 8);
  o[kRzSoft] = ue8m0_soft<ScaleRound::kZero>(lo) |
               (ue8m0_soft<ScaleRound::kZero>(hi) << 8);
  o[kE2Soft] = e2m1_soft(lo) | (e2m1_soft(hi) << 4);
}

__global__ void x8_kernel(float const* x, int n8, uint32_t* out) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n8) return;
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = x[8 * i + j];
  out[2 * i] = e2m1x8_rn(v);
  out[2 * i + 1] = e4m3x4_rn(v[0], v[1], v[2], v[3]);
}

__global__ void decode_kernel(float* dec, float* inv) {
  int const c = threadIdx.x;
  dec[c] = ue8m0_to_f32(c);
  inv[c] = ue8m0_to_inv_f32(c);
}

static uint32_t rng = 12345u;
static uint32_t xr() {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}
static float bf(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint32_t fb(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
static bool is_sub(float f) {
  uint32_t const u = fb(f) & 0x7fffffffu;
  return u != 0u && u < 0x00800000u;
}
static bool is_neg(float f) { return fb(f) >> 31; }

static uint32_t ref_e8(float v, cudaRoundMode r) {
  return static_cast<uint32_t>(__nv_cvt_float_to_e8m0(v, __NV_SATFINITE, r));
}
static uint32_t ref_e4(float v) {
  return static_cast<uint32_t>(
      __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}
static uint32_t ref_e2(float v) {
  return static_cast<uint32_t>(
             __nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest)) &
         0xfu;
}

int main() {
  int flags[5] = {0, 0, 0, 0, 0};
  {
    int* d;
    CHECK(cudaMalloc(&d, sizeof(flags)));
    flags_kernel<<<1, 1>>>(d);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(flags, d, sizeof(flags), cudaMemcpyDeviceToHost));
    cudaFree(d);
  }
  printf(
      "FLAGS fp4_ue8m0_hw=%d e4m3_hw=%d cuda_arch=%d family_specific=%d "
      "feat_sm120_all=%d\n",
      flags[0], flags[1], flags[2], flags[3], flags[4]);

  // Inputs: specials, random bit patterns (no NaN), small decimal values.
  std::vector<float> h;
  float const sp[] = {0.f,
                      -0.f,
                      1.f,
                      -1.f,
                      0.5f,
                      1.5f,
                      3.f,
                      6.f,
                      7.f,
                      448.f,
                      449.f,
                      464.f,
                      1e30f,
                      -1e30f,
                      bf(0x7f800000),
                      bf(0xff800000),
                      bf(0x00000001),
                      bf(0x007fffff),
                      bf(0x00400000),
                      bf(0x00400001),
                      bf(0x003fffff),
                      bf(0x00800000),
                      bf(0x7f7fffff),
                      bf(0x7f000000),
                      bf(0x7f000001),
                      bf(0x3f800001),
                      bf(0x3fffffff),
                      0.25f,
                      0.75f,
                      1.25f,
                      1.75f,
                      2.5f,
                      3.5f,
                      5.f,
                      4.5f};
  for (float f : sp) h.push_back(f);
  while (h.size() < (1u << 22)) {
    uint32_t const u = xr();
    if (((u >> 23) & 0xff) == 0xff && (u & 0x7fffff)) continue;
    h.push_back(bf(u));
  }
  for (int i = 0; i < (1 << 20); ++i) {
    float const m = static_cast<float>(xr() % 2000001) / 1000.f - 1000.f;
    h.push_back(m / static_cast<float>(1 + xr() % 64));
  }
  // Subnormal sweep around 2^-127 (UE8M0 code 0).
  for (uint32_t u = 0x003ff000u; u < 0x00401000u; ++u) h.push_back(bf(u));
  size_t const n_finite = h.size();
  float const nan_in[] = {
      bf(0x7fc00000), 1.f, bf(0xffc00000), 3.f, bf(0x7f800001),
      0.5f,           6.f, bf(0xff800001)};
  for (float f : nan_in) h.push_back(f);
  while (h.size() % 8) h.push_back(0.5f);
  int const n = static_cast<int>(h.size()), np = n / 2, n8 = n / 8;

  float* dx;
  uint32_t *dout, *d8;
  CHECK(cudaMalloc(&dx, static_cast<size_t>(n) * 4));
  CHECK(cudaMalloc(&dout, static_cast<size_t>(np) * kOut * 4));
  CHECK(cudaMalloc(&d8, static_cast<size_t>(n8) * 8));
  CHECK(cudaMemcpy(dx, h.data(), static_cast<size_t>(n) * 4,
                   cudaMemcpyHostToDevice));
  pair_kernel<<<(np + 255) / 256, 256>>>(dx, np, dout);
  x8_kernel<<<(n8 + 255) / 256, 256>>>(dx, n8, d8);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> out(static_cast<size_t>(np) * kOut),
      o8(static_cast<size_t>(n8) * 2);
  CHECK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(o8.data(), d8, o8.size() * 4, cudaMemcpyDeviceToHost));

  char const* nm[kOut] = {"ue8m0x2_rp", "ue8m0x2_rz",    "e4m3x2_rn",
                          "e2m1x2_rn",  "ue8m0_soft_rp", "ue8m0_soft_rz",
                          "e2m1_soft"};
  // bad[t][0] negative, [1] subnormal, [2] other.
  long bad[kOut][3] = {}, checked = 0;
  int shown[kOut] = {};
  bool ok = true;
  for (int i = 0; i < np; ++i) {
    float const lo = h[2 * i], hi = h[2 * i + 1];
    if (static_cast<size_t>(2 * i + 1) >= n_finite) {
      uint32_t const* o = &out[static_cast<size_t>(i) * kOut];
      printf(
          "NAN_PAIR lo=%08x hi=%08x rp=%04x rz=%04x e4=%04x e2=%02x "
          "rp_soft=%04x rz_soft=%04x e2_soft=%02x\n",
          fb(lo), fb(hi), o[kRp], o[kRz], o[kE4], o[kE2], o[kRpSoft],
          o[kRzSoft], o[kE2Soft]);
      continue;
    }
    ++checked;
    uint32_t const rp =
        ref_e8(lo, cudaRoundPosInf) | (ref_e8(hi, cudaRoundPosInf) << 8);
    uint32_t const rz =
        ref_e8(lo, cudaRoundZero) | (ref_e8(hi, cudaRoundZero) << 8);
    uint32_t const e4 = ref_e4(lo) | (ref_e4(hi) << 8);
    uint32_t const e2 = ref_e2(lo) | (ref_e2(hi) << 4);
    uint32_t const ref[kOut] = {rp, rz, e4, e2, rp, rz, e2};
    int const cat =
        (is_neg(lo) || is_neg(hi)) ? 0 : ((is_sub(lo) || is_sub(hi)) ? 1 : 2);
    for (int t = 0; t < kOut; ++t) {
      uint32_t const got = out[static_cast<size_t>(i) * kOut + t];
      if (got != ref[t]) {
        ++bad[t][cat];
        if (shown[t]++ < 6)
          printf("MISMATCH %s lo=%a hi=%a got=%x ref=%x\n", nm[t], lo, hi, got,
                 ref[t]);
      }
    }
  }
  for (int t = 0; t < kOut; ++t) {
    long const tot = bad[t][0] + bad[t][1] + bad[t][2];
    printf("CVT %s pairs=%ld mismatch=%ld (neg=%ld sub=%ld other=%ld) %s\n",
           nm[t], checked, tot, bad[t][0], bad[t][1], bad[t][2],
           tot ? "FAIL" : "PASS");
    ok = ok && tot == 0;
  }

  long bad8 = 0, bad4 = 0;
  for (int i = 0; i < n8; ++i) {
    if (static_cast<size_t>(8 * i + 7) >= n_finite) continue;
    float const* v = &h[static_cast<size_t>(8) * i];
    uint32_t w = 0;
    for (int j = 0; j < 8; ++j) w |= ref_e2(v[j]) << (4 * j);
    uint32_t q = 0;
    for (int j = 0; j < 4; ++j) q |= ref_e4(v[j]) << (8 * j);
    bad8 += o8[2 * i] != w;
    bad4 += o8[2 * i + 1] != q;
  }
  printf("CVT e2m1x8_rn words=%d mismatch=%ld %s\n", n8, bad8,
         bad8 ? "FAIL" : "PASS");
  printf("CVT e4m3x4_rn words=%d mismatch=%ld %s\n", n8, bad4,
         bad4 ? "FAIL" : "PASS");
  ok = ok && bad8 == 0 && bad4 == 0;

  // Decoders on all 256 codes.
  float *ddec, *dinv;
  CHECK(cudaMalloc(&ddec, 256 * 4));
  CHECK(cudaMalloc(&dinv, 256 * 4));
  decode_kernel<<<1, 256>>>(ddec, dinv);
  CHECK(cudaGetLastError());
  float dec[256], inv[256];
  CHECK(cudaMemcpy(dec, ddec, sizeof(dec), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(inv, dinv, sizeof(inv), cudaMemcpyDeviceToHost));
  long bad_dec = 0, bad_inv = 0, bad_prod = 0;
  for (int c = 0; c < 256; ++c) {
    if (c == 255) {
      bad_dec += !std::isnan(dec[c]);
      bad_inv += !std::isnan(inv[c]);
      continue;
    }
    float const want = std::ldexp(1.0f, c - 127),
                want_inv = std::ldexp(1.0f, 127 - c);
    bad_dec += fb(dec[c]) != fb(want);
    bad_inv += fb(inv[c]) != fb(want_inv);
    bad_prod += dec[c] * inv[c] != 1.0f;
    // The encoder must round-trip exact powers of two.
    bad_dec += ref_e8(want, cudaRoundZero) != static_cast<uint32_t>(c);
  }
  printf(
      "DECODE codes=256 to_f32_mismatch=%ld to_inv_mismatch=%ld "
      "product_not_one=%ld %s (code0=%a code254_inv=%a)\n",
      bad_dec, bad_inv, bad_prod,
      (bad_dec || bad_inv || bad_prod) ? "FAIL" : "PASS", dec[0], inv[254]);
  ok = ok && !bad_dec && !bad_inv && !bad_prod;
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

#include "mx_cvt.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CHECK(c)                                                       \
  do {                                                                 \
    if (auto error = (c); error != cudaSuccess) {                      \
      std::fprintf(stderr, "%s: %s\n", #c, cudaGetErrorString(error)); \
      std::exit(2);                                                    \
    }                                                                  \
  } while (false)

constexpr int kBlockSize = 32;
constexpr int kBlocks = 65536;

__global__ void convert(uint16_t const* bf16, uint8_t* scale, uint32_t* data) {
  int lane = threadIdx.x % 4;
  int block = (blockIdx.x * blockDim.x + threadIdx.x) / 4;
  if (block >= kBlocks) return;
  float values[8];
  float amax = 0.0f;
  for (int i = 0; i < 8; ++i) {
    values[i] = __uint_as_float(
        uint32_t(bf16[block * kBlockSize + lane * 8 + i]) << 16);
    amax = fmaxf(amax, fabsf(values[i]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 4));
  auto code = uccl::mx::mxfp8_scale(amax);
  if (lane == 0) scale[block] = code;
  float multiplier = uccl::mx::ue8m0_to_inv_f32(code);
  for (float& v : values) v *= multiplier;
  data[block * 8 + lane * 2] =
      uccl::mx::e4m3x4_rn(values[0], values[1], values[2], values[3]);
  data[block * 8 + lane * 2 + 1] =
      uccl::mx::e4m3x4_rn(values[4], values[5], values[6], values[7]);
}

static float expand(uint16_t bf16) {
  auto bits = uint32_t(bf16) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

int main() {
  std::vector<uint16_t> input(kBlocks * kBlockSize);
  for (int block = 0; block < kBlocks; ++block) {
    unsigned bits = block & 0x7fff;
    // Test every finite magnitude at every exponent, with varied lower
    // magnitudes, both signs, and input maxima in each group of eight.
    if (bits >= 0x7f80) bits = block % 2 ? 0x7f7f : 0;
    auto* row = &input[block * kBlockSize];
    for (int i = 0; i < kBlockSize; ++i)
      row[i] = (bits > unsigned(i * 3) ? bits - unsigned(i * 3) : 0) |
               (((i ^ (block >> 15)) & 1) << 15);
    std::rotate(row, row + block % kBlockSize, row + kBlockSize);
  }
  uint16_t* d_input;
  uint8_t* d_scale;
  uint32_t* d_data;
  CHECK(cudaMalloc(&d_input, input.size() * sizeof(uint16_t)));
  CHECK(cudaMalloc(&d_scale, kBlocks));
  CHECK(cudaMalloc(&d_data, input.size()));
  CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice));
  convert<<<(kBlocks * 4 + 127) / 128, 128>>>(d_input, d_scale, d_data);
  CHECK(cudaGetLastError());
  std::vector<uint8_t> scale(kBlocks), data(input.size());
  CHECK(
      cudaMemcpy(scale.data(), d_scale, scale.size(), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(data.data(), d_data, data.size(), cudaMemcpyDeviceToHost));
  size_t bad_scale = 0, bad_data = 0;
  for (int block = 0; block < kBlocks; ++block) {
    auto row = &input[block * kBlockSize];
    double amax = 0;
    for (int i = 0; i < kBlockSize; ++i)
      amax = std::max(amax, std::fabs(double(expand(row[i]))));
    int exponent = amax == 0 ? -127 : std::max(-127, std::ilogb(amax) - 8);
    bad_scale += scale[block] != exponent + 127;
    for (int i = 0; i < kBlockSize; ++i) {
      auto ref = __nv_cvt_float_to_fp8(std::ldexp(expand(row[i]), -exponent),
                                       __NV_SATFINITE, __NV_E4M3);
      bad_data += ref != data[block * kBlockSize + i];
    }
  }
  std::printf(
      "MXFP8 blocks=%d elements=%zu scale_mismatch=%zu data_mismatch=%zu\n",
      kBlocks, data.size(), bad_scale, bad_data);
  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_scale));
  CHECK(cudaFree(d_input));
  return bad_scale || bad_data ? 1 : 0;
}

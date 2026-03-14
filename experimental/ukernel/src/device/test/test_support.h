#pragma once

#include "../gpu_rt.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace UKernel {
namespace Device {
namespace Testing {

inline void ck(gpuError_t e, char const* msg) {
  if (e != gpuSuccess) {
    std::cerr << "CUDA error: " << msg << ": " << gpuGetErrorString(e) << "\n";
    std::exit(1);
  }
}

inline bool feq(float a, float b, float rtol = 1e-5f, float atol = 1e-6f) {
  float diff = std::fabs(a - b);
  return diff <= (atol + rtol * std::fabs(b));
}

inline void fill(std::vector<float>& v, float base, float step) {
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = base + step * static_cast<float>(i);
  }
}

}  // namespace Testing
}  // namespace Device
}  // namespace UKernel

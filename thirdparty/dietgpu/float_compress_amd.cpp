#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <iostream>
#include <vector>

// DietGPU float codec API
#include "dietgpu/float/GpuFloatCodec_hip.h"

// ✅ HIP 版 StackDeviceMemory（根据你贴的文件）
#include "dietgpu/utils/StackDeviceMemory_hip.h"

using namespace dietgpu;

#define HIP_CHECK(cmd)                                                     \
  do {                                                                     \
    hipError_t e = (cmd);                                                  \
    if (e != hipSuccess) {                                                 \
      std::cerr << "HIP error: " << hipGetErrorString(e)                   \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

int main() {
  // -----------------------------
  // 1) HIP stream
  // -----------------------------
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr uint32_t kNumFloats = 1024;

  // -----------------------------
  // 2) Host input (FP16)
  // -----------------------------
  std::vector<__half> h_input(kNumFloats);
  for (uint32_t i = 0; i < kNumFloats; ++i) {
    h_input[i] = __float2half(static_cast<float>(i));
  }

  // -----------------------------
  // 3) Device input
  // -----------------------------
  void* d_input = nullptr;
  HIP_CHECK(hipMalloc(&d_input, kNumFloats * sizeof(__half)));
  HIP_CHECK(hipMemcpyAsync(
      d_input, h_input.data(), kNumFloats * sizeof(__half),
      hipMemcpyHostToDevice, stream));

  // -----------------------------
  // 4) Max compressed size (bytes)
  // -----------------------------
  uint32_t maxCompressedBytes =
      getMaxFloatCompressedSize(FloatType::kFloat16, kNumFloats);

  std::cout << "Max compressed bytes = " << maxCompressedBytes << "\n";

  // -----------------------------
  // 5) Device output buffer
  // -----------------------------
  void* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, maxCompressedBytes));

  // Actual compressed size output (optional)
  uint32_t* d_outSize = nullptr;
  HIP_CHECK(hipMalloc(&d_outSize, sizeof(uint32_t)));

  // -----------------------------
  // 6) Batch arrays (HOST arrays of DEVICE pointers)
  // -----------------------------
  const void* in_ptrs[1] = { d_input };
  uint32_t in_sizes[1]   = { kNumFloats };
  void* out_ptrs[1]      = { d_output };

  // -----------------------------
  // 7) Compression config
  // -----------------------------
  FloatCompressConfig config;
  config.floatType = FloatType::kFloat16;
  config.useChecksum = false;
  config.is16ByteAligned = true;

  // -----------------------------
  // 8) ✅ StackDeviceMemory (HIP): use factory
  // -----------------------------
  // This creates a StackDeviceMemory for the current device and pre-allocates
  // temporary memory (default 256MB). You can tune it if needed.
  auto scratch = makeStackMemory(/*bytes=*/256 * 1024 * 1024);

  // -----------------------------
  // 9) Compress
  // -----------------------------
  floatCompress(
      scratch,
      config,
      /*numInBatch=*/1,
      in_ptrs,
      in_sizes,
      out_ptrs,
      d_outSize,
      stream);

  HIP_CHECK(hipStreamSynchronize(stream));

  uint32_t h_outSize = 0;
  HIP_CHECK(hipMemcpy(
      &h_outSize, d_outSize, sizeof(uint32_t),
      hipMemcpyDeviceToHost));

  std::cout << "Actual compressed bytes = " << h_outSize << "\n";

  // -----------------------------
  // 10) Cleanup
  // -----------------------------
  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));
  HIP_CHECK(hipFree(d_outSize));
  HIP_CHECK(hipStreamDestroy(stream));

  return 0;
}

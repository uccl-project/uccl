/**
 * Standalone C++ test for dietgpu float compression/decompression.
 * Verifies the float codec by:
 *   1. Generating random float16 data on GPU
 *   2. Compressing it with floatCompress
 *   3. Decompressing the result with floatDecompress
 *   4. Comparing the decompressed output against the original
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <glog/logging.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

#define CHECK_CUDA(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(err));                                      \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

// Generate pseudo-random float16 data on host, then copy to device
void generateTestData(half** d_data, uint32_t numElements) {
  std::vector<half> hostData(numElements);
  srand(42);
  for (uint32_t i = 0; i < numElements; i++) {
    // Normal-ish distribution: values in [-2, 2]
    float val = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
    hostData[i] = __float2half(val);
  }

  CHECK_CUDA(cudaMalloc(d_data, numElements * sizeof(half)));
  CHECK_CUDA(cudaMemcpy(*d_data, hostData.data(),
                         numElements * sizeof(half), cudaMemcpyHostToDevice));
}

bool runTest(uint32_t numElements) {
  printf("=== Testing with %u float16 elements (%.2f KB) ===\n",
         numElements, numElements * sizeof(half) / 1024.0f);

  // 1. Setup
  int device = 0;
  CHECK_CUDA(cudaSetDevice(device));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Allocate 256 MB of temp memory for dietgpu
  StackDeviceMemory res(device, 256 * 1024 * 1024);

  // 2. Generate input data
  half* d_input = nullptr;
  generateTestData(&d_input, numElements);

  // 3. Allocate output buffer for compression
  uint32_t maxCompSize = getMaxFloatCompressedSize(FloatType::kFloat16, numElements);
  void* d_compressed = nullptr;
  CHECK_CUDA(cudaMalloc(&d_compressed, maxCompSize));
  printf("  Max compressed size: %u bytes\n", maxCompSize);

  // 4. Compress (batch of 1)
  const void* inPtrs[1] = {d_input};
  uint32_t inSizes[1] = {numElements};  // in float words, not bytes
  void* outPtrs[1] = {d_compressed};

  uint32_t* d_compressedSize = nullptr;
  CHECK_CUDA(cudaMalloc(&d_compressedSize, sizeof(uint32_t)));

  FloatCompressConfig compConfig;
  compConfig.floatType = FloatType::kFloat16;
  compConfig.useChecksum = false;
  compConfig.is16ByteAligned = false;

  floatCompress(res, compConfig, 1, inPtrs, inSizes, outPtrs,
                d_compressedSize, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Read back compressed size
  uint32_t compressedSize = 0;
  CHECK_CUDA(cudaMemcpy(&compressedSize, d_compressedSize, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost));

  float ratio = (float)(numElements * sizeof(half)) / compressedSize;
  printf("  Compressed size: %u bytes (ratio: %.2fx)\n", compressedSize, ratio);

  // 5. Decompress
  half* d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_output, numElements * sizeof(half)));
  CHECK_CUDA(cudaMemset(d_output, 0, numElements * sizeof(half)));

  const void* decInPtrs[1] = {d_compressed};
  void* decOutPtrs[1] = {d_output};
  uint32_t outCapacities[1] = {numElements};  // in float words

  uint8_t* d_success = nullptr;
  CHECK_CUDA(cudaMalloc(&d_success, sizeof(uint8_t)));

  uint32_t* d_decompSize = nullptr;
  CHECK_CUDA(cudaMalloc(&d_decompSize, sizeof(uint32_t)));

  FloatDecompressConfig decConfig;
  decConfig.floatType = FloatType::kFloat16;
  decConfig.useChecksum = false;
  decConfig.is16ByteAligned = false;

  FloatDecompressStatus status = floatDecompress(
      res, decConfig, 1, decInPtrs, decOutPtrs, outCapacities,
      d_success, d_decompSize, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (status.error != FloatDecompressError::None) {
    printf("  FAIL: Decompression returned error\n");
    for (auto& e : status.errorInfo) {
      printf("    Batch %d: %s\n", e.first, e.second.c_str());
    }
    return false;
  }

  // Check success flag
  uint8_t success = 0;
  CHECK_CUDA(cudaMemcpy(&success, d_success, sizeof(uint8_t),
                         cudaMemcpyDeviceToHost));
  if (!success) {
    printf("  FAIL: Decompression success flag is false\n");
    return false;
  }

  // 6. Verify: compare original vs decompressed
  std::vector<half> origHost(numElements);
  std::vector<half> decompHost(numElements);
  CHECK_CUDA(cudaMemcpy(origHost.data(), d_input, numElements * sizeof(half),
                         cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(decompHost.data(), d_output, numElements * sizeof(half),
                         cudaMemcpyDeviceToHost));

  bool match = true;
  for (uint32_t i = 0; i < numElements; i++) {
    if (memcmp(&origHost[i], &decompHost[i], sizeof(half)) != 0) {
      printf("  FAIL: Mismatch at index %u: orig=0x%04x, decomp=0x%04x\n",
             i, *(uint16_t*)&origHost[i], *(uint16_t*)&decompHost[i]);
      match = false;
      break;
    }
  }

  if (match) {
    printf("  PASS: Decompressed data matches original exactly\n");
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_compressed));
  CHECK_CUDA(cudaFree(d_compressedSize));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_success));
  CHECK_CUDA(cudaFree(d_decompSize));
  CHECK_CUDA(cudaStreamDestroy(stream));

  return match;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  int deviceCount = 0;
  CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "No CUDA devices found\n");
    return 1;
  }

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("Using GPU: %s (compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

  bool allPassed = true;

  // Test with various sizes
  // dietgpu works best with >= 512 KiB of data
  allPassed &= runTest(512 * 1024);       // 512K elements = 1 MB
  allPassed &= runTest(128 * 512 * 1024); // 64M elements = 128 MB
  allPassed &= runTest(1024);             // Small test (1K elements)

  printf("\n%s\n", allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return allPassed ? 0 : 1;
}

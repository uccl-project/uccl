/**
 * @file test_compression.cpp
 * @brief Unit test for compression.h Compressor class and dietgpu compression
 *
 * Build (CUDA):
 *   g++ -O3 -g test_compression.cpp -o test_compression \
 *       -I../../include -I../../../include -I../../../../thirdparty/dietgpu \
 *       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
 *       -L../../../../thirdparty/dietgpu/dietgpu/float -ldietgpu_float \
 *       -Wl,-rpath,../../../../thirdparty/dietgpu/dietgpu/float \
 *       -lcudart -lcuda -libverbs -lglog -lgflags -lpthread -std=c++17
 *
 * Build (ROCm/HIP):
 *   g++ -O3 -g test_compression.cpp -o test_compression \
 *       -D__HIP_PLATFORM_AMD__ \
 *       -I../../include -I../../../include -I../../../../thirdparty/dietgpu \
 *       -I/opt/rocm/include -L/opt/rocm/lib \
 *       -L../../../../thirdparty/dietgpu/dietgpu/float -ldietgpu_float \
 *       -Wl,-rpath,../../../../thirdparty/dietgpu/dietgpu/float \
 *       -lamdhip64 -libverbs -lglog -lgflags -lpthread -std=c++17
 *
 * Run:
 *   ./test_compression [--gpu_index=0] [--buffer_size=1048576] [--float_type=fp32]
 */

#include "../compression.h"
#include "../memory_allocator.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Command line flags
DEFINE_int32(gpu_index, 0, "GPU index to use");
DEFINE_uint64(buffer_size, 1024 * 1024, "Buffer size in bytes (default 1MB)");
DEFINE_string(float_type, "fp32", "Float type: fp16, bf16, or fp32");
DEFINE_int32(iterations, 10, "Number of iterations for the test");
DEFINE_bool(verbose, false, "Enable verbose output");

// Helper to get FloatType from string
dietgpu::FloatType getFloatTypeFromString(const std::string& type_str) {
  if (type_str == "fp16" || type_str == "float16") {
    return dietgpu::FloatType::kFloat16;
  } else if (type_str == "bf16" || type_str == "bfloat16") {
    return dietgpu::FloatType::kBFloat16;
  } else if (type_str == "fp32" || type_str == "float32") {
    return dietgpu::FloatType::kFloat32;
  } else {
    LOG(WARNING) << "Unknown float type '" << type_str << "', defaulting to fp32";
    return dietgpu::FloatType::kFloat32;
  }
}

// Get element size for a given float type
size_t getElementSize(dietgpu::FloatType float_type) {
  switch (float_type) {
    case dietgpu::FloatType::kFloat16:
    case dietgpu::FloatType::kBFloat16:
      return 2;
    case dietgpu::FloatType::kFloat32:
      return 4;
    default:
      return 4;
  }
}

// Fill buffer with random float data
void fillRandomFloatData(void* host_buf, size_t num_elements,
                         dietgpu::FloatType float_type) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  if (float_type == dietgpu::FloatType::kFloat32) {
    float* data = static_cast<float*>(host_buf);
    for (size_t i = 0; i < num_elements; ++i) {
      data[i] = dist(gen);
    }
  } else if (float_type == dietgpu::FloatType::kFloat16) {
    // For fp16, we still generate as float and will let GPU handle conversion
    // Here we just store as uint16_t placeholder
    uint16_t* data = static_cast<uint16_t*>(host_buf);
    for (size_t i = 0; i < num_elements; ++i) {
      float val = dist(gen);
      // Simple float to fp16 conversion (truncation)
      uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
      uint16_t sign = (bits >> 16) & 0x8000;
      int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
      uint32_t mant = (bits >> 13) & 0x03FF;
      if (exp <= 0) {
        data[i] = sign;  // zero/denormal
      } else if (exp >= 31) {
        data[i] = sign | 0x7C00;  // inf
      } else {
        data[i] = sign | (exp << 10) | mant;
      }
    }
  } else if (float_type == dietgpu::FloatType::kBFloat16) {
    uint16_t* data = static_cast<uint16_t*>(host_buf);
    for (size_t i = 0; i < num_elements; ++i) {
      float val = dist(gen);
      // BF16 is just the upper 16 bits of float32
      uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
      data[i] = static_cast<uint16_t>(bits >> 16);
    }
  }
}

// Compare two buffers with tolerance
bool compareBuffers(const void* buf1, const void* buf2, size_t num_elements,
                    dietgpu::FloatType float_type, float tolerance = 1e-5f) {
  size_t mismatches = 0;
  const size_t max_print = 10;

  if (float_type == dietgpu::FloatType::kFloat32) {
    const float* data1 = static_cast<const float*>(buf1);
    const float* data2 = static_cast<const float*>(buf2);
    for (size_t i = 0; i < num_elements; ++i) {
      if (std::fabs(data1[i] - data2[i]) > tolerance) {
        if (mismatches < max_print) {
          LOG(WARNING) << "Mismatch at index " << i << ": " << data1[i]
                       << " vs " << data2[i];
        }
        mismatches++;
      }
    }
  } else {
    // For fp16/bf16, compare raw bits (lossless compression should be exact)
    const uint16_t* data1 = static_cast<const uint16_t*>(buf1);
    const uint16_t* data2 = static_cast<const uint16_t*>(buf2);
    for (size_t i = 0; i < num_elements; ++i) {
      if (data1[i] != data2[i]) {
        if (mismatches < max_print) {
          LOG(WARNING) << "Mismatch at index " << i << ": 0x" << std::hex
                       << data1[i] << " vs 0x" << data2[i] << std::dec;
        }
        mismatches++;
      }
    }
  }

  if (mismatches > 0) {
    LOG(ERROR) << "Total mismatches: " << mismatches << " out of "
               << num_elements << " elements";
    return false;
  }
  return true;
}

// Test basic compression and decompression roundtrip
bool testCompressionRoundtrip(Compressor& compressor,
                              MemoryAllocator& allocator,
                              size_t buffer_size,
                              dietgpu::FloatType float_type) {
  LOG(INFO) << "=== Testing Compression Roundtrip ===";

  size_t elem_size = getElementSize(float_type);
  size_t num_elements = buffer_size / elem_size;

  LOG(INFO) << "Buffer size: " << buffer_size << " bytes";
  LOG(INFO) << "Number of elements: " << num_elements;
  LOG(INFO) << "Element size: " << elem_size << " bytes";

  // Allocate GPU buffers
  auto input_mem = allocator.allocate(buffer_size, MemoryType::GPU, nullptr);
  auto output_mem = allocator.allocate(buffer_size, MemoryType::GPU, nullptr);

  // Allocate host buffers for verification
  std::vector<char> h_input(buffer_size);
  std::vector<char> h_output(buffer_size);

  // Fill input with random data
  fillRandomFloatData(h_input.data(), num_elements, float_type);

  // Copy input to GPU
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  GPU_CHECK(hipMemcpy(input_mem->addr, h_input.data(), buffer_size,
                      hipMemcpyHostToDevice));
  GPU_CHECK(hipMemset(output_mem->addr, 0, buffer_size));
#else
  GPU_CHECK(cudaMemcpy(input_mem->addr, h_input.data(), buffer_size,
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemset(output_mem->addr, 0, buffer_size));
#endif

  // Create a send request for compression
  auto remote_mem = std::make_shared<RemoteMemInfo>();
  auto send_req = std::make_shared<RDMASendRequest>(input_mem, remote_mem);
  send_req->float_type = float_type;

  // Compress
  auto start_compress = std::chrono::high_resolution_clock::now();
  bool compress_ok = compressor.compress(send_req);
  auto end_compress = std::chrono::high_resolution_clock::now();

  if (!compress_ok) {
    LOG(ERROR) << "Compression failed!";
    return false;
  }

  auto compress_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           end_compress - start_compress)
                           .count();
  LOG(INFO) << "Compression time: " << compress_time << " us";

  size_t compressed_size = send_req->local_mem->size;
  float compression_ratio =
      static_cast<float>(buffer_size) / static_cast<float>(compressed_size);
  LOG(INFO) << "Compressed size: " << compressed_size << " bytes";
  LOG(INFO) << "Compression ratio: " << compression_ratio << "x";

  // Prepare for decompression using the overload with RemoteMemInfo and
  // RegMemBlock
  RemoteMemInfo compressed_input;
  compressed_input.addr =
      reinterpret_cast<uint64_t>(send_req->local_mem->addr);
  compressed_input.length = compressed_size;

  // Decompress
  auto start_decompress = std::chrono::high_resolution_clock::now();
  bool decompress_ok =
      compressor.decompress(compressed_input, *output_mem, float_type);
  auto end_decompress = std::chrono::high_resolution_clock::now();

  if (!decompress_ok) {
    LOG(ERROR) << "Decompression failed!";
    return false;
  }

  auto decompress_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             end_decompress - start_decompress)
                             .count();
  LOG(INFO) << "Decompression time: " << decompress_time << " us";

  // Copy output back to host
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  GPU_CHECK(hipMemcpy(h_output.data(), output_mem->addr, buffer_size,
                      hipMemcpyDeviceToHost));
#else
  GPU_CHECK(cudaMemcpy(h_output.data(), output_mem->addr, buffer_size,
                       cudaMemcpyDeviceToHost));
#endif

  // Compare
  bool match = compareBuffers(h_input.data(), h_output.data(), num_elements,
                              float_type);

  if (match) {
    LOG(INFO) << "Roundtrip test PASSED!";
  } else {
    LOG(ERROR) << "Roundtrip test FAILED!";
  }

  return match;
}

// Test with RDMASendRequest and RDMARecvRequest workflow
bool testRequestWorkflow(Compressor& compressor, MemoryAllocator& allocator,
                         size_t buffer_size, dietgpu::FloatType float_type) {
  LOG(INFO) << "=== Testing Request Workflow ===";

  size_t elem_size = getElementSize(float_type);
  size_t num_elements = buffer_size / elem_size;

  // Allocate GPU buffers
  auto sender_input_mem =
      allocator.allocate(buffer_size, MemoryType::GPU, nullptr);
  auto receiver_output_mem =
      allocator.allocate(buffer_size, MemoryType::GPU, nullptr);

  // Allocate host buffers for verification
  std::vector<char> h_input(buffer_size);
  std::vector<char> h_output(buffer_size);

  // Fill input with random data
  fillRandomFloatData(h_input.data(), num_elements, float_type);

  // Copy input to GPU
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  GPU_CHECK(hipMemcpy(sender_input_mem->addr, h_input.data(), buffer_size,
                      hipMemcpyHostToDevice));
  GPU_CHECK(hipMemset(receiver_output_mem->addr, 0, buffer_size));
#else
  GPU_CHECK(cudaMemcpy(sender_input_mem->addr, h_input.data(), buffer_size,
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemset(receiver_output_mem->addr, 0, buffer_size));
#endif

  // === Sender side ===
  auto remote_mem = std::make_shared<RemoteMemInfo>();
  auto send_req = std::make_shared<RDMASendRequest>(sender_input_mem, remote_mem);
  send_req->float_type = float_type;

  // Compress on sender
  bool compress_ok = compressor.compress(send_req);
  if (!compress_ok) {
    LOG(ERROR) << "Sender compression failed!";
    return false;
  }

  size_t compressed_size = send_req->local_mem->size;
  LOG(INFO) << "Sender compressed data: " << buffer_size << " -> "
            << compressed_size << " bytes";

  // Simulate network transfer: copy compressed data to receiver's buffer
  // (In real RDMA, this would be done by the NIC via RDMA write)
  auto receiver_compressed_mem =
      allocator.allocate(compressed_size, MemoryType::GPU, nullptr);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  GPU_CHECK(hipMemcpy(receiver_compressed_mem->addr, send_req->local_mem->addr,
                      compressed_size, hipMemcpyDeviceToDevice));
#else
  GPU_CHECK(cudaMemcpy(receiver_compressed_mem->addr, send_req->local_mem->addr,
                       compressed_size, cudaMemcpyDeviceToDevice));
#endif

  // === Receiver side ===
  // Receiver has compressed data and needs to decompress to output buffer
  auto recv_req = std::make_shared<RDMARecvRequest>(receiver_output_mem);
  recv_req->float_type = float_type;

  // First prepare: set up local_mem to compressed buffer, backup original to
  // local_compression_mem
  recv_req->local_compression_mem = receiver_output_mem;  // Final destination
  recv_req->local_mem = receiver_compressed_mem;          // Compressed data

  // Decompress
  bool decompress_ok = compressor.decompress(recv_req);
  if (!decompress_ok) {
    LOG(ERROR) << "Receiver decompression failed!";
    return false;
  }

  LOG(INFO) << "Receiver decompressed data to output buffer";

  // Copy output back to host
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  GPU_CHECK(hipMemcpy(h_output.data(), receiver_output_mem->addr, buffer_size,
                      hipMemcpyDeviceToHost));
#else
  GPU_CHECK(cudaMemcpy(h_output.data(), receiver_output_mem->addr, buffer_size,
                       cudaMemcpyDeviceToHost));
#endif

  // Compare
  bool match = compareBuffers(h_input.data(), h_output.data(), num_elements,
                              float_type);

  if (match) {
    LOG(INFO) << "Request workflow test PASSED!";
  } else {
    LOG(ERROR) << "Request workflow test FAILED!";
  }

  return match;
}

// Bandwidth test
void testBandwidth(Compressor& compressor, MemoryAllocator& allocator,
                   size_t buffer_size, dietgpu::FloatType float_type,
                   int iterations) {
  LOG(INFO) << "=== Testing Bandwidth (" << iterations << " iterations) ===";

  size_t elem_size = getElementSize(float_type);
  size_t num_elements = buffer_size / elem_size;

  // Allocate GPU buffers
  auto input_mem = allocator.allocate(buffer_size, MemoryType::GPU, nullptr);
  auto output_mem = allocator.allocate(buffer_size, MemoryType::GPU, nullptr);

  // Allocate host buffer and fill with random data
  std::vector<char> h_input(buffer_size);
  fillRandomFloatData(h_input.data(), num_elements, float_type);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  GPU_CHECK(hipMemcpy(input_mem->addr, h_input.data(), buffer_size,
                      hipMemcpyHostToDevice));
#else
  GPU_CHECK(cudaMemcpy(input_mem->addr, h_input.data(), buffer_size,
                       cudaMemcpyHostToDevice));
#endif

  double total_compress_time = 0;
  double total_decompress_time = 0;
  size_t total_compressed_bytes = 0;

  for (int i = 0; i < iterations; ++i) {
    // Create fresh request for each iteration
    auto remote_mem = std::make_shared<RemoteMemInfo>();
    auto send_req = std::make_shared<RDMASendRequest>(input_mem, remote_mem);
    send_req->float_type = float_type;

    // Need to reset input_mem since compress modifies send_req->local_mem
    send_req->local_mem = input_mem;

    // Compress
    auto start_compress = std::chrono::high_resolution_clock::now();
    compressor.compress(send_req);
    auto end_compress = std::chrono::high_resolution_clock::now();

    total_compress_time +=
        std::chrono::duration<double, std::milli>(end_compress - start_compress)
            .count();
    total_compressed_bytes += send_req->local_mem->size;

    // Decompress
    RemoteMemInfo compressed_input;
    compressed_input.addr =
        reinterpret_cast<uint64_t>(send_req->local_mem->addr);
    compressed_input.length = send_req->local_mem->size;

    auto start_decompress = std::chrono::high_resolution_clock::now();
    compressor.decompress(compressed_input, *output_mem, float_type);
    auto end_decompress = std::chrono::high_resolution_clock::now();

    total_decompress_time += std::chrono::duration<double, std::milli>(
                                 end_decompress - start_decompress)
                                 .count();
  }

  double avg_compress_time = total_compress_time / iterations;
  double avg_decompress_time = total_decompress_time / iterations;
  double avg_compressed_size =
      static_cast<double>(total_compressed_bytes) / iterations;
  double compression_ratio = buffer_size / avg_compressed_size;

  double compress_bandwidth =
      (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (avg_compress_time / 1000.0);
  double decompress_bandwidth =
      (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (avg_decompress_time / 1000.0);

  LOG(INFO) << "Results:";
  LOG(INFO) << "  Buffer size: " << buffer_size / (1024.0 * 1024.0) << " MB";
  LOG(INFO) << "  Average compression time: " << avg_compress_time << " ms";
  LOG(INFO) << "  Average decompression time: " << avg_decompress_time << " ms";
  LOG(INFO) << "  Average compression ratio: " << compression_ratio << "x";
  LOG(INFO) << "  Compression bandwidth: " << compress_bandwidth << " GB/s";
  LOG(INFO) << "  Decompression bandwidth: " << decompress_bandwidth << " GB/s";
}

// Test shouldCompress static method
void testShouldCompress() {
  LOG(INFO) << "=== Testing shouldCompress() ===";

  struct TestCase {
    size_t size;
    bool expected;
  };

  std::vector<TestCase> test_cases = {
      {0, false},
      {1024, false},                          // 1KB - below threshold
      {kMinCompressBytes - 1, false},         // Just below threshold
      {kMinCompressBytes, true},              // Exactly at threshold
      {kMinCompressBytes + 1, true},          // Just above threshold
      {1024 * 1024, true},                    // 1MB
      {100 * 1024 * 1024, true},              // 100MB
  };

  bool all_passed = true;
  for (const auto& tc : test_cases) {
    bool result = Compressor::shouldCompress(tc.size);
    bool passed = (result == tc.expected);
    if (FLAGS_verbose || !passed) {
      LOG(INFO) << "  size=" << tc.size << " bytes: expected=" << tc.expected
                << ", got=" << result << " -> " << (passed ? "PASS" : "FAIL");
    }
    if (!passed) all_passed = false;
  }

  if (all_passed) {
    LOG(INFO) << "shouldCompress() test PASSED!";
  } else {
    LOG(ERROR) << "shouldCompress() test FAILED!";
  }
}

int main(int argc, char* argv[]) {
  // Initialize gflags and glog
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  LOG(INFO) << "========================================";
  LOG(INFO) << "   Compressor Unit Test";
  LOG(INFO) << "========================================";

  // Set GPU device
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  GPU_CHECK(hipSetDevice(FLAGS_gpu_index));
  LOG(INFO) << "Using HIP device: " << FLAGS_gpu_index;
#else
  GPU_CHECK(cudaSetDevice(FLAGS_gpu_index));
  LOG(INFO) << "Using CUDA device: " << FLAGS_gpu_index;
#endif

  // Get float type
  dietgpu::FloatType float_type = getFloatTypeFromString(FLAGS_float_type);
  LOG(INFO) << "Float type: " << FLAGS_float_type;
  LOG(INFO) << "Buffer size: " << FLAGS_buffer_size << " bytes";
  LOG(INFO) << "";

  // Create allocator and compressor
  MemoryAllocator allocator;
  Compressor compressor;

  bool all_passed = true;

  // Test shouldCompress
  testShouldCompress();
  LOG(INFO) << "";

  // Test compression roundtrip
  if (!testCompressionRoundtrip(compressor, allocator, FLAGS_buffer_size,
                                 float_type)) {
    all_passed = false;
  }
  LOG(INFO) << "";

  // Test request workflow
  if (!testRequestWorkflow(compressor, allocator, FLAGS_buffer_size,
                           float_type)) {
    all_passed = false;
  }
  LOG(INFO) << "";

  // Bandwidth test
  testBandwidth(compressor, allocator, FLAGS_buffer_size, float_type,
                FLAGS_iterations);
  LOG(INFO) << "";

  LOG(INFO) << "========================================";
  if (all_passed) {
    LOG(INFO) << "   ALL TESTS PASSED";
  } else {
    LOG(ERROR) << "   SOME TESTS FAILED";
  }
  LOG(INFO) << "========================================";

  return all_passed ? 0 : 1;
}

// SPDX-License-Identifier: Apache-2.0
#include "traffic_staging.cuh"
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(call) do { auto status = (call); if (status != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); std::exit(1); \
} } while (0)
#define REQUIRE(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "failed: %s\n", #condition); std::exit(2); \
} } while (0)

using nccl_ep::hybridep::traffic::Path;
namespace traffic = nccl_ep::hybridep::traffic;

struct Options {
  int words = 1 << 24, ctas = 148, passes = 1, repeats = 5, offset = 0;
  bool tails = false;
};

Options parse(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--tails") == 0) { result.tails = true; continue; }
    REQUIRE(i + 1 < argc);
    char const* name = argv[i];
    char const* value = argv[++i];
    int number;
    auto parsed = std::from_chars(value, value + std::strlen(value), number);
    REQUIRE(parsed.ec == std::errc{} && *parsed.ptr == '\0' && number >= 0);
    if (std::strcmp(name, "--words") == 0) result.words = number;
    else if (std::strcmp(name, "--ctas") == 0) result.ctas = number;
    else if (std::strcmp(name, "--passes") == 0) result.passes = number;
    else if (std::strcmp(name, "--repeats") == 0) result.repeats = number;
    else if (std::strcmp(name, "--offset") == 0) result.offset = number;
    else REQUIRE(false);
  }
  REQUIRE(result.words > 0 && result.ctas > 0 && result.ctas <= 4096);
  REQUIRE(result.passes > 0 && result.passes <= 16 && result.repeats > 0 && result.repeats <= 1000);
  REQUIRE(result.offset <= 3);
  return result;
}

void measure(Options const& o, Path path, cudaDeviceProp const& device) {
  std::vector<uint32_t> input(o.words);
  std::mt19937 random(13);
  for (auto& word : input) word = random();
  std::size_t prefix = 4 + o.offset;
  std::size_t storage_words = prefix + o.words + 4;
  std::size_t bytes = input.size() * 4;
  uint32_t *source, *destination;
  CK(cudaMalloc(&source, storage_words * 4)); CK(cudaMalloc(&destination, storage_words * 4));
  CK(cudaMemset(source, 0xa5, storage_words * 4));
  CK(cudaMemcpy(source + prefix, input.data(), bytes, cudaMemcpyHostToDevice));
  cudaStream_t stream;
  cudaEvent_t start, stop;
  CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
  nccl_ep::hybridep::KernelResources resources;
  if (path != Path::Copy) CK(traffic::query(path, o.ctas, device, resources));
  std::vector<float> samples;
  for (int round = -3; round < o.repeats; ++round) {
    CK(cudaEventRecord(start, stream));
    CK(traffic::launch(path, source + prefix, destination + prefix, bytes, o.ctas, o.passes, stream));
    CK(cudaEventRecord(stop, stream)); CK(cudaEventSynchronize(stop));
    if (round >= 0) {
      float ms;
      CK(cudaEventElapsedTime(&ms, start, stop)); samples.push_back(ms);
    }
  }
  // Validate a fresh poisoned wave, including source preservation and both guards.
  CK(cudaMemsetAsync(destination, 0xa5, storage_words * 4, stream));
  CK(traffic::launch(path, source + prefix, destination + prefix, bytes, o.ctas, o.passes, stream));
  CK(cudaStreamSynchronize(stream));
  std::vector<uint32_t> result(storage_words), preserved(storage_words);
  CK(cudaMemcpy(result.data(), destination, result.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(preserved.data(), source, preserved.size() * 4, cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < storage_words; ++i) {
    uint32_t expected = i >= prefix && i < prefix + input.size() ? input[i - prefix] : 0xa5a5a5a5u;
    REQUIRE(result[i] == expected && preserved[i] == expected);
  }
  std::printf("{\"status\":\"PASS\",\"path\":\"%s\",\"sm_major\":%d,\"sm_minor\":%d,"
              "\"sms\":%d,\"words\":%d,\"ctas\":%d,\"passes\":%d,\"offset_words\":%d,"
              "\"registers\":%d,\"static_smem\":%zu,\"resident_blocks_api\":%d,"
              "\"tmem_allocations_per_launched_cta\":%d,\"logical_payload_read_bytes_per_wave\":%zu,"
              "\"logical_payload_write_bytes_per_wave\":%zu,\"gpu_wave_ms\":[",
              traffic::name(path), device.major, device.minor, device.multiProcessorCount,
              o.words, o.ctas, o.passes, o.offset, resources.registers, resources.static_smem,
              resources.resident_blocks, path == Path::TensorInput || path == Path::TensorOutput,
              bytes * o.passes, bytes * o.passes);
  for (std::size_t i = 0; i < samples.size(); ++i) std::printf("%s%.9g", i ? "," : "", samples[i]);
  std::printf("]}\n");
  CK(cudaEventDestroy(start)); CK(cudaEventDestroy(stop)); CK(cudaStreamDestroy(stream));
  CK(cudaFree(destination)); CK(cudaFree(source));
}

int main(int argc, char** argv) {
  Options o = parse(argc, argv);
  CK(cudaSetDevice(0));
  cudaDeviceProp device;
  CK(cudaGetDeviceProperties(&device, 0));
  REQUIRE(device.major == 10 && traffic::tensor_compiled);  // Native SM100-family TMEM qualification.
  std::vector<int> sizes = o.tails ? std::vector<int>{1, 127, 128, 511, 512, 513, 1025, 4097} :
                                   std::vector<int>{o.words};
  for (int words : sizes) {
    o.words = words;
    for (Path path : {Path::Copy, Path::Direct, Path::Tma, Path::TensorInput, Path::TensorOutput})
      measure(o, path, device);
  }
  return 0;
}

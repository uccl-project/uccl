// SPDX-License-Identifier: Apache-2.0
#include "device/hybrid_ep.cuh"
#include <cstdio>
#include <future>
#include <thread>

constexpr int large = 160 * 1024, small = 32 * 1024;

__global__ void touch_shared(int* output) {
  extern __shared__ int words[];
  if (threadIdx.x == 0) words[0] = 7;
  __syncthreads();
  if (threadIdx.x == 0) *output = words[0];
}

struct LaunchResult {
  cudaError_t status;
  int value, configured_max;
};

LaunchResult launch(int* output) {
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, touch_shared));
  touch_shared<<<1, 32, large>>>(output);
  auto status = cudaGetLastError();
  int value = 0;
  if (status == cudaSuccess) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&value, output, sizeof(value), cudaMemcpyDeviceToHost));
  }
  return {status, value, attributes.maxDynamicSharedSizeBytes};
}

// The large requester stays alive while a fresh host thread configures small.
// Legacy mode reproduces the old setter; the owner then skips its cached setup.
LaunchResult interleaved(int* output, int device, bool legacy) {
  std::promise<void> large_ready, small_done;
  auto ready = large_ready.get_future();
  auto done = small_done.get_future();
  LaunchResult result{};
  std::thread owner([&] {
    CUDA_CHECK(cudaSetDevice(device));
    if (legacy)
      CUDA_CHECK(cudaFuncSetAttribute(touch_shared, cudaFuncAttributeMaxDynamicSharedMemorySize, large));
    else
      hybrid_ep::ensure_dynamic_smem(touch_shared, large);
    large_ready.set_value();
    done.wait();
    result = launch(output);
  });
  std::thread newcomer([&] {
    CUDA_CHECK(cudaSetDevice(device));
    ready.wait();
    if (legacy)
      CUDA_CHECK(cudaFuncSetAttribute(touch_shared, cudaFuncAttributeMaxDynamicSharedMemorySize, small));
    else
      hybrid_ep::ensure_dynamic_smem(touch_shared, small);
    small_done.set_value();
  });
  owner.join(); newcomer.join();
  return result;
}

int main() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  int* output;
  CUDA_CHECK(cudaMalloc(&output, sizeof(int)));
  auto negative = interleaved(output, device, true);
  auto fixed = interleaved(output, device, false);

  // Mix queries and launch setup from eight host threads on the same device.
  std::promise<void> start;
  auto gate = start.get_future().share();
  std::vector<std::thread> workers;
  for (int thread = 0; thread < 8; ++thread) workers.emplace_back([&, thread] {
    CUDA_CHECK(cudaSetDevice(device));
    gate.wait();
    for (int round = 0; round < 128; ++round) {
      int bytes = (thread + round) % 2 ? small : large;
      if (round % 2) hybrid_ep::ensure_dynamic_smem(touch_shared, bytes);
      else hybrid_ep::query_kernel_resources(touch_shared, device, bytes, 32, 1);
    }
  });
  start.set_value();
  for (auto& worker : workers) worker.join();
  auto first = hybrid_ep::query_kernel_resources(touch_shared, device, large, 32, 1);
  auto second = hybrid_ep::query_kernel_resources(touch_shared, device, small, 32, 1);
  auto final = launch(output);
  bool passed = negative.status != cudaSuccess && negative.configured_max == small &&
      fixed.status == cudaSuccess && fixed.value == 7 && fixed.configured_max >= large &&
      final.status == cudaSuccess && final.value == 7 && final.configured_max >= large;
  std::printf("{\"status\":\"%s\",\"configured_max\":%d,\"large_bytes\":%d,\"small_bytes\":%d,"
              "\"large_resident\":%d,\"small_resident\":%d,\"launch_status\":%d,\"output\":%d,"
              "\"negative_launch_status\":%d,\"negative_configured_max\":%d,"
              "\"interleaved_launch_status\":%d,\"interleaved_configured_max\":%d,"
              "\"host_threads\":8,\"mixed_configuration_calls\":1024}\n",
      passed ? "PASS" : "FAIL", final.configured_max, large, small,
      first.resident_blocks, second.resident_blocks, int(final.status), final.value,
      int(negative.status), negative.configured_max, int(fixed.status), fixed.configured_max);
  CUDA_CHECK(cudaFree(output));
  return passed ? 0 : 2;
}

#include "../../ccl/executor.h"
#include "../../ccl/device_backend.h"
#include "../persistent.h"
#include "test_support.h"
#include <iostream>
#include <vector>

namespace {

constexpr int kNumElems = 1024;
constexpr int kGatherElemsPerRank = kNumElems / 2;

}  // namespace

int main() {
  using namespace UKernel;
  using Device::Testing::ck;
  using Device::Testing::feq;
  using Device::Testing::fill;

  Device::TaskManager::instance().init(1024);

  Device::PersistentKernelConfig config;
  config.numBlocks = 2;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 16;
  config.smemSize = 0;

  std::vector<float> h_gather_remote_input(kNumElems, 0.0f);
  std::vector<float> h_gather_output(kNumElems, 0.0f);
  std::vector<float> h_gather_out(kNumElems, 0.0f);
  for (int i = 0; i < kGatherElemsPerRank; ++i) {
    h_gather_output[i] = 100.0f + static_cast<float>(i);
    h_gather_remote_input[kGatherElemsPerRank + i] =
        200.0f + static_cast<float>(i);
  }

  std::vector<float> h_reduce_local_input(kNumElems, 0.0f);
  std::vector<float> h_reduce_remote_input(kNumElems, 0.0f);
  std::vector<float> h_reduce_remote_reduced(kNumElems, 0.0f);
  std::vector<float> h_reduce_output(kNumElems, 0.0f);
  std::vector<float> h_reduce_out(kNumElems, 0.0f);
  fill(h_reduce_local_input, 10.0f, 0.5f);
  for (int i = 0; i < kGatherElemsPerRank; ++i) {
    h_reduce_remote_reduced[i] =
        h_reduce_local_input[i] + (1.0f + 0.25f * static_cast<float>(i));
  }
  for (int i = kGatherElemsPerRank; i < kNumElems; ++i) {
    h_reduce_remote_input[i] = 1.0f + 0.25f * static_cast<float>(i);
  }

  float* d_gather_remote_input = nullptr;
  float* d_gather_output = nullptr;
  float* d_reduce_local_input = nullptr;
  float* d_reduce_remote_input = nullptr;
  float* d_reduce_remote_reduced = nullptr;
  float* d_reduce_output = nullptr;
  float* d_reduce_staging = nullptr;
  ck(gpuMalloc(&d_gather_remote_input, sizeof(float) * kNumElems),
     "gpuMalloc d_gather_remote_input");
  ck(gpuMalloc(&d_gather_output, sizeof(float) * kNumElems),
     "gpuMalloc d_gather_output");
  ck(gpuMalloc(&d_reduce_local_input, sizeof(float) * kNumElems),
     "gpuMalloc d_reduce_local_input");
  ck(gpuMalloc(&d_reduce_remote_input, sizeof(float) * kNumElems),
     "gpuMalloc d_reduce_remote_input");
  ck(gpuMalloc(&d_reduce_remote_reduced, sizeof(float) * kNumElems),
     "gpuMalloc d_reduce_remote_reduced");
  ck(gpuMalloc(&d_reduce_output, sizeof(float) * kNumElems),
     "gpuMalloc d_reduce_output");
  ck(gpuMalloc(&d_reduce_staging, sizeof(float) * kNumElems),
     "gpuMalloc d_reduce_staging");
  ck(gpuMemcpy(d_gather_remote_input, h_gather_remote_input.data(),
               sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy gather remote input");
  ck(gpuMemcpy(d_gather_output, h_gather_output.data(), sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy gather output");
  ck(gpuMemcpy(d_reduce_local_input, h_reduce_local_input.data(),
               sizeof(float) * kNumElems, gpuMemcpyHostToDevice),
     "copy reduce local input");
  ck(gpuMemcpy(d_reduce_remote_input, h_reduce_remote_input.data(),
               sizeof(float) * kNumElems, gpuMemcpyHostToDevice),
     "copy reduce remote input");
  ck(gpuMemcpy(d_reduce_remote_reduced, h_reduce_remote_reduced.data(),
               sizeof(float) * kNumElems, gpuMemcpyHostToDevice),
     "copy reduce remote reduced");
  ck(gpuMemcpy(d_reduce_output, h_reduce_output.data(), sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy reduce output");
  ck(gpuMemset(d_reduce_staging, 0, sizeof(float) * kNumElems),
     "memset reduce staging");

  Device::PersistentKernel<Device::Task> kernel(config);
  kernel.launch();

  CCL::CollectiveBuffers gather_buffers{};
  gather_buffers.remote_input = d_gather_remote_input;
  gather_buffers.final_output = d_gather_output;
  CCL::PersistentDeviceBackend gather_backend(
      kernel, gather_buffers, Device::DataType::Fp32,
      Device::ReduceType::Sum, Device::TransferPath::RegisterOp,
      config.numBlocks);
  CCL::CollectiveBuffers reduce_buffers{};
  reduce_buffers.local_input = d_reduce_local_input;
  reduce_buffers.remote_input = d_reduce_remote_input;
  reduce_buffers.remote_reduced = d_reduce_remote_reduced;
  reduce_buffers.final_output = d_reduce_output;
  reduce_buffers.recv_staging = d_reduce_staging;
  CCL::PersistentDeviceBackend reduce_backend(
      kernel, reduce_buffers, Device::DataType::Fp32,
      Device::ReduceType::Sum, Device::TransferPath::RegisterOp,
      config.numBlocks);

  CCL::CollectiveConfig gather_request{};
  gather_request.algorithm = CCL::AlgorithmKind::Ring;
  gather_request.nranks = 2;
  gather_request.rank = 0;
  gather_request.channels = 2;
  gather_request.bytes_per_rank =
      sizeof(float) * static_cast<size_t>(kGatherElemsPerRank);
  gather_request.chunk_bytes = gather_request.bytes_per_rank / 2;

  CCL::CollectiveConfig reduce_request{};
  reduce_request.algorithm = CCL::AlgorithmKind::Ring;
  reduce_request.nranks = 2;
  reduce_request.rank = 0;
  reduce_request.channels = 2;
  reduce_request.bytes_per_rank =
      sizeof(float) * static_cast<size_t>(kNumElems);
  reduce_request.chunk_bytes = reduce_request.bytes_per_rank / 4;

  CCL::ExecutorBackends gather_backends{};
  gather_backends.persistent_kernel = &gather_backend;
  CCL::Executor gather_executor(gather_backends);

  CCL::CollectiveOpHandle gather_handle =
      gather_executor.submit_allgather(gather_request);
  gather_executor.wait(gather_handle);
  if (gather_executor.status(gather_handle) !=
      CCL::CollectiveOpStatus::Completed) {
    std::cerr << "allgather executor failed\n";
    return 2;
  }
  gather_executor.release(gather_handle);

  CCL::ExecutorBackends reduce_backends{};
  reduce_backends.persistent_kernel = &reduce_backend;
  CCL::Executor reduce_executor(reduce_backends);

  CCL::CollectiveOpHandle reduce_handle =
      reduce_executor.submit_allreduce(reduce_request);
  reduce_executor.wait(reduce_handle);
  if (reduce_executor.status(reduce_handle) !=
      CCL::CollectiveOpStatus::Completed) {
    std::cerr << "allreduce executor failed\n";
    return 3;
  }
  reduce_executor.release(reduce_handle);

  kernel.stop();

  ck(gpuMemcpy(h_gather_out.data(), d_gather_output, sizeof(float) * kNumElems,
               gpuMemcpyDeviceToHost),
     "copy gather out");
  ck(gpuMemcpy(h_reduce_out.data(), d_reduce_output, sizeof(float) * kNumElems,
               gpuMemcpyDeviceToHost),
     "copy reduce out");

  size_t gather_bad = 0;
  for (int i = 0; i < kGatherElemsPerRank; ++i) {
    float local_expected = 100.0f + static_cast<float>(i);
    float remote_expected = 200.0f + static_cast<float>(i);
    if (!feq(h_gather_out[i], local_expected)) {
      if (gather_bad < 8) {
        std::cerr << "[ALLGATHER LOCAL MISMATCH] i=" << i
                  << " got=" << h_gather_out[i]
                  << " exp=" << local_expected << "\n";
      }
      ++gather_bad;
    }
    if (!feq(h_gather_out[kGatherElemsPerRank + i], remote_expected)) {
      if (gather_bad < 8) {
        std::cerr << "[ALLGATHER REMOTE MISMATCH] i=" << i
                  << " got=" << h_gather_out[kGatherElemsPerRank + i]
                  << " exp=" << remote_expected << "\n";
      }
      ++gather_bad;
    }
  }
  if (gather_bad) {
    std::cerr << "CCL persistent allgather FAILED mismatches=" << gather_bad
              << "\n";
    return 4;
  }

  size_t reduce_bad = 0;
  for (int i = 0; i < kNumElems; ++i) {
    float expected = h_reduce_local_input[i] +
                     (1.0f + 0.25f * static_cast<float>(i));
    if (!feq(h_reduce_out[i], expected)) {
      if (reduce_bad < 8) {
        std::cerr << "[ALLREDUCE MISMATCH] i=" << i << " got="
                  << h_reduce_out[i] << " exp=" << expected << "\n";
      }
      ++reduce_bad;
    }
  }
  if (reduce_bad) {
    std::cerr << "CCL persistent allreduce FAILED mismatches=" << reduce_bad
              << "\n";
    return 5;
  }

  std::cout << "CCL persistent allgather PASSED\n";
  std::cout << "CCL persistent allreduce PASSED\n";

  ck(gpuFree(d_gather_remote_input), "free gather remote input");
  ck(gpuFree(d_gather_output), "free gather output");
  ck(gpuFree(d_reduce_local_input), "free reduce local input");
  ck(gpuFree(d_reduce_remote_input), "free reduce remote input");
  ck(gpuFree(d_reduce_remote_reduced), "free reduce remote reduced");
  ck(gpuFree(d_reduce_output), "free reduce output");
  ck(gpuFree(d_reduce_staging), "free reduce staging");
  return 0;
}

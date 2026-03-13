#include "ccl_api.h"
#include "gpu_rt.h"
#include "persistent.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kNumElems = 1024;
constexpr int kGatherElemsPerRank = kNumElems / 2;

static void ck(gpuError_t e, char const* msg) {
  if (e != gpuSuccess) {
    std::cerr << "CUDA error: " << msg << ": " << gpuGetErrorString(e) << "\n";
    std::exit(1);
  }
}

static bool feq(float a, float b, float rtol = 1e-5f, float atol = 1e-6f) {
  float diff = std::fabs(a - b);
  return diff <= (atol + rtol * std::fabs(b));
}

static void fill(std::vector<float>& v, float base, float step) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = base + step * static_cast<float>(i);
}

}  // namespace

int main() {
  using namespace UKernel;

  Compute::TaskManager::instance().init(1024, 256);

  Compute::PersistentKernelConfig kernel_cfg;
  kernel_cfg.numBlocks = 2;
  kernel_cfg.threadsPerBlock = 64;
  kernel_cfg.fifoCapacity = 16;

  Compute::PersistentKernel<Compute::Task> kernel(kernel_cfg);
  kernel.launch();

  std::vector<float> h_gather_remote_input(kNumElems, 0.0f);
  std::vector<float> h_gather_output(kNumElems, 0.0f);
  std::vector<float> h_gather_out(kNumElems, 0.0f);
  for (int i = 0; i < kGatherElemsPerRank; ++i) {
    h_gather_output[i] = 7.0f + static_cast<float>(i);
    h_gather_remote_input[kGatherElemsPerRank + i] =
        17.0f + static_cast<float>(i);
  }

  std::vector<float> h_reduce_local_input(kNumElems, 0.0f);
  std::vector<float> h_reduce_remote_input(kNumElems, 0.0f);
  std::vector<float> h_reduce_remote_reduced(kNumElems, 0.0f);
  std::vector<float> h_reduce_output(kNumElems, 0.0f);
  std::vector<float> h_reduce_out(kNumElems, 0.0f);
  fill(h_reduce_local_input, 2.0f, 0.5f);
  h_reduce_output = h_reduce_local_input;
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
     "gpuMalloc gather remote input");
  ck(gpuMalloc(&d_gather_output, sizeof(float) * kNumElems),
     "gpuMalloc gather output");
  ck(gpuMalloc(&d_reduce_local_input, sizeof(float) * kNumElems),
     "gpuMalloc reduce local input");
  ck(gpuMalloc(&d_reduce_remote_input, sizeof(float) * kNumElems),
     "gpuMalloc reduce remote input");
  ck(gpuMalloc(&d_reduce_remote_reduced, sizeof(float) * kNumElems),
     "gpuMalloc reduce remote reduced");
  ck(gpuMalloc(&d_reduce_output, sizeof(float) * kNumElems),
     "gpuMalloc reduce output");
  ck(gpuMalloc(&d_reduce_staging, sizeof(float) * kNumElems),
     "gpuMalloc reduce staging");

  ck(gpuMemcpy(d_gather_remote_input, h_gather_remote_input.data(),
               sizeof(float) * kNumElems, gpuMemcpyHostToDevice),
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

  Compute::CollectiveBuffers gather_buffers{};
  gather_buffers.remote_input = d_gather_remote_input;
  gather_buffers.final_output = d_gather_output;

  Compute::CollectiveBuffers reduce_buffers{};
  reduce_buffers.local_input = d_reduce_local_input;
  reduce_buffers.remote_input = d_reduce_remote_input;
  reduce_buffers.remote_reduced = d_reduce_remote_reduced;
  reduce_buffers.final_output = d_reduce_output;
  reduce_buffers.recv_staging = d_reduce_staging;

  Compute::CollectiveHostApiConfig gather_api_cfg{};
  gather_api_cfg.kernel = &kernel;
  gather_api_cfg.buffers = gather_buffers;
  gather_api_cfg.dtype = Compute::DataType::Fp32;
  gather_api_cfg.num_blocks = kernel_cfg.numBlocks;
  gather_api_cfg.enable_copy_engine = true;

  Compute::CollectiveHostApiConfig reduce_api_cfg{};
  reduce_api_cfg.kernel = &kernel;
  reduce_api_cfg.buffers = reduce_buffers;
  reduce_api_cfg.dtype = Compute::DataType::Fp32;
  reduce_api_cfg.reduce_type = Compute::ReduceType::Sum;
  reduce_api_cfg.num_blocks = kernel_cfg.numBlocks;
  reduce_api_cfg.enable_copy_engine = true;

  Compute::CollectiveHostApi gather_api(gather_api_cfg);
  Compute::CollectiveHostApi reduce_api(reduce_api_cfg);

  CCL::CollectiveConfig gather_cfg{};
  gather_cfg.nranks = 2;
  gather_cfg.rank = 0;
  gather_cfg.channels = 2;
  gather_cfg.bytes_per_rank =
      sizeof(float) * static_cast<size_t>(kGatherElemsPerRank);
  gather_cfg.chunk_bytes = gather_cfg.bytes_per_rank / 2;
  gather_cfg.requested_cpu_backend = Compute::CpuBackendKind::Auto;
  gather_cfg.device_caps.has_copy_engine_path = true;
  gather_cfg.cpu_selector.copy_engine_threshold_bytes = 1;

  CCL::CollectiveConfig reduce_cfg{};
  reduce_cfg.nranks = 2;
  reduce_cfg.rank = 0;
  reduce_cfg.channels = 2;
  reduce_cfg.bytes_per_rank = sizeof(float) * static_cast<size_t>(kNumElems);
  reduce_cfg.chunk_bytes = reduce_cfg.bytes_per_rank / 4;
  reduce_cfg.requested_cpu_backend = Compute::CpuBackendKind::Auto;
  reduce_cfg.device_caps.has_copy_engine_path = true;
  reduce_cfg.cpu_selector.copy_engine_threshold_bytes = 1;

  auto gather_handle = gather_api.submit_allgather(gather_cfg);
  gather_api.wait(gather_handle);
  if (gather_api.status(gather_handle) != CCL::CollectiveOpStatus::Completed) {
    std::cerr << "host api allgather failed\n";
    return 2;
  }
  gather_api.release(gather_handle);

  auto reduce_handle = reduce_api.submit_allreduce(reduce_cfg);
  reduce_api.wait(reduce_handle);
  if (reduce_api.status(reduce_handle) != CCL::CollectiveOpStatus::Completed) {
    std::cerr << "host api allreduce failed\n";
    return 3;
  }
  reduce_api.release(reduce_handle);

  kernel.stop();

  ck(gpuMemcpy(h_gather_out.data(), d_gather_output, sizeof(float) * kNumElems,
               gpuMemcpyDeviceToHost),
     "copy gather out");
  ck(gpuMemcpy(h_reduce_out.data(), d_reduce_output, sizeof(float) * kNumElems,
               gpuMemcpyDeviceToHost),
     "copy reduce out");

  size_t gather_bad = 0;
  for (int i = 0; i < kGatherElemsPerRank; ++i) {
    if (!feq(h_gather_out[i], 7.0f + static_cast<float>(i))) ++gather_bad;
    if (!feq(h_gather_out[kGatherElemsPerRank + i],
             17.0f + static_cast<float>(i))) {
      ++gather_bad;
    }
  }
  if (gather_bad) {
    std::cerr << "host api allgather FAILED mismatches=" << gather_bad << "\n";
    return 4;
  }

  size_t reduce_bad = 0;
  for (int i = 0; i < kNumElems; ++i) {
    float expected = h_reduce_local_input[i] +
                     (1.0f + 0.25f * static_cast<float>(i));
    if (!feq(h_reduce_out[i], expected)) {
      if (reduce_bad < 8) {
        std::cerr << "[HOST API ALLREDUCE MISMATCH] i=" << i << " got="
                  << h_reduce_out[i] << " exp=" << expected << "\n";
      }
      ++reduce_bad;
    }
  }
  if (reduce_bad) {
    std::cerr << "host api allreduce FAILED mismatches=" << reduce_bad << "\n";
    return 5;
  }

  std::cout << "CCL host API allgather PASSED\n";
  std::cout << "CCL host API allreduce PASSED\n";

  ck(gpuFree(d_gather_remote_input), "free gather remote input");
  ck(gpuFree(d_gather_output), "free gather output");
  ck(gpuFree(d_reduce_local_input), "free reduce local input");
  ck(gpuFree(d_reduce_remote_input), "free reduce remote input");
  ck(gpuFree(d_reduce_remote_reduced), "free reduce remote reduced");
  ck(gpuFree(d_reduce_output), "free reduce output");
  ck(gpuFree(d_reduce_staging), "free reduce staging");
  return 0;
}

#include "../ccl/executor.h"
#include "ccl_backend.h"
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

  Compute::PersistentKernelConfig config;
  config.numBlocks = 2;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 16;

  std::vector<float> h_gather_src(kNumElems, 0.0f);
  std::vector<float> h_gather_dst(kNumElems, 0.0f);
  std::vector<float> h_gather_out(kNumElems, 0.0f);
  for (int i = 0; i < kGatherElemsPerRank; ++i) {
    h_gather_dst[i] = 10.0f + static_cast<float>(i);
    h_gather_src[kGatherElemsPerRank + i] = 30.0f + static_cast<float>(i);
  }

  std::vector<float> h_reduce_src(kNumElems, 0.0f);
  std::vector<float> h_reduce_dst(kNumElems, 0.0f);
  std::vector<float> h_reduce_out(kNumElems, 0.0f);
  fill(h_reduce_dst, 4.0f, 0.5f);
  fill(h_reduce_src, 1.0f, 0.25f);

  float* d_gather_src = nullptr;
  float* d_gather_dst = nullptr;
  float* d_reduce_src = nullptr;
  float* d_reduce_dst = nullptr;
  float* d_reduce_staging = nullptr;
  ck(gpuMalloc(&d_gather_src, sizeof(float) * kNumElems), "gpuMalloc gather src");
  ck(gpuMalloc(&d_gather_dst, sizeof(float) * kNumElems), "gpuMalloc gather dst");
  ck(gpuMalloc(&d_reduce_src, sizeof(float) * kNumElems), "gpuMalloc reduce src");
  ck(gpuMalloc(&d_reduce_dst, sizeof(float) * kNumElems), "gpuMalloc reduce dst");
  ck(gpuMalloc(&d_reduce_staging, sizeof(float) * kNumElems),
     "gpuMalloc reduce staging");

  ck(gpuMemcpy(d_gather_src, h_gather_src.data(), sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy gather src");
  ck(gpuMemcpy(d_gather_dst, h_gather_dst.data(), sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy gather dst");
  ck(gpuMemcpy(d_reduce_src, h_reduce_src.data(), sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy reduce src");
  ck(gpuMemcpy(d_reduce_dst, h_reduce_dst.data(), sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy reduce dst");
  ck(gpuMemset(d_reduce_staging, 0, sizeof(float) * kNumElems),
     "memset reduce staging");

  Compute::PersistentKernel<Compute::Task> kernel(config);
  kernel.launch();

  Compute::ComputeCopyEngineBackend ce_gather_backend(d_gather_dst, d_gather_src);
  Compute::ComputeCopyEngineBackend ce_reduce_backend(
      d_reduce_dst, d_reduce_src, -1, -1, d_reduce_staging);
  Compute::ComputePersistentKernelBackend pk_reduce_backend(
      kernel, d_reduce_dst, d_reduce_src, Compute::DataType::Fp32,
      Compute::ReduceType::Sum, Compute::TransferPath::RegisterOp,
      config.numBlocks, d_reduce_staging);

  CCL::CollectiveConfig gather_cfg{};
  gather_cfg.nranks = 2;
  gather_cfg.rank = 0;
  gather_cfg.channels = 2;
  gather_cfg.bytes_per_rank = sizeof(float) * static_cast<size_t>(kGatherElemsPerRank);
  gather_cfg.chunk_bytes = gather_cfg.bytes_per_rank / 2;
  gather_cfg.requested_cpu_backend = Compute::CpuBackendKind::Auto;
  gather_cfg.device_caps.has_copy_engine_path = true;
  gather_cfg.cpu_selector.copy_engine_threshold_bytes = 1;

  CCL::ExecutorBackends gather_backends{};
  gather_backends.ce = &ce_gather_backend;
  CCL::Executor gather_executor(gather_backends);
  auto gather_handle = gather_executor.submit_allgather(gather_cfg);
  gather_executor.wait(gather_handle);
  if (gather_executor.status(gather_handle) != CCL::CollectiveOpStatus::Completed) {
    std::cerr << "CE allgather executor failed\n";
    return 2;
  }
  gather_executor.release(gather_handle);

  CCL::CollectiveConfig reduce_cfg{};
  reduce_cfg.nranks = 2;
  reduce_cfg.rank = 0;
  reduce_cfg.channels = 2;
  reduce_cfg.bytes_per_rank = sizeof(float) * static_cast<size_t>(kNumElems);
  reduce_cfg.chunk_bytes = reduce_cfg.bytes_per_rank / 4;
  reduce_cfg.requested_cpu_backend = Compute::CpuBackendKind::Auto;
  reduce_cfg.device_caps.has_copy_engine_path = true;
  reduce_cfg.cpu_selector.copy_engine_threshold_bytes = 1;

  CCL::ExecutorBackends reduce_backends{};
  reduce_backends.ce = &ce_reduce_backend;
  reduce_backends.persistent = &pk_reduce_backend;
  CCL::Executor reduce_executor(reduce_backends);
  auto reduce_handle = reduce_executor.submit_allreduce(reduce_cfg);
  reduce_executor.wait(reduce_handle);
  if (reduce_executor.status(reduce_handle) != CCL::CollectiveOpStatus::Completed) {
    std::cerr << "CE allreduce executor failed\n";
    return 3;
  }
  reduce_executor.release(reduce_handle);

  kernel.stop();

  ck(gpuMemcpy(h_gather_out.data(), d_gather_dst, sizeof(float) * kNumElems,
               gpuMemcpyDeviceToHost),
     "copy gather out");
  ck(gpuMemcpy(h_reduce_out.data(), d_reduce_dst, sizeof(float) * kNumElems,
               gpuMemcpyDeviceToHost),
     "copy reduce out");

  size_t gather_bad = 0;
  for (int i = 0; i < kGatherElemsPerRank; ++i) {
    if (!feq(h_gather_out[i], 10.0f + static_cast<float>(i))) ++gather_bad;
    if (!feq(h_gather_out[kGatherElemsPerRank + i],
             30.0f + static_cast<float>(i))) {
      ++gather_bad;
    }
  }
  if (gather_bad) {
    std::cerr << "CE allgather FAILED mismatches=" << gather_bad << "\n";
    return 4;
  }

  size_t reduce_bad = 0;
  for (int i = 0; i < kNumElems; ++i) {
    float expected = (4.0f + 0.5f * static_cast<float>(i)) +
                     (1.0f + 0.25f * static_cast<float>(i));
    if (!feq(h_reduce_out[i], expected)) {
      if (reduce_bad < 8) {
        std::cerr << "[CE ALLREDUCE MISMATCH] i=" << i << " got="
                  << h_reduce_out[i] << " exp=" << expected << "\n";
      }
      ++reduce_bad;
    }
  }
  if (reduce_bad) {
    std::cerr << "CE allreduce FAILED mismatches=" << reduce_bad << "\n";
    return 5;
  }

  std::cout << "CCL copy-engine allgather PASSED\n";
  std::cout << "CCL copy-engine allreduce PASSED\n";

  ck(gpuFree(d_gather_src), "free gather src");
  ck(gpuFree(d_gather_dst), "free gather dst");
  ck(gpuFree(d_reduce_src), "free reduce src");
  ck(gpuFree(d_reduce_dst), "free reduce dst");
  ck(gpuFree(d_reduce_staging), "free reduce staging");
  return 0;
}

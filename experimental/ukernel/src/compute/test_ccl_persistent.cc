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

UKernel::CCL::CollectivePlan make_plan(size_t bytes) {
  using namespace UKernel::CCL;

  CollectivePlan plan;
  plan.collective = CollectiveKind::AllReduce;
  plan.algorithm = AlgorithmKind::Ring;
  plan.nranks = 1;
  plan.rank = 0;
  plan.channels = 2;
  plan.bytes_per_rank = bytes;
  plan.chunk_bytes = bytes / 2;

  CollectiveStep copy_step;
  copy_step.step_id = 0;
  copy_step.phase = StepPhase::DirectCopy;
  copy_step.src_rank = 0;
  copy_step.dst_rank = 0;
  copy_step.chunk = ChunkRange{0, 0, 0, 0, bytes / 2};
  copy_step.ops.push_back(
      ExecutionOp{0, ExecutionOpKind::PkCopy, 0, 0, copy_step.chunk, {}});
  plan.steps.push_back(copy_step);

  CollectiveStep reduce_step;
  reduce_step.step_id = 1;
  reduce_step.phase = StepPhase::ReduceScatter;
  reduce_step.src_rank = 0;
  reduce_step.dst_rank = 0;
  reduce_step.predecessors = {0};
  reduce_step.chunk = ChunkRange{0, 1, 1, bytes / 2, bytes / 2};
  reduce_step.ops.push_back(
      ExecutionOp{1, ExecutionOpKind::PkReduce, 0, 0, reduce_step.chunk, {}});
  plan.steps.push_back(reduce_step);

  return plan;
}

}  // namespace

int main() {
  using namespace UKernel;

  Compute::TaskManager::instance().init(1024, 256);

  Compute::PersistentKernelConfig config;
  config.numBlocks = 2;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 16;
  config.smemSize = 0;

  std::vector<float> h_src(kNumElems);
  std::vector<float> h_dst(kNumElems);
  std::vector<float> h_out(kNumElems, 0.0f);
  fill(h_src, 1.0f, 0.25f);
  fill(h_dst, 10.0f, 0.5f);

  float* d_src = nullptr;
  float* d_dst = nullptr;
  ck(gpuMalloc(&d_src, sizeof(float) * kNumElems), "gpuMalloc d_src");
  ck(gpuMalloc(&d_dst, sizeof(float) * kNumElems), "gpuMalloc d_dst");
  ck(gpuMemcpy(d_src, h_src.data(), sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy src");
  ck(gpuMemcpy(d_dst, h_dst.data(), sizeof(float) * kNumElems,
               gpuMemcpyHostToDevice),
     "copy dst");

  Compute::PersistentKernel<Compute::Task> kernel(config);
  kernel.launch();

  Compute::ComputePersistentKernelBackend pk_backend(
      kernel, d_dst, d_src, Compute::DataType::Fp32, Compute::ReduceType::Sum,
      Compute::TransferPath::RegisterOp, config.numBlocks);

  CCL::ExecutorBackends backends{};
  backends.persistent = &pk_backend;

  CCL::Executor executor(backends);
  CCL::CollectiveOpHandle handle =
      executor.submit(make_plan(sizeof(float) * kNumElems));
  executor.wait(handle);
  if (executor.status(handle) != CCL::CollectiveOpStatus::Completed) {
    std::cerr << "executor failed\n";
    return 2;
  }
  executor.release(handle);

  kernel.stop();

  ck(gpuMemcpy(h_out.data(), d_dst, sizeof(float) * kNumElems,
               gpuMemcpyDeviceToHost),
     "copy out");

  size_t bad = 0;
  size_t half = static_cast<size_t>(kNumElems / 2);
  for (size_t i = 0; i < half; ++i) {
    if (!feq(h_out[i], h_src[i])) {
      if (bad < 8) {
        std::cerr << "[COPY MISMATCH] i=" << i << " got=" << h_out[i]
                  << " exp=" << h_src[i] << "\n";
      }
      ++bad;
    }
  }
  for (size_t i = half; i < static_cast<size_t>(kNumElems); ++i) {
    float expected = h_dst[i] + h_src[i];
    if (!feq(h_out[i], expected)) {
      if (bad < 8) {
        std::cerr << "[REDUCE MISMATCH] i=" << i << " got=" << h_out[i]
                  << " exp=" << expected << "\n";
      }
      ++bad;
    }
  }
  if (bad) {
    std::cerr << "CCL persistent backend FAILED mismatches=" << bad << "\n";
    return 3;
  }

  std::cout << "CCL persistent backend PASSED\n";

  ck(gpuFree(d_src), "free src");
  ck(gpuFree(d_dst), "free dst");
  return 0;
}

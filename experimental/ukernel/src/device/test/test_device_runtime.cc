#include "../persistent.h"
#include "test_support.h"
#include <iostream>
#include <vector>

#define N 1024

uint64_t submit_copy_task(
    UKernel::Device::PersistentKernel<UKernel::Device::Task>& kernel,
    void* dst, void const* src, uint64_t bytes,
    UKernel::Device::DataType dtype, uint32_t block_id,
    UKernel::Device::TransferPath path =
        UKernel::Device::TransferPath::Auto,
    uint64_t op_id = 0, uint32_t step_id = 0, uint32_t chunk_id = 0) {
  UKernel::Device::CollArgs h{};
  h.src = const_cast<void*>(src);
  h.src2 = nullptr;
  h.dst = dst;
  h.bytes = bytes;
  h.op_id = op_id;
  h.step_id = step_id;
  h.chunk_id = chunk_id;
  h.completion_cookie = chunk_id;
  h.src_rank = 0;
  h.dst_rank = 0;
  h.src_device = 0;
  h.dst_device = 0;
  h.flags = 0;
  h.redType = UKernel::Device::ReduceType::None;
  h.requested_path = path;
  h.resolved_path = UKernel::Device::TransferPath::Auto;

  UKernel::Device::Task t =
      UKernel::Device::TaskManager::instance().create_coll_task(
          h, UKernel::Device::TaskType::CollCopy, dtype, block_id);

  return kernel.submit(t);
}

uint64_t submit_reduce_task(
    UKernel::Device::PersistentKernel<UKernel::Device::Task>& kernel,
    void* dst, void const* src, uint64_t bytes,
    UKernel::Device::DataType dtype, UKernel::Device::ReduceType redop,
    uint32_t block_id,
    UKernel::Device::TransferPath path =
        UKernel::Device::TransferPath::Auto,
    uint64_t op_id = 0, uint32_t step_id = 0, uint32_t chunk_id = 0) {
  UKernel::Device::CollArgs h{};
  h.src = const_cast<void*>(src);
  h.src2 = nullptr;
  h.dst = dst;
  h.bytes = bytes;
  h.op_id = op_id;
  h.step_id = step_id;
  h.chunk_id = chunk_id;
  h.completion_cookie = chunk_id;
  h.src_rank = 0;
  h.dst_rank = 0;
  h.src_device = 0;
  h.dst_device = 0;
  h.flags = 0;
  h.redType = redop;
  h.requested_path = path;
  h.resolved_path = UKernel::Device::TransferPath::Auto;

  UKernel::Device::Task t =
      UKernel::Device::TaskManager::instance().create_coll_task(
          h, UKernel::Device::TaskType::CollReduce, dtype, block_id);

  return kernel.submit(t);
}

int main() {
  using UKernel::Device::Testing::ck;
  using UKernel::Device::Testing::feq;
  using UKernel::Device::Testing::fill;

  UKernel::Device::TaskManager::instance().init(1024);

  auto caps = UKernel::Device::query_transfer_capabilities();
  UKernel::Device::PkSelectorConfig selector_cfg{};
  uint64_t auto_select_bytes = 32768;
  auto auto_path = UKernel::Device::resolve_pk_transfer_path(
      UKernel::Device::TransferPath::Auto, auto_select_bytes, caps,
      selector_cfg);
  auto reg_path = UKernel::Device::resolve_pk_transfer_path(
      UKernel::Device::TransferPath::RegisterOp, N * sizeof(float), caps,
      selector_cfg);
  auto tma_path = UKernel::Device::resolve_pk_transfer_path(
      UKernel::Device::TransferPath::TmaOp, N * sizeof(float), caps,
      selector_cfg);
  auto expected_auto_path =
      caps.can_use_tma ? UKernel::Device::TransferPath::TmaOp
                       : UKernel::Device::TransferPath::RegisterOp;
  auto expected_tma_path =
      caps.can_use_tma ? UKernel::Device::TransferPath::TmaOp
                       : UKernel::Device::TransferPath::RegisterOp;
  if (auto_path != expected_auto_path ||
      reg_path != UKernel::Device::TransferPath::RegisterOp ||
      tma_path != expected_tma_path ||
      (caps.can_use_tma &&
       !UKernel::Device::supports_native_tma_copy(
           UKernel::Device::DataType::Fp32, N * sizeof(float)))) {
    std::cerr << "Selector bootstrap FAILED\n";
    return 4;
  }
  std::cout << "Selector bootstrap PASSED\n";

  UKernel::Device::PersistentKernelConfig config;
  config.numBlocks = 3;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 16;
  config.smemSize = 0;

  uint32_t test_block_id = 0;
  uint32_t test_block_id_2 = 1;

  float *dst_copy = nullptr, *src_copy = nullptr;
  float *dst_reduce = nullptr, *src_reduce = nullptr;

  ck(gpuMalloc(&dst_copy, N * sizeof(float)), "gpuMalloc dst_copy");
  ck(gpuMalloc(&src_copy, N * sizeof(float)), "gpuMalloc src_copy");
  ck(gpuMalloc(&dst_reduce, N * sizeof(float)), "gpuMalloc dst_reduce");
  ck(gpuMalloc(&src_reduce, N * sizeof(float)), "gpuMalloc src_reduce");

  std::vector<float> h_src_copy(N), h_dst_copy(N, 0.0f);
  std::vector<float> h_dst0(N), h_src_red(N), h_dst1(N, 0.0f);

  fill(h_src_copy, 1.25f, 0.5f);
  fill(h_dst0, 2.0f, 0.25f);
  fill(h_src_red, -1.0f, 0.125f);

  ck(gpuMemcpy(src_copy, h_src_copy.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D src_copy");
  ck(gpuMemset(dst_copy, 0, N * sizeof(float)), "memset dst_copy");

  ck(gpuMemcpy(dst_reduce, h_dst0.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D dst_reduce");
  ck(gpuMemcpy(src_reduce, h_src_red.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D src_reduce");

  UKernel::Device::PersistentKernel<UKernel::Device::Task> kernel(config);
  kernel.launch();
  std::cout << "Persistent kernel launched.\n";

  uint64_t id =
      submit_copy_task(kernel, dst_copy, src_copy, N * sizeof(float),
                       UKernel::Device::DataType::Fp32, test_block_id,
                       UKernel::Device::TransferPath::Auto, 1, 0, 0);

  while (!kernel.is_done(test_block_id, id)) {
  }
  std::cout << "COPY AUTO DONE\n";

  id = submit_reduce_task(kernel, dst_reduce, src_reduce, N * sizeof(float),
                          UKernel::Device::DataType::Fp32,
                          UKernel::Device::ReduceType::Sum, test_block_id_2,
                          UKernel::Device::TransferPath::Auto, 2, 1, 0);

  while (!kernel.is_done(test_block_id_2, id)) {
  }
  std::cout << "REDUCE DONE\n";

  id = submit_copy_task(kernel, dst_copy, src_copy, N * sizeof(float),
                        UKernel::Device::DataType::Fp32, test_block_id,
                        UKernel::Device::TransferPath::TmaOp, 3, 2, 0);

  while (!kernel.is_done(test_block_id, id)) {
  }
  std::cout << "COPY TMA DONE\n";

  kernel.stop();
  std::cout << "Stop signal sent.\n";

  ck(gpuMemcpy(h_dst_copy.data(), dst_copy, N * sizeof(float),
               gpuMemcpyDeviceToHost),
     "D2H dst_copy");
  ck(gpuMemcpy(h_dst1.data(), dst_reduce, N * sizeof(float),
               gpuMemcpyDeviceToHost),
     "D2H dst_reduce");

  {
    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
      if (!feq(h_dst_copy[i], h_src_copy[i])) {
        if (bad < 8)
          std::cerr << "[COPY MISMATCH] i=" << i << " got=" << h_dst_copy[i]
                    << " exp=" << h_src_copy[i] << "\n";
        ++bad;
      }
    }
    if (bad) {
      std::cerr << "COPY FAILED mismatches=" << bad << "/" << N << "\n";
      return 2;
    }
    std::cout << "COPY PASSED\n";
  }

  {
    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
      float exp = h_dst0[i] + h_src_red[i];
      if (!feq(h_dst1[i], exp)) {
        if (bad < 8)
          std::cerr << "[REDUCE MISMATCH] i=" << i << " got=" << h_dst1[i]
                    << " exp=" << exp << "\n";
        ++bad;
      }
    }
    if (bad) {
      std::cerr << "REDUCE FAILED mismatches=" << bad << "/" << N << "\n";
      return 3;
    }
    std::cout << "REDUCE PASSED\n";
  }

  ck(gpuFree(dst_copy), "gpuFree dst_copy");
  ck(gpuFree(src_copy), "gpuFree src_copy");
  ck(gpuFree(dst_reduce), "gpuFree dst_reduce");
  ck(gpuFree(src_reduce), "gpuFree src_reduce");

  return 0;
}

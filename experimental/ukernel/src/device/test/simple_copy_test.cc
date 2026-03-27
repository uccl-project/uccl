#include "gpu_rt.h"
#include "task.h"
#include "worker.h"
#include <iostream>
#include <vector>

using namespace UKernel::Device;

int main(int argc, char** argv) {
  std::cout << "=== Two Workers Test: Copy + Reduce ===" << std::endl;

  TaskManager::instance().init(256);

  WorkerPool::Config config;
  config.numMaxWorkers = 4;
  config.threadsPerBlock = 256;
  config.fifoCapacity = 16;

  WorkerPool wp(config);

  const size_t bytes = 64 * 1024 * 1024;  // 64MB
  const size_t count = bytes / sizeof(float);

  void* d_src_copy;
  void* d_dst_copy;
  void* d_src_reduce;
  void* d_dst_reduce;
  GPU_RT_CHECK(gpuMalloc(&d_src_copy, bytes));
  GPU_RT_CHECK(gpuMalloc(&d_dst_copy, bytes));
  GPU_RT_CHECK(gpuMalloc(&d_src_reduce, bytes));
  GPU_RT_CHECK(gpuMalloc(&d_dst_reduce, bytes));

  std::vector<char> h_src_copy(bytes);
  std::vector<char> h_dst_copy(bytes, 0);
  std::vector<char> h_dst_reduce(bytes, 0);
  for (size_t i = 0; i < bytes; i++) {
    h_src_copy[i] = static_cast<char>(i);
  }

  std::vector<float> h_src_reduce(count);
  std::vector<float> h_dst_reduce_init(count, 0.0f);
  for (size_t i = 0; i < count; i++) {
    h_src_reduce[i] = static_cast<float>(i + 1);
    h_dst_reduce_init[i] = static_cast<float>(1000 + i);
  }

  GPU_RT_CHECK(
      gpuMemcpy(d_src_copy, h_src_copy.data(), bytes, gpuMemcpyHostToDevice));
  GPU_RT_CHECK(gpuMemset(d_dst_copy, 0, bytes));
  GPU_RT_CHECK(gpuMemcpy(d_src_reduce, h_src_reduce.data(), bytes,
                         gpuMemcpyHostToDevice));
  GPU_RT_CHECK(gpuMemcpy(d_dst_reduce, h_dst_reduce_init.data(), bytes,
                         gpuMemcpyHostToDevice));

  wp.createWorker(0, 1);
  wp.createWorker(1, 1);
  wp.waitWorker(0);
  wp.waitWorker(1);

  TaskArgs args_copy;
  memset(&args_copy, 0, sizeof(args_copy));
  args_copy.src = d_src_copy;
  args_copy.dst = d_dst_copy;
  args_copy.bytes = bytes;

  TaskArgs args_reduce;
  memset(&args_reduce, 0, sizeof(args_reduce));
  args_reduce.src = d_src_reduce;
  args_reduce.dst = d_dst_reduce;
  args_reduce.bytes = bytes;
  args_reduce.set_red_type(ReduceType::Sum);

  Task copy_task = TaskManager::instance().create_task(
      args_copy, TaskType::CollCopy, DataType::Int8, 0);
  Task reduce_task = TaskManager::instance().create_task(
      args_reduce, TaskType::CollReduce, DataType::Fp32, 0);

  uint64_t copy_taskId = wp.enqueue(copy_task, 0);
  uint64_t reduce_taskId = wp.enqueue(reduce_task, 1);
  std::cout << "Enqueued copy task " << copy_taskId << " on fifo 0"
            << std::endl;
  std::cout << "Enqueued reduce task " << reduce_taskId << " on fifo 1"
            << std::endl;

  while (!wp.is_done(copy_taskId, 0)) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  std::cout << "Copy task " << copy_taskId << " done" << std::endl;

  while (!wp.is_done(reduce_taskId, 1)) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  std::cout << "Reduce task " << reduce_taskId << " done" << std::endl;

  GPU_RT_CHECK(
      gpuMemcpy(h_dst_copy.data(), d_dst_copy, bytes, gpuMemcpyDeviceToHost));
  GPU_RT_CHECK(gpuMemcpy(h_dst_reduce.data(), d_dst_reduce, bytes,
                         gpuMemcpyDeviceToHost));

  bool copy_match = true;
  for (size_t i = 0; i < bytes; i++) {
    if (h_dst_copy[i] != h_src_copy[i]) {
      std::cerr << "Copy mismatch at " << i << ": got " << (int)h_dst_copy[i]
                << ", expected " << (int)h_src_copy[i] << std::endl;
      copy_match = false;
      break;
    }
  }

  bool reduce_match = true;
  float* dst_reduce_float = reinterpret_cast<float*>(h_dst_reduce.data());
  for (size_t i = 0; i < count; i++) {
    float expected = h_dst_reduce_init[i] + h_src_reduce[i];
    if (abs(dst_reduce_float[i] - expected) > 0.01f) {
      std::cerr << "Reduce mismatch at " << i << ": got " << dst_reduce_float[i]
                << ", expected " << expected << std::endl;
      reduce_match = false;
      break;
    }
  }

  // std::cout << "Sleeping 5 seconds..." << std::endl;
  // std::this_thread::sleep_for(std::chrono::seconds(5));

  wp.shutdown_all();

  gpuFree(d_src_copy);
  gpuFree(d_dst_copy);
  gpuFree(d_src_reduce);
  gpuFree(d_dst_reduce);
  TaskManager::instance().release();

  GPU_RT_CHECK(gpuDeviceSynchronize());

  if (copy_match && reduce_match) {
    std::cout << "[PASS] Two workers test" << std::endl;
    return 0;
  } else {
    std::cout << "[FAIL] Two workers test" << std::endl;
    return 1;
  }
}

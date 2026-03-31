#include "benchmarks/bench_support.h"
#include "worker.h"
#include "gpu_rt.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using UKernel::Device::Task;
using UKernel::Device::TaskArgs;
using UKernel::Device::TaskManager;
using UKernel::Device::TaskType;
using UKernel::Device::DataType;
using UKernel::Device::ReduceType;
using UKernel::Device::WorkerPool;

enum class BenchOp {
  Nop,
  Copy,
  Reduce,
};

char const* op_name(BenchOp op) {
  switch (op) {
    case BenchOp::Nop:
      return "nop";
    case BenchOp::Copy:
      return "copy";
    case BenchOp::Reduce:
      return "reduce";
  }
  return "unknown";
}

__global__ void bench_empty_kernel() {}

__global__ void bench_copy_kernel(float const* src, float* dst,
                                  size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < count; i += stride) {
    dst[i] = src[i];
  }
}

__global__ void bench_reduce_kernel(float const* src, float* dst,
                                    size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < count; i += stride) {
    dst[i] += src[i];
  }
}

uint64_t enqueue_until_accepted(WorkerPool& pool, Task const& task,
                                uint32_t fifo_id) {
  while (true) {
    uint64_t task_id = pool.enqueue(task, fifo_id);
    if (task_id != WorkerPool::kInvalidTaskId) {
      return task_id;
    }
    std::this_thread::yield();
  }
}

void wait_until_done(WorkerPool& pool, uint64_t task_id, uint32_t fifo_id) {
  while (!pool.is_done(task_id, fifo_id)) {
    std::this_thread::yield();
  }
}

struct DeviceBuffers {
  void* src = nullptr;
  void* dst = nullptr;
  size_t bytes = 0;

  ~DeviceBuffers() {
    if (dst != nullptr) {
      gpuFree(dst);
    }
    if (src != nullptr) {
      gpuFree(src);
    }
  }
};

DeviceBuffers make_buffers(size_t bytes) {
  DeviceBuffers bufs;
  bufs.bytes = bytes;
  GPU_RT_CHECK(gpuMalloc(&bufs.src, bytes));
  GPU_RT_CHECK(gpuMalloc(&bufs.dst, bytes));

  std::vector<float> host(bytes / sizeof(float), 1.0f);
  GPU_RT_CHECK(
      gpuMemcpy(bufs.src, host.data(), bytes, gpuMemcpyHostToDevice));
  GPU_RT_CHECK(gpuMemset(bufs.dst, 0, bytes));
  return bufs;
}

void reset_buffers(BenchOp op, DeviceBuffers const& bufs) {
  if (op == BenchOp::Nop) {
    return;
  }
  GPU_RT_CHECK(gpuMemset(bufs.dst, 0, bufs.bytes));
}

void launch_one_kernel(BenchOp op, DeviceBuffers const& bufs, size_t count) {
  switch (op) {
    case BenchOp::Nop:
      bench_empty_kernel<<<1, 64>>>();
      break;
    case BenchOp::Copy:
      bench_copy_kernel<<<1, 64>>>(
          reinterpret_cast<float const*>(bufs.src),
          reinterpret_cast<float*>(bufs.dst), count);
      break;
    case BenchOp::Reduce:
      bench_reduce_kernel<<<1, 64>>>(
          reinterpret_cast<float const*>(bufs.src),
          reinterpret_cast<float*>(bufs.dst), count);
      break;
  }
}

double run_kernel_launch_path(BenchOp op, int tasks_per_batch, int rounds,
                              int warmup, size_t bytes) {
  DeviceBuffers bufs = make_buffers(bytes);
  size_t count = bytes / sizeof(float);
  reset_buffers(op, bufs);

  for (int i = 0; i < warmup; ++i) {
    for (int j = 0; j < tasks_per_batch; ++j) {
      launch_one_kernel(op, bufs, count);
    }
    GPU_RT_CHECK(gpuDeviceSynchronize());
  }

  reset_buffers(op, bufs);
  uint64_t t0 = now_ns();
  for (int i = 0; i < rounds; ++i) {
    for (int j = 0; j < tasks_per_batch; ++j) {
      launch_one_kernel(op, bufs, count);
    }
    GPU_RT_CHECK(gpuDeviceSynchronize());
  }
  uint64_t t1 = now_ns();
  return (t1 - t0) * 1e-9;
}

Task make_worker_task(BenchOp op, DeviceBuffers const& bufs) {
  switch (op) {
    case BenchOp::Nop:
      return Task(TaskType::BenchNop, DataType::Fp32, /*block=*/0, /*args=*/0);
    case BenchOp::Copy: {
      TaskArgs args{};
      args.src = bufs.src;
      args.dst = bufs.dst;
      args.bytes = bufs.bytes;
      return TaskManager::instance().create_task(args, TaskType::CollCopy,
                                                 DataType::Fp32, 0);
    }
    case BenchOp::Reduce: {
      TaskArgs args{};
      args.src = bufs.src;
      args.dst = bufs.dst;
      args.bytes = bufs.bytes;
      args.set_red_type(ReduceType::Sum);
      return TaskManager::instance().create_task(args, TaskType::CollReduce,
                                                 DataType::Fp32, 0);
    }
  }
  return Task(TaskType::BenchNop, DataType::Fp32, 0, 0);
}

double run_persistent_worker_path(BenchOp op, int tasks_per_batch, int rounds,
                                  int warmup, size_t bytes) {
  TaskManager::instance().init(1);
  DeviceBuffers bufs = make_buffers(bytes);

  WorkerPool::Config cfg;
  cfg.numMaxWorkers = 1;
  cfg.threadsPerBlock = 64;
  cfg.fifoCapacity = std::max(16, tasks_per_batch * 2);

  WorkerPool pool(cfg);
  if (!pool.createWorker(0, 1)) {
    throw std::runtime_error("failed to create persistent worker");
  }
  pool.waitWorker(0);

  Task task = make_worker_task(op, bufs);
  reset_buffers(op, bufs);

  for (int i = 0; i < warmup; ++i) {
    uint64_t last_id = 0;
    for (int j = 0; j < tasks_per_batch; ++j) {
      last_id = enqueue_until_accepted(pool, task, 0);
    }
    wait_until_done(pool, last_id, 0);
  }

  reset_buffers(op, bufs);
  uint64_t t0 = now_ns();
  for (int i = 0; i < rounds; ++i) {
    uint64_t last_id = 0;
    for (int j = 0; j < tasks_per_batch; ++j) {
      last_id = enqueue_until_accepted(pool, task, 0);
    }
    wait_until_done(pool, last_id, 0);
  }
  uint64_t t1 = now_ns();

  pool.shutdown_all();
  if (op != BenchOp::Nop) {
    TaskManager::instance().free_task_args(task.args_index());
  }
  TaskManager::instance().release();
  return (t1 - t0) * 1e-9;
}

void print_result(char const* label, int tasks_per_batch, int rounds,
                  double seconds) {
  double total_tasks = static_cast<double>(tasks_per_batch) * rounds;
  double batch_us = seconds * 1e6 / rounds;
  double task_us = seconds * 1e6 / total_tasks;
  double tasks_per_sec = total_tasks / seconds;

  std::printf("[%s]\n", label);
  std::printf("  Total time      : %.6f s\n", seconds);
  std::printf("  Batch latency   : %.3f us\n", batch_us);
  std::printf("  Task latency    : %.3f us/task\n", task_us);
  std::printf("  Task throughput : %.2f K tasks/s\n", tasks_per_sec / 1e3);
}

}  // namespace

int main(int argc, char** argv) {
  int tasks_per_batch = 100;
  int rounds = 1000;
  int warmup = 100;
  size_t bytes = 4096;

  if (argc >= 2) tasks_per_batch = std::max(1, std::atoi(argv[1]));
  if (argc >= 3) rounds = std::max(1, std::atoi(argv[2]));
  if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
  if (argc >= 5) bytes = std::max<size_t>(sizeof(float), std::atoi(argv[4]));
  bytes = (bytes / sizeof(float)) * sizeof(float);

  std::printf("Kernel Launch vs Persistent Worker\n");
  std::printf("Tasks per batch: %d\n", tasks_per_batch);
  std::printf("Rounds         : %d\n", rounds);
  std::printf("Warmup rounds  : %d\n", warmup);
  std::printf("Payload bytes  : %zu\n", bytes);

  GPU_RT_CHECK(gpuFree(0));
  GPU_RT_CHECK(gpuDeviceSynchronize());

  for (BenchOp op : {BenchOp::Nop, BenchOp::Copy, BenchOp::Reduce}) {
    double launch_seconds =
        run_kernel_launch_path(op, tasks_per_batch, rounds, warmup, bytes);
    double worker_seconds =
        run_persistent_worker_path(op, tasks_per_batch, rounds, warmup, bytes);

    std::printf("\n=== %s ===\n", op_name(op));
    print_result("Launch kernels", tasks_per_batch, rounds, launch_seconds);
    print_result("Persistent worker", tasks_per_batch, rounds, worker_seconds);
    std::printf("Speedup (launch / worker): %.2fx\n",
                launch_seconds / worker_seconds);
  }
  return 0;
}

#include "benchmarks/bench_support.h"
#include "gpu_rt.h"
#include "persistent_kernel_ops.h"
#include "worker.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using UKernel::Device::DataType;
using UKernel::Device::ReduceType;
using UKernel::Device::Task;
using UKernel::Device::TaskArgs;
using UKernel::Device::TaskManager;
using UKernel::Device::TaskType;
using UKernel::Device::WorkerPool;

// Zero a device buffer with a kernel instead of cudaMemset. On the L40S
// nodes the copy-engine memset's writes do not reliably drain before a
// subsequent worker reduce's read-modify-write: residual 0-writes land
// after the reduce's store and the first round silently loses those
// elements (measured ~2% sparse 128B-line losses, 64MB-segment pattern).
// A kernel's writes are ordered by kernel completion, so zeroing this
// way is race-free. If you must use cudaMemset here, insert a short
// delay (~200ms) after it before running the worker.
__global__ void zero_f32_kernel(float* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) p[i] = 0.0f;
}

void zero_buffer(void* ptr, size_t bytes) {
  size_t n = bytes / sizeof(float);
  unsigned blocks = static_cast<unsigned>((n + 255) / 256);
  if (blocks == 0) blocks = 1;
  zero_f32_kernel<<<blocks, 256>>>(static_cast<float*>(ptr), n);
  GPU_RT_CHECK(gpuGetLastError());
  GPU_RT_CHECK(gpuDeviceSynchronize());
}

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

uint64_t enqueue_batch_until_accepted(WorkerPool& pool,
                                      std::vector<Task> const& tasks,
                                      uint32_t fifo_id) {
  while (true) {
    uint64_t task_id = pool.enqueue_batch(tasks, fifo_id);
    if (task_id != WorkerPool::kInvalidTaskId) {
      return task_id;
    }
    std::this_thread::yield();
  }
}

struct DeviceBuffers {
  void* src = nullptr;
  void* dst = nullptr;
  size_t bytes = 0;

  DeviceBuffers() = default;
  DeviceBuffers(DeviceBuffers const&) = delete;
  DeviceBuffers& operator=(DeviceBuffers const&) = delete;

  DeviceBuffers(DeviceBuffers&& other) noexcept
      : src(other.src), dst(other.dst), bytes(other.bytes) {
    other.src = nullptr;
    other.dst = nullptr;
    other.bytes = 0;
  }

  ~DeviceBuffers() {
    if (dst != nullptr) {
      gpuFree(dst);
    }
    if (src != nullptr) {
      gpuFree(src);
    }
  }
};

void verify_reduce_result(DeviceBuffers const& bufs, char const* label,
                          float expected);

TaskArgs make_launch_args(BenchOp op, DeviceBuffers const& bufs) {
  TaskArgs args{};
  args.src = bufs.src;
  args.dst = bufs.dst;
  args.bytes = bufs.bytes;
  if (op == BenchOp::Reduce) {
    args.set_red_type(ReduceType::Sum);
  }
  return args;
}

void quiesce_device() { GPU_RT_CHECK(gpuDeviceSynchronize()); }

DeviceBuffers make_buffers(size_t bytes) {
  DeviceBuffers bufs;
  bufs.bytes = bytes;
  GPU_RT_CHECK(gpuMalloc(&bufs.src, bytes));
  GPU_RT_CHECK(gpuMalloc(&bufs.dst, bytes));

  std::vector<float> host(bytes / sizeof(float), 1.0f);
  GPU_RT_CHECK(gpuMemcpy(bufs.src, host.data(), bytes, gpuMemcpyHostToDevice));
  zero_buffer(bufs.dst, bytes);
  return bufs;
}

void reset_buffers(BenchOp op, DeviceBuffers const& bufs) {
  if (op == BenchOp::Nop) {
    return;
  }
  zero_buffer(bufs.dst, bufs.bytes);
}

void launch_one_kernel(BenchOp op, TaskArgs const& args, uint32_t num_blocks,
                       uint32_t threads_per_block) {
  switch (op) {
    case BenchOp::Nop:
      UKernel::Device::
          benchDispatchNopKernel<<<num_blocks, threads_per_block>>>();
      break;
    case BenchOp::Copy:
      UKernel::Device::
          benchDispatchCopyFp32Kernel<<<num_blocks, threads_per_block>>>(args);
      break;
    case BenchOp::Reduce:
      UKernel::Device::
          benchDispatchReduceFp32Kernel<<<num_blocks, threads_per_block>>>(
              args);
      break;
  }
}

double run_kernel_launch_path(BenchOp op, int tasks_per_batch, int rounds,
                              int warmup, size_t bytes, uint32_t num_blocks,
                              uint32_t threads_per_block) {
  DeviceBuffers bufs = make_buffers(bytes);
  TaskArgs args = make_launch_args(op, bufs);
  reset_buffers(op, bufs);

  for (int i = 0; i < warmup; ++i) {
    for (int j = 0; j < tasks_per_batch; ++j) {
      launch_one_kernel(op, args, num_blocks, threads_per_block);
    }
    GPU_RT_CHECK(gpuDeviceSynchronize());
  }

  reset_buffers(op, bufs);
  uint64_t t0 = now_ns();
  for (int i = 0; i < rounds; ++i) {
    for (int j = 0; j < tasks_per_batch; ++j) {
      launch_one_kernel(op, args, num_blocks, threads_per_block);
    }
    GPU_RT_CHECK(gpuDeviceSynchronize());
  }
  uint64_t t1 = now_ns();
  if (op == BenchOp::Reduce)
    verify_reduce_result(bufs, "launch",
                         static_cast<float>(rounds * tasks_per_batch));
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
                                  int warmup, size_t bytes, uint32_t num_blocks,
                                  uint32_t threads_per_block,
                                  uint32_t smem_size) {
  TaskManager::instance().init(1);
  DeviceBuffers bufs =
      (op != BenchOp::Nop) ? make_buffers(bytes) : DeviceBuffers{};

  WorkerPool::Config cfg;
  cfg.numMaxWorkers = 1;
  cfg.threadsPerBlock = threads_per_block;
  cfg.fifoCapacity = 1024;
  cfg.smemSize = smem_size;
  // The bench is not an app: keep the worker always resident (pool.sync is
  // the completion signal; idle-exit + relaunch cycles trip a CUDA 13.3
  // context hang in the bench's rapid create/destroy pattern).
  cfg.idleExitAfterUs = 1000000;

  WorkerPool pool(cfg);
  if (!pool.createWorker(0, num_blocks)) {
    throw std::runtime_error("failed to create persistent worker");
  }
  pool.waitWorker(0);

  Task task = make_worker_task(op, bufs);
  if (op != BenchOp::Nop) {
    reset_buffers(op, bufs);
  }

  for (int i = 0; i < warmup; ++i) {
    uint64_t last_id = 0;
    for (int j = 0; j < tasks_per_batch; ++j) {
      last_id = enqueue_until_accepted(pool, task, 0);
    }
    pool.sync(last_id, 0);
  }

  if (op != BenchOp::Nop) {
    reset_buffers(op, bufs);
  }
  uint64_t t0 = now_ns();
  for (int i = 0; i < rounds; ++i) {
    uint64_t last_id = 0;
    for (int j = 0; j < tasks_per_batch; ++j) {
      last_id = enqueue_until_accepted(pool, task, 0);
    }
    pool.sync(last_id, 0);
  }
  uint64_t t1 = now_ns();

  if (op == BenchOp::Reduce)
    verify_reduce_result(bufs, "worker-single", static_cast<float>(rounds));
  pool.shutdown_all();
  if (op != BenchOp::Nop) {
    TaskManager::instance().free_task_args(task.args_index());
  }
  TaskManager::instance().release();
  return (t1 - t0) * 1e-9;
}

double run_persistent_worker_batch_path(BenchOp op, int tasks_per_batch,
                                        int rounds, int warmup, size_t bytes,
                                        uint32_t num_blocks,
                                        uint32_t threads_per_block,
                                        uint32_t smem_size) {
  TaskManager::instance().init(1);
  DeviceBuffers bufs =
      (op != BenchOp::Nop) ? make_buffers(bytes) : DeviceBuffers{};

  WorkerPool::Config cfg;
  cfg.numMaxWorkers = 1;
  cfg.threadsPerBlock = threads_per_block;
  cfg.fifoCapacity = 1024;
  cfg.smemSize = smem_size;

  if (static_cast<size_t>(tasks_per_batch) > cfg.fifoCapacity) {
    throw std::runtime_error(
        "tasks_per_batch exceeds fifoCapacity for true batch enqueue");
  }

  WorkerPool pool(cfg);
  if (!pool.createWorker(0, num_blocks)) {
    throw std::runtime_error("failed to create persistent worker");
  }
  pool.waitWorker(0);

  Task task = make_worker_task(op, bufs);
  std::vector<Task> batch(static_cast<size_t>(tasks_per_batch), task);
  if (op != BenchOp::Nop) {
    reset_buffers(op, bufs);
  }

  for (int i = 0; i < warmup; ++i) {
    uint64_t first_id = enqueue_batch_until_accepted(pool, batch, 0);
    pool.sync(first_id + batch.size() - 1, 0);
  }

  if (op != BenchOp::Nop) {
    reset_buffers(op, bufs);
  }
  uint64_t t0 = now_ns();
  for (int i = 0; i < rounds; ++i) {
    uint64_t first_id = enqueue_batch_until_accepted(pool, batch, 0);
    pool.sync(first_id + batch.size() - 1, 0);
  }
  uint64_t t1 = now_ns();

  if (op == BenchOp::Reduce)
    verify_reduce_result(bufs, "worker-batch",
                         static_cast<float>(rounds * tasks_per_batch));
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

void verify_reduce_result(DeviceBuffers const& bufs, char const* label,
                          float expected) {
  std::vector<float> host(bufs.bytes / sizeof(float));
  GPU_RT_CHECK(gpuMemcpy(host.data(), bufs.dst, bufs.bytes,
                         gpuMemcpyDeviceToHost));
  size_t wrong = 0;
  size_t first = static_cast<size_t>(-1);
  for (size_t i = 0; i < host.size(); ++i) {
    if (host[i] != expected) {
      ++wrong;
      if (first == static_cast<size_t>(-1)) first = i;
      if (wrong <= 16)
        std::printf("[check %s] wrong[%zu] idx=%zu val=%f\n", label, wrong, i,
                    host[i]);
    }
  }
  std::printf("[check %s] wrong=%zu first=%zu\n", label, wrong, first);
}

}  // namespace

int main(int argc, char** argv) {
  int tasks_per_batch = 100;
  int rounds = 1000;
  int warmup = 100;
  size_t bytes = 4096;
  uint32_t num_blocks = 1;
  uint32_t threads_per_block = 64;
  uint32_t smem_size = 0;

  if (argc >= 2) tasks_per_batch = std::max(1, std::atoi(argv[1]));
  if (argc >= 3) rounds = std::max(1, std::atoi(argv[2]));
  if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
  if (argc >= 5) bytes = std::max<size_t>(sizeof(float), std::atoi(argv[4]));
  if (argc >= 6)
    num_blocks = static_cast<uint32_t>(std::max(1, std::atoi(argv[5])));
  if (argc >= 7)
    threads_per_block = static_cast<uint32_t>(std::max(1, std::atoi(argv[6])));
  if (argc >= 8)
    smem_size = static_cast<uint32_t>(std::max(0, std::atoi(argv[7])));
  bytes = (bytes / sizeof(float)) * sizeof(float);

  std::printf("Kernel Launch vs Persistent Worker\n");
  std::printf("Tasks per batch: %d\n", tasks_per_batch);
  std::printf("Rounds         : %d\n", rounds);
  std::printf("Warmup rounds  : %d\n", warmup);
  std::printf("Payload bytes  : %zu\n", bytes);
  std::printf("Blocks/grid    : %u\n", num_blocks);
  std::printf("Threads/block  : %u\n", threads_per_block);
  std::printf("Shared memory  : %u\n", smem_size);

  GPU_RT_CHECK(gpuFree(0));
  GPU_RT_CHECK(gpuDeviceSynchronize());

  for (BenchOp op : {BenchOp::Nop, BenchOp::Copy, BenchOp::Reduce}) {
    std::printf("\n=== %s ===\n", op_name(op));
    std::printf("Running persistent worker (single enqueue)...\n");
    double worker_single_seconds =
        run_persistent_worker_path(op, tasks_per_batch, rounds, warmup, bytes,
                                   num_blocks, threads_per_block, smem_size);
    print_result("Persistent worker (single enqueue)", tasks_per_batch, rounds,
                 worker_single_seconds);
    quiesce_device();

    std::printf("Running persistent worker (batch enqueue)...\n");
    double worker_batch_seconds = run_persistent_worker_batch_path(
        op, tasks_per_batch, rounds, warmup, bytes, num_blocks,
        threads_per_block, smem_size);
    print_result("Persistent worker (batch enqueue)", tasks_per_batch, rounds,
                 worker_batch_seconds);

    quiesce_device();
    std::printf("Running launch path (single stream)...\n");
    double launch_seconds =
        run_kernel_launch_path(op, tasks_per_batch, rounds, warmup, bytes,
                               num_blocks, threads_per_block);
    print_result("Launch kernels (single stream)", tasks_per_batch, rounds,
                 launch_seconds);
    quiesce_device();

    std::printf("Speedup (launch / worker single): %.2fx\n",
                launch_seconds / worker_single_seconds);
    std::printf("Speedup (launch / worker batch) : %.2fx\n",
                launch_seconds / worker_batch_seconds);
  }
  return 0;
}

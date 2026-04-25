#include "test_utils.h"
#include <vector>

namespace UKernel {
namespace Device {

namespace {

using TestUtil::default_config;
using TestUtil::DeviceBuffer;
using TestUtil::download_vector;
using TestUtil::require;
using TestUtil::run_case;
using TestUtil::TaskManagerScope;
using TestUtil::upload_vector;
using TestUtil::verify_bytes;
using TestUtil::verify_floats;
using TestUtil::wait_until_done;
using TestUtil::zero_buffer;

TaskArgs make_copy_args(void* src, void* dst, size_t bytes) {
  TaskArgs args{};
  args.src = src;
  args.dst = dst;
  args.bytes = bytes;
  return args;
}

TaskArgs make_reduce_args(void* src, void* dst, size_t bytes) {
  TaskArgs args = make_copy_args(src, dst, bytes);
  args.set_red_type(ReduceType::Sum);
  return args;
}

void test_two_workers_copy_and_reduce() {
  TaskManagerScope scope(64);
  WorkerPool pool(default_config(/*max_workers=*/4, /*fifo_capacity=*/16));

  constexpr size_t kBytes = 64 << 20;
  constexpr size_t kCount = kBytes / sizeof(float);

  DeviceBuffer copy_src(kBytes);
  DeviceBuffer copy_dst(kBytes);
  DeviceBuffer reduce_src(kBytes);
  DeviceBuffer reduce_dst(kBytes);

  std::vector<char> copy_host(kBytes);
  for (size_t i = 0; i < copy_host.size(); ++i) {
    copy_host[i] = static_cast<char>(i);
  }
  std::vector<float> reduce_src_host(kCount);
  std::vector<float> reduce_dst_init(kCount);
  std::vector<float> reduce_expected(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    reduce_src_host[i] = static_cast<float>(i + 1);
    reduce_dst_init[i] = static_cast<float>(1000 + i);
    reduce_expected[i] = reduce_dst_init[i] + reduce_src_host[i];
  }

  upload_vector(copy_src.ptr, copy_host);
  zero_buffer(copy_dst.ptr, kBytes);
  upload_vector(reduce_src.ptr, reduce_src_host);
  upload_vector(reduce_dst.ptr, reduce_dst_init);

  require(pool.createWorker(0, 1), "copy worker creation failed");
  require(pool.createWorker(1, 1), "reduce worker creation failed");
  pool.waitWorker(0);
  pool.waitWorker(1);

  Task copy_task = TaskManager::instance().create_task(
      make_copy_args(copy_src.ptr, copy_dst.ptr, kBytes), TaskType::CollCopy,
      DataType::Int8, 0);
  Task reduce_task = TaskManager::instance().create_task(
      make_reduce_args(reduce_src.ptr, reduce_dst.ptr, kBytes),
      TaskType::CollReduce, DataType::Fp32, 0);

  uint64_t copy_task_id = pool.enqueue(copy_task, 0);
  uint64_t reduce_task_id = pool.enqueue(reduce_task, 1);
  require(copy_task_id != WorkerPool::kInvalidTaskId, "copy enqueue failed");
  require(reduce_task_id != WorkerPool::kInvalidTaskId,
          "reduce enqueue failed");

  wait_until_done(pool, copy_task_id, 0, std::chrono::seconds(10));
  wait_until_done(pool, reduce_task_id, 1, std::chrono::seconds(10));

  verify_bytes(download_vector<char>(copy_dst.ptr, kBytes), copy_host,
               "two-worker copy");
  verify_floats(download_vector<float>(reduce_dst.ptr, kBytes), reduce_expected,
                "two-worker reduce");
  pool.shutdown_all();
}

void test_same_flow_reduce_pipeline() {
  TaskManagerScope scope(64);
  WorkerPool pool(default_config(/*max_workers=*/2, /*fifo_capacity=*/16));

  constexpr size_t kTileElems = (64 << 10) / sizeof(float);
  constexpr size_t kTiles = 4;
  constexpr size_t kElems = kTileElems * kTiles;
  constexpr size_t kBytes = kElems * sizeof(float);

  DeviceBuffer src(kBytes);
  DeviceBuffer dst(kBytes);
  std::vector<float> src_host(kElems);
  std::vector<float> dst_init(kElems);
  std::vector<float> expected(kElems);
  for (size_t i = 0; i < kElems; ++i) {
    src_host[i] = static_cast<float>(1000 + 2 * i);
    dst_init[i] = static_cast<float>(2000 + i);
    expected[i] = src_host[i] + dst_init[i];
  }

  upload_vector(src.ptr, src_host);
  upload_vector(dst.ptr, dst_init);

  require(pool.createWorker(0, 1), "pipeline worker creation failed");
  pool.waitWorker(0);

  std::vector<uint64_t> task_ids;
  task_ids.reserve(kTiles);
  for (size_t tile = 0; tile < kTiles; ++tile) {
    size_t offset = tile * kTileElems * sizeof(float);
    TaskArgs args = make_reduce_args(static_cast<char*>(src.ptr) + offset,
                                     static_cast<char*>(dst.ptr) + offset,
                                     kTileElems * sizeof(float));
    Task task = TaskManager::instance().create_task(args, TaskType::CollReduce,
                                                    DataType::Fp32, 0);
    uint64_t task_id = pool.enqueue(task, 0);
    require(task_id != WorkerPool::kInvalidTaskId, "pipeline enqueue failed");
    task_ids.push_back(task_id);
  }

  for (uint64_t task_id : task_ids) {
    wait_until_done(pool, task_id, 0, std::chrono::seconds(10));
  }

  verify_floats(download_vector<float>(dst.ptr, kBytes), expected,
                "same-flow reduce pipeline");
  pool.shutdown_all();
}

void test_multi_block_reduce() {
  TaskManagerScope scope(32);
  WorkerPool pool(default_config(/*max_workers=*/2, /*fifo_capacity=*/8));

  constexpr size_t kElems = 1 << 20;
  constexpr size_t kBytes = kElems * sizeof(float);

  DeviceBuffer src(kBytes);
  DeviceBuffer dst(kBytes);
  std::vector<float> src_host(kElems);
  std::vector<float> dst_init(kElems);
  std::vector<float> expected(kElems);
  for (size_t i = 0; i < kElems; ++i) {
    src_host[i] = static_cast<float>(i % 97);
    dst_init[i] = static_cast<float>(500 + (i % 13));
    expected[i] = src_host[i] + dst_init[i];
  }

  upload_vector(src.ptr, src_host);
  upload_vector(dst.ptr, dst_init);

  require(pool.createWorker(0, 4), "multi-block reduce worker creation failed");
  pool.waitWorker(0);

  Task task = TaskManager::instance().create_task(
      make_reduce_args(src.ptr, dst.ptr, kBytes), TaskType::CollReduce,
      DataType::Fp32, 0);
  uint64_t task_id = pool.enqueue(task, 0);
  require(task_id != WorkerPool::kInvalidTaskId,
          "multi-block reduce enqueue failed");
  wait_until_done(pool, task_id, 0, std::chrono::seconds(10));

  verify_floats(download_vector<float>(dst.ptr, kBytes), expected,
                "multi-block reduce");
  pool.shutdown_all();
}

}  // namespace

}  // namespace Device
}  // namespace UKernel

int main() {
  try {
    GPU_RT_CHECK(gpuSetDevice(0));
    std::cout << "=== Device Integration Tests ===" << std::endl << std::endl;
    using namespace UKernel::Device;
    run_case("device integration", "two-worker copy and reduce",
             test_two_workers_copy_and_reduce);
    run_case("device integration", "same-flow reduce pipeline",
             test_same_flow_reduce_pipeline);
    run_case("device integration", "multi-block reduce",
             test_multi_block_reduce);
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << std::endl
              << "=== Device integration tests PASSED ===" << std::endl;
    return 0;
  } catch (std::exception const& ex) {
    std::cerr << "[device integration test] fatal: " << ex.what() << std::endl;
    return 2;
  }
}

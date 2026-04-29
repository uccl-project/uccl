#include "cuda_bf16.h"
#include "test_utils.h"
#include <algorithm>
#include <array>
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

TaskArgs make_reduce_args(void* src, void* dst, size_t bytes,
                          ReduceType reduce_type) {
  TaskArgs args = make_copy_args(src, dst, bytes);
  args.set_red_type(reduce_type);
  return args;
}

void test_task_encoding() {
  Task task(TaskType::CollReduce, DataType::Fp32, /*block=*/7, /*args=*/33);
  require(task.type_u8() == static_cast<uint8_t>(TaskType::CollReduce),
          "task type encoding mismatch");
  require(task.dtype_u8() == static_cast<uint8_t>(DataType::Fp32),
          "task dtype encoding mismatch");
  require(task.block_index() == 7, "task block index mismatch");
  require(task.args_index() == 33, "task args index mismatch");
  require(sizeof(Task) == 16, "task ABI size mismatch");
}

void test_task_manager_publish_and_reuse() {
  TaskManagerScope scope(4);

  TaskArgs args =
      make_reduce_args(reinterpret_cast<void*>(0x1000),
                       reinterpret_cast<void*>(0x2000), 1024, ReduceType::Sum);
  args.src_rank = 1;
  args.dst_rank = 2;

  Task first = TaskManager::instance().create_task(
      args, TaskType::CollReduce, DataType::Fp32, /*blockId=*/3);
  TaskArgs staged{};
  GPU_RT_CHECK(gpuMemcpy(
      &staged, TaskManager::instance().d_task_args() + first.args_index(),
      sizeof(staged), gpuMemcpyDeviceToHost));

  require(staged.src == args.src, "task manager staged src mismatch");
  require(staged.dst == args.dst, "task manager staged dst mismatch");
  require(staged.bytes == args.bytes, "task manager staged bytes mismatch");
  require(staged.red_type() == ReduceType::Sum,
          "task manager staged reduce type mismatch");
  require(staged.is_published(), "task manager should publish task args");

  TaskManager::instance().free_task_args(first.args_index());
  GPU_RT_CHECK(gpuMemcpy(
      &staged, TaskManager::instance().d_task_args() + first.args_index(),
      sizeof(staged), gpuMemcpyDeviceToHost));
  require(!staged.is_published(),
          "freed task args should clear publish marker");

  Task second = TaskManager::instance().create_task(
      args, TaskType::CollReduce, DataType::Fp32, /*blockId=*/3);
  require(second.args_index() == first.args_index(),
          "task args slot should be reusable after free");
}

void test_worker_lifecycle_and_streams() {
  TaskManagerScope scope(8);

  gpuStream_t external_control = nullptr;
  GPU_RT_CHECK(
      gpuStreamCreateWithFlags(&external_control, gpuStreamNonBlocking));

  auto cleanup = [&] {
    if (external_control != nullptr) {
      gpuStreamDestroy(external_control);
    }
  };

  try {
    WorkerPool::Config cfg = default_config(/*max_workers=*/2);
    cfg.controlStream = external_control;
    WorkerPool pool(cfg);

    require(pool.control_stream() == external_control,
            "worker pool should use provided control stream");
    require(!pool.pollWorker(0),
            "worker should not poll ready before creation");
    require(pool.createWorker(0, 1), "worker creation should succeed");
    pool.waitWorker(0);
    require(pool.pollWorker(0), "worker should become ready");
    require(pool.getWorkerStream(0) != nullptr,
            "worker execution stream should be created");
    require(pool.getWorkerStream(0) != pool.control_stream(),
            "worker execution stream should differ from control stream");
    require(!pool.createWorker(0, 1), "same fifo should not bind two workers");

    int device = 0;
    int sm_count = 0;
    GPU_RT_CHECK(gpuGetDevice(&device));
    GPU_RT_CHECK(gpuDeviceGetAttribute(&sm_count, gpuDevAttrMultiProcessorCount,
                                       device));
    require(!pool.createWorker(1, static_cast<uint32_t>(sm_count + 1)),
            "creating more blocks than SMs should fail");

    pool.destroyWorker(0);
    require(!pool.pollWorker(0),
            "destroyed worker should no longer report ready");
    require(pool.getWorkerStream(0) == nullptr,
            "destroyed worker should no longer expose execution stream");
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();
}

void test_enqueue_without_worker_fails() {
  TaskManagerScope scope(4);
  WorkerPool pool(default_config());

  DeviceBuffer src(64);
  DeviceBuffer dst(64);
  TaskArgs args = make_copy_args(src.ptr, dst.ptr, 64);
  Task task = TaskManager::instance().create_task(args, TaskType::CollCopy,
                                                  DataType::Int8, 0);
  uint64_t task_id = pool.enqueue(task, 0);
  require(task_id == WorkerPool::kInvalidTaskId,
          "enqueue without a bound worker should fail");
  TaskManager::instance().free_task_args(task.args_index());
}

void test_single_block_copy() {
  TaskManagerScope scope(16);
  WorkerPool pool(default_config());

  constexpr size_t kBytes = 1 << 20;
  DeviceBuffer src(kBytes);
  DeviceBuffer dst(kBytes);
  std::vector<char> host_src(kBytes);
  for (size_t i = 0; i < host_src.size(); ++i) {
    host_src[i] = static_cast<char>(i & 0x7F);
  }
  upload_vector(src.ptr, host_src);
  zero_buffer(dst.ptr, kBytes);

  require(pool.createWorker(0, 1), "single-block worker creation failed");
  pool.waitWorker(0);

  Task task = TaskManager::instance().create_task(
      make_copy_args(src.ptr, dst.ptr, kBytes), TaskType::CollCopy,
      DataType::Int8, 0);
  uint64_t task_id = pool.enqueue(task, 0);
  require(task_id != WorkerPool::kInvalidTaskId, "single-block enqueue failed");
  wait_until_done(pool, task_id, 0);

  verify_bytes(download_vector<char>(dst.ptr, kBytes), host_src,
               "single-block copy");
  pool.shutdown_all();
}

void test_multi_block_copy() {
  TaskManagerScope scope(16);
  WorkerPool pool(default_config());

  constexpr size_t kBytes = 4 << 20;
  DeviceBuffer src(kBytes);
  DeviceBuffer dst(kBytes);
  std::vector<char> host_src(kBytes);
  for (size_t i = 0; i < host_src.size(); ++i) {
    host_src[i] = static_cast<char>((i * 3) & 0xFF);
  }
  upload_vector(src.ptr, host_src);
  zero_buffer(dst.ptr, kBytes);

  require(pool.createWorker(0, 4), "multi-block worker creation failed");
  pool.waitWorker(0);

  Task task = TaskManager::instance().create_task(
      make_copy_args(src.ptr, dst.ptr, kBytes), TaskType::CollCopy,
      DataType::Int8, 0);
  uint64_t task_id = pool.enqueue(task, 0);
  require(task_id != WorkerPool::kInvalidTaskId, "multi-block enqueue failed");
  wait_until_done(pool, task_id, 0, std::chrono::seconds(10));

  verify_bytes(download_vector<char>(dst.ptr, kBytes), host_src,
               "multi-block copy");
  pool.shutdown_all();
}

void test_copy_supports_fp32_and_fp16() {
  TaskManagerScope scope(16);
  WorkerPool pool(default_config());
  require(pool.createWorker(0, 1), "worker creation failed");
  pool.waitWorker(0);

  constexpr size_t kElems = 256;
  DeviceBuffer src_f32(kElems * sizeof(float));
  DeviceBuffer dst_f32(kElems * sizeof(float));
  DeviceBuffer src_f16(kElems * sizeof(__half));
  DeviceBuffer dst_f16(kElems * sizeof(__half));

  std::vector<float> host_f32(kElems);
  std::vector<float> expect_f32(kElems);
  std::vector<__half> host_f16(kElems);
  std::vector<float> expect_f16(kElems);
  for (size_t i = 0; i < kElems; ++i) {
    expect_f32[i] = static_cast<float>(i) * 2.5f;
    host_f32[i] = expect_f32[i];
    expect_f16[i] = static_cast<float>(i) * 1.5f;
    host_f16[i] = __float2half(expect_f16[i]);
  }
  upload_vector(src_f32.ptr, host_f32);
  upload_vector(src_f16.ptr, host_f16);
  zero_buffer(dst_f32.ptr, dst_f32.bytes);
  zero_buffer(dst_f16.ptr, dst_f16.bytes);

  Task task_f32 = TaskManager::instance().create_task(
      make_copy_args(src_f32.ptr, dst_f32.ptr, dst_f32.bytes),
      TaskType::CollCopy, DataType::Fp32, 0);
  uint64_t task_f32_id = pool.enqueue(task_f32, 0);
  require(task_f32_id != WorkerPool::kInvalidTaskId, "fp32 enqueue failed");
  wait_until_done(pool, task_f32_id, 0);
  verify_floats(download_vector<float>(dst_f32.ptr, dst_f32.bytes), expect_f32,
                "fp32 copy");

  Task task_f16 = TaskManager::instance().create_task(
      make_copy_args(src_f16.ptr, dst_f16.ptr, dst_f16.bytes),
      TaskType::CollCopy, DataType::Fp16, 0);
  uint64_t task_f16_id = pool.enqueue(task_f16, 0);
  require(task_f16_id != WorkerPool::kInvalidTaskId, "fp16 enqueue failed");
  wait_until_done(pool, task_f16_id, 0);

  std::vector<__half> actual_f16 =
      download_vector<__half>(dst_f16.ptr, dst_f16.bytes);
  std::vector<float> actual_f16_float(kElems, 0.0f);
  for (size_t i = 0; i < kElems; ++i) {
    actual_f16_float[i] = __half2float(actual_f16[i]);
  }
  verify_floats(actual_f16_float, expect_f16, "fp16 copy");
  pool.shutdown_all();
}

void test_multiple_fifos_copy() {
  TaskManagerScope scope(32);
  WorkerPool pool(default_config(/*max_workers=*/4));

  constexpr size_t kBytes = 512;
  std::array<DeviceBuffer, 3> src = {DeviceBuffer(kBytes), DeviceBuffer(kBytes),
                                     DeviceBuffer(kBytes)};
  std::array<DeviceBuffer, 3> dst = {DeviceBuffer(kBytes), DeviceBuffer(kBytes),
                                     DeviceBuffer(kBytes)};

  std::array<std::vector<char>, 3> host_src = {std::vector<char>(kBytes),
                                               std::vector<char>(kBytes),
                                               std::vector<char>(kBytes)};
  for (size_t fifo = 0; fifo < host_src.size(); ++fifo) {
    for (size_t i = 0; i < kBytes; ++i) {
      host_src[fifo][i] = static_cast<char>(fifo * 37 + i);
    }
    upload_vector(src[fifo].ptr, host_src[fifo]);
    zero_buffer(dst[fifo].ptr, kBytes);
    require(pool.createWorker(static_cast<uint32_t>(fifo), 1),
            "multi-fifo worker creation failed");
    pool.waitWorker(static_cast<uint32_t>(fifo));
  }

  for (size_t fifo = 0; fifo < host_src.size(); ++fifo) {
    Task task = TaskManager::instance().create_task(
        make_copy_args(src[fifo].ptr, dst[fifo].ptr, kBytes),
        TaskType::CollCopy, DataType::Int8, 0);
    uint64_t task_id = pool.enqueue(task, static_cast<uint32_t>(fifo));
    require(task_id != WorkerPool::kInvalidTaskId, "multi-fifo enqueue failed");
    wait_until_done(pool, task_id, static_cast<uint32_t>(fifo));
  }

  for (size_t fifo = 0; fifo < host_src.size(); ++fifo) {
    verify_bytes(download_vector<char>(dst[fifo].ptr, kBytes), host_src[fifo],
                 "fifo copy");
  }
  pool.shutdown_all();
}

}  // namespace

}  // namespace Device
}  // namespace UKernel

int main() {
  try {
    int device_count = 0;
    GPU_RT_CHECK(gpuGetDeviceCount(&device_count));
    UKernel::Device::TestUtil::require(device_count > 0,
                                       "no CUDA devices available");
    GPU_RT_CHECK(gpuSetDevice(0));

    std::cout << "=== Device Unit Tests ===" << std::endl << std::endl;
    using namespace UKernel::Device;
    run_case("device unit", "task encoding", test_task_encoding);
    run_case("device unit", "task manager publish and reuse",
             test_task_manager_publish_and_reuse);
    run_case("device unit", "worker lifecycle and streams",
             test_worker_lifecycle_and_streams);
    run_case("device unit", "enqueue without worker fails",
             test_enqueue_without_worker_fails);
    run_case("device unit", "single-block copy", test_single_block_copy);
    run_case("device unit", "multi-block copy", test_multi_block_copy);
    run_case("device unit", "copy supports fp32 and fp16",
             test_copy_supports_fp32_and_fp16);
    run_case("device unit", "multiple fifo copy", test_multiple_fifos_copy);
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << std::endl << "=== Device unit tests PASSED ===" << std::endl;
    return 0;
  } catch (std::exception const& ex) {
    std::cerr << "[device unit test] fatal: " << ex.what() << std::endl;
    return 2;
  }
}

// Test-only single-device receiver. Used inside a real model's MoE layers;
// retains production GIN/FIFO encoding but supplies local copy completions.
#include "queue_fixture.hpp"
#include "thirdparty/nccl-ep/adapter/uccl_gin_net.cuh"
#include <algorithm>

#ifndef MODEL_EXPECT_SCALAR_FLUSH
#define MODEL_EXPECT_SCALAR_FLUSH 0
#endif

using namespace local_test;
using nccl_ep_adapter::UcclGinNet;

__global__ void transfer_activation(uccl_gin::UCCLGinResources resources,
                                    size_t bytes, int groups,
                                    uint64_t generation) {
  int const group = blockIdx.x;
  int const producer = group * 32 + threadIdx.x;
  size_t const words = bytes / sizeof(uint32_t);
  size_t const begin = words * producer / (groups * 32);
  size_t const end = words * (producer + 1) / (groups * 32);
  size_t const receive = resources.window_bytes / 2;
  UcclGinNet net(resources, group);
  net.put({}, 1, {}, receive + begin * sizeof(uint32_t), {},
          begin * sizeof(uint32_t), (end - begin) * sizeof(uint32_t),
          ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
          cuda::thread_scope_thread, cuda::thread_scope_device,
          ncclGinOptFlagsAggregateRequests);
  net.signal({}, 1, ncclGin_SignalAdd{static_cast<uint32_t>(group), 1},
             ncclCoopWarp{}, ncclGin_None{}, cuda::thread_scope_thread,
             cuda::thread_scope_thread);
#if MODEL_EXPECT_SCALAR_FLUSH
  // The original adapter forwards every participating member to scalar flush.
  // The typed put/signal prerequisite is identical in both model builds.
  net.gin.flush();
#else
  net.flush(ncclCoopWarp{}, cuda::memory_order_acquire);
#endif
  auto* source = reinterpret_cast<uint32_t*>(resources.window_base);
  for (size_t word = begin; word < end; ++word) source[word] = 0xdeadbeefu;
  net.waitSignal(ncclCoopWarp{}, group, generation);
}

struct ModelTransport {
  QueueFixture fixture;
  size_t const maximum;
  uint8_t* window;
  uint8_t *expected, *captured;
  int64_t *host_counters, *device_counters;
  cudaStream_t copy_stream;
  cudaEvent_t copied;
  uint64_t generation = 0, calls = 0, writes = 0, signals = 0, quiets = 0;
  uint64_t payload_bytes = 0;

  ModelTransport(int device, int queues, size_t maximum)
      : fixture(device, queues, 4096), maximum(maximum) {
    require(queues > 0 && queues <= 64 && maximum % 4 == 0 &&
                maximum * 2 <= UINT32_MAX,
            "model window size/queues");
    CUDA_CHECK(cudaMalloc(&window, maximum * 2));
    CUDA_CHECK(cudaHostAlloc(&expected, maximum, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&captured, maximum, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host_counters, 1024 * sizeof(int64_t),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&device_counters, host_counters, 0));
    std::memset(host_counters, 0, 1024 * sizeof(int64_t));
    fixture.resources.window_base = reinterpret_cast<uint64_t>(window);
    fixture.resources.window_bytes = maximum * 2;
    fixture.resources.atomic_tail_base =
        reinterpret_cast<uint64_t>(device_counters);
    fixture.resources.num_scaleout_ranks = 2;
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copied, copy_stream));
    CUDA_CHECK(cudaEventSynchronize(copied));
  }

  void copy(void const* input, void* output, size_t bytes, int groups,
            cudaStream_t input_stream) {
    require(groups > 0 && groups <= 64 && bytes <= maximum && bytes % 4 == 0 &&
                bytes >= static_cast<size_t>(groups * 32 * 4),
            "activation shape exceeds transport bounds");
    CUDA_CHECK(cudaSetDevice(fixture.device));
    // Both model producers and this staging copy precede the FIFO kernel on
    // the model stream. Allocate/initialize every copy resource beforehand.
    CUDA_CHECK(cudaMemcpyAsync(window, input, bytes, cudaMemcpyDeviceToDevice,
                               input_stream));
    CUDA_CHECK(cudaMemcpyAsync(expected, window, bytes, cudaMemcpyDeviceToHost,
                               input_stream));
    CUDA_CHECK(cudaStreamSynchronize(input_stream));
    ++generation;
    // Calls are serialized by the model executor. The previous kernel and
    // consumer have joined before an inactive group's counter is advanced.
    for (int group = 0; group < groups; ++group)
      __atomic_store_n(host_counters + group,
                       static_cast<int64_t>(generation - 1), __ATOMIC_RELEASE);
    std::vector<int> group_writes(groups, 0);
    std::vector<bool> group_signaled(groups, false);
    std::vector<bool> producer_seen(groups * 32, false);
    uint64_t call_writes = 0, call_signals = 0, call_quiets = 0,
             copied_bytes = 0;
    size_t const words = bytes / sizeof(uint32_t);
    fixture.start([&](size_t queue, TransferCmd const& cmd) {
      auto const kind = get_base_cmd(cmd.cmd_type);
      if (kind == CmdType::QUIET) {
        ++call_quiets;
        return;
      }
      require(cmd.dst_rank == 1, "model rail peer");
      if (kind == CmdType::WRITE) {
        size_t const offset = cmd.req_lptr << kWriteAddrShiftNormal;
        size_t const destination = cmd.req_rptr << kWriteAddrShiftNormal;
        require(cmd.bytes > 0 && offset + cmd.bytes <= bytes &&
                    destination == maximum + offset,
                "model WRITE window");
        // Match each producer's exact partition, including uneven division.
        int const producer = std::min<int>(
            ((offset / sizeof(uint32_t) + 1) * groups * 32 - 1) / words,
            groups * 32 - 1);
        size_t const begin =
            words * producer / (groups * 32) * sizeof(uint32_t);
        size_t const end =
            words * (producer + 1) / (groups * 32) * sizeof(uint32_t);
        int const group = producer / 32;
        require(!producer_seen[producer] && offset == begin &&
                    cmd.bytes == end - begin &&
                    queue == static_cast<size_t>(group %
                                                 fixture.resources.num_queues),
                "model producer/channel partition");
        require(!group_signaled[group], "model signal overtook data");
        producer_seen[producer] = true;
        CUDA_CHECK(cudaMemcpyAsync(captured + offset, window + offset,
                                   cmd.bytes, cudaMemcpyDeviceToHost,
                                   copy_stream));
        CUDA_CHECK(cudaEventRecord(copied, copy_stream));
        CUDA_CHECK(cudaEventSynchronize(copied));
        require(
            std::memcmp(captured + offset, expected + offset, cmd.bytes) == 0,
            "model source reused before completion");
        CUDA_CHECK(cudaMemcpyAsync(window + destination, captured + offset,
                                   cmd.bytes, cudaMemcpyHostToDevice,
                                   copy_stream));
        CUDA_CHECK(cudaEventRecord(copied, copy_stream));
        CUDA_CHECK(cudaEventSynchronize(copied));
        ++group_writes[group];
        ++call_writes;
        copied_bytes += cmd.bytes;
      } else {
        require(kind == CmdType::ATOMIC && cmd.atomic_offset == 1 &&
                    cmd.value == 1 && cmd.req_rptr % sizeof(int64_t) == 0,
                "model ordered signal encoding");
        size_t const group = cmd.req_rptr / sizeof(int64_t);
        require(group < static_cast<size_t>(groups) && !group_signaled[group] &&
                    group_writes[group] == 32 &&
                    queue == group % fixture.resources.num_queues,
                "model signal ordering/channel/count");
        group_signaled[group] = true;
        __atomic_fetch_add(host_counters + group, int64_t{1}, __ATOMIC_RELEASE);
        ++call_signals;
      }
    });
    transfer_activation<<<groups, 32, 0, fixture.stream>>>(
        fixture.resources, bytes, groups, generation);
    fixture.finish();
    uint64_t const expected_quiets = static_cast<uint64_t>(groups) *
                                     fixture.resources.num_queues *
                                     (MODEL_EXPECT_SCALAR_FLUSH ? 32 : 1);
    require(call_writes == static_cast<uint64_t>(groups * 32) &&
                call_signals == static_cast<uint64_t>(groups) &&
                call_quiets == expected_quiets && copied_bytes == bytes,
            "model missing/duplicate payload or completion commands");
    CUDA_CHECK(cudaMemcpyAsync(output, window + maximum, bytes,
                               cudaMemcpyDeviceToDevice, input_stream));
    CUDA_CHECK(cudaStreamSynchronize(input_stream));
    ++calls;
    writes += call_writes;
    signals += call_signals;
    quiets += call_quiets;
    payload_bytes += copied_bytes;
  }

  ~ModelTransport() {
    CUDA_CHECK(cudaSetDevice(fixture.device));
    CUDA_CHECK(cudaEventDestroy(copied));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaFreeHost(host_counters));
    CUDA_CHECK(cudaFreeHost(captured));
    CUDA_CHECK(cudaFreeHost(expected));
    CUDA_CHECK(cudaFree(window));
  }
};

extern "C" void* model_transport_create(int device, int queues,
                                        size_t maximum) {
  return new ModelTransport(device, queues, maximum);
}
extern "C" void model_transport_copy(void* handle, void const* input,
                                     void* output, size_t bytes, int groups,
                                     void* stream) {
  static_cast<ModelTransport*>(handle)->copy(input, output, bytes, groups,
                                             static_cast<cudaStream_t>(stream));
}
extern "C" void model_transport_stats(void* handle, uint64_t* result) {
  auto const& transport = *static_cast<ModelTransport*>(handle);
  result[0] = transport.calls;
  result[1] = transport.writes;
  result[2] = transport.signals;
  result[3] = transport.quiets;
  result[4] = transport.payload_bytes;
}
extern "C" void model_transport_destroy(void* handle) {
  delete static_cast<ModelTransport*>(handle);
}

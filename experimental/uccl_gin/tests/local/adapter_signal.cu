#include "queue_fixture.hpp"
#include "thirdparty/nccl-ep/adapter/uccl_gin_net.cuh"
#include <array>

using namespace local_test;
using nccl_ep_adapter::UcclGinNet;
constexpr int kChannel = 7, kSignalTestGenerations = 10;

template <int Width, int Arity>
__global__ void send_signals(uccl_gin::UCCLGinResources resources,
                             ncclGinSignal_t id) {
  using Coop = std::conditional_t<Width == 32, ncclCoopWarp, ncclCoopThread>;
  UcclGinNet net(resources, kChannel);
  Coop coop;
  for (int generation = 1; generation <= kSignalTestGenerations; ++generation) {
    // A late non-leader makes a missing entry barrier observable.
    if (threadIdx.x == Width - 1) __nanosleep(100000);
    size_t const word = (generation - 1) * Width + threadIdx.x;
    auto* source = reinterpret_cast<uint64_t*>(resources.window_base);
    source[word] = generation * 1000 + threadIdx.x;
    net.put({}, 1, {},
            (kSignalTestGenerations * Width + word) * sizeof(uint64_t), {},
            word * sizeof(uint64_t), sizeof(uint64_t), ncclGin_None{},
            ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            cuda::thread_scope_thread, cuda::thread_scope_device,
            ncclGinOptFlagsAggregateRequests);
    if constexpr (Arity == 3)
      net.signal({}, 1, ncclGin_SignalAdd{id, 1});
    else if constexpr (Arity == 7)
      net.signal({}, 1, ncclGin_SignalAdd{id, 1}, coop, ncclGin_None{},
                 cuda::thread_scope_thread, cuda::thread_scope_thread);
    else
      net.signal({}, 1, ncclGin_SignalAdd{id, 1}, coop, ncclGin_None{},
                 cuda::thread_scope_thread, cuda::thread_scope_thread,
                 ncclGinOptFlagsDefault);
    net.waitSignal(coop, id, generation);
  }
  net.flush(coop);
}

template <int Width, int Arity>
void check_signals(int device, int local_world, ncclGinSignal_t id) {
  QueueFixture fixture(device, 3, 4096);
  uint64_t* window;
  int64_t *host_counters, *device_counters;
  CUDA_CHECK(cudaMalloc(&window,
                        2 * kSignalTestGenerations * Width * sizeof(uint64_t)));
  CUDA_CHECK(cudaHostAlloc(&host_counters, 1024 * sizeof(int64_t),
                           cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&device_counters, host_counters, 0));
  std::memset(host_counters, 0, 1024 * sizeof(int64_t));
  fixture.resources.window_base = reinterpret_cast<uint64_t>(window);
  fixture.resources.window_bytes =
      2 * kSignalTestGenerations * Width * sizeof(uint64_t);
  fixture.resources.atomic_tail_base =
      reinterpret_cast<uint64_t>(device_counters);
  fixture.resources.num_scaleout_ranks = 2;
  fixture.resources.num_scaleup_ranks = local_world;
  fixture.resources.scaleup_rank = local_world - 1;
  auto capture = mscclpp::detail::gpuCallocHostUnique<uint64_t>(1);
  cudaStream_t copy_stream;
  cudaEvent_t copied;
  CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(copied, copy_stream));
  CUDA_CHECK(cudaEventSynchronize(copied));
  int writes = 0, signals = 0, quiets = 0;
  std::array<bool, kSignalTestGenerations * Width> seen{};
  fixture.start([&](size_t queue, TransferCmd const& cmd) {
    CmdType const kind = get_base_cmd(cmd.cmd_type);
    if (kind == CmdType::QUIET) {
      ++quiets;
      return;
    }
    require(queue == kChannel % 3, "put/signal changed channel");
    require(cmd.dst_rank == 2 * local_world - 1, "rail peer mapping");
    if (kind == CmdType::WRITE) {
      size_t const word =
          (cmd.req_lptr << kWriteAddrShiftNormal) / sizeof(uint64_t);
      require(word < seen.size() && !seen[word],
              "duplicate/invalid source offset");
      require((cmd.req_rptr << kWriteAddrShiftNormal) ==
                  (kSignalTestGenerations * Width + word) * sizeof(uint64_t),
              "destination offset");
      require(cmd.bytes == sizeof(uint64_t), "put byte count");
      CUDA_CHECK(cudaMemcpyAsync(capture.get(), window + word, sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, copy_stream));
      CUDA_CHECK(cudaEventRecord(copied, copy_stream));
      CUDA_CHECK(cudaEventSynchronize(copied));
      require(*capture == (word / Width + 1) * 1000 + word % Width,
              "source visibility");
      seen[word] = true;
      ++writes;
    } else {
      require(kind == CmdType::ATOMIC && cmd.atomic_offset == 1 &&
                  cmd.req_rptr == uint64_t{id} * sizeof(int64_t) &&
                  cmd.value == 1,
              "ordered ATOMIC encoding, including signal ID zero");
      ++signals;
      require(writes == signals * Width, "signal overtook a member's put");
      __atomic_fetch_add(host_counters + id, int64_t{1}, __ATOMIC_RELEASE);
    }
  });
  send_signals<Width, Arity>
      <<<1, Width, 0, fixture.stream>>>(fixture.resources, id);
  fixture.finish();
  require(writes == kSignalTestGenerations * Width &&
              signals == kSignalTestGenerations && quiets == 3,
          "logical signal/flush count");
  require(host_counters[id] == kSignalTestGenerations, "counter value");
  CUDA_CHECK(cudaEventDestroy(copied));
  CUDA_CHECK(cudaStreamDestroy(copy_stream));
  CUDA_CHECK(cudaFreeHost(host_counters));
  CUDA_CHECK(cudaFree(window));
}

int main(int argc, char** argv) {
  int const device = integer(argc, argv, "--device", 0);
  int cases = 0;
  for (int local_world : {1, 2, 8})
    for (ncclGinSignal_t id : {0u, 1023u}) {
      check_signals<1, 3>(device, local_world, id);
      check_signals<1, 7>(device, local_world, id);
      check_signals<32, 7>(device, local_world, id);
      check_signals<32, 8>(device, local_world, id);
      cases += 4;
    }
  std::printf(
      "{\"scope\":\"native CUDA + production FIFO, test consumer; no network\","
      "\"device\":%d,\"signal_cases\":%d,\"pass\":true}\n",
      device, cases);
}

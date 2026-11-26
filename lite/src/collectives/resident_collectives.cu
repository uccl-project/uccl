// Experimental input-ready, host-blocking collectives. A prepared resident CTA
// on each GPU replaces per-call launches. This is a separate execution contract
// from stream-enqueued collectives, with an identically scoped NCCL control.
#define main uccl_lite_peer_benchmark_main
#include "peer_ag_rs.cu"
#undef main

namespace uccl_lite {
constexpr size_t kResidentRsLlBytes = 8192;
constexpr size_t kResidentArLlBytes = 2048;
struct alignas(128) ResidentWord {
  uint32_t value;
};
struct alignas(128) ResidentDoorbell {
  uint64_t value;
};
struct ResidentCommand {
  ResidentDoorbell doorbell;
  ResidentWord ready[4], done[4], active[4];
};
__host__ __device__ float residentValue(size_t i, int rank, int generation) {
  uint32_t x = uint32_t(i) ^ (uint32_t(rank + 1) * 0x9e3779b9u) ^
               (uint32_t(generation + 1) * 0x85ebca6bu);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  return (int(x % 200001) - 100000) / 100003.0f;
}
__global__ void residentFill(float* input, size_t count, int rank, int gen) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    input[i] = residentValue(i, rank, gen);
}
__global__ void residentCheck(float const* output, unsigned int* wrong,
                              size_t count, int op, int rank, int gen) {
  size_t shard = count / 4, n = op == 1 ? shard : count;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    double expected = 0;
    if (op == 0)
      expected = residentValue(i % shard, int(i / shard), gen);
    else
      for (int r = 0; r < 4; ++r)
        expected +=
            residentValue(op == 1 ? size_t(rank) * shard + i : i, r, gen);
    double tolerance = op == 0 ? 0 : 1e-6 + 1e-6 * fabs(expected);
    if (!isfinite(output[i]) || fabs(double(output[i]) - expected) > tolerance)
      atomicAdd(wrong, 1u);
  }
}
// A resident grid spans many producer launches; bypass its noncoherent L1
// when reading inputs or receive slots written by other SMs/GPUs.
__device__ uint4 residentLoad4(void const* from) {
  uint4 value;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(from)
               : "memory");
  return value;
}
__device__ float residentLoad(float const* from) {
  float value;
  asm volatile("ld.global.cg.f32 %0, [%1];"
               : "=f"(value)
               : "l"(from)
               : "memory");
  return value;
}
__device__ float4 residentLoadFloat4(float const* from) {
  uint4 value = residentLoad4(from);
  return make_float4(__uint_as_float(value.x), __uint_as_float(value.y),
                     __uint_as_float(value.z), __uint_as_float(value.w));
}
__device__ void residentCopy(float* to, float const* from, size_t count) {
  if (count % 4 == 0) {
    for (size_t i = threadIdx.x; i < count / 4; i += blockDim.x)
      reinterpret_cast<uint4*>(to)[i] = residentLoad4(from + 4 * i);
  } else
    for (size_t i = threadIdx.x; i < count; i += blockDim.x)
      to[i] = residentLoad(from + i);
}
__global__ void residentCollective(
    ResidentCommand* command, int rank, float* input, float* output, float* o0,
    float* o1, float* o2, float* o3, float* s0, float* s1, float* s2, float* s3,
    float* i0, float* i1, float* i2, float* i3, mscclpp::LL16Packet* p0,
    mscclpp::LL16Packet* p1, mscclpp::LL16Packet* p2, mscclpp::LL16Packet* p3,
    bool packetMode, uint32_t initialEpoch) {
  float* outputs[4] = {o0, o1, o2, o3};
  float* inputs[4] = {i0, i1, i2, i3};
  float* scratch[4] = {s0, s1, s2, s3};
  mscclpp::LL16Packet* packets[4] = {p0, p1, p2, p3};
  __shared__ uint32_t epoch, op;
  __shared__ size_t count;
  uint32_t previous = initialEpoch;
  if (threadIdx.x == 0) releaseSystemFlag(&command->active[rank].value, 1);
  while (true) {
    if (threadIdx.x == 0) {
      uint64_t word;
      do {
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                     : "=l"(word)
                     : "l"(&command->doorbell.value)
                     : "memory");
        __nanosleep(64);
      } while (uint32_t(word >> 32) == previous);
      epoch = uint32_t(word >> 32);
      op = uint32_t(word & 3);
      count = size_t((word >> 2) & 0x3fffffffULL);
    }
    __syncthreads();
    if (op == 3) break;
    size_t shard = count / 4;
    bool packetReduction =
        packetMode &&
        ((op == 1 && count * sizeof(float) <= kResidentRsLlBytes) ||
         (op == 2 && count * sizeof(float) <= kResidentArLlBytes));
    if (packetReduction) {
      size_t extent = op == 1 ? shard : count;
      size_t units = (extent + 1) / 2;
      for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
        size_t index = 2 * i;
        for (int phase = 1; phase < 4; ++phase) {
          int dest = (rank + phase) % 4;
          float const* from = input + (op == 1 ? size_t(dest) * shard : 0);
          uint2 value =
              make_uint2(__float_as_uint(residentLoad(from + index)),
                         index + 1 < extent
                             ? __float_as_uint(residentLoad(from + index + 1))
                             : 0);
          packets[dest][size_t(rank) * units + i].write(value, epoch);
        }
        float const* local = input + (op == 1 ? size_t(rank) * shard : 0);
        float a = residentLoad(local + index);
        float b = index + 1 < extent ? residentLoad(local + index + 1) : 0;
        for (int source = 0; source < 4; ++source) {
          if (source == rank) continue;
          uint2 value = packets[rank][size_t(source) * units + i].read(epoch);
          a += __uint_as_float(value.x);
          b += __uint_as_float(value.y);
        }
        float* result = op == 1 ? output : input;
        result[index] = a;
        if (index + 1 < extent) result[index + 1] = b;
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0)
        releaseSystemFlag(&command->done[rank].value, epoch);
      previous = epoch;
      __syncthreads();
      continue;
    }
    if (op == 0) {
      for (int phase = 0; phase < 4; ++phase) {
        int dest = (rank + phase) % 4;
        residentCopy(outputs[dest] + size_t(rank) * shard, input, shard);
        __syncthreads();
      }
    } else {
      size_t extent = shard;
      for (int phase = 1; phase < 4; ++phase) {
        int dest = (rank + phase) % 4;
        residentCopy(scratch[dest] + size_t(rank) * extent,
                     input + size_t(dest) * shard, extent);
        __syncthreads();
      }
    }
    __threadfence_system();
    __syncthreads();
    if (op == 0) {
      if (threadIdx.x == 0)
        releaseSystemFlag(&command->done[rank].value, epoch);
      previous = epoch;
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0) {
      releaseSystemFlag(&command->ready[rank].value, epoch);
      for (int source = 0; source < 4; ++source)
        while (acquireSystemFlag(&command->ready[source].value) != epoch)
          __nanosleep(64);
    }
    __syncthreads();
    if (op != 0) {
      size_t extent = shard;
      float const* local = input + size_t(rank) * shard;
      float* result = op == 1 ? output : input + size_t(rank) * shard;
      if (extent % 4 == 0) {
        for (size_t i = threadIdx.x; i < extent / 4; i += blockDim.x) {
          float4 value = residentLoadFloat4(local + 4 * i);
          for (int source = 0; source < 4; ++source) {
            if (source == rank) continue;
            float4 b = residentLoadFloat4(scratch[rank] +
                                          size_t(source) * extent + 4 * i);
            value.x += b.x;
            value.y += b.y;
            value.z += b.z;
            value.w += b.w;
          }
          reinterpret_cast<float4*>(result)[i] = value;
        }
      } else {
        for (size_t i = threadIdx.x; i < extent; i += blockDim.x) {
          float value = residentLoad(local + i);
          for (int source = 0; source < 4; ++source)
            if (source != rank)
              value +=
                  residentLoad(scratch[rank] + size_t(source) * extent + i);
          result[i] = value;
        }
      }
    }
    if (op == 2) {
      // Only local SMs read the newly reduced shard before forwarding it;
      // device scope suffices here. The final system fence publishes peer
      // writes.
      __threadfence();
      __syncthreads();
      for (int phase = 1; phase < 4; ++phase) {
        int dest = (rank + phase) % 4;
        residentCopy(inputs[dest] + size_t(rank) * shard,
                     input + size_t(rank) * shard, shard);
        __syncthreads();
      }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) releaseSystemFlag(&command->done[rank].value, epoch);
    previous = epoch;
    __syncthreads();
  }
  if (threadIdx.x == 0) releaseSystemFlag(&command->active[rank].value, 0);
}
struct Resident {
  std::array<Device*, 4> devices;
  std::array<float*, 4> scratch{};
  std::array<mscclpp::LL16Packet*, 4> packets{};
  bool packetMode;
  size_t maxCount;
  std::array<cudaStream_t, 4> streams{};
  ResidentCommand* command = nullptr;
  uint32_t epoch = 0;
  bool running = false;
  Resident(std::array<Device*, 4> d, size_t bytes, bool packet)
      : devices(d), packetMode(packet), maxCount(bytes / sizeof(float)) {
    if (maxCount < 4 || maxCount >= (1ULL << 30) || bytes % 16)
      throw std::invalid_argument("invalid resident capacity");
    cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&command),
                            sizeof(*command),
                            cudaHostAllocPortable | cudaHostAllocMapped));
    for (int r = 0; r < 4; ++r) {
      cudaCheck(cudaSetDevice(r));
      cudaCheck(cudaMalloc(&scratch[r], bytes));
      if (packetMode) {
        size_t arExtent = std::min(bytes, kResidentArLlBytes) / sizeof(float);
        size_t rsExtent =
            std::min(bytes, kResidentRsLlBytes) / (4 * sizeof(float));
        size_t packetBytes = 4 * ((std::max(arExtent, rsExtent) + 1) / 2) *
                             sizeof(mscclpp::LL16Packet);
        cudaCheck(cudaMalloc(&packets[r], packetBytes));
        cudaCheck(cudaMemset(packets[r], 0, packetBytes));
      }
      cudaCheck(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
    }
  }
  void wait(ResidentWord* words, uint32_t value) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (int r = 0; r < 4; ++r)
      while (__atomic_load_n(&words[r].value, __ATOMIC_ACQUIRE) != value) {
        if (std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error("resident collective timeout");
        asm volatile("pause");
      }
  }
  void start() {
    if (running) throw std::logic_error("resident already running");
    if (epoch == UINT32_MAX)
      throw std::runtime_error("resident epoch exhausted");
    std::memset(command, 0, sizeof(*command));
    __atomic_store_n(&command->doorbell.value, uint64_t(epoch) << 32,
                     __ATOMIC_RELEASE);
    running = true;
    for (int r = 0; r < 4; ++r) {
      cudaCheck(cudaSetDevice(r));
      residentCollective<<<1, 256, 0, streams[r]>>>(
          command, r, devices[r]->input, devices[r]->output, devices[0]->output,
          devices[1]->output, devices[2]->output, devices[3]->output,
          scratch[0], scratch[1], scratch[2], scratch[3], devices[0]->input,
          devices[1]->input, devices[2]->input, devices[3]->input, packets[0],
          packets[1], packets[2], packets[3], packetMode, epoch);
      cudaCheck(cudaGetLastError());
    }
    running = true;
    wait(command->active, 1);
  }
  void launch(int op, size_t count) {
    if (op < 0 || op > 2 || count < 4 || count % 4 || count > maxCount)
      throw std::invalid_argument("invalid resident command");
    if (epoch == UINT32_MAX)
      throw std::runtime_error("resident epoch exhausted");
    ++epoch;
    // One atomic descriptor read replaces separate PCIe reads of epoch,
    // operation and extent. Inputs are ready before this release publication.
    uint64_t word =
        (uint64_t(epoch) << 32) | (uint64_t(count) << 2) | uint64_t(op);
    __atomic_store_n(&command->doorbell.value, word, __ATOMIC_RELEASE);
    wait(command->done, epoch);
  }
  void stop() {
    if (!running) return;
    uint32_t stopEpoch = epoch == UINT32_MAX ? 0 : ++epoch;
    __atomic_store_n(&command->doorbell.value, (uint64_t(stopEpoch) << 32) | 3,
                     __ATOMIC_RELEASE);
    wait(command->active, 0);
    for (int r = 0; r < 4; ++r) {
      cudaCheck(cudaSetDevice(r));
      cudaCheck(cudaStreamSynchronize(streams[r]));
    }
    running = false;
  }
  ~Resident() {
    try {
      stop();
    } catch (...) {
      std::terminate();
    }
    for (int r = 0; r < 4; ++r) {
      cudaSetDevice(r);
      cudaStreamDestroy(streams[r]);
      cudaFree(scratch[r]);
      if (packets[r]) cudaFree(packets[r]);
    }
    cudaFreeHost(command);
  }
};
}  // namespace uccl_lite
#ifndef UCCL_LITE_RESIDENT_NO_MAIN
int main(int argc, char** argv) {
  using namespace uccl_lite;
  try {
    bool packetMode = true;
    int ncclGraph = 0;
    std::vector<char*> args{argv[0]};
    for (int i = 1; i < argc; i += 2) {
      std::string option = argv[i];
      if (option == "--help") {
        std::puts(
            "Usage: uccl_lite_resident_collectives -b BYTES -e BYTES "
            "-w WARMUPS -n ITERATIONS -r REPEATS [--packets 0|1] [--nccl-graph "
            "0|1]; four "
            "peer-accessible GPUs; "
            "input-ready host-blocking experiment, not stream enqueue");
        return 0;
      }
      if (i + 1 == argc)
        throw std::invalid_argument("missing resident option value");
      if (option == "--packets" || option == "--nccl-graph") {
        std::string value = argv[i + 1];
        if (value != "0" && value != "1" &&
            !(option == "--nccl-graph" && value == "2"))
          throw std::invalid_argument("invalid protocol mode");
        if (option == "--packets")
          packetMode = value == "1";
        else
          ncclGraph = std::stoi(value);
        continue;
      }
      args.push_back(argv[i]);
      args.push_back(argv[i + 1]);
      if (option != "-b" && option != "-e" && option != "-w" &&
          option != "-n" && option != "-r")
        throw std::invalid_argument("unsupported resident option: " + option);
    }
    auto cfg = parse(int(args.size()), args.data());
    if (cfg.graph)
      throw std::invalid_argument("resident mode is host-blocking");
    std::array<std::unique_ptr<Device>, 4> owned;
    std::array<Device*, 4> devices{};
    for (int r = 0; r < 4; ++r) {
      cudaCheck(cudaSetDevice(r));
      for (int p = 0; p < 4; ++p)
        if (r != p) {
          auto status = cudaDeviceEnablePeerAccess(p, 0);
          if (status == cudaErrorPeerAccessAlreadyEnabled)
            cudaGetLastError();
          else
            cudaCheck(status);
        }
      owned[r] = std::make_unique<Device>(r, cfg.end, 0, false);
      devices[r] = owned[r].get();
    }
    std::array<ncclComm_t, 4> comms{};
    int ranks[4] = {0, 1, 2, 3};
    ncclCheck(ncclCommInitAll(comms.data(), 4, ranks));
    int ncclVersion = 0;
    ncclCheck(ncclGetVersion(&ncclVersion));
    std::printf("# NCCL runtime version=%d\n", ncclVersion);
    // Load validation kernels/runtime copy paths before the persistent CTA:
    // lazy CUDA module loading may otherwise synchronize a running context.
    for (auto* d : devices) {
      cudaCheck(cudaSetDevice(d->rank));
      cudaCheck(cudaMemsetAsync(d->wrong, 0, sizeof(unsigned int), d->stream));
      residentFill<<<1, 32, 0, d->stream>>>(d->input, 0, d->rank, 0);
      residentCheck<<<1, 32, 0, d->stream>>>(d->output, d->wrong, 0, 0, d->rank,
                                             0);
      unsigned int warm = 0;
      cudaCheck(cudaMemcpyAsync(&warm, d->wrong, sizeof(warm),
                                cudaMemcpyDeviceToHost, d->stream));
      cudaCheck(cudaStreamSynchronize(d->stream));
    }
    Resident resident(devices, cfg.end, packetMode);
    std::printf(
        "# four L4 resident experiment; FP32 signed random; AG/RS "
        "out-of-place; "
        "AR in-place; input-ready host post-to-completion; prepare excluded; "
        "warmups=%d iterations=%d repeats=%d; no resident CTA during NCCL\n",
        cfg.warmups, cfg.iterations, cfg.repeats);
    std::puts(
        "# resident command uses one packed 64-bit release/acquire doorbell");
    std::printf(
        "# packet-mode=%d; LL RS through %zu bytes, AR through %zu bytes; "
        "larger AR uses RS+AG\n",
        int(packetMode), kResidentRsLlBytes, kResidentArLlBytes);
    std::printf(
        "# NCCL baseline=%s; graph mode preselects root context before both "
        "timers\n",
        ncclGraph == 2 ? "per-device-graph"
        : ncclGraph    ? "global-graph"
                       : "eager");
    for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
      size_t count = bytes / 4;
      for (int op = 0; op < 3; ++op) {
        char const* name = op == 0   ? "all_gather"
                           : op == 1 ? "reduce_scatter"
                                     : "all_reduce";
        char const* baselineName = ncclGraph == 2 ? "nccl-graph-per-device"
                                   : ncclGraph    ? "nccl-graph"
                                                  : "nccl";
        auto enqueueNccl = [&] {
          ncclCheck(ncclGroupStart());
          for (auto* d : devices) {
            cudaCheck(cudaSetDevice(d->rank));
            if (op == 0)
              ncclCheck(ncclAllGather(d->input, d->reference, count / 4,
                                      ncclFloat, comms[d->rank], d->stream));
            else if (op == 1)
              ncclCheck(ncclReduceScatter(d->input, d->reference, count / 4,
                                          ncclFloat, ncclSum, comms[d->rank],
                                          d->stream));
            else
              ncclCheck(ncclAllReduce(d->reference, d->reference, count,
                                      ncclFloat, ncclSum, comms[d->rank],
                                      d->stream));
          }
          ncclCheck(ncclGroupEnd());
        };
        auto checkResult = [&](bool baseline, int gen) {
          for (auto* d : devices) {
            cudaCheck(cudaSetDevice(d->rank));
            cudaCheck(
                cudaMemsetAsync(d->wrong, 0, sizeof(unsigned int), d->stream));
            residentCheck<<<256, 256, 0, d->stream>>>(baseline  ? d->reference
                                                      : op == 2 ? d->input
                                                                : d->output,
                                                      d->wrong, count, op,
                                                      d->rank, gen);
            unsigned int wrong = 0;
            cudaCheck(cudaMemcpyAsync(&wrong, d->wrong, sizeof(wrong),
                                      cudaMemcpyDeviceToHost, d->stream));
            cudaCheck(cudaStreamSynchronize(d->stream));
            if (wrong)
              throw std::runtime_error("resident signed-input mismatch");
          }
        };
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        std::array<cudaGraph_t, 4> graphs{};
        std::array<cudaGraphExec_t, 4> executables{};
        if (ncclGraph) {
          for (auto* d : devices) {
            cudaCheck(cudaSetDevice(d->rank));
            residentFill<<<256, 256, 0, d->stream>>>(
                op == 2 ? d->reference : d->input, op == 0 ? count / 4 : count,
                d->rank, 0);
          }
          synchronize(devices);
          enqueueNccl();
          synchronize(devices);
          checkResult(true, 0);
          if (ncclGraph == 2) {
            for (auto* d : devices) {
              cudaCheck(cudaSetDevice(d->rank));
              cudaCheck(cudaStreamBeginCapture(d->stream,
                                               cudaStreamCaptureModeGlobal));
            }
            enqueueNccl();
            for (auto* d : devices) {
              cudaCheck(cudaSetDevice(d->rank));
              cudaCheck(cudaStreamEndCapture(d->stream, &graphs[d->rank]));
            }
            for (auto* d : devices) {
              cudaCheck(cudaSetDevice(d->rank));
              cudaCheck(cudaGraphInstantiate(
                  &executables[d->rank], graphs[d->rank], nullptr, nullptr, 0));
            }
          } else {
            cudaCheck(cudaSetDevice(0));
            cudaCheck(cudaStreamBeginCapture(devices[0]->stream,
                                             cudaStreamCaptureModeGlobal));
            cudaCheck(cudaEventRecord(devices[0]->transferAllDone,
                                      devices[0]->stream));
            for (int r = 1; r < 4; ++r) {
              cudaCheck(cudaSetDevice(r));
              cudaCheck(cudaStreamWaitEvent(devices[r]->stream,
                                            devices[0]->transferAllDone, 0));
            }
            enqueueNccl();
            for (auto* d : devices) {
              cudaCheck(cudaSetDevice(d->rank));
              cudaCheck(cudaEventRecord(d->transferDone, d->stream));
            }
            cudaCheck(cudaSetDevice(0));
            for (int r = 1; r < 4; ++r)
              cudaCheck(cudaStreamWaitEvent(devices[0]->stream,
                                            devices[r]->transferDone, 0));
            cudaCheck(cudaStreamEndCapture(devices[0]->stream, &graph));
            cudaCheck(
                cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
          }
        }
        std::array<std::vector<double>, 2> means;
        for (int repeat = 0; repeat < cfg.repeats; ++repeat)
          for (int ordinal = 0; ordinal < 2; ++ordinal) {
            int backend = (repeat + ordinal) % 2;
            if (backend == 0) resident.start();
            double total = 0;
            for (int call = 0; call < cfg.warmups + cfg.iterations; ++call) {
              int gen = 10 + repeat * (cfg.warmups + cfg.iterations) + call;
              for (auto* d : devices) {
                cudaCheck(cudaSetDevice(d->rank));
                residentFill<<<256, 256, 0, d->stream>>>(
                    backend && op == 2 ? d->reference : d->input,
                    op == 0 ? count / 4 : count, d->rank, gen);
              }
              synchronize(devices);  // only caller streams; never synchronize
                                     // the device
              if (ncclGraph) cudaCheck(cudaSetDevice(0));
              auto start = std::chrono::steady_clock::now();
              if (backend == 0)
                resident.launch(op, count);
              else {
                if (ncclGraph == 2) {
                  for (auto* d : devices) {
                    cudaCheck(cudaSetDevice(d->rank));
                    cudaCheck(cudaGraphLaunch(executables[d->rank], d->stream));
                  }
                  synchronize(devices);
                } else if (ncclGraph) {
                  cudaCheck(cudaGraphLaunch(executable, devices[0]->stream));
                  cudaCheck(cudaStreamSynchronize(devices[0]->stream));
                } else {
                  enqueueNccl();
                  synchronize(devices);
                }
              }
              double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - start)
                              .count();
              checkResult(backend != 0, gen);
              if (call >= cfg.warmups) {
                total += us;
                std::printf("# sample,%s,%zu,%s,%d,%d,%.3f\n", name, bytes,
                            backend ? baselineName : "resident", repeat + 1,
                            call - cfg.warmups + 1, us);
              }
            }
            if (backend == 0) resident.stop();
            means[backend].push_back(total / cfg.iterations);
          }
        for (int b = 0; b < 2; ++b)
          std::printf("%s,%zu,%s,%.3f\n", name, bytes,
                      b ? baselineName : "resident", median(means[b]));
        std::fflush(stdout);
        for (int r = 0; r < 4; ++r) {
          if (executables[r]) cudaCheck(cudaGraphExecDestroy(executables[r]));
          if (graphs[r]) cudaCheck(cudaGraphDestroy(graphs[r]));
        }
        if (executable) cudaCheck(cudaGraphExecDestroy(executable));
        if (graph) {
          cudaCheck(cudaGraphDestroy(graph));
          for (auto* d : devices) {
            cudaCheck(cudaSetDevice(d->rank));
            cudaCheck(cudaEventRecord(d->transferDone, d->stream));
            cudaCheck(cudaEventRecord(d->transferAllDone, d->stream));
          }
          synchronize(devices);
        }
      }
      if (bytes > cfg.end / 2) break;
    }
    for (auto comm : comms) ncclCheck(ncclCommDestroy(comm));
  } catch (std::exception const& error) {
    std::fprintf(stderr, "resident collectives: %s\n", error.what());
    return 1;
  }
}

#endif

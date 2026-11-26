// Paired experimental BF16 mini-host/resident/router/NCCL benchmark.
#include "compact_transport.cuh"
#include "host_ring.cuh"
#include "uccl_lite/collective_dispatch.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <nccl.h>
#include <sched.h>

using namespace uccl_lite;
using namespace uccl_lite::compact_experiment;
namespace {
void ncclCheck(ncclResult_t e) {
  if (e != ncclSuccess) throw std::runtime_error(ncclGetErrorString(e));
}
// All sums, including hierarchical BF16 partials, are exactly representable.
// Independent host-double validation is also available in --check mode.
__host__ __device__ float value(size_t i, int rank, unsigned generation) {
  uint32_t x = uint32_t(i) ^ (uint32_t(rank + 1) * 0x9e3779b9u) ^
               (generation * 0x85ebca6bu);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  return (int(x % 63) - 31) / 32.0f;
}
__global__ void fillData(__nv_bfloat16* in, __nv_bfloat16* out, size_t n,
                         size_t result, int rank, unsigned gen) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < result + 16;
       i += gridDim.x * blockDim.x)
    out[i] = __float2bfloat16(-123);
  // In-place calls fill the input in a separate invocation.
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    in[i] = __float2bfloat16(value(i, rank, gen));
}
__global__ void verify(__nv_bfloat16 const* out, size_t count, int rank,
                       int ranks, int op, unsigned gen, unsigned* errors) {
  size_t shard = count / ranks, n = op == 1 ? shard : count;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n + 16;
       i += gridDim.x * blockDim.x) {
    double expected = -123;
    if (i < n) {
      if (op == 0)
        expected = value(i % shard, int(i / shard), gen);
      else {
        expected = 0;
        for (int r = 0; r < ranks; ++r)
          expected += value(op == 1 ? rank * shard + i : i, r, gen);
      }
    }
    if (__bfloat162float(out[i]) != expected) atomicAdd(errors, 1u);
  }
}
__global__ void delay(unsigned long long cycles) {
  auto start = clock64();
  while (clock64() - start < cycles) __nanosleep(64);
}
struct Buffers {
  int ranks;
  size_t capacity;
  std::vector<int> ids;
  std::vector<PacketPort> ports;
  std::vector<__nv_bfloat16*> inputs, outputs;
  std::vector<unsigned*> errors;
  std::vector<cudaEvent_t> begins, ends;
  Buffers(int n, size_t bytes) : ranks(n), capacity(bytes) {
    for (int r = 0; r < n; ++r) {
      ids.push_back(r);
      check(cudaSetDevice(r));
      __nv_bfloat16 *in, *out;
      unsigned* err;
      cudaStream_t stream;
      cudaEvent_t a, b;
      check(cudaMalloc(&in, bytes + 32));
      check(cudaMalloc(&out, bytes + 32));
      check(cudaMalloc(&err, sizeof(unsigned)));
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      check(cudaEventCreate(&a));
      check(cudaEventCreate(&b));
      inputs.push_back(in);
      outputs.push_back(out);
      errors.push_back(err);
      begins.push_back(a);
      ends.push_back(b);
      ports.push_back({in, out, stream, r});
    }
  }
  void bind(int op) {
    for (int r = 0; r < ranks; ++r)
      ports[r].output = op == 2 ? inputs[r] : outputs[r];
  }
  void sync() {
    for (auto const& p : ports) {
      check(cudaSetDevice(p.device));
      check(cudaStreamSynchronize(p.stream));
    }
  }
  void warmKernels() {
    // Load every producer/check kernel and runtime copy/memset path on every
    // device before starting persistent work. Lazy module loads may otherwise
    // synchronize that context, including its never-ending resident kernel.
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(r));
      delay<<<1, 32, 0, ports[r].stream>>>(1);
      fillData<<<1, 32, 0, ports[r].stream>>>(inputs[r], outputs[r], 0, 0, r,
                                              0);
      check(cudaMemsetAsync(errors[r], 0, sizeof(unsigned), ports[r].stream));
      verify<<<1, 32, 0, ports[r].stream>>>(outputs[r], 0, r, ranks, 0, 0,
                                            errors[r]);
      unsigned unused;
      check(cudaMemcpyAsync(&unused, errors[r], sizeof(unused),
                            cudaMemcpyDeviceToHost, ports[r].stream));
      check(cudaStreamSynchronize(ports[r].stream));
    }
  }
  void fill(int op, size_t count, unsigned gen, int delayed = -1) {
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(r));
      if (r == delayed) delay<<<1, 32, 0, ports[r].stream>>>(100000);
      size_t inputCount = op == 0 ? count / ranks : count;
      size_t outCount = op == 1 ? count / ranks : count;
      // Fill output/guard independently so no cross-CTA initialization race
      // is possible for in-place AR.
      fillData<<<64, 256, 0, ports[r].stream>>>(inputs[r], outputs[r],
                                                inputCount, outCount, r, gen);
      if (op == 2) {
        check(cudaMemsetAsync(inputs[r] + count, 0xf6, 32, ports[r].stream));
        // Use the same finite sentinel as every other operation.
        fillData<<<1, 32, 0, ports[r].stream>>>(inputs[r], inputs[r] + count, 0,
                                                0, r, gen);
      }
      check(cudaGetLastError());
    }
  }
  void validate(int op, size_t count, unsigned gen, bool host = false) {
    // Enqueue every rank's independent checker before waiting for any rank.
    // This preserves full per-call validation without serializing eight large
    // checks on the host. All checking remains outside the collective timer.
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(r));
      auto* out = static_cast<__nv_bfloat16*>(ports[r].output);
      check(cudaMemsetAsync(errors[r], 0, sizeof(unsigned), ports[r].stream));
      verify<<<64, 256, 0, ports[r].stream>>>(out, count, r, ranks, op, gen,
                                              errors[r]);
    }
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(r));
      auto* out = static_cast<__nv_bfloat16*>(ports[r].output);
      unsigned bad;
      check(cudaMemcpyAsync(&bad, errors[r], sizeof(bad),
                            cudaMemcpyDeviceToHost, ports[r].stream));
      check(cudaStreamSynchronize(ports[r].stream));
      if (bad)
        throw std::runtime_error("output/guard mismatch rank " +
                                 std::to_string(r) + " gen " +
                                 std::to_string(gen));
      if (host) {
        size_t shard = count / ranks, n = op == 1 ? shard : count;
        std::vector<__nv_bfloat16> got(n + 16);
        check(cudaMemcpyAsync(got.data(), out, got.size() * 2,
                              cudaMemcpyDeviceToHost, ports[r].stream));
        check(cudaStreamSynchronize(ports[r].stream));
        for (size_t i = 0; i < got.size(); ++i) {
          double expected = -123;
          if (i < n) {
            expected = 0;
            if (op == 0)
              expected = value(i % shard, int(i / shard), gen);
            else
              for (int s = 0; s < ranks; ++s)
                expected += double(value(op == 1 ? r * shard + i : i, s, gen));
          }
          if (double(__bfloat162float(got[i])) != expected)
            throw std::runtime_error("independent host oracle mismatch");
        }
      }
    }
  }
  ~Buffers() {
    sync();
    for (int r = 0; r < ranks; ++r) {
      cudaSetDevice(r);
      cudaFree(inputs[r]);
      cudaFree(outputs[r]);
      cudaFree(errors[r]);
      cudaEventDestroy(begins[r]);
      cudaEventDestroy(ends[r]);
      cudaStreamDestroy(ports[r].stream);
    }
  }
};
void enqueueNccl(Buffers& b, std::vector<ncclComm_t> const& comms, int op,
                 size_t count) {
  ncclCheck(ncclGroupStart());
  for (int r = 0; r < b.ranks; ++r) {
    check(cudaSetDevice(r));
    auto const& p = b.ports[r];
    if (op == 0)
      ncclCheck(ncclAllGather(p.input, p.output, count / b.ranks, ncclBfloat16,
                              comms[r], p.stream));
    else if (op == 1)
      ncclCheck(ncclReduceScatter(p.input, p.output, count / b.ranks,
                                  ncclBfloat16, ncclSum, comms[r], p.stream));
    else
      ncclCheck(ncclAllReduce(p.input, p.output, count, ncclBfloat16, ncclSum,
                              comms[r], p.stream));
  }
  ncclCheck(ncclGroupEnd());
}
struct NcclGraph {
  cudaGraphExec_t exec = nullptr;
  std::vector<cudaEvent_t> join;
  cudaEvent_t fork;
  Buffers& b;
  NcclGraph(Buffers& buffers, std::vector<ncclComm_t> const& comms, int op,
            size_t count)
      : b(buffers) {
    enqueueNccl(b, comms, op, count);
    b.sync();
    for (int r = 0; r < b.ranks; ++r) {
      check(cudaSetDevice(r));
      cudaEvent_t e;
      check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      join.push_back(e);
    }
    check(cudaSetDevice(0));
    check(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    auto origin = b.ports[0].stream;
    check(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
    check(cudaEventRecord(fork, origin));
    for (int r = 1; r < b.ranks; ++r) {
      check(cudaSetDevice(r));
      check(cudaStreamWaitEvent(b.ports[r].stream, fork, 0));
    }
    enqueueNccl(b, comms, op, count);
    for (int r = 1; r < b.ranks; ++r) {
      check(cudaSetDevice(r));
      check(cudaEventRecord(join[r], b.ports[r].stream));
    }
    check(cudaSetDevice(0));
    for (int r = 1; r < b.ranks; ++r)
      check(cudaStreamWaitEvent(origin, join[r], 0));
    cudaGraph_t graph;
    check(cudaStreamEndCapture(origin, &graph));
    check(cudaGraphInstantiate(&exec, graph, 0));
    check(cudaGraphDestroy(graph));
  }
  void launch() {
    check(cudaSetDevice(0));
    check(cudaGraphLaunch(exec, b.ports[0].stream));
  }
  ~NcclGraph() {
    cudaSetDevice(0);
    cudaGraphExecDestroy(exec);
    cudaEventDestroy(fork);
    for (int r = 0; r < b.ranks; ++r) {
      cudaSetDevice(r);
      cudaEventDestroy(join[r]);
    }
  }
};
std::vector<std::string> split(std::string const& s) {
  std::vector<std::string> out;
  std::stringstream in(s);
  std::string item;
  while (std::getline(in, item, ',')) out.push_back(item);
  return out;
}
CollectiveCall callFor(Buffers& b, int op, size_t count) {
  CollectiveCall c;
  c.collective = op == 0   ? Collective::AllGather
                 : op == 1 ? Collective::ReduceScatter
                           : Collective::AllReduce;
  c.count = op == 2 ? count : count / b.ranks;
  c.inPlace = op == 2;
  c.ports = b.ports;
  return c;
}
void correctness(int ranks) {
  size_t capacity = 1UL << 20;
  Buffers b(ranks, capacity);
  size_t calls = 0;
  // Distinct allocations preserve each queued result and each input producer.
  for (size_t bytes : {size_t(16 * ranks), size_t(132 * ranks), size_t(32768),
                       size_t(65536), capacity}) {
    std::fprintf(stderr, "check mini bytes=%zu\n", bytes);
    std::vector<std::unique_ptr<Buffers>> queue;
    for (int q = 0; q < 4; ++q)
      queue.emplace_back(std::make_unique<Buffers>(ranks, capacity));
    // The protocol requires fixed ordered caller streams; use b's streams.
    Transport mini(b.ports, capacity, 2 * ranks);
    for (int round = 0; round < 3; ++round) {
      for (int q = 0; q < 4; ++q) {
        auto& x = *queue[q];
        x.fill(0, bytes / 2, round * 4 + q + 1);
        x.sync();
        auto ports = x.ports;
        for (int r = 0; r < ranks; ++r) ports[r].stream = b.ports[r].stream;
        check(cudaSetDevice(q % ranks));
        delay<<<1, 32, 0, b.ports[q % ranks].stream>>>(100000);
        mini.gather(bytes / 2, ports);
      }
      b.sync();
      for (int q = 0; q < 4; ++q) {
        queue[q]->validate(0, bytes / 2, round * 4 + q + 1, true);
        ++calls;
      }
    }
  }
  // A fixed out-of-place binding tests mixed operations, changed extents,
  // repeated start/stop and delayed producer streams. In-place AR is exercised
  // by the paired benchmark on every invocation.
  Transport persistent(b.ports, capacity, 2 * ranks);
  b.warmKernels();
  for (int round = 0; round < 3; ++round) {
    std::fprintf(stderr, "check resident round=%d start\n", round);
    persistent.start();
    for (size_t bytes : {size_t(16 * ranks), size_t(132 * ranks), size_t(2048),
                         size_t(8192), size_t(65536), capacity})
      for (int op = 0; op < 3; ++op) {
        std::fprintf(stderr, "check resident bytes=%zu op=%d\n", bytes, op);
        unsigned gen = unsigned(++calls);
        b.fill(op, bytes / 2, gen, int(calls % ranks));
        b.sync();
        persistent.run(op, bytes / 2);
        b.validate(op, bytes / 2, gen, true);
      }
    persistent.stop();
  }
  bool rejected = false;
  try {
    persistent.run(0, capacity / 2 + 1);
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("invalid extent accepted");
  std::printf(
      "PASS compact check ranks=%d calls=%zu: queued mini, mixed resident, "
      "tails, guards, delayed producers, restart\n",
      ranks, calls);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    int ranks = 8, op = 0, warmup = 10, iters = 20, repeats = 3, blocks = 16,
        idle = 0;
    size_t begin = 1024, end = 1UL << 20;
    bool host = false, checking = false, reverse = false, shmOnly = false;
    size_t rsPacket = 128UL << 10, rsMinPacket = 128UL << 10;
    size_t rsGrain = 8192, rsTransportGrain = 16UL << 10;
    int rsWorkers = 4, rsBlocks = 64, rsThreads = 256;
    bool rsGpu = false, rsInterleave = false, rsDirect = false;
    auto variants = split("nccl,router-tuned,mini");
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() {
        if (++i >= argc) throw std::invalid_argument("missing option value");
        return std::string(argv[i]);
      };
      if (a == "--gpus")
        ranks = std::stoi(next());
      else if (a == "--op") {
        auto s = next();
        op = s == "ag" ? 0 : s == "rs" ? 1 : s == "ar" ? 2 : -1;
      } else if (a == "--begin")
        begin = std::stoull(next());
      else if (a == "--end")
        end = std::stoull(next());
      else if (a == "--warmup")
        warmup = std::stoi(next());
      else if (a == "--iterations")
        iters = std::stoi(next());
      else if (a == "--repeats")
        repeats = std::stoi(next());
      else if (a == "--blocks")
        blocks = std::stoi(next());
      else if (a == "--idle-us")
        idle = std::stoi(next());
      else if (a == "--variants")
        variants = split(next());
      else if (a == "--host-timing")
        host = true;
      else if (a == "--check")
        checking = true;
      else if (a == "--reverse")
        reverse = true;
      else if (a == "--shm-only")
        shmOnly = true;
      else if (a == "--rs-packet-bytes")
        rsPacket = std::stoull(next());
      else if (a == "--rs-min-packet-bytes")
        rsMinPacket = std::stoull(next());
      else if (a == "--rs-workers")
        rsWorkers = std::stoi(next());
      else if (a == "--rs-blocks")
        rsBlocks = std::stoi(next());
      else if (a == "--rs-threads")
        rsThreads = std::stoi(next());
      else if (a == "--rs-cpu-grain")
        rsGrain = std::stoull(next());
      else if (a == "--rs-transport-grain")
        rsTransportGrain = std::stoull(next());
      else if (a == "--rs-gpu")
        rsGpu = true;
      else if (a == "--rs-interleave")
        rsInterleave = true;
      else if (a == "--rs-direct")
        rsDirect = true;
      else
        throw std::invalid_argument("unknown option " + a);
    }
    if (ranks != 4 && ranks != 8)
      throw std::invalid_argument("use four or eight GPUs");
    if (shmOnly) {
      if (setenv("NCCL_P2P_DISABLE", "1", 1) ||
          setenv("NCCL_SHM_DISABLE", "0", 1))
        throw std::runtime_error("cannot configure NCCL SHM");
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(r));
        for (int peer = 0; peer < ranks; ++peer) {
          if (peer == r) continue;
          auto status = cudaDeviceDisablePeerAccess(peer);
          if (status == cudaErrorPeerAccessNotEnabled)
            cudaGetLastError();
          else
            check(status);
        }
      }
    }
    if (checking) {
      correctness(ranks);
      return 0;
    }
    if (op < 0 || begin < 16 * size_t(ranks) || begin % (2 * ranks) ||
        end < begin || end % (2 * ranks) || end > (1UL << 30) || warmup < 1 ||
        iters < 1 || repeats < 1 || idle < 0)
      throw std::invalid_argument("invalid benchmark configuration");
    for (auto const& v : variants) {
      if (v != "nccl" && v != "nccl-graph" && v != "router-base" &&
          v != "router-progress" && v != "router-tuned" &&
          v != "router-single" && v != "router-full-ag" &&
          v != "router-full-ag-compact" && v != "router-bulk" &&
          v != "current" && v != "previous" && v != "mini" && v != "resident" &&
          v != "router-ring" && v != "rs-candidate" && v != "rs-ring" &&
          v != "rs-halving" && v != "rs-ring-cta" && v != "rs-hier")
        throw std::invalid_argument("unknown variant " + v);
      if ((v == "rs-candidate" || v == "router-ring") && op != 1)
        throw std::invalid_argument("RS candidate requires ReduceScatter");
      if ((v == "rs-ring" || v == "rs-halving" || v == "rs-ring-cta" ||
           v == "rs-hier") &&
          (op != 1 || end > (16UL << 20)))
        throw std::invalid_argument(
            "RS ring prototype requires RS through 16 MiB");
      if (v == "mini" && op != 0)
        throw std::invalid_argument("mini supports AG only");
      if ((v == "mini" || v == "resident") && end > (16UL << 20))
        throw std::invalid_argument(
            "standalone mini/resident extent exceeds tested capacity");
      if ((v == "router-full-ag" || v == "router-full-ag-compact") && op != 0)
        throw std::invalid_argument("full AG progress supports AG only");
      if ((v == "resident" || v == "nccl-graph") && !host)
        throw std::invalid_argument("resident/graph require host timing");
    }
    if (reverse) std::reverse(variants.begin(), variants.end());
    Buffers b(ranks, end);
    b.bind(op);
    b.warmKernels();
    // Inspect the full CPU allowance before pinning the submission thread.
    auto topology = inspectTopology(b.ids);
    if (shmOnly) {
      std::fill(topology.peerAccess.begin(), topology.peerAccess.end(), false);
      topology.identity += ";logical-no-p2p-shm-v1";
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
      throw std::runtime_error("get affinity");
    if (!CPU_ISSET(47, &allowed))
      throw std::runtime_error("CPU 47 unavailable");
    cpu_set_t pin;
    CPU_ZERO(&pin);
    CPU_SET(47, &pin);
    if (sched_setaffinity(0, sizeof(pin), &pin))
      throw std::runtime_error("set affinity");
    std::vector<ncclComm_t> comms(ranks);
    ncclCheck(ncclCommInitAll(comms.data(), ranks, b.ids.data()));
    int version;
    ncclCheck(ncclGetVersion(&version));
    std::printf(
        "# BF16 %d GPUs op=%d AR-in-place AG/RS-out-of-place NCCL=%d "
        "input-ready %s timing CPU=47 idle_us=%d blocks=%d warmup=%d "
        "iterations=%d repeats=%d\n",
        ranks, op, version,
        host ? "host-global-completion" : "slowest-GPU-event", idle, blocks,
        warmup, iters, repeats);
    std::printf("# testbed=%s; host-packet UCCL transport; NCCL P2P %s\n",
                shmOnly ? "shm" : "gpu", shmOnly ? "disabled" : "default");
    std::puts("kind,bytes,variant,repeat,iteration,us");
    unsigned gen = 0;
    std::vector<size_t> sizes;
    for (size_t bytes = begin; bytes <= end; bytes *= 2) {
      sizes.push_back(bytes);
      if (bytes > end / 2) break;
    }
    if (reverse) std::reverse(sizes.begin(), sizes.end());
    for (size_t bytes : sizes) {
      size_t count = bytes / 2;
      auto call = callFor(b, op, count);
      for (int repeat = 0; repeat < repeats; ++repeat) {
        for (size_t j = 0; j < variants.size(); ++j) {
          auto const& requested = variants[(j + repeat) % variants.size()];
          std::string name = requested;
          if (name == "current" || name == "previous") {
            if (name == "current" && op == 1 && bytes >= (1UL << 20) &&
                bytes <= (8UL << 20))
              name = "router-ring";
            else if (op == 0 && bytes >= (256UL << 10) && bytes <= (1UL << 20))
              name = "mini";
            else if (op == 0 && bytes >= (2UL << 20))
              name = "router-full-ag";
            else if (op == 2 && bytes > (4UL << 20))
              name = "router-bulk";
            else
              name = "router-tuned";
          }
          std::printf("# implementation,%zu,%s,%s\n", bytes, requested.c_str(),
                      name.c_str());
          std::unique_ptr<Transport> compact;
          std::unique_ptr<HostRing> ring;
          std::unique_ptr<CollectiveDispatcher> router;
          std::unique_ptr<NcclGraph> graph;
          std::function<void()> invoke;
          if (name == "rs-ring" || name == "rs-halving" ||
              name == "rs-ring-cta" || name == "rs-hier") {
            ring = std::make_unique<HostRing>(
                b.ports, end, blocks, name == "rs-halving",
                name == "rs-ring-cta", name == "rs-hier");
            invoke = [&]() { ring->scatter(count, b.ports); };
          } else if (name == "mini" || name == "resident") {
            compact = std::make_unique<Transport>(
                b.ports, std::min(end, 16UL << 20), blocks);
            if (name == "resident") {
              compact->start();
              invoke = [&]() { compact->run(op, count); };
            } else
              invoke = [&]() { compact->gather(count, b.ports); };
          } else if (name == "nccl")
            invoke = [&]() { enqueueNccl(b, comms, op, count); };
          else if (name == "nccl-graph") {
            b.fill(op, count, ++gen);
            b.sync();
            graph = std::make_unique<NcclGraph>(b, comms, op, count);
            invoke = [&]() { graph->launch(); };
          } else {
            cpu_router::CpuRouterOptions options;
            options.maxTensorBytes = end;
            options.smallAllReducePacketBytes = 128UL << 10;
            options.smallReduceScatterPacketBytes = 128UL << 10;
            options.smallAllReduceWorkersPerNode = 4;
            options.smallAllReduceTransportBlocks = 16;
            options.largeAllGatherPacketBytes = 2UL << 20;
            if (name != "router-base") {
              options.batchCpuJobs = true;
              options.smallReduceScatterWorkersPerNode = 4;
              options.smallAllReduceGpuProgressMaxBytes = 8192;
              options.smallAllGatherGpuProgressMaxBytes = 1UL << 20;
              options.smallReduceScatterGpuProgressMaxBytes = 65536;
              if (op == 2 && name != "router-bulk") {
                options.workersPerNode = 4;
                options.transportBlocks = 16;
              }
            }
            if (name == "router-tuned" || name == "router-single" ||
                name == "router-full-ag" || name == "router-full-ag-compact" ||
                name == "router-bulk" || name == "rs-candidate")
              options.gpuLaunchWorkersPerNode = 4;
            if (name == "router-single") options.singleCtaBytes = 65536;
            if (name == "router-full-ag" || name == "router-full-ag-compact") {
              options.reduction = cpu_router::ReductionPlacement::Gpu;
              options.gpuProgress = true;
              options.agPerSourceProgress = true;
              if (name == "router-full-ag-compact")
                options.transportBlocks = 16;
            }
            if (name == "rs-candidate") {
              options.packetBytes = rsPacket;
              options.minPacketBytes = rsMinPacket;
              options.smallAllReducePacketBytes = 0;
              options.smallReduceScatterPacketBytes = 0;
              options.smallReduceScatterGpuProgress = false;
              options.smallReduceScatterGpuProgressMaxBytes = 0;
              options.smallReduceScatterWorkersPerNode = 0;
              options.scatterWorkersPerNode = rsWorkers;
              options.transportBlocks = rsBlocks;
              options.transportThreads = rsThreads;
              options.transportGrainBytes = rsTransportGrain;
              options.grainElements = rsGrain;
              options.directScatterResults = rsDirect;
              options.interleaveScatter = rsInterleave;
              options.gpuProgress = rsGpu;
              options.reduction = rsGpu ? cpu_router::ReductionPlacement::Gpu
                                        : cpu_router::ReductionPlacement::Cpu;
              std::printf(
                  "# RS candidate packet=%zu min=%zu workers=%d blocks=%d "
                  "threads=%d cpu-grain=%zu transport-grain=%zu gpu=%d "
                  "striped=%d direct=%d\n",
                  rsPacket, rsMinPacket, rsWorkers, rsBlocks, rsThreads,
                  rsGrain, rsTransportGrain, int(rsGpu), int(rsInterleave),
                  int(rsDirect));
            }
            if (name == "router-ring") {
              options.hostRingReduceScatter = true;
              options.gpuProgress = true;
              options.reduction = cpu_router::ReductionPlacement::Gpu;
              options.packetBytes =
                  std::max(size_t(128UL << 10), bytes / ranks);
              options.transportBlocks = blocks;
              options.gpuLaunchWorkersPerNode = 4;
              options.smallReduceScatterGpuProgress = false;
              options.smallReduceScatterGpuProgressMaxBytes = 0;
            }
            BackendPolicy policy;
            policy.mode = RouterMode::Force;
            router = std::make_unique<CollectiveDispatcher>(
                topology, policy,
                [](CollectiveCall const&) {
                  throw std::logic_error("unexpected fallback");
                },
                options);
            router->prepare(call);
            invoke = [&]() { router->enqueue(call); };
          }
          std::vector<double> times;
          for (int it = -warmup; it < iters; ++it) {
            if (idle)
              std::this_thread::sleep_for(std::chrono::microseconds(idle));
            b.fill(op, count, ++gen);
            b.sync();
            if (!host)
              for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(r));
                check(cudaEventRecord(b.begins[r], b.ports[r].stream));
              }
            if (host) check(cudaSetDevice(0));
            auto start = std::chrono::steady_clock::now();
            invoke();
            if (!host)
              for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(r));
                check(cudaEventRecord(b.ends[r], b.ports[r].stream));
              }
            // Resident run already returns after all ranks' system-release
            // completion. CUDA synchronization would add unrelated API cost.
            if (name == "nccl-graph") {
              check(cudaStreamSynchronize(b.ports[0].stream));
            } else if (name != "resident")
              b.sync();
            double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - start)
                            .count();
            if (!host) {
              us = 0;
              for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(r));
                float ms;
                check(cudaEventElapsedTime(&ms, b.begins[r], b.ends[r]));
                us = std::max(us, double(ms) * 1000);
              }
            }
            b.validate(op, count, gen);
            if (it >= 0) {
              times.push_back(us);
              std::printf("sample,%zu,%s,%d,%d,%.6f\n", bytes,
                          requested.c_str(), repeat, it, us);
            }
          }
          if (compact) compact->stop();
          std::sort(times.begin(), times.end());
          double median =
              (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
          std::printf("median,%zu,%s,%d,-1,%.6f\n", bytes, requested.c_str(),
                      repeat, median);
          std::fflush(stdout);
        }
      }
    }
    for (auto comm : comms) ncclCheck(ncclCommDestroy(comm));
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "compact experiment: %s\n", e.what());
    return 1;
  }
}

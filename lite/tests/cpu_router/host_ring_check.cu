#include "uccl_lite/cpu_router/packet_pipeline.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <immintrin.h>
using namespace uccl_lite::cpu_router;
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
float decode(uint16_t x) {
  uint32_t u = uint32_t(x) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t encode(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffU) > 0x7f800000U)
    u |= 0x00400000U;
  else
    u += 0x7fffU + ((u >> 16) & 1U);
  return u >> 16;
}
__host__ __device__ uint16_t pattern(size_t i, int rank, unsigned epoch) {
  // Odd multiplier enumerates all 65,536 encodings in each rank. Other epochs
  // permute signs/exponents and include cancellation and overflow cases.
  return uint16_t(i * (epoch * 2 + 1) + rank * 193 + epoch * 7919);
}
__global__ void fillRaw(uint16_t* p, size_t n, int rank, unsigned epoch) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    p[i] = pattern(i, rank, epoch);
}
__global__ void fillExact(__nv_bfloat16* p, __nv_bfloat16* out, size_t n,
                          size_t shard, int rank, unsigned epoch) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    p[i] = __float2bfloat16_rn(float(int((i + epoch) % 7) - 3 + rank) / 32);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < shard + 8;
       i += gridDim.x * blockDim.x)
    out[i] = __float2bfloat16_rn(-123);
}
__global__ void validateExact(__nv_bfloat16 const* out, size_t shard, int rank,
                              int ranks, unsigned epoch, unsigned* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < shard + 8;
       i += gridDim.x * blockDim.x) {
    float want = -123;
    if (i < shard) {
      int v = int((rank * shard + i + epoch) % 7) - 3;
      want = float(ranks * v + ranks * (ranks - 1) / 2) / 32;
    }
    if (__bfloat162float(out[i]) != want) atomicAdd(errors, 1U);
  }
}
template <class F>
void reject(F&& f) {
  bool rejected = false;
  try {
    f();
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("unsupported ring call accepted");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    int ranks = argc > 1 ? std::stoi(argv[1]) : 8;
    if (ranks != 4 && ranks != 8)
      throw std::invalid_argument("use 4 or 8 ranks");
    unsigned mxcsr = _mm_getcsr();
    _mm_setcsr(mxcsr &
               ~unsigned(0x8040));  // Independent CPU oracle: no FTZ/DAZ.
    int available = 0;
    check(cudaGetDeviceCount(&available));
    if (available < ranks) throw std::runtime_error("not enough GPUs");
    std::vector<int> devices =
        ranks == 4 && available >= 8 ? std::vector<int>{0, 1, 4, 5}
        : ranks == 4                 ? std::vector<int>{0, 1, 2, 3}
                                     : std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
    constexpr size_t maxShard = 65536;
    std::vector<PacketPort> ports;
    std::vector<unsigned*> errors(ranks);
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices[r]));
      void *in, *out;
      cudaStream_t stream;
      check(cudaMalloc(&in, maxShard * ranks * 2));
      check(cudaMalloc(&out, (maxShard + 8) * 2));
      check(cudaMalloc(&errors[r], sizeof(unsigned)));
      check(cudaMemset(errors[r], 0, sizeof(unsigned)));
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      ports.push_back({in, out, stream, devices[r]});
    }
    auto sync = [&] {
      for (auto const& p : ports) {
        check(cudaSetDevice(p.device));
        check(cudaStreamSynchronize(p.stream));
      }
    };
    PacketPipelineOptions options;
    options.hostRingReduceScatter = true;
    options.reduction = ReductionPlacement::Gpu;
    options.gpuProgress = true;
    options.transportBlocks = 16;
    options.packetBytes = options.minPacketBytes = maxShard * 2;
    options.maxTensorBytes = maxShard * ranks * 2;
    options.workersPerNode = 2;
    options.gpuLaunchWorkersPerNode = 2;
    auto router = std::make_unique<PacketPipeline>(options);
    router->prepare(ports);
    size_t checked = 0;
    for (unsigned epoch : {0U, 1U, 17U, 113U}) {
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(devices[r]));
        fillRaw<<<32, 256, 0, ports[r].stream>>>(
            static_cast<uint16_t*>(const_cast<void*>(ports[r].input)),
            maxShard * ranks, r, epoch);
      }
      router->enqueue(PacketProgram::reduceScatter(maxShard, ranks), ports);
      sync();
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(devices[r]));
        std::vector<uint16_t> got(maxShard);
        check(cudaMemcpy(got.data(), ports[r].output, maxShard * 2,
                         cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < maxShard; ++i) {
          size_t offset = r * maxShard + i;
          uint16_t want = pattern(offset, (r + 1) % ranks, epoch);
          for (int step = 2; step <= ranks; ++step)
            want = encode(decode(want) +
                          decode(pattern(offset, (r + step) % ranks, epoch)));
          if (std::isnan(decode(want)) ? !std::isnan(decode(got[i]))
                                       : got[i] != want) {
            std::fprintf(stderr, "rank=%d i=%zu epoch=%u got=%04x want=%04x\n",
                         r, i, epoch, got[i], want);
            throw std::runtime_error("host ring scalar oracle mismatch");
          }
          ++checked;
        }
      }
    }
    auto exact = [&](size_t shard, unsigned epoch) {
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(devices[r]));
        fillExact<<<8, 256, 0, ports[r].stream>>>(
            static_cast<__nv_bfloat16*>(const_cast<void*>(ports[r].input)),
            static_cast<__nv_bfloat16*>(ports[r].output), shard * ranks, shard,
            r, epoch);
      }
      router->enqueue(PacketProgram::reduceScatter(shard, ranks), ports);
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(devices[r]));
        validateExact<<<8, 256, 0, ports[r].stream>>>(
            static_cast<__nv_bfloat16 const*>(ports[r].output), shard, r, ranks,
            epoch, errors[r]);
      }
    };
    // Rejection must not advance any command or ownership generation.
    auto plan = PacketProgram::reduceScatter(256, ranks);
    reject(
        [&] { router->enqueue(PacketProgram::allReduce(256, ranks), ports); });
    auto masked = plan;
    masked.sources ^= 1;
    reject([&] { router->enqueue(masked, ports); });
    reject([&] {
      router->enqueue(PacketProgram::reduceScatter(257, ranks), ports);
    });
    auto bad = ports;
    bad[0].output = const_cast<void*>(bad[0].input);
    reject([&] { router->enqueue(plan, bad); });
    bad = ports;
    bad[0].output = static_cast<char*>(bad[0].output) + 2;
    reject([&] { router->enqueue(plan, bad); });
    exact(256, 7);
    // Queue depth is 128: this crosses it repeatedly without host GPU waits.
    constexpr size_t extents[] = {8, 256, 16384, 24, 65536, 512};
    for (unsigned i = 0; i < 270; ++i) exact(extents[i % 6], i);
    sync();
    router->beginCapture();
    std::vector<cudaGraph_t> graphs(ranks);
    std::vector<cudaGraphExec_t> execs(ranks);
    for (auto const& p : ports) {
      check(cudaSetDevice(p.device));
      check(cudaStreamBeginCapture(p.stream, cudaStreamCaptureModeRelaxed));
    }
    for (unsigned i = 0; i < 140; ++i) exact(extents[i % 6], i + 1000);
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices[r]));
      check(cudaStreamEndCapture(ports[r].stream, &graphs[r]));
      check(cudaGraphInstantiate(&execs[r], graphs[r], 0));
    }
    router->endCapture();
    for (int replay = 0; replay < 3; ++replay) {
      router->beginReplay();
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(devices[r]));
        check(cudaGraphLaunch(execs[r], ports[r].stream));
      }
      sync();
      router->endReplay();
    }
    exact(24, 2000);
    sync();
    router.reset();
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices[r]));
      unsigned count;
      check(
          cudaMemcpy(&count, errors[r], sizeof(count), cudaMemcpyDeviceToHost));
      if (count)
        throw std::runtime_error("async/graph output or guard mismatch");
      check(cudaGraphExecDestroy(execs[r]));
      check(cudaGraphDestroy(graphs[r]));
      check(cudaStreamDestroy(ports[r].stream));
      check(cudaFree(const_cast<void*>(ports[r].input)));
      check(cudaFree(ports[r].output));
      check(cudaFree(errors[r]));
    }
    _mm_setcsr(mxcsr);
    std::printf(
        "host ring %d ranks: %zu independent scalar checks, 270 asynchronous "
        "calls, 3 x 140 graph calls, changing extents, guards and rejection "
        "recovery passed\n",
        ranks, checked);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "host ring check: %s\n", e.what());
    return 1;
  }
}

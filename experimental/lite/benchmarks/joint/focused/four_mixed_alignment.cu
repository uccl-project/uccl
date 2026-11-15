// Frozen common BF16 benchmark: NCCL_REFERENCE_ONLY removes all solution links.
#include "../cpu_router/nccl_identity.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#ifndef NCCL_REFERENCE_ONLY
#include "uccl_lite/joint_collectives.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
extern char **environ;
namespace {
void ck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
void nk(ncclResult_t e) {
  if (e != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(e));
}
char const *opName(int op) {
  return op == 0 ? "allreduce" : op == 1 ? "allgather" : "reducescatter";
}
__host__ __device__ unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
// All sums of 2/4 ranks are exactly BF16-representable; independent generation
// and rank/index hash prevents constant-input, stale-output and permutation
// bugs.
__host__ __device__ float value(size_t index, int rank, unsigned generation) {
  unsigned h = mix(unsigned(index) ^ mix(unsigned(index >> 32)) ^
                   mix(unsigned(rank + 1) * 8179U) ^ mix(generation));
  return float(int(h % 63) - 31) / 32.0f;
}
constexpr size_t kGuard = 128;
constexpr unsigned short kCanary = 0x55aa;
__global__ void initialize(__nv_bfloat16 *in, __nv_bfloat16 *out, size_t inputs,
                           size_t outputs, int rank, unsigned generation,
                           unsigned long long delay) {
  if (delay) {
    unsigned long long begin = clock64();
    while (clock64() - begin < delay) {
    }
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputs;
       i += blockDim.x * gridDim.x)
    in[i] = __float2bfloat16(value(i, rank, generation));
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outputs;
       i += blockDim.x * gridDim.x)
    reinterpret_cast<unsigned short *>(out)[i] = 0x7fc1;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kGuard;
       i += blockDim.x * gridDim.x) {
    reinterpret_cast<unsigned short *>(in)[ptrdiff_t(i) - ptrdiff_t(kGuard)] =
        kCanary;
    reinterpret_cast<unsigned short *>(in)[inputs + i] = kCanary;
    reinterpret_cast<unsigned short *>(out)[ptrdiff_t(i) - ptrdiff_t(kGuard)] =
        kCanary;
    reinterpret_cast<unsigned short *>(out)[outputs + i] = kCanary;
  }
}
__global__ void verify(__nv_bfloat16 const *in, __nv_bfloat16 const *out,
                       size_t inputs, size_t outputs, int rank, int ranks,
                       int op, size_t count, unsigned generation, int *errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outputs;
       i += blockDim.x * gridDim.x) {
    float expected = 0;
    if (op == 1)
      expected = value(i % count, int(i / count), generation);
    else
      for (int source = 0; source < ranks; ++source)
        expected += value(op == 2 ? rank * count + i : i, source, generation);
    if (__bfloat162float(out[i]) != expected)
      atomicAdd(errors, 1);
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputs;
       i += blockDim.x * gridDim.x)
    if (__bfloat162float(in[i]) != value(i, rank, generation))
      atomicAdd(errors, 1);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kGuard;
       i += blockDim.x * gridDim.x) {
    auto *a = reinterpret_cast<unsigned short const *>(in);
    auto *b = reinterpret_cast<unsigned short const *>(out);
    if (a[ptrdiff_t(i) - ptrdiff_t(kGuard)] != kCanary ||
        a[inputs + i] != kCanary ||
        b[ptrdiff_t(i) - ptrdiff_t(kGuard)] != kCanary ||
        b[outputs + i] != kCanary)
      atomicAdd(errors, 1);
  }
}
struct Config {
  int gpus = 4, warmup = 100, repeats = 5, iterations = 20;
  size_t begin = 1024, end = 1UL << 30;
  unsigned seed = 20261114;
  std::string backend = "joint", operation = "all", profile, source, samples;
  bool reverse = false, shmOnly = false, selftest = false;
};
struct Buffers {
  std::vector<int> devices;
  std::vector<__nv_bfloat16 *> in, out;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> start, stop;
  std::vector<int *> errors;
  Buffers(int ranks, size_t bytes) {
    for (int r = 0; r < ranks; ++r) {
      ck(cudaSetDevice(r));
      devices.push_back(r);
      __nv_bfloat16 *a, *b;
      int *e;
      cudaStream_t s;
      cudaEvent_t x, y;
      ck(cudaMalloc(&a, bytes + 4 * kGuard + 16));
      ck(cudaMalloc(&b, bytes + 4 * kGuard + 16));
      ck(cudaMalloc(&e, sizeof(int)));
      ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      ck(cudaEventCreate(&x));
      ck(cudaEventCreate(&y));
      in.push_back(a + kGuard + (r == ranks - 1 ? 1 : 0));
      out.push_back(b + kGuard + (r == ranks - 1 ? 1 : 0));
      errors.push_back(e);
      streams.push_back(s);
      start.push_back(x);
      stop.push_back(y);
    }
  }
  ~Buffers() {
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      cudaStreamSynchronize(streams[r]);
      cudaFree(in[r] - kGuard - (r == devices.size() - 1 ? 1 : 0));
      cudaFree(out[r] - kGuard - (r == devices.size() - 1 ? 1 : 0));
      cudaFree(errors[r]);
      cudaEventDestroy(start[r]);
      cudaEventDestroy(stop[r]);
      cudaStreamDestroy(streams[r]);
    }
  }
#ifndef NCCL_REFERENCE_ONLY
  uccl_lite::CollectiveCall call(int op, size_t bytes) {
    uccl_lite::CollectiveCall c;
    c.collective = static_cast<uccl_lite::Collective>(op);
    c.count = bytes / 2 / (op == 0 ? 1 : devices.size());
    for (size_t r = 0; r < devices.size(); ++r)
      c.ports.push_back({in[r], out[r], streams[r], devices[r]});
    return c;
  }
#endif
};
double median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  size_t n = x.size();
  return n % 2 ? x[n / 2] : (x[n / 2 - 1] + x[n / 2]) / 2;
}
struct Stats {
  double minimum, median, maximum;
};
Stats summarize(std::vector<double> const &x) {
  return {*std::min_element(x.begin(), x.end()), median(x),
          *std::max_element(x.begin(), x.end())};
}
using Enqueue = std::function<void()>;
double invocation(Buffers &b, int op, size_t bytes, unsigned generation,
                  Enqueue const &enqueue, bool delayed = false) {
  int ranks = int(b.devices.size());
  size_t count = bytes / 2 / (op == 0 ? 1 : ranks);
  size_t inputs = count * (op == 2 ? ranks : 1),
         outputs = count * (op == 1 ? ranks : 1);
  for (int r = 0; r < ranks; ++r) {
    ck(cudaSetDevice(r));
    initialize<<<128, 256, 0, b.streams[r]>>>(
        b.in[r], b.out[r], inputs, outputs, r, generation,
        delayed && r == ranks - 1 ? 500000 : 0);
    ck(cudaMemsetAsync(b.errors[r], 0, sizeof(int), b.streams[r]));
    ck(cudaEventRecord(b.start[r], b.streams[r]));
  }
  // Deliberately no producer-stream synchronization before candidate enqueue.
  enqueue();
  for (int r = 0; r < ranks; ++r) {
    ck(cudaSetDevice(r));
    ck(cudaEventRecord(b.stop[r], b.streams[r]));
    verify<<<128, 256, 0, b.streams[r]>>>(b.in[r], b.out[r], inputs, outputs, r,
                                          ranks, op, count, generation,
                                          b.errors[r]);
    ck(cudaGetLastError());
  }
  double worst = 0;
  for (int r = 0; r < ranks; ++r) {
    ck(cudaSetDevice(r));
    ck(cudaStreamSynchronize(b.streams[r]));
    int errors;
    ck(cudaMemcpy(&errors, b.errors[r], sizeof(int), cudaMemcpyDeviceToHost));
    if (errors)
      throw std::runtime_error(std::string("correctness: ") + opName(op) +
                               " bytes=" + std::to_string(bytes) +
                               " rank=" + std::to_string(r) +
                               " generation=" + std::to_string(generation) +
                               " errors=" + std::to_string(errors));
    float ms;
    ck(cudaEventElapsedTime(&ms, b.start[r], b.stop[r]));
    worst = std::max(worst, double(ms) * 1000);
  }
  return worst;
}
Stats measure(Config const &cfg, Buffers &b, int op, size_t bytes,
              Enqueue const &enqueue, std::string const &backend,
              std::string const &selected, std::ostream *raw,
              unsigned &generation) {
  for (int i = 0; i < cfg.warmup; ++i)
    invocation(b, op, bytes, ++generation, enqueue);
  std::vector<double> roundMedians;
  for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
    std::vector<double> samples;
    for (int i = 0; i < cfg.iterations; ++i) {
      double us = invocation(b, op, bytes, ++generation, enqueue);
      samples.push_back(us);
      if (raw)
        *raw << opName(op) << ',' << bytes << ',' << backend << ',' << selected
             << ',' << repeat << ',' << i << ',' << us << ',' << generation
             << '\n';
    }
    roundMedians.push_back(median(samples));
  }
  return summarize(roundMedians);
}
struct Nccl {
  Buffers &b;
  std::vector<ncclComm_t> comm;
  explicit Nccl(Buffers &buffers) : b(buffers), comm(b.devices.size()) {
    nk(ncclCommInitAll(comm.data(), int(comm.size()), b.devices.data()));
  }
  ~Nccl() {
    for (auto c : comm)
      ncclCommDestroy(c);
  }
  void enqueue(int op, size_t bytes) {
    size_t count = bytes / 2 / (op == 0 ? 1 : comm.size());
    nk(ncclGroupStart());
    for (size_t r = 0; r < comm.size(); ++r) {
      ck(cudaSetDevice(int(r)));
      if (op == 0)
        nk(ncclAllReduce(b.in[r], b.out[r], count, ncclBfloat16, ncclSum,
                         comm[r], b.streams[r]));
      else if (op == 1)
        nk(ncclAllGather(b.in[r], b.out[r], count, ncclBfloat16, comm[r],
                         b.streams[r]));
      else
        nk(ncclReduceScatter(b.in[r], b.out[r], count, ncclBfloat16, ncclSum,
                             comm[r], b.streams[r]));
    }
    nk(ncclGroupEnd());
  }
};
#ifndef NCCL_REFERENCE_ONLY
using namespace uccl_lite;
Topology topology(Config const &cfg, Buffers const &b) {
  auto t = inspectTopology(b.devices);
  t.identity += ";joint-source=" + cfg.source;
  return t;
}
BackendPolicy policy(Config const &cfg) {
  BackendPolicy p;
  if (!cfg.profile.empty() && std::filesystem::exists(cfg.profile))
    p.measurements = loadRouterProfile(cfg.profile);
  p.fallbackIdentity = JointCollectives::peerIdentity();
  return p;
}
void streamBurst(Buffers &b, int op, size_t bytes, unsigned seed,
                 Enqueue const &enqueue) {
  int ranks = int(b.devices.size());
  size_t count = bytes / 2 / (op == 0 ? 1 : ranks);
  size_t inputs = count * (op == 2 ? ranks : 1),
         outputs = count * (op == 1 ? ranks : 1);
  for (int r = 0; r < ranks; ++r) {
    ck(cudaSetDevice(r));
    ck(cudaMemsetAsync(b.errors[r], 0, sizeof(int), b.streams[r]));
  }
  for (int iteration = 0; iteration < 12; ++iteration) {
    unsigned generation = seed + iteration;
    for (int r = 0; r < ranks; ++r) {
      ck(cudaSetDevice(r));
      initialize<<<32, 128, 0, b.streams[r]>>>(
          b.in[r], b.out[r], inputs, outputs, r, generation,
          r == iteration % ranks ? 200000 : 0);
    }
    enqueue();
    for (int r = 0; r < ranks; ++r) {
      ck(cudaSetDevice(r));
      verify<<<32, 128, 0, b.streams[r]>>>(b.in[r], b.out[r], inputs, outputs,
                                           r, ranks, op, count, generation,
                                           b.errors[r]);
    }
  }
  for (int r = 0; r < ranks; ++r) {
    ck(cudaSetDevice(r));
    ck(cudaStreamSynchronize(b.streams[r]));
    int errors;
    ck(cudaMemcpy(&errors, b.errors[r], sizeof(int), cudaMemcpyDeviceToHost));
    if (errors)
      throw std::runtime_error("async stream reuse burst failed");
  }
}
void hostOracle(Config const &cfg, Buffers &b, JointMode mode) {
  int ranks = int(b.devices.size());
  for (int op = 0; op < 3; ++op) {
    // Odd element count exercises tails, independently of the scored
    // power-of-two grid.
    size_t count = 257, bytes = count * 2 * (op == 0 ? 1 : ranks);
    size_t inputs = count * (op == 2 ? ranks : 1),
           outputs = count * (op == 1 ? ranks : 1);
    auto c = b.call(op, bytes);
    JointCollectives impl(topology(cfg, b), mode);
    impl.prepare(c);
    std::vector<std::vector<__nv_bfloat16>> host(
        ranks, std::vector<__nv_bfloat16>(inputs));
    for (int r = 0; r < ranks; ++r) {
      for (size_t i = 0; i < inputs; ++i) {
        unsigned h = mix(unsigned(i) + 19937U * (r + 1) + cfg.seed);
        float x = std::ldexp(float(int(h % 511) - 255) / 128.0f,
                             int((h >> 16) % 17) - 8);
        host[r][i] = __float2bfloat16_rn(x);
        if (op == 1) {
          static unsigned short const special[] = {
              0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0x7f81, 1, 0x8001};
          reinterpret_cast<unsigned short *>(host[r].data())[i] =
              i < 8 ? special[(i + r) % 8] : unsigned(h);
        }
      }
      ck(cudaSetDevice(r));
      ck(cudaMemcpyAsync(b.in[r], host[r].data(), inputs * 2,
                         cudaMemcpyHostToDevice, b.streams[r]));
      ck(cudaMemsetAsync(b.out[r], 0xff, outputs * 2, b.streams[r]));
    }
    impl.enqueue(c);
    for (int r = 0; r < ranks; ++r) {
      ck(cudaSetDevice(r));
      ck(cudaStreamSynchronize(b.streams[r]));
      std::vector<__nv_bfloat16> actual(outputs);
      ck(cudaMemcpy(actual.data(), b.out[r], outputs * 2,
                    cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < outputs; ++i) {
        if (op == 1) {
          if (reinterpret_cast<unsigned short const *>(actual.data())[i] !=
              reinterpret_cast<unsigned short const *>(
                  host[i / count].data())[i % count])
            throw std::runtime_error(
                "allgather BF16 bit-pattern oracle failed");
          continue;
        }
        double expected = 0, magnitude = 0;
        if (op == 1)
          expected = __bfloat162float(host[i / count][i % count]);
        else
          for (int s = 0; s < ranks; ++s) {
            double x = __bfloat162float(host[s][op == 2 ? r * count + i : i]);
            expected += x;
            magnitude += std::abs(x);
          }
        double observed = __bfloat162float(actual[i]);
        // Finite BF16 source values, double reference. Allow source-tree BF16
        // intermediate rounding (router/NCCL), but reject nonfinite output.
        double tolerance = op == 1 ? 0 : 0.02 * magnitude + 1e-7;
        if (!std::isfinite(observed) ||
            std::abs(observed - expected) > tolerance)
          throw std::runtime_error("independent host BF16 oracle failed");
      }
    }
  }
}
void selftest(Config const &cfg, Buffers &b) {
  auto t = topology(cfg, b);
  for (int op = 0; op < 3; ++op) {
    auto c = b.call(op, 1024);
    JointCollectives missing(t, JointMode::Auto);
    if (!JointCollectives::peerSupported(t))
      throw std::runtime_error("selftest requires available physical peers");
    if (missing.decision(c).cpuRouter || missing.routerInitialized())
      throw std::runtime_error("missing profile must fall back to peer without "
                               "router initialization");
    missing.prepare(c);
    for (int i = 0; i < 8; ++i)
      invocation(
          b, op, 1024, cfg.seed + i, [&] { missing.enqueue(c); }, true);
    Measurement m;
    m.topologyIdentity = missing.topology().identity;
    m.fallbackIdentity = JointCollectives::peerIdentity();
    m.workload = c.workload();
    m.router = {1, 1, 1};
    m.fallback = {10, 10, 10};
    m.repetitions = 5;
    m.correct = true;
    BackendPolicy p;
    p.measurements = {m};
    JointCollectives matching(t, JointMode::Auto, p);
    if (!matching.decision(c).cpuRouter)
      throw std::runtime_error("matching calibration must select router");
    matching.prepare(c);
    invocation(
        b, op, 1024, cfg.seed + 101, [&] { matching.enqueue(c); }, true);
    streamBurst(b, op, 1024, cfg.seed + 500, [&] { missing.enqueue(c); });
    streamBurst(b, op, 1024, cfg.seed + 700, [&] { matching.enqueue(c); });
    auto noPeer = t;
    for (size_t r = 0; r < noPeer.devices.size(); ++r)
      for (size_t q = 0; q < noPeer.devices.size(); ++q)
        if (r != q)
          noPeer.peerAccess[r * noPeer.devices.size() + q] = false;
    JointCollectives fallback(noPeer, JointMode::Auto);
    if (!fallback.decision(c).cpuRouter)
      throw std::runtime_error("unavailable peer must select valid router");
    bool unavailableRejected = false;
    try {
      JointCollectives unavailable(noPeer, JointMode::Peer);
    } catch (std::invalid_argument const &) {
      unavailableRejected = true;
    }
    if (!unavailableRejected)
      throw std::runtime_error("unavailable explicit peer was accepted");
    for (int mutation = 0; mutation < 5; ++mutation) {
      auto stale = m;
      if (mutation == 0)
        stale.topologyIdentity += "stale";
      if (mutation == 1)
        stale.fallbackIdentity += "stale";
      if (mutation == 2)
        stale.workload.bytes *= 2;
      if (mutation == 3)
        stale.workload.inPlace = true;
      if (mutation == 4)
        stale.workload.alignment = m.workload.alignment == 2 ? 16 : 2;
      p.measurements = {stale};
      JointCollectives rejected(t, JointMode::Auto, p);
      if (rejected.decision(c).cpuRouter || rejected.routerInitialized())
        throw std::runtime_error("stale/mismatched profile enabled router");
    }
    // Capture is deliberately outside the supported eager contract; reject
    // before the candidate launches any work or creates cross-stream waits.
    ck(cudaSetDevice(0));
    ck(cudaStreamBeginCapture(b.streams[0], cudaStreamCaptureModeThreadLocal));
    bool rejected = false;
    try {
      missing.enqueue(c);
    } catch (std::exception const &) {
      rejected = true;
    }
    cudaGraph_t graph = nullptr;
    auto end = cudaStreamEndCapture(b.streams[0], &graph);
    if (graph)
      ck(cudaGraphDestroy(graph));
    if (!rejected)
      throw std::runtime_error("peer graph capture not rejected");
    if (end != cudaSuccess && end != cudaErrorStreamCaptureInvalidated)
      ck(end);
    cudaGetLastError();
    invocation(
        b, op, 1024, cfg.seed + 102, [&] { missing.enqueue(c); }, true);
  }
  hostOracle(cfg, b, JointMode::Peer);
  hostOracle(cfg, b, JointMode::Router);
  std::cerr
      << "SELFTEST PASS: independent broad finite BF16 double oracle, odd "
         "tails, async input reuse, no-peer, "
         "missing/stale/profile/placement/alignment, producer streams, guards, "
         "source preservation, re-instantiation, capture rejection/recovery\n";
}
#endif
void row(int op, size_t bytes, std::string const &backend,
         std::string const &selected, Stats s) {
  std::cout << opName(op) << ',' << bytes << ',' << backend << ',' << selected
            << ',' << s.minimum << ',' << s.median << ',' << s.maximum << ",1\n"
            << std::flush;
}
} // namespace
int main(int argc, char **argv) {
  try {
    Config cfg;
#ifdef NCCL_REFERENCE_ONLY
    cfg.backend = "nccl";
#endif
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&] {
        if (++i == argc)
          throw std::invalid_argument("missing option value");
        return std::string(argv[i]);
      };
      if (a == "--gpus")
        cfg.gpus = std::stoi(next());
      else if (a == "--begin")
        cfg.begin = std::stoull(next());
      else if (a == "--end")
        cfg.end = std::stoull(next());
      else if (a == "--warmup")
        cfg.warmup = std::stoi(next());
      else if (a == "--repeats")
        cfg.repeats = std::stoi(next());
      else if (a == "--iters" || a == "--iterations")
        cfg.iterations = std::stoi(next());
      else if (a == "--backend")
        cfg.backend = next();
      else if (a == "--op")
        cfg.operation = next();
      else if (a == "--profile")
        cfg.profile = next();
      else if (a == "--source-identity")
        cfg.source = next();
      else if (a == "--samples")
        cfg.samples = next();
      else if (a == "--seed")
        cfg.seed = std::stoul(next());
      else if (a == "--reverse")
        cfg.reverse = true;
      else if (a == "--shm-only")
        cfg.shmOnly = true;
      else if (a == "--selftest")
        cfg.selftest = true;
      else
        throw std::invalid_argument("unknown option " + a);
    }
    if ((cfg.gpus != 2 && cfg.gpus != 4) || cfg.begin < 8 ||
        cfg.end < cfg.begin || cfg.end > (1UL << 30) ||
        cfg.begin % (2 * cfg.gpus) || cfg.warmup < 0 || cfg.repeats < 1 ||
        cfg.iterations < 1)
      throw std::invalid_argument("invalid shape/repetitions");
    if (cfg.backend != "nccl" && cfg.backend != "router" &&
        cfg.backend != "peer" && cfg.backend != "joint" &&
        cfg.backend != "calibrate")
      throw std::invalid_argument("unknown backend");
#ifdef NCCL_REFERENCE_ONLY
    if (cfg.backend != "nccl" || cfg.selftest)
      throw std::invalid_argument("frozen NCCL reference only");
#endif
    if (cfg.backend == "calibrate" &&
        (cfg.profile.empty() || std::filesystem::exists(cfg.profile)))
      throw std::invalid_argument("calibration needs a new profile path");
    if (cfg.backend != "nccl" && cfg.source.empty())
      throw std::invalid_argument("candidate requires --source-identity");
    if (cfg.shmOnly) {
      if (cfg.backend != "nccl")
        throw std::invalid_argument("SHM applies only to reference");
      setenv("NCCL_P2P_DISABLE", "1", 1);
      setenv("NCCL_SHM_DISABLE", "0", 1);
      setenv("NCCL_IB_DISABLE", "1", 1);
      setenv("NCCL_NET", "Socket", 1);
    }
    std::ofstream raw;
    if (!cfg.samples.empty()) {
      if (std::filesystem::exists(cfg.samples))
        throw std::invalid_argument("sample output exists");
      raw.exceptions(std::ios::badbit | std::ios::failbit);
      raw.open(cfg.samples);
      raw << std::setprecision(10);
      raw << "operation,bytes,backend,selected_backend,repeat,iteration,"
             "slowest_rank_us,generation\n";
    }
    std::cout << std::setprecision(10)
              << "operation,bytes,backend,selected_backend,minimum_us,median_"
                 "us,maximum_us,correct\n";
    std::cerr << "CONFIG backend=" << cfg.backend << " ranks=" << cfg.gpus
              << " dtype=bf16 placement=outofplace warmup=" << cfg.warmup
              << " repeats=" << cfg.repeats << " iterations=" << cfg.iterations
              << " seed=" << cfg.seed << " source=" << cfg.source
              << " profile=" << cfg.profile << '\n';
    Buffers b(cfg.gpus, std::max(cfg.end, size_t(4096)));
#ifndef NCCL_REFERENCE_ONLY
    if (cfg.selftest) {
      selftest(cfg, b);
      return 0;
    }
#endif
    std::vector<int> ops;
    for (int op = 0; op < 3; ++op)
      if (cfg.operation == "all" || cfg.operation == opName(op))
        ops.push_back(op);
    if (ops.empty())
      throw std::invalid_argument("unknown operation");
    std::vector<size_t> sizes;
    for (size_t n = cfg.begin; n <= cfg.end; n *= 2)
      sizes.push_back(n);
    if (cfg.reverse) {
      std::reverse(sizes.begin(), sizes.end());
      std::reverse(ops.begin(), ops.end());
    }
    unsigned generation = cfg.seed;
    std::unique_ptr<Nccl> nccl;
    if (cfg.backend == "nccl") {
      int version;
      nk(ncclGetVersion(&version));
      std::cerr << "NCCL_IDENTITY "
                << uccl_lite::benchmark::ncclFallbackIdentity(
                       version,
                       uccl_lite::benchmark::ncclConfiguration(environ))
                << '\n';
      nccl = std::make_unique<Nccl>(b);
    }
#ifndef NCCL_REFERENCE_ONLY
    std::vector<Measurement> measurements;
#endif
    for (int op : ops)
      for (size_t bytes : sizes) {
        if (nccl) {
          auto s = measure(
              cfg, b, op, bytes, [&] { nccl->enqueue(op, bytes); }, "nccl",
              cfg.shmOnly ? "nccl_shm" : "nccl_default", raw ? &raw : nullptr,
              generation);
          row(op, bytes, "nccl", cfg.shmOnly ? "nccl_shm" : "nccl_default", s);
          continue;
        }
#ifndef NCCL_REFERENCE_ONLY
        auto c = b.call(op, bytes);
        auto t = topology(cfg, b);
        if (cfg.backend == "calibrate") {
          Measurement m;
          m.workload = c.workload();
          m.fallbackIdentity = JointCollectives::peerIdentity();
          m.repetitions = cfg.repeats;
          m.correct = true;
          // Destroy each backend before constructing the other: peer timings
          // never include live CPU-router workers from the comparison process.
          for (int step = 0; step < 2; ++step) {
            bool router =
                ((step + int(bytes / cfg.begin % 3) + op + cfg.reverse) % 2) ==
                0;
            JointCollectives impl(t,
                                  router ? JointMode::Router : JointMode::Peer);
            impl.prepare(c);
            m.topologyIdentity = impl.topology().identity;
            auto s = measure(
                cfg, b, op, bytes, [&] { impl.enqueue(c); }, "calibrate",
                router ? "router" : "peer", raw ? &raw : nullptr, generation);
            if (router)
              m.router = {s.minimum, s.median, s.maximum};
            else
              m.fallback = {s.minimum, s.median, s.maximum};
            row(op, bytes, "calibrate", router ? "router" : "peer", s);
          }
          measurements.push_back(m);
        } else {
          auto mode = cfg.backend == "router" ? JointMode::Router
                      : cfg.backend == "peer" ? JointMode::Peer
                                              : JointMode::Auto;
          JointCollectives impl(t, mode, policy(cfg));
          auto selected = impl.prepare(c);
          std::string choice = selected.cpuRouter ? "router" : "peer";
          std::cerr << "DECISION " << opName(op) << ' ' << bytes << ' '
                    << choice << ' ' << selected.reason << '\n';
          auto s = measure(
              cfg, b, op, bytes, [&] { impl.enqueue(c); }, cfg.backend, choice,
              raw ? &raw : nullptr, generation);
          row(op, bytes, cfg.backend, choice, s);
        }
#endif
      }
#ifndef NCCL_REFERENCE_ONLY
    if (cfg.backend == "calibrate")
      saveRouterProfile(cfg.profile, measurements);
#endif
    std::cout
        << "# correctness: PASS (every invocation, full output/input/guards)\n";
    return 0;
  } catch (std::exception const &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}

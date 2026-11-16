// Eight-rank NCCL control for UCCL Lite's two-node AR/AG/RS comparison.
// Host submission plus completion is timed; every invocation is fully checked.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

namespace {
constexpr int kRanks = 8;
void cc(cudaError_t x) {
  if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x));
}
void nc(ncclResult_t x) {
  if (x != ncclSuccess) throw std::runtime_error(ncclGetErrorString(x));
}
void mp(int x) {
  if (x != MPI_SUCCESS) throw std::runtime_error("MPI call failed");
}
enum class Op { AR, AG, RS };
__global__ void fill(float* p, size_t count, int rank, int gen) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x)
    p[i] = float((rank + 1 + gen * kRanks) * (1 + i % 97));
}
__global__ void check(float const* p, size_t count, size_t shard, int rank,
                      int gen, Op op, unsigned int* wrong) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    float want;
    if (op == Op::AG) {
      int source = int(i / shard);
      want = float((source + 1 + gen * kRanks) * (1 + (i % shard) % 97));
    } else {
      size_t global = op == Op::RS ? size_t(rank) * shard + i : i;
      want = float((36 + gen * kRanks * kRanks) * (1 + global % 97));
    }
    if (p[i] != want) atomicAdd(wrong, 1);
  }
}
struct Cfg {
  Op op = Op::AR;
  size_t begin = 1024, end = 1ull << 30;
  int warm = 5, iters = 10;
  int gpusPerProcess = 1;
};
Cfg parse(int argc, char** argv) {
  Cfg c;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (++i == argc) throw std::runtime_error("missing option value");
    std::string value = argv[i];
    if (key == "--op") {
      if (value == "allreduce")
        c.op = Op::AR;
      else if (value == "allgather")
        c.op = Op::AG;
      else if (value == "reducescatter")
        c.op = Op::RS;
      else
        throw std::runtime_error("unknown collective");
    } else if (key == "-b")
      c.begin = std::strtoull(value.c_str(), nullptr, 0);
    else if (key == "-e")
      c.end = std::strtoull(value.c_str(), nullptr, 0);
    else if (key == "--gpus-per-process")
      c.gpusPerProcess = std::stoi(value);
    else if (key == "-w")
      c.warm = std::atoi(value.c_str());
    else if (key == "-n")
      c.iters = std::atoi(value.c_str());
    else
      throw std::runtime_error("unknown option");
  }
  if (c.begin < 1024 || c.begin > c.end || c.begin % 512 || c.end % 512 ||
      c.warm < 5 || c.iters < 10)
    throw std::runtime_error("invalid protocol");
  if (c.gpusPerProcess != 1 &&
      (c.gpusPerProcess != 2 || c.op != Op::AG))
    throw std::runtime_error("two GPUs per process requires AllGather");
  return c;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Cfg cfg = parse(argc, argv);
    mp(MPI_Init(&argc, &argv));
    int rank = -1, world = 0;
    mp(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mp(MPI_Comm_size(MPI_COMM_WORLD, &world));
    if (world * cfg.gpusPerProcess != kRanks)
      throw std::runtime_error("requires eight GPU ranks");
    MPI_Comm local;
    mp(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                           MPI_INFO_NULL, &local));
    int localRank = -1, localSize = 0;
    mp(MPI_Comm_rank(local, &localRank));
    mp(MPI_Comm_size(local, &localSize));
    if (localSize * cfg.gpusPerProcess != 4)
      throw std::runtime_error("requires four GPUs per node");
    struct Device {
      int physical, rank;
      cudaStream_t stream;
      ncclComm_t comm;
      float *input, *output;
      unsigned int* wrong;
    };
    std::vector<Device> devices(cfg.gpusPerProcess);
    for (int gpu = 0; gpu < cfg.gpusPerProcess; ++gpu) {
      auto& device = devices[gpu];
      device.physical = localRank * cfg.gpusPerProcess + gpu;
      device.rank = rank * cfg.gpusPerProcess + gpu;
      cc(cudaSetDevice(device.physical));
      cc(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
    }
    ncclUniqueId id;
    if (rank == 0) nc(ncclGetUniqueId(&id));
    mp(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    if (cfg.gpusPerProcess > 1) nc(ncclGroupStart());
    for (auto& device : devices) {
      cc(cudaSetDevice(device.physical));
      nc(ncclCommInitRank(&device.comm, kRanks, id, device.rank));
    }
    if (cfg.gpusPerProcess > 1) nc(ncclGroupEnd());
    int version = 0;
    nc(ncclGetVersion(&version));
    if (rank == 0) {
      std::printf(
          "# two nodes x four L4; NCCL %d; float32; %s; %s; host "
          "enqueue+completion\n",
          version,
          cfg.op == Op::AR   ? "AllReduce"
          : cfg.op == Op::AG ? "AllGather"
                             : "ReduceScatter",
          cfg.op == Op::AR ? "in-place" : "out-of-place");
      std::printf(
          "# warmups=%d iterations=%d; every invocation fully checked\n",
          cfg.warm, cfg.iters);
      std::printf("# MPI ranks=%d GPUs/process=%d; bytes=complete tensor/GPU\n",
                  world, cfg.gpusPerProcess);
      std::puts("backend,phase,iteration,bytes,time_us,wrong");
    }
    for (size_t bytes = cfg.begin; bytes <= cfg.end;) {
      size_t full = bytes / sizeof(float), shard = full / kRanks;
      size_t inputCount = cfg.op == Op::AG ? shard : full;
      size_t outputCount = cfg.op == Op::RS ? shard : full;
      for (auto& device : devices) {
        cc(cudaSetDevice(device.physical));
        cc(cudaMalloc(&device.input, inputCount * sizeof(float)));
        if (cfg.op == Op::AR)
          device.output = device.input;
        else
          cc(cudaMalloc(&device.output, outputCount * sizeof(float)));
        cc(cudaMalloc(&device.wrong, sizeof(unsigned int)));
      }
      double timedSum = 0;
      for (int i = 0; i < cfg.warm + cfg.iters; ++i) {
        int gen = 10 + i;
        for (auto& device : devices) {
          cc(cudaSetDevice(device.physical));
          fill<<<256, 256, 0, device.stream>>>(device.input, inputCount,
                                              device.rank, gen);
          cc(cudaGetLastError());
          cc(cudaStreamSynchronize(device.stream));
        }
        mp(MPI_Barrier(MPI_COMM_WORLD));
        auto start = std::chrono::steady_clock::now();
        if (cfg.gpusPerProcess > 1) nc(ncclGroupStart());
        for (auto& device : devices) {
          if (cfg.gpusPerProcess > 1) cc(cudaSetDevice(device.physical));
          if (cfg.op == Op::AR)
            nc(ncclAllReduce(device.input, device.output, full, ncclFloat,
                             ncclSum, device.comm, device.stream));
          else if (cfg.op == Op::AG)
            nc(ncclAllGather(device.input, device.output, shard, ncclFloat,
                             device.comm, device.stream));
          else
            nc(ncclReduceScatter(device.input, device.output, shard, ncclFloat,
                                 ncclSum, device.comm, device.stream));
        }
        if (cfg.gpusPerProcess > 1) nc(ncclGroupEnd());
        for (auto& device : devices) {
          if (cfg.gpusPerProcess > 1) cc(cudaSetDevice(device.physical));
          cc(cudaStreamSynchronize(device.stream));
        }
        auto end = std::chrono::steady_clock::now();
        double localUs =
            std::chrono::duration<double, std::micro>(end - start).count();
        double slowestUs = 0;
        mp(MPI_Allreduce(&localUs, &slowestUs, 1, MPI_DOUBLE, MPI_MAX,
                         MPI_COMM_WORLD));
        unsigned int localWrong = 0, totalWrong = 0;
        for (auto& device : devices) {
          cc(cudaSetDevice(device.physical));
          cc(cudaMemsetAsync(device.wrong, 0, sizeof(unsigned int), device.stream));
          check<<<256, 256, 0, device.stream>>>(device.output, outputCount, shard,
                                               device.rank, gen, cfg.op,
                                               device.wrong);
          cc(cudaGetLastError());
          unsigned int wrong = 0;
          cc(cudaMemcpyAsync(&wrong, device.wrong, sizeof(unsigned int),
                             cudaMemcpyDeviceToHost, device.stream));
          cc(cudaStreamSynchronize(device.stream));
          localWrong += wrong;
        }
        mp(MPI_Allreduce(&localWrong, &totalWrong, 1, MPI_UNSIGNED, MPI_SUM,
                         MPI_COMM_WORLD));
        bool timed = i >= cfg.warm;
        if (rank == 0) {
          std::printf("nccl,%s,%d,%zu,%.3f,%u\n", timed ? "timed" : "warmup",
                      timed ? i - cfg.warm + 1 : i + 1, bytes, slowestUs,
                      totalWrong);
          if (timed) timedSum += slowestUs;
        }
        if (totalWrong) throw std::runtime_error("output mismatch");
      }
      if (rank == 0) {
        std::printf("# mean,%s,%zu,%.3f\n",
                    cfg.op == Op::AR   ? "allreduce"
                    : cfg.op == Op::AG ? "allgather"
                                       : "reducescatter",
                    bytes, timedSum / cfg.iters);
        std::fflush(stdout);
      }
      for (auto& device : devices) {
        cc(cudaSetDevice(device.physical));
        cc(cudaFree(device.wrong));
        if (cfg.op != Op::AR) cc(cudaFree(device.output));
        cc(cudaFree(device.input));
      }
      if (bytes > cfg.end / 2) break;
      bytes *= 2;
    }
    for (auto& device : devices) {
      cc(cudaSetDevice(device.physical));
      nc(ncclCommDestroy(device.comm));
      cc(cudaStreamDestroy(device.stream));
    }
    mp(MPI_Comm_free(&local));
    mp(MPI_Finalize());
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "NCCL two-node baseline: %s\n", error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

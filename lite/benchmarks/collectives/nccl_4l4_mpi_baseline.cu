// Four-rank NCCL AllReduce control with the same process count and host timing
// scope as the native UCCL Lite benchmark. Every invocation is fully checked.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

namespace {
constexpr int kRanks = 4;
void cc(cudaError_t x) {
  if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x));
}
void nc(ncclResult_t x) {
  if (x != ncclSuccess) throw std::runtime_error(ncclGetErrorString(x));
}
void mp(int x) {
  if (x != MPI_SUCCESS) throw std::runtime_error("MPI call failed");
}
__global__ void fill(float* data, size_t count, int rank, int generation) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x)
    data[i] = float(rank * 1000000 + generation * 1024 + int(i & 1023));
}
__global__ void check(float const* data, size_t count, int generation,
                      unsigned int* wrong) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    float want = float(6000000 + 4 * generation * 1024 + 4 * int(i & 1023));
    if (data[i] != want) atomicAdd(wrong, 1);
  }
}
struct Config {
  size_t begin = 1024, end = 1ull << 30;
  int warmups = 5, iterations = 10;
};
Config parse(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    if (++i == argc) throw std::runtime_error("missing option value");
    std::string value = argv[i];
    if (option == "--op") {
      if (value != "allreduce") throw std::runtime_error("requires AllReduce");
    } else if (option == "-b")
      cfg.begin = std::strtoull(value.c_str(), nullptr, 0);
    else if (option == "-e")
      cfg.end = std::strtoull(value.c_str(), nullptr, 0);
    else if (option == "-w")
      cfg.warmups = std::atoi(value.c_str());
    else if (option == "-n")
      cfg.iterations = std::atoi(value.c_str());
    else
      throw std::runtime_error("unknown option");
  }
  if (cfg.begin < 1024 || cfg.begin > cfg.end || cfg.begin % 16 ||
      cfg.end % 16 || cfg.warmups < 5 || cfg.iterations < 10)
    throw std::runtime_error("invalid protocol");
  return cfg;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg = parse(argc, argv);
    mp(MPI_Init(&argc, &argv));
    int rank = -1, world = 0;
    mp(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mp(MPI_Comm_size(MPI_COMM_WORLD, &world));
    if (world != kRanks) throw std::runtime_error("requires four ranks");
    MPI_Comm local;
    mp(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                           MPI_INFO_NULL, &local));
    int localSize = 0;
    mp(MPI_Comm_size(local, &localSize));
    if (localSize != kRanks) throw std::runtime_error("requires one node");
    cc(cudaSetDevice(rank));
    cudaStream_t stream;
    cc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ncclUniqueId id;
    if (rank == 0) nc(ncclGetUniqueId(&id));
    mp(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclComm_t comm;
    nc(ncclCommInitRank(&comm, kRanks, id, rank));
    int version = 0;
    nc(ncclGetVersion(&version));
    if (rank == 0) {
      std::printf(
          "# four L4 MPI ranks; NCCL %d; AllReduce; float32; in-place; host "
          "enqueue+completion\n",
          version);
      std::printf(
          "# warmups=%d iterations=%d; every invocation fully checked\n",
          cfg.warmups, cfg.iterations);
      std::puts("backend,phase,iteration,bytes,time_us,wrong");
    }
    int sizeIndex = 0;
    for (size_t bytes = cfg.begin; bytes <= cfg.end;) {
      size_t count = bytes / sizeof(float);
      float* data = nullptr;
      unsigned int* deviceWrong = nullptr;
      cc(cudaMalloc(&data, bytes));
      cc(cudaMalloc(&deviceWrong, sizeof(unsigned int)));
      double timedSum = 0;
      for (int iteration = 0; iteration < cfg.warmups + cfg.iterations;
           ++iteration) {
        int generation = 10 + sizeIndex * 100 + iteration;
        fill<<<256, 256, 0, stream>>>(data, count, rank, generation);
        cc(cudaGetLastError());
        cc(cudaStreamSynchronize(stream));
        mp(MPI_Barrier(MPI_COMM_WORLD));
        auto start = std::chrono::steady_clock::now();
        nc(ncclAllReduce(data, data, count, ncclFloat, ncclSum, comm, stream));
        cc(cudaStreamSynchronize(stream));
        auto end = std::chrono::steady_clock::now();
        double localUs =
            std::chrono::duration<double, std::micro>(end - start).count();
        double slowestUs = 0;
        mp(MPI_Allreduce(&localUs, &slowestUs, 1, MPI_DOUBLE, MPI_MAX,
                         MPI_COMM_WORLD));
        cc(cudaMemsetAsync(deviceWrong, 0, sizeof(unsigned int), stream));
        check<<<256, 256, 0, stream>>>(data, count, generation, deviceWrong);
        cc(cudaGetLastError());
        unsigned int localWrong = 0, totalWrong = 0;
        cc(cudaMemcpyAsync(&localWrong, deviceWrong, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost, stream));
        cc(cudaStreamSynchronize(stream));
        mp(MPI_Allreduce(&localWrong, &totalWrong, 1, MPI_UNSIGNED, MPI_SUM,
                         MPI_COMM_WORLD));
        bool timed = iteration >= cfg.warmups;
        if (rank == 0) {
          std::printf("nccl,%s,%d,%zu,%.3f,%u\n", timed ? "timed" : "warmup",
                      timed ? iteration - cfg.warmups + 1 : iteration + 1,
                      bytes, slowestUs, totalWrong);
          if (timed) timedSum += slowestUs;
        }
        if (totalWrong) throw std::runtime_error("output mismatch");
      }
      if (rank == 0) {
        std::printf("# mean,allreduce,%zu,%.3f\n", bytes,
                    timedSum / cfg.iterations);
        std::fflush(stdout);
      }
      cc(cudaFree(deviceWrong));
      cc(cudaFree(data));
      if (bytes > cfg.end / 2) break;
      bytes *= 2;
      ++sizeIndex;
    }
    nc(ncclCommDestroy(comm));
    cc(cudaStreamDestroy(stream));
    mp(MPI_Comm_free(&local));
    mp(MPI_Finalize());
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "NCCL four-L4 MPI baseline: %s\n", error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

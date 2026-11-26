// Strict host-only fallback for systems without GPU P2P or GPUDirect RDMA.
// Every collective byte follows GPU -> pinned host -> MPI/UCX -> pinned host ->
// GPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>

#define CUDACHECK(cmd)                                           \
  do {                                                           \
    cudaError_t e = (cmd);                                       \
    if (e != cudaSuccess) {                                      \
      std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); \
      MPI_Abort(MPI_COMM_WORLD, 1);                              \
    }                                                            \
  } while (0)
#define MPICHECK(cmd)                            \
  do {                                           \
    int e = (cmd);                               \
    if (e != MPI_SUCCESS) {                      \
      std::fprintf(stderr, "MPI error %d\n", e); \
      MPI_Abort(MPI_COMM_WORLD, e);              \
    }                                            \
  } while (0)

struct Config {
  size_t minBytes = 1UL << 10;
  size_t maxBytes = 1UL << 30;
  size_t pieceBytes = 4UL << 20;
  int warmup = 5;
  int iters = 10;
  bool profile = false;
  bool check = true;
};

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  int rank = 0, world = 0, localRank = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                               MPI_INFO_NULL, &localComm));
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  CUDACHECK(cudaSetDevice(localRank));

  Config cfg;
  try {
    for (int i = 1; i < argc; i++) {
      std::string key = argv[i];
      auto next = [&]() {
        if (++i >= argc) throw std::runtime_error("missing " + key);
        return argv[i];
      };
      if (key == "-b")
        cfg.minBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-e")
        cfg.maxBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-k")
        cfg.pieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-w")
        cfg.warmup = std::atoi(next());
      else if (key == "-n")
        cfg.iters = std::atoi(next());
      else if (key == "-P")
        cfg.profile = std::atoi(next()) != 0;
      else if (key == "-c")
        cfg.check = std::atoi(next()) != 0;
      else
        throw std::runtime_error("unknown option " + key);
    }
  } catch (std::exception const& error) {
    if (rank == 0) std::fprintf(stderr, "%s\n", error.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  char* device = nullptr;
  CUDACHECK(cudaMalloc(&device, cfg.maxBytes));
  size_t slotBytes = std::min(cfg.maxBytes, cfg.pieceBytes);
  float* host[2]{};
  cudaStream_t streams[2]{};
  for (int slot = 0; slot < 2; slot++) {
    CUDACHECK(cudaHostAlloc(&host[slot], slotBytes, cudaHostAllocPortable));
    CUDACHECK(cudaStreamCreateWithFlags(&streams[slot], cudaStreamNonBlocking));
  }
  std::vector<float> input, output;

  auto fill = [&](size_t count) {
    input.resize(count);
    for (size_t i = 0; i < count; i++)
      input[i] = (float)(rank + 1) * (float)(1 + i % 97);
    CUDACHECK(cudaMemcpy(device, input.data(), count * sizeof(float),
                         cudaMemcpyHostToDevice));
  };

  double d2hUs = 0.0, mpiUs = 0.0, h2dUs = 0.0;
  auto launch = [&](size_t bytes) {
    size_t piece =
        std::max<size_t>(sizeof(float), cfg.pieceBytes & ~(sizeof(float) - 1));
    int pieces = (int)((bytes + piece - 1) / piece);
    for (int p = 0; p < pieces; p++) {
      int slot = p & 1;
      CUDACHECK(cudaStreamSynchronize(streams[slot]));
      size_t offset = (size_t)p * piece;
      size_t length = std::min(piece, bytes - offset);
      auto stageBegin = std::chrono::steady_clock::now();
      CUDACHECK(cudaMemcpy(host[slot], device + offset, length,
                           cudaMemcpyDeviceToHost));
      auto stageEnd = std::chrono::steady_clock::now();
      if (cfg.profile)
        d2hUs +=
            std::chrono::duration<double, std::micro>(stageEnd - stageBegin)
                .count();
      stageBegin = std::chrono::steady_clock::now();
      MPICHECK(MPI_Allreduce(MPI_IN_PLACE, host[slot],
                             (int)(length / sizeof(float)), MPI_FLOAT, MPI_SUM,
                             MPI_COMM_WORLD));
      stageEnd = std::chrono::steady_clock::now();
      if (cfg.profile)
        mpiUs +=
            std::chrono::duration<double, std::micro>(stageEnd - stageBegin)
                .count();
      stageBegin = std::chrono::steady_clock::now();
      CUDACHECK(cudaMemcpy(device + offset, host[slot], length,
                           cudaMemcpyHostToDevice));
      stageEnd = std::chrono::steady_clock::now();
      if (cfg.profile)
        h2dUs +=
            std::chrono::duration<double, std::micro>(stageEnd - stageBegin)
                .count();
    }
    CUDACHECK(cudaStreamSynchronize(streams[0]));
    CUDACHECK(cudaStreamSynchronize(streams[1]));
  };

  auto verify = [&](size_t count) {
    output.resize(count);
    CUDACHECK(cudaMemcpy(output.data(), device, count * sizeof(float),
                         cudaMemcpyDeviceToHost));
    double rankSum = (double)world * (world + 1) / 2.0;
    double worst = 0.0;
    size_t firstBad = count;
    for (size_t i = 0; i < count; i++) {
      double expected = rankSum * (double)(1 + i % 97);
      double error = std::fabs((double)output[i] - expected) / expected;
      if (error > 1e-5 && firstBad == count) firstBad = i;
      worst = std::max(worst, error);
    }
    if (firstBad != count)
      std::fprintf(stderr, "rank %d first bad %zu got %.1f expected %.1f\n",
                   rank, firstBad, (double)output[firstBad],
                   rankSum * (double)(1 + firstBad % 97));
    return worst;
  };

  if (rank == 0) {
    std::printf(
        "# UCCL Lite strict host-only AllReduce; no GPU P2P; no GPUDirect "
        "RDMA\n");
    std::printf("# D2H -> blocking MPI_Allreduce(host) -> H2D; piece %zu B\n",
                cfg.pieceBytes);
    std::printf("# %12s %12s %10s %10s %10s %10s\n", "size(B)", "count",
                "time(us)", "algbw", "busbw", "maxrelerr");
  }
  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    bytes = (bytes + 511) & ~(size_t)511;
    size_t count = bytes / sizeof(float);
    double error = -1.0;
    if (cfg.check) {
      fill(count);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      launch(bytes);
      error = verify(count);
    }
    fill(count);
    for (int i = 0; i < cfg.warmup; i++) launch(bytes);
    d2hUs = mpiUs = h2dUs = 0.0;
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; i++) launch(bytes);
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - begin).count() /
                cfg.iters;
    double worstUs = 0.0, worstError = 0.0;
    MPICHECK(
        MPI_Reduce(&us, &worstUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&error, &worstError, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    double profileLocal[3]{d2hUs / cfg.iters, mpiUs / cfg.iters,
                           h2dUs / cfg.iters};
    double profileWorst[3]{};
    MPICHECK(MPI_Reduce(profileLocal, profileWorst, 3, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    if (rank == 0) {
      double algbw = (double)bytes / worstUs / 1000.0;
      std::printf("  %12zu %12zu %10.2f %10.2f %10.2f %10.1e\n", bytes, count,
                  worstUs, algbw, algbw * 1.75, worstError);
      if (cfg.profile)
        std::printf("# profile size=%zu d2h_us=%.2f mpi_us=%.2f h2d_us=%.2f\n",
                    bytes, profileWorst[0], profileWorst[1], profileWorst[2]);
    }
    if (bytes > cfg.maxBytes / 2) break;
  }

  for (int slot = 0; slot < 2; slot++) {
    CUDACHECK(cudaStreamDestroy(streams[slot]));
    CUDACHECK(cudaFreeHost(host[slot]));
  }
  CUDACHECK(cudaFree(device));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}

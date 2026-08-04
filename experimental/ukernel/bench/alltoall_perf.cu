// alltoall_perf.cu — minimal ncclAllToAll bandwidth benchmark.
//
// Same binary runs against either the ukernel shim (LD_LIBRARY_PATH ->
// build/nccl/lib) or native NCCL (system libnccl), so the comparison is
// apples-to-apples on the standard API. nccl-tests' alltoall_perf uses
// ncclSend/ncclRecv, which the shim does not implement; this bench uses
// ncclAllToAll directly (shim implements it, native has it).
//
// Usage (2 ranks, 1 GPU each, same node):
//   mpirun -np 1 ./alltoall_perf --rank=0 --bytes=268435456 \
//        : -np 1 ./alltoall_perf --rank=1 --bytes=268435456
// rank 0 writes the NCCL unique id to /tmp/uk_a2a_id; rank 1 polls for it.
// Report: algbw = (nranks-1)/nranks * total_bytes / avg_time (nccl-tests
// convention), busbw = algbw * nranks/(nranks-1). total_bytes is the full
// per-rank buffer (nranks * count * elemsz).

#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

static const char* kIdPath = "/tmp/uk_a2a_id";

static long get_long_arg(int argc, char** argv, const char* name, long def) {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == name) return std::atol(argv[i + 1]);
  }
  return def;
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now()
                                           .time_since_epoch())
      .count();
}

#define CUDACHK(c)                                                   \
  do {                                                               \
    cudaError_t e = (c);                                             \
    if (e != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
              __FILE__, __LINE__);                                   \
      exit(1);                                                       \
    }                                                                \
  } while (0)

#define NCCLCHK(c)                                                     \
  do {                                                                 \
    ncclResult_t r = (c);                                              \
    if (r != ncclSuccess) {                                            \
      fprintf(stderr, "NCCL error %d at %s:%d\n", r, __FILE__, __LINE__); \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

int main(int argc, char** argv) {
  int rank = (int)get_long_arg(argc, argv, "--rank", 0);
  size_t total_bytes = (size_t)get_long_arg(argc, argv, "--bytes", 1 << 28);
  int iters = (int)get_long_arg(argc, argv, "--iters", 20);
  int warmup = (int)get_long_arg(argc, argv, "--warmup", 5);
  const int nranks = 2;

  int dev = rank;
  CUDACHK(cudaSetDevice(dev));

  ncclUniqueId id;
  if (rank == 0) {
    NCCLCHK(ncclGetUniqueId(&id));
    FILE* f = fopen(kIdPath, "wb");
    if (!f) { perror("fopen"); exit(1); }
    fwrite(&id, sizeof(id), 1, f);
    fclose(f);
  } else {
    // Poll for rank 0's id file (mpirun starts ranks together).
    FILE* f = nullptr;
    for (int i = 0; i < 500 && !f; ++i) {
      f = fopen(kIdPath, "rb");
      if (!f) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!f) { fprintf(stderr, "rank1: no unique id file\n"); exit(1); }
    if (fread(&id, sizeof(id), 1, f) != 1) { fprintf(stderr, "id read fail\n"); exit(1); }
    fclose(f);
    remove(kIdPath);
  }

  ncclComm_t comm;
  NCCLCHK(ncclCommInitRank(&comm, nranks, id, rank));

  void* buf = nullptr;
  CUDACHK(cudaMalloc(&buf, total_bytes));
  CUDACHK(cudaMemset(buf, 0, total_bytes));

  // count = elements per rank pair; total = nranks * count * 4 (float).
  size_t count = total_bytes / (sizeof(float) * nranks);

  cudaStream_t stream;
  CUDACHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  for (int i = 0; i < warmup; ++i)
    NCCLCHK(ncclAllToAll(buf, buf, count, ncclFloat, comm, stream));
  CUDACHK(cudaStreamSynchronize(stream));

  std::vector<double> times;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_s();
    NCCLCHK(ncclAllToAll(buf, buf, count, ncclFloat, comm, stream));
    CUDACHK(cudaStreamSynchronize(stream));
    double dt = now_s() - t0;
    if (i >= 2) times.push_back(dt);
  }

  double avg = 0;
  for (double t : times) avg += t;
  avg /= times.size();
  double algbw = (double)(nranks - 1) / nranks * total_bytes / avg / 1e9;
  double busbw = algbw * nranks / (nranks - 1);
  fprintf(stderr,
          "[r%d] total=%zuMB iters=%zu avg=%.1fus algbw=%.1fGB/s busbw=%.1fGB/s\n",
          rank, total_bytes >> 20, times.size(), avg * 1e6, algbw, busbw);

  NCCLCHK(ncclCommDestroy(comm));
  CUDACHK(cudaFree(buf));
  return 0;
}

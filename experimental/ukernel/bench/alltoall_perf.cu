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

// Build with -DUSE_SHIM_API for the ukernel shim (its nccl.h adds a
// ncclAllToAll extension; native NCCL has no such API). Without the
// define the harness uses ncclSend/ncclRecv to build the same 2-rank
// alltoall exchange, which is what nccl-tests' alltoall_perf does on
// native NCCL. Same buffers, same count semantics.
//
// 2-rank exchange (in-place, buf = 2*count floats):
//   rank r sends buf[r*count .. (r+1)*count) to peer, receives the
//   peer's slice into buf[(1-r)*count .. (2-r)*count).

static long get_long_arg(int argc, char** argv, const char* name, long def) {
  std::string n(name);
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == n && i + 1 < argc) return std::atol(argv[i + 1]);
    if (a.size() > n.size() && a.compare(0, n.size(), n) == 0 &&
        a[n.size()] == '=')
      return std::atol(a.c_str() + n.size() + 1);
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

static void run_one(void* buf, size_t count, int rank, ncclComm_t comm,
                    cudaStream_t stream) {
#ifdef USE_SHIM_API
  NCCLCHK(ncclAllToAll(buf, buf, count, ncclFloat, comm, stream));
#else
  // Native NCCL: no ncclAllToAll. 2-rank ring exchange (in-place):
  // send my slice to the peer, receive the peer's slice.
  int peer = 1 - rank;
  // Standard alltoall: partition i of my buffer goes to rank i. With 2
  // ranks I send partition peer and receive the peer's partition peer
  // (= my own partition's slot), i.e. both offsets are peer * count.
  size_t off = static_cast<size_t>(peer) * count;
  // nccl-tests pattern: group the send/recv pair (recv first, the
  // conventional order) so the exchange is one collective operation.
  NCCLCHK(ncclGroupStart());
  NCCLCHK(ncclRecv(static_cast<char*>(buf) + off * sizeof(float), count,
                   ncclFloat, peer, comm, stream));
  NCCLCHK(ncclSend(static_cast<char*>(buf) + off * sizeof(float), count,
                   ncclFloat, peer, comm, stream));
  NCCLCHK(ncclGroupEnd());
#endif
}

// Fill buf with rank-specific values, run one exchange, and check that
// partition peer now holds the peer's original values.
static bool verify_exchange(void* buf, size_t count, int rank,
                            ncclComm_t comm, cudaStream_t stream) {
  std::vector<float> fill(2 * count, static_cast<float>(rank + 1));
  CUDACHK(cudaMemcpy(buf, fill.data(), fill.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  run_one(buf, count, rank, comm, stream);
  CUDACHK(cudaStreamSynchronize(stream));
  int peer = 1 - rank;
  std::vector<float> got(count);
  CUDACHK(cudaMemcpy(got.data(),
                     static_cast<char*>(buf) +
                         static_cast<size_t>(peer) * count * sizeof(float),
                     count * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> mine(count);
  CUDACHK(cudaMemcpy(mine.data(),
                     static_cast<char*>(buf) +
                         static_cast<size_t>(rank) * count * sizeof(float),
                     count * sizeof(float), cudaMemcpyDeviceToHost));
  fprintf(stderr, "[r%d] my[0..3]=%.0f,%.0f,%.0f,%.0f peer[0..3]=%.0f,%.0f,%.0f,%.0f\n",
          rank, mine[0], mine[1], mine[2], mine[3], got[0], got[1], got[2],
          got[3]);
  bool ok = true;
  for (size_t i = 0; i < count; ++i) {
    if (got[i] != static_cast<float>(peer + 1)) {
      ok = false;
      if (i < 8)
        fprintf(stderr, "[r%d] verify bad[%zu]=%.0f want=%.0f\n", rank, i,
                got[i], static_cast<float>(peer + 1));
      break;
    }
  }
  return ok;
}

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

  bool vok = verify_exchange(buf, count, rank, comm, stream);
  fprintf(stderr, "[r%d] verify %s\n", rank, vok ? "OK" : "FAIL");

  for (int i = 0; i < warmup; ++i)
    run_one(buf, count, rank, comm, stream);
  CUDACHK(cudaStreamSynchronize(stream));

  std::vector<double> times;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_s();
    run_one(buf, count, rank, comm, stream);
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

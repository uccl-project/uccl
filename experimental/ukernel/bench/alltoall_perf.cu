// alltoall_perf.cu — minimal ncclAllToAll bandwidth benchmark.
//
// Same binary runs against either the ukernel shim (LD_LIBRARY_PATH ->
// build/nccl/lib) or native NCCL (system libnccl), so the comparison is
// apples-to-apples on the standard API. nccl-tests' alltoall_perf uses
// ncclSend/ncclRecv, which the shim does not implement; this bench uses
// ncclAllToAll directly (shim implements it, native has it).
//
// Usage (N ranks, 1 GPU each, same node):
//   mpirun -np 1 ./alltoall_perf --rank=0 --nranks=N --bytes=268435456 \
//        : -np 1 ./alltoall_perf --rank=1 --nranks=N --bytes=268435456 \
//        : ... : -np 1 ./alltoall_perf --rank=N-1 --nranks=N --bytes=...
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
#include <unistd.h>

static const char* kIdPath = "/tmp/uk_a2a_id";
static const char* kFillPath = "/tmp/uk_a2a_fill";

// Build with -DUSE_SHIM_API for the ukernel shim (its nccl.h adds a
// ncclAllToAll extension; native NCCL has no such API). Without the
// define the harness uses ncclSend/ncclRecv to build the same N-rank
// alltoall exchange, which is what nccl-tests' alltoall_perf does on
// native NCCL. Same buffers, same count semantics (count = elements per
// rank pair; partition r of rank x is sent to rank r and partition r is
// received from rank r).

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

static void run_one(void* buf, size_t count, int rank, int nranks,
                    ncclComm_t comm, cudaStream_t stream) {
#ifdef USE_SHIM_API
  NCCLCHK(ncclAllToAll(buf, buf, count, ncclFloat, comm, stream));
#else
  // Native NCCL: no ncclAllToAll. nccl-tests pattern — one grouped
  // send/recv pair per peer (recv first, the conventional order),
  // including self (harmless no-op), so the exchange is one collective.
  NCCLCHK(ncclGroupStart());
  for (int p = 0; p < nranks; ++p) {
    size_t off = static_cast<size_t>(p) * count;
    NCCLCHK(ncclRecv(static_cast<char*>(buf) + off * sizeof(float), count,
                     ncclFloat, p, comm, stream));
    NCCLCHK(ncclSend(static_cast<char*>(buf) + off * sizeof(float), count,
                     ncclFloat, p, comm, stream));
  }
  NCCLCHK(ncclGroupEnd());
#endif
}

// Fill partition j with (rank*1000 + j), run one exchange, and check
// that partition i now holds (i*1000 + rank) — the value rank i stored
// in its partition `rank` before the exchange.
static bool verify_exchange(void* buf, size_t count, int rank, int nranks,
                            ncclComm_t comm, cudaStream_t stream) {
  std::vector<float> fill(static_cast<size_t>(nranks) * count);
  for (int p = 0; p < nranks; ++p)
    for (size_t i = 0; i < count; ++i)
      fill[static_cast<size_t>(p) * count + i] =
          static_cast<float>(rank * 1000 + p);
  CUDACHK(cudaMemcpy(buf, fill.data(), fill.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  // IPC puts are one-sided writes into the peer's buffer, so a peer's
  // verify-fill can race our puts (fill after a put lands overwrites the
  // exchanged data). Native NCCL has no ncclBarrier (shim extension), so
  // use a portable file handshake: every rank announces its fill done,
  // then waits until all N flags exist before the exchange starts.
  CUDACHK(cudaStreamSynchronize(stream));
  char flag[256];
  snprintf(flag, sizeof(flag), "%s_%d", kFillPath, rank);
  FILE* ff = fopen(flag, "w");
  if (!ff) { perror("fopen fill flag"); exit(1); }
  fclose(ff);
  for (int waited = 0; waited < 500; ++waited) {
    bool all = true;
    for (int p = 0; p < nranks; ++p) {
      char pf[256];
      snprintf(pf, sizeof(pf), "%s_%d", kFillPath, p);
      if (access(pf, F_OK) != 0) { all = false; break; }
    }
    if (all) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (rank == 0)
    for (int p = 0; p < nranks; ++p) {
      char pf[256];
      snprintf(pf, sizeof(pf), "%s_%d", kFillPath, p);
      remove(pf);
    }
  run_one(buf, count, rank, nranks, comm, stream);
  // Full device sync: the shim's IPC puts run on the adapter's own
  // streams; stream-syncing only our stream may race their completion.
  // (The shim's collective-completion signal should order this, but
  // verify must not depend on it — check the data after everything
  // lands.)
  CUDACHK(cudaDeviceSynchronize());
  // Scan the whole buffer for where peer data landed (diagnostic).
  std::vector<float> whole(static_cast<size_t>(nranks) * count);
  CUDACHK(cudaMemcpy(whole.data(), buf, whole.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
  size_t bad = 0, first_bad = whole.size();
  for (int p = 0; p < nranks; ++p) {
    float want = static_cast<float>(p * 1000 + rank);
    for (size_t i = 0; i < count; ++i) {
      float got = whole[static_cast<size_t>(p) * count + i];
      if (got != want) {
        ++bad;
        if (first_bad == whole.size()) first_bad = static_cast<size_t>(p) * count + i;
        if (bad <= 8)
          fprintf(stderr, "[r%d] verify bad[%zu] (part %d, idx %zu)=%.0f want=%.0f\n",
                  rank, static_cast<size_t>(p) * count + i, p, i, got, want);
      }
    }
  }
  fprintf(stderr, "[r%d] verify: bad=%zu/%zu first@%zu\n", rank, bad,
          whole.size(), first_bad);
  return bad == 0;
}

int main(int argc, char** argv) {
  int rank = (int)get_long_arg(argc, argv, "--rank", 0);
  int nranks = (int)get_long_arg(argc, argv, "--nranks", 2);
  size_t total_bytes = (size_t)get_long_arg(argc, argv, "--bytes", 1 << 28);
  int iters = (int)get_long_arg(argc, argv, "--iters", 20);
  int warmup = (int)get_long_arg(argc, argv, "--warmup", 5);
  int skip_verify = (int)get_long_arg(argc, argv, "--skip-verify", 0);

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
    if (!f) { fprintf(stderr, "r%d: no unique id file\n", rank); exit(1); }
    if (fread(&id, sizeof(id), 1, f) != 1) { fprintf(stderr, "id read fail\n"); exit(1); }
    fclose(f);
  }

  ncclComm_t comm;
  NCCLCHK(ncclCommInitRank(&comm, nranks, id, rank));

  void* buf = nullptr;
  CUDACHK(cudaMalloc(&buf, total_bytes));
  CUDACHK(cudaMemset(buf, 0, total_bytes));
  fprintf(stderr, "[r%d] buf=%p total=%zu\n", rank, buf, total_bytes);

  // count = elements per rank pair; total = nranks * count * 4 (float).
  size_t count = total_bytes / (sizeof(float) * nranks);

  cudaStream_t stream;
  CUDACHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // ncclCommInitRank blocks until every rank has joined, so by now all
  // ranks have read the id file — rank 0 may drop it for the next run.
  if (rank == 0) remove(kIdPath);

  if (!skip_verify) {
    bool vok = verify_exchange(buf, count, rank, nranks, comm, stream);
    fprintf(stderr, "[r%d] verify %s\n", rank, vok ? "OK" : "FAIL");
  }

  for (int i = 0; i < warmup; ++i)
    run_one(buf, count, rank, nranks, comm, stream);
  CUDACHK(cudaStreamSynchronize(stream));

  std::vector<double> times;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_s();
    run_one(buf, count, rank, nranks, comm, stream);
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

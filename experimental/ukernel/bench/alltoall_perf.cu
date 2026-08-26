// alltoall_perf.cu — minimal ncclAllToAll bandwidth benchmark.
//
// Same binary runs against either the ukernel shim (LD_LIBRARY_PATH ->
// build/nccl/lib) or native NCCL (system libnccl), so the comparison is
// apples-to-apples on the standard API. nccl-tests' alltoall_perf uses
// ncclSend/ncclRecv, which the shim does not implement; this bench uses
// ncclAllToAll directly (shim implements it, native has it).
//
// Usage (N ranks, 1 GPU each, same or multiple nodes):
//   mpirun -np N ./alltoall_perf --bytes=268435456 --iters=20 --warmup=5
// rank and nranks come from MPI; rank 0 generates the NCCL unique id and
// MPI_Bcast distributes it, so no shared file is needed across nodes.
// Report: algbw = (nranks-1)/nranks * total_bytes / avg_time (nccl-tests
// convention), busbw = algbw * nranks/(nranks-1). total_bytes is the full
// per-rank buffer (nranks * count * elemsz).

#include <mpi.h>
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

static const char* kFillPath = "/tmp/uk_a2a_fill";

// Zero a device buffer with a kernel instead of cudaMemset: on the L40S
// nodes the copy-engine memset's writes can still be draining when the
// shim's worker reduce/put reads the buffer, so the first round silently
// loses elements. Kernel-zero is ordered by kernel completion.
__global__ void zero_f32(float* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) p[i] = 0.0f;
}

// Uses the standard ncclAlltoAll (NCCL >= 2.19; the shim implements it
// too). Same count semantics (count = elements per rank pair;
// partition r of rank x is sent to rank r and partition r is received
// from rank r). Buffers: nccl-tests alltoall is NOT in-place — native
// Send/Recv with sendbuff==recvbuff aliases and corrupts the exchange
// ("We don't support in-place alltoall"). The harness always allocates
// a separate send and recv buffer; the shim path (ncclAllToAll) is
// in-place-only, so it exchanges recvbuf.

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

static void run_one(void* sendbuf, void* recvbuf, size_t count, int rank,
                    int nranks, ncclComm_t comm, cudaStream_t stream) {
  // Standard ncclAlltoAll: the shim implements it and native NCCL has it
  // (NCCL >= 2.19). One binary runs against either library.
  NCCLCHK(ncclAlltoAll(sendbuf, recvbuf, count, ncclFloat, comm, stream));
}

// Fill partition j with (rank*1000 + j), run one exchange, and check
// that partition i now holds (i*1000 + rank) — the value rank i stored
// in its partition `rank` before the exchange.
static bool verify_exchange(void* sendbuf, void* recvbuf, size_t count,
                            int rank, int nranks, ncclComm_t comm,
                            cudaStream_t stream) {
  std::vector<float> fill(static_cast<size_t>(nranks) * count);
  for (int p = 0; p < nranks; ++p)
    for (size_t i = 0; i < count; ++i)
      fill[static_cast<size_t>(p) * count + i] =
          static_cast<float>(rank * 1000 + p);
  CUDACHK(cudaMemcpy(sendbuf, fill.data(), fill.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(recvbuf, fill.data(), fill.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  // IPC puts are one-sided writes into the peer's buffer, so a peer's
  // verify-fill can race our puts (fill after a put lands overwrites the
  // exchanged data). NCCL (native or shim) exposes no barrier primitive,
  // so use a portable file handshake: every rank announces its fill done,
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
  run_one(sendbuf, recvbuf, count, rank, nranks, comm, stream);
  // Full device sync: the shim's IPC puts run on the adapter's own
  // streams; stream-syncing only our stream may race their completion.
  // (The shim's collective-completion signal should order this, but
  // verify must not depend on it — check the data after everything
  // lands.)
  CUDACHK(cudaDeviceSynchronize());
  // Scan the whole buffer for where peer data landed (diagnostic).
  std::vector<float> whole(static_cast<size_t>(nranks) * count);
  CUDACHK(cudaMemcpy(whole.data(), recvbuf, whole.size() * sizeof(float),
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
  MPI_Init(&argc, &argv);
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int rank = (int)get_long_arg(argc, argv, "--rank", mpi_rank);
  int nranks = (int)get_long_arg(argc, argv, "--nranks", mpi_size);
  int dev = (int)get_long_arg(argc, argv, "--dev", rank);
  size_t total_bytes = (size_t)get_long_arg(argc, argv, "--bytes", 1 << 28);
  int iters = (int)get_long_arg(argc, argv, "--iters", 20);
  int warmup = (int)get_long_arg(argc, argv, "--warmup", 5);
  int skip_verify = (int)get_long_arg(argc, argv, "--skip-verify", 0);

  CUDACHK(cudaSetDevice(dev));

  ncclUniqueId id;
  if (mpi_rank == 0) NCCLCHK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  NCCLCHK(ncclCommInitRank(&comm, nranks, id, rank));

  void* sendbuf = nullptr;
  void* recvbuf = nullptr;
  CUDACHK(cudaMalloc(&sendbuf, total_bytes));
  CUDACHK(cudaMalloc(&recvbuf, total_bytes));
  {
    size_t n = total_bytes / sizeof(float);
    unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    if (blocks == 0) blocks = 1;
    zero_f32<<<blocks, 256>>>(static_cast<float*>(recvbuf), n);
    CUDACHK(cudaGetLastError());
    CUDACHK(cudaDeviceSynchronize());
  }
  fprintf(stderr, "[r%d] send=%p recv=%p total=%zu\n", rank, sendbuf,
          recvbuf, total_bytes);

  // count = elements per rank pair; total = nranks * count * 4 (float).
  size_t count = total_bytes / (sizeof(float) * nranks);

  cudaStream_t stream;
  CUDACHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  if (!skip_verify) {
    bool vok = verify_exchange(sendbuf, recvbuf, count, rank, nranks, comm,
                               stream);
    fprintf(stderr, "[r%d] verify %s\n", rank, vok ? "OK" : "FAIL");
  }

  for (int i = 0; i < warmup; ++i)
    run_one(sendbuf, recvbuf, count, rank, nranks, comm, stream);
  CUDACHK(cudaStreamSynchronize(stream));

  std::vector<double> times;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_s();
    run_one(sendbuf, recvbuf, count, rank, nranks, comm, stream);
    CUDACHK(cudaStreamSynchronize(stream));
    double dt = now_s() - t0;
    if (i >= 2) times.push_back(dt);
  }

  double avg = 0;
  for (double t : times) avg += t;
  avg /= times.size();
  // nccl-tests alltoall convention (AlltoAllGetBw): algbw = full
  // per-rank buffer / time; busbw = algbw * (nranks-1)/nranks.
  double algbw = total_bytes / avg / 1e9;
  double busbw = algbw * (double)(nranks - 1) / nranks;
  fprintf(stderr,
          "[r%d] total=%zuMB iters=%zu avg=%.1fus algbw=%.1fGB/s busbw=%.1fGB/s\n",
          rank, total_bytes >> 20, times.size(), avg * 1e6, algbw, busbw);

  NCCLCHK(ncclCommDestroy(comm));
  CUDACHK(cudaFree(sendbuf));
  CUDACHK(cudaFree(recvbuf));
  MPI_Finalize();
  return 0;
}

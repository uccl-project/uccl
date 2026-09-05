// stream_concurrent.cu - concurrent-collective benchmark across CUDA
// streams (G1 validation; primary target L40S).
//
// Scenarios (--scenario):
//   fsdp2   : 2 streams: AllGather (next-layer param prefetch) +
//             ReduceScatter (current-layer grads)  -- FSDP backward
//             prefetch shape
//   fsdp4   : 4 streams: AG, RS, AG, RS
//   ar2/ar4 : c streams of AllReduce (same-shape control)
//   seqfsdp : the fsdp2 workload issued back-to-back on ONE stream
//             (sequential reference)
//
// Layer bytes W = per-rank full parameter tensor. Per NCCL convention
// the AllGather input is W/nranks and the ReduceScatter output is
// W/nranks; nccl-tests busbw for AG/RS is (n-1)/n * W / t and for AR
// it is 2(n-1)/n * W / t, so aggregate busbw below uses those factors.
//
// The same binary runs against the shim (LD_LIBRARY_PATH ->
// build/nccl/lib) or native NCCL. --group-mode selects how groups wrap
// the API calls: per-op (one ncclGroupStart/End per collective, which
// is what PyTorch's ProcessGroupNCCL does internally), batch (one
// group around all ops of an iteration), or none (bare calls).
// --comm-mode shared uses one ncclComm for all ops (FSDP drop-in);
// per-op gives every op its own communicator (FSDP2 / NCCL comm-split
// semantics).

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define CUDACHK(c)                                                       \
  do {                                                                   \
    cudaError_t e = (c);                                                 \
    if (e != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
              cudaGetErrorString(e), __FILE__, __LINE__);                \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

#define NCCLCHK(c)                                                       \
  do {                                                                   \
    ncclResult_t r = (c);                                                \
    if (r != ncclSuccess) {                                              \
      fprintf(stderr, "NCCL error %d at %s:%d\n", r, __FILE__, __LINE__); \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

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

static const char* get_str_arg(int argc, char** argv, const char* name,
                               const char* def) {
  std::string n(name);
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == n && i + 1 < argc) return argv[i + 1];
    if (a.size() > n.size() && a.compare(0, n.size(), n) == 0)
      return a.c_str() + n.size() + 1;
  }
  return def;
}

__global__ void fill_f32(float* p, size_t n, float v) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) p[i] = v;
}

struct Op {
  int kind;  // 0 = AG, 1 = RS, 2 = AR
  cudaStream_t stream;
  void* a = nullptr;  // AG shard-in / RS full-in / AR input
  void* b = nullptr;  // AG full-out / RS shard-out / AR output
  size_t a_elems = 0;
  size_t b_elems = 0;
  std::vector<float> lat_ms;  // per-iteration elapsed between markers
};

static float median(std::vector<float>& v) {
  if (v.empty()) return 0.f;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static float percentile(std::vector<float>& v, float p) {
  if (v.empty()) return 0.f;
  std::sort(v.begin(), v.end());
  size_t idx = (size_t)(p * (float)(v.size() - 1));
  return v[idx];
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  const char* scenario = get_str_arg(argc, argv, "--scenario", "fsdp2");
  const char* comm_mode = get_str_arg(argc, argv, "--comm-mode", "shared");
  const char* group_mode = get_str_arg(argc, argv, "--group-mode", "per-op");
  size_t W = (size_t)get_long_arg(argc, argv, "--layer-bytes", 64 << 20);
  int iters = (int)get_long_arg(argc, argv, "--iters", 30);
  int warmup = (int)get_long_arg(argc, argv, "--warmup", 5);
  int skip_verify = (int)get_long_arg(argc, argv, "--skip-verify", 0);
  // --sync-every K: issue K batches, then cudaDeviceSynchronize.
  // K=1 measures end-to-end wall per batch (host waits every batch);
  // larger K lets host dispatch run ahead of GPU execution (the FSDP
  // prefetch regime), K=iters is fully pipelined. Average wall per
  // batch is reported either way.
  int sync_every = (int)get_long_arg(argc, argv, "--sync-every", 1);

  int dev = mpi_rank;
  if (const char* lr = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
    dev = std::atoi(lr);
  CUDACHK(cudaSetDevice(dev));

  int n = mpi_size;
  if (W < (size_t)n) W = (size_t)n;  // shard must not be empty
  size_t shard = W / (size_t)n;
  size_t shard_elems = shard / sizeof(float);
  size_t full_elems = W / sizeof(float);
  if (shard_elems == 0 || full_elems == 0) {
    fprintf(stderr, "[r%d] W=%zu n=%d too small\n", mpi_rank, W, n);
    return 2;
  }

  // Scenario -> per-op kinds and streams.
  std::vector<Op> ops;
  std::string scen(scenario);
  bool seq = false;
  if (scen == "fsdp2") {
    ops.resize(2);
    ops[0].kind = 0; ops[1].kind = 1;
  } else if (scen == "fsdp4") {
    ops.resize(4);
    ops[0].kind = 0; ops[1].kind = 1; ops[2].kind = 0; ops[3].kind = 1;
  } else if (scen == "ar2" || scen == "ar4") {
    int c = scen == "ar2" ? 2 : 4;
    ops.resize(c);
    for (auto& o : ops) o.kind = 2;
  } else if (scen == "seqfsdp") {
    ops.resize(2);
    ops[0].kind = 0; ops[1].kind = 1;
    seq = true;
  } else {
    fprintf(stderr, "unknown scenario %s\n", scenario);
    return 2;
  }

  // One communicator (shared) or one per op (per-op).
  std::string cm(comm_mode);
  int ncomms = cm == "per-op" ? (int)ops.size() : 1;
  if (cm != "shared" && cm != "per-op") {
    fprintf(stderr, "unknown comm-mode %s\n", comm_mode);
    return 2;
  }
  std::vector<ncclUniqueId> ids(ncomms);
  if (mpi_rank == 0)
    for (int i = 0; i < ncomms; ++i) NCCLCHK(ncclGetUniqueId(&ids[i]));
  for (int i = 0; i < ncomms; ++i)
    MPI_Bcast(&ids[i], (int)sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  std::vector<ncclComm_t> comms(ncomms);
  for (int i = 0; i < ncomms; ++i)
    NCCLCHK(ncclCommInitRank(&comms[i], mpi_size, ids[i], mpi_rank));

  // Allocate per-op buffers and streams.
  for (size_t i = 0; i < ops.size(); ++i) {
    auto& o = ops[i];
    if (seq && i > 0) {
      o.stream = ops[0].stream;  // serialize on one stream
    } else {
      CUDACHK(cudaStreamCreateWithFlags(&o.stream, cudaStreamNonBlocking));
    }
    if (o.kind == 0) {  // AG: shard in, full out
      o.a_elems = shard_elems;
      o.b_elems = full_elems;
    } else if (o.kind == 1) {  // RS: full in, shard out
      o.a_elems = full_elems;
      o.b_elems = shard_elems;
    } else {  // AR: full in, full out
      o.a_elems = full_elems;
      o.b_elems = full_elems;
    }
    CUDACHK(cudaMalloc(&o.a, o.a_elems * sizeof(float)));
    CUDACHK(cudaMalloc(&o.b, o.b_elems * sizeof(float)));
    unsigned blocks_a = (unsigned)((o.a_elems + 255) / 256);
    unsigned blocks_b = (unsigned)((o.b_elems + 255) / 256);
    fill_f32<<<blocks_a, 256>>>((float*)o.a, o.a_elems, 1.0f);
    fill_f32<<<blocks_b, 256>>>((float*)o.b, o.b_elems, 0.0f);
    CUDACHK(cudaGetLastError());
  }
  CUDACHK(cudaDeviceSynchronize());

  std::string gm(group_mode);
  bool gm_batch = gm == "batch";
  bool gm_perop = gm == "per-op";
  bool gm_none = gm == "none";
  if (!gm_batch && !gm_perop && !gm_none) {
    fprintf(stderr, "unknown group-mode %s\n", group_mode);
    return 2;
  }

  auto launch_batch = [&](std::vector<cudaEvent_t>& starts,
                          std::vector<cudaEvent_t>& stops) {
    if (gm_batch) NCCLCHK(ncclGroupStart());
    for (size_t i = 0; i < ops.size(); ++i) {
      auto& o = ops[i];
      ncclComm_t comm = cm == "per-op" ? comms[i] : comms[0];
      CUDACHK(cudaEventRecord(starts[i], o.stream));
      if (gm_perop) NCCLCHK(ncclGroupStart());
      if (o.kind == 0) {
        NCCLCHK(ncclAllGather(o.a, o.b, shard_elems, ncclFloat, comm,
                              o.stream));
      } else if (o.kind == 1) {
        NCCLCHK(ncclReduceScatter(o.a, o.b, shard_elems, ncclFloat,
                                  ncclSum, comm, o.stream));
      } else {
        NCCLCHK(ncclAllReduce(o.a, o.b, full_elems, ncclFloat, ncclSum,
                              comm, o.stream));
      }
      if (gm_perop) NCCLCHK(ncclGroupEnd());
      CUDACHK(cudaEventRecord(stops[i], o.stream));
    }
    if (gm_batch) NCCLCHK(ncclGroupEnd());
  };

  std::vector<cudaEvent_t> starts(ops.size()), stops(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    CUDACHK(cudaEventCreate(&starts[i]));
    CUDACHK(cudaEventCreate(&stops[i]));
  }

  // Warmup.
  for (int i = 0; i < warmup; ++i) {
    launch_batch(starts, stops);
    CUDACHK(cudaDeviceSynchronize());
  }

  // One-shot correctness: inputs are 1.0; AG output must be 1.0,
  // RS (sum over n shards) and AR outputs must be n.
  if (!skip_verify) {
    launch_batch(starts, stops);
    CUDACHK(cudaDeviceSynchronize());
    for (auto& o : ops) {
      std::vector<float> host(o.b_elems);
      CUDACHK(cudaMemcpy(host.data(), o.b, o.b_elems * sizeof(float),
                         cudaMemcpyDeviceToHost));
      float want = o.kind == 0 ? 1.0f : (float)n;
      size_t bad = 0;
      for (size_t k = 0; k < host.size(); ++k)
        if (host[k] != want) ++bad;
      if (bad) {
        fprintf(stderr, "[r%d] op kind %d verify bad=%zu/%zu\n", mpi_rank,
                o.kind, bad, host.size());
        return 3;
      }
    }
  }

  // Timed loop: sync every K batches, report average wall per batch.
  // Aggregate busbw uses nccl-tests factors.
  int K = sync_every;
  if (K < 1) K = 1;
  if (K > iters) K = iters;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    launch_batch(starts, stops);
    if ((i + 1) % K == 0) CUDACHK(cudaDeviceSynchronize());
  }
  if (iters % K != 0) CUDACHK(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  double wall_med_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() /
      (double)iters;
  for (size_t k = 0; k < ops.size(); ++k) {
    for (int i = 0; i < iters; ++i) {
      float ms = 0.f;
      CUDACHK(cudaEventElapsedTime(&ms, starts[k], stops[k]));
      ops[k].lat_ms.push_back(ms);
    }
  }

  double factor = (double)(n - 1) / (double)n;  // AG/RS
  double bus_bytes = 0.0;
  for (auto& o : ops)
    bus_bytes += (o.kind == 2 ? 2.0 : 1.0) * factor * (double)W;
  double agg_busbw = bus_bytes / (wall_med_us * 1e-6) / 1e9;

  if (mpi_rank == 0) {
    fprintf(stderr,
            "[r0] scenario=%s comm=%s group=%s n=%d W=%zuMB iters=%d "
            "syncK=%d wall_us=%.1f agg_busbw=%.2fGB/s\n",
            scenario, comm_mode, group_mode, n, W >> 20, iters, K, wall_med_us,
            agg_busbw);
    for (size_t i = 0; i < ops.size(); ++i) {
      fprintf(stderr,
              "  op%zu kind=%d lat_ms p50=%.3f p99=%.3f mean=%.3f\n", i,
              ops[i].kind, median(ops[i].lat_ms),
              percentile(ops[i].lat_ms, 0.99),
              ops[i].lat_ms.empty()
                  ? 0.f
                  : std::accumulate(ops[i].lat_ms.begin(),
                                    ops[i].lat_ms.end(), 0.f) /
                        (float)ops[i].lat_ms.size());
    }
  }

  for (size_t i = 0; i < ops.size(); ++i) {
    CUDACHK(cudaEventDestroy(starts[i]));
    CUDACHK(cudaEventDestroy(stops[i]));
  }
  for (int i = 0; i < ncomms; ++i) NCCLCHK(ncclCommDestroy(comms[i]));
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

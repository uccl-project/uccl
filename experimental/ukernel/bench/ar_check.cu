// ar_check.cu — locate the exact bad indices in a fused allreduce.
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mpi.h>

#define CUDACHK(c)                                                          \
  do {                                                                      \
    cudaError_t e = (c);                                                    \
    if (e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),    \
              __FILE__, __LINE__);                                          \
      return 1;                                                             \
    }                                                                       \
  } while (0)

#define NCCLCHK(c)                                                          \
  do {                                                                      \
    ncclResult_t r = (c);                                                   \
    if (r != ncclSuccess) {                                                 \
      fprintf(stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString(r),    \
              __FILE__, __LINE__);                                          \
      return 1;                                                             \
    }                                                                       \
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

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int rank = (int)get_long_arg(argc, argv, "--rank", mpi_rank);
  int nranks = (int)get_long_arg(argc, argv, "--nranks", mpi_size);
  int dev = (int)get_long_arg(argc, argv, "--dev", rank);
  size_t bytes = (size_t)get_long_arg(argc, argv, "--bytes", 8 << 20);

  CUDACHK(cudaSetDevice(dev));
  ncclUniqueId id;
  if (mpi_rank == 0) NCCLCHK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  NCCLCHK(ncclCommInitRank(&comm, nranks, id, rank));

  size_t count = bytes / sizeof(float);
  float* sendbuf = nullptr;
  float* recvbuf = nullptr;
  CUDACHK(cudaMalloc(&sendbuf, bytes));
  CUDACHK(cudaMalloc(&recvbuf, bytes));
  {
    std::vector<float> h(count);
    for (size_t i = 0; i < count; ++i) h[i] = (float)((i * 2654435761u + rank * 97u) % 1000000u) / 1000000.0f;
    CUDACHK(cudaMemcpy(sendbuf, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemset(recvbuf, 0, bytes));
  }
  cudaStream_t stream;
  CUDACHK(cudaStreamCreate(&stream));

  NCCLCHK(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream));
  CUDACHK(cudaStreamSynchronize(stream));

  std::vector<float> out(count);
  CUDACHK(cudaMemcpy(out.data(), recvbuf, bytes, cudaMemcpyDeviceToHost));
  // Exact expected sum via MPI_Allreduce of the host copies.
  std::vector<float> h(count);
  CUDACHK(cudaMemcpy(h.data(), sendbuf, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> expect(count, 0.0f);
  MPI_Allreduce(h.data(), expect.data(), (int)count, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
  int bad = 0;
  size_t first[32];
  float vals[32];
  float devs[32];
  size_t bad_lo = count, bad_hi = 0;
  size_t bad_total = 0;
  for (size_t i = 0; i < count && bad < 32; ++i) {
    float dev = out[i] - expect[i];
    if (dev < -1e-2f || dev > 1e-2f) {
      first[bad] = i;
      vals[bad] = out[i];
      devs[bad] = dev;
      ++bad;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    float dev = out[i] - expect[i];
    if (dev < -1e-2f || dev > 1e-2f) {
      ++bad_total;
      if (i < bad_lo) bad_lo = i;
      if (i > bad_hi) bad_hi = i;
    }
  }
  fprintf(stderr,
          "[ar-check r%d] count=%zu bad_total=%zu range=[%zu,%zu] "
          "first=%d\n",
          rank, count, bad_total, bad_lo, bad_hi, bad);
  for (int k = 0; k < bad; ++k)
    fprintf(stderr, "[ar-check r%d] bad[%zu] = %f expect=%f dev=%f\n", rank,
            first[k], vals[k], expect[first[k]], devs[k]);

  NCCLCHK(ncclCommDestroy(comm));
  CUDACHK(cudaFree(sendbuf));
  CUDACHK(cudaFree(recvbuf));
  MPI_Finalize();
  return 0;
}

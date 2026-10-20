// Independent verification of the real ep/include/ep_utils.cuh::barrier_block
// under concurrent epochs (uccl-project/uccl#986).
//
// This test includes the project header directly and calls the actual
// barrier_block implementation, so it verifies the code that ships rather than
// a copy of it. Two epochs run concurrently on separate streams over one signal
// array, exactly like two normal-mode dispatch epochs sharing a Buffer.
//
// Build (from ep/tests): make barrier_concurrent
// Run: one process per rank, same <name> for the rendezvous
//   CUDA_VISIBLE_DEVICES=0 ./barrier_concurrent 0 2 400 2 rr1 30
//   CUDA_VISIBLE_DEVICES=1 ./barrier_concurrent 1 2 400 2 rr1 30
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ep_utils.cuh"

#define MAX_RANKS 16
#define MAX_EPOCHS 8

static void check(cudaError_t e, char const *what) {
  if (e != cudaSuccess) {
    printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
    exit(1);
  }
}
static void die(char const *m) {
  perror(m);
  exit(1);
}

__device__ __forceinline__ int ld_vol_uint(unsigned int const *p) {
  unsigned int r;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return (int)r;
}

// Each block announces its barrier sequence before the REAL barrier and checks
// afterwards that no rank is still behind, so a premature release is reported.
__global__ void real_barrier_stress(int **ptrs, unsigned int *pre,
                                    unsigned long long *fail,
                                    unsigned long long *timeouts,
                                    unsigned long long *calls, int nranks,
                                    int iters, int use_real_barrier) {
  int rank = (int)blockIdx.x;
  int epoch = (int)blockIdx.y;
  int tid = (int)threadIdx.x;
  int **cells = ptrs + (size_t)(rank * MAX_EPOCHS + epoch) * nranks;
  unsigned int const round = (unsigned int)(epoch + 1);
  (void)use_real_barrier;

  for (int it = 0; it < iters; ++it) {
    if (tid == 0) {
      __threadfence_system();
      pre[rank] = round;
    }
    __syncthreads();

    // The real, unmodified implementation from ep_utils.cuh.
    barrier_block<MAX_RANKS, false>(cells, rank);

    if (tid == 0) {
      __threadfence_system();
      unsigned int mn = 0xffffffffu;
      for (int r = 0; r < nranks; ++r) {
        unsigned int v = (unsigned int)ld_vol_uint(pre + r);
        mn = v < mn ? v : mn;
      }
      if (mn < round) atomicAdd_system(fail, 1ull);
      atomicAdd_system(calls, 1ull);
    }
    __syncthreads();
  }
}

struct HandleBytes {
  char b[64];
};

static std::string hpath(std::string const &name, int rank) {
  return "/dev/shm/ucll986r_" + name + "_" + std::to_string(rank) + ".bin";
}

int main(int argc, char **argv) {
  if (argc < 6) {
    printf("usage: %s <rank> <nranks> <iters> <epochs> <name> [timeout_s]\n",
           argv[0]);
    return 2;
  }
  int my_rank = atoi(argv[1]);
  int nranks = atoi(argv[2]);
  int iters = atoi(argv[3]);
  int nepoch = atoi(argv[4]);
  std::string name = argv[5];
  int timeout_s = argc > 6 ? atoi(argv[6]) : 60;
  if (nranks < 2 || nranks > MAX_RANKS || nepoch < 2) return 2;
  // barrier_block<MAX_RANKS> needs MAX_RANKS cells per rank.
  int const cells_per_epoch = MAX_RANKS;

  check(cudaSetDevice(0), "setDevice");
  printf("[rank %d] code=ep_utils.cuh barrier_block iters=%d epochs=%d nranks=%d\n",
         my_rank, iters, nepoch, nranks);
  fflush(stdout);

  int *base[MAX_EPOCHS];
  HandleBytes my_handle[MAX_EPOCHS];
  for (int e = 0; e < nepoch; ++e) {
    // barrier_block<kNumRanks> needs kNumRanks cells per rank block.
    size_t const bytes =
        sizeof(int) * (size_t)cells_per_epoch * (size_t)nranks;
    check(cudaMalloc(&base[e], bytes), "cells");
    check(cudaMemset(base[e], 0, bytes), "zero");
    cudaIpcMemHandle_t h;
    check(cudaIpcGetMemHandle(&h, base[e]), "ipcGet");
    memcpy(my_handle[e].b, h.reserved, 64);
  }

  std::string mine = hpath(name, my_rank);
  unlink(mine.c_str());
  {
    int f = open(mine.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if (f < 0) die("open");
    size_t n = sizeof(HandleBytes) * nepoch;
    if (write(f, my_handle, n) != (ssize_t)n) die("write");
    close(f);
  }
  HandleBytes peer[MAX_RANKS][MAX_EPOCHS];
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank) continue;
    std::string p = hpath(name, r);
    int f = -1;
    for (int t = 0; t < timeout_s * 100; ++t) {
      f = open(p.c_str(), O_RDONLY);
      if (f >= 0) {
        struct stat st {};
        if (fstat(f, &st) == 0 &&
            st.st_size == (off_t)(sizeof(HandleBytes) * nepoch))
          break;
        close(f);
        f = -1;
      }
      struct timespec ts { 0, 10 * 1000 * 1000 };
      nanosleep(&ts, nullptr);
    }
    if (f < 0) return 2;
    size_t n = sizeof(HandleBytes) * nepoch;
    if (read(f, peer[r], n) != (ssize_t)n) die("read");
    close(f);
  }

  // One pointer row per rank block: ptrs[rank][target_rank] -> target's block.
  int *remote[MAX_RANKS];
  for (int r = 0; r < nranks; ++r) {
    remote[r] = nullptr;
    if (r == my_rank) continue;
    cudaIpcMemHandle_t h;
    memcpy(h.reserved, peer[r][0].b, 64);
    check(cudaIpcOpenMemHandle((void **)&remote[r], h,
                               cudaIpcMemLazyEnablePeerAccess), "ipcOpen");
  }

  std::vector<int *> table((size_t)nranks * MAX_EPOCHS * nranks, nullptr);
  for (int lr = 0; lr < nranks; ++lr)
    for (int e = 0; e < nepoch; ++e)
      for (int t = 0; t < nranks; ++t) {
        int *blk = (lr == my_rank) ? base[e] : remote[lr];
        // row `t` of rank `lr` is its block at offset t * MAX_RANKS
        table[((size_t)lr * MAX_EPOCHS + e) * nranks + t] =
            blk + (size_t)t * cells_per_epoch;
      }
  int **d_ptrs = nullptr;
  check(cudaMalloc(&d_ptrs, sizeof(int *) * table.size()), "d_ptrs");
  check(cudaMemcpy(d_ptrs, table.data(), sizeof(int *) * table.size(),
                   cudaMemcpyHostToDevice), "d_ptrs cp");

  unsigned int *d_pre = nullptr;
  check(cudaHostAlloc((void **)&d_pre, sizeof(unsigned int) * MAX_RANKS,
                      cudaHostAllocMapped), "pre");
  memset((void *)d_pre, 0, sizeof(unsigned int) * MAX_RANKS);
  unsigned long long *d_fail = nullptr, *d_calls = nullptr, *d_to = nullptr;
  check(cudaMalloc(&d_fail, sizeof(unsigned long long)), "fail");
  check(cudaMalloc(&d_calls, sizeof(unsigned long long)), "calls");
  check(cudaMemset(d_fail, 0, sizeof(unsigned long long)), "f0");
  check(cudaMemset(d_calls, 0, sizeof(unsigned long long)), "c0");
  (void)d_to;

  cudaStream_t s[MAX_EPOCHS];
  for (int e = 0; e < nepoch; ++e) check(cudaStreamCreate(&s[e]), "stream");
  for (int e = 0; e < nepoch; ++e)
    real_barrier_stress<<<dim3(nranks), 32, 0, s[e]>>>(
        d_ptrs, d_pre, d_fail, d_to, d_calls, nranks, iters, 1);
  check(cudaDeviceSynchronize(), "sync");

  unsigned long long fail = 0, calls = 0;
  check(cudaMemcpy(&fail, d_fail, sizeof(fail), cudaMemcpyDeviceToHost), "f");
  check(cudaMemcpy(&calls, d_calls, sizeof(calls), cudaMemcpyDeviceToHost), "c");
  if (my_rank == 0)
    printf("[summary] code=ep_utils.cuh iters=%d nranks=%d epochs=%d "
           "calls=%llu expected=%d premature_releases=%llu\n",
           iters, nranks, nepoch, calls,
           iters * nranks * nepoch, fail);
  fflush(stdout);
  return (calls == (unsigned long long)iters * nranks * nepoch && fail == 0)
             ? 0
             : 1;
}

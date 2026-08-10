// CE (copy engine) contention microbenchmark.
//
// 8 ranks alltoall with cudaMemcpyAsync P2P, in two modes:
//   unsync : each rank runs its own loop (ranks drift apart) — the
//            copies never all peak at once.
//   sync   : a file barrier before every iteration — all 8 ranks issue
//            their 7 copies at the same instant (the shim's collective
//            pattern).
// The contrast isolates the CE/NVLink queueing under a synchronized
// peak: if the same copies take much longer in sync mode, the CE is the
// bottleneck, not the host.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define CK(c)                                                          \
  do {                                                                 \
    cudaError_t e = (c);                                               \
    if (e != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA %s %d\n", cudaGetErrorString(e), __LINE__); \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

static long garg(int argc, char** argv, const char* name, long def) {
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
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Simple vectorized SM copy (used in "sm" mode to contrast with the CE).
// Reads peer memory over NVLink, writes local; one kernel per peer chunk.
__global__ void copy_kernel(const char4* __restrict__ src,
                            char4* __restrict__ dst, size_t n4) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  size_t stride = gridDim.x * static_cast<size_t>(blockDim.x);
  for (; i < n4; i += stride) dst[i] = src[i];
}

// Shared-memory sense-reversing barrier (mmap'd file).
struct BarState {
  std::atomic<uint32_t> arrive;
  std::atomic<uint32_t> gen;
};
static BarState* g_bar = nullptr;

static void barrier_all(int rank, int n);

static void barrier_init(int rank, int n, const char* tag) {
  char p[64];
  snprintf(p, sizeof(p), "/tmp/ce_bar_%s", tag);
  int fd = -1;
  if (rank == 0) {
    fd = open(p, O_CREAT | O_TRUNC | O_RDWR, 0666);
    if (fd < 0 || ftruncate(fd, sizeof(BarState)) != 0) {
      fprintf(stderr, "[r%d] barrier create failed\n", rank);
      exit(1);
    }
  } else {
    for (int i = 0; i < 500 && fd < 0; ++i) {
      fd = open(p, O_RDWR);
      if (fd < 0) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
  if (fd < 0) {
    fprintf(stderr, "[r%d] barrier open failed\n", rank);
    exit(1);
  }
  void* m = mmap(nullptr, sizeof(BarState), PROT_READ | PROT_WRITE, MAP_SHARED,
                 fd, 0);
  close(fd);
  g_bar = static_cast<BarState*>(m);
  if (rank == 0) {
    g_bar->arrive.store(0, std::memory_order_relaxed);
    g_bar->gen.store(0, std::memory_order_relaxed);
  }
  barrier_all(rank, n);  // sync the init
}

static void barrier_all(int rank, int n) {
  uint32_t mygen = g_bar->gen.load(std::memory_order_acquire);
  if (g_bar->arrive.fetch_add(1, std::memory_order_acq_rel) ==
      static_cast<uint32_t>(n - 1)) {
    g_bar->arrive.store(0, std::memory_order_relaxed);
    g_bar->gen.fetch_add(1, std::memory_order_release);
  } else {
    while (g_bar->gen.load(std::memory_order_acquire) == mygen) {
      std::this_thread::yield();
    }
  }
}

int main(int argc, char** argv) {
  setbuf(stderr, NULL);
  int rank = static_cast<int>(garg(argc, argv, "--rank", 0));
  int n = static_cast<int>(garg(argc, argv, "--nranks", 8));
  size_t total = static_cast<size_t>(garg(argc, argv, "--bytes", 1 << 28));
  int iters = static_cast<int>(garg(argc, argv, "--iters", 20));
  int sm = static_cast<int>(garg(argc, argv, "--sm", 0));
  int serial = static_cast<int>(garg(argc, argv, "--serial", 0));
  CK(cudaSetDevice(rank));
  barrier_init(rank, n, "main");

  // IPC handle exchange: write my handle + rdy flag, wait for all rdys,
  // then read all handles.
  cudaIpcMemHandle_t* handles = new cudaIpcMemHandle_t[n];
  void* send = nullptr;
  CK(cudaMalloc(&send, total));
  CK(cudaMemset(send, 0, total));
  char idp[64], rdp[64];
  snprintf(idp, sizeof(idp), "/tmp/ce_h_%d", rank);
  snprintf(rdp, sizeof(rdp), "/tmp/ce_rdy_%d", rank);
  remove(idp);
  remove(rdp);
  FILE* f = fopen(idp, "wb");
  CK(cudaIpcGetMemHandle(&handles[rank], send));
  fwrite(&handles[rank], sizeof(cudaIpcMemHandle_t), 1, f);
  fclose(f);
  f = fopen(rdp, "w");
  fclose(f);
  for (int r = 0; r < n; ++r) {
    char p[64];
    snprintf(p, sizeof(p), "/tmp/ce_rdy_%d", r);
    for (int i = 0; i < 500; ++i) {
      FILE* fp = fopen(p, "r");
      if (fp) {
        fclose(fp);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
  std::vector<void*> recvs(static_cast<size_t>(n));
  std::vector<cudaStream_t> st(static_cast<size_t>(n));
  for (int r = 0; r < n; ++r) {
    char p[64];
    snprintf(p, sizeof(p), "/tmp/ce_h_%d", r);
    f = fopen(p, "rb");
    if (!f) {
      fprintf(stderr, "no handle %d\n", r);
      exit(1);
    }
    if (r != rank)
      if (fread(&handles[r], sizeof(cudaIpcMemHandle_t), 1, f) != 1) {
        fprintf(stderr, "bad handle read %d\n", r);
        exit(1);
      }
    fclose(f);
    if (r == rank) {
      recvs[static_cast<size_t>(r)] = send;
    } else {
      CK(cudaIpcOpenMemHandle(&recvs[static_cast<size_t>(r)],
                              handles[static_cast<size_t>(r)],
                              cudaIpcMemLazyEnablePeerAccess));
    }
    CK(cudaStreamCreateWithFlags(&st[static_cast<size_t>(r)],
                                 cudaStreamNonBlocking));
  }

  size_t part = total / static_cast<size_t>(n);
  auto exchange = [&] {
    for (int p = 0; p < n; ++p) {
      if (p == rank) continue;
      char* dst = static_cast<char*>(recvs[static_cast<size_t>(p)]) +
                  part * static_cast<size_t>(rank);
      const char* src =
          static_cast<const char*>(send) + part * static_cast<size_t>(p);
      cudaStream_t s = serial ? st[0] : st[static_cast<size_t>(p)];
      if (sm) {
        copy_kernel<<<512, 256, 0, s>>>(
            reinterpret_cast<const char4*>(src),
            reinterpret_cast<char4*>(dst), part / sizeof(char4));
      } else {
        CK(cudaMemcpyAsync(dst, src, part, cudaMemcpyDeviceToDevice, s));
      }
    }
    if (serial) {
      CK(cudaStreamSynchronize(st[0]));
    } else {
      for (int p = 0; p < n; ++p)
        if (p != rank) CK(cudaStreamSynchronize(st[static_cast<size_t>(p)]));
    }
  };

  auto measure = [&](const char* mode, bool sync) {
    for (int i = 0; i < 5; ++i) {
      if (sync) barrier_all(rank, n);
      exchange();
    }
    std::vector<double> ts;
    for (int i = 0; i < iters; ++i) {
      if (sync) barrier_all(rank, n);
      double t0 = now_s();
      exchange();
      ts.push_back(now_s() - t0);
    }
    double avg = 0;
    for (double t : ts) avg += t;
    avg /= ts.size();
    double per_rank_bw = static_cast<double>(n - 1) / n * total / avg / 1e9;
    fprintf(stderr,
            "[r%d] %s: avg=%.1fus per-rank-out=%.1fGB/s per-copy=%.1fGB/s "
            "(copy=%.1fus)\n",
            rank, mode, avg * 1e6, per_rank_bw,
            per_rank_bw / static_cast<double>(n - 1),
            avg * 1e6 / static_cast<double>(n - 1));
    return avg;
  };

  const char* tag = serial ? (sm ? "sm-serial" : "serial") : (sm ? "sm" : "");
  measure((std::string(tag) + "-unsync").c_str(), false);
  measure((std::string(tag) + "-sync").c_str(), true);
  return 0;
}

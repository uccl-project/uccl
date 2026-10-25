// Round-pairing test for barrier_block in ep/include/ep_utils.cuh
// (uccl-project/uccl#986). It includes the real header and calls the real
// barrier_block<kNumRanks>, instantiated for the participating rank count.
//
// Each rank owns a signal block and every rank holds a pointer table to all of
// them, as Buffer lays them out in uccl_ep.cc. In every round a rank publishes
// its arrival, enters the barrier with 128 threads (the notify_dispatch block
// size) and then checks from every warp that all peers have arrived at this
// round. One rank is delayed on some rounds, so a release before a peer arrives
// or a warp that leaves before the block is released is counted. When all ranks
// have finished, every signal cell must be back to zero.
//
// A single-epoch run always comes first and must pass. The requested epochs
// then share the same signal array, like two microbatches sharing a Buffer:
//   serial   all epochs on one stream per rank, which is how uccl_ep.cc
//            launches every barrier (on comm_stream); must not release early
//   overlap  one stream per epoch running concurrently; reports how often a
//            round was released by another epoch's arrival
//
// Usage (from ep/tests, after `make barrier_concurrent`):
//   ./barrier_concurrent local <nranks> <iters> <epochs> <serial|overlap>
//       [timeout_s]
//       every rank is a block on the current GPU (protocol check, no IPC)
//   ./barrier_concurrent ipc <rank> <nranks> <iters> <epochs>
//       <serial|overlap> <name> [timeout_s]
//       one process per rank on visible GPU <rank> % <GPU count>, signal
//       blocks shared over CUDA IPC; all ranks pass the same unused <name>:
//   ./barrier_concurrent ipc 0 2 400 2 serial run1 &
//   ./barrier_concurrent ipc 1 2 400 2 serial run1
#include "ep_utils.cuh"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(call)                                     \
  do {                                                  \
    cudaError_t e = (call);                             \
    if (e != cudaSuccess) {                             \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                   \
      exit(1);                                          \
    }                                                   \
  } while (0)

constexpr int kMaxRanks = 8;
constexpr int kMaxEpochs = 4;
// Arrival slot of the single-epoch run; the requested epochs use 0..epochs-1.
constexpr int kSingleSlot = kMaxEpochs;
constexpr int kNumThreads = 128;
constexpr long long kDelayCycles = 200000;

struct alignas(16) RankBlock {
  int signal[kMaxRanks];
  int arrived[kMaxEpochs + 1];
};

struct Counters {
  unsigned long long calls;
  unsigned long long early;
};

// Rendezvous for ipc mode, in a file under /dev/shm shared by all ranks.
struct Rendezvous {
  std::atomic<int> published;
  std::atomic<int> finished;
  std::atomic<int> left;
  cudaIpcMemHandle_t handles[kMaxRanks];
};

template <int kNumRanks>
__global__ void barrier_rounds(int** signal_ptrs, RankBlock** blocks,
                               int first_rank, int slot_begin, int num_slots,
                               int iters, Counters* counters) {
  int const rank = first_rank + static_cast<int>(blockIdx.x);
  int const lane = static_cast<int>(threadIdx.x) % WARP_SIZE;

  for (int it = 0; it < iters; ++it) {
    for (int slot = slot_begin; slot < slot_begin + num_slots; ++slot) {
      int const round = it + 1;
      if (threadIdx.x == 0) {
        if (it % 4 == 1 and rank == (it / 4 + slot) % kNumRanks) {
          auto const start = clock64();
          while (clock64() - start < kDelayCycles)
            ;
        }
        st_release_sys_global(blocks[rank]->arrived + slot, round);
      }
      __syncthreads();

      barrier_block<kNumRanks>(signal_ptrs, rank);

      // Every warp checks on its own, so a warp that returns before the block
      // is released is caught as well. The fence keeps the arrival loads after
      // the barrier's own loads.
      bool early = false;
      if (lane == 0) {
        memory_fence();
        for (int peer = 0; peer < kNumRanks; ++peer)
          early |= ld_acquire_sys_global(blocks[peer]->arrived + slot) < round;
      }
      // One count per call, whichever warps saw it.
      if (__syncthreads_or(early) and threadIdx.x == 0)
        atomicAdd(&counters->early, 1ull);
      if (threadIdx.x == 0) atomicAdd(&counters->calls, 1ull);
    }
  }
}

struct Setup {
  char tag[16];
  bool local;
  int rank, num_ranks, iters, epochs;
  bool overlap;
  int** signal_ptrs;
  RankBlock** blocks;
  RankBlock* own;  // all blocks in local mode, this rank's block in ipc mode
  cudaStream_t streams[kMaxEpochs];
};

// Launches one phase and returns its counters. `per_stream` gives every slot
// its own stream; otherwise all slots run in order in one kernel.
template <int kNumRanks>
Counters run_phase(Setup const& s, int slot_begin, int num_slots,
                   bool per_stream) {
  Counters* counters;
  CHECK(cudaMalloc(&counters, sizeof(Counters)));
  CHECK(cudaMemset(counters, 0, sizeof(Counters)));
  int const grid = s.local ? s.num_ranks : 1;
  int const first_rank = s.local ? 0 : s.rank;
  if (per_stream) {
    for (int i = 0; i < num_slots; ++i)
      barrier_rounds<kNumRanks><<<grid, kNumThreads, 0, s.streams[i]>>>(
          s.signal_ptrs, s.blocks, first_rank, slot_begin + i, 1, s.iters,
          counters);
  } else {
    barrier_rounds<kNumRanks><<<grid, kNumThreads, 0, s.streams[0]>>>(
        s.signal_ptrs, s.blocks, first_rank, slot_begin, num_slots, s.iters,
        counters);
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  Counters result;
  CHECK(cudaMemcpy(&result, counters, sizeof(result), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(counters));
  return result;
}

// Number of nonzero cells in the signal blocks this process owns.
int count_dirty_cells(Setup const& s) {
  int const num_blocks = s.local ? s.num_ranks : 1;
  std::vector<RankBlock> host(num_blocks);
  CHECK(cudaMemcpy(host.data(), s.own, sizeof(RankBlock) * num_blocks,
                   cudaMemcpyDeviceToHost));
  int dirty = 0;
  for (auto const& block : host)
    for (int i = 0; i < s.num_ranks; ++i) dirty += block.signal[i] != 0;
  return dirty;
}

template <int kNumRanks>
bool run(Setup const& s) {
  int const blocks_per_launch = s.local ? s.num_ranks : 1;
  unsigned long long const single_calls = 1ull * blocks_per_launch * s.iters;
  unsigned long long const epoch_calls = single_calls * s.epochs;

  auto const single = run_phase<kNumRanks>(s, kSingleSlot, 1, false);
  bool const single_ok = single.calls == single_calls and single.early == 0;
  printf("%s single epoch: calls=%llu/%llu early_releases=%llu -> %s\n", s.tag,
         single.calls, single_calls, single.early, single_ok ? "PASS" : "FAIL");
  if (not single_ok) return false;

  auto const multi = run_phase<kNumRanks>(s, 0, s.epochs, s.overlap);
  bool multi_ok = multi.calls == epoch_calls;
  if (s.overlap) {
    printf(
        "%s %d overlapping epochs: calls=%llu/%llu "
        "released_by_another_epoch=%llu\n",
        s.tag, s.epochs, multi.calls, epoch_calls, multi.early);
  } else {
    multi_ok = multi_ok and multi.early == 0;
    printf(
        "%s %d serial epochs: calls=%llu/%llu early_releases=%llu -> "
        "%s\n",
        s.tag, s.epochs, multi.calls, epoch_calls, multi.early,
        multi_ok ? "PASS" : "FAIL");
  }
  return multi_ok;
}

bool dispatch(Setup const& s) {
  // Same rank counts as SWITCH_RANKS in ep_launch.cuh.
  switch (s.num_ranks) {
    case 2:
      return run<2>(s);
    case 4:
      return run<4>(s);
    case 8:
      return run<8>(s);
    default:
      fprintf(stderr, "nranks must be 2, 4 or 8\n");
      return false;
  }
}

void upload_tables(Setup& s, std::vector<RankBlock*> const& blocks) {
  std::vector<int*> signal_ptrs(kMaxRanks, nullptr);
  for (int i = 0; i < s.num_ranks; ++i) signal_ptrs[i] = blocks[i]->signal;
  CHECK(cudaMalloc(&s.signal_ptrs, sizeof(int*) * kMaxRanks));
  CHECK(cudaMalloc(&s.blocks, sizeof(RankBlock*) * kMaxRanks));
  CHECK(cudaMemcpy(s.signal_ptrs, signal_ptrs.data(), sizeof(int*) * kMaxRanks,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(s.blocks, blocks.data(), sizeof(RankBlock*) * kMaxRanks,
                   cudaMemcpyHostToDevice));
}

void wait_for(std::atomic<int> const& counter, int target) {
  while (counter.load(std::memory_order_acquire) < target) usleep(1000);
}

int usage(char const* argv0) {
  fprintf(stderr,
          "usage: %s local <nranks> <iters> <epochs> <serial|overlap> "
          "[timeout_s]\n"
          "       %s ipc <rank> <nranks> <iters> <epochs> <serial|overlap> "
          "<name> [timeout_s]\n",
          argv0, argv0);
  return 2;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage(argv[0]);
  Setup s{};
  s.local = strcmp(argv[1], "local") == 0;
  if (not s.local and strcmp(argv[1], "ipc") != 0) return usage(argv[0]);
  if (argc < (s.local ? 6 : 8)) return usage(argv[0]);
  int arg = 2;
  s.rank = s.local ? 0 : atoi(argv[arg++]);
  s.num_ranks = atoi(argv[arg++]);
  s.iters = atoi(argv[arg++]);
  s.epochs = atoi(argv[arg++]);
  std::string const mode = argv[arg++];
  s.overlap = mode == "overlap";
  std::string const name = s.local ? "" : argv[arg++];
  int const timeout_s = argc > arg ? atoi(argv[arg]) : 120;
  if (s.num_ranks < 2 or s.num_ranks > kMaxRanks or s.rank < 0 or
      s.rank >= s.num_ranks or s.iters < 1 or s.epochs < 1 or
      s.epochs > kMaxEpochs or (not s.overlap and mode != "serial"))
    return usage(argv[0]);
  // A hang ends the process with SIGALRM instead of waiting for the
  // barrier's own device timeout.
  alarm(timeout_s);

  int num_devices = 0;
  CHECK(cudaGetDeviceCount(&num_devices));
  int const device = s.local ? 0 : s.rank % num_devices;
  CHECK(cudaSetDevice(device));
  for (int i = 0; i < kMaxEpochs; ++i)
    CHECK(cudaStreamCreateWithFlags(&s.streams[i], cudaStreamNonBlocking));
  if (s.local)
    snprintf(s.tag, sizeof(s.tag), "[local]");
  else
    snprintf(s.tag, sizeof(s.tag), "[rank %d]", s.rank);
  printf("%s gpu=%d nranks=%d iters=%d epochs=%d %s\n", s.tag, device,
         s.num_ranks, s.iters, s.epochs, mode.c_str());
  fflush(stdout);

  bool ok;
  if (s.local) {
    CHECK(cudaMalloc(&s.own, sizeof(RankBlock) * s.num_ranks));
    CHECK(cudaMemset(s.own, 0, sizeof(RankBlock) * s.num_ranks));
    std::vector<RankBlock*> blocks(kMaxRanks, nullptr);
    for (int i = 0; i < s.num_ranks; ++i) blocks[i] = s.own + i;
    upload_tables(s, blocks);
    ok = dispatch(s);
  } else {
    std::string const path = "/dev/shm/uccl_barrier_concurrent_" + name;
    int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0 or ftruncate(fd, sizeof(Rendezvous)) != 0) {
      perror(path.c_str());
      return 1;
    }
    auto* rv = static_cast<Rendezvous*>(mmap(nullptr, sizeof(Rendezvous),
                                             PROT_READ | PROT_WRITE, MAP_SHARED,
                                             fd, 0));
    close(fd);
    if (rv == MAP_FAILED) {
      perror("mmap");
      return 1;
    }
    if (rv->published.load() >= s.num_ranks or rv->left.load() > 0) {
      fprintf(stderr, "%s is left over from an earlier run; use a new name\n",
              path.c_str());
      return 1;
    }

    CHECK(cudaMalloc(&s.own, sizeof(RankBlock)));
    CHECK(cudaMemset(s.own, 0, sizeof(RankBlock)));
    CHECK(cudaIpcGetMemHandle(&rv->handles[s.rank], s.own));
    rv->published.fetch_add(1, std::memory_order_release);
    wait_for(rv->published, s.num_ranks);

    std::vector<RankBlock*> blocks(kMaxRanks, nullptr);
    for (int i = 0; i < s.num_ranks; ++i) {
      if (i == s.rank) {
        blocks[i] = s.own;
        continue;
      }
      CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&blocks[i]),
                                 rv->handles[i],
                                 cudaIpcMemLazyEnablePeerAccess));
    }
    upload_tables(s, blocks);
    ok = dispatch(s);

    // Peers may still add to or subtract from our cells until they are done.
    rv->finished.fetch_add(1, std::memory_order_acq_rel);
    wait_for(rv->finished, s.num_ranks);
    if (rv->left.fetch_add(1) == s.num_ranks - 1) unlink(path.c_str());
  }

  int const dirty = count_dirty_cells(s);
  printf("%s nonzero signal cells after the run: %d\n", s.tag, dirty);
  ok = ok and dirty == 0;
  printf("%s %s\n", s.tag, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

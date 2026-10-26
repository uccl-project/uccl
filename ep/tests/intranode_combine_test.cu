// Build with `make intranode_combine_test` and run on peer-connected GPUs.
#include "intranode.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_bf16.h>

#define CHECK(call)                                     \
  do {                                                  \
    cudaError_t e = (call);                             \
    if (e != cudaSuccess) {                             \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                   \
      exit(1);                                          \
    }                                                   \
  } while (0)

template <class T>
T* allocate(size_t n) {
  T* ptr;
  CHECK(cudaMalloc(&ptr, std::max(size_t(1), n) * sizeof(T)));
  return ptr;
}
template <class T>
T* upload(std::vector<T> const& data) {
  auto ptr = allocate<T>(data.size());
  if (data.size())
    CHECK(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T),
                     cudaMemcpyHostToDevice));
  return ptr;
}
struct RankState {
  cudaStream_t stream;
  void* buffer;
  int* signal;
  void** buffers;
  int** signals;
  std::vector<void*> owned;
  int *counts, *expert_counts, *recv_count, *recv_experts;
  int *channel_prefix, *rank_prefix, *recv_offset, *send_head, *src_idx;
  unsigned char* mask;
  __nv_bfloat16 *input, *dispatched, *output;
  int tokens, received;
  template <class T>
  T* alloc(size_t n) {
    auto p = allocate<T>(n);
    owned.push_back(p);
    return p;
  }
  template <class T>
  T* put(std::vector<T> const& v) {
    auto p = upload(v);
    owned.push_back(p);
    return p;
  }
};

void synchronize_ranks(std::vector<RankState>& ranks) {
  for (int r = 0; r < ranks.size(); ++r) {
    CHECK(cudaSetDevice(r));
    CHECK(cudaStreamSynchronize(ranks[r].stream));
  }
}

// Patterns cover dense routing, 37-token runs, mixed routing, and sparse
// traffic to rank 0. The sparse case also leaves the final source rank empty.
bool route(int source, int token, int dest, int ranks, int pattern) {
  if (pattern == 0) return true;
  if (pattern == 1) return dest == (source + token / 37) % ranks;
  if (pattern == 2) return (token * 13 + source * 7 + dest * 3) % 8 < 3;
  return dest == 0 && (token % 97 == 0 || token % 97 == 96);
}

void run(int nranks, int tokens, int hidden, int channels, int pattern,
         bool timing) {
  std::vector<RankState> ranks(nranks);
  std::vector<void*> buffers(nranks);
  std::vector<int*> signals(nranks);
  std::vector<std::vector<__nv_bfloat16>> expected(nranks);
  int sms = channels * 2, queue = 256, chunk = 6;
  size_t buffer_bytes =
      size_t(nranks) * nranks * sizeof(int) +
      size_t(channels) * nranks *
          (4 * sizeof(int) +
           queue * (hidden * sizeof(__nv_bfloat16) + sizeof(int)));
  for (int r = 0; r < nranks; ++r) {
    CHECK(cudaSetDevice(r));
    auto& state = ranks[r];
    CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
    state.buffer = state.alloc<unsigned char>(buffer_bytes);
    buffers[r] = state.buffer;
    state.signal = state.alloc<int>(nranks);
    signals[r] = state.signal;
    CHECK(cudaMemset(state.signal, 0, nranks * sizeof(int)));
    state.tokens = pattern == 3 && r == nranks - 1 ? 0 : tokens;
    std::vector<int> counts(nranks, 0);
    std::vector<unsigned char> mask(size_t(state.tokens) * nranks);
    expected[r].resize(size_t(state.tokens) * hidden);
    std::vector<__nv_bfloat16> x(expected[r].size());
    for (int t = 0; t < state.tokens; ++t) {
      int copies = 0;
      for (int dest = 0; dest < nranks; ++dest) {
        bool live = route(r, t, dest, nranks, pattern);
        mask[t * nranks + dest] = live;
        counts[dest] += live;
        copies += live;
      }
      for (int h = 0; h < hidden; ++h) {
        float v = float((t * 3 + h + r * 7) % 17 - 8) / 8;
        x[size_t(t) * hidden + h] = __float2bfloat16(v);
        float combined = 0.0f;
        for (int i = 0; i < copies; ++i) combined += v;
        expected[r][size_t(t) * hidden + h] = __float2bfloat16(combined);
      }
    }
    state.counts = state.put(counts);
    state.expert_counts = state.put(counts);
    state.mask = state.put(mask);
    state.input = state.put(x);
    state.channel_prefix = state.alloc<int>(nranks * channels);
    state.rank_prefix = state.alloc<int>(nranks * nranks);
    state.recv_offset = state.alloc<int>(nranks * channels);
    state.send_head = state.alloc<int>(size_t(state.tokens) * nranks);
    state.output = state.alloc<__nv_bfloat16>(size_t(state.tokens) * hidden);
    CHECK(cudaHostAlloc(&state.recv_count, sizeof(int), cudaHostAllocMapped));
    *state.recv_count = -1;
    CHECK(cudaHostAlloc(&state.recv_experts, sizeof(int), cudaHostAllocMapped));
    *state.recv_experts = -1;
  }
  for (int r = 0; r < nranks; ++r) {
    CHECK(cudaSetDevice(r));
    auto& state = ranks[r];
    state.buffers = state.put(buffers);
    state.signals = state.put(signals);
  }
  for (int r = 0; r < nranks; ++r) {
    CHECK(cudaSetDevice(r));
    auto& state = ranks[r];
    uccl::intranode::notify_dispatch(
        state.counts, state.recv_count, nranks, state.expert_counts,
        state.recv_experts, nranks, state.tokens,
        reinterpret_cast<bool*>(state.mask), state.channel_prefix,
        state.rank_prefix, channels * nranks * 4, 1, state.buffers,
        state.signals, r, state.stream, channels);
  }
  synchronize_ranks(ranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK(cudaSetDevice(r));
    auto& state = ranks[r];
    state.received = *state.recv_count;
    int wanted = 0;
    for (int s = 0; s < nranks; ++s)
      for (int t = 0; t < ranks[s].tokens; ++t)
        wanted += route(s, t, r, nranks, pattern);
    if (state.received != wanted) {
      fprintf(stderr, "receive count mismatch %d %d\n", state.received, wanted);
      exit(2);
    }
    state.dispatched =
        state.alloc<__nv_bfloat16>(size_t(state.received) * hidden);
    state.src_idx = state.alloc<int>(state.received);
  }
  for (int r = 0; r < nranks; ++r) {
    CHECK(cudaSetDevice(r));
    auto& state = ranks[r];
    uccl::intranode::dispatch(
        state.dispatched, nullptr, state.src_idx, nullptr, nullptr,
        state.recv_offset, state.send_head, state.input, nullptr, nullptr,
        nullptr, reinterpret_cast<bool*>(state.mask), state.channel_prefix,
        state.tokens, 0, hidden / 8, 0, nranks, 0, 0, 0, state.buffers, r,
        nranks, state.stream, sms, chunk, queue);
  }
  synchronize_ranks(ranks);
  auto launch = [&](int r) {
    auto& state = ranks[r];
    uccl::intranode::cached_notify_combine(
        state.buffers, state.send_head, channels, state.tokens,
        channels * nranks * 2, state.signals, r, nranks, state.stream);
    uccl::intranode::combine(
        CUDA_R_16BF, state.output, nullptr, state.dispatched, nullptr, nullptr,
        nullptr, state.src_idx, state.rank_prefix, state.recv_offset,
        state.send_head, state.received, state.tokens, hidden, 0, state.buffers,
        r, nranks, state.stream, sms, chunk, queue);
  };
  std::vector<std::vector<int>> expected_heads(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK(cudaSetDevice(r));
    auto& state = ranks[r];
    auto& heads = expected_heads[r];
    heads.resize(size_t(state.tokens) * nranks);
    if (heads.size())
      CHECK(cudaMemcpy(heads.data(), state.send_head,
                       heads.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int channel_size = (state.tokens + channels - 1) / channels;
    for (int channel = 0; channel < channels; ++channel)
      for (int peer = 0; peer < nranks; ++peer) {
        int next = 1 << 25;
        for (int t = std::min(state.tokens, (channel + 1) * channel_size) - 1;
             t >= channel * channel_size; --t) {
          int& head = heads[t * nranks + peer];
          if (head >= 0)
            next = head;
          else
            head = -next - 1;
        }
      }
  }
  // Reusing a dispatch handle preserves both the payload and head encoding.
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (int r = 0; r < nranks; ++r) {
      CHECK(cudaSetDevice(r));
      launch(r);
    }
    synchronize_ranks(ranks);
    for (int r = 0; r < nranks; ++r) {
      CHECK(cudaSetDevice(r));
      auto& state = ranks[r];
      std::vector<__nv_bfloat16> got(expected[r].size());
      if (got.size())
        CHECK(cudaMemcpy(got.data(), state.output,
                         got.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost));
      if (got.size() && std::memcmp(got.data(), expected[r].data(),
                                    got.size() * sizeof(__nv_bfloat16))) {
        fprintf(stderr, "output mismatch R%d N%d C%d P%d rank%d\n", nranks,
                tokens, channels, pattern, r);
        exit(3);
      }
      std::vector<int> heads(expected_heads[r].size());
      if (heads.size())
        CHECK(cudaMemcpy(heads.data(), state.send_head,
                         heads.size() * sizeof(int), cudaMemcpyDeviceToHost));
      if (heads != expected_heads[r]) {
        fprintf(stderr, "head mismatch\n");
        exit(4);
      }
    }
  }
  if (timing) {
    std::vector<cudaGraphExec_t> exec(nranks);
    std::vector<cudaEvent_t> start(nranks), end(nranks);
    int const repeats = 100;
    for (int r = 0; r < nranks; ++r) {
      CHECK(cudaSetDevice(r));
      CHECK(cudaEventCreate(&start[r]));
      CHECK(cudaEventCreate(&end[r]));
      cudaGraph_t graph;
      CHECK(
          cudaStreamBeginCapture(ranks[r].stream, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < repeats; ++i) launch(r);
      CHECK(cudaStreamEndCapture(ranks[r].stream, &graph));
      CHECK(cudaGraphInstantiate(&exec[r], graph, nullptr, nullptr, 0));
      CHECK(cudaGraphDestroy(graph));
    }
    std::vector<float> times;
    for (int round = 0; round < 8; ++round) {
      for (int r = 0; r < nranks; ++r) {
        CHECK(cudaSetDevice(r));
        CHECK(cudaEventRecord(start[r], ranks[r].stream));
        CHECK(cudaGraphLaunch(exec[r], ranks[r].stream));
        CHECK(cudaEventRecord(end[r], ranks[r].stream));
      }
      synchronize_ranks(ranks);
      float elapsed = 0;
      for (int r = 0; r < nranks; ++r) {
        CHECK(cudaSetDevice(r));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start[r], end[r]));
        elapsed = std::max(elapsed, ms * 1000 / repeats);
      }
      if (round > 1) times.push_back(elapsed);
    }
    std::sort(times.begin(), times.end());
    printf("combine R%d N%d H%d C%d P%d: %.4f us\n", nranks, tokens, hidden,
           channels, pattern, times[times.size() / 2]);
    for (int r = 0; r < nranks; ++r) {
      CHECK(cudaSetDevice(r));
      CHECK(cudaGraphExecDestroy(exec[r]));
      CHECK(cudaEventDestroy(start[r]));
      CHECK(cudaEventDestroy(end[r]));
    }
  }
  for (int r = 0; r < nranks; ++r) {
    CHECK(cudaSetDevice(r));
    auto& state = ranks[r];
    for (auto p : state.owned) CHECK(cudaFree(p));
    CHECK(cudaFreeHost(state.recv_count));
    CHECK(cudaFreeHost(state.recv_experts));
    CHECK(cudaStreamDestroy(state.stream));
  }
  printf("PASS R%d N%d H%d C%d P%d\n", nranks, tokens, hidden, channels,
         pattern);
  fflush(stdout);
}

int main(int argc, char** argv) {
  int devices;
  CHECK(cudaGetDeviceCount(&devices));
  int nranks = argc > 1 ? std::atoi(argv[1]) : devices;
  if ((nranks != 2 && nranks != 4 && nranks != 8) || nranks > devices) {
    fprintf(stderr,
            "Usage: intranode_combine_test [2|4|8]; requested GPUs must be "
            "available\n");
    return 5;
  }
  for (int r = 0; r < nranks; ++r) {
    CHECK(cudaSetDevice(r));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, r));
    printf("GPU %d %s\n", r, prop.name);
    for (int p = 0; p < nranks; ++p)
      if (p != r) {
        int access, atomics;
        CHECK(cudaDeviceCanAccessPeer(&access, r, p));
        CHECK(cudaDeviceGetP2PAttribute(
            &atomics, cudaDevP2PAttrNativeAtomicSupported, r, p));
        if (!access || !atomics) {
          fprintf(stderr, "Peer access with native atomics required\n");
          return 6;
        }
        CHECK(cudaDeviceEnablePeerAccess(p, 0));
      }
  }
  for (int n : {0, 1, 31, 32, 33, 63, 64, 65, 127, 129, 1023, 4097})
    for (int channels : {3, 20})
      for (int pattern : {0, 1, 2, 3})
        run(nranks, n, 4096, channels, pattern, false);
  for (int n : {128, 1024, 4096, 16384})
    for (int pattern : {1, 2}) run(nranks, n, 7168, 20, pattern, true);
  puts("PASS intranode combine");
}

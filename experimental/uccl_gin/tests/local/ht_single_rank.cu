#include "device/hybridep_adapter.cuh"
#include "queue_fixture.hpp"
#include <algorithm>
#include <deque>
#include <cuda_bf16.h>

using local_test::require;
namespace ht = nccl_ep::hybridep;
constexpr int kHidden = 1024, kExperts = 32, kTopK = 8;

template <typename T>
class Buffer {
 public:
  T* data = nullptr;
  size_t const size;
  explicit Buffer(size_t size) : size(size) {
    CUDA_CHECK(cudaMalloc(&data, size * sizeof(T)));
    CUDA_CHECK(cudaMemset(data, 0, size * sizeof(T)));
  }
  ~Buffer() { CUDA_CHECK(cudaFree(data)); }
  Buffer(Buffer const&) = delete;
  Buffer& operator=(Buffer const&) = delete;
  void upload(std::vector<T> const& values, cudaStream_t stream) {
    require(values.size() == size, "input shape");
    CUDA_CHECK(cudaMemcpyAsync(data, values.data(), size * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }
  std::vector<T> read() const {
    std::vector<T> values(size);
    CUDA_CHECK(cudaMemcpy(values.data(), data, size * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return values;
  }
};

uint16_t exact_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  require((bits & 0xffffu) == 0, "CPU oracle value is exactly BF16");
  return static_cast<uint16_t>(bits >> 16);
}

__global__ void expert_twice(uint16_t const* input, uint16_t* output,
                             int32_t const* received) {
  int const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < received[0] * kHidden)
    output[index] = __bfloat16_as_ushort(__float2bfloat16_rn(
        2.0f * __bfloat162float(__ushort_as_bfloat16(input[index]))));
}

uint64_t digest(uint64_t state, void const* pointer, size_t bytes) {
  auto const* values = static_cast<uint8_t const*>(pointer);
  for (size_t i = 0; i < bytes; ++i)
    state = (state ^ values[i]) * 1099511628211ull;
  return state;
}

struct BatchResult {
  float gpu_ms;
  uint64_t hash;
  int received;
};

class SingleRank {
 public:
  local_test::QueueFixture queue;
  int const tokens;
  Buffer<uint16_t> input, dispatched, expert, combined;
  Buffer<float> input_prob, dispatched_prob;
  Buffer<uint8_t> routing, rdma_map, rank_mask, local_routing;
  Buffer<int32_t> sparse_map, received, expert_counts;
  Buffer<uint32_t> dispatch_flags, combine_flags, grid_counter;
  Buffer<uint8_t> scan_tmp;
  Buffer<int64_t> counters;
  cudaEvent_t begin, end;
  uint32_t generation = 0;

  explicit SingleRank(int device, int tokens)
      : queue(device, 32, 4096),
        tokens(tokens),
        input(tokens * kHidden),
        dispatched(tokens * kHidden),
        expert(tokens * kHidden),
        combined(tokens * kHidden),
        input_prob(tokens * kExperts),
        dispatched_prob(tokens * kExperts),
        routing(tokens * (kExperts / 8)),
        rdma_map((tokens + 15) / 16 * 16),
        rank_mask(tokens),
        local_routing(tokens * kExperts),
        sparse_map(tokens),
        received(1),
        expert_counts(kExperts),
        dispatch_flags(1),
        combine_flags(1),
        grid_counter(1),
        scan_tmp(ht::get_preprocessing_scan_tmp_size(1)),
        counters(1024) {
    queue.resources.window_base = reinterpret_cast<uint64_t>(input.data);
    queue.resources.window_bytes = input.size * sizeof(uint16_t);
    queue.resources.atomic_tail_base =
        reinterpret_cast<uint64_t>(counters.data);
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
  }

  ~SingleRank() {
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
  }

  BatchResult run(int seed) {
    std::vector<uint16_t> host_input(input.size), expected_dispatch;
    std::vector<uint16_t> expected_combined(combined.size);
    std::vector<float> host_prob(input_prob.size), expected_prob;
    std::vector<uint8_t> host_routing(routing.size), expected_local;
    std::vector<uint8_t> expected_mask(tokens), expected_rdma(rdma_map.size);
    std::vector<int32_t> expected_map(tokens, -1), expected_counts(kExperts);
    int count = 0;
    for (int token = 0; token < tokens; ++token) {
      bool const active = (token + seed) % 7 != 0;
      for (int column = 0; column < kHidden; ++column) {
        float const value = ((token + column + seed) % 64 - 32) / 16.0f;
        host_input[token * kHidden + column] = exact_bf16(value);
        expected_combined[token * kHidden + column] =
            active ? exact_bf16(value * 2) : 0;
      }
      if (!active) continue;
      expected_map[token] = count++;
      expected_mask[token] = expected_rdma[token] = 1;
      for (int k = 0; k < kTopK; ++k) {
        int const expert_id = (token * 5 + k * 3 + seed) % kExperts;
        host_routing[token * (kExperts / 8) + expert_id / 8] |=
            1u << (expert_id % 8);
        host_prob[token * kExperts + expert_id] = (k + 1) / 16.0f;
        ++expected_counts[expert_id];
      }
      for (int e = 0; e < kExperts; ++e) {
        expected_local.push_back(host_prob[token * kExperts + e] != 0);
        expected_prob.push_back(host_prob[token * kExperts + e]);
      }
      expected_dispatch.insert(expected_dispatch.end(),
                               host_input.begin() + token * kHidden,
                               host_input.begin() + (token + 1) * kHidden);
    }
    input.upload(host_input, queue.stream);
    input_prob.upload(host_prob, queue.stream);
    routing.upload(host_routing, queue.stream);
    CUDA_CHECK(cudaMemsetAsync(dispatched.data, 0xa5,
                               dispatched.size * sizeof(uint16_t),
                               queue.stream));
    CUDA_CHECK(cudaMemsetAsync(combined.data, 0xa5,
                               combined.size * sizeof(uint16_t), queue.stream));

    uint16_t* dispatch_outputs[] = {dispatched.data};
    float* probability_outputs[] = {dispatched_prob.data};
    uint16_t* expert_inputs[] = {expert.data};
    ht::DispatchParams dispatch{};
    dispatch.hidden_dim = kHidden;
    dispatch.experts_per_rank = kExperts;
    dispatch.num_ranks_per_node = 1;
    dispatch.attn_input_token = input.data;
    dispatch.attn_input_prob = input_prob.data;
    dispatch.expert_output_token_ptrs =
        reinterpret_cast<void* const*>(dispatch_outputs);
    dispatch.expert_output_prob_ptrs = probability_outputs;
    dispatch.rdma_to_attn_map = reinterpret_cast<bool*>(rdma_map.data);
    dispatch.sparse_to_dense_map = sparse_map.data;
    dispatch.intra_node_write_completion_flags = dispatch_flags.data;
    dispatch.dispatch_grid_barrier_counter = grid_counter.data;
    dispatch.expected_intra_node_flag_value = ++generation;
    dispatch.num_tokens_per_rank = tokens;
    ht::CombineParams combine{};
    combine.hidden_dim = kHidden;
    combine.experts_per_rank = kExperts;
    combine.num_ranks_per_node = 1;
    combine.expert_input_token_ptrs = expert_inputs;
    combine.attn_output_token = combined.data;
    combine.rdma_to_attn_map = reinterpret_cast<bool*>(rdma_map.data);
    combine.sparse_to_dense_map = sparse_map.data;
    combine.combine_intra_node_write_completion_flags = combine_flags.data;
    combine.combine_expected_intra_node_flag_value = generation;
    combine.num_tokens_per_rank = tokens;
    combine.num_recv_tokens = count;
#ifdef NCCL_EP_USE_UCCL_GIN
    dispatch.uccl_resources = combine.uccl_resources = queue.resources;
#endif
    CUDA_CHECK(cudaEventRecord(begin, queue.stream));
    ht::call_metadata_preprocessing(
        routing.data, sparse_map.data, reinterpret_cast<bool*>(rdma_map.data),
        nullptr, rank_mask.data, received.data,
        reinterpret_cast<bool*>(local_routing.data), expert_counts.data,
        scan_tmp.data, 0, 0, tokens, kHidden, 1, 1, kExperts, queue.stream);
    ht::call_dispatch(dispatch, 8192, 1, false, true, queue.stream);
    expert_twice<<<(tokens * kHidden + 255) / 256, 256, 0, queue.stream>>>(
        dispatched.data, expert.data, received.data);
    CUDA_CHECK(cudaGetLastError());
    ht::call_combine(combine, 8192, 1, false, queue.stream);
    CUDA_CHECK(cudaEventRecord(end, queue.stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));

    require(received.read()[0] == count, "received token count");
    require(sparse_map.read() == expected_map, "complete sparse-to-dense map");
    require(expert_counts.read() == expected_counts, "all per-expert counts");
    require(rank_mask.read() == expected_mask, "complete rank mask");
    auto const actual_rdma = rdma_map.read();
    require(std::equal(expected_rdma.begin(), expected_rdma.begin() + tokens,
                       actual_rdma.begin()),
            "routing flags");
    auto const actual_local = local_routing.read();
    require(std::equal(expected_local.begin(), expected_local.end(),
                       actual_local.begin()),
            "all local expert routing");
    auto const actual_dispatch = dispatched.read();
    require(std::equal(expected_dispatch.begin(), expected_dispatch.end(),
                       actual_dispatch.begin()),
            "every dispatched BF16 value");
    auto const actual_prob = dispatched_prob.read();
    require(std::equal(expected_prob.begin(), expected_prob.end(),
                       actual_prob.begin()),
            "every dispatched probability");
    auto const actual_combined = combined.read();
    require(actual_combined == expected_combined,
            "every combined BF16 value including dropped tokens");
    require(dispatch_flags.read()[0] == generation &&
                combine_flags.read()[0] == generation,
            "monotonic dispatch/combine generations");
    require(grid_counter.read()[0] == 0, "dispatch grid counter restored");
    for (auto const& fifo : queue.queues)
      require(fifo->poll().fst == 0,
              "single-rank HT must generate no network commands");
    uint64_t hash = digest(14695981039346656037ull, actual_combined.data(),
                           actual_combined.size() * sizeof(uint16_t));
    hash = digest(hash, expected_counts.data(),
                  expected_counts.size() * sizeof(int32_t));
    return {elapsed, hash, count};
  }
};

int main(int argc, char** argv) {
  int const batch = local_test::integer(argc, argv, "--batch", 64);
  int const concurrency = local_test::integer(argc, argv, "--concurrency", 128);
  int const rounds = local_test::integer(argc, argv, "--rounds", 7);
  int const tail = local_test::integer(argc, argv, "--token-tail", 0);
  require(batch >= 8 && batch <= 64 && concurrency >= 32 &&
              concurrency % batch == 0 && rounds > 2,
          "high-batch/high-concurrency finite matrix");
  require(tail <= 124 && tail % 4 == 0,
          "last request retains an aligned token tail");
  int const tokens = batch * 128 - tail;
  SingleRank test(local_test::integer(argc, argv, "--device", 0), tokens);
#ifdef NCCL_EP_USE_UCCL_GIN
  char const* backend = "uccl";
#else
  char const* backend = "nccl";
#endif
  for (int round = 0; round < rounds; ++round) {
    std::deque<int> pending;
    for (int id = 0; id < concurrency; ++id) pending.push_back(id);
    size_t const peak_pending = pending.size();
    auto const started = std::chrono::steady_clock::now();
    float gpu_ms = 0;
    uint64_t hash = 14695981039346656037ull;
    int actual_batches = 0, received_tokens = 0;
    std::vector<int> completed;
    while (!pending.empty()) {
      auto const result =
          test.run(round * (concurrency / batch) + actual_batches);
      gpu_ms += result.gpu_ms;
      received_tokens += result.received;
      hash = digest(hash, &result.hash, sizeof(result.hash));
      ++actual_batches;
      for (int j = 0; j < batch; ++j) {
        completed.push_back(pending.front());
        pending.pop_front();
      }
    }
    double const wall_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    for (int id = 0; id < concurrency; ++id)
      require(completed[id] == id, "every queued request completed once");
    std::printf(
        "{\"test\":\"native_ht_single_rank\",\"backend\":\"%s\",\"batch\":%d,"
        "\"concurrency\":%d,"
        "\"round\":%d,\"warmup\":%s,\"tokens_per_batch\":%d,\"peak_pending_"
        "requests\":%zu,"
        "\"actual_batches\":%d,\"completed_requests\":%zu,\"received_tokens\":%"
        "d,\"gpu_ms\":%.6f,"
        "\"wave_ms_with_oracle\":%.6f,\"oracle_hash\":\"%016llx\",\"gin_"
        "commands\":0,\"pass\":true}\n",
        backend, batch, concurrency, round, round < 2 ? "true" : "false",
        tokens, peak_pending, actual_batches, completed.size(), received_tokens,
        gpu_ms, wall_ms, static_cast<unsigned long long>(hash));
    std::fflush(stdout);
  }
}

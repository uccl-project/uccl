#include "kernels.h"
#include "numeric.h"
#include "route_plan.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

using namespace uccl::musa_ep;
namespace {
void check(musaError_t error) {
  if (error != musaSuccess)
    throw std::runtime_error("MUSA runtime failure: " +
                             std::to_string(int(error)));
}
void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class T>
struct DeviceArray {
  T* data = nullptr;
  size_t size;
  explicit DeviceArray(size_t n) : size(n) {
    if (n) check(musaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
  }
  ~DeviceArray() {
    if (data) musaFree(data);
  }
  DeviceArray(DeviceArray const&) = delete;
  void upload(std::vector<T> const& values) {
    require(values.size() == size, "test upload shape");
    if (size)
      check(musaMemcpy(data, values.data(), size * sizeof(T),
                       musaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> values(size);
    if (size)
      check(musaMemcpy(values.data(), data, size * sizeof(T),
                       musaMemcpyDeviceToHost));
    return values;
  }
};

void run_case(size_t hidden, size_t topk, unsigned seed, bool unaligned,
              musaStream_t stream) {
  size_t const tokens = seed % 13 == 0 ? 0 : seed % 17 + 1;
  std::mt19937 rng(seed);
  std::vector<int64_t> indices(tokens * topk);
  std::vector<float> weights(tokens * topk);
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = seed % 11 == 0 ? -1 : int(rng() % 5);
    weights[i] = float(int(rng() % 9) - 4) / 4;
  }
  auto plan = make_plan(0, 1, 5, tokens, hidden, topk, indices, weights);
  size_t const rows = plan.routes.size(), offset = unaligned ? 5 : 4;
  constexpr float guard = -12345;
  std::vector<float> input(tokens * hidden);
  for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 19) - 9);
  std::vector<float> guarded(input.size() + 10, guard);
  std::copy(input.begin(), input.end(), guarded.begin() + offset);
  DeviceArray<float> x(guarded.size()), packed(rows * hidden + 10),
      permuted(rows * hidden + 10), output(tokens * hidden + 10);
  x.upload(guarded);
  packed.upload(std::vector<float>(packed.size, guard));
  permuted.upload(std::vector<float>(permuted.size, guard));
  output.upload(std::vector<float>(output.size, guard));
  std::vector<size_t> tokens_map, reverse(rows);
  for (auto const& route : plan.routes) tokens_map.push_back(route.token);
  for (size_t i = 0; i < rows; ++i) reverse[i] = rows - i - 1;
  DeviceArray<size_t> d_tokens(rows), d_reverse(rows), d_lanes(tokens * topk);
  DeviceArray<float> d_weights(tokens * topk);
  d_tokens.upload(tokens_map);
  d_reverse.upload(reverse);
  d_lanes.upload(plan.lane_to_route);
  d_weights.upload(weights);
  check(gather_rows(x.data + offset, packed.data + offset, d_tokens.data, rows,
                    hidden, stream));
  check(musaStreamSynchronize(stream));
  auto values = packed.download();
  auto reference = pack_reference(plan, input);
  auto verify_pack = [&](std::vector<float> const& values) {
    for (size_t i = 0; i < values.size(); ++i) {
      float expected = i >= offset && i < offset + reference.size()
                           ? reference[i - offset]
                           : guard;
      require(std::memcmp(&values[i], &expected, sizeof(float)) == 0,
              "pack/inverse/canary bit mismatch");
    }
  };
  verify_pack(values);
  check(gather_rows(packed.data + offset, permuted.data + offset,
                    d_reverse.data, rows, hidden, stream));
  check(scatter_rows(permuted.data + offset, packed.data + offset,
                     d_reverse.data, rows, hidden, stream));
  check(musaStreamSynchronize(stream));
  verify_pack(packed.download());
  check(weighted_combine(packed.data + offset, output.data + offset,
                         d_lanes.data, d_weights.data, tokens, hidden, topk,
                         stream));
  check(musaStreamSynchronize(stream));
  values = output.download();
  auto exact = combine_reference(plan, reference);
  for (size_t i = 0; i < values.size(); ++i) {
    float expected =
        i >= offset && i < offset + exact.size() ? exact[i - offset] : guard;
    require(values[i] == expected, "combine exact/canary mismatch");
  }
  // Nonbinary weights are checked against a separate FP64 lane oracle.
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = float(int(rng() % 101) - 50) / 37;
  d_weights.upload(weights);
  check(weighted_combine(packed.data + offset, output.data + offset,
                         d_lanes.data, d_weights.data, tokens, hidden, topk,
                         stream));
  check(musaStreamSynchronize(stream));
  values = output.download();
  for (size_t t = 0; t < tokens; ++t)
    for (size_t h = 0; h < hidden; ++h) {
      double expected = 0;
      for (size_t k = 0; k < topk; ++k)
        if (indices[t * topk + k] != -1)
          expected += double(weights[t * topk + k]) * input[t * hidden + h];
      require(test::matches(values[offset + t * hidden + h], expected),
              "combine tolerance mismatch");
    }
}

void benchmark(musaStream_t stream) {
  for (size_t hidden : {128, 1024, 7168}) {
    size_t const tokens = 1024, rows = tokens * 2;
    DeviceArray<float> x(tokens * hidden), packed(rows * hidden),
        y(tokens * hidden);
    DeviceArray<size_t> map(rows), lanes(rows);
    DeviceArray<float> weights(rows);
    std::vector<size_t> h_map(rows), h_lanes(rows);
    for (size_t i = 0; i < rows; ++i) {
      h_map[i] = i / 2;
      h_lanes[i] = i;
    }
    map.upload(h_map);
    lanes.upload(h_lanes);
    weights.upload(std::vector<float>(rows, 0.5f));
    x.upload(std::vector<float>(tokens * hidden, 1));
    musaEvent_t start, end;
    check(musaEventCreate(&start));
    check(musaEventCreate(&end));
    for (int warm = 0; warm < 10; ++warm) {
      check(gather_rows(x.data, packed.data, map.data, rows, hidden, stream));
      check(weighted_combine(packed.data, y.data, lanes.data, weights.data,
                             tokens, hidden, 2, stream));
    }
    for (int operation = 0; operation < 2; ++operation) {
      check(musaEventRecord(start, stream));
      for (int repeat = 0; repeat < 100; ++repeat) {
        if (operation == 0)
          check(
              gather_rows(x.data, packed.data, map.data, rows, hidden, stream));
        else
          check(weighted_combine(packed.data, y.data, lanes.data, weights.data,
                                 tokens, hidden, 2, stream));
      }
      check(musaEventRecord(end, stream));
      check(musaEventSynchronize(end));
      float ms;
      check(musaEventElapsedTime(&ms, start, end));
      std::printf("kernel=%s H=%zu T=%zu K=2 iterations=100 mean_us=%.3f\n",
                  operation == 0 ? "gather" : "combine", hidden, tokens,
                  ms * 10);
    }
    check(musaEventDestroy(start));
    check(musaEventDestroy(end));
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    check(musaSetDevice(0));
    musaStream_t stream;
    check(musaStreamCreate(&stream));
    for (size_t hidden : {1, 3, 4, 7, 128, 8192})
      for (size_t topk : {1, 2, 8})
        for (unsigned seed = 0; seed < 100; ++seed)
          run_case(hidden, topk, seed, seed % 2, stream);
    if (argc == 2 && std::string(argv[1]) == "--bench") benchmark(stream);
    check(musaStreamDestroy(stream));
    std::puts(
        "MUSA EP device kernels: PASS "
        "(exact/tolerance/aligned/unaligned/empty)");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

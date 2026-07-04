#include "algo/chunk_graph.h"
#include "lower.h"
#include "test_config.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace UKernel {
namespace CCL {
namespace {

// ── Layer 1: coll_types ─────────────────────────────────────────────────

void test_scalar_type_sizes() {
  printf("[test] scalar type sizes...\n");
  assert(scalar_type_size(ScalarType::UInt8) == 1);
  assert(scalar_type_size(ScalarType::Int8) == 1);
  assert(scalar_type_size(ScalarType::Bool) == 1);
  assert(scalar_type_size(ScalarType::Int16) == 2);
  assert(scalar_type_size(ScalarType::Float16) == 2);
  assert(scalar_type_size(ScalarType::BFloat16) == 2);
  assert(scalar_type_size(ScalarType::Int32) == 4);
  assert(scalar_type_size(ScalarType::Float32) == 4);
  assert(scalar_type_size(ScalarType::Int64) == 8);
  assert(scalar_type_size(ScalarType::Float64) == 8);
}

void test_enum_distinct_values() {
  printf("[test] enum distinct values...\n");
  assert(static_cast<uint32_t>(CollKind::AllReduceRing) !=
         static_cast<uint32_t>(CollKind::AllToAllPairwise));

  assert(static_cast<uint32_t>(AlgoOpKind::Put) !=
         static_cast<uint32_t>(AlgoOpKind::Recv));
  assert(static_cast<uint32_t>(AlgoOpKind::Put) !=
         static_cast<uint32_t>(AlgoOpKind::RecvReduce));

  assert(static_cast<uint32_t>(ExecOpKind::Put) !=
         static_cast<uint32_t>(ExecOpKind::Reduce));
  assert(static_cast<uint32_t>(ExecOpKind::Put) !=
         static_cast<uint32_t>(ExecOpKind::Signal));
  assert(static_cast<uint32_t>(ExecOpKind::Put) !=
         static_cast<uint32_t>(ExecOpKind::WaitSignal));
  assert(static_cast<uint32_t>(ExecOpKind::Reduce) !=
         static_cast<uint32_t>(ExecOpKind::WaitSignal));

  assert(static_cast<uint32_t>(ReductionKind::None) !=
         static_cast<uint32_t>(ReductionKind::Sum));
}

// ── Layer 2: coll_config ────────────────────────────────────────────────

void test_collective_config_defaults() {
  printf("[test] collective config defaults...\n");
  CollectiveConfig cfg;
  assert(cfg.kind == CollKind::AllReduceRing);
  assert(cfg.nranks == 1);
  assert(cfg.rank == 0);
  assert(cfg.input_bytes == 0);
  assert(cfg.output_bytes == 0);
  assert(cfg.tile_bytes == 0);
  assert(cfg.input_split_bytes.empty());
  assert(cfg.output_split_bytes.empty());
  assert(cfg.kind == CollKind::AllReduceRing);
  assert(cfg.dtype == ScalarType::Float32);
  assert(cfg.reduction == ReductionKind::Sum);
}

void test_collective_config_field_assignment() {
  printf("[test] collective config field assignment...\n");
  CollectiveConfig cfg;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.nranks = 8;
  cfg.rank = 3;
  cfg.input_bytes = 65536;
  cfg.output_bytes = 65536;
  cfg.tile_bytes = 512;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.dtype = ScalarType::Float16;
  cfg.reduction = ReductionKind::Prod;
  cfg.input_split_bytes = {16, 32};
  cfg.output_split_bytes = {16, 32};
  assert(cfg.kind == CollKind::AllToAllPairwise);
  assert(cfg.nranks == 8);
  assert(cfg.rank == 3);
  assert(cfg.input_bytes == 65536);
  assert(cfg.output_bytes == 65536);
  assert(cfg.tile_bytes == 512);
  assert(cfg.kind == CollKind::AllToAllPairwise);
  assert(cfg.dtype == ScalarType::Float16);
  assert(cfg.reduction == ReductionKind::Prod);
  assert(cfg.input_split_bytes.size() == 2);
  assert(cfg.output_split_bytes.size() == 2);
}

// ── Layer 3: coll_algo ──────────────────────────────────────────────────

void test_chunk_defaults() {
  printf("[test] Chunk defaults...\n");
  Chunk chunk;
  assert(chunk.op == AlgoOpKind::Put);
  assert(chunk.bytes == 0);
  assert(chunk.src_off == 0);
  assert(chunk.dst_off == 0);
  assert(chunk.src_rank == -1);
  assert(chunk.dst_rank == -1);
  assert(chunk.deps.empty());
}

void test_coll_algo_defaults() {
  printf("[test] CollAlgo defaults...\n");
  CollAlgo algo;
  assert(algo.kind == CollKind::AllReduceRing);
  assert(algo.nranks == 1);
  assert(algo.rank == 0);
  assert(algo.input_bytes == 0);
  assert(algo.output_bytes == 0);
  assert(algo.reduction == ReductionKind::None);
  assert(algo.chunks.empty());
}

void test_build_coll_algo_empty_ops_for_zero_data() {
  printf("[test] build_coll_algo rejects zero data...\n");
  CollectiveConfig cfg;
  cfg.nranks = 4;
  cfg.rank = 1;
  cfg.input_bytes = 0;
  cfg.output_bytes = 0;
  cfg.tile_bytes = 512;
  cfg.dtype = ScalarType::Float32;
  bool threw = false;
  try {
    build_coll_algo(cfg, /*inplace=*/false);
  } catch (std::invalid_argument const&) {
    threw = true;
  }
  assert(threw);
}

void test_build_coll_algo_ring_allreduce_basic() {
  printf("[test] build_coll_algo ring allreduce basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.dtype = ScalarType::Float32;
  cfg.reduction = ReductionKind::Sum;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.kind == CollKind::AllReduceRing);
  assert(algo.nranks == 4);
  assert(algo.rank == 1);
  assert(algo.input_bytes == 4096);
  assert(algo.reduction == ReductionKind::Sum);

  // 4 ranks ring allreduce: 2 phases × 3 ring_steps,
  // phase 1: Put + Recv + RecvReduce per step (3 ops × 3 steps = 9)
  // phase 2: Put + Recv per step (2 ops × 3 steps = 6)
  // = 15 abstract ops.
  assert(!algo.chunks.empty());
  assert(algo.chunks.size() == 15);

  // Phase 1 (reduce-scatter) should have Recv and RecvReduce ops.
  bool saw_recv = false;
  bool saw_recv_reduce = false;
  bool saw_send = false;
  for (auto const& chunk : algo.chunks) {
    if (chunk.op == AlgoOpKind::Recv) saw_recv = true;
    if (chunk.op == AlgoOpKind::RecvReduce) saw_recv_reduce = true;
    if (chunk.op == AlgoOpKind::Put) saw_send = true;
  }
  assert(saw_recv);
  assert(saw_recv_reduce);
  assert(saw_send);

  // Dependencies should form a chain across ring steps.
  bool saw_dep = false;
  for (auto const& chunk : algo.chunks)
    if (!chunk.deps.empty()) saw_dep = true;
  assert(saw_dep);
}

void test_build_coll_algo_alltoall_basic() {
  printf("[test] build_coll_algo alltoall basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllToAllPairwise;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.kind == CollKind::AllToAllPairwise);
  assert(!algo.chunks.empty());
  bool saw_recv = false, saw_send = false;
  for (auto const& chunk : algo.chunks) {
    if (chunk.op == AlgoOpKind::Recv) saw_recv = true;
    if (chunk.op == AlgoOpKind::Put) saw_send = true;
  }
  assert(saw_recv);
  assert(saw_send);
}

// ── Layer 4: lower ──────────────────────────────────────────────────

void test_lower_algo_rejects_zero_tile_bytes() {
  printf("[test] lower_algo rejects zero tile_bytes...\n");
  CollAlgo algo;
  algo.chunks.push_back({});
  bool threw = false;
  try {
    lower_algo(algo, 0);
  } catch (std::invalid_argument const&) {
    threw = true;
  }
  assert(threw);
}

void test_lower_algo_ring_basic() {
  printf("[test] tile_and_schedule ring basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, /*tile_bytes=*/512);
  assert(tiled.input_bytes > 0);
  assert(!tiled.ops.empty());
  // Ring allreduce with 4 ranks, 1024-byte shards, 512-byte tiles:
  // Each abstract op → 2 tiles, plus per-tile Signal/WaitSignal.
  assert(tiled.ops.size() > 30);

  // All tiled ops should have bytes <= tile_bytes (signals have 0).
  for (auto const& op : tiled.ops) assert(op.bytes <= 512);

  // Signal and WaitSignal ops should be present with non-zero tags.
  bool saw_signal = false, saw_waitsig = false;
  for (auto const& op : tiled.ops) {
    if (op.kind == ExecOpKind::Signal) saw_signal = true;
    if (op.kind == ExecOpKind::WaitSignal) saw_waitsig = true;
  }
  assert(saw_signal);
  assert(saw_waitsig);
}

void test_lower_algo_alltoall_basic() {
  printf("[test] tile_and_schedule alltoall...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 2048, 256);
  cfg.kind = CollKind::AllToAllPairwise;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, /*tile_bytes=*/256);
  assert(!tiled.ops.empty());
  bool has_signal = false;
  for (auto const& op : tiled.ops)
    if (op.kind == ExecOpKind::Signal) has_signal = true;
  assert(has_signal);
}

void test_full_pipeline_ring_allreduce() {
  printf("[test] full pipeline ring allreduce...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, cfg.tile_bytes);

  // Verify all data ops are valid (signals have zero bytes).
  for (auto const& op : tiled.ops) {
    if (op.kind == ExecOpKind::Signal || op.kind == ExecOpKind::WaitSignal)
      continue;
    assert(op.bytes > 0);
  }
}

void test_full_pipeline_alltoall() {
  printf("[test] full pipeline alltoall...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 2, 8192, 512);
  cfg.kind = CollKind::AllToAllPairwise;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, cfg.tile_bytes);
  assert(!tiled.ops.empty());
}

void bench_lower_algo_large_ring() {
  printf("[bench] tile_and_schedule large ring (8r, 1MB, 64KB tile)...\n");
  CollectiveConfig cfg;
  cfg.nranks = 8;
  cfg.rank = 0;
  cfg.input_bytes = 1 << 20;
  cfg.output_bytes = 1 << 20;
  cfg.tile_bytes = 1 << 16;
  cfg.dtype = ScalarType::Float32;
  cfg.reduction = ReductionKind::Sum;
  cfg.kind = CollKind::AllReduceRing;

  constexpr int kWarmup = 5;
  constexpr int kIters = 200;
  for (int i = 0; i < kWarmup; ++i) build_tiled(cfg, false);

  auto t0 = std::chrono::steady_clock::now();
  size_t total_ops = 0;
  for (int i = 0; i < kIters; ++i) {
    TiledResult r = build_tiled(cfg, false);
    total_ops += r.ops.size();
  }
  auto t1 = std::chrono::steady_clock::now();
  auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  printf("  %zu ops/iter × %d iters in %.1f ms  (%.0f ops/ms)\n",
         total_ops / kIters, kIters, us / 1000.0,
         static_cast<double>(total_ops) / (us / 1000.0));
}

void bench_lower_algo_large_alltoall() {
  printf(
      "[bench] tile_and_schedule large alltoall dma (8r, 1MB, 64KB tile)...\n");
  CollectiveConfig cfg;
  cfg.nranks = 8;
  cfg.rank = 0;
  cfg.input_bytes = 1 << 20;
  cfg.output_bytes = 1 << 20;
  cfg.tile_bytes = 1 << 16;
  cfg.dtype = ScalarType::Float32;
  cfg.kind = CollKind::AllToAllPairwise;

  constexpr int kWarmup = 5;
  constexpr int kIters = 200;
  for (int i = 0; i < kWarmup; ++i) build_tiled(cfg, false);

  auto t0 = std::chrono::steady_clock::now();
  size_t total_ops = 0;
  for (int i = 0; i < kIters; ++i) {
    TiledResult r = build_tiled(cfg, false);
    total_ops += r.ops.size();
  }
  auto t1 = std::chrono::steady_clock::now();
  auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  printf("  %zu ops/iter × %d iters in %.1f ms  (%.0f ops/ms)\n",
         total_ops / kIters, kIters, us / 1000.0,
         static_cast<double>(total_ops) / (us / 1000.0));
}

}  // namespace
}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;

  // Layer 1
  test_scalar_type_sizes();
  test_enum_distinct_values();

  // Layer 2
  test_collective_config_defaults();
  test_collective_config_field_assignment();

  // Layer 3
  test_chunk_defaults();
  test_coll_algo_defaults();
  test_build_coll_algo_empty_ops_for_zero_data();
  test_build_coll_algo_ring_allreduce_basic();
  test_build_coll_algo_alltoall_basic();

  // Layer 4
  test_lower_algo_rejects_zero_tile_bytes();
  test_lower_algo_ring_basic();
  test_lower_algo_alltoall_basic();

  // Integration
  test_full_pipeline_ring_allreduce();
  test_full_pipeline_alltoall();

  // Benchmarks
  bench_lower_algo_large_ring();
  bench_lower_algo_large_alltoall();

  printf("\n=== Module tests PASSED ===\n");
  return 0;
}

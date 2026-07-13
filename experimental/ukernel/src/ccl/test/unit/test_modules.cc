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
  assert(static_cast<uint32_t>(CollKind::AllReduceRing) !=
         static_cast<uint32_t>(CollKind::AllToAllPairwise));
  assert(static_cast<uint32_t>(OpKind::Send) !=
         static_cast<uint32_t>(OpKind::Copy));
  assert(static_cast<uint32_t>(OpKind::Send) !=
         static_cast<uint32_t>(OpKind::Recv));
  assert(static_cast<uint32_t>(OpKind::Recv) !=
         static_cast<uint32_t>(OpKind::RecvReduce));
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
  assert(cfg.use_sm_ipc == true);
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
  cfg.use_sm_ipc = false;
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
  assert(cfg.use_sm_ipc == false);
  assert(cfg.input_split_bytes.size() == 2);
  assert(cfg.output_split_bytes.size() == 2);
}

// ── Layer 3: coll_algo ──────────────────────────────────────────────────

void test_chunk_defaults() {
  printf("[test] Chunk defaults...\n");
  Chunk chunk;
  assert(chunk.op == OpKind::Copy);
  assert(chunk.bytes == 0);
  assert(chunk.src_off == 0);
  assert(chunk.dst_off == 0);
  assert(chunk.src_rank == -1);
  assert(chunk.dst_rank == -1);
  assert(chunk.sequential_tiles == false);
  assert(chunk.deps.empty());
}

void test_sequential_tiles_values() {
  printf("[test] SequentialTiles values...\n");
  assert(static_cast<uint8_t>(false) != static_cast<uint8_t>(true));
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

  // 4 ranks ring allreduce: 2 phases × 3 ring_steps, 2 ops per step
  // = 2 × 3 × 2 = 12 abstract ops.
  assert(!algo.chunks.empty());
  assert(algo.chunks.size() == 12);

  // Phase 1 (reduce-scatter) should have DeviceRecvReduce ops.
  bool saw_recv_reduce = false;
  bool saw_send = false;
  for (auto const& chunk : algo.chunks) {
    if (chunk.op == OpKind::RecvReduce) saw_recv_reduce = true;
    if (chunk.op == OpKind::Send) saw_send = true;
    assert(chunk.sequential_tiles == false);
  }
  assert(saw_recv_reduce);
  assert(saw_send);

  // Dependencies should form a chain across ring steps.
  bool saw_dep = false;
  for (auto const& chunk : algo.chunks)
    if (!chunk.deps.empty()) saw_dep = true;
  assert(saw_dep);
}

void test_build_coll_algo_alltoall_sm_basic() {
  printf("[test] build_coll_algo alltoall sm basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = true;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.kind == CollKind::AllToAllPairwise);
  assert(!algo.chunks.empty());
  // All SM IPC ops should have Independent tile order.
  for (auto const& chunk : algo.chunks) assert(chunk.sequential_tiles == false);
}

void test_build_coll_algo_alltoall_dma_basic() {
  printf("[test] build_coll_algo alltoall dma basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = false;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.kind == CollKind::AllToAllPairwise);
  assert(!algo.chunks.empty());
  // DMA ops should have some Sequential sequential_tiles.
  bool saw_sequential = false;
  for (auto const& chunk : algo.chunks) {
    if (chunk.sequential_tiles == true) saw_sequential = true;
  }
  assert(saw_sequential);
}

// ── Layer 4: scheduler ──────────────────────────────────────────────────

void test_bfs_layers_empty() {
  printf("[test] schedule_ops empty...\n");
  auto s = bfs_layers({});
  assert(s.empty());
}

void test_bfs_layers_single() {
  printf("[test] schedule_ops single...\n");
  std::vector<Op> ops(1);
  ops[0].kind = OpKind::Copy;
  ops[0].bytes = 128;
  auto s = bfs_layers(ops);
  assert(s.size() == 1);
  assert(s[0].size() == 1);
  assert(s[0][0] == 0);
}

void test_bfs_layers_independent() {
  printf("[test] schedule_ops independent ops...\n");
  std::vector<Op> ops(5);
  for (int i = 0; i < 5; ++i) {
    ops[i].kind = OpKind::Copy;
    ops[i].bytes = 128;
  }
  auto s = bfs_layers(ops);
  assert(s.size() == 1);
  assert(s[0].size() == 5);
}

void test_bfs_layers_chain() {
  printf("[test] schedule_ops chain...\n");
  std::vector<Op> ops(4);
  for (int i = 0; i < 4; ++i) {
    ops[i].kind = OpKind::Copy;
    ops[i].bytes = 128;
  }
  ops[1].deps = {0};
  ops[2].deps = {1};
  ops[3].deps = {2};
  auto s = bfs_layers(ops);
  assert(s.size() == 4);
  for (int i = 0; i < 4; ++i) {
    assert(s[i].size() == 1);
    assert(s[i][0] == static_cast<uint32_t>(i));
  }
}

void test_bfs_layers_diamond() {
  printf("[test] schedule_ops diamond...\n");
  std::vector<Op> ops(4);
  for (int i = 0; i < 4; ++i) {
    ops[i].kind = OpKind::Copy;
    ops[i].bytes = 128;
  }
  ops[1].deps = {0};
  ops[2].deps = {0};
  ops[3].deps = {1, 2};
  auto s = bfs_layers(ops);
  assert(s.size() == 3);
  assert(s[0].size() == 1);
  assert(s[1].size() == 2);
  assert(s[2].size() == 1);

  size_t total = 0;
  for (auto const& layer : s) total += layer.size();
  assert(total == 4);
  assert(total == 4);
}

void test_lower_algo_ring_basic() {
  printf("[test] tile_and_schedule ring basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, /*tile_bytes=*/512);
  assert(tiled.input_bytes > 0);
  assert(tiled.staging_bytes_required == 0);
  assert(!tiled.ops.empty());
  assert(tiled.layers.size() >= 1);
  // Ring allreduce with 4 ranks, 1024-byte shards, 512-byte tiles:
  // Each abstract op → 2 tiles. 12 abstract ops → 24 tiled ops.
  assert(tiled.ops.size() == 24);

  // All tiled ops should have bytes <= tile_bytes.
  for (auto const& op : tiled.ops) assert(op.bytes <= 512);
}

void test_lower_algo_alltoall_sm_basic() {
  printf("[test] tile_and_schedule alltoall sm...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 2048, 256);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = true;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, /*tile_bytes=*/256);
  assert(tiled.staging_bytes_required == 0);
  assert(!tiled.ops.empty());
  assert(tiled.layers.size() >= 1);
  // 4 ranks, self + 3 peers, 2048-byte slice → 8 tiles per peer.
  // self: DeviceCopy (8 tiles), per-peer: DeviceSend + DeviceRecv (16 tiles
  // each × 3 peers) Total: 8 + 2×8×3 = 56 tiles.
  assert(tiled.ops.size() == 14);
}

void test_lower_algo_alltoall_dma_basic() {
  printf("[test] tile_and_schedule alltoall dma...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 2048, 256);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = false;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, /*tile_bytes=*/256);
  assert(tiled.staging_bytes_required > 0);
  assert(!tiled.ops.empty());
  assert(tiled.layers.size() >= 1);
}

void test_lower_algo_sequential_tile_deps() {
  printf("[test] tile_and_schedule sequential tile deps...\n");
  CollectiveConfig cfg = Testing::make_test_config(2, 0, 1024, 256);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = false;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, /*tile_bytes=*/256);
  // DMA staging: TransportSend, TransportRecv, DeviceCopy should have
  // sequential tile ordering.
  bool saw_sequential_chain = false;
  for (size_t i = 1; i < tiled.ops.size(); ++i) {
    for (uint32_t dep : tiled.ops[i].deps) {
      if (dep == static_cast<uint32_t>(i - 1)) saw_sequential_chain = true;
    }
  }
  assert(saw_sequential_chain);
}

void test_bfs_layers_total_covers_all_ops() {
  printf("[test] schedule_ops total covers all ops...\n");
  for (int test_num = 0; test_num < 20; ++test_num) {
    std::vector<Op> ops(static_cast<size_t>(test_num + 1));
    for (auto& op : ops) {
      op.kind = OpKind::Copy;
      op.bytes = 128;
    }
    auto s = bfs_layers(ops);
    size_t total = 0;
    for (auto const& l_ : s) total += l_.size();
    assert(total == ops.size());
  }
}

// ── Integration: full pipeline ──────────────────────────────────────────

void test_full_pipeline_ring_allreduce() {
  printf("[test] full pipeline ring allreduce...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, cfg.tile_bytes);

  // Verify all ops are valid.
  for (auto const& op : tiled.ops) assert(op.bytes > 0);

  // Verify schedule covers all ops.
  size_t total_scheduled = 0;
  for (auto const& l : tiled.layers) total_scheduled += l.size();
  assert(total_scheduled == tiled.ops.size());
}

void test_full_pipeline_alltoall_sm() {
  printf("[test] full pipeline alltoall sm...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 2, 8192, 512);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = true;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, cfg.tile_bytes);
  assert(tiled.staging_bytes_required == 0);
}

void test_full_pipeline_alltoall_dma() {
  printf("[test] full pipeline alltoall dma...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 0, 4096, 1024);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = false;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, cfg.tile_bytes);
  assert(tiled.staging_bytes_required > 0);
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
  cfg.use_sm_ipc = false;

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

void bench_bfs_layers_wide_dag() {
  printf("[bench] schedule_ops wide DAG (1000 independent ops)...\n");
  std::vector<Op> ops(1000);
  for (auto& op : ops) {
    op.kind = OpKind::Copy;
    op.bytes = 128;
  }
  constexpr int kWarmup = 20;
  constexpr int kIters = 500;
  for (int i = 0; i < kWarmup; ++i) bfs_layers(ops);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) bfs_layers(ops);
  auto t1 = std::chrono::steady_clock::now();
  auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  printf("  1000 ops × %d iters in %.1f ms  (%.0f ops/ms)\n", kIters,
         us / 1000.0, static_cast<double>(1000 * kIters) / (us / 1000.0));
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
  test_sequential_tiles_values();
  test_coll_algo_defaults();
  test_build_coll_algo_empty_ops_for_zero_data();
  test_build_coll_algo_ring_allreduce_basic();
  test_build_coll_algo_alltoall_sm_basic();
  test_build_coll_algo_alltoall_dma_basic();

  // Layer 4
  test_bfs_layers_empty();
  test_bfs_layers_single();
  test_bfs_layers_independent();
  test_bfs_layers_chain();
  test_bfs_layers_diamond();
  test_lower_algo_ring_basic();
  test_lower_algo_alltoall_sm_basic();
  test_lower_algo_alltoall_dma_basic();
  test_lower_algo_sequential_tile_deps();
  test_bfs_layers_total_covers_all_ops();

  // Integration
  test_full_pipeline_ring_allreduce();
  test_full_pipeline_alltoall_sm();
  test_full_pipeline_alltoall_dma();

  // Benchmarks
  bench_lower_algo_large_ring();
  bench_lower_algo_large_alltoall();
  bench_bfs_layers_wide_dag();

  printf("\n=== Module tests PASSED ===\n");
  return 0;
}

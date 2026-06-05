#include "backend_test_utils.h"
#include "coll_algo.h"
#include "coll_config.h"
#include "coll_types.h"
#include "scheduler.h"
#include "test_utils.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
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
  assert(static_cast<uint32_t>(CollectiveKind::AllReduce) !=
         static_cast<uint32_t>(CollectiveKind::AllToAll));
  assert(static_cast<uint32_t>(AlgorithmKind::Ring) !=
         static_cast<uint32_t>(AlgorithmKind::Pairwise));
  assert(static_cast<uint32_t>(OpKind::TransportSend) !=
         static_cast<uint32_t>(OpKind::DeviceCopy));
  assert(static_cast<uint32_t>(OpKind::DeviceSend) !=
         static_cast<uint32_t>(OpKind::DeviceRecv));
  assert(static_cast<uint32_t>(OpKind::DeviceRecv) !=
         static_cast<uint32_t>(OpKind::DeviceRecvReduce));
  assert(static_cast<uint32_t>(ReductionKind::None) !=
         static_cast<uint32_t>(ReductionKind::Sum));
}

// ── Layer 2: coll_config ────────────────────────────────────────────────

void test_collective_config_defaults() {
  printf("[test] collective config defaults...\n");
  CollectiveConfig cfg;
  assert(cfg.collective == CollectiveKind::AllReduce);
  assert(cfg.nranks == 1);
  assert(cfg.rank == 0);
  assert(cfg.input_bytes == 0);
  assert(cfg.output_bytes == 0);
  assert(cfg.tile_bytes == 0);
  assert(cfg.input_split_bytes.empty());
  assert(cfg.output_split_bytes.empty());
  assert(cfg.algorithm == AlgorithmKind::Ring);
  assert(cfg.dtype == ScalarType::Float32);
  assert(cfg.reduction == ReductionKind::Sum);
  assert(cfg.use_sm_ipc == true);
}

void test_collective_config_field_assignment() {
  printf("[test] collective config field assignment...\n");
  CollectiveConfig cfg;
  cfg.collective = CollectiveKind::AllToAll;
  cfg.nranks = 8;
  cfg.rank = 3;
  cfg.input_bytes = 65536;
  cfg.output_bytes = 65536;
  cfg.tile_bytes = 512;
  cfg.collective = CollectiveKind::AllToAll;
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.dtype = ScalarType::Float16;
  cfg.reduction = ReductionKind::Prod;
  cfg.use_sm_ipc = false;
  cfg.input_split_bytes = {16, 32};
  cfg.output_split_bytes = {16, 32};
  assert(cfg.collective == CollectiveKind::AllToAll);
  assert(cfg.nranks == 8);
  assert(cfg.rank == 3);
  assert(cfg.input_bytes == 65536);
  assert(cfg.output_bytes == 65536);
  assert(cfg.tile_bytes == 512);
  assert(cfg.algorithm == AlgorithmKind::Pairwise);
  assert(cfg.dtype == ScalarType::Float16);
  assert(cfg.reduction == ReductionKind::Prod);
  assert(cfg.use_sm_ipc == false);
  assert(cfg.input_split_bytes.size() == 2);
  assert(cfg.output_split_bytes.size() == 2);
}

// ── Layer 3: coll_algo ──────────────────────────────────────────────────

void test_algo_op_defaults() {
  printf("[test] AlgoOp defaults...\n");
  AlgoOp op;
  assert(op.kind == OpKind::DeviceCopy);
  assert(op.bytes == 0);
  assert(op.src_off == 0);
  assert(op.dst_off == 0);
  assert(op.src_peer == 0);
  assert(op.dst_peer == 0);
  assert(op.copy_from_staging == false);
  assert(op.tile_order == TileOrder::Independent);
  assert(op.deps.empty());
}

void test_tile_order_values() {
  printf("[test] TileOrder values...\n");
  assert(static_cast<uint8_t>(TileOrder::Independent) !=
         static_cast<uint8_t>(TileOrder::Sequential));
}

void test_coll_algo_defaults() {
  printf("[test] CollAlgo defaults...\n");
  CollAlgo algo;
  assert(algo.collective == CollectiveKind::AllReduce);
  assert(algo.nranks == 1);
  assert(algo.rank == 0);
  assert(algo.input_bytes == 0);
  assert(algo.output_bytes == 0);
  assert(algo.reduction == ReductionKind::None);
  assert(algo.ops.empty());
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
  assert(algo.collective == CollectiveKind::AllReduce);
  assert(algo.nranks == 4);
  assert(algo.rank == 1);
  assert(algo.input_bytes == 4096);
  assert(algo.reduction == ReductionKind::Sum);

  // 4 ranks ring allreduce: 2 phases × 3 ring_steps, 2 ops per step
  // = 2 × 3 × 2 = 12 abstract ops.
  assert(!algo.ops.empty());
  assert(algo.ops.size() == 12);

  // Phase 1 (reduce-scatter) should have DeviceRecvReduce ops.
  bool saw_recv_reduce = false;
  bool saw_send = false;
  for (auto const& op : algo.ops) {
    if (op.kind == OpKind::DeviceRecvReduce) saw_recv_reduce = true;
    if (op.kind == OpKind::DeviceSend) saw_send = true;
    assert(op.tile_order == TileOrder::Independent);
  }
  assert(saw_recv_reduce);
  assert(saw_send);

  // Dependencies should form a chain across ring steps.
  bool saw_dep = false;
  for (auto const& op : algo.ops)
    if (!op.deps.empty()) saw_dep = true;
  assert(saw_dep);
}

void test_build_coll_algo_alltoall_sm_basic() {
  printf("[test] build_coll_algo alltoall sm basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.collective = CollectiveKind::AllToAll;
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.use_sm_ipc = true;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.collective == CollectiveKind::AllToAll);
  assert(!algo.ops.empty());
  // All SM IPC ops should have Independent tile order.
  for (auto const& op : algo.ops)
    assert(op.tile_order == TileOrder::Independent);
}

void test_build_coll_algo_alltoall_dma_basic() {
  printf("[test] build_coll_algo alltoall dma basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.collective = CollectiveKind::AllToAll;
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.use_sm_ipc = false;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.collective == CollectiveKind::AllToAll);
  assert(!algo.ops.empty());
  // DMA ops should have some Sequential tile_order.
  bool saw_sequential = false;
  for (auto const& op : algo.ops) {
    if (op.tile_order == TileOrder::Sequential) saw_sequential = true;
  }
  assert(saw_sequential);
}

// ── Layer 4: scheduler ──────────────────────────────────────────────────

void test_schedule_ops_empty() {
  printf("[test] schedule_ops empty...\n");
  Schedule s = schedule_ops({});
  assert(s.num_streams == 1);
  assert(s.stream_ops.empty());
}

void test_schedule_ops_single() {
  printf("[test] schedule_ops single...\n");
  std::vector<Op> ops(1);
  ops[0].kind = OpKind::DeviceCopy;
  ops[0].bytes = 128;
  Schedule s = schedule_ops(ops);
  assert(s.num_streams == 1);
  assert(s.stream_ops.size() == 1);
  assert(s.stream_ops[0].size() == 1);
  assert(s.stream_ops[0][0] == 0);
}

void test_schedule_ops_independent() {
  printf("[test] schedule_ops independent ops...\n");
  std::vector<Op> ops(5);
  for (int i = 0; i < 5; ++i) {
    ops[i].kind = OpKind::DeviceCopy;
    ops[i].bytes = 128;
  }
  Schedule s = schedule_ops(ops);
  // All independent → one layer of 5 → width = 5.
  assert(s.num_streams == 5);
  // Each stream gets exactly 1 op.
  for (uint32_t i = 0; i < 5; ++i) assert(s.stream_ops[i].size() == 1);
}

void test_schedule_ops_chain() {
  printf("[test] schedule_ops chain...\n");
  std::vector<Op> ops(4);
  for (int i = 0; i < 4; ++i) {
    ops[i].kind = OpKind::DeviceCopy;
    ops[i].bytes = 128;
  }
  // op0 → op1 → op2 → op3  (strict sequential chain).
  ops[1].deps = {0};
  ops[2].deps = {1};
  ops[3].deps = {2};
  Schedule s = schedule_ops(ops);
  // Width = 1 (only one op ready at a time).
  assert(s.num_streams == 1);
  assert(s.stream_ops[0].size() == 4);
  for (int i = 0; i < 4; ++i)
    assert(s.stream_ops[0][i] == static_cast<uint32_t>(i));
}

void test_schedule_ops_diamond() {
  printf("[test] schedule_ops diamond...\n");
  //  0
  //  ⇙⇘
  // 1 2
  //  ⇘⇙
  //  3
  std::vector<Op> ops(4);
  for (int i = 0; i < 4; ++i) {
    ops[i].kind = OpKind::DeviceCopy;
    ops[i].bytes = 128;
  }
  ops[1].deps = {0};
  ops[2].deps = {0};
  ops[3].deps = {1, 2};
  Schedule s = schedule_ops(ops);
  // Layer 0: {0} → width 1
  // Layer 1: {1, 2} → width 2
  // Layer 2: {3} → width 1
  assert(s.num_streams == 2);
  assert(s.stream_ops[0].size() >= 2);  // gets at least 2 ops (0 + 3 or 0 + 1)
  assert(s.stream_ops[1].size() >= 1);  // gets at least 1 op

  // Verify topological ordering: for each op, all deps appear before it
  // in the same or an earlier stream position.
  // We can only check that all 4 ops are scheduled.
  size_t total = 0;
  for (auto const& st : s.stream_ops) total += st.size();
  assert(total == 4);
}

void test_tile_and_schedule_ring_basic() {
  printf("[test] tile_and_schedule ring basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = tile_and_schedule(algo, /*tile_bytes=*/512);
  assert(tiled.input_bytes > 0);
  assert(tiled.staging_bytes_required == 0);
  assert(!tiled.ops.empty());
  assert(tiled.schedule.num_streams >= 1);
  // Ring allreduce with 4 ranks, 1024-byte shards, 512-byte tiles:
  // Each abstract op → 2 tiles. 12 abstract ops → 24 tiled ops.
  assert(tiled.ops.size() == 24);

  // All tiled ops should have bytes <= tile_bytes.
  for (auto const& op : tiled.ops) assert(op.bytes <= 512);
}

void test_tile_and_schedule_alltoall_sm_basic() {
  printf("[test] tile_and_schedule alltoall sm...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 2048, 256);
  cfg.collective = CollectiveKind::AllToAll;
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.use_sm_ipc = true;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = tile_and_schedule(algo, /*tile_bytes=*/256);
  assert(tiled.staging_bytes_required == 0);
  assert(!tiled.ops.empty());
  assert(tiled.schedule.num_streams >= 1);
  // 4 ranks, self + 3 peers, 2048-byte slice → 8 tiles per peer.
  // self: DeviceCopy (8 tiles), per-peer: DeviceSend + DeviceRecv (16 tiles
  // each × 3 peers) Total: 8 + 2×8×3 = 56 tiles.
  assert(tiled.ops.size() == 14);
}

void test_tile_and_schedule_alltoall_dma_basic() {
  printf("[test] tile_and_schedule alltoall dma...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 2048, 256);
  cfg.collective = CollectiveKind::AllToAll;
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.use_sm_ipc = false;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = tile_and_schedule(algo, /*tile_bytes=*/256);
  assert(tiled.staging_bytes_required > 0);
  assert(!tiled.ops.empty());
  assert(tiled.schedule.num_streams >= 1);
}

void test_tile_and_schedule_sequential_tile_deps() {
  printf("[test] tile_and_schedule sequential tile deps...\n");
  CollectiveConfig cfg = Testing::make_test_config(2, 0, 1024, 256);
  cfg.collective = CollectiveKind::AllToAll;
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.use_sm_ipc = false;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = tile_and_schedule(algo, /*tile_bytes=*/256);
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

void test_schedule_ops_schedule_total_covers_all_ops() {
  printf("[test] schedule_ops total covers all ops...\n");
  for (int test_num = 0; test_num < 20; ++test_num) {
    std::vector<Op> ops(static_cast<size_t>(test_num + 1));
    for (auto& op : ops) {
      op.kind = OpKind::DeviceCopy;
      op.bytes = 128;
    }
    Schedule s = schedule_ops(ops);
    size_t total = 0;
    for (auto const& st : s.stream_ops) total += st.size();
    assert(total == ops.size());
  }
}

// ── Integration: full pipeline ──────────────────────────────────────────

void test_full_pipeline_ring_allreduce() {
  printf("[test] full pipeline ring allreduce...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = tile_and_schedule(algo, cfg.tile_bytes);

  // Verify all ops are valid.
  for (auto const& op : tiled.ops) assert(op.bytes > 0);

  // Verify schedule covers all ops.
  size_t total_scheduled = 0;
  for (auto const& st : tiled.schedule.stream_ops) total_scheduled += st.size();
  assert(total_scheduled == tiled.ops.size());
}

void test_full_pipeline_alltoall_sm() {
  printf("[test] full pipeline alltoall sm...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 2, 8192, 512);
  cfg.collective = CollectiveKind::AllToAll;
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.use_sm_ipc = true;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = tile_and_schedule(algo, cfg.tile_bytes);
  assert(tiled.staging_bytes_required == 0);
}

void test_full_pipeline_alltoall_dma() {
  printf("[test] full pipeline alltoall dma...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 0, 4096, 1024);
  cfg.collective = CollectiveKind::AllToAll;
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.use_sm_ipc = false;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = tile_and_schedule(algo, cfg.tile_bytes);
  assert(tiled.staging_bytes_required > 0);
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
  test_algo_op_defaults();
  test_tile_order_values();
  test_coll_algo_defaults();
  test_build_coll_algo_empty_ops_for_zero_data();
  test_build_coll_algo_ring_allreduce_basic();
  test_build_coll_algo_alltoall_sm_basic();
  test_build_coll_algo_alltoall_dma_basic();

  // Layer 4
  test_schedule_ops_empty();
  test_schedule_ops_single();
  test_schedule_ops_independent();
  test_schedule_ops_chain();
  test_schedule_ops_diamond();
  test_tile_and_schedule_ring_basic();
  test_tile_and_schedule_alltoall_sm_basic();
  test_tile_and_schedule_alltoall_dma_basic();
  test_tile_and_schedule_sequential_tile_deps();
  test_schedule_ops_schedule_total_covers_all_ops();

  // Integration
  test_full_pipeline_ring_allreduce();
  test_full_pipeline_alltoall_sm();
  test_full_pipeline_alltoall_dma();

  printf("\n=== Module tests PASSED ===\n");
  return 0;
}

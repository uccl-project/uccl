#include "coll_algo.h"
#include "lower.h"
#include "test_config.h"
#include "utils.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace UKernel {
namespace CCL {
namespace {

// Layer 1: coll_types

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

// Layer 2: coll_config

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

// Layer 3: coll_algo

void test_chunk_defaults() {
  printf("[test] MacroOp defaults...\n");
  MacroOp chunk;
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

  // Buffer roles are explicit on every data chunk: the RS-phase
  // first-touch Put reads Input, later RS Puts and every AG-phase Put
  // read Output; every RecvReduce is Input -> Output. (Even pair_id =
  // RS phase, odd = AG phase.)
  size_t rs_in = 0, rs_out = 0, ag_puts = 0;
  for (auto const& chunk : algo.chunks) {
    if (chunk.op == AlgoOpKind::RecvReduce) {
      assert(chunk.src.space == BufSpace::Input);
      assert(chunk.dst.space == BufSpace::Output);
    }
    if (chunk.op != AlgoOpKind::Put) continue;
    assert(chunk.dst.space == BufSpace::Output);
    bool phase2 = (chunk.pair_id % 2) == 1;
    if (phase2) {
      assert(chunk.src.space == BufSpace::Output);
      ++ag_puts;
    } else if (chunk.deps.empty()) {
      assert(chunk.src.space == BufSpace::Input);
      ++rs_in;
    } else {
      assert(chunk.src.space == BufSpace::Output);
      ++rs_out;
    }
  }
  assert(rs_in == 1 && rs_out == 2 && ag_puts == 3);
}

void test_build_coll_algo_alltoall_basic() {
  printf("[test] build_coll_algo alltoall basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllToAllPairwise;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/true);
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

void test_build_coll_algo_reduce_scatter_basic() {
  printf("[test] build_coll_algo ring reduce-scatter basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::ReduceScatterRing;
  cfg.output_bytes = 1024;  // this rank's shard of the 4096-byte input
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.kind == CollKind::ReduceScatterRing);

  // 4 ranks: 3 ring steps x (Put + Recv + RecvReduce) + final local copy.
  assert(algo.chunks.size() == 10);
  size_t nputs = 0, nrecvs = 0, nreduces = 0;
  for (auto const& chunk : algo.chunks) {
    if (chunk.op == AlgoOpKind::Put) ++nputs;
    if (chunk.op == AlgoOpKind::Recv) ++nrecvs;
    if (chunk.op == AlgoOpKind::RecvReduce) ++nreduces;
    // Peers' puts and all reduces target the full-layout Scratch.
    if (chunk.op == AlgoOpKind::RecvReduce) {
      assert(chunk.src.space == BufSpace::Input);
      assert(chunk.dst.space == BufSpace::Tmp);
    }
  }
  assert(nputs == 4 && nrecvs == 3 && nreduces == 3);

  // The last chunk is the own-shard copy Scratch -> Output[0].
  auto const& tail = algo.chunks.back();
  assert(tail.op == AlgoOpKind::Put);
  assert(tail.pair_id == kNoPairId);
  assert(tail.src.space == BufSpace::Tmp);
  assert(tail.dst.space == BufSpace::Output);
  assert(tail.dst_off == 0);
  assert(tail.bytes == 1024);
  assert(tail.src_off == 1024);  // rank 1's shard offset (uniform shards)
  assert(!tail.deps.empty());

  // Forwarding puts land in the peer's Scratch; the first-touch put of a
  // shard reads Input, later ones read Scratch.
  size_t first_touch = 0, forward = 0;
  for (auto const& chunk : algo.chunks) {
    if (chunk.op != AlgoOpKind::Put || chunk.pair_id == kNoPairId) continue;
    assert(chunk.dst.space == BufSpace::Tmp);
    if (chunk.src.space == BufSpace::Input) {
      assert(chunk.deps.empty());
      ++first_touch;
    } else {
      assert(chunk.src.space == BufSpace::Tmp);
      assert(!chunk.deps.empty());
      ++forward;
    }
  }
  assert(first_touch == 1 && forward == 2);
}

void test_build_coll_algo_allgather_basic() {
  printf("[test] build_coll_algo ring all-gather basic...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllGatherRing;
  cfg.input_bytes = 1024;  // this rank's shard of the 4096-byte output
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.kind == CollKind::AllGatherRing);

  // 1 local copy + 3 ring steps x (Put + Recv).
  assert(algo.chunks.size() == 7);

  // First chunk: own-shard copy Input[0] -> Output[shard_offset].
  auto const& head = algo.chunks.front();
  assert(head.op == AlgoOpKind::Put);
  assert(head.pair_id == kNoPairId);
  assert(head.src.space == BufSpace::Input);
  assert(head.dst.space == BufSpace::Output);
  assert(head.src_off == 0);
  assert(head.dst_off == 1024);  // rank 1's shard offset (uniform shards)
  assert(head.bytes == 1024);

  // Ring puts: the first-step own-shard send reads Input[0] with no
  // deps; later sends forward received shards and depend on their Recv.
  size_t nputs = 0, nrecvs = 0, first_step = 0;
  for (auto const& chunk : algo.chunks) {
    if (chunk.op == AlgoOpKind::Put && chunk.pair_id != kNoPairId) {
      ++nputs;
      if (chunk.src.space == BufSpace::Input) {
        assert(chunk.deps.empty());
        assert(chunk.src_off == 0);
        assert(chunk.dst.space == BufSpace::Output);
        ++first_step;
      } else {
        assert(chunk.src.space == BufSpace::Output);
        assert(chunk.dst.space == BufSpace::Output);
        assert(!chunk.deps.empty());
      }
    }
    if (chunk.op == AlgoOpKind::Recv) ++nrecvs;
  }
  assert(nputs == 3 && nrecvs == 3 && first_step == 1);
}

void test_build_coll_algo_allgather_inplace() {
  printf("[test] build_coll_algo ring all-gather in-place...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllGatherRing;
  cfg.input_bytes = 1024;  // this rank's shard of the 4096-byte output
  cfg.inplace = true;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/true);
  assert(algo.kind == CollKind::AllGatherRing);

  // In-place skips the local publish copy: 3 ring steps x (Put + Recv)
  // only — no pairless local Put.
  assert(algo.chunks.size() == 6);
  size_t nputs = 0, nrecvs = 0;
  for (auto const& chunk : algo.chunks) {
    if (chunk.op == AlgoOpKind::Put && chunk.pair_id != kNoPairId) {
      ++nputs;
      // Own-shard send (step 0) reads Output[offset(rank)] — the shard
      // already sits in the output layout; forwarded shards read Output
      // as in the out-of-place form.
      assert(chunk.src.space == BufSpace::Output);
      assert(chunk.dst.space == BufSpace::Output);
    }
    if (chunk.op == AlgoOpKind::Recv) {
      ++nrecvs;
      assert(chunk.dst.space == BufSpace::Output);
    }
  }
  assert(nputs == 3 && nrecvs == 3);
  // No local copy chunk (pair_id == kNoPairId).
  for (auto const& chunk : algo.chunks)
    assert(!(chunk.op == AlgoOpKind::Put && chunk.pair_id == kNoPairId));
}

void test_verify_algo_pairing_all_kinds() {
  printf("[test] verify_algo_pairing all kinds x nranks...\n");
  auto rank0_shard = [](size_t total_bytes, int nranks) {
    size_t elems = total_bytes / 4;
    return (elems / static_cast<size_t>(nranks) +
            (elems % static_cast<size_t>(nranks) ? 1 : 0)) * 4;
  };
  for (int n : {2, 3, 4, 8}) {
    // AllReduce: both placements, divisible and non-divisible shards.
    for (bool inplace : {false, true}) {
      for (size_t bytes : {4096UL, 4100UL}) {
        CollectiveConfig cfg;
        cfg.kind = CollKind::AllReduceRing;
        cfg.nranks = n;
        cfg.rank = 0;
        cfg.input_bytes = bytes;
        cfg.output_bytes = bytes;
        cfg.tile_bytes = 512;
        cfg.dtype = ScalarType::Float32;
        verify_algo_pairing(cfg, inplace);
        // Binary tree (out-of-place only).
        cfg.kind = CollKind::AllReduceTree;
        verify_algo_pairing(cfg, /*inplace=*/false);
      }
    }
    // AllToAll (inplace only): equal split. NOTE: jointly-consistent
    // variable splits (rank r's slice for p == rank p's slice for r)
    // cannot be expressed with one shared per-rank config — that is a
    // user-level joint contract, and the byte check here is exactly
    // what would flag a violation of it.
    {
      CollectiveConfig cfg;
      cfg.kind = CollKind::AllToAllPairwise;
      cfg.nranks = n;
      cfg.rank = 0;
      cfg.input_bytes = 1024 * static_cast<size_t>(n);
      cfg.output_bytes = cfg.input_bytes;
      cfg.tile_bytes = 512;
      cfg.dtype = ScalarType::Float32;
      verify_algo_pairing(cfg, /*inplace=*/true);
    }
    // ReduceScatter / AllGather: divisible and non-divisible shards.
    for (size_t extra : {0UL, 4UL}) {
      size_t bytes = 1024 * static_cast<size_t>(n) + extra;
      CollectiveConfig rs;
      rs.kind = CollKind::ReduceScatterRing;
      rs.nranks = n;
      rs.rank = 0;
      rs.input_bytes = bytes;
      rs.output_bytes = rank0_shard(bytes, n);
      rs.tile_bytes = 512;
      rs.dtype = ScalarType::Float32;
      verify_algo_pairing(rs, /*inplace=*/false);

      CollectiveConfig ag;
      ag.kind = CollKind::AllGatherRing;
      ag.nranks = n;
      ag.rank = 0;
      ag.input_bytes = rank0_shard(bytes, n);
      ag.output_bytes = bytes;
      ag.tile_bytes = 512;
      ag.dtype = ScalarType::Float32;
      verify_algo_pairing(ag, /*inplace=*/false);
      // In-place AllGather: same pairing invariants (puts still pair
      // 1:1 with recvs landing in Output); the builder just skips the
      // local publish copy and sources the step-0 send from Output.
      CollectiveConfig agi = ag;
      agi.inplace = true;
      verify_algo_pairing(agi, /*inplace=*/true);
      // In-place ReduceScatter: algorithm unchanged (partials in Tmp),
      // pairing invariants hold as out-of-place.
      CollectiveConfig rsi = rs;
      rsi.inplace = true;
      verify_algo_pairing(rsi, /*inplace=*/true);
    }
  }
}

void test_build_coll_algo_tree_basic() {
  printf("[test] build_coll_algo binary-tree allreduce...\n");

  // rank 1 of 4: children {3} (single -> lands Output), parent 0 with
  // two children -> this rank's up-put lands in the parent's Tmp(0).
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllReduceTree;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  assert(algo.kind == CollKind::AllReduceTree);
  // Tmp declared on EVERY rank (rank-symmetric): a rank whose up-put
  // targets a two-children parent's Tmp(0) addresses it with its own
  // scratch id, which must match the parent's — that requires symmetric
  // declaration (see build_allreduce_tree_algo).
  assert(algo.tmp_bytes.size() == 1 && algo.tmp_bytes[0] == 4096);
  // Recv(3), RecvReduce, Put->0, Recv<-0, Put->3.
  assert(algo.chunks.size() == 5);
  auto const& red = algo.chunks[1];
  assert(red.op == AlgoOpKind::RecvReduce);
  assert(red.src.space == BufSpace::Input && red.dst.space == BufSpace::Output);
  auto const& up = algo.chunks[2];
  assert(up.op == AlgoOpKind::Put && up.dst_rank == 0);
  assert(up.pair_id == (1 * 4 + 0) * 2);
  assert(up.src.space == BufSpace::Output);
  assert(up.dst.space == BufSpace::Tmp);  // first child of a 2-child parent
  assert(!up.deps.empty());
  auto const& dn = algo.chunks.back();
  assert(dn.op == AlgoOpKind::Put && dn.dst_rank == 3);
  assert(dn.pair_id == (1 * 4 + 3) * 2 + 1);
  assert(dn.src.space == BufSpace::Output && !dn.deps.empty());

  // Root (rank 0, children {1,2}): first child reduced into Tmp, last
  // into Output; no up-put; down-puts to both children from Output.
  cfg.rank = 0;
  CollAlgo root = build_coll_algo(cfg, /*inplace=*/false);
  assert(root.tmp_bytes.size() == 1 && root.tmp_bytes[0] == 4096);
  assert(root.chunks.size() == 6);
  assert(root.chunks[0].pair_id == (1 * 4 + 0) * 2);
  assert(root.chunks[1].dst.space == BufSpace::Tmp);
  assert(root.chunks[2].pair_id == (2 * 4 + 0) * 2);
  assert(root.chunks[3].dst.space == BufSpace::Output);
  assert(root.chunks[3].src.space == BufSpace::Tmp);
  for (auto const& c : root.chunks)
    if (c.op == AlgoOpKind::Put)
      assert(c.dst_rank == 1 || c.dst_rank == 2);
  assert(root.chunks[4].pair_id == (0 * 4 + 1) * 2 + 1);
  assert(root.chunks[5].pair_id == (0 * 4 + 2) * 2 + 1);

  // Leaf (rank 3, no children): sends Input, receives result in Output.
  cfg.rank = 3;
  CollAlgo leaf = build_coll_algo(cfg, /*inplace=*/false);
  assert(leaf.chunks.size() == 2);
  assert(leaf.chunks[0].op == AlgoOpKind::Put);
  assert(leaf.chunks[0].src.space == BufSpace::Input);
  assert(leaf.chunks[0].dst.space == BufSpace::Output);  // 1-child parent
  assert(leaf.chunks[1].op == AlgoOpKind::Recv);
  assert(leaf.chunks[1].dst.space == BufSpace::Output);

  // In-place is rejected for now.
  bool threw = false;
  try {
    build_coll_algo(cfg, /*inplace=*/true);
  } catch (std::invalid_argument const&) {
    threw = true;
  }
  assert(threw);
}

void test_build_coll_algo_ag_rs_validation() {
  printf("[test] build_coll_algo AG/RS validation...\n");

  // Non-divisible shards: 1025 f32 elements (4100 B) over 4 ranks ->
  // shard elems 257/256/256/256 = 1028/1024/1024/1024 bytes.
  auto make_rs = [](int rank, size_t out_bytes) {
    CollectiveConfig cfg;
    cfg.kind = CollKind::ReduceScatterRing;
    cfg.nranks = 4;
    cfg.rank = rank;
    cfg.input_bytes = 4100;
    cfg.output_bytes = out_bytes;
    cfg.tile_bytes = 512;
    cfg.dtype = ScalarType::Float32;
    return cfg;
  };
  // Rank 0's shard is 1028 B: accepted.
  build_coll_algo(make_rs(0, 1028), /*inplace=*/false);
  // Rank 1's shard is 1024 B: 1028 must be rejected, 1024 accepted.
  bool threw = false;
  try {
    build_coll_algo(make_rs(1, 1028), /*inplace=*/false);
  } catch (std::invalid_argument const&) {
    threw = true;
  }
  assert(threw);
  CollAlgo rs1 = build_coll_algo(make_rs(1, 1024), /*inplace=*/false);
  assert(rs1.chunks.back().dst_off == 0 && rs1.chunks.back().bytes == 1024);

  // Non-divisible offsets: rank 3's shard starts at (3*256 + 1) elems.
  CollAlgo rs3 = build_coll_algo(make_rs(3, 1024), /*inplace=*/false);
  assert(rs3.chunks.back().src_off == (3 * 256 + 1) * 4);

  auto make_ag = [](int rank, size_t in_bytes) {
    CollectiveConfig cfg;
    cfg.kind = CollKind::AllGatherRing;
    cfg.nranks = 4;
    cfg.rank = rank;
    cfg.input_bytes = in_bytes;
    cfg.output_bytes = 4100;
    cfg.tile_bytes = 512;
    cfg.dtype = ScalarType::Float32;
    return cfg;
  };
  build_coll_algo(make_ag(1, 1024), /*inplace=*/false);
  threw = false;
  try {
    build_coll_algo(make_ag(1, 1028), /*inplace=*/false);
  } catch (std::invalid_argument const&) {
    threw = true;
  }
  assert(threw);
  // Misaligned to dtype size.
  threw = false;
  try {
    build_coll_algo(make_ag(1, 1023), /*inplace=*/false);
  } catch (std::invalid_argument const&) {
    threw = true;
  }
  assert(threw);
}

// Layer 4: lower

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
    if (op.kind == LogicalOpKind::Signal) saw_signal = true;
    if (op.kind == LogicalOpKind::Wait) saw_waitsig = true;
  }
  assert(saw_signal);
  assert(saw_waitsig);

  // G=1: every Signal op is a fusion-eligible group of one Put.
  size_t nsig = 0;
  for (auto const& op : tiled.ops)
    if (op.kind == LogicalOpKind::Signal) ++nsig;
  assert(tiled.fused_put_signal.size() == nsig);
  for (auto [s, p] : tiled.fused_put_signal) {
    assert(tiled.ops[s].kind == LogicalOpKind::Signal);
    assert(tiled.ops[p].kind == LogicalOpKind::Put);
    assert(tiled.ops[s].deps.size() == 1 && tiled.ops[s].deps[0] == p);
  }
  assert(tiled.sig_group_size.size() == nsig);
  for (auto [s, g] : tiled.sig_group_size) {
    assert(tiled.ops[s].kind == LogicalOpKind::Signal);
    assert(g == 1);
  }
}

void test_lower_algo_signal_grouping() {
  printf("[test] lower_algo signal grouping...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);

  auto count_kind = [](TiledResult const& t, LogicalOpKind k) {
    size_t n = 0;
    for (auto const& op : t.ops)
      if (op.kind == k) ++n;
    return n;
  };

  TiledResult g1 = lower_algo(algo, 512, /*signal_group_tiles=*/1);
  TiledResult g2 = lower_algo(algo, 512, /*signal_group_tiles=*/2);

  size_t sig1 = count_kind(g1, LogicalOpKind::Signal);
  size_t ws1 = count_kind(g1, LogicalOpKind::Wait);
  size_t sig2 = count_kind(g2, LogicalOpKind::Signal);
  size_t ws2 = count_kind(g2, LogicalOpKind::Wait);

  // 1024-byte shards / 512-byte tiles → exactly 2 tiles per chunk pair,
  // so G=2 exactly halves the signal/wait counts.
  assert(sig1 > 0 && sig2 * 2 == sig1);
  assert(ws1 > 0 && ws2 * 2 == ws1);

  // Data-moving op counts are unaffected by grouping.
  assert(count_kind(g2, LogicalOpKind::Put) ==
         count_kind(g1, LogicalOpKind::Put));
  assert(count_kind(g2, LogicalOpKind::Reduce) ==
         count_kind(g1, LogicalOpKind::Reduce));

  // With G=2 every Signal depends on both Puts of its (full) group.
  for (auto const& op : g2.ops)
    if (op.kind == LogicalOpKind::Signal) assert(op.deps.size() == 2);

  // 2-tile groups collapse to group index 0 in the tag's low
  // tag_group_bits bits (adaptive layout; all-ones value reserved).
  for (auto const& op : g2.ops) {
    if (op.kind == LogicalOpKind::Signal || op.kind == LogicalOpKind::Wait)
      assert((op.tag & ((1ull << g2.tag_group_bits) - 1)) == 0);
  }

  // G=2: each group contributes 2 fusion-carrying puts, group size 2.
  assert(g2.fused_put_signal.size() == sig2 * 2);
  assert(!g1.fused_put_signal.empty());
  for (auto [s, g] : g2.sig_group_size) assert(g == 2);
  assert(!g2.wait_group_size.empty());
  for (auto [w, g] : g2.wait_group_size) assert(g == 2);
}

void test_lower_algo_reduce_scatter() {
  printf("[test] lower_algo reduce-scatter...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::ReduceScatterRing;
  cfg.output_bytes = 1024;
  TiledResult tiled = build_tiled(cfg, /*inplace=*/false);

  // Full-input-layout scratch for partial shards.
  assert(tiled.staging_bytes_required == 4096);

  size_t nreduce = 0, copy_tiles = 0;
  bool saw_signal = false, saw_wait = false;
  for (auto const& op : tiled.ops) {
    if (op.kind == LogicalOpKind::Signal) saw_signal = true;
    if (op.kind == LogicalOpKind::Wait) saw_wait = true;
    if (op.kind == LogicalOpKind::Reduce) {
      // Reduce: Scratch (peer partial) += Input (my contribution).
      assert(op.src_buf_role == CollectiveBufferRole::Input);
      assert(op.dst_buf_role == CollectiveBufferRole::Scratch);
      ++nreduce;
    }
    if (op.kind == LogicalOpKind::Put) {
      if (op.dst_peer == ~0u) {
        // Local own-shard copy tiles -> Output[0..shard).
        assert(op.src_buf_role == CollectiveBufferRole::Scratch);
        assert(op.dst_buf_role == CollectiveBufferRole::Output);
        assert(op.dst_off < 1024);
        ++copy_tiles;
      } else {
        // Remote puts always land in the peer's Scratch.
        assert(op.dst_buf_role == CollectiveBufferRole::Scratch);
      }
    }
  }
  assert(saw_signal && saw_wait);
  assert(nreduce == 6);    // 3 RecvReduce chunks x 2 tiles
  assert(copy_tiles == 2); // 1024-byte copy / 512-byte tiles
}

void test_lower_algo_allgather() {
  printf("[test] lower_algo all-gather...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllGatherRing;
  cfg.input_bytes = 1024;
  TiledResult tiled = build_tiled(cfg, /*inplace=*/false);

  assert(tiled.staging_bytes_required == 0);

  size_t copy_tiles = 0, remote_puts = 0, in_src_puts = 0;
  for (auto const& op : tiled.ops) {
    if (op.kind != LogicalOpKind::Put) continue;
    if (op.dst_peer == ~0u) {
      // Local own-shard copy Input[0] -> Output[offset].
      assert(op.src_buf_role == CollectiveBufferRole::Input);
      assert(op.dst_buf_role == CollectiveBufferRole::Output);
      assert(op.dst_off >= 1024 && op.dst_off < 2048);
      ++copy_tiles;
    } else {
      // Ring puts land in the peer's Output layout; the first-step own
      // shard send reads Input[0], forwards read Output.
      assert(op.dst_buf_role == CollectiveBufferRole::Output);
      if (op.src_buf_role == CollectiveBufferRole::Input)
        ++in_src_puts;
      else
        assert(op.src_buf_role == CollectiveBufferRole::Output);
      ++remote_puts;
    }
  }
  assert(copy_tiles == 2);   // 1024-byte copy / 512-byte tiles
  assert(remote_puts == 6);  // 3 ring Put chunks x 2 tiles
  assert(in_src_puts == 2);  // first-step send: 1 chunk x 2 tiles
}

void test_lower_algo_alltoall_basic() {
  printf("[test] tile_and_schedule alltoall...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 2048, 256);
  cfg.kind = CollKind::AllToAllPairwise;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/true);
  TiledResult tiled = lower_algo(algo, /*tile_bytes=*/256);
  assert(!tiled.ops.empty());
  bool has_signal = false;
  for (auto const& op : tiled.ops)
    if (op.kind == LogicalOpKind::Signal) has_signal = true;
  assert(has_signal);
}

void test_full_pipeline_ring_allreduce() {
  printf("[test] full pipeline ring allreduce...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/false);
  TiledResult tiled = lower_algo(algo, cfg.tile_bytes);

  // Verify all data ops are valid (signals have zero bytes).
  for (auto const& op : tiled.ops) {
    if (op.kind == LogicalOpKind::Signal || op.kind == LogicalOpKind::Wait)
      continue;
    assert(op.bytes > 0);
  }
}

void test_full_pipeline_alltoall() {
  printf("[test] full pipeline alltoall...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 2, 8192, 512);
  cfg.kind = CollKind::AllToAllPairwise;
  CollAlgo algo = build_coll_algo(cfg, /*inplace=*/true);
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
  // AllToAll is always inplace (input == output).
  for (int i = 0; i < kWarmup; ++i) build_tiled(cfg, true);

  auto t0 = std::chrono::steady_clock::now();
  size_t total_ops = 0;
  for (int i = 0; i < kIters; ++i) {
    TiledResult r = build_tiled(cfg, true);
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
  test_build_coll_algo_reduce_scatter_basic();
  test_build_coll_algo_allgather_basic();
  test_build_coll_algo_allgather_inplace();
  test_build_coll_algo_tree_basic();
  test_build_coll_algo_ag_rs_validation();
  test_verify_algo_pairing_all_kinds();

  // Layer 4
  test_lower_algo_rejects_zero_tile_bytes();
  test_lower_algo_ring_basic();
  test_lower_algo_signal_grouping();
  test_lower_algo_reduce_scatter();
  test_lower_algo_allgather();
  test_lower_algo_alltoall_basic();

  // Integration
  test_full_pipeline_ring_allreduce();
  test_full_pipeline_alltoall();

  // Benchmarks
  bench_lower_algo_large_ring();
  bench_lower_algo_large_alltoall();

  printf("\nModule tests PASSED\n");
  return 0;
}

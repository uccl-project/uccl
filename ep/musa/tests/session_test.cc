#include "session.h"
#include "kernel_math.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <thread>

namespace uccl::musa_ep {
// Explicit host substitutes. The index/reduction functions are production
// code, but these do not execute .mu kernels or validate stream ordering.
musaError_t gather_rows(float const* x, float* y, size_t const* rows,
                        size_t count, size_t hidden, musaStream_t) {
  for (size_t i = 0; i < count * hidden; ++i)
    y[i] = x[gather_index(i, hidden, rows)];
  return musaSuccess;
}
musaError_t scatter_rows(float const* x, float* y, size_t const* rows,
                         size_t count, size_t hidden, musaStream_t) {
  for (size_t i = 0; i < count * hidden; ++i)
    y[gather_index(i, hidden, rows)] = x[i];
  return musaSuccess;
}
musaError_t weighted_combine(float const* x, float* y, size_t const* lanes,
                             float const* weights, size_t tokens, size_t hidden,
                             size_t topk, musaStream_t) {
  for (size_t i = 0; i < tokens * hidden; ++i)
    y[i] = combine_element(x, lanes, weights, i, hidden, topk);
  return musaSuccess;
}
}  // namespace uccl::musa_ep

using namespace uccl::musa_ep;
auto const ready = reinterpret_cast<musaEvent_t>(1);

template <class F>
void rejects(F f) {
  bool caught = false;
  int const transfers = fake::transfers;
  try {
    f();
  } catch (std::exception const&) {
    caught = true;
  }
  assert(caught && fake::transfers == transfers);
}

void run_rank(int rank, int ranks) {
  fake::rank = rank;
  rejects([&] {
    Session bad_config(
        {size_t(ranks * 2), size_t(rank == 0 ? 4 : 7), 3, 65536, 1 << 24}, 0,
        0);
  });
  Session session({size_t(ranks * 2), 7, 3, 65536, 1 << 24}, 0, 0);
  RouteHandle stale;
  for (unsigned seed = 0; seed < 100; ++seed) {
    std::mt19937 rng(seed * 101 + rank);
    size_t const tokens = seed % 11 == 0 ? 0 : (seed + rank) % 17;
    std::vector<int64_t> indices(tokens * 3);
    std::vector<float> weights(tokens * 3), x(tokens * 7);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 19) - 9);
    for (size_t i = 0; i < indices.size(); ++i) {
      indices[i] = seed % 7 == 0 ? -1 : int(rng() % (ranks * 2));
      weights[i] = float(int(rng() % 9) - 4) / 4;
    }
    auto result = session.dispatch(x.data(), tokens, indices, weights, ready);
    rejects([&] { session.close(); });
    if (seed) rejects([&] { session.combine(nullptr, 0, 7, stale, ready); });
    std::vector<float> expert(result.rows * result.hidden);
    auto const& offsets = session.expert_offsets();
    for (size_t e = 0; e + 1 < offsets.size(); ++e)
      for (size_t row = offsets[e]; row < offsets[e + 1]; ++row)
        for (size_t h = 0; h < 7; ++h) {
          float v = result.data[row * 7 + h];
          expert[row * 7 + h] =
              float(rank * 2 + e + 1) * v - float(rank * 2 + e);
        }
    // One rank's bad shape must reject the operation on every rank.
    rejects([&] {
      session.combine(expert.data(), result.rows + (rank == 0), 7,
                      result.handle, ready);
    });
    auto y =
        session.combine(expert.data(), result.rows, 7, result.handle, ready);
    stale = result.handle;
    rejects([&] {
      session.combine(expert.data(), result.rows, 7, result.handle, ready);
    });
    for (size_t t = 0; t < tokens; ++t)
      for (size_t h = 0; h < 7; ++h) {
        float expected = 0;
        for (size_t lane = 0; lane < 3; ++lane) {
          size_t i = t * 3 + lane;
          if (indices[i] >= 0)
            expected += weights[i] * (float(indices[i] + 1) * x[t * 7 + h] -
                                      float(indices[i]));
        }
        assert(y.data()[t * 7 + h] == expected);
      }
  }
  std::vector<int64_t> idx(3, rank * 2);
  std::vector<float> weights(3, 1), x(7, 1);
  auto bad_idx = idx;
  if (rank == 0) bad_idx[0] = ranks * 2;
  rejects([&] { session.dispatch(x.data(), 1, bad_idx, weights, ready); });
  rejects([&] {
    session.dispatch(x.data(), 1, idx, weights, rank == 0 ? nullptr : ready);
  });
  // Fail one rank's arena/output allocation, then retry the same round.
  if (rank == 0) fake::fail_alloc = 0;
  int ok = 1, all;
  MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, 0);
  rejects([&] { session.dispatch(x.data(), 1, idx, weights, ready); });
  auto result = session.dispatch(x.data(), 1, idx, weights, ready);
  rejects([&] {
    if (rank == 0)
      session.cancel_completed_round(result.handle, ready);
    else
      session.combine(nullptr, 0, 7, result.handle, ready);
  });
  session.cancel_completed_round(result.handle, ready);
  auto next = session.dispatch(x.data(), 1, idx, weights, ready);
  rejects([&] { session.cancel_completed_round(result.handle, ready); });
  session.cancel_completed_round(next.handle, ready);
  // A one-rank capacity failure must reject before any peer posts payload.
  Session small({size_t(ranks * 2), 7, 3, 3, 1 << 24}, 0, 0);
  std::vector<int64_t> skew(3, 0);
  rejects([&] { small.dispatch(x.data(), 1, skew, weights, ready); });
  small.close();
  // Identity agreement also catches different Session instances with the
  // same topology, state and generation on a shared control communicator.
  Session other({size_t(ranks * 2), 7, 3, 65536, 1 << 24}, 0, 0);
  rejects([&] {
    if (rank == 0)
      session.dispatch(x.data(), 1, idx, weights, ready);
    else
      other.dispatch(x.data(), 1, idx, weights, ready);
  });
  auto a = session.dispatch(x.data(), 1, idx, weights, ready);
  auto b = other.dispatch(x.data(), 1, idx, weights, ready);
  rejects([&] { session.combine(b.data, b.rows, 7, b.handle, ready); });
  rejects([&] { other.cancel_completed_round(b.handle, nullptr); });
  other.cancel_completed_round(b.handle, ready);
  session.cancel_completed_round(a.handle, ready);
  other.close();
  for (unsigned repeat = 0; repeat < 1000; ++repeat) {
    auto r = session.dispatch(x.data(), 1, idx, weights, ready);
    auto y = session.combine(r.data, r.rows, 7, r.handle, ready);
    for (size_t h = 0; h < 7; ++h) assert(y.data()[h] == 3);
  }
  session.close();
  rejects([&] { session.dispatch(x.data(), 1, idx, weights, ready); });
}

int main() {
  for (int ranks : {2, 4}) {
    fake::ranks = ranks;
    std::vector<std::thread> workers;
    for (int rank = 0; rank < ranks; ++rank)
      workers.emplace_back(run_rank, rank, ranks);
    for (auto& worker : workers) worker.join();
    assert(fake::live_allocations == 0);
  }
  assert(fake::waits && fake::syncs && fake::transfers);
  std::cout
      << "EP Session dispatch/combine/control: PASS (2/4 ranks, 100 seeds)\n";
}

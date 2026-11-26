#include "exchange_plan.h"
#include "numeric.h"
#include "session.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>

using namespace uccl::musa_ep;
musaError_t expert_affine(float const*, float*, size_t const*, size_t, size_t,
                          musaStream_t);
musaError_t delay_work(unsigned, musaStream_t);
namespace {
void gpu(musaError_t error) {
  if (error != musaSuccess)
    throw std::runtime_error("MUSA test failure " + std::to_string(int(error)));
}
void mpi(int error) {
  if (error != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, error);
}
template <class F>
void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (std::invalid_argument const&) {
    caught = true;
  }
  if (!caught) throw std::runtime_error("expected collective rejection");
}
}  // namespace
int main(int argc, char** argv) {
  mpi(MPI_Init(&argc, &argv));
  int rank, ranks, local_rank;
  mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  mpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  try {
    if (argc < 5 || argc > 7)
      throw std::invalid_argument(
          "usage: ep_test H K rounds seed [tokens [delay_steps]]");
    auto number = [](char const* text) {
      size_t end = 0;
      auto value = std::stoull(text, &end);
      if (!*text || *text == '-' || text[end])
        throw std::invalid_argument("invalid integer");
      return size_t(value);
    };
    size_t hidden = number(argv[1]), topk = number(argv[2]),
           rounds = number(argv[3]), seed = number(argv[4]);
    size_t const fixed_tokens = argc >= 6 ? number(argv[5]) : 0;
    size_t const delay_steps = argc == 7 ? number(argv[6]) : 0;
    if (!rounds || !hidden || hidden > 8192 || !topk || topk > 8 || ranks < 2)
      throw std::invalid_argument("invalid EP test shape");
    if (fixed_tokens > 65536 / topk || delay_steps > 10000000)
      throw std::invalid_argument("invalid test capacity or delay");
    MPI_Comm local, control;
    mpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &local));
    mpi(MPI_Comm_rank(local, &local_rank));
    mpi(MPI_Comm_dup(MPI_COMM_WORLD, &control));
    int devices;
    gpu(musaGetDeviceCount(&devices));
    if (local_rank >= devices)
      throw std::runtime_error("one visible GPU per local rank required");
    gpu(musaSetDevice(local_rank));
    mcclUniqueId id;
    if (rank == 0 && mcclGetUniqueId(&id) != mcclSuccess)
      throw std::runtime_error("MCCL ID");
    mpi(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    mcclComm_t comm;
    if (mcclCommInitRank(&comm, ranks, id, rank) != mcclSuccess)
      throw std::runtime_error("MCCL init");
    musaStream_t producer, expert_stream;
    musaEvent_t ready, expert_ready;
    gpu(musaStreamCreate(&producer));
    gpu(musaStreamCreate(&expert_stream));
    gpu(musaEventCreate(&ready));
    gpu(musaEventCreate(&expert_ready));
    Session session({size_t(ranks * 2), hidden, topk}, control, comm);
    double dispatch_us = 0, combine_us = 0;
    for (size_t iteration = 0; iteration < rounds; ++iteration) {
      std::mt19937 rng(unsigned(seed + iteration * 101 + rank));
      // Empty source can still own experts and receive nonempty peer work.
      size_t tokens = iteration % 13 == 0 ? (rank == 0 ? 0 : 17)
                                          : (iteration + rank) % 17 + 1;
      if (argc >= 6) tokens = fixed_tokens;
      std::vector<int64_t> indices(checked_product(tokens, topk));
      std::vector<float> weights(indices.size()),
          input(checked_product(tokens, hidden));
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = float(int((i + iteration + rank) % 19) - 9);
      for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = iteration % 11 == 0  ? -1
                     : iteration % 7 == 0 ? 0
                                          : int(rng() % (ranks * 2));
        weights[i] = iteration % 2 == 0 ? float(int(rng() % 9) - 4) / 4
                                        : float(int(rng() % 101) - 50) / 37;
      }
      DeviceTensor x(tokens, hidden);
      gpu(delay_work(unsigned(delay_steps), producer));
      if (!input.empty())
        gpu(musaMemcpyAsync(x.data(), input.data(),
                            input.size() * sizeof(float),
                            musaMemcpyHostToDevice, producer));
      gpu(musaEventRecord(ready, producer));
      auto start = std::chrono::steady_clock::now();
      auto result = session.dispatch(x.data(), tokens, indices, weights, ready);
      auto middle = std::chrono::steady_clock::now();
      DeviceTensor expert(result.rows, hidden);
      std::vector<size_t> expert_ids(result.rows);
      auto const& offsets = session.expert_offsets();
      for (size_t e = 0; e + 1 < offsets.size(); ++e)
        for (size_t row = offsets[e]; row < offsets[e + 1]; ++row)
          expert_ids[row] = size_t(rank * 2) + e;
      void* device_ids = nullptr;
      if (!expert_ids.empty()) {
        gpu(musaMalloc(&device_ids, expert_ids.size() * sizeof(size_t)));
        gpu(musaMemcpyAsync(device_ids, expert_ids.data(),
                            expert_ids.size() * sizeof(size_t),
                            musaMemcpyHostToDevice, expert_stream));
      }
      gpu(delay_work(unsigned(delay_steps), expert_stream));
      gpu(expert_affine(result.data, expert.data(),
                        static_cast<size_t*>(device_ids), result.rows, hidden,
                        expert_stream));
      gpu(musaEventRecord(expert_ready, expert_stream));
      if (iteration == 0) {
        rejects([&] { session.close(); });
        rejects([&] {
          session.combine(expert.data(), result.rows + (rank == 0), hidden,
                          result.handle, expert_ready);
        });
      }
      auto before_combine = std::chrono::steady_clock::now();
      auto y = session.combine(expert.data(), result.rows, hidden,
                               result.handle, expert_ready);
      auto end = std::chrono::steady_clock::now();
      dispatch_us +=
          std::chrono::duration<double, std::micro>(middle - start).count();
      combine_us +=
          std::chrono::duration<double, std::micro>(end - before_combine)
              .count();
      if (device_ids) gpu(musaFree(device_ids));
      std::vector<float> output(tokens * hidden);
      if (!output.empty())
        gpu(musaMemcpy(output.data(), y.data(), output.size() * sizeof(float),
                       musaMemcpyDeviceToHost));
      for (size_t t = 0; t < tokens; ++t)
        for (size_t h = 0; h < hidden; ++h) {
          double expected = 0;
          for (size_t lane = 0; lane < topk; ++lane) {
            size_t const i = t * topk + lane;
            if (indices[i] != -1)
              expected += double(weights[i]) *
                          ((indices[i] + 1) * double(input[t * hidden + h]) -
                           indices[i]);
          }
          if (!test::matches(output[t * hidden + h], expected,
                             iteration % 2 == 0))
            throw std::runtime_error("affine oracle mismatch");
        }
      if (iteration == 0)
        rejects([&] {
          session.combine(expert.data(), result.rows, hidden, result.handle,
                          expert_ready);
        });
    }
    session.close();
    gpu(musaEventDestroy(ready));
    gpu(musaEventDestroy(expert_ready));
    gpu(musaStreamDestroy(producer));
    gpu(musaStreamDestroy(expert_stream));
    if (mcclCommDestroy(comm) != mcclSuccess)
      throw std::runtime_error("MCCL destroy");
    mpi(MPI_Comm_free(&control));
    mpi(MPI_Comm_free(&local));
    std::printf(
        "EP affine rank=%d H=%zu K=%zu rounds=%zu seed=%zu: PASS "
        "dispatch_mean_us=%.3f combine_mean_us=%.3f\n",
        rank, hidden, topk, rounds, seed, dispatch_us / rounds,
        combine_us / rounds);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "EP test rank=%d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  mpi(MPI_Finalize());
}

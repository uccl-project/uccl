// Direct registry consumer: verify explicit rail compatibility and collective
// rejection before transport setup on the retained {4,4} executor.
#include "uccl_lite/runtime/collective_backend.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
using namespace uccl_lite;
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &localComm);
  int localRank, localRanks;
  MPI_Comm_rank(localComm, &localRank);
  MPI_Comm_size(localComm, &localRanks);
  try {
    bool badHosts = argc >= 2 && std::string(argv[1]) == "bad-hosts";
    if (ranks != 8 || (!badHosts && localRanks != 4) || (argc < 2 || argc > 4))
      throw std::invalid_argument(
          "two four-rank hosts; explicit|invalid|automatic|uncovered mode; "
          "optional reverse device assignment and full size sweep");
    bool reverse = false, full = false;
    for (int i = 2; i < argc; ++i) {
      if (std::string(argv[i]) == "reverse")
        reverse = true;
      else if (std::string(argv[i]) == "full")
        full = true;
      else
        throw std::invalid_argument("unknown checker option");
    }
    bool invalid = std::string(argv[1]) == "invalid";
    bool duplicate = std::string(argv[1]) == "duplicate";
    bool disabled = std::string(argv[1]) == "gdrcopy-disabled";
    bool automatic = std::string(argv[1]) == "automatic";
    bool uncovered = std::string(argv[1]) == "uncovered";
    if (!invalid && !duplicate && !badHosts && !disabled && !automatic &&
        !uncovered && std::string(argv[1]) != "explicit")
      throw std::invalid_argument("invalid mode");
    if (disabled &&
        (!std::getenv("MSCCLPP_FORCE_DISABLE_GDR") ||
         std::string(std::getenv("MSCCLPP_FORCE_DISABLE_GDR")) != "1"))
      throw std::invalid_argument(
          "gdrcopy-disabled mode requires MSCCLPP_FORCE_DISABLE_GDR=1");
    int device = reverse ? 3 - (localRank % 4) : localRank % 4;
    if (duplicate && localRank == 3) device = reverse ? 3 : 0;
    check(cudaSetDevice(device));
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      runtime::BackendContext context;
      context.rank = rank;
      context.ranks = ranks;
      context.device = device;
      context.rail = automatic || uncovered ? -1 : invalid && rank == 7 ? 7 : 0;
      context.capacity = full ? 32UL << 20 : 1UL << 20;
      context.localDevices = {0, 1, 2, 3};
      auto registration = runtime::networkHierarchyBackend();
      if (!rank) context.id = registration.uniqueId();
      MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
      auto backend = registration.create(context);
      int failed = 0, failures;
      try {
        backend->prepare();
      } catch (std::exception const& error) {
        failed = 1;
        std::cerr << "rank " << rank << " prepare: " << error.what() << '\n';
      }
      MPI_Allreduce(&failed, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (invalid || duplicate || badHosts || disabled || uncovered) {
        if (failures != ranks || backend->initialized())
          throw std::runtime_error(
              "invalid explicit rail did not reject collectively");
        continue;
      }
      if (failures || !backend->initialized())
        throw std::runtime_error("valid explicit rail rejected");
      cudaStream_t stream;
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      std::vector<size_t> counts{4096};
      if (full) counts.push_back(1UL << 20);
      for (size_t count : counts) {
        constexpr size_t guard = 32;
        size_t extent = count * ranks + 2 * guard;
        float *input, *output;
        check(cudaMalloc(&input, extent * sizeof(float)));
        check(cudaMalloc(&output, extent * sizeof(float)));
        for (int layout = 0; layout < 2; ++layout)
          for (int generation = 0; generation < 3; ++generation)
            for (int op = 1; op <= 3; ++op) {
              bool alias = layout && (rank & 1);
              bool inputB = alias && op == 1, outputA = alias && op != 1;
              size_t in = count * (op == 2 ? ranks : 1),
                     out = count * (op == 1 ? ranks : 1),
                     inputOffset = guard + (inputB ? rank * count : 0),
                     outputOffset =
                         guard + (alias && op == 2 ? rank * count : 0);
              std::vector<float> a(extent, -99), b(extent, -99), actual(extent);
              auto& source = inputB ? b : a;
              for (size_t i = 0; i < in; ++i)
                source[inputOffset + i] = rank + 1 + float(i % 7) + generation;
              check(cudaMemcpy(input, a.data(), extent * sizeof(float),
                               cudaMemcpyHostToDevice));
              check(cudaMemcpy(output, b.data(), extent * sizeof(float),
                               cudaMemcpyHostToDevice));
              auto& expected = outputA ? a : b;
              for (size_t i = 0; i < out; ++i)
                expected[outputOffset + i] =
                    op == 1
                        ? float(i / count + 1 + (i % count) % 7 + generation)
                        : float(36 +
                                8 * (((op == 2 ? size_t(rank) * count + i : i) %
                                      7) +
                                     generation));
              backend->enqueue(
                  NetworkOperation(op), (inputB ? output : input) + inputOffset,
                  (outputA ? input : output) + outputOffset, count, stream);
              check(cudaStreamSynchronize(stream));
              backend->synchronize();
              check(cudaMemcpy(actual.data(), input, extent * sizeof(float),
                               cudaMemcpyDeviceToHost));
              if (actual != a)
                throw std::runtime_error(
                    "logical-rank input/alias/guard correctness failed");
              check(cudaMemcpy(actual.data(), output, extent * sizeof(float),
                               cudaMemcpyDeviceToHost));
              if (actual != b)
                throw std::runtime_error(
                    "logical-rank output/guard correctness failed");
            }
        check(cudaFree(input));
        check(cudaFree(output));
      }
      check(cudaStreamDestroy(stream));
    }
    if (!rank)
      std::cout << "PASS: network registry " << argv[1]
                << " rails, two lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&localComm);
  MPI_Finalize();
}

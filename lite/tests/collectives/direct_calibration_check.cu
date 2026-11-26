#include "qualification.cuh"
#include "uccl_lite/collectives/direct_exchange.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include <linux/mempolicy.h>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
using namespace uccl_lite;
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
template <class Bits>
Bits encode(float x) {
  if constexpr (sizeof(Bits) == 4) {
    Bits bits;
    std::memcpy(&bits, &x, 4);
    return bits;
  } else
    return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
template <class Bits>
float decode(Bits x) {
  if constexpr (sizeof(Bits) == 4) {
    float value;
    std::memcpy(&value, &x, 4);
    return value;
  } else
    return __bfloat162float(__ushort_as_bfloat16(x));
}
template <class Bits>
Bits value(size_t i, int rank, int generation, bool gather) {
  if (gather)
    return Bits(uint32_t(i) * 2246822519u + rank * 7919u + generation);
  float x;
  if (i % 11 == 0 && rank < 4) {
    float cancellation[4] = {16777216.0f, float(1 + generation), -16777216.0f,
                             3.0f};
    x = cancellation[rank];
  } else
    x = float((int(i % 17) - 8) * (rank + 1)) / float(rank + 3) +
        generation * .125f;
  return encode<Bits>(x);
}
__global__ void skew() {
  auto start = clock64();
  while (clock64() - start < 2000000) {
  }
}
template <class Bits, class Backend>
void exercise(Backend& peer, int rank, int ranks, int op, size_t count,
              bool alias, std::array<cudaStream_t, 2> streams) {
  size_t in = count * (op == 2 ? ranks : 1),
         out = count * (op == 1 ? ranks : 1), extent = std::max(in, out) + 128;
  size_t aOffset = 1 + (alias && op == 1 ? size_t(rank) * count : 0),
         bOffset = 1 + (alias && op == 2 ? size_t(rank) * count : 0);
  bool inputB = alias && op == 1, outputA = alias && op != 1;
  Bits *a, *b;
  check(cudaMalloc(&a, extent * sizeof(Bits)));
  check(cudaMalloc(&b, extent * sizeof(Bits)));
  auto* input = (inputB ? b : a) + aOffset;
  auto* output = (outputA ? a : b) + bOffset;
  for (int generation = 0; generation < 3; ++generation) {
    Bits guard = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
    std::vector<Bits> ha(extent, guard), hb(extent, guard), actual(extent);
    for (size_t i = 0; i < in; ++i)
      (inputB ? hb : ha)[aOffset + i] =
          value<Bits>(i, rank, generation, op == 1);
    // Pageable synchronous H2D may return before DMA completion. Produce on
    // the consumer stream so nonblocking streams have an explicit dependency.
    auto stream = streams[generation % 2];
    check(cudaMemcpyAsync(a, ha.data(), extent * sizeof(Bits),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(b, hb.data(), extent * sizeof(Bits),
                          cudaMemcpyHostToDevice, stream));
    if (rank == ranks - 1) skew<<<1, 32, 0, stream>>>();
    check(cudaGetLastError());
    peer.enqueue(NetworkOperation(op), input, output, count, stream);
    peer.synchronize();
    for (size_t i = 0; i < out; ++i) {
      Bits wanted;
      if (op == 1)
        wanted = value<Bits>(i % count, int(i / count), generation, true);
      else {
        // Independent source-ordered FP32 reference. Cancellation inputs
        // distinguish this contract from a reordered partial-sum tree.
        float sum = 0;
        for (int source = 0; source < ranks; ++source)
          sum += decode(value<Bits>(op == 2 ? size_t(rank) * count + i : i,
                                    source, generation, false));
        wanted = encode<Bits>(sum);
      }
      (outputA ? ha : hb)[bOffset + i] = wanted;
    }
    for (auto pair : {std::make_pair(a, &ha), {b, &hb}}) {
      check(cudaMemcpy(actual.data(), pair.first, extent * sizeof(Bits),
                       cudaMemcpyDeviceToHost));
      if (actual != *pair.second) {
        for (size_t i = 0; i < extent; ++i)
          if (actual[i] != (*pair.second)[i]) {
            std::cerr << "direct exchange mismatch rank=" << rank
                      << " op=" << op << " count=" << count
                      << " alias=" << alias << " generation=" << generation
                      << " buffer=" << (pair.first == a ? "a" : "b")
                      << " index=" << i << " got=" << uint64_t(actual[i])
                      << " expected=" << uint64_t((*pair.second)[i]) << '\n';
            break;
          }
        throw std::runtime_error(
            "direct exchange full-buffer/source-order/input/guard oracle "
            "mismatch");
      }
    }
  }
  check(cudaFree(a));
  check(cudaFree(b));
}

template <class Bits>
void run(int rank, int ranks, int device, size_t chunk) {
  auto dtype = sizeof(Bits) == 4 ? ScalarType::Float32 : ScalarType::BFloat16;
  NetworkId id;
  if (!rank) id = DirectExchangeCollectives::createUniqueId();
  MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  size_t capacity = chunk * size_t(ranks) * 2;
  DirectExchangeCollectives direct(id, rank, ranks, device, capacity, dtype,
                                   chunk);
  direct.prepare();
  auto profile = direct.calibrateModel();
  if (profile->observations.size() !=
          size_t(11 * ranks + 2 * ranks * (ranks - 1)) ||
      profile->configurationIdentities.size() != size_t(ranks))
    throw std::runtime_error("incomplete direct calibration observations");
  for (auto const& o : profile->observations) {
    if (o.microseconds.size() != 9)
      throw std::runtime_error("raw repetition samples missing");
    for (auto value : o.microseconds)
      if (!std::isfinite(value) || value < 0)
        throw std::runtime_error("invalid model sample");
    if (!rank) {
      std::cout << "OBS " << std::quoted(o.resource) << ' '
                << std::quoted(o.scope) << ' ' << o.bytes << ' '
                << o.trafficFactor << ' ' << std::quoted(o.rejection);
      for (auto us : o.microseconds)
        std::cout << ' ' << std::setprecision(17) << us;
      std::cout << '\n';
    }
  }
  std::array<cudaStream_t, 2> streams;
  for (auto& stream : streams)
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int op = 1; op <= 3; ++op) {
    auto operation = NetworkOperation(op);
    auto collective = op == 1   ? Collective::AllGather
                      : op == 2 ? Collective::ReduceScatter
                                : Collective::AllReduce;
    for (size_t count :
         {size_t(257), chunk / sizeof(Bits), chunk / sizeof(Bits) + 1}) {
      Workload work{collective, dtype,
                    count * sizeof(Bits) * (op == 3 ? 1 : ranks), false,
                    unsigned(sizeof(Bits))};
      auto estimate = estimateDirectModel(*profile, work);
      for (bool alias : {false, true})
        exercise<Bits>(direct, rank, ranks, op, count, alias, streams);
      uccl_lite::runtime::CollectiveRequest request{
          operation, count,
          std::vector<uccl_lite::runtime::RankBufferLayout>(
              ranks, {false, unsigned(sizeof(Bits))})};
      uccl_lite::runtime::qualification::Probe probe;
      probe.allocate(request, rank, ranks, dtype);
      std::vector<double> observed;
      for (int repeat = 0; repeat < 5; ++repeat) {
        probe.initialize(repeat);
        check(cudaStreamSynchronize(probe.stream));
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::steady_clock::now();
        direct.enqueue(operation, probe.input(), probe.output(), count,
                       probe.stream);
        double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count(),
               maximum;
        MPI_Allreduce(&us, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (repeat >= 2) observed.push_back(maximum);
        check(cudaMemsetAsync(probe.wrong, 0, sizeof(unsigned), probe.stream));
        probe.verify(repeat);
        unsigned wrong;
        check(cudaMemcpyAsync(&wrong, probe.wrong, sizeof(wrong),
                              cudaMemcpyDeviceToHost, probe.stream));
        check(cudaStreamSynchronize(probe.stream));
        if (wrong)
          throw std::runtime_error("calibrated execution full-buffer mismatch");
      }
      if (!rank) {
        std::cout << "MODEL op=" << op << " count=" << count
                  << " available=" << estimate.estimated
                  << " predicted_us=" << estimate.microseconds
                  << " reason=" << std::quoted(estimate.reason)
                  << " observed_us=";
        for (auto us : observed) std::cout << us << ',';
        std::cout << '\n';
      }
    }
  }
  for (auto stream : streams) check(cudaStreamDestroy(stream));
  if (!rank)
    std::cout << "PASS direct operational calibration ranks=" << ranks
              << " observations=" << profile->observations.size()
              << " identity=" << profile->identity
              << " omitted_cuts=" << profile->unmodeledResources.size() << '\n';
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks, localRank, localRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local);
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localRanks);
  try {
    if (argc != 3) throw std::invalid_argument("fp32|bf16 CHUNK_BYTES");
    int visible;
    check(cudaGetDeviceCount(&visible));
    int device = visible == 1 ? 0 : localRanks - 1 - localRank;
    check(cudaSetDevice(device));
    if (std::string(argv[1]) == "fp32")
      run<uint32_t>(rank, ranks, device, std::stoull(argv[2]));
    else
      run<uint16_t>(rank, ranks, device, std::stoull(argv[2]));
  } catch (std::exception const& e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

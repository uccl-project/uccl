#include "scalar_oracle.cuh"
#include "uccl_lite/runtime/collective_backend.hpp"
#include <iostream>
#include <numeric>
#include <mpi.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
using scalar_test::check;
void session(BackendContext& c, BackendRegistration const& registration) {
  if (!c.rank) c.id = registration.uniqueId();
  MPI_Bcast(c.id.data(), c.id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
}
void chain(CollectiveBackend& backend, int rank,
           std::array<cudaStream_t, 2> streams) {
  constexpr size_t count = 65536, guard = 32, extent = count * 4 + 2 * guard;
  float *a, *b, *c;
  check(cudaMalloc(&a, extent * 4));
  check(cudaMalloc(&b, extent * 4));
  check(cudaMalloc(&c, extent * 4));
  for (int generation = 0; generation < 3; ++generation) {
    std::vector<float> ha(extent, -99), hb(extent, -99), hc(extent, -99),
        actual(extent);
    for (size_t i = 0; i < count; ++i) ha[guard + i] = rank + 1 + generation;
    check(cudaMemcpy(a, ha.data(), extent * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, hb.data(), extent * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(c, hc.data(), extent * 4, cudaMemcpyHostToDevice));
    backend.enqueue(NetworkOperation::AllGather, a + guard, b + guard, count,
                    streams[0]);
    backend.enqueue(NetworkOperation::ReduceScatter, b + guard, c + guard,
                    count, streams[1]);
    backend.enqueue(NetworkOperation::AllReduce, c + guard, c + guard, count,
                    streams[0]);
    backend.synchronize();
    for (size_t i = 0; i < 4 * count; ++i)
      hb[guard + i] = i / count + 1 + generation;
    for (size_t i = 0; i < count; ++i)
      hc[guard + i] = 4 * (10 + 4 * generation);
    for (auto pair : {std::pair<float*, std::vector<float> const*>{a, &ha},
                      {b, &hb},
                      {c, &hc}}) {
      check(cudaMemcpy(actual.data(), pair.first, extent * 4,
                       cudaMemcpyDeviceToHost));
      if (actual != *pair.second)
        throw std::runtime_error("permuted producer/consumer/guard failure");
    }
  }
  check(cudaFree(a));
  check(cudaFree(b));
  check(cudaFree(c));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (ranks != 4 || argc != 2 ||
        (std::string(argv[1]) != "quick" && std::string(argv[1]) != "full"))
      throw std::invalid_argument("four ranks; quick|full");
    check(cudaSetDevice(rank));
    BackendContext c;
    c.rank = c.device = rank;
    c.ranks = ranks;
    c.capacity = 64UL << 20;
    c.localDevices = {0, 1, 2, 3};
    auto registration = peerPacketBackend();
    session(c, registration);
    auto bad = c;
    if (rank == 3) std::swap(bad.localDevices[0], bad.localDevices[1]);
    auto rejected = registration.create(bad);
    bool failed = false;
    try {
      rejected->prepare();
    } catch (std::invalid_argument const&) {
      failed = true;
    }
    if (!failed || rejected->initialized())
      throw std::runtime_error("different rank mapping accepted");
    rejected.reset();
    std::array<cudaStream_t, 2> streams;
    for (auto& s : streams)
      check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::vector<size_t> counts{64, 4096, 65536};
    if (std::string(argv[1]) == "full") {
      counts.push_back(1UL << 20);
      counts.push_back(4UL << 20);
    }
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      session(c, registration);
      auto backend = registration.create(c);
      backend->prepare();
      for (size_t count : counts)
        for (bool alias : {false, true})
          for (int op = 1; op <= 3; ++op)
            scalar_test::exercise<uint32_t>(
                op, count, alias && (rank & 1), 4u << (rank % 3), rank, ranks,
                streams,
                [&](int op, void const* input, void* output, size_t count,
                    cudaStream_t stream) {
                  backend->enqueue(NetworkOperation(op), input, output, count,
                                   stream);
                });
      chain(*backend, rank, streams);
      cudaGraph_t graph;
      check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      bool rejected = false;
      try {
        backend->enqueue(NetworkOperation::AllReduce,
                         reinterpret_cast<void*>(256),
                         reinterpret_cast<void*>(512), 64, streams[0]);
      } catch (std::invalid_argument const&) {
        rejected = true;
      }
      check(cudaStreamEndCapture(streams[0], &graph));
      check(cudaGraphDestroy(graph));
      if (!rejected) throw std::runtime_error("unsupported capture accepted");
      chain(*backend, rank, streams);
    }
    for (auto s : streams) check(cudaStreamDestroy(s));
    if (!rank)
      std::cout << "PASS: topology-mapped peer packet " << argv[1]
                << "; full guards, logical AG/RS order, FP32 reductions, mixed "
                   "aliases, cross-stream chains, mapping consensus and two "
                   "lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

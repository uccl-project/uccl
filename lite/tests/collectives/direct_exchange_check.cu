#include "uccl_lite/collectives/direct_exchange.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include <linux/mempolicy.h>
#include <array>
#include <cstring>
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
void run(int rank, int ranks, int device, size_t capacity, size_t chunk) {
  std::array<cudaStream_t, 2> streams;
  for (auto& stream : streams)
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto dtype = sizeof(Bits) == 4 ? ScalarType::Float32 : ScalarType::BFloat16;
  check(cudaGetLastError());
  {
    DirectExchangeCollectives unused({}, rank, ranks, device + 1024, capacity,
                                     dtype, chunk);
    if (unused.initialized())
      throw std::runtime_error("unprepared candidate initialized resources");
  }
  check(cudaGetLastError());  // Destruction must not touch the deferred device.

  for (int lifetime = 0; lifetime < 2; ++lifetime) {
    NetworkId id;
    if (!rank) id = DirectExchangeCollectives::createUniqueId();
    MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    {
      DirectExchangeCollectives bad(
          id, rank, ranks, device,
          capacity + (rank == ranks - 1 ? sizeof(Bits) : 0), dtype, chunk);
      bool rejected = false;
      try {
        bad.prepare();
      } catch (std::exception const&) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("inconsistent capacity accepted");
    }
    if (!rank) id = DirectExchangeCollectives::createUniqueId();
    MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    DirectExchangeCollectives peer(id, rank, ranks, device, capacity, dtype,
                                   chunk);
    if (peer.initialized() || !peer.routes().empty())
      throw std::runtime_error("constructor prepared resources");
    cpu_set_t before{}, after{};
    int modeBefore = 0, modeAfter = 0;
    std::array<unsigned long, 16> nodesBefore{}, nodesAfter{};
    if (sched_getaffinity(0, sizeof(before), &before) ||
        syscall(SYS_get_mempolicy, &modeBefore, nodesBefore.data(),
                sizeof(nodesBefore) * 8, nullptr, 0))
      throw std::runtime_error("cannot capture test placement");
    peer.prepare();
    if (sched_getaffinity(0, sizeof(after), &after) ||
        syscall(SYS_get_mempolicy, &modeAfter, nodesAfter.data(),
                sizeof(nodesAfter) * 8, nullptr, 0) ||
        !CPU_EQUAL(&before, &after) || modeBefore != modeAfter ||
        nodesBefore != nodesAfter)
      throw std::runtime_error("prepare changed caller placement");
    if (!peer.initialized() || peer.routes().size() != size_t(ranks - 1))
      throw std::runtime_error("incomplete route set");
    std::array<int, 33> row{};
    row.fill(-1);
    row[32] = device;
    for (auto route : peer.routes()) row[route.peer] = route.cudaIpc ? 1 : 0;
    std::vector<int> rows(size_t(ranks) * row.size());
    MPI_Gather(row.data(), row.size(), MPI_INT, rows.data(), row.size(),
               MPI_INT, 0, MPI_COMM_WORLD);
    int peerEdges = 0;
    if (!rank)
      for (int r = 0; r < ranks; ++r)
        for (int p = 0; p < ranks; ++p)
          if (r != p && rows[r * row.size() + p] == 1) ++peerEdges;
    MPI_Bcast(&peerEdges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int op = 1; op <= 3; ++op) {
      auto traffic = peer.traffic(NetworkOperation(op), 257);
      size_t payload = 257 * sizeof(Bits);
      if (traffic.peerPayloadBytes != size_t(peerEdges) * payload ||
          traffic.networkPayloadBytes !=
              (size_t(ranks) * (ranks - 1) - peerEdges) * payload ||
          traffic.chunks != (payload + chunk - 1) / chunk)
        throw std::runtime_error(
            "runtime traffic does not match actual transport routes");
    }
    if (!rank) {
      for (int r = 0; r < ranks; ++r) {
        std::cout << "ROUTES rank=" << r
                  << " device=" << rows[r * row.size() + 32];
        for (int p = 0; p < ranks; ++p)
          if (p != r)
            std::cout << ' ' << p << ':'
                      << (rows[r * row.size() + p] ? "ipc" : "ib");
        std::cout << '\n';
      }
      std::cout << "IDENTITY " << peer.identity() << '\n';
    }
    Bits* guard;
    check(cudaMalloc(&guard, 128 * sizeof(Bits)));
    check(cudaMemsetAsync(guard, 0x5a, 128 * sizeof(Bits), streams[0]));
    check(cudaStreamSynchronize(streams[0]));
    for (int mode = 0; mode < 3; ++mode) {
      if (mode == 2 && !rank)
        check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      bool rejected = false;
      try {
        peer.enqueue(NetworkOperation::AllReduce, guard,
                     mode == 1 && !rank ? guard + 1 : guard + 64,
                     mode == 0 && !rank ? 1 : 16, streams[0]);
      } catch (std::exception const&) {
        rejected = true;
      }
      if (mode == 2 && !rank) {
        cudaGraph_t graph;
        check(cudaStreamEndCapture(streams[0], &graph));
        check(cudaGraphDestroy(graph));
      }
      if (!rejected)
        throw std::runtime_error("inconsistent/aliased/captured call accepted");
      std::vector<unsigned char> bytes(128 * sizeof(Bits));
      check(cudaMemcpy(bytes.data(), guard, bytes.size(),
                       cudaMemcpyDeviceToHost));
      for (auto b : bytes)
        if (b != 0x5a)
          throw std::runtime_error("rejected call modified buffers");
    }
    check(cudaFree(guard));
    size_t width = chunk / sizeof(Bits);
    for (size_t count : {size_t(1), size_t(257), width - 1, width, width + 1,
                         capacity / (ranks * sizeof(Bits))})
      for (int op = 1; op <= 3; ++op)
        for (bool alias : {false, true})
          exercise<Bits>(peer, rank, ranks, op, count, alias, streams);
    exercise<Bits>(peer, rank, ranks, 3, capacity / sizeof(Bits), true,
                   streams);
  }
  auto registration = runtime::directExchangeBackend(dtype);
  if (runtime::availableBackends().size() != 2 || registration.spec.automatic ||
      !registration.spec.acceptsLayout({ranks}) ||
      !registration.spec.acceptsLayout({1, ranks - 1}) ||
      registration.spec.acceptsLayout({32, 1}))
    throw std::runtime_error("standalone candidate catalog/extent mismatch");
  runtime::BackendContext context;
  if (!rank) context.id = registration.uniqueId();
  MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  context.rank = rank;
  context.ranks = ranks;
  context.device = device;
  context.capacity = capacity;
  {
    auto backend = registration.create(context);
    if (backend->initialized() || backend->supportsGraphs())
      throw std::runtime_error("invalid generic backend capabilities");
    backend->prepare();
    auto traffic =
        backend->traffic(NetworkOperation::AllReduce, capacity / sizeof(Bits));
    if (traffic.at("gpu_memory_bytes") !=
            size_t(3 * ranks * ranks + 3 * ranks) * capacity ||
        traffic.at("chunks") != (capacity + 256 * 1024 - 1) / (256 * 1024))
      throw std::runtime_error(
          "backend traffic query lost the prepared chunk geometry");
    for (int op = 1; op <= 3; ++op)
      for (bool alias : {false, true})
        exercise<Bits>(*backend, rank, ranks, op, 257, alias, streams);
    if (!backend->idle())
      throw std::runtime_error("synchronous backend is not idle");
  }
  for (auto stream : streams) check(cudaStreamDestroy(stream));
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
    if (argc != 4)
      throw std::invalid_argument("bf16|fp32 CAPACITY_BYTES CHUNK_BYTES");
    size_t capacity = std::stoull(argv[2]), chunk = std::stoull(argv[3]);
    if (chunk < 1024 || capacity < chunk * size_t(ranks) * 2)
      throw std::invalid_argument("test capacity too small");
    int visible;
    check(cudaGetDeviceCount(&visible));
    int device = visible == 1 ? 0 : localRanks - 1 - localRank;
    check(cudaSetDevice(device));
    std::string type = argv[1];
    if (type == "fp32")
      run<uint32_t>(rank, ranks, device, capacity, chunk);
    else if (type == "bf16")
      run<uint16_t>(rank, ranks, device, capacity, chunk);
    else
      throw std::invalid_argument("bad dtype");
    if (!rank)
      std::cout
          << "PASS direct exchange " << type << " ranks=" << ranks
          << "; all operations, source order, full buffers/guards, aliases, "
             "tails, chunk reuse, call rejection, two lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

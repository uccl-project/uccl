#include "native_fallback.hpp"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include "uccl_lite/runtime/prepared_network.hpp"
#include <linux/mempolicy.h>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
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
  // Every partial sum is integral and representable for all supported ranks,
  // allowing an independent bitwise oracle for both native and direct paths.
  float x = float(int((i + rank) % 9) - 4 + generation);
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

using namespace uccl_lite::runtime;
void nativeCheck(ncclResult_t r) {
  if (r != ncclSuccess) throw std::runtime_error(ncclGetErrorString(r));
}
class ScheduledBaseline : public CollectiveBackend {
  std::shared_ptr<CollectiveBackend> native_;

 public:
  int calls = 0;
  explicit ScheduledBaseline(std::shared_ptr<CollectiveBackend> p)
      : native_(std::move(p)) {}
  void prepare() override { native_->prepare(); }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    native_->enqueue(op, input, output, count, stream);
    if (++calls <= 90)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  void synchronize() override { native_->synchronize(); }
  bool initialized() const override { return native_->initialized(); }
  bool idle() const override { return native_->idle(); }
};
struct View {
  PreparedRuntime& runtime;
  size_t plan;
  void enqueue(NetworkOperation, void const* a, void* b, size_t,
               cudaStream_t stream) {
    runtime.enqueue(plan, a, b, stream);
  }
  void synchronize() { runtime.synchronize(); }
};
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
    if (argc != 2 || ranks != 8 || localRanks != 4)
      throw std::invalid_argument("NATIVE_LIBRARY on two four-GPU hosts");
    int device = localRanks - 1 - localRank;
    check(cudaSetDevice(device));
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      ncclUniqueId id;
      if (!rank) nativeCheck(ncclGetUniqueId(&id));
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
      ncclComm_t comm;
      nativeCheck(ncclCommInitRank(&comm, ranks, id, rank));
      {
        RuntimeContext context;
        if (!rank) context.id = PreparedRuntime::createUniqueId();
        MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
        context.rank = rank;
        context.ranks = ranks;
        context.device = device;
        context.capacity = 1UL << 20;
        PreparationOptions options;
        options.dtype = ScalarType::Float32;
        options.fallback = uccl_lite::integration::makeNativeNcclFallback(
            context, comm, argv[1], "pool-demotion-test-only-schedule",
            ScalarType::Float32);
        auto baseline =
            std::make_shared<ScheduledBaseline>(options.fallback.backend);
        options.fallback.backend = baseline;
        options.fallback.identity += ";test-only-first90-delay20ms";
        PreparedRuntime runtime(context, options);
        CollectiveRequest request{NetworkOperation::AllReduce, 257, {}};
        for (int r = 0; r < ranks; ++r)
          request.layouts.push_back({bool(r & 1), 4});
        auto old = runtime.prepare(request);
        if (runtime.selection(old).backend != RuntimeBackend::DirectExchange ||
            baseline->calls != 30)
          throw std::runtime_error(
              "direct admission fixture/counter not reached");
        request.count = 64;
        auto added = runtime.prepare(request);
        auto const& changed = runtime.selection(old);
        if (changed.backend != RuntimeBackend::Caller ||
            changed.poolCandidateSamples.size() != 18 ||
            changed.poolBaselineSamples.size() != 18)
          throw std::runtime_error(
              "old common-runtime plan/report did not reflect pool demotion");
        if (qualifiedLocalBackend(changed.poolBaselineSamples,
                                  changed.poolCandidateSamples,
                                  true) == LocalBackend::Host)
          throw std::runtime_error(
              "demotion was not backed by failed current gain");
        bool resources =
            runtime.selection(added).backend != RuntimeBackend::Caller;
        if (runtime.acceleratedResourcesInitialized() != resources)
          throw std::runtime_error("demoted pool resources retained");
        View first{runtime, old}, second{runtime, added};
        exercise<uint32_t>(first, rank, ranks, 3, 257, bool(rank & 1), streams);
        exercise<uint32_t>(second, rank, ranks, 3, 64, bool(rank & 1), streams);
        if (!rank)
          std::cout << "DEMOTED lifetime=" << lifetime
                    << " old=" << int(changed.backend)
                    << " added=" << int(runtime.selection(added).backend)
                    << " baseline_calls=" << baseline->calls << '\n';
      }
      nativeCheck(ncclCommDestroy(comm));
    }
    for (auto stream : streams) check(cudaStreamDestroy(stream));
    if (!rank)
      std::cout << "PASS current-pool rejection updates old handles/reports; "
                   "input/output/guard correctness; two lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

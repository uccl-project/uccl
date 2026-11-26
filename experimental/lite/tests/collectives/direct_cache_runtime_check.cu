#include "native_fallback.hpp"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/network_cache.hpp"
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
              bool alias, std::array<cudaStream_t, 2> streams,
              size_t alignment = sizeof(Bits)) {
  size_t in = count * (op == 2 ? ranks : 1),
         out = count * (op == 1 ? ranks : 1), extent = std::max(in, out) + 128;
  size_t aOffset = alignment / sizeof(Bits) +
                   (alias && op == 1 ? size_t(rank) * count : 0),
         bOffset = alignment / sizeof(Bits) +
                   (alias && op == 2 ? size_t(rank) * count : 0);
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
class Baseline : public CollectiveBackend {
  std::shared_ptr<CollectiveBackend> impl_;
  int delay_;
  bool wrong_;
  size_t scalar_;

 public:
  int calls = 0;
  Baseline(std::shared_ptr<CollectiveBackend> p, int delay, bool wrong,
           size_t scalar)
      : impl_(std::move(p)), delay_(delay), wrong_(wrong), scalar_(scalar) {}
  void prepare() override { impl_->prepare(); }
  void enqueue(NetworkOperation op, void const* a, void* b, size_t n,
               cudaStream_t stream) override {
    ++calls;
    impl_->enqueue(op, a, b, n, stream);
    if (wrong_) check(cudaMemsetAsync(b, 0, n * scalar_, stream));
    if (delay_) std::this_thread::sleep_for(std::chrono::milliseconds(delay_));
  }
  void synchronize() override { impl_->synchronize(); }
  bool initialized() const override { return impl_->initialized(); }
  bool idle() const override { return impl_->idle(); }
};
struct View {
  PreparedNetworkRuntime& runtime;
  size_t plan;
  void enqueue(NetworkOperation, void const* a, void* b, size_t,
               cudaStream_t stream) {
    runtime.enqueue(plan, a, b, stream);
  }
  void synchronize() { runtime.synchronize(); }
};
template <class Bits>
void run(int rank, int ranks, int localRanks, int device, char const* library,
         std::string path, std::string mode) {
  auto dtype = sizeof(Bits) == 4 ? ScalarType::Float32 : ScalarType::BFloat16;
  ncclUniqueId id;
  if (!rank) nativeCheck(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  nativeCheck(ncclCommInitRank(&comm, ranks, id, rank));
  std::array<cudaStream_t, 2> streams;
  for (auto& stream : streams)
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (!rank && mode == "identity") {
    auto cache = loadNetworkPlanCache(path);
    if (!cache || cache->records.size() != 1)
      throw std::runtime_error("identity fixture needs one direct record");
    auto changed = cache->records[0];
    changed.configurationIdentities[0] += ";fixture-stale";
    updateNetworkPlanCache(path, *cache, changed);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  {
    BackendContext context;
    if (!rank) context.id = PreparedNetworkRuntime::createUniqueId();
    MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    context.rank = rank;
    context.ranks = ranks;
    context.device = device;
    context.capacity = 1UL << 20;
    int visible;
    check(cudaGetDeviceCount(&visible));
    if (visible == 1)
      context.localDevices = {0};
    else
      for (int i = 0; i < localRanks; ++i) context.localDevices.push_back(i);
    NetworkPreparationOptions options;
    options.dtype = dtype;
    options.allowHierarchy = false;
    options.planCache = path;
    options.fallbackOnly = mode == "fallback";
    options.fallback = uccl_lite::integration::makeNativeNcclFallback(
        context, comm, library,
        mode == "stale" ? "new-direct-cache-config" : "direct-cache-config",
        dtype);
    int delay = (mode == "cold" || mode == "warm" || mode == "identity" ||
                 mode == "layout" || mode == "stale")
                    ? 20
                    : 0;
    auto counter = std::make_shared<Baseline>(
        options.fallback.backend, delay, mode == "bad-baseline", sizeof(Bits));
    options.fallback.backend = counter;
    options.fallback.identity += ";test-only-scheduling-fixture";
    PreparedNetworkRuntime runtime(context, options);
    unsigned align = mode == "layout" ? 8 : sizeof(Bits);
    CollectiveRequest request{
        NetworkOperation::AllReduce, 257,
        std::vector<RankBufferLayout>(ranks, {false, align})};
    if (mode == "corrupt" || mode == "bad-baseline") {
      bool rejected = false;
      try {
        runtime.prepare(request);
      } catch (std::exception const&) {
        rejected = true;
      }
      if (!rejected || runtime.networkInitialized() ||
          counter->calls != (mode == "bad-baseline" ? 5 : 0))
        throw std::runtime_error(
            "invalid cache/baseline reached direct execution");
    } else {
      auto plan = runtime.prepare(request);
      auto const& p = runtime.selection(plan);
      auto const& d = p.direct;
      int calls = (mode == "warm"                            ? 10
                   : mode == "native-warm"                   ? 5
                   : mode == "fallback"                      ? 0
                   : (mode == "drift" || mode == "identity") ? 40
                                                             : 30);
      if (counter->calls != calls)
        throw std::runtime_error("unexpected direct cache native call count: " +
                                 std::to_string(counter->calls));
      bool cached = mode == "warm" || mode == "native-warm";
      if (d.fromCache != cached)
        throw std::runtime_error("direct cache hit state mismatch");
      bool winner = delay != 0;
      if (p.backend != (winner ? NetworkBackend::DirectExchange
                               : NetworkBackend::Caller) ||
          runtime.networkInitialized() != winner)
        throw std::runtime_error(
            "wrong direct cache winner/resource retention");
      if (mode != "fallback" &&
          (d.samples.size() != 18 || d.baselineSamples.size() != 18 ||
           d.configurationIdentities.size() != size_t(ranks)))
        throw std::runtime_error("original direct evidence lost");
      if (mode == "warm" && (d.validationSamples.size() != 6 ||
                             d.validationBaselineSamples.size() != 6))
        throw std::runtime_error("direct checkpoint is not ABBA");
      if (mode == "native-warm" && (!d.validationSamples.empty() ||
                                    d.validationBaselineSamples.size() != 3))
        throw std::runtime_error("cached caller constructed direct");
      if (mode == "drift" &&
          d.cacheStatus.find("performance drift") == std::string::npos)
        throw std::runtime_error("direct drift did not refresh");
      if (mode == "identity" &&
          d.cacheStatus.find("execution identity miss") == std::string::npos)
        throw std::runtime_error("stale execution identity reused");
      View view{runtime, plan};
      exercise<Bits>(view, rank, ranks, 3, 257, false, streams, align);
      if (!rank)
        std::cout << "DIRECT_CACHE mode=" << mode << " cached=" << d.fromCache
                  << " selected=" << int(p.backend)
                  << " validation=" << d.validationSamples.size()
                  << " status=" << d.cacheStatus << '\n';
    }
  }
  for (auto stream : streams) check(cudaStreamDestroy(stream));
  nativeCheck(ncclCommDestroy(comm));
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
    if (argc != 5)
      throw std::invalid_argument("NATIVE_LIBRARY CACHE_PATH fp32|bf16 MODE");
    int visible;
    check(cudaGetDeviceCount(&visible));
    int device = visible == 1 ? 0 : localRanks - 1 - localRank;
    check(cudaSetDevice(device));
    if (std::string(argv[3]) == "bf16")
      run<uint16_t>(rank, ranks, localRanks, device, argv[1], argv[2], argv[4]);
    else
      run<uint32_t>(rank, ranks, localRanks, device, argv[1], argv[2], argv[4]);
    if (!rank)
      std::cout << "PASS direct cache " << argv[4] << " dtype=" << argv[3]
                << " ranks=" << ranks
                << "; full buffers/guards and actual call-count validation\n";
  } catch (std::exception const& e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

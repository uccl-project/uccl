#include "native_fallback.hpp"
#include "scalar_oracle.cuh"
#include "uccl_lite/runtime/prepared_local.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <mpi.h>
using namespace uccl_lite;
using namespace uccl_lite::runtime;
void nativeCheck(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(status));
}
template <class Bits>
void run(int rank, int ranks, char** argv, ScalarType dtype) {
  std::string mode = argv[2];
  BackendContext context;
  context.rank = context.device = rank;
  context.ranks = ranks;
  context.capacity = 65536UL * ranks * sizeof(Bits);
  context.localDevices.resize(ranks);
  std::iota(context.localDevices.begin(), context.localDevices.end(), 0);
  if (!rank) {
    std::ifstream random("/dev/urandom", std::ios::binary);
    if (!random.read(reinterpret_cast<char*>(context.id.data()),
                     context.id.size()))
      throw std::runtime_error("session ID failed");
  }
  MPI_Bcast(context.id.data(), context.id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t native = nullptr;
  bool useNative = mode == "native" || mode == "fallback";
  if (useNative) {
    ncclUniqueId id;
    if (!rank) nativeCheck(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    nativeCheck(ncclCommInitRank(&native, ranks, id, rank));
  }
  {
    LocalPreparationOptions options;
    options.dtype = dtype;
    options.probeResources = mode == "prepared";
    options.allowPeerPacket = mode != "generic";
    options.fallbackOnly = mode == "fallback";
    if (useNative)
      options.fallback = integration::makeNativeNcclFallback(
          context, native, argv[3], "scalar-default", dtype);
    PreparedLocalRuntime runtime(context, options);
    LocalRequest invalid{
        NetworkOperation::AllReduce, 64,
        std::vector<RankBufferLayout>(ranks, {false, unsigned(sizeof(Bits))})};
    if (dtype == ScalarType::Float32) {
      invalid.layouts.back().alignment = 2;
      bool rejected = false;
      try {
        runtime.prepare(invalid);
      } catch (std::invalid_argument const&) {
        rejected = true;
      }
      if (!rejected || runtime.peerInitialized() ||
          runtime.peerPacketInitialized() || runtime.hostInitialized())
        throw std::runtime_error(
            "invalid FP32 alignment initialized a backend");
    }
    if (useNative) {
      auto mismatch = options;
      mismatch.dtype = dtype == ScalarType::Float32 ? ScalarType::BFloat16
                                                    : ScalarType::Float32;
      auto other = context;
      other.id[0] ^= 0x65;
      PreparedLocalRuntime wrong(other, mismatch);
      invalid.layouts.assign(ranks, {false, 256});
      bool rejected = false;
      try {
        wrong.prepare(invalid);
      } catch (std::invalid_argument const&) {
        rejected = true;
      }
      if (!rejected || wrong.peerInitialized() || wrong.hostInitialized() ||
          options.fallback.backend->initialized())
        throw std::runtime_error(
            "fallback dtype mismatch was not rejected before initialization");
    }
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      scalar_test::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<size_t> plans;
    bool needPeer = false, needPacket = false, needTree = false,
         sawPacket = false;
    for (size_t count : {64UL, 257UL, 65536UL})
      for (int layout = 0; layout < 2; ++layout)
        for (int op = 1; op <= 3; ++op) {
          LocalRequest request{NetworkOperation(op), count, {}};
          for (int r = 0; r < ranks; ++r)
            request.layouts.push_back(
                {layout && (count == 65536 || (r & 1)),
                 unsigned(count == 257 ? sizeof(Bits)
                                       : sizeof(Bits) * (1u << (r % 3)))});
          size_t plan = runtime.prepare(request);
          plans.push_back(plan);
          if (runtime.prepare(request) != plan)
            throw std::runtime_error("typed live cache failed");
          auto const& choice = runtime.selection(plan);
          if (choice.hostEligible || runtime.hostInitialized())
            throw std::runtime_error(
                "BF16 host candidate admitted for another dtype");
          needPeer |= choice.backend == LocalBackend::Peer;
          needPacket |= choice.backend == LocalBackend::PeerPacket;
          needTree |= choice.backend == LocalBackend::PeerTree;
          if (runtime.peerInitialized() != (useNative ? needPeer : true) ||
              runtime.peerPacketInitialized() != needPacket ||
              runtime.peerTreeInitialized() != needTree)
            throw std::runtime_error("typed resource lifetime mismatch");
          sawPacket |= choice.peerPacketEligible;
          if (choice.peerPacketEligible && !choice.candidateCorrect[2])
            throw std::runtime_error(
                "retained FP32 peer candidate failed oracle");
          if (useNative) {
            std::array<std::vector<double>, 4> samples{
                {choice.samples[0], choice.samples[1], choice.samples[3],
                 choice.samples[4]}};
            if (choice.backend !=
                qualifiedFallbackAlternative(choice.fallbackSamples, samples,
                                             choice.candidateCorrect))
              throw std::runtime_error(
                  "typed fallback admission differs from evidence");
          } else {
            auto expected =
                choice.peerPacketEligible &&
                        qualifiedLocalBackend(
                            choice.samples[0], choice.samples[3],
                            choice.candidateCorrect[2]) == LocalBackend::Host
                    ? LocalBackend::PeerPacket
                    : LocalBackend::Peer;
            if (choice.backend != expected)
              throw std::runtime_error(
                  "typed packet admission differs from evidence");
          }
          if (!rank) {
            std::cout << "dtype=" << int(dtype) << " op=" << op
                      << " count=" << count << " layout=" << layout
                      << " selected=" << int(choice.backend)
                      << " packet_eligible=" << choice.peerPacketEligible
                      << " topology=" << choice.topologyIdentity
                      << " executor=" << choice.executorIdentity;
            for (size_t candidate = 0; candidate < kLocalAlternatives.size();
                 ++candidate) {
              std::cout << " candidate" << int(kLocalAlternatives[candidate])
                        << "_us=";
              for (double us :
                   choice.samples[size_t(kLocalAlternatives[candidate])])
                std::cout << us << ',';
              std::cout << " baseline" << candidate << "_us=";
              for (double us : choice.fallbackSamples[candidate])
                std::cout << us << ',';
            }
            std::cout << std::endl;
          }
          auto l = request.layouts[rank];
          scalar_test::exercise<Bits>(
              op, count, l.inPlace, l.alignment, rank, ranks, streams,
              [&](int, void const* in, void* out, size_t, cudaStream_t stream) {
                runtime.enqueue(plan, in, out, stream);
              });
        }
    if (mode == "packet" && !sawPacket)
      throw std::runtime_error(
          "expected retained FP32 candidate was not considered");
    for (size_t plan : plans) {
      auto const& p = runtime.selection(plan).request;
      auto l = p.layouts[rank];
      scalar_test::exercise<Bits>(
          int(p.operation), p.count, l.inPlace, l.alignment, rank, ranks,
          streams,
          [&](int, void const* in, void* out, size_t, cudaStream_t stream) {
            runtime.enqueue(plan, in, out, stream);
          });
    }
    runtime.synchronize();
    for (auto stream : streams) scalar_test::check(cudaStreamDestroy(stream));
  }
  if (native) nativeCheck(ncclCommDestroy(native));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 4)
      throw std::invalid_argument(
          "fp32|fp16 prepared|generic|packet|native|fallback NATIVE_LIBRARY");
    std::string type = argv[1], mode = argv[2];
    if (mode != "prepared" && mode != "generic" && mode != "packet" &&
        mode != "native" && mode != "fallback")
      throw std::invalid_argument("invalid scalar test mode");
    scalar_test::check(cudaSetDevice(rank));
    if (type == "fp32")
      run<uint32_t>(rank, ranks, argv, ScalarType::Float32);
    else if (type == "fp16" && (mode == "native" || mode == "fallback"))
      run<uint16_t>(rank, ranks, argv, ScalarType::Float16);
    else
      throw std::invalid_argument("unsupported scalar test dtype/mode");
    if (!rank)
      std::cout << "PASS: scalar prepared runtime " << type << ' ' << mode
                << " ranks=" << ranks
                << "; bitwise AG, numerical reductions, aliases, tails and old "
                   "plans\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

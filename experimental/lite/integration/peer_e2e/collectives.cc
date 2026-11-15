// Benchmark-only, single-process two-GPU bridge. No automatic backend policy.
#include "uccl_lite/collective_dispatch.hpp"
#include "uccl_lite/collectives/peer_collectives.hpp"
#include <array>
#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <cuda_runtime.h>
#include <nccl.h>

namespace {
void cudaCheck(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void ncclCheck(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
thread_local char error[1024];
struct Context {
  std::unique_ptr<uccl_lite::PeerCollectives> peer;
  std::array<ncclComm_t, 2> comm{};
  std::set<size_t> prepared;
  bool usePeer;
  explicit Context(bool value) : usePeer(value) {
    if (usePeer) {
      uccl_lite::Topology topology;
      topology.devices = {0, 1};
      for (int r = 0; r < 2; ++r)
        for (int s = 0; s < 2; ++s) {
          int access = 1;
          if (r != s) cudaCheck(cudaDeviceCanAccessPeer(&access, r, s));
          topology.peerAccess.push_back(access);
        }
      peer = std::make_unique<uccl_lite::PeerCollectives>(topology);
    } else {
      int devices[] = {0, 1};
      ncclCheck(ncclCommInitAll(comm.data(), 2, devices));
    }
  }
  ~Context() {
    for (auto c : comm)
      if (c) ncclCommDestroy(c);
  }
  uccl_lite::CollectiveCall call(int op, size_t count, void** input,
                                 void** output, void** streams) {
    if ((op != 1 && op != 2) || !count || count > (32UL << 20))
      throw std::invalid_argument(
          "expected positive AG/RS, at most 128 MiB complete");
    uccl_lite::CollectiveCall call;
    call.collective = op == 1 ? uccl_lite::Collective::AllGather
                              : uccl_lite::Collective::ReduceScatter;
    call.count = count;
    for (int r = 0; r < 2; ++r) {
      if (!input[r] || !output[r] || input[r] == output[r])
        throw std::invalid_argument("expected out-of-place device buffers");
      cudaCheck(cudaSetDevice(r));
      cudaStreamCaptureStatus capture;
      cudaCheck(cudaStreamIsCapturing(static_cast<cudaStream_t>(streams[r]),
                                      &capture));
      if (capture != cudaStreamCaptureStatusNone)
        throw std::invalid_argument(
            "experimental adapter supports eager execution only");
      cudaPointerAttributes in{}, out{};
      cudaCheck(cudaPointerGetAttributes(&in, input[r]));
      cudaCheck(cudaPointerGetAttributes(&out, output[r]));
      if (in.type != cudaMemoryTypeDevice || out.type != cudaMemoryTypeDevice ||
          in.device != r || out.device != r)
        throw std::invalid_argument(
            "buffers must reside on their corresponding GPU");
      call.ports.push_back(
          {input[r], output[r], static_cast<cudaStream_t>(streams[r]), r});
    }
    return call;
  }
};
template <class Function>
int guarded(Function fn) {
  try {
    fn();
    return 0;
  } catch (std::exception const& e) {
    std::snprintf(error, sizeof(error), "%s", e.what());
    return -1;
  }
}
}  // namespace
extern "C" {
char const* uccl_lite_e2e_error() { return error; }
void* uccl_lite_e2e_create(int peer) {
  Context* result = nullptr;
  guarded([&] { result = new Context(peer != 0); });
  return result;
}
void uccl_lite_e2e_destroy(void* context) {
  delete static_cast<Context*>(context);
}
int uccl_lite_e2e_version() {
  int version = 0;
  ncclGetVersion(&version);
  return version;
}
int uccl_lite_e2e_prepare(void* context, size_t count, void** input,
                          void** output, void** streams) {
  return guarded([&] {
    auto& c = *static_cast<Context*>(context);
    auto call = c.call(2, count, input, output, streams);
    for (int r = 0; r < 2; ++r) {
      cudaCheck(cudaSetDevice(r));
      cudaCheck(cudaDeviceSynchronize());
    }
    if (c.usePeer) c.peer->prepare(call);
    c.prepared.insert(count);
  });
}
int uccl_lite_e2e_enqueue(void* context, int op, size_t count, void** input,
                          void** output, void** streams) {
  return guarded([&] {
    auto& c = *static_cast<Context*>(context);
    auto call = c.call(op, count, input, output, streams);
    if (!c.prepared.count(count))
      throw std::logic_error("prepare exact extent before enqueue");
    if (c.usePeer)
      c.peer->enqueue(call);
    else {
      ncclCheck(ncclGroupStart());
      ncclResult_t result = ncclSuccess;
      for (int r = 0; r < 2; ++r) {
        cudaCheck(cudaSetDevice(r));
        auto next =
            op == 1
                ? ncclAllGather(input[r], output[r], count, ncclBfloat16,
                                c.comm[r], call.ports[r].stream)
                : ncclReduceScatter(input[r], output[r], count, ncclBfloat16,
                                    ncclSum, c.comm[r], call.ports[r].stream);
        if (next != ncclSuccess) result = next;
      }
      auto end = ncclGroupEnd();
      ncclCheck(result);
      ncclCheck(end);
    }
  });
}
}

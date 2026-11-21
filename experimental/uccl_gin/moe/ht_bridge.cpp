// SPDX-License-Identifier: Apache-2.0
// Typed bridge for the qualified NCCL-EP tensor-array ABI (NCCL 2.30.4).
#include "device/ht_scheduler.hpp"
#include "nccl_ep.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(call)                                                           \
  do {                                                                     \
    auto status = (call);                                                  \
    if (status != cudaSuccess) {                                           \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
      std::abort();                                                        \
    }                                                                      \
  } while (0)
#define NC(call)                                                           \
  do {                                                                     \
    auto status = (call);                                                  \
    if (status != ncclSuccess) {                                           \
      std::fprintf(stderr, "%s: %s\n", #call, ncclGetErrorString(status)); \
      std::abort();                                                        \
    }                                                                      \
  } while (0)
#define REQUIRE(condition)                              \
  do {                                                  \
    if (!(condition)) {                                 \
      std::fprintf(stderr, "failed: %s\n", #condition); \
      std::abort();                                     \
    }                                                   \
  } while (0)

std::atomic<uint64_t> active_allocations{0}, total_allocations{0};

cudaError_t allocate(void** data, size_t bytes) {
  auto status = cudaMalloc(data, bytes);
  if (status == cudaSuccess && *data) {
    ++active_allocations;
    ++total_allocations;
  }
  return status;
}
cudaError_t release(void* data) {
  auto status = cudaFree(data);
  if (status == cudaSuccess && data) --active_allocations;
  return status;
}

struct Bridge {
  ncclComm_t comm;
  ncclEpGroup_t group;
  ncclEpHandle_t handle = nullptr;
  ncclNDTensor_t routing = nullptr;
  cudaStream_t stream;
  nccl_ep::hybridep::LaunchPolicy policy;
  int device, rank, world, local_ranks, hidden, experts, maximum, tokens = 0,
                                                                  topk = 0;
  unsigned received = 0;
  bool dispatched = false;
  uint64_t dispatch_calls = 0, combine_calls = 0;

  Bridge(int device, int rank, int world, int local_ranks, int hidden,
         int experts, int maximum, int stages, int blocks, void const* unique,
         void* cuda_stream)
      : stream(static_cast<cudaStream_t>(cuda_stream)),
        policy{stages, blocks},
        device(device),
        rank(rank),
        world(world),
        local_ranks(local_ranks),
        hidden(hidden),
        experts(experts),
        maximum(maximum) {
    REQUIRE(experts % world == 0 && world % local_ranks == 0 &&
            maximum <= 8192);
    CK(cudaSetDevice(device));
    REQUIRE(nccl_ep::hybridep::set_launch_policy(policy));
    ncclUniqueId id;
    std::memcpy(&id, unique, sizeof(id));
    NC(ncclCommInitRank(&comm, world, id, rank));
    ncclEpGroupConfig_t config{};
    config.version = 1;
    config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    config.num_experts = experts;
    config.max_tokens_per_rank = maximum;
    config.token_size_bytes = hidden * 2;
    NC(ncclEpCreateGroup(&group, comm, &config, stream, allocate, release));
  }
  ~Bridge() {
    REQUIRE(!handle && !routing && !dispatched);
    CK(cudaSetDevice(device));
    CK(cudaStreamSynchronize(stream));
    NC(ncclEpGroupDestroy(group, stream));
    NC(ncclCommDestroy(comm));
  }
  ncclNDTensor_t tensor(void* data, ncclDataType_t type, ncclEpTensorTag_t tag,
                        int rows, int columns) {
    ncclNDTensor_t result;
    NC(ncclEpTensorCreate(group, &result, 2, type, tag, data, rows, columns));
    return result;
  }
  unsigned prepare(void* indices, int count, int selected) {
    REQUIRE(!handle && !routing && !dispatched && count > 0 &&
            count <= maximum);
    CK(cudaSetDevice(device));
    REQUIRE(nccl_ep::hybridep::set_launch_policy(policy));
    tokens = count;
    topk = selected;
    routing =
        tensor(indices, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, tokens, topk);
    NC(ncclEpCreateHandle(&handle, group, routing, nullptr, 0, nullptr, stream,
                          false));
    CK(cudaStreamSynchronize(stream));
    NC(ncclEpHandleGetNumRecvTokens(handle, &received));
    return received;
  }
  void dispatch(void* input, void* probability, void* output,
                void* out_probability, void* out_indices) {
    REQUIRE(handle && routing && !dispatched);
    REQUIRE(nccl_ep::hybridep::set_launch_policy(policy));
    ncclNDTensor_t inputs[] = {
        tensor(input, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, tokens, hidden),
        tensor(probability, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
               tokens, topk),
        routing};
    ncclNDTensor_t outputs[] = {
        tensor(output, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, received,
               hidden),
        tensor(out_probability, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
               received, topk),
        tensor(out_indices, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, received,
               topk)};
    ncclEpDispatchConfig_t config{};
    NC(ncclEpDispatch(handle, inputs, 3, outputs, 3, nullptr, 0, false, &config,
                      stream));
    NC(ncclEpComplete(handle, nullptr, stream));
    for (int i = 0; i < 2; ++i) NC(ncclEpTensorDestroy(group, inputs[i]));
    for (auto t : outputs) NC(ncclEpTensorDestroy(group, t));
    dispatched = true;
    ++dispatch_calls;
  }
  void combine(void* expert_output, void* output) {
    REQUIRE(handle && routing && dispatched);
    REQUIRE(nccl_ep::hybridep::set_launch_policy(policy));
    // Forward inference has already applied expert probabilities locally.
    ncclNDTensor_t input = tensor(expert_output, ncclBfloat16,
                                  NCCL_EP_TENSOR_TAG_TOKENS, received, hidden);
    ncclNDTensor_t result =
        tensor(output, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, tokens, hidden);
    NC(ncclEpCombine(handle, &input, 1, &result, 1, nullptr, 0, false, nullptr,
                     stream));
    NC(ncclEpComplete(handle, nullptr, stream));
    CK(cudaStreamSynchronize(stream));
    NC(ncclEpTensorDestroy(group, input));
    NC(ncclEpTensorDestroy(group, result));
    NC(ncclEpHandleDestroy(handle));
    NC(ncclEpTensorDestroy(group, routing));
    handle = nullptr;
    routing = nullptr;
    dispatched = false;
    ++combine_calls;
  }
};

extern "C" {
int ht_bridge_abi_version() { return 2; }
uint64_t ht_bridge_active_allocations() { return active_allocations; }
unsigned ht_bridge_unique_id(void* output) {
  ncclUniqueId id;
  NC(ncclGetUniqueId(&id));
  std::memcpy(output, &id, sizeof(id));
  return sizeof(id);
}
void* ht_bridge_create(int device, int rank, int world, int local_ranks,
                       int hidden, int experts, int maximum, int stages,
                       int blocks, void const* unique, void* stream) {
  return new Bridge(device, rank, world, local_ranks, hidden, experts, maximum,
                    stages, blocks, unique, stream);
}
void ht_bridge_destroy(void* context) { delete static_cast<Bridge*>(context); }
unsigned ht_bridge_prepare(void* context, void* indices, int tokens, int topk) {
  return static_cast<Bridge*>(context)->prepare(indices, tokens, topk);
}
void ht_bridge_dispatch(void* context, void* input, void* probability,
                        void* output, void* out_probability,
                        void* out_indices) {
  static_cast<Bridge*>(context)->dispatch(input, probability, output,
                                          out_probability, out_indices);
}
void ht_bridge_combine(void* context, void* expert_output, void* output) {
  static_cast<Bridge*>(context)->combine(expert_output, output);
}
void ht_bridge_resources(void* context,
                         nccl_ep::hybridep::ScheduleResources* resources) {
  auto& bridge = *static_cast<Bridge*>(context);
  REQUIRE(nccl_ep::hybridep::set_launch_policy(bridge.policy));
  nccl_ep::hybridep::query_schedule_resources(
      bridge.hidden, bridge.experts / bridge.world, bridge.local_ranks,
      bridge.world / bridge.local_ranks, *resources, false);
}
void ht_bridge_stats(void* context, uint64_t* output) {
  auto& bridge = *static_cast<Bridge*>(context);
  output[0] = bridge.dispatch_calls;
  output[1] = bridge.combine_calls;
  output[2] = bool(bridge.handle);
  output[3] = bridge.dispatched;
  output[4] = active_allocations;
  output[5] = total_allocations;
}
void ht_bridge_versions(int* output) {
  CK(cudaRuntimeGetVersion(output));
  NC(ncclGetVersion(output + 1));
}
}

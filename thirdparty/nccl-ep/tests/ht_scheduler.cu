// SPDX-License-Identifier: Apache-2.0
// Real HT communication with a logical queue and a bounded, independently owned pool.
#include "nccl_ep.h"
#include "device/ht_scheduler.hpp"
#include "traffic_staging.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call) do { auto cuda_status = (call); if (cuda_status != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(cuda_status)); MPI_Abort(MPI_COMM_WORLD, 2); \
} } while (0)
#define NCCL_CHECK(call) do { auto nccl_status = (call); if (nccl_status != ncclSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, ncclGetErrorString(nccl_status)); MPI_Abort(MPI_COMM_WORLD, 3); \
} } while (0)
#define REQUIRE(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "failed: %s\n", #condition); MPI_Abort(MPI_COMM_WORLD, 4); \
} } while (0)

constexpr int hidden = 7168;
constexpr int topk = 2;
namespace traffic = nccl_ep::hybridep::traffic;
struct BalancePolicy { int passes, active, ctas; };

std::vector<BalancePolicy> parse_balance(char const* value) {
  std::vector<BalancePolicy> policies;
  char const* end = value + std::strlen(value);
  while (value != end) {
    BalancePolicy policy;
    int* fields[] = {&policy.passes, &policy.active, &policy.ctas};
    for (int field = 0; field < 3; ++field) {
      auto parsed = std::from_chars(value, end, *fields[field]);
      if (parsed.ec != std::errc{} || *fields[field] < 1) std::exit(2);
      value = parsed.ptr;
      if (field < 2) { if (value == end || *value++ != ':') std::exit(2); }
    }
    if (value != end && (*value++ != ',' || value == end)) std::exit(2);
    if (policy.passes > 16 || policy.active > 32 || policy.ctas > 4096 ||
        std::any_of(policies.begin(), policies.end(), [&](auto p) { return p.passes == policy.passes; })) std::exit(2);
    policies.push_back(policy);
  }
  if (policies.empty()) std::exit(2);
  return policies;
}

struct Options {
  int slots = 1, tokens = 4096, active = 16, stages = 0, blocks = 16, repeats = 10;
  int expert_ctas = 16, expert_passes = 1;
  int resident_groups = 0;
  std::vector<BalancePolicy> balance;
  traffic::Path expert_path = traffic::Path::Copy;
  std::string output;
  bool diagnostic = false, sweep = false, shape_stress = false, check_invalid_combine = false;
  bool nonuniform_weights = false;
  bool phase_events = false;
  bool traffic_sweep = false;
};

Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string name = argv[i];
    if (name == "--diagnostic") { options.diagnostic = true; continue; }
    if (name == "--scale-sweep") { options.sweep = true; continue; }
    if (name == "--shape-stress") { options.shape_stress = true; continue; }
    if (name == "--check-invalid-combine") { options.check_invalid_combine = true; continue; }
    if (name == "--nonuniform-weights") { options.nonuniform_weights = true; continue; }
    if (name == "--phase-events") { options.phase_events = true; continue; }
    if (name == "--traffic-sweep") { options.traffic_sweep = true; continue; }
    if (i + 1 == argc) std::exit(2);
    char const* value = argv[++i];
    if (name == "--output") { options.output = value; continue; }
    if (name == "--balance-policy") { options.balance = parse_balance(value); continue; }
    if (name == "--expert-path") {
      bool found = false;
      for (auto path : {traffic::Path::Copy, traffic::Path::Direct, traffic::Path::Tma,
                        traffic::Path::TensorInput, traffic::Path::TensorOutput})
        if (std::strcmp(value, traffic::name(path)) == 0) { options.expert_path = path; found = true; }
      if (!found) std::exit(2);
      continue;
    }
    int number;
    auto parsed = std::from_chars(value, value + std::strlen(value), number);
    if (parsed.ec != std::errc{} || *parsed.ptr != '\0' || number < 0) std::exit(2);
    if (name == "--slots") options.slots = number;
    else if (name == "--tokens") options.tokens = number;
    else if (name == "--active") options.active = number;
    else if (name == "--stages") options.stages = number;
    else if (name == "--blocks") options.blocks = number;
    else if (name == "--repeats") options.repeats = number;
    else if (name == "--expert-ctas") options.expert_ctas = number;
    else if (name == "--expert-passes") options.expert_passes = number;
    else if (name == "--resident-groups") options.resident_groups = number;
    else std::exit(2);
  }
  if (options.slots < 1 || options.slots > 64 || options.tokens < 1 || options.tokens > 8192 ||
      options.active < 1 || options.active > 32 || options.repeats < 1 || options.repeats > 1000)
    std::exit(2);
  if (options.expert_ctas < 1 || options.expert_ctas > 4096 ||
      options.expert_passes < 1 || options.expert_passes > 16 || options.resident_groups > 32 ||
      (options.traffic_sweep && options.shape_stress)) std::exit(2);
  if (!nccl_ep::hybridep::set_launch_policy({options.stages, options.blocks})) std::exit(2);
  return options;
}

std::map<void*, size_t> allocations;
std::mutex allocation_mutex;
std::atomic<unsigned> progress{0};
size_t allocated_bytes = 0, peak_bytes = 0;
cudaError_t allocate(void** pointer, size_t bytes) {
  auto result = cudaMalloc(pointer, bytes);
  if (result == cudaSuccess) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    allocations[*pointer] = bytes;
    allocated_bytes += bytes;
    peak_bytes = std::max(peak_bytes, allocated_bytes);
  }
  return result;
}
cudaError_t release(void* pointer) {
  auto result = cudaFree(pointer);
  if (result == cudaSuccess) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    auto entry = allocations.find(pointer);
    if (entry == allocations.end()) std::abort();
    allocated_bytes -= entry->second;
    allocations.erase(entry);
  }
  return result;
}

__host__ __device__ unsigned value(int source, int slot, int epoch, int token, int col) {
  if (col == 0) return source + 1;
  if (col == 1) return slot + 1;
  if (col == 2) return epoch % 128 + 1;
  if (col == 3) return token % 128 + 1;
  if (col == 4) return token / 128 + 1;
  if (col == 5) return epoch / 128 + 1;
  unsigned hash = (source + 1) * 1000003u ^ (slot + 1) * 7919u ^ epoch * 104729u ^
                  (token + 1) * 15485863u ^ col * 2654435761u;
  return hash % 127u + 1u;
}
__global__ void produce(__nv_bfloat16* data, int tokens, int rank, int slot, int epoch) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size_t(tokens) * hidden;
       i += size_t(gridDim.x) * blockDim.x)
    data[i] = __float2bfloat16(float(value(rank, slot, epoch, i / hidden, i % hidden)));
}
__global__ void check_values(__nv_bfloat16 const* data, int tokens, int rank, int slot,
                             int epoch, int factor, unsigned long long* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size_t(tokens) * hidden;
       i += size_t(gridDim.x) * blockDim.x)
    if (__bfloat162float(data[i]) != float(factor * value(rank, slot, epoch, i / hidden, i % hidden)))
      atomicAdd(errors, 1ull);
}
__host__ __device__ float probability_value(int source, int worker, int slot, int epoch,
                                           int token, int k, bool nonuniform) {
  if (!nonuniform) return 0.5f;
  unsigned hash = (source + 1) * 65521u ^ (worker + 1) * 101u ^ slot * 103u ^
                  epoch * 107u ^ token * 109u;
  float first = float(hash % 15u + 1u) / 32.0f;
  return k ? 1.0f - first : first;
}
__global__ void produce_weights(float* weights, int tokens, int rank, int worker,
                                int slot, int epoch) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < tokens * topk)
    weights[i] = probability_value(rank, worker, slot, epoch, i / topk, i % topk, true);
}
__global__ void check_combined_weights(float const* weights, int count, int rank,
                                      int worker, int slot, int epoch, bool nonuniform,
                                      unsigned long long* errors) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && weights[i] != probability_value(rank, worker, slot, epoch,
                                                  i / topk, i % topk, nonuniform))
    atomicAdd(errors, 1ull);
}
__global__ void check_dispatch(__nv_bfloat16 const* data, int64_t const* indices,
    float const* weights, int tokens, int rank, int world, int worker, int slot, int epoch,
    bool nonuniform, int* seen, unsigned long long* errors) {
  int row = blockIdx.x;
  __shared__ int source, token, bad;
  if (threadIdx.x == 0) {
    source = int(__bfloat162float(data[size_t(row) * hidden])) - 1;
    token = int(__bfloat162float(data[size_t(row) * hidden + 3])) - 1 +
            128 * (int(__bfloat162float(data[size_t(row) * hidden + 4])) - 1);
    bad = source < 0 || source >= world || token < 0 || token >= tokens ||
          (source != rank && (source + 1) % world != rank);
    if (bad) atomicAdd(errors, 1ull);
    else {
      atomicAdd(seen + source * tokens + token, 1);
      int first = (token + worker + source) % 4;
      int second = (first + 1) % 4;
      if (world == 1) {
        float first_weight = probability_value(source, worker, slot, epoch, token,
                                               first < second ? 0 : 1, nonuniform);
        float second_weight = probability_value(source, worker, slot, epoch, token,
                                                first < second ? 1 : 0, nonuniform);
        if (indices[row * 2] != min(first, second) || indices[row * 2 + 1] != max(first, second) ||
            weights[row * 2] != first_weight || weights[row * 2 + 1] != second_weight) atomicAdd(errors, 1ull);
      } else {
        int selected = source == rank ? first : second;
        float selected_weight = probability_value(source, worker, slot, epoch, token,
                                                   source == rank ? 0 : 1, nonuniform);
        if (indices[row * 2] != selected || indices[row * 2 + 1] != -1 ||
            weights[row * 2] != selected_weight || weights[row * 2 + 1] != 0.0f) atomicAdd(errors, 1ull);
      }
    }
  }
  __syncthreads();
  if (!bad) for (int col = threadIdx.x; col < hidden; col += blockDim.x)
    if (__bfloat162float(data[size_t(row) * hidden + col]) !=
        float(value(source, slot, epoch, token, col))) atomicAdd(errors, 1ull);
}
__global__ void check_seen(int const* seen, int tokens, int rank, int world,
                           unsigned long long* errors) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < world * tokens) {
    int source = i / tokens;
    int expected = source == rank || (source + 1) % world == rank;
    if (seen[i] != expected) atomicAdd(errors, 1ull);
  }
}

void* data(ncclNDTensor_t tensor) {
  void* pointer;
  NCCL_CHECK(ncclEpTensorGetData(tensor, &pointer));
  return pointer;
}
struct Worker {
  ncclComm_t comm;
  ncclEpGroup_t group;
  ncclEpHandle_t handle;
  cudaStream_t stream;
  cudaEvent_t done;
  cudaEvent_t phase_events[4];
  ncclNDTensor_t input, routing, probability, output, output_probability, output_routing;
  ncclNDTensor_t experts, combined, combine_probability;
  ncclNDTensor_t invalid_probability = nullptr;
  int* seen;
  unsigned long long* errors;
  int id, tokens, received, slot = 0, epoch = 0, completed = 0;
  bool inflight = false;

  bool diagnostic, check_invalid_combine, nonuniform_weights;
  bool measure_phases;
  traffic::Path expert_path;
  int expert_ctas, expert_passes;
  int invalid_rejections = 0;
  Worker(int worker, int count, int maximum, int rank, int world, bool trace, bool check_invalid,
         bool nonuniform, Options const& options)
      : id(worker), tokens(count), diagnostic(trace), check_invalid_combine(check_invalid),
        nonuniform_weights(nonuniform), measure_phases(options.phase_events),
        expert_path(options.expert_path), expert_ctas(options.expert_ctas), expert_passes(options.expert_passes) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    if (measure_phases) for (auto& event : phase_events) CUDA_CHECK(cudaEventCreate(&event));
    ncclUniqueId unique;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&unique));
    REQUIRE(MPI_Bcast(&unique, sizeof(unique), MPI_BYTE, 0, MPI_COMM_WORLD) == MPI_SUCCESS);
    ++progress;
    NCCL_CHECK(ncclCommInitRank(&comm, world, unique, rank));
    ++progress;
    ncclEpGroupConfig_t config{};
    config.version = 1; config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    config.num_experts = 4 * world; config.max_tokens_per_rank = maximum;
    config.token_size_bytes = hidden * 2;
    NCCL_CHECK(ncclEpCreateGroup(&group, comm, &config, stream, allocate, release));
    ++progress;
    initialize(rank, world);
  }

  void initialize(int rank, int world) {
    auto tensor = [&](ncclNDTensor_t& result, ncclDataType_t type, ncclEpTensorTag_t tag, int rows, int cols) {
      NCCL_CHECK(ncclEpTensorCreate(group, &result, 2, type, tag, nullptr, rows, cols));
    };
    tensor(routing, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, tokens, topk);
    std::vector<int64_t> indices(size_t(tokens) * topk);
    for (int token = 0; token < tokens; ++token) {
      int first = (token + id + rank) % 4;
      indices[token * 2] = rank * 4 + first;
      indices[token * 2 + 1] = ((rank + 1) % world) * 4 + (first + 1) % 4;
    }
    CUDA_CHECK(cudaMemcpyAsync(data(routing), indices.data(), indices.size() * 8,
                                cudaMemcpyHostToDevice, stream));
    NCCL_CHECK(ncclEpCreateHandle(&handle, group, routing, nullptr, 0, nullptr, stream, false));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ++progress;
    unsigned rows;
    NCCL_CHECK(ncclEpHandleGetNumRecvTokens(handle, &rows));
    received = rows;
    REQUIRE(received == (world == 1 ? tokens : 2 * tokens));
    tensor(input, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, tokens, hidden);
    tensor(probability, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, tokens, topk);
    tensor(output, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, received, hidden);
    tensor(output_probability, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, received, topk);
    if (check_invalid_combine)
      NCCL_CHECK(ncclEpTensorCreate(group, &invalid_probability, 2, ncclFloat32,
          NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, data(output_probability), received, topk + 1));
    tensor(output_routing, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, received, topk);
    tensor(experts, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, received, hidden);
    tensor(combined, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, tokens, hidden);
    tensor(combine_probability, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, tokens, topk);
    std::vector<float> weights(size_t(tokens) * topk, 0.5f);
    CUDA_CHECK(cudaMemcpyAsync(data(probability), weights.data(), weights.size() * 4,
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(allocate(reinterpret_cast<void**>(&seen), size_t(world) * tokens * 4));
    CUDA_CHECK(allocate(reinterpret_cast<void**>(&errors), sizeof(*errors)));
    CUDA_CHECK(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void enqueue(int rank, int request, int generation) {
    REQUIRE(!inflight);
    inflight = true;
    slot = request; epoch = generation;
    CUDA_CHECK(cudaMemsetAsync(data(output), 0xa5, size_t(received) * hidden * 2, stream));
    CUDA_CHECK(cudaMemsetAsync(data(combined), 0xa5, size_t(tokens) * hidden * 2, stream));
    CUDA_CHECK(cudaMemsetAsync(data(output_routing), 0xa5, size_t(received) * topk * 8, stream));
    CUDA_CHECK(cudaMemsetAsync(data(output_probability), 0xa5, size_t(received) * topk * 4, stream));
    CUDA_CHECK(cudaMemsetAsync(data(combine_probability), 0xa5, size_t(tokens) * topk * 4, stream));
    produce<<<32, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(data(input)), tokens, rank, slot, epoch);
    if (nonuniform_weights)
      produce_weights<<<(tokens * topk + 255) / 256, 256, 0, stream>>>(
          static_cast<float*>(data(probability)), tokens, rank, id, slot, epoch);
    ncclNDTensor_t inputs[] = {input, probability, routing};
    ncclNDTensor_t outputs[] = {output, output_probability, output_routing};
    ncclEpDispatchConfig_t dispatch{};
    if (measure_phases) CUDA_CHECK(cudaEventRecord(phase_events[0], stream));
    if (diagnostic) { std::printf("PHASE rank=%d slot=%d epoch=%d dispatch_enter\n", rank, slot, epoch); std::fflush(stdout); }
    NCCL_CHECK(ncclEpDispatch(handle, inputs, 3, outputs, 3, nullptr, 0, false, &dispatch, stream));
    NCCL_CHECK(ncclEpComplete(handle, nullptr, stream));
    if (diagnostic) { std::printf("PHASE rank=%d slot=%d epoch=%d dispatch_return\n", rank, slot, epoch); std::fflush(stdout); }
    ++progress;
    if (measure_phases) CUDA_CHECK(cudaEventRecord(phase_events[1], stream));
    CUDA_CHECK(traffic::launch(expert_path, data(output), data(experts), size_t(received) * hidden * 2,
                              expert_ctas, expert_passes, stream));
    if (measure_phases) CUDA_CHECK(cudaEventRecord(phase_events[2], stream));
    ncclNDTensor_t combine_inputs[] = {experts, output_probability};
    ncclNDTensor_t combine_outputs[] = {combined, combine_probability};
    if (check_invalid_combine) {
      ncclNDTensor_t invalid_inputs[] = {experts, probability};
      REQUIRE(ncclEpCombine(handle, invalid_inputs, 2, combine_outputs, 2,
                           nullptr, 0, false, nullptr, stream) == ncclInvalidArgument);
      ++invalid_rejections;
      ncclNDTensor_t invalid_width_inputs[] = {experts, invalid_probability};
      REQUIRE(ncclEpCombine(handle, invalid_width_inputs, 2, combine_outputs, 2,
                           nullptr, 0, false, nullptr, stream) == ncclInvalidArgument);
      ++invalid_rejections;
    }
    NCCL_CHECK(ncclEpCombine(handle, combine_inputs, 2, combine_outputs, 2,
                            nullptr, 0, false, nullptr, stream));
    NCCL_CHECK(ncclEpComplete(handle, nullptr, stream));
    if (diagnostic) { std::printf("PHASE rank=%d slot=%d epoch=%d combine_return\n", rank, slot, epoch); std::fflush(stdout); }
    if (measure_phases) CUDA_CHECK(cudaEventRecord(phase_events[3], stream));
    ++progress;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(done, stream));
  }

  void verify(int rank, int world) {
    REQUIRE(inflight);
    CUDA_CHECK(cudaMemsetAsync(seen, 0, size_t(world) * tokens * 4, stream));
    check_dispatch<<<received, 128, 0, stream>>>(static_cast<__nv_bfloat16*>(data(output)),
        static_cast<int64_t*>(data(output_routing)), static_cast<float*>(data(output_probability)),
        tokens, rank, world, id, slot, epoch, nonuniform_weights, seen, errors);
    check_seen<<<(world * tokens + 255) / 256, 256, 0, stream>>>(seen, tokens, rank, world, errors);
    check_values<<<32, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(data(input)),
        tokens, rank, slot, epoch, 1, errors);
    check_values<<<32, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(data(combined)),
        tokens, rank, slot, epoch, world == 1 ? 1 : 2, errors);
    check_combined_weights<<<(tokens * topk + 255) / 256, 256, 0, stream>>>(
        static_cast<float*>(data(combine_probability)), tokens * topk, rank, id, slot, epoch,
        nonuniform_weights, errors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ++completed;
    inflight = false;
    ++progress;
  }

  unsigned long long free_handle() {
    REQUIRE(!inflight);
    unsigned long long result;
    CUDA_CHECK(cudaMemcpy(&result, errors, sizeof(result), cudaMemcpyDeviceToHost));
    if (invalid_probability) {
      NCCL_CHECK(ncclEpTensorDestroy(group, invalid_probability));
      invalid_probability = nullptr;
    }
    for (auto tensor : {input, routing, probability, output, output_probability,
                        output_routing, experts, combined, combine_probability})
      NCCL_CHECK(ncclEpTensorDestroy(group, tensor));
    NCCL_CHECK(ncclEpHandleDestroy(handle));
    CUDA_CHECK(release(seen)); CUDA_CHECK(release(errors));
    ++progress;
    return result;
  }

  unsigned long long reshape(int count, int rank, int world) {
    auto result = free_handle();
    tokens = count;
    initialize(rank, world);
    return result;
  }

  unsigned long long close() {
    auto result = free_handle();
    NCCL_CHECK(ncclEpGroupDestroy(group, stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    if (measure_phases) for (auto event : phase_events) CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaEventDestroy(done)); CUDA_CHECK(cudaStreamDestroy(stream));
    ++progress;
    return result;
  }
};

void print_resources(FILE* output, char const* operation, nccl_ep::hybridep::KernelResources const& resource) {
  std::fprintf(output, "\"%s\":{\"sms\":%d,\"registers\":%d,\"threads\":%d,\"blocks\":%d,"
              "\"resident_blocks_api\":%d,\"static_smem\":%zu,\"dynamic_smem\":%zu,\"smem_per_sm\":%zu,\"smem_per_block\":%zu}",
      operation, resource.sms, resource.registers, resource.threads, resource.blocks,
      resource.resident_blocks, resource.static_smem, resource.dynamic_smem, resource.smem_per_sm, resource.smem_per_block);
}

struct WaveResult {
  int slots, tokens, phase, peak_admitted = 0, completed = 0;
  int active = 0, requested = 0, expert_ctas = 0, expert_passes = 1;
  nccl_ep::hybridep::KernelResources expert_resource;
  std::vector<float> gpu_samples;
  std::vector<float> dispatch_samples, expert_samples, combine_samples;
  std::vector<double> host_samples;
};

int main(int argc, char** argv) {
  Options options = parse(argc, argv);
  REQUIRE(MPI_Init(&argc, &argv) == MPI_SUCCESS);
  std::atomic<bool> finished{false};
  std::thread watchdog([&] {
    unsigned previous = progress.load();
    auto changed = std::chrono::steady_clock::now();
    while (!finished.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      unsigned current = progress.load();
      if (current != previous) { previous = current; changed = std::chrono::steady_clock::now(); }
      if (std::chrono::steady_clock::now() - changed > std::chrono::seconds(45)) {
        std::fprintf(stderr, "OWN_RANK_WATCHDOG no host progress for 45 seconds\n");
        std::fflush(stderr); std::_Exit(110);
      }
    }
  });
  int rank, world, local_rank, local_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
  MPI_Comm_rank(local, &local_rank); MPI_Comm_size(local, &local_size); MPI_Comm_free(&local);
  REQUIRE(world == 1 || world == 2 || world == 4 || world == 8);
  REQUIRE(local_size == 1 || local_size == 4 || local_size == 8);
  REQUIRE(!options.check_invalid_combine || world > 1);
  REQUIRE(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  CUDA_CHECK(cudaSetDevice(local_rank));
  cudaDeviceProp device;
  CUDA_CHECK(cudaGetDeviceProperties(&device, local_rank));
  int runtime_version, nccl_version;
  char const* taskset = std::getenv("UCCL_PROXY_USE_TASKSET");
  int proxy_taskset = taskset && strcmp(taskset, "1") == 0 ? 1 :
                      taskset && strcmp(taskset, "2") == 0 ? 2 : 0;
  cpu_set_t launcher_cpus;
  REQUIRE(!pthread_getaffinity_np(pthread_self(), sizeof(launcher_cpus), &launcher_cpus));
  int launcher_cpu_count = CPU_COUNT(&launcher_cpus);
  REQUIRE(launcher_cpu_count > (proxy_taskset == 2 ? 1 : 0));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  NCCL_CHECK(ncclGetVersion(&nccl_version));
  nccl_ep::hybridep::ScheduleResources resources;
  nccl_ep::hybridep::query_schedule_resources(hidden, 4, local_size, world / local_size, resources, true);
  ++progress;
  nccl_ep::hybridep::KernelResources expert_resource;
  int resident_requested = options.resident_groups ? options.resident_groups : options.active;
  int capacity = nccl_ep::hybridep::admission_limit(resources, options.slots, resident_requested), pool;
  if (options.expert_path != traffic::Path::Copy) {
    REQUIRE((device.major == 10 && traffic::tensor_compiled) ||
            (options.expert_path != traffic::Path::TensorInput && options.expert_path != traffic::Path::TensorOutput));
    CUDA_CHECK(traffic::query(options.expert_path, options.expert_ctas, device, expert_resource));
    if (!options.resident_groups)
      capacity = nccl_ep::hybridep::admission_limit(resources, expert_resource, options.slots, options.active);
  }
  REQUIRE(MPI_Allreduce(&capacity, &pool, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS);
  if (pool == 0) {
    FILE* output = options.output.empty() ? stdout : std::fopen((options.output + "/rank" +
        std::to_string(rank) + "-resources.json").c_str(), "wx");
    REQUIRE(output != nullptr);
    std::fprintf(output, "{\"schema\":2,\"combine_backward\":true,\"status\":\"UNSUPPORTED\",\"rank\":%d,\"physical_gpus\":%d,"
        "\"stages\":%d,\"blocks\":%d,\"tokens\":%d,\"hidden\":%d,",
        rank, world, options.stages, options.blocks, options.tokens, hidden);
    print_resources(output, "dispatch", resources.dispatch); std::fprintf(output, ",");
    print_resources(output, "combine", resources.combine); std::fprintf(output, "}\n");
    if (output != stdout) REQUIRE(std::fclose(output) == 0);
    MPI_Finalize(); finished.store(true); watchdog.join();
    return 77;
  }
  std::printf("HT_SCHEDULER_RESOURCES rank=%d physical_world=%d local_gpus=%d logical_slots=%d resource_pool=%d tokens=%d stages=%d blocks=%d\n",
      rank, world, local_size, options.slots, pool, options.tokens, options.stages, options.blocks);
  std::fflush(stdout);
  std::vector<Worker> workers;
  workers.reserve(pool);
  std::vector<int> shapes{options.tokens};
  if (options.shape_stress) shapes = {64, 128, 4096, 8192, 128, 8192, 64};
  std::vector<int> traffic_classes{options.expert_passes};
  if (options.traffic_sweep) {
    traffic_classes = {1, 4, 8, 4, 1};
    shapes.assign(traffic_classes.size(), options.tokens);
  }
  int maximum = *std::max_element(shapes.begin(), shapes.end());
  int resource_pool = pool;
  size_t free_bytes, total_bytes, first_group_bytes = 0, group_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t initial_free_bytes = free_bytes;
  for (int i = 0; i < resource_pool; ++i) {
    if (i) {
      int local_limit = nccl_ep::hybridep::memory_capacity(i, resource_pool, free_bytes, group_bytes);
      int agreed_limit;
      REQUIRE(MPI_Allreduce(&local_limit, &agreed_limit, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS);
      if (agreed_limit == i) break;
    }
    size_t before = free_bytes;
    workers.emplace_back(i, maximum, maximum, rank, world, options.diagnostic,
                         options.check_invalid_combine, options.nonuniform_weights, options);
    // Prime the maximum shape, including lazy communication allocations, outside timing.
    workers.back().enqueue(rank, i, 0);
    workers.back().verify(rank, world);
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    REQUIRE(free_bytes < before);
    size_t observed = before - free_bytes;
    if (!i) first_group_bytes = group_bytes = observed;
    else group_bytes = i == 1 ? observed : std::max(group_bytes, observed);
    std::printf("MEMORY_GROUP rank=%d groups=%d observed_bytes=%zu estimate_bytes=%zu free_bytes=%zu\n",
        rank, i + 1, observed, group_bytes, free_bytes); std::fflush(stdout);
  }
  pool = int(workers.size());
  REQUIRE(pool > 0);
  std::printf("HT_SCHEDULER_READY rank=%d pool=%d resource_pool=%d memory_limited=%d\n",
      rank, pool, resource_pool, pool < resource_pool); std::fflush(stdout);
  size_t pool_free_bytes = free_bytes;
  if (proxy_taskset == 2) {
    // Groups inherit the full launcher mask; proxies exclude its last CPU.
    int submit_cpu = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      if (CPU_ISSET(cpu, &launcher_cpus)) submit_cpu = cpu;
    cpu_set_t submit_mask;
    CPU_ZERO(&submit_mask); CPU_SET(submit_cpu, &submit_mask);
    REQUIRE(!pthread_setaffinity_np(pthread_self(), sizeof(submit_mask), &submit_mask));
    std::printf("SUBMIT_CPU_AFFINITY cpu=%d launcher_count=%d proxy_count=%d\n",
        sched_getcpu(), launcher_cpu_count, launcher_cpu_count - 1); std::fflush(stdout);
  }
  unsigned long long errors = 0;
  if (shapes[0] != maximum)
    for (auto& worker : workers) errors += worker.reshape(shapes[0], rank, world);
  cudaStream_t coordinator;
  cudaEvent_t start, stop, gate;
  CUDA_CHECK(cudaStreamCreateWithFlags(&coordinator, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
  std::vector<int> dimensions{options.slots};
  if (options.sweep) {
    dimensions.clear();
    for (int slots : {1, 2, 4, 8, 16, 32, 64}) if (slots <= options.slots) dimensions.push_back(slots);
  }
  std::vector<WaveResult> results;
  int epoch = 0, expected_completed = 0;
  using Clock = std::chrono::steady_clock;
  for (size_t phase = 0; phase < shapes.size(); ++phase) {
  int tokens = shapes[phase];
  if (phase && tokens != shapes[phase - 1]) for (auto& worker : workers) errors += worker.reshape(tokens, rank, world);
  int passes = options.traffic_sweep ? traffic_classes[phase] : options.expert_passes;
  BalancePolicy policy{passes, options.active, options.expert_ctas};
  if (!options.balance.empty()) {
    auto found = std::find_if(options.balance.begin(), options.balance.end(), [&](auto p) { return p.passes == passes; });
    REQUIRE(found != options.balance.end()); policy = *found;
  }
  int local_active = nccl_ep::hybridep::admission_limit(resources, pool, policy.active), active;
  if (options.expert_path != traffic::Path::Copy) {
    CUDA_CHECK(traffic::query(options.expert_path, policy.ctas, device, expert_resource));
    local_active = nccl_ep::hybridep::admission_limit(resources, expert_resource, pool, policy.active);
  }
  REQUIRE(MPI_Allreduce(&local_active, &active, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS);
  REQUIRE(active > 0);
  for (auto& worker : workers) {
    REQUIRE(!worker.inflight);
    worker.expert_ctas = policy.ctas; worker.expert_passes = passes;
  }
  cpu_set_t submit_cpus;
  REQUIRE(!pthread_getaffinity_np(pthread_self(), sizeof(submit_cpus), &submit_cpus));
  REQUIRE(CPU_COUNT(&submit_cpus) == (proxy_taskset == 2 ? 1 : launcher_cpu_count));
  std::printf("SHAPE rank=%d phase=%zu tokens=%d\n", rank, phase, tokens); std::fflush(stdout);
  for (int slots : dimensions) {
    WaveResult result{slots, tokens, int(phase)};
    result.active = active; result.requested = policy.active;
    result.expert_ctas = policy.ctas; result.expert_passes = passes; result.expert_resource = expert_resource;
    for (int round = -3; round < options.repeats; ++round) {
    ++epoch;
    REQUIRE(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
    auto host_start = Clock::now();
    float wave_ms = 0;
    float dispatch_ms = 0, expert_ms = 0, combine_ms = 0;
    for (int first = 0; first < slots; first += active) {
      int count = std::min(active, slots - first);
      int admitted = 0;
      CUDA_CHECK(cudaEventRecord(start, coordinator));
      CUDA_CHECK(cudaEventRecord(gate, coordinator));
      for (int i = 0; i < count; ++i) {
        CUDA_CHECK(cudaStreamWaitEvent(workers[i].stream, gate));
        workers[i].enqueue(rank, first + i, epoch);
        result.peak_admitted = std::max(result.peak_admitted, ++admitted);
      }
      for (int i = 0; i < count; ++i) CUDA_CHECK(cudaStreamWaitEvent(coordinator, workers[i].done));
      CUDA_CHECK(cudaEventRecord(stop, coordinator)); CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop)); wave_ms += elapsed;
      if (options.phase_events) {
        float maximum_ms[3]{};
        for (int i = 0; i < count; ++i)
          for (int phase = 0; phase < 3; ++phase) {
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, workers[i].phase_events[phase], workers[i].phase_events[phase + 1]));
            maximum_ms[phase] = std::max(maximum_ms[phase], elapsed);
          }
        dispatch_ms += maximum_ms[0]; expert_ms += maximum_ms[1]; combine_ms += maximum_ms[2];
      }
      for (int i = 0; i < count; ++i) { workers[i].verify(rank, world); ++result.completed; --admitted; }
      REQUIRE(admitted == 0);
    }
    double host_ms = std::chrono::duration<double, std::milli>(Clock::now() - host_start).count();
    if (round >= 0) {
      result.gpu_samples.push_back(wave_ms); result.host_samples.push_back(host_ms);
      if (options.phase_events) {
        result.dispatch_samples.push_back(dispatch_ms); result.expert_samples.push_back(expert_ms);
        result.combine_samples.push_back(combine_ms);
      }
    }
    }
    REQUIRE(result.completed == slots * (options.repeats + 3));
    expected_completed += result.completed;
    results.push_back(std::move(result));
  }
  }
  int completed = 0, invalid_rejections = 0;
  for (auto& worker : workers) {
    completed += worker.completed; invalid_rejections += worker.invalid_rejections;
    errors += worker.close();
  }
  REQUIRE(allocations.empty() && allocated_bytes == 0);
  REQUIRE(completed == expected_completed + pool);
  REQUIRE(invalid_rejections == (options.check_invalid_combine ? 2 * completed : 0));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaEventDestroy(gate));
  CUDA_CHECK(cudaStreamDestroy(coordinator));
  unsigned long long global_errors;
  MPI_Allreduce(&errors, &global_errors, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#ifdef HT_SCHEDULER_NATIVE
  char const* backend = "NCCL";
#else
  char const* backend = "UCCL";
#endif
  for (auto const& result : results) {
  std::string suffix = options.shape_stress || options.traffic_sweep ? "-phase" + std::to_string(result.phase) : "";
  FILE* output = options.output.empty() ? stdout : std::fopen((options.output + "/rank" +
      std::to_string(rank) + "-slots" + std::to_string(result.slots) + suffix + ".json").c_str(), "wx");
  REQUIRE(output != nullptr);
  std::fprintf(output, "{\"schema\":2,\"combine_backward\":true,\"full_probability_validation\":true,\"backend\":\"%s\",\"status\":\"%s\",\"rank\":%d,\"physical_gpus\":%d,\"nodes\":%d,"
              "\"logical_slots_per_rank\":%d,\"total_logical_slots\":%d,\"tokens\":%d,\"hidden\":%d,"
              "\"phase\":%d,\"sm_major\":%d,\"sm_minor\":%d,\"cuda_runtime\":%d,\"nccl_version\":%d,\"proxy_taskset\":%d,"
              "\"stages\":%d,\"blocks\":%d,\"requested_active\":%d,\"agreed_active\":%d,\"allocated_EP_groups\":%d,"
              "\"observed_peak_admitted\":%d,\"per_handle_peak\":1,\"completed\":%d,\"errors\":%llu,"
              "\"callback_peak_bytes\":%zu,\"callback_final_live\":0,\"normal_drain\":true,\"invalid_combine_rejections\":%d,",
      backend, global_errors ? "FAIL" : "PASS", rank, world, world / local_size, result.slots,
      world * result.slots, result.tokens, hidden, result.phase, device.major, device.minor,
      runtime_version, nccl_version, proxy_taskset, options.stages, options.blocks, result.requested,
      std::min(result.active, result.slots), pool, result.peak_admitted, result.completed, global_errors, peak_bytes,
      options.check_invalid_combine ? 2 * result.completed : 0);
  std::fprintf(output, "\"resource_pool_limit\":%d,\"memory_budget_enabled\":true,\"memory_limited\":%s,"
              "\"memory_calibration_tokens\":%d,\"calibration_completed\":%d,"
              "\"calibration_invalid_combine_rejections\":%d,\"cuda_initial_free_bytes\":%zu,"
              "\"cuda_free_after_pool_bytes\":%zu,\"cuda_total_bytes\":%zu,"
              "\"first_group_gpu_bytes\":%zu,\"incremental_group_gpu_bytes\":%zu,",
      resource_pool, pool < resource_pool ? "true" : "false", maximum, pool, options.check_invalid_combine ? 2 * pool : 0,
      initial_free_bytes, pool_free_bytes, total_bytes, first_group_bytes, group_bytes);
  std::fprintf(output, "\"launcher_cpu_budget_count\":%d,\"configured_proxy_cpu_budget_count\":%d,"
              "\"submit_cpu_budget_count\":%d,",
      launcher_cpu_count, proxy_taskset ? launcher_cpu_count - (proxy_taskset == 2) : 0,
      proxy_taskset == 2 ? 1 : launcher_cpu_count);
  std::fprintf(output, "\"nonuniform_probability_weights\":%s,", options.nonuniform_weights ? "true" : "false");
  bool uses_tensor = options.expert_path == traffic::Path::TensorInput || options.expert_path == traffic::Path::TensorOutput;
  std::fprintf(output, "\"expert_path\":\"%s\",\"expert_ctas\":%d,\"expert_passes\":%d,"
              "\"phase_events\":%s,\"tmem_reserved_bytes_per_expert_cta\":%d,",
      traffic::name(options.expert_path), result.expert_ctas, result.expert_passes,
      options.phase_events ? "true" : "false", uses_tensor ? 16384 : 0);
  std::fprintf(output, "\"requested_resident_groups\":%d,\"balance_from_profile_table\":%s,",
      resident_requested, options.balance.empty() ? "false" : "true");
  if (options.expert_path != traffic::Path::Copy) { print_resources(output, "expert", result.expert_resource); std::fprintf(output, ","); }
  print_resources(output, "dispatch", resources.dispatch); std::fprintf(output, ",");
  print_resources(output, "combine", resources.combine);
  std::fprintf(output, ",\"gpu_window_sum_ms\":[");
  for (size_t i = 0; i < result.gpu_samples.size(); ++i) std::fprintf(output, "%s%.9g", i ? "," : "", result.gpu_samples[i]);
  std::fprintf(output, "],\"host_wave_with_validation_ms\":[");
  for (size_t i = 0; i < result.host_samples.size(); ++i) std::fprintf(output, "%s%.9g", i ? "," : "", result.host_samples[i]);
  std::fprintf(output, "]");
  for (auto const& phase : {std::make_pair("dispatch", &result.dispatch_samples),
                           std::make_pair("expert", &result.expert_samples),
                           std::make_pair("combine", &result.combine_samples)}) {
    std::fprintf(output, ",\"%s_window_max_sum_ms\":[", phase.first);
    for (size_t i = 0; i < phase.second->size(); ++i) std::fprintf(output, "%s%.9g", i ? "," : "", (*phase.second)[i]);
    std::fprintf(output, "]");
  }
  std::fprintf(output, "}\n"); std::fflush(output);
  if (output != stdout) REQUIRE(std::fclose(output) == 0);
  }
  MPI_Finalize();
  finished.store(true); watchdog.join();
  return global_errors ? 5 : 0;
}

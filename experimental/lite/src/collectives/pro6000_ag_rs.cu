// SPDX-License-Identifier: Apache-2.0
// Matched out-of-place FP32 AllGather/ReduceScatter harness. All required local
// copies, reductions, communication and completion waits are inside timing.
#include "uccl_lite/allreduce_validation.cuh"
#include <mscclpp/core.hpp>
#include <chrono>
#include <memory>
#include <sys/wait.h>
#include <unistd.h>
#ifdef UCCL_LITE_NCCL_REFERENCE
#include <nccl.h>
#ifndef UCCL_LITE_RANKS
#define UCCL_LITE_RANKS 8
#endif
constexpr int kRanks = UCCL_LITE_RANKS;
#define NCCL_CHECK(call)                                              \
  do {                                                                \
    auto result = (call);                                             \
    if (result != ncclSuccess) {                                      \
      std::fprintf(stderr, "NCCL: %s\n", ncclGetErrorString(result)); \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)
#else
#include "uccl_lite/pro6000_ag_rs.cuh"
#endif
using namespace uccl_lite;
static_assert(kRanks == 4 || kRanks == 8);

struct Config {
  size_t begin = 1024, end = 1UL << 30, capacity = 1UL << 30;
  size_t tile = 16UL << 20;
  int blocks = 4, threads = 256, workers = 1, streams = 1, copyUnroll = 1,
      routing = 0;
  int warmups = 10, iterations = 20, seed = 1, stress = 0;
  bool gather = true, graph = false, storeSignals = true, fullFences = false,
       poisonKernel = false, fusedCopy = true;
  std::string mode = "ring-ce";
};

__global__ void agRsInitialize(float* data, size_t count, int rank, int seed) {
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < count;
       j += size_t(gridDim.x) * blockDim.x)
    data[j] = float(pattern(j, rank, seed));
}
__global__ void agRsPoison(float* output, size_t count) {
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < count;
       j += size_t(gridDim.x) * blockDim.x)
    output[j] = __int_as_float(0x7fffffff);
}
__global__ void agRsSkew() {
  auto begin = clock64();
  while (clock64() - begin < 100000) {
  }
}
__global__ void agRsVerify(float const* input, float const* output,
                           size_t shard, int rank, int seed, bool gather,
                           unsigned long long* errors) {
  unsigned long long bad = 0;
  size_t outputCount = gather ? shard * kRanks : shard;
  size_t inputCount = gather ? shard : shard * kRanks;
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < outputCount;
       j += size_t(gridDim.x) * blockDim.x) {
    int expected = 0;
    if (gather)
      expected = pattern(j % shard, j / shard, seed);
    else
      for (int r = 0; r < kRanks; ++r)
        expected += pattern(rank * shard + j, r, seed);
    if (!isfinite(output[j]) || output[j] != float(expected)) ++bad;
  }
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < inputCount;
       j += size_t(gridDim.x) * blockDim.x)
    if (input[j] != float(pattern(j, rank, seed))) ++bad;
  if (bad) atomicAdd(errors, bad);
}

#ifndef UCCL_LITE_NCCL_REFERENCE
struct AgRsContext {
  std::unique_ptr<mscclpp::Communicator> comm;
  std::array<std::unique_ptr<mscclpp::Communicator>, 2> copies;
  std::shared_ptr<mscclpp::ProxyService> proxy;
  std::vector<std::unique_ptr<mscclpp::MemoryChannel>> memory;
  std::vector<std::unique_ptr<mscclpp::PortChannel>> ports;
  std::vector<mscclpp::RegisteredMemory> regions;
  AgRsChannels* channels = nullptr;
  mscclpp::DeviceSyncer* syncers = nullptr;
  uint32_t* epochs = nullptr;
  uint64_t* sent = nullptr;
  char* allocation = nullptr;
  char* buffer = nullptr;
  size_t stride, packetSlot;
  AgRsLayout layout{};
  std::vector<AgRsChannels> hostChannels;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> done;
  cudaEvent_t ready = nullptr;

  AgRsContext(std::shared_ptr<mscclpp::TcpBootstrap> bootstrap, int rank,
              Config const& cfg) {
    comm = std::make_unique<mscclpp::Communicator>(bootstrap);
    stride = cfg.capacity + 2 * guardBytes;
    packetSlot = 2 * (16UL << 20) * kLevels;
    size_t total = 4 * stride + 2 * packetSlot;
    checkCuda(cudaMalloc(&allocation, total + 2 * guardBytes));
    checkCuda(cudaMemset(allocation, 0, total + 2 * guardBytes));
    buffer = allocation + guardBytes;
    layout = {0, stride, 2 * stride, 3 * stride, 4 * stride};
    auto full =
        comm->registerMemory(buffer, total, mscclpp::Transport::CudaIpc);
    for (int r = 0; r < 4; ++r)
      regions.push_back(comm->registerMemory(buffer + r * stride, cfg.capacity,
                                             mscclpp::Transport::CudaIpc));
    std::array<mscclpp::Connection, kRanks> connections;
    std::array<mscclpp::RegisteredMemory, kRanks> remoteFull;
    std::array<std::array<mscclpp::RegisteredMemory, 4>, kRanks> remote;
    std::array<std::shared_future<mscclpp::Connection>, kRanks> future;
    std::array<std::shared_future<mscclpp::RegisteredMemory>, kRanks>
        fullFuture;
    std::array<std::array<std::shared_future<mscclpp::RegisteredMemory>, 4>,
               kRanks>
        regionFuture;
    for (int p = 0; p < kRanks; ++p) {
      if (p == rank) continue;
      future[p] = comm->connect(
          mscclpp::EndpointConfig(mscclpp::Transport::CudaIpc), p, 100);
      comm->sendMemory(full, p, 10);
      fullFuture[p] = comm->recvMemory(p, 10);
      for (int r = 0; r < 4; ++r) {
        comm->sendMemory(regions[r], p, r);
        regionFuture[p][r] = comm->recvMemory(p, r);
      }
    }
    for (int p = 0; p < kRanks; ++p)
      if (p != rank) connections[p] = future[p].get();
    std::vector<AgRsChannels> handles(cfg.blocks);
    checkCuda(cudaMalloc(&sent, cfg.blocks * kRanks * sizeof(uint64_t)));
    checkCuda(cudaMemset(sent, 0, cfg.blocks * kRanks * sizeof(uint64_t)));
    for (int p = 0; p < kRanks; ++p) {
      if (p == rank) continue;
      remoteFull[p] = fullFuture[p].get();
      for (int r = 0; r < 4; ++r) remote[p][r] = regionFuture[p][r].get();
    }
    int next = ringRank((ringPosition(rank) + 1) % kRanks);
    int prev = ringRank((ringPosition(rank) + kRanks - 1) % kRanks);
    std::array<std::array<mscclpp::Connection, kRanks>, 2> copyConnections;
    if (cfg.mode == "ring-ce") {
      proxy = std::make_shared<mscclpp::ProxyService>();
      for (int c = 0; c < cfg.streams; ++c) {
        copies[c] = std::make_unique<mscclpp::Communicator>(bootstrap);
        for (int p = 0; p < kRanks; ++p)
          if (p != rank)
            future[p] = copies[c]->connect(
                mscclpp::EndpointConfig(mscclpp::Transport::CudaIpc), p,
                200 + c);
        for (int p = 0; p < kRanks; ++p)
          if (p != rank) copyConnections[c][p] = future[p].get();
      }
    }
    std::array<mscclpp::MemoryId, 4> localId{}, remoteId{};
    if (proxy)
      for (int r = 0; r < 4; ++r) {
        localId[r] = proxy->addMemory(regions[r]);
        remoteId[r] = proxy->addMemory(remote[next][r]);
      }
    for (int lane = 0; lane < cfg.blocks; ++lane) {
      handles[lane].order = cfg.routing / 2;
      handles[lane].direction = (cfg.routing & 1) && (lane & 1) ? -1 : 1;
      int position =
          agRsRingPosition(rank, handles[lane].order, handles[lane].direction);
      int laneNext = agRsRingRank((position + 1) % kRanks, handles[lane].order,
                                  handles[lane].direction);
      int lanePrev = agRsRingRank((position + kRanks - 1) % kRanks,
                                  handles[lane].order, handles[lane].direction);
      for (int p = 0; p < kRanks; ++p) {
        if (p == rank) continue;
        auto sem = comm->buildSemaphore(connections[p], p).get();
        memory.push_back(std::make_unique<mscclpp::MemoryChannel>(
            sem, remoteFull[p], full, buffer));
        AgRsPeer channel{};
        static_cast<Chan&>(channel) = memory.back()->deviceHandle();
        channel.sent = cfg.storeSignals ? sent + lane * kRanks + p : nullptr;
        handles[lane].peers[p] = channel;
        if (p == lanePrev) handles[lane].acknowledge = channel;
        if (p == laneNext) handles[lane].drained = channel;
        if (!proxy) continue;
        int c = lane % cfg.streams;
        auto sid =
            proxy->buildAndAddSemaphore(*copies[c], copyConnections[c][p]);
        auto port = [&](int dst, int src) {
          ports.push_back(std::make_unique<mscclpp::PortChannel>(
              proxy->portChannel(sid, remoteId[dst], localId[src])));
          return ports.back()->deviceHandle();
        };
        if (p == next) {
          handles[lane].first = port(3, 0);
          handles[lane].partial = port(3, 2);
          handles[lane].gather = port(1, 1);
        }
        if (p == prev) handles[lane].incoming = port(1, 1);
      }
    }
    checkCuda(cudaMalloc(&channels, handles.size() * sizeof(AgRsChannels)));
    checkCuda(cudaMemcpy(channels, handles.data(),
                         handles.size() * sizeof(AgRsChannels),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc(&syncers, cfg.blocks * sizeof(mscclpp::DeviceSyncer)));
    checkCuda(
        cudaMemset(syncers, 0, cfg.blocks * sizeof(mscclpp::DeviceSyncer)));
    checkCuda(cudaMalloc(&epochs, cfg.blocks * sizeof(uint32_t)));
    checkCuda(cudaMemset(epochs, 0, cfg.blocks * sizeof(uint32_t)));
    if (cfg.mode == "ring-host-ce") {
      hostChannels = handles;
      streams.resize(cfg.blocks);
      done.resize(cfg.blocks);
      checkCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
      for (int lane = 0; lane < cfg.blocks; ++lane) {
        checkCuda(
            cudaStreamCreateWithFlags(&streams[lane], cudaStreamNonBlocking));
        checkCuda(
            cudaEventCreateWithFlags(&done[lane], cudaEventDisableTiming));
      }
    }
    if (proxy) proxy->startProxy();
  }
  void hostLaunch(Config const& cfg, int rank, size_t shard,
                  cudaStream_t caller) {
    size_t tile = std::min(shard, cfg.tile / sizeof(float));
    int lanes = std::min<size_t>(cfg.blocks, (shard + tile - 1) / tile);
    checkCuda(cudaEventRecord(ready, caller));
    for (int lane = 0; lane < lanes; ++lane)
      checkCuda(cudaStreamWaitEvent(streams[lane], ready));
    int position = ringPosition(rank);
    int next = ringRank((position + 1) % kRanks);
    for (size_t offset = 0, index = 0; offset < shard;
         offset += tile, ++index) {
      size_t count = std::min(tile, shard - offset);
      int lane = index % lanes;
      auto work = streams[lane];
      auto ch = hostChannels[lane];
      if (cfg.gather)
        agRsHostPhase<true><<<cfg.workers, cfg.threads, 0, work>>>(
            ch, buffer, rank, shard, offset, count, layout, syncers + lane, -1);
      for (int step = 0; step < kRanks - 1; ++step) {
        int owner = ringRank((position - (cfg.gather ? 0 : 1) - step + kRanks) %
                             kRanks);
        size_t send = (owner * shard + offset) * sizeof(float);
        size_t src = cfg.gather ? layout.output
                                : (step == 0 ? layout.input : layout.partial);
        size_t dst = cfg.gather ? layout.output : layout.receive;
        auto remote = static_cast<char*>(ch.peers[next].dst_);
        checkCuda(cudaMemcpyAsync(remote + dst + send, buffer + src + send,
                                  count * sizeof(float),
                                  cudaMemcpyDeviceToDevice, work));
        if (cfg.gather)
          agRsHostPhase<true><<<1, 32, 0, work>>>(ch, buffer, rank, shard,
                                                  offset, count, layout,
                                                  syncers + lane, step);
        else
          agRsHostPhase<false><<<cfg.workers, cfg.threads, 0, work>>>(
              ch, buffer, rank, shard, offset, count, layout, syncers + lane,
              step);
      }
    }
    for (int lane = 0; lane < lanes; ++lane) {
      checkCuda(cudaEventRecord(done[lane], streams[lane]));
      checkCuda(cudaStreamWaitEvent(caller, done[lane]));
    }
  }
  ~AgRsContext() {
    if (proxy) proxy->stopProxy();
    for (auto work : streams) checkCuda(cudaStreamDestroy(work));
    for (auto event : done) checkCuda(cudaEventDestroy(event));
    if (ready) checkCuda(cudaEventDestroy(ready));
    checkCuda(cudaFree(channels));
    checkCuda(cudaFree(syncers));
    checkCuda(cudaFree(epochs));
    checkCuda(cudaFree(sent));
    ports.clear();
    memory.clear();
    proxy.reset();
    regions.clear();
    comm.reset();
    for (auto& copy : copies) copy.reset();
    checkCuda(cudaFree(allocation));
  }
};
#endif

static void runRank(int rank, mscclpp::UniqueId id, Config cfg) {
  pinRank(rank);
  checkCuda(cudaSetDevice(rank));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, kRanks);
  bootstrap->initialize(id);
  cudaStream_t stream;
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // Loading a new CUDA function can synchronize its context. Do this before
  // peer-synchronizing work: first-loading the skew kernel behind an active
  // proxy-driven collective can deadlock the CUDA runtime and proxy thread.
  cudaFuncAttributes attributes;
  checkCuda(cudaFuncGetAttributes(&attributes, agRsSkew));
  checkCuda(cudaFuncGetAttributes(&attributes, agRsPoison));
  float *input, *output;
#ifdef UCCL_LITE_NCCL_REFERENCE
  std::array<ncclUniqueId, kRanks> ids{};
  if (!rank) NCCL_CHECK(ncclGetUniqueId(&ids[0]));
  bootstrap->allGather(ids.data(), sizeof(ncclUniqueId));
  ncclComm_t nccl;
  NCCL_CHECK(ncclCommInitRank(&nccl, kRanks, ids[0], rank));
  char* allocation;
  size_t stride = cfg.capacity + 2 * guardBytes;
  checkCuda(cudaMalloc(&allocation, 2 * stride));
  input = reinterpret_cast<float*>(allocation + guardBytes);
  output = reinterpret_cast<float*>(allocation + stride + guardBytes);
  auto launch = [&](size_t shard) {
    if (cfg.gather)
      NCCL_CHECK(ncclAllGather(input, output, shard, ncclFloat, nccl, stream));
    else
      NCCL_CHECK(ncclReduceScatter(input, output, shard, ncclFloat, ncclSum,
                                   nccl, stream));
  };
#else
  int active;
  cudaDeviceProp property;
  checkCuda(cudaGetDeviceProperties(&property, rank));
  checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, cfg.gather ? agRsRingCE<true> : agRsRingCE<false>, cfg.threads,
      0));
  if (cfg.mode == "put")
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, cfg.gather ? agRsPut<true> : agRsPut<false>, cfg.threads, 0));
  if (cfg.mode == "ring-put")
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, cfg.gather ? agRsRingPut<true> : agRsRingPut<false>,
        cfg.threads, 0));
  if (cfg.mode == "ring-host-ce")
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, cfg.gather ? agRsHostPhase<true> : agRsHostPhase<false>,
        cfg.threads, 0));
  if (cfg.blocks * cfg.workers > active * property.multiProcessorCount) {
    std::fprintf(stderr, "Grid exceeds residency capacity\n");
    std::exit(2);
  }
  AgRsContext peer(bootstrap, rank, cfg);
  input = reinterpret_cast<float*>(peer.buffer + peer.layout.input);
  output = reinterpret_cast<float*>(peer.buffer + peer.layout.output);
  auto launch = [&](size_t shard) {
    if (cfg.mode == "ring-host-ce") {
      peer.hostLaunch(cfg, rank, shard, stream);
    } else if (cfg.mode == "hier-ll") {
      if (cfg.gather)
        agRsHierLL<true><<<cfg.blocks, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, peer.layout, peer.epochs,
            peer.packetSlot);
      else
        agRsHierLL<false><<<cfg.blocks, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, peer.layout, peer.epochs,
            peer.packetSlot);
    } else if (cfg.mode == "ll") {
      if (cfg.gather)
        agRsLL<true><<<cfg.blocks, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, peer.layout, peer.epochs,
            peer.packetSlot);
      else
        agRsLL<false><<<cfg.blocks, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, peer.layout, peer.epochs,
            peer.packetSlot);
    } else {
      size_t tile = std::min(shard, cfg.tile / sizeof(float));
      if (cfg.mode == "put" || cfg.mode == "ring-put")
        tile = ((shard / 4 + cfg.blocks - 1) / cfg.blocks) * 4;
      int lanes = std::min<size_t>(cfg.blocks, (shard + tile - 1) / tile);
      if (cfg.mode == "put" && cfg.gather)
        agRsPut<true><<<lanes * cfg.workers, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, tile, peer.layout,
            peer.syncers, cfg.workers, cfg.fullFences, cfg.copyUnroll);
      else if (cfg.mode == "put")
        agRsPut<false><<<lanes * cfg.workers, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, tile, peer.layout,
            peer.syncers, cfg.workers, cfg.fullFences, cfg.copyUnroll);
      else if (cfg.mode == "ring-put" && cfg.gather)
        agRsRingPut<true><<<lanes * cfg.workers, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, tile, peer.layout,
            peer.syncers, cfg.workers, cfg.fullFences, cfg.copyUnroll,
            cfg.fusedCopy);
      else if (cfg.mode == "ring-put")
        agRsRingPut<false><<<lanes * cfg.workers, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, tile, peer.layout,
            peer.syncers, cfg.workers, cfg.fullFences, cfg.copyUnroll,
            cfg.fusedCopy);
      else if (cfg.gather)
        agRsRingCE<true><<<lanes * cfg.workers, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, tile, peer.layout,
            peer.syncers, cfg.workers);
      else
        agRsRingCE<false><<<lanes * cfg.workers, cfg.threads, 0, stream>>>(
            peer.channels, peer.buffer, rank, shard, tile, peer.layout,
            peer.syncers, cfg.workers);
    }
    checkCuda(cudaGetLastError());
  };
#endif
  unsigned long long* errors;
  checkCuda(cudaMalloc(&errors, sizeof(*errors)));
  if (!rank)
    std::printf(
        "# backend=%s op=%s ranks=%d dtype=FP32 out-of-place total-bytes "
        "warmups=%d continuous=%d capacity=%zu mode=%s blocks=%d threads=%d "
        "tile_per_shard=%zu workers=%d copy_contexts=%d graph=%d seed=%d "
        "stress=%d store_signals=%d full_fences=%d poison_kernel=%d "
        "copy_unroll=%d routing=%d fused_copy=%d\n",
#ifdef UCCL_LITE_NCCL_REFERENCE
        "NCCL",
#else
        "UCCL_Lite",
#endif
        cfg.gather ? "allgather" : "reducescatter", kRanks, cfg.warmups,
        cfg.iterations, cfg.capacity, cfg.mode.c_str(), cfg.blocks, cfg.threads,
        cfg.tile, cfg.workers, cfg.streams, cfg.graph, cfg.seed, cfg.stress,
        cfg.storeSignals, cfg.fullFences, cfg.poisonKernel, cfg.copyUnroll,
        cfg.routing, cfg.fusedCopy);
  for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
    size_t shard = bytes / (sizeof(float) * kRanks);
    size_t inputBytes = cfg.gather ? bytes / kRanks : bytes;
    size_t outputBytes = cfg.gather ? bytes : bytes / kRanks;
    setGuard(reinterpret_cast<char*>(input), inputBytes, stream);
    setGuard(reinterpret_cast<char*>(output), outputBytes, stream);
#ifndef UCCL_LITE_NCCL_REFERENCE
    setGuard(peer.buffer + peer.layout.partial, bytes, stream);
    setGuard(peer.buffer + peer.layout.receive, bytes, stream);
#endif
    checkCuda(cudaStreamSynchronize(stream));
    bootstrap->barrier();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (cfg.graph) {
      checkCuda(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      launch(shard);
      checkCuda(cudaStreamEndCapture(stream, &graph));
      checkCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    auto invoke = [&]() {
      if (executable)
        checkCuda(cudaGraphLaunch(executable, stream));
      else
        launch(shard);
    };
    auto poison = [&]() {
      if (cfg.poisonKernel)
        agRsPoison<<<256, 256, 0, stream>>>(output, outputBytes / 4);
      else
        checkCuda(cudaMemsetAsync(output, 0xff, outputBytes, stream));
    };
    auto check = [&](int seed, int call) {
      checkCuda(cudaStreamSynchronize(stream));
      checkCuda(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
      agRsVerify<<<256, 256, 0, stream>>>(input, output, shard, rank, seed,
                                          cfg.gather, errors);
      unsigned long long bad;
      checkCuda(cudaMemcpyAsync(&bad, errors, sizeof(bad),
                                cudaMemcpyDeviceToHost, stream));
      checkCuda(cudaStreamSynchronize(stream));
      auto guards = checkGuard(reinterpret_cast<char*>(input), inputBytes) +
                    checkGuard(reinterpret_cast<char*>(output), outputBytes);
#ifndef UCCL_LITE_NCCL_REFERENCE
      guards += checkGuard(peer.buffer + peer.layout.partial, bytes) +
                checkGuard(peer.buffer + peer.layout.receive, bytes);
#endif
      if (bad || guards)
        std::fprintf(stderr,
                     "CHECK_FAILURE rank=%d bytes=%zu seed=%d call=%d "
                     "output_input=%llu guards=%llu\n",
                     rank, bytes, seed, call, bad, guards);
      return bad + guards;
    };
    unsigned long long bad = 0;
    // Change the immutable input before every checked call. Old packets,
    // incorrect rank permutations and stale graph epochs cannot pass this.
    for (int trial = 0; trial < 9; ++trial) {
      int seed = cfg.seed + trial;
      agRsInitialize<<<256, 256, 0, stream>>>(input, inputBytes / 4, rank,
                                              seed);
      poison();
      checkCuda(cudaStreamSynchronize(stream));
      bootstrap->barrier();
      invoke();
      bad += check(seed, trial);
      bootstrap->barrier();
    }
    // Queue changing inputs and full-output verification on the caller stream
    // without host barriers between calls. This exposes premature completion
    // and source-buffer reuse bugs that constant-input timing cannot detect.
    if (cfg.stress) {
      checkCuda(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
      bootstrap->barrier();
      for (int call = 0; call < cfg.stress; ++call) {
        int seed = cfg.seed + 100 + call;
        if (rank == call % kRanks) agRsSkew<<<1, 32, 0, stream>>>();
        agRsInitialize<<<256, 256, 0, stream>>>(input, inputBytes / 4, rank,
                                                seed);
        poison();
        invoke();
        agRsVerify<<<256, 256, 0, stream>>>(input, output, shard, rank, seed,
                                            cfg.gather, errors);
      }
      unsigned long long stressBad;
      checkCuda(cudaMemcpyAsync(&stressBad, errors, sizeof(stressBad),
                                cudaMemcpyDeviceToHost, stream));
      checkCuda(cudaStreamSynchronize(stream));
      bad += stressBad;
      bad += check(cfg.seed + 99 + cfg.stress, cfg.stress);
      std::printf("STRESS rank=%d bytes=%zu calls=%d errors=%llu\n", rank,
                  bytes, cfg.stress, stressBad);
      bootstrap->barrier();
    }
    agRsInitialize<<<256, 256, 0, stream>>>(input, inputBytes / 4, rank,
                                            cfg.seed + 10);
    checkCuda(cudaStreamSynchronize(stream));
    bootstrap->barrier();
    for (int i = 0; i < cfg.warmups; ++i) invoke();
    checkCuda(cudaStreamSynchronize(stream));
    rankThreadPlacement(rank, true, bytes == cfg.begin);
    bootstrap->barrier();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iterations; ++i) invoke();
    checkCuda(cudaStreamSynchronize(stream));
    auto stop = std::chrono::steady_clock::now();
    rankThreadPlacement(rank, false, false);
    double us =
        std::chrono::duration<double, std::micro>(stop - start).count() /
        cfg.iterations;
    bad += check(cfg.seed + 10, cfg.warmups + cfg.iterations);
    std::array<double, kRanks> times{};
    std::array<unsigned long long, kRanks> failures{};
    times[rank] = us;
    failures[rank] = bad;
    bootstrap->allGather(times.data(), sizeof(double));
    bootstrap->allGather(failures.data(), sizeof(unsigned long long));
    double worst = *std::max_element(times.begin(), times.end());
    auto failed = *std::max_element(failures.begin(), failures.end());
    if (!rank) {
      std::printf("RESULT %zu %.6f %.6f %llu\n", bytes, worst,
                  bytes * (double(kRanks - 1) / kRanks) / (worst * 1000),
                  failed);
      for (int r = 0; r < kRanks; ++r)
        std::printf("RANK %zu %d %.6f %llu\n", bytes, r, times[r], failures[r]);
      std::fflush(stdout);
    }
    if (failed) std::exit(3);
    if (executable) checkCuda(cudaGraphExecDestroy(executable));
    if (graph) checkCuda(cudaGraphDestroy(graph));
    bootstrap->barrier();
    if (bytes > cfg.end / 2) break;
  }
#ifdef UCCL_LITE_NCCL_REFERENCE
  NCCL_CHECK(ncclCommDestroy(nccl));
  checkCuda(cudaFree(allocation));
#endif
  checkCuda(cudaFree(errors));
  checkCuda(cudaStreamDestroy(stream));
  bootstrap->barrier();
}

int main(int argc, char** argv) {
  Config cfg;
  try {
    for (int i = 1; i < argc; ++i) {
      std::string option = argv[i];
      if (++i == argc) return 2;
      std::string value = argv[i];
      if (option == "--begin")
        cfg.begin = std::stoull(value);
      else if (option == "--end")
        cfg.end = std::stoull(value);
      else if (option == "--bytes")
        cfg.begin = cfg.end = std::stoull(value);
      else if (option == "--capacity")
        cfg.capacity = std::stoull(value);
      else if (option == "--mode")
        cfg.mode = value;
      else if (option == "--blocks")
        cfg.blocks = std::stoi(value);
      else if (option == "--threads")
        cfg.threads = std::stoi(value);
      else if (option == "--workers")
        cfg.workers = std::stoi(value);
      else if (option == "--streams")
        cfg.streams = std::stoi(value);
      else if (option == "--tile")
        cfg.tile = std::stoull(value);
      else if (option == "--seed")
        cfg.seed = std::stoi(value);
      else if (option == "--graph")
        cfg.graph = std::stoi(value);
      else if (option == "--warmups")
        cfg.warmups = std::stoi(value);
      else if (option == "--iterations")
        cfg.iterations = std::stoi(value);
      else if (option == "--fused-copy")
        cfg.fusedCopy = std::stoi(value);
      else if (option == "--routing")
        cfg.routing = std::stoi(value);
      else if (option == "--copy-unroll")
        cfg.copyUnroll = std::stoi(value);
      else if (option == "--poison-kernel")
        cfg.poisonKernel = std::stoi(value);
      else if (option == "--full-fences")
        cfg.fullFences = std::stoi(value);
      else if (option == "--store-signals")
        cfg.storeSignals = std::stoi(value);
      else if (option == "--stress")
        cfg.stress = std::stoi(value);
      else if (option == "--op" &&
               (value == "allgather" || value == "reducescatter"))
        cfg.gather = value == "allgather";
      else
        return 2;
    }
  } catch (std::exception const&) {
    return 2;
  }
  if (cfg.begin < 16 * kRanks || cfg.begin % (16 * kRanks) ||
      cfg.end % (16 * kRanks) || cfg.begin > cfg.end ||
      cfg.end > cfg.capacity || cfg.capacity > (1UL << 30) ||
      cfg.capacity % 4096 || cfg.tile < 16 || cfg.tile % 16 || cfg.blocks < 1 ||
      cfg.workers < 1 || cfg.blocks > 128 || cfg.workers > 128 ||
      cfg.blocks * cfg.workers > 128 || cfg.streams < 1 || cfg.streams > 2 ||
      cfg.threads < 32 || cfg.threads > 1024 || cfg.threads % 32 ||
      cfg.warmups < 1 || cfg.iterations < 1 || cfg.iterations > 10000 ||
      cfg.seed < 0 || cfg.seed > 100000 || cfg.stress < 0 ||
      cfg.stress > 10000 || cfg.routing < 0 || cfg.routing > 3 ||
      (cfg.routing && cfg.mode != "ring-put") ||
      (cfg.copyUnroll != 1 && cfg.copyUnroll != 4 && cfg.copyUnroll != 8) ||
      (cfg.mode != "ring-ce" && cfg.mode != "ll" && cfg.mode != "hier-ll" &&
       cfg.mode != "put" && cfg.mode != "ring-put" &&
       cfg.mode != "ring-host-ce") ||
      ((cfg.mode == "ll" || cfg.mode == "hier-ll") && cfg.end > (16UL << 20)))
    return 2;
  auto id = mscclpp::TcpBootstrap::createUniqueId();
  std::vector<pid_t> children;
  for (int rank = 1; rank < kRanks; ++rank) {
    pid_t pid = fork();
    if (!pid) {
      runRank(rank, id, cfg);
      std::fflush(stdout);
      return 0;
    }
    if (pid < 0) return 1;
    children.push_back(pid);
  }
  runRank(0, id, cfg);
  int result = 0;
  for (auto pid : children) {
    int status;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) result = 1;
  }
  return result;
}

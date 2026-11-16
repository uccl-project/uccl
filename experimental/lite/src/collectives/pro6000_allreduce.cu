// SPDX-License-Identifier: Apache-2.0
// Experimental matched FP32/C++ AllReduce harness and topology-specific
// kernels.
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
#define NCCL_CHECK(call)                                         \
  do {                                                           \
    auto r = (call);                                             \
    if (r != ncclSuccess) {                                      \
      std::fprintf(stderr, "NCCL: %s\n", ncclGetErrorString(r)); \
      std::exit(1);                                              \
    }                                                            \
  } while (0)
#else
#include "uccl_lite/pro6000_allreduce.cuh"
#endif
using namespace uccl_lite;

struct Config {
  size_t begin = 1024, end = 1UL << 30, capacity = 1UL << 30;
  size_t tile = 16UL << 20;
  int blocks = 16, threads = 256, seed = 1, workers = 1, streams = 1;
  int warmups = 10, iterations = 20;
  bool profile = false;
  std::string mode = "chunk-ce";
};

#ifndef UCCL_LITE_NCCL_REFERENCE
struct PeerContext {
  std::unique_ptr<mscclpp::Communicator> comm;
  std::array<std::unique_ptr<mscclpp::Communicator>, 3> transportComms;
  std::array<std::unique_ptr<mscclpp::Communicator>, 2> ringComms;
  std::array<mscclpp::Communicator*, 3> transport;
  std::shared_ptr<mscclpp::ProxyService> proxy;
  std::vector<std::unique_ptr<mscclpp::MemoryChannel>> memory;
  std::vector<std::unique_ptr<mscclpp::PortChannel>> ports;
  std::array<Chan, 3> mem{};
  std::array<PortChan, 3> rs{}, ag{};
  ChunkChannels* device = nullptr;
  RingChannels* ring = nullptr;
  mscclpp::DeviceSyncer* syncers = nullptr;
  Layout layout{};
  size_t packetBase, packetSlot;
  char* allocation = nullptr;
  char* buffer = nullptr;
  size_t stride;
  std::vector<std::array<Chan, 3>> hostChannels;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> done;
  cudaEvent_t ready = nullptr;
  std::vector<mscclpp::RegisteredMemory> regions;

  PeerContext(std::shared_ptr<mscclpp::TcpBootstrap> bootstrap, int rank,
              Config cfg) {
    comm = std::make_unique<mscclpp::Communicator>(bootstrap);
    // All sweeps use identical 1-GiB capacity, including narrow calibration
    // jobs.
    stride = cfg.capacity + 2 * uccl_lite::guardBytes;
    packetSlot = 4UL << 20;
    packetBase = 7 * stride;
    size_t total = packetBase + 4 * kLevels * packetSlot;
    checkCuda(cudaMalloc(&allocation, total + 2 * uccl_lite::guardBytes));
    checkCuda(cudaMemset(allocation, 0, total + 2 * uccl_lite::guardBytes));
    buffer = allocation + uccl_lite::guardBytes;
    layout.data = layout.send[0] = 0;
    for (int i = 1; i < 4; ++i) layout.send[i] = i * stride;
    for (int i = 0; i < kLevels; ++i) layout.recv[i] = (4 + i) * stride;
    auto full =
        comm->registerMemory(buffer, total, mscclpp::Transport::CudaIpc);
    for (int i = 0; i < 7; ++i)
      regions.push_back(comm->registerMemory(buffer + i * stride, cfg.capacity,
                                             mscclpp::Transport::CudaIpc));
    // This MSCCL++ install shares one CUDA IPC stream per Context. Separate
    // contexts allow PIX, NODE and SYS transfers to overlap without patching
    // it.
    for (int l = 0; l < kLevels; ++l) {
      if (cfg.streams == 3)
        transportComms[l] = std::make_unique<mscclpp::Communicator>(bootstrap);
      transport[l] = cfg.streams == 3 ? transportComms[l].get() : comm.get();
    }
    std::array<mscclpp::Connection, 3> connections;
    std::array<mscclpp::RegisteredMemory, 3> remoteFull;
    std::array<std::array<mscclpp::RegisteredMemory, 7>, 3> remote;
    std::array<std::shared_future<mscclpp::Connection>, 3> connectionF;
    std::array<std::shared_future<mscclpp::RegisteredMemory>, 3> fullF;
    std::array<std::array<std::shared_future<mscclpp::RegisteredMemory>, 7>, 3>
        regionF;
    for (int l = 0; l < kLevels; ++l) {
      int peer = rank ^ (1 << l);
      comm->sendMemory(full, peer, 10);
      fullF[l] = comm->recvMemory(peer, 10);
      for (int r = 0; r < 7; ++r) {
        comm->sendMemory(regions[r], peer, r);
        regionF[l][r] = comm->recvMemory(peer, r);
      }
      connectionF[l] = transport[l]->connect(
          mscclpp::EndpointConfig(mscclpp::Transport::CudaIpc), peer, 100 + l);
    }
    for (int l = 0; l < kLevels; ++l) connections[l] = connectionF[l].get();
    std::array<mscclpp::Semaphore, 3> sem;
    for (int l = 0; l < kLevels; ++l)
      sem[l] = comm->buildSemaphore(connections[l], rank ^ (1 << l)).get();
    for (int l = 0; l < kLevels; ++l) {
      remoteFull[l] = fullF[l].get();
      for (int r = 0; r < 7; ++r) remote[l][r] = regionF[l][r].get();
      memory.push_back(std::make_unique<mscclpp::MemoryChannel>(
          sem[l], remoteFull[l], full, buffer));
    }
    for (int l = 0; l < kLevels; ++l) mem[l] = memory[l]->deviceHandle();
    if (cfg.mode == "host-ce" || cfg.mode == "ring-host-ce") {
      hostChannels.resize(cfg.blocks);
      streams.resize(cfg.blocks);
      done.resize(cfg.blocks);
      checkCuda(
          cudaMalloc(&syncers, cfg.blocks * sizeof(mscclpp::DeviceSyncer)));
      checkCuda(
          cudaMemset(syncers, 0, cfg.blocks * sizeof(mscclpp::DeviceSyncer)));
      checkCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
      for (int lane = 0; lane < cfg.blocks; ++lane) {
        checkCuda(
            cudaStreamCreateWithFlags(&streams[lane], cudaStreamNonBlocking));
        checkCuda(
            cudaEventCreateWithFlags(&done[lane], cudaEventDisableTiming));
        for (int l = 0; l < kLevels; ++l) {
          auto sem =
              comm->buildSemaphore(connections[l], rank ^ (1 << l)).get();
          memory.push_back(std::make_unique<mscclpp::MemoryChannel>(
              sem, remoteFull[l], full, buffer));
          hostChannels[lane][l] = memory.back()->deviceHandle();
        }
      }
      return;
    }
    bool ce = cfg.mode == "drained-ce" || cfg.mode == "staged-ce" ||
              cfg.mode == "chunk-ce" || cfg.mode == "ring-ce";
    if (!ce) return;
    proxy = std::make_shared<mscclpp::ProxyService>();
    std::array<mscclpp::MemoryId, 7> localId;
    std::array<std::array<mscclpp::MemoryId, 7>, 3> remoteId;
    for (int r = 0; r < 7; ++r) localId[r] = proxy->addMemory(regions[r]);
    for (int l = 0; l < kLevels; ++l)
      for (int r = 0; r < 7; ++r)
        remoteId[l][r] = proxy->addMemory(remote[l][r]);
    auto port = [&](mscclpp::SemaphoreId sid, int l, int dst, int src) {
      ports.push_back(std::make_unique<mscclpp::PortChannel>(
          proxy->portChannel(sid, remoteId[l][dst], localId[src])));
      return ports.back()->deviceHandle();
    };
    if (cfg.mode == "ring-ce") {
      std::vector<RingChannels> handles(cfg.blocks);
      int next = ringLevel(rank, 1), prev = ringLevel(rank, -1);
      std::array<std::array<mscclpp::Connection, 3>, 2> ringConnections;
      if (cfg.streams == 2) {
        // The runtime has one copy stream per Context. Alternating lanes
        // between two Contexts lets signal transfers overlap peer payloads.
        for (int context = 0; context < 2; ++context) {
          ringComms[context] =
              std::make_unique<mscclpp::Communicator>(bootstrap);
          std::array<std::shared_future<mscclpp::Connection>, 3> futures;
          for (int l = 0; l < kLevels; ++l)
            futures[l] = ringComms[context]->connect(
                mscclpp::EndpointConfig(mscclpp::Transport::CudaIpc),
                rank ^ (1 << l), 200 + context * 3 + l);
          for (int l = 0; l < kLevels; ++l)
            ringConnections[context][l] = futures[l].get();
        }
      }
      for (int lane = 0; lane < cfg.blocks; ++lane) {
        for (int l = 0; l < kLevels; ++l) {
          auto sid =
              cfg.streams == 2
                  ? proxy->buildAndAddSemaphore(*ringComms[lane % 2],
                                                ringConnections[lane % 2][l])
                  : proxy->buildAndAddSemaphore(*transport[l], connections[l]);
          auto sem =
              comm->buildSemaphore(connections[l], rank ^ (1 << l)).get();
          memory.push_back(std::make_unique<mscclpp::MemoryChannel>(
              sem, remoteFull[l], full, buffer));
          if (l == next) {
            handles[lane].first = port(sid, l, 4, 0);
            handles[lane].reduced = port(sid, l, 4, 1);
            handles[lane].gather = port(sid, l, 0, 0);
            handles[lane].drained = memory.back()->deviceHandle();
          }
          if (l == prev) {
            handles[lane].incoming = port(sid, l, 0, 0);
            handles[lane].acknowledge = memory.back()->deviceHandle();
          }
        }
      }
      checkCuda(
          cudaMalloc(&syncers, cfg.blocks * sizeof(mscclpp::DeviceSyncer)));
      checkCuda(
          cudaMemset(syncers, 0, cfg.blocks * sizeof(mscclpp::DeviceSyncer)));
      checkCuda(cudaMalloc(&ring, handles.size() * sizeof(RingChannels)));
      checkCuda(cudaMemcpy(ring, handles.data(),
                           handles.size() * sizeof(RingChannels),
                           cudaMemcpyHostToDevice));
    } else if (cfg.mode == "drained-ce" || cfg.mode == "staged-ce") {
      if (kRanks != 8) std::exit(2);
      for (int l = 0; l < kLevels; ++l) {
        auto sid = proxy->buildAndAddSemaphore(*transport[l], connections[l]);
        rs[l] = port(sid, l, 4 + l, l);
        if (cfg.mode == "staged-ce")
          ag[l] = port(sid, l, 0, l == 2 ? 3 : 0);
        else
          ag[l] = port(sid, l, l == 2 ? 6 : (l == 1 ? 5 : 0),
                       l == 2 ? 3 : (l == 1 ? 6 : 5));
      }
    } else {
      std::vector<ChunkChannels> handles(cfg.blocks);
      for (int lane = 0; lane < cfg.blocks; ++lane) {
        for (int l = 0; l < kLevels; ++l) {
          auto sid = proxy->buildAndAddSemaphore(*transport[l], connections[l]);
          handles[lane].rs[l] = port(sid, l, 4 + l, 0);
          handles[lane].ag[l] = port(sid, l, 0, 0);
          auto sem =
              comm->buildSemaphore(connections[l], rank ^ (1 << l)).get();
          memory.push_back(std::make_unique<mscclpp::MemoryChannel>(
              sem, remoteFull[l], full, buffer));
          handles[lane].ack[l] = memory.back()->deviceHandle();
        }
      }
      checkCuda(
          cudaMalloc(&syncers, cfg.blocks * sizeof(mscclpp::DeviceSyncer)));
      checkCuda(
          cudaMemset(syncers, 0, cfg.blocks * sizeof(mscclpp::DeviceSyncer)));
      checkCuda(cudaMalloc(&device, handles.size() * sizeof(ChunkChannels)));
      checkCuda(cudaMemcpy(device, handles.data(),
                           handles.size() * sizeof(ChunkChannels),
                           cudaMemcpyHostToDevice));
    }
    proxy->startProxy();
  }
  void hostLaunch(size_t count, int rank, Config cfg, cudaStream_t caller) {
    size_t tile = std::min(count, cfg.tile / 4);
    int lanes = std::min<size_t>(cfg.blocks, count / tile);
    checkCuda(cudaEventRecord(ready, caller));
    for (int lane = 0; lane < lanes; ++lane)
      checkCuda(cudaStreamWaitEvent(streams[lane], ready));
    for (size_t offset = 0, index = 0; offset < count;
         offset += tile, ++index) {
      int lane = index % lanes;
      auto work = streams[lane];
      if (cfg.mode == "ring-host-ce") {
        int position = ringPosition(rank);
        auto next = hostChannels[lane][ringLevel(rank, 1)];
        auto prev = hostChannels[lane][ringLevel(rank, -1)];
        char* remote = static_cast<char*>(next.dst_);
        size_t shard = tile / kRanks;
        for (int phase = 0; phase < 2 * (kRanks - 1); ++phase) {
          bool rs = phase < kRanks - 1;
          int step = rs ? phase : phase - (kRanks - 1);
          int sendShard = (position + (rs ? 0 : 1) - step + kRanks) % kRanks;
          int recvShard = (sendShard - 1 + kRanks) % kRanks;
          size_t send = offset + sendShard * shard;
          size_t keep = offset + recvShard * shard;
          size_t src = rs && step ? layout.send[1] : layout.data;
          size_t dst = rs ? layout.recv[0] : layout.data;
          checkCuda(cudaMemcpyAsync(remote + dst + send * sizeof(float),
                                    buffer + src + send * sizeof(float),
                                    shard * sizeof(float),
                                    cudaMemcpyDeviceToDevice, work));
          arRingHostPhase<<<rs ? cfg.workers : 1, rs ? cfg.threads : 32, 0,
                            work>>>(
              next, prev, reinterpret_cast<float*>(buffer + layout.data),
              reinterpret_cast<float*>(buffer + layout.send[1]),
              reinterpret_cast<float*>(buffer + layout.recv[0]), keep, shard,
              rs, step == kRanks - 2, syncers + lane);
        }
        continue;
      }
      size_t segBase = offset, segN = tile;
      for (int phase = 0; phase < 2 * kLevels; ++phase) {
        bool rs = phase < kLevels;
        int l = rs ? phase : 2 * kLevels - 1 - phase;
        size_t half = rs ? segN / 2 : segN;
        size_t keep = rs ? segBase + ((rank >> l) & 1) * half : segBase;
        size_t send = rs ? segBase + (((rank >> l) & 1) ^ 1) * half : segBase;
        char* remote = static_cast<char*>(hostChannels[lane][l].dst_);
        size_t target = rs ? layout.recv[l] : layout.data;
        checkCuda(cudaMemcpyAsync(remote + target + send * 4, buffer + send * 4,
                                  half * 4, cudaMemcpyDeviceToDevice, work));
        arHostPhase<<<cfg.workers, cfg.threads, 0, work>>>(
            hostChannels[lane][l], reinterpret_cast<float*>(buffer),
            reinterpret_cast<float*>(buffer + layout.recv[l]), keep, half, rs,
            syncers + lane);
        if (rs) {
          segBase = keep;
          segN = half;
        } else {
          segBase = std::min(segBase, segBase ^ segN);
          segN *= 2;
        }
      }
    }
    for (int lane = 0; lane < lanes; ++lane) {
      checkCuda(cudaEventRecord(done[lane], streams[lane]));
      checkCuda(cudaStreamWaitEvent(caller, done[lane]));
    }
  }
  Layout packets(uint32_t flag) {
    auto l = layout;
    for (int p = 0; p < 2 * kLevels; ++p)
      l.pkt[p] = packetBase + (2 * p + (flag & 1)) * packetSlot;
    return l;
  }
  void guards(size_t bytes, cudaStream_t stream) {
    for (int r = 0; r < 7; ++r)
      uccl_lite::setGuard(buffer + r * stride, bytes, stream);
  }
  unsigned long long verifyGuards(size_t bytes) {
    unsigned long long bad = 0;
    for (int r = 0; r < 7; ++r)
      bad += uccl_lite::checkGuard(buffer + r * stride, bytes);
    return bad;
  }
  ~PeerContext() {
    if (proxy) proxy->stopProxy();
    for (auto work : streams) checkCuda(cudaStreamDestroy(work));
    for (auto event : done) checkCuda(cudaEventDestroy(event));
    if (ready) checkCuda(cudaEventDestroy(ready));
    if (device) checkCuda(cudaFree(device));
    if (ring) checkCuda(cudaFree(ring));
    if (syncers) checkCuda(cudaFree(syncers));
    // Registrations and IPC mappings are destroyed before their backing memory.
    ports.clear();
    memory.clear();
    proxy.reset();
    regions.clear();
    comm.reset();
    for (auto& transportComm : transportComms) transportComm.reset();
    for (auto& ringComm : ringComms) ringComm.reset();
    checkCuda(cudaFree(allocation));
  }
};
#endif

static void runRank(int rank, mscclpp::UniqueId id, Config cfg) {
  uccl_lite::pinRank(rank);
  checkCuda(cudaSetDevice(rank));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, kRanks);
  bootstrap->initialize(id);
  cudaStream_t stream;
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  float* data;
#ifdef UCCL_LITE_NCCL_REFERENCE
  std::array<ncclUniqueId, kRanks> ids{};
  if (!rank) NCCL_CHECK(ncclGetUniqueId(&ids[0]));
  bootstrap->allGather(ids.data(), sizeof(ncclUniqueId));
  ncclComm_t nccl;
  NCCL_CHECK(ncclCommInitRank(&nccl, kRanks, ids[0], rank));
  char* allocation;
  checkCuda(cudaMalloc(&allocation, cfg.capacity + 2 * uccl_lite::guardBytes));
  data = reinterpret_cast<float*>(allocation + uccl_lite::guardBytes);
  auto guards = [&](size_t bytes) {
    uccl_lite::setGuard(reinterpret_cast<char*>(data), bytes, stream);
  };
  auto verifyGuards = [&](size_t bytes) {
    return uccl_lite::checkGuard(reinterpret_cast<char*>(data), bytes);
  };
  auto launch = [&](size_t count) {
    NCCL_CHECK(
        ncclAllReduce(data, data, count, ncclFloat, ncclSum, nccl, stream));
    checkCuda(cudaGetLastError());
  };
#else
  int active = 0;
  cudaDeviceProp property;
  checkCuda(cudaGetDeviceProperties(&property, rank));
  int grid = cfg.blocks;
  if (cfg.mode == "ring-ce") {
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, arRingCE,
                                                            cfg.threads, 0));
    grid *= cfg.workers;
  } else if (cfg.mode == "chunk-ce") {
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, arChunkCE,
                                                            cfg.threads, 0));
    grid *= cfg.workers;
  } else if (cfg.mode == "ring-host-ce") {
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, arRingHostPhase, cfg.threads, 0));
    grid *= cfg.workers;
  } else if (cfg.mode == "host-ce") {
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, arHostPhase, cfg.threads, 0));
    grid *= cfg.workers;
  } else if (cfg.mode == "drained-ce") {
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, arDrainedCE, cfg.threads, 0));
  } else if (cfg.mode == "staged-ce") {
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, arStagedCE,
                                                            cfg.threads, 0));
  } else {
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, ar8Put,
                                                            cfg.threads, 0));
  }
  if (grid > active * property.multiProcessorCount) {
    std::fprintf(stderr,
                 "Cooperative grid exceeds device residency capacity\n");
    std::exit(2);
  }
  PeerContext peer(bootstrap, rank, cfg);
  data = reinterpret_cast<float*>(peer.buffer);
  uint32_t flag = 1;
  unsigned long long* profile = nullptr;
  if (cfg.profile) {
    checkCuda(cudaMalloc(&profile, 12 * sizeof(*profile)));
    checkCuda(cudaMemset(profile, 0, 12 * sizeof(*profile)));
  }
  auto guards = [&](size_t bytes) { peer.guards(bytes, stream); };
  auto verifyGuards = [&](size_t bytes) { return peer.verifyGuards(bytes); };
  auto launch = [&](size_t count) {
    auto l = peer.packets(flag);
    if (cfg.mode == "ring-ce") {
      size_t tile = std::min(cfg.tile / 4, count);
      int lanes = std::min<size_t>(cfg.blocks, count / tile);
      arRingCE<<<lanes * cfg.workers, cfg.threads, 0, stream>>>(
          peer.ring, peer.buffer, rank, count, tile, l, peer.syncers,
          cfg.workers);
    } else if (cfg.mode == "host-ce" || cfg.mode == "ring-host-ce") {
      peer.hostLaunch(count, rank, cfg, stream);
    } else if (cfg.mode == "ll") {
      int blocks = std::min<size_t>(cfg.blocks, count / (kRanks * 2));
      if (count * 4 > peer.packetSlot || count % (blocks * kRanks * 2))
        std::exit(2);
      arChunkLL<<<blocks, cfg.threads, 0, stream>>>(
          peer.mem[0], peer.mem[1], peer.mem[2], rank, count, l, flag++);
    } else if (cfg.mode == "put") {
      ar8Put<<<cfg.blocks, cfg.threads, 0, stream>>>(
          peer.mem[0], peer.mem[1], peer.mem[2], rank, count, l);
    } else if (cfg.mode == "staged-ce") {
      arStagedCE<<<cfg.blocks, cfg.threads, 0, stream>>>(
          peer.rs[0], peer.rs[1], peer.rs[2], peer.ag[0], peer.ag[1],
          peer.ag[2], data, rank, count, l, profile, peer.mem[0]);
    } else if (cfg.mode == "drained-ce") {
      arDrainedCE<<<cfg.blocks, cfg.threads, 0, stream>>>(
          peer.rs[0], peer.rs[1], peer.rs[2], peer.ag[0], peer.ag[1],
          peer.ag[2], data, rank, count, l, profile, peer.mem[0]);
    } else {
      size_t tile = std::min(cfg.tile / 4, count);
      int lanes = std::min<size_t>(cfg.blocks, count / tile);
      arChunkCE<<<lanes * cfg.workers, cfg.threads, 0, stream>>>(
          peer.device, peer.buffer, rank, count, tile, l, profile, peer.syncers,
          cfg.workers);
    }
    checkCuda(cudaGetLastError());
  };
#endif
  unsigned long long* errors;
  checkCuda(cudaMalloc(&errors, sizeof(*errors)));
  if (!rank)
    std::printf(
        "# backend=%s ranks=%d FP32 in-place SUM warmups=%d continuous=%d "
        "capacity=%zu mode=%s blocks=%d threads=%d tile=%zu seed=%d\n",
#ifdef UCCL_LITE_NCCL_REFERENCE
        "NCCL CE",
#else
        "UCCL Lite",
#endif
        kRanks, cfg.warmups, cfg.iterations, cfg.capacity, cfg.mode.c_str(),
        cfg.blocks, cfg.threads, cfg.tile, cfg.seed);
  if (!rank)
    std::printf("# workers=%d copy_contexts=%d profile=%d\n", cfg.workers,
                cfg.streams, cfg.profile);
  std::fflush(stdout);
  for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
    size_t count = bytes / 4;
    unsigned long long bad = 0;
    guards(bytes);
    checkCuda(cudaStreamSynchronize(stream));
    bootstrap->barrier();
    // Changing inputs and repeated calls exercise every buffer reuse boundary.
    for (int trial = 0; trial < 3; ++trial) {
      int seed = cfg.seed + trial;
      uccl_lite::initialize<<<256, 256, 0, stream>>>(data, count, rank, seed);
      checkCuda(cudaStreamSynchronize(stream));
      bootstrap->barrier();
      for (int call = 1; call <= 3; ++call) {
        launch(count);
        auto outputBad = uccl_lite::checkOutput(data, count, kRanks, seed, call,
                                                stream, errors);
        auto guardBad = verifyGuards(bytes);
        bad += outputBad + guardBad;
        if (outputBad || guardBad)
          std::fprintf(stderr,
                       "CHECK_FAILURE rank=%d bytes=%zu seed=%d calls=%d "
                       "output=%llu guards=%llu\n",
                       rank, bytes, seed, call, outputBad, guardBad);
        bootstrap->barrier();
      }
    }
    uccl_lite::initialize<<<256, 256, 0, stream>>>(data, count, rank,
                                                   cfg.seed + 10);
    checkCuda(cudaStreamSynchronize(stream));
    bootstrap->barrier();
    for (int i = 0; i < cfg.warmups; ++i) launch(count);
    checkCuda(cudaStreamSynchronize(stream));
    uccl_lite::rankThreadPlacement(rank, true, bytes == cfg.begin);
    bootstrap->barrier();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iterations; ++i) launch(count);
    checkCuda(cudaStreamSynchronize(stream));
    auto stop = std::chrono::steady_clock::now();
    uccl_lite::rankThreadPlacement(rank, false, false);
    double us =
        std::chrono::duration<double, std::micro>(stop - start).count() /
        cfg.iterations;
    auto outputBad =
        uccl_lite::checkOutput(data, count, kRanks, cfg.seed + 10,
                               cfg.warmups + cfg.iterations, stream, errors);
    auto guardBad = verifyGuards(bytes);
    bad += outputBad + guardBad;
    if (outputBad || guardBad)
      std::fprintf(stderr,
                   "CHECK_FAILURE rank=%d bytes=%zu seed=%d calls=%d "
                   "output=%llu guards=%llu\n",
                   rank, bytes, cfg.seed + 10, cfg.warmups + cfg.iterations,
                   outputBad, guardBad);
    std::array<double, kRanks> times{};
    std::array<unsigned long long, kRanks> failures{};
    times[rank] = us;
    failures[rank] = bad;
    bootstrap->allGather(times.data(), sizeof(double));
    bootstrap->allGather(failures.data(), sizeof(unsigned long long));
    auto worst = *std::max_element(times.begin(), times.end());
    auto failed = *std::max_element(failures.begin(), failures.end());
    if (!rank) {
      std::printf("RESULT %zu %.6f %.6f %llu\n", bytes, worst,
                  bytes * (2.0 * (kRanks - 1) / kRanks) / (worst * 1000),
                  failed);
      for (int r = 0; r < kRanks; ++r)
        std::printf("RANK %zu %d %.6f %llu\n", bytes, r, times[r], failures[r]);
      std::fflush(stdout);
    }
    if (failed) std::exit(3);
    bootstrap->barrier();
    if (bytes > cfg.end / 2) break;
  }
#ifndef UCCL_LITE_NCCL_REFERENCE
  if (profile) {
    std::array<unsigned long long, 12> values{};
    checkCuda(cudaMemcpy(values.data(), profile, sizeof(values),
                         cudaMemcpyDeviceToHost));
    for (int p = 0; p < 2 * kLevels; ++p)
      std::printf(
          "PROFILE rank=%d phase=%d transfer_ack_cycles=%llu "
          "local_cycles=%llu\n",
          rank, p, values[2 * p], values[2 * p + 1]);
    checkCuda(cudaFree(profile));
  }
#else
  NCCL_CHECK(ncclCommDestroy(nccl));
  checkCuda(cudaFree(allocation));
#endif
  checkCuda(cudaFree(errors));
  checkCuda(cudaStreamDestroy(stream));
  bootstrap->barrier();
}

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    if (++i == argc) return 2;
    std::string value = argv[i];
    if (option == "--begin")
      cfg.begin = std::stoull(value);
    else if (option == "--end")
      cfg.end = std::stoull(value);
    else if (option == "--mode")
      cfg.mode = value;
    else if (option == "--blocks")
      cfg.blocks = std::stoi(value);
    else if (option == "--threads")
      cfg.threads = std::stoi(value);
    else if (option == "--streams")
      cfg.streams = std::stoi(value);
    else if (option == "--workers")
      cfg.workers = std::stoi(value);
    else if (option == "--tile")
      cfg.tile = std::stoull(value);
    else if (option == "--seed")
      cfg.seed = std::stoi(value);
    else if (option == "--profile")
      cfg.profile = std::stoi(value);
    else
      return 2;
  }
  auto pow2 = [](size_t x) { return x && !(x & (x - 1)); };
  if (!pow2(cfg.begin) || !pow2(cfg.end) || !pow2(cfg.tile) ||
      !pow2(cfg.blocks) || cfg.begin < 1024 || cfg.end > cfg.capacity ||
      cfg.begin > cfg.end || cfg.tile < 1024 || cfg.blocks > 128 ||
      (cfg.streams != 1 && cfg.streams != 2 && cfg.streams != 3) ||
      (cfg.streams == 2 && cfg.mode != "ring-ce") || !pow2(cfg.workers) ||
      cfg.blocks * cfg.workers > 128 || cfg.threads < 32 ||
      cfg.threads > 1024 || cfg.threads % 32 || cfg.seed < 0 ||
      cfg.seed > 100000)
    return 2;
  if (cfg.mode != "ll" && cfg.mode != "put" && cfg.mode != "drained-ce" &&
      cfg.mode != "chunk-ce" && cfg.mode != "staged-ce" &&
      cfg.mode != "host-ce" && cfg.mode != "ring-ce" &&
      cfg.mode != "ring-host-ce")
    return 2;
  auto id = mscclpp::TcpBootstrap::createUniqueId();
  std::vector<pid_t> children;
  for (int rank = 1; rank < kRanks; ++rank) {
    pid_t pid = fork();
    if (!pid) {
      runRank(rank, id, cfg);
      std::fflush(stdout);
      // Allow CUDA/profiler atexit handlers to flush child-process traces.
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

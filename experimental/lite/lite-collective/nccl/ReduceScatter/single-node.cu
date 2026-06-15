#ifndef MSCCLPP_NCCL_REDUCESCATTER_SINGLE_NODE_INCLUDED
#error "ReduceScatter/single-node.cu is included by multi-node.cu; do not compile it directly."
#endif

// Included from multi-node.cu so the single-node path can share the same
// file-local context and helper kernels without exposing them as public ABI.

ncclResult_t runLocalFourRankRingReduceScatter(
    RsContext& ctx, void const* sendbuff, void* recvbuff, size_t recvcount,
    size_t bytesPerRank, cudaStream_t stream,
    std::shared_ptr<Communicator> bootstrapComm, void* scratchBuffer,
    size_t scratchBufferSize) {
  (void)recvcount;
  ensureLocalScratchIpc(ctx, bootstrapComm, ctx.rank, ctx.worldSize,
                        ctx.nRanksPerNode, scratchBuffer, scratchBufferSize);
  bool ringEvents = useLocalFourRingEvents();
  if (ringEvents) {
    ensureLocalScratchEvents(ctx, bootstrapComm, ctx.rank, ctx.worldSize,
                             ctx.nRanksPerNode);
  }
  size_t ringChunkBytes =
      std::min(bytesPerRank, configuredLocalFourRingChunkBytes());
  ringChunkBytes = ringChunkBytes / sizeof(float) * sizeof(float);
  if (ringChunkBytes == 0) return ncclInvalidUsage;
  size_t rowStrideBytes = ringChunkBytes;
  size_t slotBytes = 2 * rowStrideBytes;
  if (slotBytes == 0 || slotBytes > scratchBufferSize) {
    return ncclInvalidUsage;
  }
  size_t slotCount = std::min<size_t>(scratchBufferSize / slotBytes, 1024);
  if (ringEvents) {
    slotCount = std::min<size_t>(
        slotCount, static_cast<size_t>(ctx.pipelineSlots) /
                       static_cast<size_t>(ctx.nRanksPerNode - 1));
  }
  if (slotCount == 0) return ncclInvalidUsage;

  int const n = ctx.nRanksPerNode;
  int const nextLocal = (ctx.localRank + 1) % n;
  int const prevLocal = (ctx.localRank + n - 1) % n;
  int const nextIdx = localPeerIndex(ctx.localRank, nextLocal);
  int const prevIdx = localPeerIndex(ctx.localRank, prevLocal);
  char const* send = static_cast<char const*>(sendbuff);
  char* recv = static_cast<char*>(recvbuff);
  char* scratch = static_cast<char*>(scratchBuffer);

  for (size_t chunkOffset = 0; chunkOffset < bytesPerRank;) {
    size_t chunkBytes = std::min(ringChunkBytes, bytesPerRank - chunkOffset);
    size_t chunkElems = chunkBytes / sizeof(float);
    uint64_t epoch = ++ctx.epoch;
    size_t slot = static_cast<size_t>((epoch - 1) % slotCount);
    if (epoch > slotCount) {
      uint64_t reuseEpoch = epoch - slotCount;
      for (int i = 0; i < ctx.nRanksPerNode; ++i) {
        waitForEpoch(ctx.ctrl->localCrossReady[i], reuseEpoch);
      }
      if (ringEvents) {
        for (int peerLocal = 0; peerLocal < ctx.nRanksPerNode; ++peerLocal) {
          if (peerLocal == ctx.localRank) continue;
          int peerIdx = localPeerIndex(ctx.localRank, peerLocal);
          MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
              stream, ctx.remoteCrossEvents[peerIdx][slot], 0));
        }
      }
    }

    size_t slotBase = slot * slotBytes;
    uint64_t signalBase = epoch * 8;

    for (int step = 0; step < n - 1; ++step) {
      int sendChunk = step == 0 ? (ctx.localRank + n - 1) % n
                                : (ctx.localRank - step - 1 + n) % n;
      char const* src =
          step == 0 ? send + static_cast<size_t>(sendChunk) * bytesPerRank +
                          chunkOffset
                    : scratch + slotBase +
                          static_cast<size_t>((step - 1) & 1) * rowStrideBytes;
      char* dst = ctx.remoteScratchPtrs[nextIdx] + slotBase +
                  static_cast<size_t>(step & 1) * rowStrideBytes;
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, chunkBytes,
                                        cudaMemcpyDeviceToDevice, stream));
      size_t eventIndex = 0;
      if (ringEvents) {
        eventIndex = slot * static_cast<size_t>(n - 1) +
                     static_cast<size_t>(step);
        MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCopyEvents[eventIndex],
                                          stream));
      } else {
        waitForCudaStream(stream);
      }

      uint64_t signal = signalBase + static_cast<uint64_t>(step + 1);
      ctx.ctrl->localCopyReady[ctx.localRank].store(
          signal, std::memory_order_release);
      waitForEpoch(ctx.ctrl->localCopyReady[prevLocal], signal);
      if (ringEvents) {
        MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
            stream, ctx.remoteCopyEvents[prevIdx][eventIndex], 0));
      }

      int recvChunk = (ctx.localRank - step - 2 + n) % n;
      char* incoming = scratch + slotBase +
                       static_cast<size_t>(step & 1) * rowStrideBytes;
      char const* local =
          send + static_cast<size_t>(recvChunk) * bytesPerRank + chunkOffset;
      void* out = step == n - 2 ? recv + chunkOffset : incoming;
      ncclResult_t result = launchAdd(local, incoming, out, chunkElems, stream);
      if (result != ncclSuccess) return result;
    }

    if (ringEvents) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCrossEvents[slot], stream));
    } else {
      waitForCudaStream(stream);
    }
    ctx.ctrl->localCrossReady[ctx.localRank].store(epoch,
                                                   std::memory_order_release);
    chunkOffset += chunkBytes;
  }
  return ncclSuccess;
}

ncclResult_t runP2pRingReduceScatter(void const* sendbuff, void* recvbuff,
                                     size_t recvcount, size_t bytesPerRank,
                                     ncclDataType_t datatype, ncclComm_t comm,
                                     cudaStream_t stream, int rank, int nRanks,
                                     void* scratchBuffer,
                                     size_t scratchBufferSize) {
  if (datatype != ncclFloat32) return ncclInvalidUsage;
  size_t slotBytes = bytesPerRank;
  if (slotBytes == 0 || scratchBufferSize < 2 * slotBytes) {
    return ncclInvalidUsage;
  }
  int next = (rank + 1) % nRanks;
  int prev = (rank + nRanks - 1) % nRanks;
  auto const* send = static_cast<char const*>(sendbuff);
  auto* scratch = static_cast<char*>(scratchBuffer);

  for (int step = 0; step < nRanks - 1; ++step) {
    int sendChunk = step == 0 ? (rank + nRanks - 1) % nRanks
                              : (rank - step - 1 + nRanks) % nRanks;
    char const* src =
        step == 0 ? send + static_cast<size_t>(sendChunk) * bytesPerRank
                  : scratch + static_cast<size_t>((step - 1) & 1) * slotBytes;
    char* dst = scratch + static_cast<size_t>(step & 1) * slotBytes;

    ncclResult_t result = ncclGroupStart();
    if (result != ncclSuccess) return result;
    ncclResult_t enqueueResult =
        ncclSend(src, recvcount, datatype, next, comm, stream);
    if (enqueueResult == ncclSuccess) {
      enqueueResult = ncclRecv(dst, recvcount, datatype, prev, comm, stream);
    }
    result = ncclGroupEnd();
    if (enqueueResult != ncclSuccess) return enqueueResult;
    if (result != ncclSuccess) return result;

    int recvChunk = (rank - step - 2 + nRanks) % nRanks;
    char const* local =
        send + static_cast<size_t>(recvChunk) * bytesPerRank;
    void* out = step == nRanks - 2 ? recvbuff : dst;
    result = launchAdd(local, dst, out, recvcount, stream);
    if (result != ncclSuccess) return result;
  }
  return ncclSuccess;
}

ncclResult_t runLocalFourRankReduceScatter(
    RsContext& ctx, void const* sendbuff, void* recvbuff, size_t recvcount,
    size_t bytesPerRank, cudaStream_t stream,
    std::shared_ptr<Communicator> bootstrapComm, void* scratchBuffer,
    size_t scratchBufferSize) {
  if (useLocalFourRingFor(bytesPerRank)) {
    return runLocalFourRankRingReduceScatter(
        ctx, sendbuff, recvbuff, recvcount, bytesPerRank, stream,
        bootstrapComm, scratchBuffer, scratchBufferSize);
  }
  if (scratchBuffer == nullptr || scratchBufferSize == 0) {
    return ncclInvalidArgument;
  }
  ensureLocalScratchIpc(ctx, bootstrapComm, ctx.rank, ctx.worldSize,
                        ctx.nRanksPerNode, scratchBuffer, scratchBufferSize);
  ensureLocalScratchEvents(ctx, bootstrapComm, ctx.rank, ctx.worldSize,
                           ctx.nRanksPerNode);

  size_t rowStrideBytes = (bytesPerRank + sizeof(float) - 1) / sizeof(float) *
                          sizeof(float);
  size_t flagBytes = 256;
  size_t slotBytes =
      flagBytes + static_cast<size_t>(ctx.nRanksPerNode) * rowStrideBytes;
  if (slotBytes == 0 || slotBytes > scratchBufferSize) {
    return ncclInvalidUsage;
  }
  size_t slotCount = scratchBufferSize / slotBytes;
  slotCount = std::min<size_t>(slotCount, 1024);
  if (slotCount == 0) return ncclInvalidUsage;
  bool rowStrideInitialized = ctx.localFourRowStrideBytes != 0;
  bool rowStrideChanged = rowStrideInitialized &&
                          ctx.localFourRowStrideBytes != rowStrideBytes;
  if (rowStrideChanged) {
    uint64_t prevEpoch = ctx.epoch;
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    ctx.ctrl->localScratchDone[ctx.localRank].store(
        prevEpoch, std::memory_order_release);
    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->localScratchDone[i], prevEpoch);
    }
  }
  if (!rowStrideInitialized || rowStrideChanged) {
    ctx.localFourLayoutStartEpoch = ctx.epoch;
  }
  ctx.localFourRowStrideBytes = rowStrideBytes;

  uint64_t epoch = ++ctx.epoch;
  uint64_t layoutEpoch = epoch - ctx.localFourLayoutStartEpoch;
  size_t slot = static_cast<size_t>((layoutEpoch - 1) % slotCount);
  int eventSlot = static_cast<int>(slot);
  bool useGpuFlags = ctx.ctrlDeviceSlab != nullptr && useLocalFourGpuFlags();
  bool useDeviceFlagScatter =
      !useGpuFlags && slotCount >= 16 &&
      bytesPerRank <= configuredLocalFourDeviceFlagMaxBytes();
  bool useDeviceFlags = useDeviceFlagScatter;
  bool skipSelfCopy = !useDeviceFlags && !useGpuFlags &&
                      bytesPerRank <= 32 * 1024;
  bool useParallelCopies = !useDeviceFlags && !useGpuFlags && !skipSelfCopy &&
                           bytesPerRank >= 64 * 1024 &&
                           useLocalFourParallelCopies();
  if (useParallelCopies) ensureLocalParallelCopyResources(ctx);
  if (layoutEpoch > slotCount && !rowStrideChanged) {
    uint64_t reuseEpoch = epoch - slotCount;
    if (useDeviceFlags) {
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      bootstrapComm->bootstrap()->barrier();
    } else {
      for (int i = 0; i < ctx.nRanksPerNode; ++i) {
        waitForEpoch(ctx.ctrl->localCrossReady[i], reuseEpoch);
      }
      if (!useGpuFlags) {
        for (int peerLocal = 0; peerLocal < ctx.nRanksPerNode; ++peerLocal) {
          if (peerLocal == ctx.localRank) continue;
          int peerIdx = localPeerIndex(ctx.localRank, peerLocal);
          MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
              stream, ctx.remoteCrossEvents[peerIdx][eventSlot], 0));
        }
      }
    }
  }

  auto const* send = static_cast<char const*>(sendbuff);
  auto* scratch = static_cast<char*>(scratchBuffer);
  size_t slotBase = slot * slotBytes;
  size_t dataBase = slotBase + flagBytes;
  unsigned long long* flagPtrs[4] = {};
  char* dstPtrs[4] = {};
  if (useParallelCopies) {
    MSCCLPP_CUDATHROW(
        cudaEventRecord(ctx.localCopyStartEvents[eventSlot], stream));
  }
  for (int targetLocal = 0; targetLocal < ctx.nRanksPerNode; ++targetLocal) {
    size_t sourceOffset = static_cast<size_t>(targetLocal) * bytesPerRank;
    char* targetScratch = nullptr;
    if (targetLocal == ctx.localRank) {
      targetScratch = scratch;
    } else {
      int peerIdx = localPeerIndex(ctx.localRank, targetLocal);
      targetScratch = ctx.remoteScratchPtrs[peerIdx];
    }
    char* dst = targetScratch + dataBase +
                static_cast<size_t>(ctx.localRank) * rowStrideBytes;
    dstPtrs[targetLocal] = dst;
    flagPtrs[targetLocal] = reinterpret_cast<unsigned long long*>(
        targetScratch + slotBase +
        static_cast<size_t>(ctx.localRank) * sizeof(unsigned long long));
    if (!useDeviceFlagScatter &&
        !(skipSelfCopy && targetLocal == ctx.localRank)) {
      cudaStream_t copyStream = stream;
      if (useParallelCopies) {
        copyStream = ctx.localCopyStreams[static_cast<size_t>(targetLocal)];
        MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
            copyStream, ctx.localCopyStartEvents[eventSlot], 0));
      }
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, send + sourceOffset, bytesPerRank,
                                        cudaMemcpyDeviceToDevice, copyStream));
      if (useParallelCopies) {
        MSCCLPP_CUDATHROW(cudaEventRecord(
            ctx.localCopyDoneEvents[static_cast<size_t>(targetLocal)],
            copyStream));
      }
    }
  }
  if (useParallelCopies) {
    for (int targetLocal = 0; targetLocal < ctx.nRanksPerNode; ++targetLocal) {
      if (skipSelfCopy && targetLocal == ctx.localRank) continue;
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
          stream, ctx.localCopyDoneEvents[static_cast<size_t>(targetLocal)],
          0));
    }
  }

  if (useDeviceFlags) {
    ncclResult_t result = launchLocalFourRankTinyScatter(
        sendbuff, dstPtrs[0], dstPtrs[1], dstPtrs[2], dstPtrs[3], flagPtrs[0],
        flagPtrs[1], flagPtrs[2], flagPtrs[3], recvcount, epoch, stream);
    if (result != ncclSuccess) return result;
    result = launchLocalFourRankDeviceFlagReduce(
        reinterpret_cast<unsigned long long volatile*>(scratch + slotBase),
        scratch + dataBase, recvbuff, recvcount, rowStrideBytes, epoch, stream);
    if (result != ncclSuccess) return result;
    ctx.ctrl->localCrossReady[ctx.localRank].store(epoch,
                                                   std::memory_order_release);
    return ncclSuccess;
  }

  if (useGpuFlags) {
    ncclResult_t result = launchLocalFourRankScratchWaitReduce(
        ctx.ctrlDeviceSlab, offsetof(RsControl, localCopyReady),
        offsetof(RsControl, localCrossReady), scratch + dataBase, recvbuff,
        recvcount, rowStrideBytes, ctx.localRank, epoch, stream);
    if (result != ncclSuccess) return result;
    return ncclSuccess;
  } else {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCopyEvents[eventSlot], stream));
    ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->localCopyReady[i], epoch);
    }
    for (int peerLocal = 0; peerLocal < ctx.nRanksPerNode; ++peerLocal) {
      if (peerLocal == ctx.localRank) continue;
      int peerIdx = localPeerIndex(ctx.localRank, peerLocal);
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
          stream, ctx.remoteCopyEvents[peerIdx][eventSlot], 0));
    }

    ncclResult_t result =
        skipSelfCopy
            ? launchLocalFourRankScratchReduceSelf(
                  scratch + dataBase,
                  send + static_cast<size_t>(ctx.localRank) * bytesPerRank,
                  recvbuff, recvcount, rowStrideBytes, ctx.localRank, stream)
            : launchLocalFourRankScratchReduce(
                  scratch + dataBase, recvbuff, recvcount, rowStrideBytes,
                  stream);
    if (result != ncclSuccess) return result;

    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCrossEvents[eventSlot], stream));
    ctx.ctrl->localCrossReady[ctx.localRank].store(epoch,
                                                   std::memory_order_release);
  }
  return ncclSuccess;
}

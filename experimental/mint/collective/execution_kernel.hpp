// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_KERNEL_HPP_
#define MSCCLPP_EXECUTION_KERNEL_HPP_

#include "executor.hpp"
#if defined(ENABLE_NPKIT)
#include "npkit.hpp"
#endif
#include "concurrency_device.hpp"
#include "device.hpp"
#include "execution_common.hpp"
#include "gpu_data_types.hpp"
#include "memory_channel.hpp"
#include "packet_device.hpp"
#include "port_channel.hpp"
#include "reduce_kernel.hpp"
#include "switch_channel_device.hpp"
namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE)

#define MAX_DEVICE_FUNCTIONS_IN_PIPELINE 16
__device__ DeviceSyncer deviceSyncers[MAX_DEVICE_SYNCERS];
__device__ DeviceSemaphore* deviceSemaphores;

__shared__ DeviceHandle<BaseMemoryChannel>* memoryChannels_;
__shared__ DeviceHandle<BasePortChannel>* portChannels_;
__shared__ DeviceHandle<SwitchChannel>* nvlsChannels_;
__shared__ void** memoryChannelBufferPtrs_;
__shared__ MemoryId* portChannelBufferIds_;
__shared__ BufferType* memoryChannelBufferTypes_;
__shared__ BufferType* portChannelBufferTypes_;
__shared__ uint32_t flag_;
__shared__ uint32_t scratchChunkSize_;
__shared__ uint32_t scratchOffset_;
__shared__ MemoryId localMemoryIdBegin_;
#if defined(ENABLE_NPKIT)
__shared__ NpKitEvent* eventBuffer_;
#endif

template <typename T>
MSCCLPP_DEVICE_INLINE T* getBuffer(T* input, T* output, T* scratch,
                                   BufferType bufferType) {
  if (bufferType == BufferType::INPUT) {
    return input;
  }
  if (bufferType == BufferType::OUTPUT) {
    return output;
  }
  if (bufferType == BufferType::SCRATCH) {
    return reinterpret_cast<T*>((char*)scratch + scratchOffset_);
  }
  return nullptr;
}

template <bool ReuseScratch>
MSCCLPP_DEVICE_INLINE uint32_t getOffset(BufferType bufferType,
                                         uint32_t offset) {
  if constexpr (!ReuseScratch) {
    return offset;
  } else if (bufferType != BufferType::SCRATCH) {
    return offset;
  } else {
    return offset % scratchChunkSize_;
  }
}

template <typename T, typename PacketType, bool ReuseScratch = false>
MSCCLPP_DEVICE_INLINE void executeDeviceFunction(
    Operation const& op, T* input, T* output, T* scratch,
    uint8_t* nSteps = nullptr, uint32_t offset = 0,
    uint32_t unitSize = UINT32_MAX);

MSCCLPP_DEVICE_INLINE void handleNop() { __syncthreads(); }

MSCCLPP_DEVICE_INLINE void handleBarrier(Operation const& op) {
  DeviceSyncer* syncer = &deviceSyncers[op.deviceSyncerIndex];
  syncer->sync(op.nThreadBlocks);
}

template <bool Relaxed = false>
MSCCLPP_DEVICE_INLINE void handleSignal(Operation const& op) {
  int nChannels = op.nChannels;
  ChannelType chType = op.channelType;
  uint8_t const* channelIndex = op.channelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::MEMORY) {
    if constexpr (Relaxed) {
      memoryChannels_[channelIndex[tid]].relaxedSignal();
    } else {
      memoryChannels_[channelIndex[tid]].signal();
    }
    return;
  }
  if (tid < nChannels && chType == ChannelType::PORT) {
    portChannels_[channelIndex[tid]].signal();
  }
}

template <bool Relaxed = false>
MSCCLPP_DEVICE_INLINE void handleWait(Operation const& op) {
  int nChannels = op.nChannels;
  ChannelType chType = op.channelType;
  uint8_t const* channelIndex = op.channelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::MEMORY) {
    if constexpr (Relaxed) {
      memoryChannels_[channelIndex[tid]].relaxedWait();
    } else {
      memoryChannels_[channelIndex[tid]].wait();
    }
    return;
  }
  if (tid < nChannels && chType == ChannelType::PORT) {
    portChannels_[channelIndex[tid]].wait();
  }
}

MSCCLPP_DEVICE_INLINE void handleFlush(Operation const& op) {
  int nChannels = op.nOutputs;
  uint8_t const* channelIndexes = op.channelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels) {
    portChannels_[channelIndexes[tid]].flush();
  }
}

template <bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void handleGet(Operation const& op, void* input,
                                     void* output, void* scratch,
                                     uint32_t offset, uint32_t unitSize) {
  const uint32_t count = op.nInputs;
  uint32_t const* sizes = op.inputBufferSizes;
  uint32_t const* srcOffsets = op.inputOffsets;
  uint32_t const* dstOffsets = op.outputOffsets;
  for (uint32_t i = 0; i < count; i++) {
    uint32_t dstOffset =
        dstOffsets[i] +
        getOffset<ReuseScratch>(op.outputBufferRefs[i].type, offset);
    uint32_t srcOffset =
        srcOffsets[i] +
        getOffset<ReuseScratch>(
            memoryChannelBufferTypes_[op.inputBufferRefs[i].id], offset);
    uint32_t size = min(sizes[i] - offset, unitSize);
    char* remoteMemory =
        static_cast<char*>(memoryChannelBufferPtrs_[op.inputBufferRefs[i].id]);
    mscclpp::copy(static_cast<char*>(getBuffer(input, output, scratch,
                                               op.outputBufferRefs[i].type)) +
                      srcOffset,
                  remoteMemory + dstOffset, size, threadIdx.x, blockDim.x);
  }
}

template <bool ReuseScratch, bool PutWithSignal = false,
          bool PutWithSignalAndFlush = false>
MSCCLPP_DEVICE_INLINE void handlePut(Operation const& op, void* input,
                                     void* output, void* scratch,
                                     uint32_t offset, uint32_t unitSize) {
  ChannelType chType = op.channelType;
  uint32_t count = op.nOutputs;
  uint8_t const* channelIndexes = op.channelIndexes;
  uint32_t const* dstOffsets = op.outputOffsets;
  uint32_t const* srcOffsets = op.inputOffsets;
  uint32_t const* outputSizes = op.outputBufferSizes;
  char* src = static_cast<char*>(
      getBuffer(input, output, scratch, op.inputBufferRefs[0].type));
  if (chType == ChannelType::MEMORY) {
    for (uint32_t i = 0; i < count; i++) {
      uint32_t dstOffset =
          dstOffsets[i] +
          getOffset<ReuseScratch>(
              memoryChannelBufferTypes_[op.outputBufferRefs[i].id], offset);
      uint32_t srcOffset =
          srcOffsets[i] +
          getOffset<ReuseScratch>(op.inputBufferRefs[i].type, offset);
      uint32_t size = min(outputSizes[i] - offset, unitSize);
      char* remoteMemory = static_cast<char*>(
          memoryChannelBufferPtrs_[op.outputBufferRefs[i].id]);
      mscclpp::copy(remoteMemory + dstOffset, src + srcOffset, size,
                    threadIdx.x, blockDim.x);
    }
    return;
  }
  if (chType == ChannelType::PORT) {
    uint32_t tid = threadIdx.x;
    if (tid < count) {
      uint32_t size = min(outputSizes[tid] - offset, unitSize);
      MemoryId dstMemoryId = portChannelBufferIds_[op.outputBufferRefs[tid].id];
      MemoryId srcMemoryId =
          static_cast<MemoryId>(op.inputBufferRefs[tid].type) +
          localMemoryIdBegin_;
      uint32_t dstOffset =
          dstOffsets[tid] +
          getOffset<ReuseScratch>(
              portChannelBufferTypes_[op.outputBufferRefs[tid].id], offset);
      uint32_t srcOffset =
          srcOffsets[tid] +
          getOffset<ReuseScratch>(op.inputBufferRefs[tid].type, offset);
      if constexpr (PutWithSignal) {
        portChannels_[channelIndexes[tid]].putWithSignal(
            dstMemoryId, dstOffset, srcMemoryId, srcOffset, size);
      } else if constexpr (PutWithSignalAndFlush) {
        portChannels_[channelIndexes[tid]].putWithSignalAndFlush(
            dstMemoryId, (uint64_t)dstOffset, srcMemoryId, (uint64_t)srcOffsets,
            size);
      } else {
        portChannels_[channelIndexes[tid]].put(dstMemoryId, dstOffset,
                                               srcMemoryId, srcOffset, size);
      }
    }
  }
}

template <typename T, bool ReuseScratch, bool SendToRemote = true,
          ReduceOp OpType = SUM>
MSCCLPP_DEVICE_INLINE void handleReadReduceSend(Operation const& op,
                                                void* input, void* output,
                                                void* scratch, uint32_t offset,
                                                uint32_t unitSize) {
  const uint32_t size = min(op.inputBufferSizes[0] - offset, unitSize);
  const uint32_t nInt4 = size / sizeof(int4);
  const uint32_t inputOffset4 =
      (op.inputOffsets[0] +
       getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset)) /
      sizeof(int4);
  const uint32_t outputOffset4 =
      (op.outputOffsets[0] +
       getOffset<ReuseScratch>(op.outputBufferRefs[0].type, offset)) /
      sizeof(int4);
  const uint8_t nRemoteInputs = op.nInputs - 1;
  const uint8_t nRemoteOutputs = op.nOutputs - 1;
  uint32_t const* srcOffsets = op.inputOffsets + 1;
  uint32_t const* dstOffsets = op.outputOffsets + 1;
  int4* input4 = static_cast<int4*>(
      getBuffer(input, output, scratch, op.inputBufferRefs[0].type));
  int4* output4 = static_cast<int4*>(
      getBuffer(input, output, scratch, op.outputBufferRefs[0].type));
  for (uint32_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = input4[inputOffset4 + idx];
    for (int index = 0; index < nRemoteInputs; ++index) {
      int4 val;
      uint32_t srcOffset =
          (srcOffsets[index] +
           getOffset<ReuseScratch>(
               memoryChannelBufferTypes_[op.inputBufferRefs[index + 1].id],
               offset)) /
          sizeof(int4);
      void* remoteMemory = static_cast<char*>(
          memoryChannelBufferPtrs_[op.inputBufferRefs[index + 1].id]);
      val = mscclpp::read<int4>(remoteMemory, srcOffset + idx);
      tmp = cal_vector<T, OpType>(tmp, val);
    }
    output4[outputOffset4 + idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nRemoteOutputs; ++index) {
        uint32_t dstOffset =
            (dstOffsets[index] +
             getOffset<ReuseScratch>(
                 memoryChannelBufferTypes_[op.outputBufferRefs[index + 1].id],
                 offset)) /
            sizeof(int4);
        void* remoteMemory = static_cast<char*>(
            memoryChannelBufferPtrs_[op.outputBufferRefs[index + 1].id]);
        mscclpp::write<int4>(remoteMemory, dstOffset + idx, tmp);
      }
    }
  }
  // handle rest of data
  uint32_t processed = nInt4 * sizeof(int4);
  const uint32_t startIdx =
      (op.inputOffsets[0] +
       getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset) +
       processed) /
      sizeof(T);
  const uint32_t endIdx =
      (op.inputOffsets[0] +
       getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset) + size) /
      sizeof(T);
  for (uint32_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = static_cast<T*>(input)[idx];
    for (int index = 0; index < nRemoteInputs; ++index) {
      uint32_t srcOffset =
          (srcOffsets[index] +
           getOffset<ReuseScratch>(
               memoryChannelBufferTypes_[op.inputBufferRefs[index + 1].id],
               offset)) /
          sizeof(T);
      void* remoteMemory = static_cast<char*>(
          memoryChannelBufferPtrs_[op.inputBufferRefs[index + 1].id]);
      tmp = tmp + mscclpp::read<T>(remoteMemory, srcOffset + idx);
    }
    static_cast<T*>(output)[idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nRemoteOutputs; ++index) {
        uint32_t dstOffset =
            (dstOffsets[index] +
             getOffset<ReuseScratch>(
                 memoryChannelBufferTypes_[op.outputBufferRefs[index + 1].id],
                 offset)) /
            sizeof(T);
        void* remoteMemory = static_cast<char*>(
            memoryChannelBufferPtrs_[op.outputBufferRefs[index + 1].id]);
        mscclpp::write<T>(remoteMemory, dstOffset + idx, tmp);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handlePutPackets(Operation const& op, void* input,
                                            void* output, void* scratch) {
  ChannelType chType = op.channelType;
  uint16_t nDstChannels = op.nOutputs;
  uint32_t const* dstOffsets = op.outputOffsets;
  uint32_t const* srcOffsets = op.inputOffsets;
  uint32_t const* sizes = op.inputBufferSizes;
  uint8_t const* channelIndexes = op.channelIndexes;
  void* inputBuff =
      getBuffer(input, output, scratch, op.inputBufferRefs[0].type);
  if (chType == ChannelType::MEMORY) {
    for (int index = 0; index < nDstChannels; ++index) {
      uint32_t size = sizes[index];
      mscclpp::copyToPackets<PacketType>(
          (char*)memoryChannelBufferPtrs_[op.outputBufferRefs[index].id] +
              (dstOffsets[index] << 1) + scratchOffset_,
          (char*)inputBuff + srcOffsets[index], size, threadIdx.x, blockDim.x,
          flag_);
    }
  }
  if (chType == ChannelType::PORT) {
    int tid = threadIdx.x;
    if (tid >= nDstChannels) {
      return;
    }
    // For port channel, we assume src and dst are in packet format
    // TODO: support non-packet format and remove packet format(packet format
    // should be handle in handleReadPutPacket)
    uint32_t size = sizes[tid];
    uint32_t constOffset = op.inputBufferRefs[tid].type == BufferType::SCRATCH
                               ? scratchOffset_
                               : 0;
    uint32_t dstOffset = (dstOffsets[tid] << 1) + scratchOffset_;
    uint32_t srcOffset = (srcOffsets[tid] << 1) + constOffset;
    MemoryId srcMemoryId = static_cast<MemoryId>(op.inputBufferRefs[tid].type);
    MemoryId dstMemoryId = portChannelBufferIds_[op.outputBufferRefs[tid].id];
    portChannels_[channelIndexes[tid]].put(dstMemoryId, dstOffset, srcMemoryId,
                                           srcOffset, size << 1);
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleReadPutPackets(Operation const& op,
                                                void* scratch) {
  uint32_t nOutput = op.nOutputs;
  uint32_t const* dstOffsets = op.outputOffsets;
  uint32_t const* srcOffsets = op.inputOffsets;
  uint8_t const* channelIndexes = op.channelIndexes;
  uint32_t size = op.inputBufferSizes[0];
  ChannelType chType = op.channelType;
  if (chType == ChannelType::MEMORY) {
    size_t nPackets = size / sizeof(PacketPayload<PacketType>);
    PacketType* pkts =
        (PacketType*)((char*)scratch + scratchOffset_ + (srcOffsets[0] << 1));
    for (size_t pktIdx = threadIdx.x; pktIdx < nPackets; pktIdx += blockDim.x) {
      PacketPayload<PacketType> data = pkts[pktIdx].read(flag_);
      PacketType pkt(data, flag_);
      for (uint32_t idx = 0; idx < nOutput; ++idx) {
        size_t offset =
            (scratchOffset_ + (dstOffsets[idx] << 1)) / sizeof(PacketType);
        void* remoteMemory = static_cast<char*>(
            memoryChannelBufferPtrs_[op.outputBufferRefs[idx].id]);
        mscclpp::write<PacketType>(remoteMemory, offset + pktIdx, pkt);
      }
    }
  } else if (chType == ChannelType::PORT) {
    // Ensuring Data Is Ready
    size_t nPackets = size / sizeof(PacketPayload<PacketType>);
    for (size_t pktIdx = threadIdx.x; pktIdx < nPackets; pktIdx += blockDim.x) {
      PacketType* pkts =
          (PacketType*)((char*)scratch + scratchOffset_ + (srcOffsets[0] << 1));
      pkts[pktIdx].read(flag_);
    }
    __syncthreads();

    // Putting the data
    uint32_t chIdx = threadIdx.x;
    if (chIdx >= nOutput) {
      return;
    }
    uint32_t dstOffset = (dstOffsets[chIdx] << 1) + scratchOffset_;
    uint32_t srcOffset = (srcOffsets[0] << 1) + scratchOffset_;
    MemoryId dstMemoryId = portChannelBufferIds_[op.outputBufferRefs[chIdx].id];
    portChannels_[channelIndexes[chIdx]].put(
        dstMemoryId, dstOffset,
        static_cast<MemoryId>(BufferType::SCRATCH) + localMemoryIdBegin_,
        srcOffset, size << 1);
  }
}

template <typename T, typename PacketType, bool SendToRemote = true,
          ReduceOp OpType = SUM>
MSCCLPP_DEVICE_INLINE void handleReduceSendPackets(Operation const& op,
                                                   void* input, void* output,
                                                   void* scratch) {
  uint32_t size = op.inputBufferSizes[0];
  const uint32_t nSrcs = op.nInputs - 1;
  const uint32_t nDstChannels = op.nOutputs - 1;
  const uint32_t srcOffsetByBytes = op.inputOffsets[0];
  const uint32_t dstOffsetByBytes = op.outputOffsets[0];
  uint32_t const* inputOffsets = op.inputOffsets + 1;
  uint32_t const* outputOffsets = op.outputOffsets + 1;
  BufferRef const* outputBufferRefs = op.outputBufferRefs + 1;

  uint32_t nPackets = size / sizeof(PacketPayload<PacketType>);
  const uint32_t srcOffset =
      srcOffsetByBytes / sizeof(PacketPayload<PacketType>);
  const uint32_t dstOffset =
      dstOffsetByBytes / sizeof(PacketPayload<PacketType>);
  PacketPayload<PacketType>* srcPacketPayload =
      (PacketPayload<PacketType>*)getBuffer(input, output, scratch,
                                            op.inputBufferRefs[0].type) +
      srcOffset;
  PacketPayload<PacketType>* dstPacketPayload =
      (PacketPayload<PacketType>*)getBuffer(input, output, scratch,
                                            op.outputBufferRefs[0].type) +
      dstOffset;
  for (uint32_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = {};
    for (uint32_t index = 0; index < nSrcs; ++index) {
      PacketType* pkt = (PacketType*)((char*)scratch + scratchOffset_ +
                                      2 * inputOffsets[index]);
      PacketPayload<PacketType> val = pkt[idx].read(flag_);
      data = cal_vector<T, OpType>(data, val);
    }
    data = cal_vector<T, OpType>(data, srcPacketPayload[idx]);
    dstPacketPayload[idx] = data;

    if constexpr (SendToRemote) {
      PacketType pkt(data, flag_);
      for (uint32_t index = 0; index < nDstChannels; ++index) {
        uint32_t offset =
            (scratchOffset_ + outputOffsets[index] * 2) / sizeof(PacketType);
        void* remoteMemory = static_cast<char*>(
            memoryChannelBufferPtrs_[outputBufferRefs[index].id]);
        mscclpp::write<PacketType>(remoteMemory, offset + idx, pkt);
      }
    }
  }
}

template <typename T, typename PacketType, bool SendToRemote = true,
          ReduceOp OpType = SUM>
MSCCLPP_DEVICE_INLINE void handleReduceCopySendPackets(Operation const& op,
                                                       void* input,
                                                       void* output,
                                                       void* scratch) {
  uint32_t size = op.inputBufferSizes[0];
  const uint32_t nSrcs = op.nInputs - 1;
  const uint32_t nDstChannels = op.nOutputs - 2;
  const uint32_t srcOffsetByBytes = op.inputOffsets[0];
  const uint32_t dstOffsetByBytes = op.outputOffsets[0];
  uint32_t const* inputOffsets = op.inputOffsets + 1;
  uint32_t const* outputOffsets = op.outputOffsets + 2;
  BufferRef const* outputBufferRefs = op.outputBufferRefs + 2;

  PacketType* dstPkt =
      (PacketType*)((char*)getBuffer(input, output, scratch,
                                     op.outputBufferRefs[1].type) +
                    2 * op.outputOffsets[1]);
  uint32_t nPackets = size / sizeof(PacketPayload<PacketType>);
  const uint32_t srcOffset =
      srcOffsetByBytes / sizeof(PacketPayload<PacketType>);
  const uint32_t dstOffset =
      dstOffsetByBytes / sizeof(PacketPayload<PacketType>);
  PacketPayload<PacketType>* srcPacketPayload =
      (PacketPayload<PacketType>*)getBuffer(input, output, scratch,
                                            op.inputBufferRefs[0].type) +
      srcOffset;
  PacketPayload<PacketType>* dstPacketPayload =
      (PacketPayload<PacketType>*)getBuffer(input, output, scratch,
                                            op.outputBufferRefs[0].type) +
      dstOffset;
  for (uint32_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = {};
    for (uint32_t index = 0; index < nSrcs; ++index) {
      PacketType* pkt = (PacketType*)((char*)scratch + scratchOffset_ +
                                      2 * inputOffsets[index]);
      PacketPayload<PacketType> val = pkt[idx].read(flag_);
      data = cal_vector<T, OpType>(data, val);
    }
    data = cal_vector<T, OpType>(data, srcPacketPayload[idx]);
    dstPacketPayload[idx] = data;
    PacketType* dst_val = &dstPkt[idx];
    dst_val->write(data, flag_);

    if constexpr (SendToRemote) {
      PacketType pkt(data, flag_);
      for (uint32_t index = 0; index < nDstChannels; ++index) {
        uint32_t offset =
            (scratchOffset_ + outputOffsets[index] * 2) / sizeof(PacketType);
        void* remoteMemory = static_cast<char*>(
            memoryChannelBufferPtrs_[outputBufferRefs[index].id]);
        mscclpp::write<PacketType>(remoteMemory, offset + idx, pkt);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleUnpackPackets(Operation const& op, void* input,
                                               void* output, void* scratch) {
  const uint32_t size = op.inputBufferSizes[0];
  const uint32_t dstOffset = op.outputOffsets[0];
  const uint32_t srcOffset = op.inputOffsets[0];
  PacketType* srcPackets = (PacketType*)(static_cast<char*>(scratch) +
                                         scratchOffset_ + (srcOffset << 1));
  PacketPayload<PacketType>* result =
      (PacketPayload<PacketType>*)(static_cast<char*>(
                                       getBuffer(input, output, scratch,
                                                 op.outputBufferRefs[0].type)) +
                                   dstOffset);
  uint32_t nPackets = size / sizeof(PacketPayload<PacketType>);
  for (uint32_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = srcPackets[idx].read(flag_);
    result[idx] = data;
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleCopyPackets(Operation const& op, void* input,
                                             void* output, void* scratch) {
  uint32_t size = op.inputBufferSizes[0];
  uint32_t dstOffset = op.outputOffsets[0];
  uint32_t srcOffset = op.inputOffsets[0];
  dstOffset = dstOffset << 1;
  char* dst = static_cast<char*>(getBuffer(input, output, scratch,
                                           op.outputBufferRefs[0].type)) +
              dstOffset;
  char* src = static_cast<char*>(getBuffer(input, output, scratch,
                                           op.inputBufferRefs[0].type)) +
              srcOffset;
  mscclpp::copyToPackets<PacketType>(dst, src, size, threadIdx.x, blockDim.x,
                                     flag_);
}

template <typename T, bool ReuseScratch, bool SendToRemote = true,
          ReduceOp OpType = SUM>
MSCCLPP_DEVICE_INLINE void handleReduceSend(Operation const& op, void* input,
                                            void* output, void* scratch,
                                            uint32_t offset,
                                            uint32_t unitSize) {
  const uint32_t size = min(op.inputBufferSizes[0] - offset, unitSize);
  const uint32_t nInt4 = size / sizeof(int4);
  int nInput = op.nInputs - 1;
  int nOutput = op.nOutputs - 1;
  uint32_t const* inputOffsets = op.inputOffsets + 1;
  uint32_t const* outputOffsets = op.outputOffsets + 1;
  BufferRef const* inputBufferRefs = op.inputBufferRefs + 1;
  BufferRef const* outputBufferRefs = op.outputBufferRefs + 1;
  uint32_t srcOffsetByBytes =
      op.inputOffsets[0] +
      getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset);
  uint32_t dstOffsetByBytes =
      op.outputOffsets[0] +
      getOffset<ReuseScratch>(op.outputBufferRefs[0].type, offset);

  const uint32_t srcOffset4 = srcOffsetByBytes / sizeof(int4);
  const uint32_t dstOffset4 = dstOffsetByBytes / sizeof(int4);
  int4* src4 =
      (int4*)getBuffer(input, output, scratch, op.inputBufferRefs[0].type);
  int4* dst4 =
      (int4*)getBuffer(input, output, scratch, op.outputBufferRefs[0].type);
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = src4[srcOffset4 + idx];
    for (int index = 0; index < nInput; ++index) {
      int4* buff4 = static_cast<int4*>(
          getBuffer(input, output, scratch, inputBufferRefs[index].type));
      size_t buffOffset =
          (inputOffsets[index] +
           getOffset<ReuseScratch>(outputBufferRefs[index].type, offset)) /
          sizeof(int4);
      int4 val = buff4[buffOffset + idx];
      tmp = cal_vector<T, OpType>(tmp, val);
    }
    dst4[dstOffset4 + idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nOutput; ++index) {
        size_t outOffset =
            (outputOffsets[index] +
             getOffset<ReuseScratch>(
                 memoryChannelBufferTypes_[outputBufferRefs[index].id],
                 offset)) /
            sizeof(int4);
        void* remoteMemory =
            memoryChannelBufferPtrs_[outputBufferRefs[index].id];
        mscclpp::write(remoteMemory, outOffset + idx, tmp);
      }
    }
  }
  // handle rest of data
  uint32_t processed = nInt4 * sizeof(int4);
  const uint32_t startIdx = (srcOffsetByBytes + processed) / sizeof(T);
  const uint32_t endIdx = (srcOffsetByBytes + size) / sizeof(T);
  T* src = (T*)src4;
  T* dst = (T*)dst4;
  for (uint32_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = src[idx];
    for (int index = 0; index < nInput; ++index) {
      T* buff = static_cast<T*>(
          getBuffer(input, output, scratch, inputBufferRefs[index].type));
      uint32_t buffOffset =
          (inputOffsets[index] +
           getOffset<ReuseScratch>(inputBufferRefs[index].type, offset)) /
          sizeof(T);
      tmp = tmp + buff[buffOffset + idx];
    }
    dst[idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nOutput; ++index) {
        uint32_t outOffset =
            (outputOffsets[index] +
             getOffset<ReuseScratch>(
                 memoryChannelBufferTypes_[outputBufferRefs[index].id],
                 offset)) /
            sizeof(T);
        void* remoteMemory =
            memoryChannelBufferPtrs_[outputBufferRefs[index].id];
        mscclpp::write<T>(remoteMemory, outOffset + idx, tmp);
      }
    }
  }
}

template <bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void handleCopy(Operation const& op, void* input,
                                      void* output, void* scratch,
                                      uint32_t offset, uint32_t unitSize) {
  uint32_t size = min(op.inputBufferSizes[0] - offset, unitSize);
  if (size <= 0) {
    return;
  }
  uint32_t dstOffset =
      op.outputOffsets[0] +
      getOffset<ReuseScratch>(op.outputBufferRefs[0].type, offset);
  uint32_t srcOffset =
      op.inputOffsets[0] +
      getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset);
  char* srcData = static_cast<char*>(getBuffer(input, output, scratch,
                                               op.inputBufferRefs[0].type)) +
                  srcOffset;
  char* dstData = static_cast<char*>(getBuffer(input, output, scratch,
                                               op.outputBufferRefs[0].type)) +
                  dstOffset;
  mscclpp::copy(dstData, srcData, size, threadIdx.x, blockDim.x);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <typename T, bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void handleMultiLoadReduceStore(Operation const& op,
                                                      uint32_t offset,
                                                      uint32_t unitSize) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    assert(false &&
           "MULTI_LOAD_REDUCE_STORE is not supported for uint8_t data type");
    return;
  } else {
    static_assert(sizeof(T) <= 8, "Only support type with size <= 8 bytes");
    const uint32_t size = min(op.inputBufferSizes[0] - offset, unitSize);
    if (size <= 0) {
      return;
    }
    const uint32_t srcOffset =
        op.inputOffsets[0] +
        getOffset<ReuseScratch>(op.nvlsInputBufferType, offset);
    const uint32_t dstOffset =
        op.outputOffsets[0] +
        getOffset<ReuseScratch>(op.nvlsOutputBufferType, offset);
    assert(size % sizeof(T) == 0);
    assert(srcOffset % sizeof(T) == 0);
    assert(dstOffset % sizeof(T) == 0);

    T* src = (T*)nvlsChannels_[op.nvlsInputIndex].mcPtr;
    T* dst = (T*)nvlsChannels_[op.nvlsOutputIndex].mcPtr;
    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
      const size_t nElem = size / sizeof(T);
      const size_t srcOffsetElem = srcOffset / sizeof(T);
      const size_t dstOffsetElem = dstOffset / sizeof(T);
      VectorType<T, 1>* srcElem =
          reinterpret_cast<VectorType<T, 1>*>(src + srcOffsetElem);
      VectorType<T, 1>* dstElem =
          reinterpret_cast<VectorType<T, 1>*>(dst + dstOffsetElem);
      for (size_t idx = threadIdx.x; idx < nElem; idx += blockDim.x) {
        auto val = SwitchChannelDeviceHandle::multimemLoadReduce(srcElem + idx);
        SwitchChannelDeviceHandle::multimemStore(val, dstElem + idx);
      }
    } else {
      // handle data in 16-byte unit
      using Type16 = mscclpp::VectorType<T, 16 / sizeof(T)>;
      const size_t nType16 = size / sizeof(Type16);
      const size_t srcOffset16 = srcOffset / sizeof(Type16);
      const size_t dstOffset16 = dstOffset / sizeof(Type16);
      Type16* src16 = reinterpret_cast<Type16*>(src) + srcOffset16;
      Type16* dst16 = reinterpret_cast<Type16*>(dst) + dstOffset16;
      for (size_t idx = threadIdx.x; idx < nType16; idx += blockDim.x) {
        Type16 val = SwitchChannelDeviceHandle::multimemLoadReduce(src16 + idx);
        SwitchChannelDeviceHandle::multimemStore(val, dst16 + idx);
      }
      // handle rest of data
      constexpr int RedBytes = (sizeof(T) == 8) ? 8 : 4;
      using TypeRest = mscclpp::VectorType<T, RedBytes / sizeof(T)>;
      const size_t processed = nType16 * sizeof(Type16);
      const size_t nRest = (size - processed) / sizeof(TypeRest);
      TypeRest* srcR = reinterpret_cast<TypeRest*>(src + srcOffset + processed);
      TypeRest* dstR = reinterpret_cast<TypeRest*>(dst + dstOffset + processed);
      for (size_t idx = threadIdx.x; idx < nRest; idx += blockDim.x) {
        TypeRest val =
            SwitchChannelDeviceHandle::multimemLoadReduce(srcR + idx);
        SwitchChannelDeviceHandle::multimemStore(val, dstR + idx);
      }
    }
  }
}
#endif

template <typename T, typename PacketType, bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void handlePipeline(Operation const& op, T* input,
                                          T* output, T* scratch
#if defined(ENABLE_NPKIT)
                                          ,
                                          uint64_t& eventBufferHead
#endif
) {
  uint16_t nIterations = op.nIterations;
  uint16_t nOperations = op.nOperations;
  uint32_t unitSize = op.unitSize;
  Operation const* operations = &op + 1;
  for (uint16_t i = 0; i < nIterations; i++) {
    uint32_t offset = i * unitSize;
    for (uint8_t opId = 0; opId < nOperations; opId++) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY)
      executeDeviceFunction<T, PacketType, ReuseScratch>(
          operations[opId], input, output, scratch, nullptr, offset, unitSize,
          eventBufferHead);
#else
      executeDeviceFunction<T, PacketType, ReuseScratch>(
          operations[opId], input, output, scratch, nullptr, offset, unitSize);
#endif
    }
  }
}

MSCCLPP_DEVICE_INLINE void handleSemRelease(Operation const& op) {
  uint32_t tid = threadIdx.x;
  if (tid < op.nDeviceSemaphores) {
    DeviceSemaphore* sem = &deviceSemaphores[op.deviceSemaphoreIds[tid]];
    sem->release();
  }
}

MSCCLPP_DEVICE_INLINE void handleSemAcquire(Operation const& op) {
  uint32_t tid = threadIdx.x;
  if (tid < op.nDeviceSemaphores) {
    DeviceSemaphore* sem = &deviceSemaphores[op.deviceSemaphoreIds[tid]];
    sem->acquire();
  }
}

#if defined(ENABLE_NPKIT)
MSCCLPP_DEVICE_INLINE uint32_t getOpSize(Operation const& op, uint32_t offset,
                                         uint32_t unitSize) {
  if (op.type == OperationType::BARRIER || op.type == OperationType::WAIT ||
      op.type == OperationType::SIGNAL ||
      op.type == OperationType::RELAXED_WAIT ||
      op.type == OperationType::RELAXED_SIGNAL ||
      op.type == OperationType::NOP || op.type == OperationType::FLUSH ||
      op.type == OperationType::PIPELINE ||
      op.type == OperationType::SEM_ACQUIRE ||
      op.type == OperationType::SEM_RELEASE) {
    return 0;
  }
  return min(op.inputBufferSizes[0] - offset, unitSize);
}
#endif

template <typename T, typename PacketType, bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void executeDeviceFunction(Operation const& op, T* input,
                                                 T* output, T* scratch,
                                                 uint8_t* nSteps,
                                                 uint32_t offset,
                                                 uint32_t unitSize
#if defined(ENABLE_NPKIT)
                                                 ,
                                                 uint64_t& eventBufferHead
#endif
) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY)
  uint32_t opSize = 0;
  if (unitSize < UINT32_MAX) {
    opSize = getOpSize(op, offset, unitSize);
  }
  if (op.type != OperationType::PIPELINE) {
    NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY + (int)op.type,
                              opSize, 0, NPKIT_GET_GPU_TIMESTAMP(),
                              eventBuffer_, &eventBufferHead);
  }
#endif
  if (nSteps != nullptr) {
    *nSteps = 1;
  }
  OperationType opType = op.type;
  if (opType == OperationType::NOP) {
    handleNop();
  } else if (opType == OperationType::BARRIER) {
    handleBarrier(op);
  } else if (opType == OperationType::SIGNAL) {
    handleSignal(op);
  } else if (opType == OperationType::WAIT) {
    handleWait(op);
  } else if (opType == OperationType::RELAXED_SIGNAL) {
    handleSignal<true>(op);
  } else if (opType == OperationType::RELAXED_WAIT) {
    handleWait<true>(op);
  } else if (opType == OperationType::FLUSH) {
    handleFlush(op);
  } else if (opType == OperationType::PUT) {
    handlePut<ReuseScratch>(op, input, output, scratch, offset, unitSize);
  } else if (opType == OperationType::PUT_WITH_SIGNAL) {
    handlePut<ReuseScratch, true>(op, input, output, scratch, offset, unitSize);
  } else if (opType == OperationType::PUT_WITH_SIGNAL_AND_FLUSH) {
    handlePut<ReuseScratch, true, true>(op, input, output, scratch, offset,
                                        unitSize);
  } else if (opType == OperationType::PUT_PACKETS) {
    handlePutPackets<PacketType>(op, input, output, scratch);
  } else if (opType == OperationType::READ_PUT_PACKETS) {
    handleReadPutPackets<PacketType>(op, scratch);
  } else if (opType == OperationType::GET) {
    handleGet<ReuseScratch>(op, input, output, scratch, offset, unitSize);
  } else if (opType == OperationType::READ_REDUCE_SEND) {
    handleReadReduceSend<T, ReuseScratch, true>(op, input, output, scratch,
                                                offset, unitSize);
  } else if (opType == OperationType::READ_REDUCE) {
    handleReadReduceSend<T, ReuseScratch, false>(op, input, output, scratch,
                                                 offset, unitSize);
  } else if (opType == OperationType::COPY) {
    handleCopy<ReuseScratch>(op, input, output, scratch, offset, unitSize);
  } else if (opType == OperationType::REDUCE_SEND) {
    handleReduceSend<T, ReuseScratch>(op, input, output, scratch, offset,
                                      unitSize);
  } else if (opType == OperationType::REDUCE) {
    handleReduceSend<T, ReuseScratch, false>(op, input, output, scratch, offset,
                                             unitSize);
  } else if (opType == OperationType::REDUCE_SEND_PACKETS) {
    handleReduceSendPackets<T, PacketType>(op, input, output, scratch);
  } else if (opType == OperationType::REDUCE_PACKETS) {
    handleReduceSendPackets<T, PacketType, false>(op, input, output, scratch);
  } else if (opType == OperationType::REDUCE_COPY_SEND_PACKETS) {
    handleReduceCopySendPackets<T, PacketType>(op, input, output, scratch);
  } else if (opType == OperationType::REDUCE_COPY_PACKETS) {
    handleReduceCopySendPackets<T, PacketType, false>(op, input, output,
                                                      scratch);
  } else if (opType == OperationType::UNPACK_PACKETS) {
    handleUnpackPackets<PacketType>(op, input, output, scratch);
  } else if (opType == OperationType::COPY_PACKETS) {
    handleCopyPackets<PacketType>(op, input, output, scratch);
  } else if (opType == OperationType::SEM_ACQUIRE) {
    handleSemAcquire(op);
  } else if (opType == OperationType::SEM_RELEASE) {
    handleSemRelease(op);
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  else if (opType == OperationType::MULTI_LOAD_REDUCE_STORE) {
    handleMultiLoadReduceStore<T, ReuseScratch>(op, offset, unitSize);
  }
#endif
  else if (opType == OperationType::PIPELINE) {
    *nSteps = op.nOperations + 1;
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_EXIT)
    handlePipeline<T, PacketType, ReuseScratch>(op, input, output, scratch,
                                                eventBufferHead);
#else
    handlePipeline<T, PacketType, ReuseScratch>(op, input, output, scratch);
#endif
  }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_EXIT)
  if (op.type != OperationType::PIPELINE) {
    NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_OP_BASE_EXIT + (int)op.type,
                              opSize, 0, NPKIT_GET_GPU_TIMESTAMP(),
                              eventBuffer_, &eventBufferHead);
  }
#endif
  return;
}

template <typename T, typename PacketType = LL16Packet,
          bool ReuseScratch = false>
__global__ __launch_bounds__(1024, 1) void executionKernel(
    [[maybe_unused]] int rank /*for debug*/, T* input, T* output, T* scratch,
    uint32_t scratchOffset, uint32_t scratchChunkSize,
    DeviceExecutionPlan* plan, DeviceSemaphore* semaphores,
    uint32_t localMemoryIdBegin, uint32_t flag
#if defined(ENABLE_NPKIT)
    ,
    NpKitEventCollectContext* npKitEventCollectContexts,
    uint64_t* cpuTimestamp) {
#else
) {
#endif
  extern __shared__ int4 sharedMem[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
#if defined(ENABLE_NPKIT)
  eventBuffer_ = (NpKitEvent*)((char*)sharedMem + sizeof(DeviceExecutionPlan));
  uint64_t eventBufferHead = 0;
#if defined(ENABLE_NPKIT_EVENT_EXECUTOR_INIT_ENTRY) && \
    defined(ENABLE_NPKIT_EVENT_EXECUTOR_INIT_EXIT)
  uint64_t npkitTimestampEntry = 0;
  if (tid == 0) {
    npkitTimestampEntry = NPKIT_GET_GPU_TIMESTAMP();
  }
#endif
#endif
  DeviceExecutionPlan* localPlan = plan + bid;
  for (size_t i = tid; i < sizeof(DeviceExecutionPlan) / sizeof(int4);
       i += blockDim.x) {
    sharedMem[i] = ((int4*)localPlan)[i];
  }
  __syncshm();
  deviceSemaphores = semaphores;
  localPlan = (DeviceExecutionPlan*)sharedMem;
  int nOperations = localPlan->nOperations;
  Operation* operations = (Operation*)localPlan->operations;
  memoryChannels_ = localPlan->channels.memoryChannels;
  portChannels_ = localPlan->channels.portChannels;
  nvlsChannels_ = localPlan->channels.nvlsChannels;
  memoryChannelBufferPtrs_ = localPlan->remoteBuffers.memoryChannelBufferPtrs;
  portChannelBufferIds_ = localPlan->remoteBuffers.portChannelBufferIds;
  memoryChannelBufferTypes_ = localPlan->remoteBuffers.memoryChannelBufferTypes;
  portChannelBufferTypes_ = localPlan->remoteBuffers.portChannelBufferTypes;
  flag_ = flag;
  scratchChunkSize_ = scratchChunkSize;
  scratchOffset_ = scratchOffset;
  localMemoryIdBegin_ = localMemoryIdBegin;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
#if defined(MSCCLPP_DEVICE_HIP)
  NpKit::CollectGpuEventShm(
      NPKIT_EVENT_TIME_SYNC_CPU, 0, 0,
      NPKIT_LOAD_CPU_TIMESTAMP_PER_BLOCK(cpuTimestamp, bid),
#else
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
#endif
      eventBuffer_, &eventBufferHead);
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0,
                            NPKIT_GET_GPU_TIMESTAMP(), eventBuffer_,
                            &eventBufferHead);
#endif

#if defined(ENABLE_NPKIT) &&                           \
    defined(ENABLE_NPKIT_EVENT_EXECUTOR_INIT_ENTRY) && \
    defined(ENABLE_NPKIT_EVENT_EXECUTOR_INIT_EXIT)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_INIT_ENTRY, 0, 0,
                            npkitTimestampEntry, eventBuffer_,
                            &eventBufferHead);
  NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_INIT_EXIT, 0, 0,
                            NPKIT_GET_GPU_TIMESTAMP(), eventBuffer_,
                            &eventBufferHead);
#endif

  for (int i = 0; i < nOperations;) {
    Operation& op = operations[i];
    uint8_t nSteps = 0;
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY)
    executeDeviceFunction<T, PacketType, ReuseScratch>(
        op, input, output, scratch, &nSteps, 0, UINT32_MAX, eventBufferHead);
#else
    executeDeviceFunction<T, PacketType, ReuseScratch>(op, input, output,
                                                       scratch, &nSteps);
#endif
    i += nSteps;
  }

#if defined(ENABLE_NPKIT)
  NpKit::StoreGpuEventShm(npKitEventCollectContexts, eventBuffer_,
                          eventBufferHead);
#endif
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

class ExecutionKernel {
 public:
#if defined(MSCCLPP_DEVICE_HIP)
  template <typename PacketType, bool ReuseScratch>
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src,
                           void* dst, void* scratch, uint32_t scratchOffset,
                           uint32_t scratchChunkSize, DataType dataType,
                           DeviceExecutionPlan* plan,
                           DeviceSemaphore* semaphores,
                           uint32_t localMemoryIdBegin, uint32_t sharedMemSize,
                           cudaStream_t stream, uint32_t flag = 0) {
    switch (dataType) {
      case DataType::INT32:
        executionKernel<int32_t, PacketType, ReuseScratch>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
                rank, (int32_t*)src, (int32_t*)dst, (int32_t*)scratch,
                scratchOffset, scratchChunkSize, plan, semaphores,
                localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
                ,
                NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
            );
#endif
        break;
      case DataType::UINT32:
        executionKernel<uint32_t, PacketType, ReuseScratch>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
                rank, (uint32_t*)src, (uint32_t*)dst, (uint32_t*)scratch,
                scratchOffset, scratchChunkSize, plan, semaphores,
                localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
                ,
                NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
            );
#endif
        break;
      case DataType::FLOAT16:
        executionKernel<half, PacketType, ReuseScratch>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
                rank, (half*)src, (half*)dst, (half*)scratch, scratchOffset,
                scratchChunkSize, plan, semaphores, localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
                ,
                NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
            );
#endif
        break;
      case DataType::FLOAT32:
        executionKernel<float, PacketType, ReuseScratch>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
                rank, (float*)src, (float*)dst, (float*)scratch, scratchOffset,
                scratchChunkSize, plan, semaphores, localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
                ,
                NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
            );
#endif
        break;
      case DataType::BFLOAT16:
        executionKernel<__bfloat16, PacketType, ReuseScratch>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
                rank, (__bfloat16*)src, (__bfloat16*)dst, (__bfloat16*)scratch,
                scratchOffset, scratchChunkSize, plan, semaphores,
                localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
                ,
                NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
            );
#endif
        break;
#if defined(__FP8_TYPES_EXIST__)
      case DataType::FLOAT8_E4M3:
        executionKernel<__fp8_e4m3, PacketType, ReuseScratch>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
                rank, (__fp8_e4m3*)src, (__fp8_e4m3*)dst, (__fp8_e4m3*)scratch,
                scratchOffset, scratchChunkSize, plan, semaphores,
                localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
                ,
                NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
            );
#endif
        break;
      case DataType::FLOAT8_E5M2:
        executionKernel<__fp8_e5m2, PacketType, ReuseScratch>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
                rank, (__fp8_e5m2*)src, (__fp8_e5m2*)dst, (__fp8_e5m2*)scratch,
                scratchOffset, scratchChunkSize, plan, semaphores,
                localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
                ,
                NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
            );
#endif
        break;
#endif  // __FP8_TYPES_EXIST__
      case DataType::UINT8:
        executionKernel<uint8_t, PacketType, ReuseScratch>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
                rank, (uint8_t*)src, (uint8_t*)dst, (uint8_t*)scratch,
                scratchOffset, scratchChunkSize, plan, semaphores,
                localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
                ,
                NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
            );
#endif
        break;
    }
  }
#else   // !defined(MSCCLPP_DEVICE_HIP)
  template <typename PacketType, bool ReuseScratch>
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src,
                           void* dst, void* scratch, uint32_t scratchOffset,
                           uint32_t scratchChunkSize, DataType dataType,
                           DeviceExecutionPlan* plan,
                           DeviceSemaphore* semaphores,
                           uint32_t localMemoryIdBegin, uint32_t sharedMemSize,
                           cudaStream_t stream, uint32_t flag = 0);
#endif  // !defined(MSCCLPP_DEVICE_HIP)
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_KERNEL_HPP_

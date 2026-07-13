// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_COMMON_HPP_
#define MSCCLPP_EXECUTION_COMMON_HPP_

#include "memory_channel.hpp"
#include "port_channel.hpp"
#include "switch_channel.hpp"

namespace mscclpp {

constexpr int MAX_LOCAL_BUFFER_PER_OPERATION = 2;

constexpr int MAX_CHANNEL = 16;
constexpr int MAX_CHANNEL_PER_OPERATION = 8;
constexpr int MAX_OPERATION = 64;

constexpr int MAX_DEVICE_SYNCERS = 16;
constexpr int MAX_DEVICE_SEMAPHORES = 16;

constexpr uint32_t PREDFINED_SCRATCH_SIZE = 1 << 26;  // 64 MB

enum class BufferType : uint8_t {
  NONE = UINT8_MAX,
  INPUT = 0,
  OUTPUT = 1,
  SCRATCH = 2,
};

enum class ChannelType : uint8_t {
  NONE,
  MEMORY,
  PORT,
  SWITCH,
};

// NOTE(chhwang): any modification here requires corresponding updates in
// `tools/npkit/npkit_trace_generator.py`. As well as
// NPKIT_EVENT_EXECUTOR_OP_BASE_EXIT in npkit_event.hpp
enum class OperationType : uint8_t {
  NOP,
  BARRIER,
  PUT,
  PUT_PACKETS,
  READ_PUT_PACKETS,
  PUT_WITH_SIGNAL,
  PUT_WITH_SIGNAL_AND_FLUSH,
  GET,
  COPY,
  COPY_PACKETS,
  UNPACK_PACKETS,
  SIGNAL,
  WAIT,
  FLUSH,
  REDUCE,
  REDUCE_PACKETS,
  REDUCE_COPY_PACKETS,
  REDUCE_SEND,
  REDUCE_SEND_PACKETS,
  REDUCE_COPY_SEND_PACKETS,
  READ_REDUCE,
  READ_REDUCE_SEND,
  MULTI_LOAD_REDUCE_STORE,
  RELAXED_SIGNAL,
  RELAXED_WAIT,
  PIPELINE,
  SEM_RELEASE,
  SEM_ACQUIRE,
};

struct Channels {
  mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel> memoryChannels[MAX_CHANNEL];
  mscclpp::DeviceHandle<mscclpp::BasePortChannel> portChannels[MAX_CHANNEL];
  mscclpp::DeviceHandle<mscclpp::SwitchChannel> nvlsChannels[MAX_CHANNEL];
};

struct RemoteBuffers {
  // For buffer accessed via memory channel
  BufferType memoryChannelBufferTypes[MAX_CHANNEL];
  void* memoryChannelBufferPtrs[MAX_CHANNEL];

  // for buffer access via port channel
  BufferType portChannelBufferTypes[MAX_CHANNEL];
  MemoryId portChannelBufferIds[MAX_CHANNEL];
};

union BufferRef {
  uint8_t id;
  BufferType type;
};

struct Operation {
  OperationType type;
  ChannelType channelType;
  union {
    BufferRef inputBufferRefs[MAX_LOCAL_BUFFER_PER_OPERATION +
                              MAX_CHANNEL_PER_OPERATION];
    struct {
      uint8_t nvlsInputIndex;
      BufferType nvlsInputBufferType;
    };
  };
  union {
    BufferRef outputBufferRefs[MAX_LOCAL_BUFFER_PER_OPERATION +
                               MAX_CHANNEL_PER_OPERATION];
    struct {
      uint8_t nvlsOutputIndex;
      BufferType nvlsOutputBufferType;
    };
  };

  union {
    struct {
      uint8_t channelIndexes[MAX_CHANNEL_PER_OPERATION];
      uint32_t inputOffsets[MAX_LOCAL_BUFFER_PER_OPERATION +
                            MAX_CHANNEL_PER_OPERATION];
      uint32_t outputOffsets[MAX_LOCAL_BUFFER_PER_OPERATION +
                             MAX_CHANNEL_PER_OPERATION];
      uint32_t inputBufferSizes[MAX_LOCAL_BUFFER_PER_OPERATION +
                                MAX_CHANNEL_PER_OPERATION];
      uint32_t outputBufferSizes[MAX_LOCAL_BUFFER_PER_OPERATION +
                                 MAX_CHANNEL_PER_OPERATION];

      uint8_t nChannels;
      uint8_t nInputs;
      uint8_t nOutputs;
    };
    struct {
      uint32_t unitSize;
      uint32_t nIterations;
      uint8_t nOperations;
    };
    struct {
      uint32_t deviceSyncerIndex;
      uint32_t nThreadBlocks;
    };
    struct {
      uint32_t deviceSemaphoreIds[MAX_DEVICE_SEMAPHORES];
      uint32_t nDeviceSemaphores;
    };
  };
};

// total size = 2016 + 10240 + 4 + 12(padding) = 12272 bytes
struct __attribute__((aligned(16))) DeviceExecutionPlan {
  uint8_t nMemoryChannels;              // 1 bytes
  uint8_t nPortChannels;                // 1 bytes
  uint16_t nOperations;                 // 2 bytes
  Channels channels;                    // 1792 bytes
  RemoteBuffers remoteBuffers;          // 224 bytes
  Operation operations[MAX_OPERATION];  // 64 * 160 = 10240 bytes
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_COMMON_HPP_

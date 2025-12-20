#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

namespace eccl {

using OpTaskType = uint64_t; // OpTaskBitsType = 8
constexpr OpTaskType OpTaskCopy = 0x0;  // a copy task
constexpr OpTaskType OpTaskReduce = 0x1;  // a reduce task

using OpDataType = uint64_t; // OpDataBitsType = 8
constexpr OpDataType OpDataFp8 = 0x0;
constexpr OpDataType OpDataFp16 = 0x1;
constexpr OpDataType OpDataFp32 = 0x3;

using OpRedType = uint64_t; // OpRedBitsType = 8
constexpr OpRedType OpRedSum = 0x1;
constexpr OpRedType OpRedMax = 0x2;

constexpr unsigned int OpTaskBitsType = 8;            // OpTaskType
constexpr unsigned int OpTaskBitsData = 8;            // OpDataType
constexpr unsigned int OpTaskBitsRed = 8;             // OpRedType
constexpr unsigned int OpTaskBitsWPT = 8;             // Works per Thread
constexpr unsigned int OpTaskBitsFifoReserved = 1;    // reserved bits for alignment

// 16B无法容纳完整64位地址，32B作为task的开销如何
/// 32B unsigned integers used as a OpTask.
/// Used as a work element in the concurrent FIFO.
union alignas(16) OpTask {
  struct {
    uint64_t src;
    uint64_t dst;
    uint64_t size;
    uint64_t meta;
  };
  // The summation of number of bits must be 128 or less.
  struct {
    // First 64 bits: value[0]
    uint64_t src;

    // Second 64 bits: value[1]
    uint64_t dst;

    // Third 64: value[2]
    uint64_t size;

    // Fourth 64: value[3]
    uint64_t taskType : OpTaskBitsType;
    uint64_t dataType : OpTaskBitsData;
    uint64_t redType : OpTaskBitsRed;
    uint64_t wpt : OpTaskBitsWPT; // works per thread
    uint64_t : (64 - OpTaskBitsType - OpTaskBitsData - OpTaskBitsRed - OpTaskBitsWPT - 
                OpTaskBitsFifoReserved);  // ensure 64-bit alignment
    uint64_t reserved : OpTaskBitsFifoReserved;
  } fields;


  // Default constructor for CPU side initialization
  OpTask() = default;

  /// Constructor for CPU initialization (without GPU specific macros)
  /// @param src The source addr for the data (for kCopy or kReduce).
  /// @param dst The destination addr for the data (for kCopy or kReduce).
  /// @param size The number of bytes for the operation (e.g., for kCopy or kReduce).
  /// @param ttype The type of the OpTask (e.g., OpTaskCopy, OpTaskReduce).
  /// @param dtype The data type for the operation (e.g., OpDataFp32, OpDataI32).
  /// @param redop The reduction operation type (only for kReduce).
  /// @param wpt The works per thread
  OpTask(uint64_t src, uint64_t dst, uint64_t size,
         OpTaskType ttype, OpDataType dtype, OpRedType redop, uint64_t wpt) {
    assert(ttype < (1ULL << OpTaskBitsType));
    assert(dtype < (1ULL << OpTaskBitsData));
    assert(redop < (1ULL << OpTaskBitsRed));
    assert(wpt  < (1ULL << OpTaskBitsWPT));

    this->src = src;
    this->dst = dst;
    this->size = size;

    constexpr uint64_t maskTType = (1ULL << OpTaskBitsType) - 1;
    constexpr uint64_t maskDType = (1ULL << OpTaskBitsData) - 1;
    constexpr uint64_t maskRedOp = (1ULL << OpTaskBitsRed) - 1;
    constexpr uint64_t maskWPT = (1ULL << OpTaskBitsWPT) - 1;

    // Packing fields
    this->meta = ((((((uint64_t)wpt & maskWPT) << OpTaskBitsWPT)
           + (((uint64_t)redop & maskRedOp) << OpTaskBitsRed)) 
           + (((uint64_t)dtype & maskDType) << OpTaskBitsData)) 
           + (((uint64_t)ttype & maskTType) << OpTaskBitsType));
  }
};

}
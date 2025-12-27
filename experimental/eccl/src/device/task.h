#pragma once

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

namespace eccl {

using OpTaskType = uint64_t;
constexpr OpTaskType OpTaskCopy = 0x0;
constexpr OpTaskType OpTaskReduce = 0x1;

using OpDataType = uint64_t;
constexpr OpDataType OpDataFp8 = 0x0;
constexpr OpDataType OpDataFp16 = 0x1;
constexpr OpDataType OpDataFp32 = 0x3;

using OpRedType = uint64_t;
constexpr OpRedType OpRedSum = 0x1;
constexpr OpRedType OpRedMax = 0x2;

constexpr unsigned int OpTaskBitsType = 8;
constexpr unsigned int OpTaskBitsData = 8;
constexpr unsigned int OpTaskBitsRed = 8;
constexpr unsigned int OpTaskBitsWPT = 8;

constexpr unsigned int OpTaskMetaShiftType = 0;
constexpr unsigned int OpTaskMetaShiftData = 8;
constexpr unsigned int OpTaskMetaShiftRed = 16;
constexpr unsigned int OpTaskMetaShiftWPT = 24;

static __host__ __device__ __forceinline__ uint32_t opTaskType(uint64_t meta) {
  return (uint32_t)((meta >> OpTaskMetaShiftType) & 0xFFu);
}
static __host__ __device__ __forceinline__ uint32_t opDataType(uint64_t meta) {
  return (uint32_t)((meta >> OpTaskMetaShiftData) & 0xFFu);
}
static __host__ __device__ __forceinline__ uint32_t opRedType(uint64_t meta) {
  return (uint32_t)((meta >> OpTaskMetaShiftRed) & 0xFFu);
}
static __host__ __device__ __forceinline__ uint32_t opWpt(uint64_t meta) {
  return (uint32_t)((meta >> OpTaskMetaShiftWPT) & 0xFFu);
}

// TODO: 16B task with args ptr
/// 32B unsigned integers used as a OpTask.
/// Used as a work element in the concurrent FIFO.
union alignas(16) OpTask {
  struct {
    uint64_t src;
    uint64_t dst;
    uint64_t size;
    uint64_t meta;
  };

  OpTask() = default;

  OpTask(uint64_t src, uint64_t dst, uint64_t size, OpTaskType ttype,
         OpDataType dtype, OpRedType redop, uint64_t wpt) {
    assert(ttype < (1ULL << OpTaskBitsType));
    assert(dtype < (1ULL << OpTaskBitsData));
    assert(redop < (1ULL << OpTaskBitsRed));
    assert(wpt < (1ULL << OpTaskBitsWPT));

    this->src = src;
    this->dst = dst;
    this->size = size;

    this->meta = ((uint64_t)(ttype & 0xFFu) << OpTaskMetaShiftType) |
                 ((uint64_t)(dtype & 0xFFu) << OpTaskMetaShiftData) |
                 ((uint64_t)(redop & 0xFFu) << OpTaskMetaShiftRed) |
                 ((uint64_t)(wpt & 0xFFu) << OpTaskMetaShiftWPT);
  }
};

}  // namespace eccl
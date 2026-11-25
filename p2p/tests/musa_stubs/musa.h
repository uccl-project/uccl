// Driver test double, limited to the allocation-range adapter contract.
#pragma once
#include <cstddef>
#include <cstdint>
enum MUresult { MUSA_SUCCESS = 0, MUSA_ERROR_INVALID_VALUE = 1 };
using MUdeviceptr = uintptr_t;
inline MUresult range_result = MUSA_SUCCESS;
inline MUresult muMemGetAddressRange(MUdeviceptr* base, size_t* size,
                                     MUdeviceptr ptr) {
  if (range_result == MUSA_SUCCESS) {
    *base = ptr & ~uintptr_t(4095);
    *size = 4096;
  }
  return range_result;
}

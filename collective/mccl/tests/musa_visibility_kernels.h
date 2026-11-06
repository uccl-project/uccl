#pragma once
#include <cstddef>
#include <cstdint>
#include <musa_runtime.h>

musaError_t fill_payload(uint8_t* data, size_t bytes, unsigned epoch,
                         musaStream_t stream);
musaError_t check_payload(uint8_t* allocation, size_t bytes, unsigned epoch,
                          unsigned* errors, musaStream_t stream);
constexpr size_t kVisibilityGuard = 16;

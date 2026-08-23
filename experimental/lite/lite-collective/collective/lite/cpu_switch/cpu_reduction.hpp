// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cstddef>

namespace mscclpp::lite::detail {

// Runtime-dispatched scalar/AVX-512 float-sum backends used by the public
// CpuSwitch templates.
void reduceFloatSum(float const* const* inputs, size_t inputCount,
                    float* output, size_t count);

void reduceTwoFloatSum(float const* const* inputs, size_t inputCount,
                       size_t inputStride, size_t firstOffset,
                       float* firstOutput, size_t secondOffset,
                       float* secondOutput, size_t count);

}  // namespace mscclpp::lite::detail

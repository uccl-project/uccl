// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "cpu_reduction.hpp"

#include <stdexcept>

namespace mscclpp::lite::detail {

namespace {

void reduceFloatSumScalar(float const* const* inputs, size_t inputCount,
                          float* output, size_t count) {
  for (size_t element = 0; element < count; ++element) {
    float value = inputs[0][element];
    for (size_t input = 1; input < inputCount; ++input) {
      value += inputs[input][element];
    }
    output[element] = value;
  }
}

void reduceTwoFloatSumScalar(float const* const* inputs, size_t inputCount,
                             size_t inputStride, size_t firstOffset,
                             float* firstOutput, size_t secondOffset,
                             float* secondOutput, size_t count) {
  size_t firstBase = firstOffset * inputStride;
  size_t secondBase = secondOffset * inputStride;
  for (size_t element = 0; element < count; ++element) {
    float first = inputs[0][firstBase + element];
    float second = inputs[0][secondBase + element];
    for (size_t input = 1; input < inputCount; ++input) {
      first += inputs[input][firstBase + element];
      second += inputs[input][secondBase + element];
    }
    firstOutput[element] = first;
    secondOutput[element] = second;
  }
}

#if defined(__x86_64__) && defined(__GNUC__)
using Avx512Float __attribute__((vector_size(64), aligned(1))) = float;

__attribute__((target("avx512f")))
void reduceFloatSumAvx512(float const* const* inputs, size_t inputCount,
                          float* output, size_t count) {
  size_t element = 0;
  for (; element + 16 <= count; element += 16) {
    Avx512Float value =
        *reinterpret_cast<Avx512Float const*>(inputs[0] + element);
    for (size_t input = 1; input < inputCount; ++input) {
      value += *reinterpret_cast<Avx512Float const*>(inputs[input] + element);
    }
    *reinterpret_cast<Avx512Float*>(output + element) = value;
  }
  for (; element < count; ++element) {
    float value = inputs[0][element];
    for (size_t input = 1; input < inputCount; ++input) {
      value += inputs[input][element];
    }
    output[element] = value;
  }
}

__attribute__((target("avx512f")))
void reduceTwoFloatSumAvx512(float const* const* inputs, size_t inputCount,
                             size_t inputStride, size_t firstOffset,
                             float* firstOutput, size_t secondOffset,
                             float* secondOutput, size_t count) {
  size_t firstBase = firstOffset * inputStride;
  size_t secondBase = secondOffset * inputStride;
  size_t element = 0;
  for (; element + 16 <= count; element += 16) {
    Avx512Float first = *reinterpret_cast<Avx512Float const*>(
        inputs[0] + firstBase + element);
    Avx512Float second = *reinterpret_cast<Avx512Float const*>(
        inputs[0] + secondBase + element);
    for (size_t input = 1; input < inputCount; ++input) {
      first += *reinterpret_cast<Avx512Float const*>(
          inputs[input] + firstBase + element);
      second += *reinterpret_cast<Avx512Float const*>(
          inputs[input] + secondBase + element);
    }
    *reinterpret_cast<Avx512Float*>(firstOutput + element) = first;
    *reinterpret_cast<Avx512Float*>(secondOutput + element) = second;
  }
  for (; element < count; ++element) {
    float first = inputs[0][firstBase + element];
    float second = inputs[0][secondBase + element];
    for (size_t input = 1; input < inputCount; ++input) {
      first += inputs[input][firstBase + element];
      second += inputs[input][secondBase + element];
    }
    firstOutput[element] = first;
    secondOutput[element] = second;
  }
}

bool hasAvx512F() {
  static bool const available = [] {
    __builtin_cpu_init();
    return static_cast<bool>(__builtin_cpu_supports("avx512f"));
  }();
  return available;
}
#endif

}  // namespace

void reduceFloatSum(float const* const* inputs, size_t inputCount,
                    float* output, size_t count) {
  if (inputCount == 0) {
    throw std::invalid_argument("CpuSwitch reduction needs an input");
  }
#if defined(__x86_64__) && defined(__GNUC__)
  if (hasAvx512F()) {
    reduceFloatSumAvx512(inputs, inputCount, output, count);
    return;
  }
#endif
  reduceFloatSumScalar(inputs, inputCount, output, count);
}

void reduceTwoFloatSum(float const* const* inputs, size_t inputCount,
                       size_t inputStride, size_t firstOffset,
                       float* firstOutput, size_t secondOffset,
                       float* secondOutput, size_t count) {
  if (inputCount == 0) {
    throw std::invalid_argument("CpuSwitch reduction needs an input");
  }
#if defined(__x86_64__) && defined(__GNUC__)
  if (hasAvx512F()) {
    reduceTwoFloatSumAvx512(inputs, inputCount, inputStride, firstOffset,
                            firstOutput, secondOffset, secondOutput, count);
    return;
  }
#endif
  reduceTwoFloatSumScalar(inputs, inputCount, inputStride, firstOffset,
                          firstOutput, secondOffset, secondOutput, count);
}

}  // namespace mscclpp::lite::detail

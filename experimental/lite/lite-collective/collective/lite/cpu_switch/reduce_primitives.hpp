// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "cpu_reduction.hpp"
#include "types.hpp"

#include <type_traits>
#include <vector>

namespace mscclpp::lite {

template <typename T>
struct Sum {
  static T apply(T lhs, T rhs) { return lhs + rhs; }
};

template <typename T>
struct Product {
  static T apply(T lhs, T rhs) { return lhs * rhs; }
};

template <typename T>
struct Min {
  static T apply(T lhs, T rhs) { return rhs < lhs ? rhs : lhs; }
};

template <typename T>
struct Max {
  static T apply(T lhs, T rhs) { return lhs < rhs ? rhs : lhs; }
};

namespace reduce_detail {

template <typename T, typename RedOp>
struct IsFloatSum : std::false_type {};

template <>
struct IsFloatSum<float, Sum<float>> : std::true_type {};

}  // namespace reduce_detail

template <typename T, typename RedOp>
class ReducePrimitives {
 public:
  // Element-wise reduction of all contiguous input spans into output.
  void reduce(std::vector<Span<T const>> const& inputs, Span<T> output) const {
    checkReduction(inputs, output);
    if constexpr (reduce_detail::IsFloatSum<T, RedOp>::value) {
      std::vector<float const*> pointers(inputs.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        pointers[i] = inputs[i].data;
      }
      detail::reduceFloatSum(pointers.data(), pointers.size(), output.data,
                             output.count);
    } else {
      reduceScalar(inputs, output);
    }
  }

  /**
  Reduce the target row from every source matrix into one output span.

  The vector contains exactly the sources participating in this operation;
  therefore a separate sourceCount argument is unnecessary. All source rows
  and the output must have the same element count.
  */
  void reduceRows(std::vector<Rows<T const>> const& inputs, size_t targetRow,
                  Span<T> output) const {
    std::vector<Span<T const>> rows(inputs.size());
    for (size_t source = 0; source < inputs.size(); ++source) {
      checkRows(inputs[source]);
      rows[source] = inputs[source].row(targetRow);
    }
    reduce(rows, output);
  }

  /**
  Reduce two target rows from every source matrix into two output spans.

  For float addition, both rows are reduced in one fused pass. Other reduction
  operations reuse the single-row primitive twice. The vector must contain at
  least one source, and all sources participate in both reductions.
  */
  void reduceTwoRows(std::vector<Rows<T const>> const& inputs,
                     size_t firstTarget, Span<T> firstOutput,
                     size_t secondTarget,
                     Span<T> secondOutput) const {
    if (inputs.empty()) {
      throw std::invalid_argument("CpuSwitch reduction needs an input");
    }
    if ((firstOutput.data == nullptr && firstOutput.count != 0) ||
        (secondOutput.data == nullptr && secondOutput.count != 0)) {
      throw std::invalid_argument(
          "CpuSwitch two-row reduction output is null");
    }
    for (size_t source = 0; source < inputs.size(); ++source) {
      auto const& input = inputs[source];
      checkRows(input);
      if (firstTarget >= input.rowCount || secondTarget >= input.rowCount ||
          input.stride != inputs[0].stride ||
          input.columnCount != firstOutput.count ||
          input.columnCount != secondOutput.count) {
        throw std::invalid_argument(
            "CpuSwitch two-row reduction shape mismatch");
      }
    }
    if constexpr (reduce_detail::IsFloatSum<T, RedOp>::value) {
      std::vector<float const*> pointers(inputs.size());
      for (size_t source = 0; source < inputs.size(); ++source) {
        pointers[source] = inputs[source].data;
      }
      detail::reduceTwoFloatSum(
          pointers.data(), pointers.size(), inputs[0].stride, firstTarget,
          firstOutput.data, secondTarget, secondOutput.data, firstOutput.count);
    } else {
      reduceRows(inputs, firstTarget, firstOutput);
      reduceRows(inputs, secondTarget, secondOutput);
    }
  }

  // Reduce input with accumulator, and overwrite accumulator with the result.
  void reduceInPlace(Span<T> accumulator, Span<T const> input) const {
    if (accumulator.count != input.count ||
        (accumulator.data == nullptr && accumulator.count != 0) ||
        (input.data == nullptr && input.count != 0)) {
      throw std::invalid_argument("CpuSwitch in-place reduction shape mismatch");
    }
    std::vector<Span<T const>> inputs{
        Span<T const>{accumulator.data, accumulator.count,
                      accumulator.numaNode, accumulator.device},
        input};
    reduce(inputs, accumulator);
  }

 private:
  template <typename RowT>
  static void checkRows(Rows<RowT> rows) {
    // Each row must fit within its stride, and non-empty rows need valid data.
    if (rows.stride < rows.columnCount ||
        (rows.data == nullptr && rows.rowCount * rows.columnCount != 0)) {
      throw std::invalid_argument("CpuSwitch rows are invalid");
    }
  }

  static void checkReduction(std::vector<Span<T const>> const& inputs,
                             Span<T> output) {
    if (inputs.empty()) {
      throw std::invalid_argument("CpuSwitch reduction needs an input");
    }
    if (output.data == nullptr && output.count != 0) {
      throw std::invalid_argument("CpuSwitch reduction output is null");
    }
    for (auto const& input : inputs) {
      if (input.count != output.count ||
          (input.data == nullptr && input.count != 0)) {
        throw std::invalid_argument("CpuSwitch reduction shapes do not match");
      }
    }
  }

  static void reduceScalar(std::vector<Span<T const>> const& inputs,
                           Span<T> output) {
    for (size_t element = 0; element < output.count; ++element) {
      T value = inputs[0].data[element];
      for (size_t source = 1; source < inputs.size(); ++source) {
        value = RedOp::apply(value, inputs[source].data[element]);
      }
      output.data[element] = value;
    }
  }
};

}  // namespace mscclpp::lite

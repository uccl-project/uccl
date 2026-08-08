// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SERIALIZATION_HPP_
#define MSCCLPP_SERIALIZATION_HPP_

#include <algorithm>
#include <vector>

namespace mscclpp::detail {

template <typename T>
void serialize(std::vector<char>& buffer, const T& value) {
  char const* data = reinterpret_cast<char const*>(&value);
  std::copy_n(data, sizeof(T), std::back_inserter(buffer));
}

template <typename T>
std::vector<char>::const_iterator deserialize(
    const std::vector<char>::const_iterator& pos, T& value) {
  std::copy_n(pos, sizeof(T), reinterpret_cast<char*>(&value));
  return pos + sizeof(T);
}

}  // namespace mscclpp::detail

#endif  // MSCCLPP_SERIALIZATION_HPP_

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_UTILS_HPP_
#define MSCCLPP_UTILS_HPP_

#include "core.hpp"
#include <functional>
#include <string>

namespace mscclpp {
namespace detail {

// Refer
// https://www.boost.org/doc/libs/1_86_0/libs/container_hash/doc/html/hash.html#combine
template <typename T>
inline void hashCombine(std::size_t& seed, const T& value) {
  std::hash<T> hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace detail

/// Get the host name of the system.
/// @param maxlen The maximum length of the returned string.
/// @param delim The delimiter to use for the host name; if the delimiter is
/// found before maxlen characters, the host name will be truncated at that
/// point.
/// @return The host name of the system, truncated to maxlen characters if
/// necessary, and split by the specified delimiter.
/// @throw Error if it fails to retrieve the host name (error code:
/// SystemError).
std::string getHostName(int maxlen, char const delim);

/// Get the number of available InfiniBand devices.
///
/// @return The number of available InfiniBand devices.
int getIBDeviceCount();

/// Get the name of the InfiniBand device associated with the specified
/// transport.
///
/// @param ibTransport The InfiniBand transport to get the device name for.
/// @return The name of the InfiniBand device associated with the specified
/// transport.
std::string getIBDeviceName(Transport ibTransport);

/// Get the InfiniBand transport associated with the specified device name.
///
/// @param ibDeviceName The name of the InfiniBand device to get the transport
/// for.
/// @return The InfiniBand transport associated with the specified device name.
Transport getIBTransportByDeviceName(std::string const& ibDeviceName);

}  // namespace mscclpp

#endif  // MSCCLPP_UTILS_HPP_

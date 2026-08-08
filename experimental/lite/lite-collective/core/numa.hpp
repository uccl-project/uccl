// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_NUMA_HPP_
#define MSCCLPP_NUMA_HPP_

namespace mscclpp {

/// Return the NUMA node ID of the given GPU device ID.
/// @param deviceId The GPU device ID.
/// @return The NUMA node ID of the device.
/// @throw Error if the device ID is invalid or if the NUMA node cannot be
/// determined.
int getDeviceNumaNode(int deviceId);

/// NUMA bind the current thread to the specified NUMA node.
/// @param node The NUMA node ID to bind to.
/// @throw Error if the given NUMA node ID is invalid.
void numaBind(int node);

}  // namespace mscclpp

#endif  // MSCCLPP_NUMA_HPP_

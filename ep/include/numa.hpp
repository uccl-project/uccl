// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_NUMA_HPP_
#define MSCCLPP_NUMA_HPP_

namespace mscclpp {

int getDeviceNumaNode(int cudaDev);
void numaBind(int node);

}  // namespace mscclpp

#endif  // MSCCLPP_NUMA_HPP_

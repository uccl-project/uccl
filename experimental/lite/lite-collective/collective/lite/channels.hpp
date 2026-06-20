// channels.hpp — all four channel types in one include.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │          CPU-initiated                  GPU-initiated                   │
// ├─────────────────────────────────────────────────────────────────────────┤
// │ CpuStagingChannel   │ local D2H/H2D │ GpuStagingChannel                │
// │ cpu_staging_channel.hpp  (stream API)│ gpu_staging_channel.hpp (ring)  │
// ├─────────────────────────────────────────────────────────────────────────┤
// │ CpuPortChannel      │ inter-node    │ GpuPortChannel                   │
// │ cpu_port_channel.hpp  (direct RDMA) │ gpu_port_channel.hpp (ring proxy)│
// └─────────────────────────────────────────────────────────────────────────┘
//
// Design invariants:
//   - {Cpu,Gpu}StagingChannel share CscCtrl and slab layout: wait()/get()
//     from CpuStagingChannel can consume data staged by GpuStagingChannel.
//   - {Cpu,Gpu}PortChannel share GpcCtrl semaphore: wait() from either CPU
//     or GPU (GpcDeviceHandle::wait()) can observe a signal from either side.

#pragma once
#include "cpu_staging_channel.hpp"
#include "gpu_staging_channel.hpp"
#include "cpu_port_channel.hpp"
#include "gpu_port_channel.hpp"

// host_staging_buffer.hpp — backward-compatibility shim.
// The canonical implementation now lives in cpu_staging_channel.hpp.
#pragma once
#include "cpu_staging_channel.hpp"

// Legacy type aliases.
using HsbCounter     = CscCounter;
using HsbCtrl        = CscCtrl;
using HsbDeviceHandle = CscDeviceHandle;
using HostStagingBuffer = CpuStagingChannel;

static constexpr int kHsbMaxRanks  = kCscMaxRanks;
static constexpr int kHsbMaxSlots  = kCscMaxSlots;
static constexpr int kHsbMaxChunks = kCscMaxChunks;

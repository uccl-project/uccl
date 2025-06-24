#pragma once
#include "common.hpp"
#include "copy_ring.hpp"
#include <atomic>
#include <thread>
#include <cuda_runtime.h>

extern CopyRing g_ring;
extern std::atomic<bool> g_run;
#ifdef ENABLE_PROXY_CUDA_MEMCPY
extern thread_local uint64_t async_memcpy_count;
extern thread_local uint64_t async_memcpy_total_time;
#endif
extern int src_device;

void peer_copy_worker(CopyRing& g_ring, int idx);
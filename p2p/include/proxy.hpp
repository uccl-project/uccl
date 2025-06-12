#ifndef PROXY_HPP
#define PROXY_HPP

#include "ring_buffer.cuh"
#include <chrono>
#include <thread>
#include <vector>
#include <immintrin.h>

struct ProxyCtx {
  RingBuffer* rb_host;  // host pointer (CPU visible address of RingBuffer)
  int my_rank;          // rank id for this proxy (if simulating multiple)
};

void cpu_proxy(RingBuffer* rb, int block_idx, void* gpu_buffer,
               size_t total_size, int rank, char const* peer_ip);
void cpu_proxy_local(RingBuffer* rb, int block_idx);

#endif  // PROXY_HPP
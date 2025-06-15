#ifndef PROXY_HPP
#define PROXY_HPP

#include "ring_buffer.cuh"
#include <chrono>
#include <thread>
#include <unordered_map>
#include <vector>
#include <immintrin.h>

struct ProxyCtx {
  RingBuffer* rb_host;  // host pointer (CPU visible address of RingBuffer)
  int my_rank;          // rank id for this proxy (if simulating multiple)
};

void cpu_proxy(RingBuffer* rb, int block_idx, void* gpu_buffer,
               size_t total_size, int rank, char const* peer_ip);
void cpu_proxy_local(RingBuffer* rb, int block_idx);
void remote_cpu_proxy(RingBuffer* rb, int block_idx, void* gpu_buffer,
                      size_t total_size, int rank, char const* peer_ip);

// Proxy id to start time unordered_map
extern thread_local std::unordered_map<
    int, std::chrono::high_resolution_clock::time_point>
    wr_id_to_start_time;

extern thread_local uint64_t completion_count;
extern thread_local uint64_t wr_time_total;
#endif  // PROXY_HPP
#ifndef RING_BUFFER_FC_CUH
#define RING_BUFFER_FC_CUH

#include "ring_buffer.cuh"
#include <atomic>
#include <cuda_runtime.h>

namespace uccl {
namespace flat_combining {

// Maximum number of warps and proxies for this benchmark
constexpr uint32_t MAX_WARPS = 1024;
constexpr uint32_t MAX_PROXIES = 8;
constexpr uint32_t MAX_BATCH_SIZE = 32;

// Proxy warp list for efficient combiner scanning
struct ProxyWarpList {
  uint32_t* warp_ids;  // Dynamic array of warp IDs assigned to this proxy
  uint32_t count;      // Number of warps assigned to this proxy

  __host__ __device__ void init() {
    warp_ids = nullptr;
    count = 0;
  }
};

// Publication Record - each warp has one record
struct alignas(128) PublicationRecord {
  // Request state
  uint64_t volatile request_flag;  // 0=empty, 1=has_request, 2=being_processed
  TransferCmd cmd;                 // Command to be published
  uint32_t warp_id;                // Source warp ID

  // Payload information
  void* payload_ptr;      // Pointer to payload data (if any)
  uint32_t payload_size;  // Size of payload data

  // Response state
  uint64_t volatile response_flag;  // 0=not_completed, 1=completed

  // Timing for benchmarking
  uint64_t submit_time;    // GPU clock when submitted
  uint64_t complete_time;  // GPU clock when completed

  // Padding to avoid false sharing
  uint8_t padding[256 - sizeof(request_flag) - sizeof(cmd) - sizeof(warp_id) -
                  sizeof(payload_ptr) - sizeof(payload_size) -
                  sizeof(response_flag) - sizeof(submit_time) -
                  sizeof(complete_time)];

  __host__ __device__ void init() {
    request_flag = 0;
    response_flag = 0;
    submit_time = 0;
    complete_time = 0;
    warp_id = 0;
  }

  __device__ void submit_request(TransferCmd const& new_cmd, uint32_t w_id,
                                 void* payload = nullptr, uint32_t size = 0) {
    cmd = new_cmd;
    warp_id = w_id;
    payload_ptr = payload;
    payload_size = size;
    submit_time = clock64();
    __threadfence();
    request_flag = 1;
  }

  __device__ void wait_completion() {
    while (response_flag == 0) {
      __threadfence();
    }
    complete_time = clock64();
    response_flag = 0;  // Reset for next use
  }

  __device__ uint64_t get_latency_cycles() const {
    return complete_time - submit_time;
  }
};

// Publication List Manager
struct PublicationList {
  PublicationRecord* records;  // Array of records, one per warp
  uint32_t num_warps;

  __host__ __device__ void init(uint32_t n_warps) {
    num_warps = n_warps;
    for (uint32_t i = 0; i < num_warps; i++) {
      records[i].init();
    }
  }

  __host__ __device__ PublicationRecord* get_record(uint32_t warp_id) {
    return &records[warp_id];
  }
};

// Flat Combining Ring Buffer Manager
class FCRingBufferManager {
 private:
  // Ring buffers (one per proxy)
  DeviceToHostCmdBuffer** ring_buffers_;
  uint32_t num_proxies_;
  uint32_t num_warps_;

  // Publication lists
  PublicationList* pub_lists_;

  // Warp to proxy mapping
  uint32_t* warp_to_proxy_map_;
  uint32_t*
      proxy_to_combiner_map_;  // Which warp is the combiner for each proxy

  // Optimized combiner scanning - each proxy has a list of its warps
  ProxyWarpList* proxy_warp_lists_;

  // Payload buffer pools (one per proxy for staging payload data)
  uint8_t** payload_buffers_;     // Pinned memory for payload staging
  uint32_t payload_buffer_size_;  // Size of each buffer
  uint32_t* payload_write_ptrs_;  // Current write position in each buffer

  // Statistics
  struct Stats {
    uint64_t total_requests;
    uint64_t total_batches;
    uint64_t total_combines;
    uint64_t total_latency_cycles;
    uint64_t max_batch_size;
    uint64_t total_payload_bytes;
  } stats_;

 public:
  __host__ __device__ void init(
      DeviceToHostCmdBuffer** rbs, uint32_t num_proxies, uint32_t num_warps,
      PublicationList* pub_list, uint32_t* warp_to_proxy,
      uint32_t* proxy_to_combiner, uint8_t** payload_bufs = nullptr,
      uint32_t payload_buf_size = 0, uint32_t* payload_write_ptrs = nullptr,
      ProxyWarpList* proxy_warp_lists = nullptr) {
    ring_buffers_ = rbs;
    num_proxies_ = num_proxies;
    num_warps_ = num_warps;
    pub_lists_ = pub_list;
    warp_to_proxy_map_ = warp_to_proxy;
    proxy_to_combiner_map_ = proxy_to_combiner;
    payload_buffers_ = payload_bufs;
    payload_buffer_size_ = payload_buf_size;
    payload_write_ptrs_ = payload_write_ptrs;

    // Set proxy warp lists
    proxy_warp_lists_ = proxy_warp_lists;

    // Initialize stats
    stats_.total_requests = 0;
    stats_.total_batches = 0;
    stats_.total_combines = 0;
    stats_.total_latency_cycles = 0;
    stats_.max_batch_size = 0;
    stats_.total_payload_bytes = 0;
  }

  // Submit request (called by regular warps)
  __device__ void submit_request(uint32_t warp_id, TransferCmd const& cmd) {
    auto* record = pub_lists_->get_record(warp_id);
    record->submit_request(cmd, warp_id, nullptr, 0);
    record->wait_completion();
  }

  // Submit request with payload data
  __device__ void submit_request_with_payload(uint32_t warp_id,
                                              TransferCmd const& cmd,
                                              void* payload_data,
                                              uint32_t payload_size) {
    auto* record = pub_lists_->get_record(warp_id);
    record->submit_request(cmd, warp_id, payload_data, payload_size);
    record->wait_completion();
  }

  // Combiner main loop (called by combiner warps)
  __device__ void run_combiner(uint32_t combiner_warp_id, uint32_t proxy_id,
                               bool volatile* stop_flag) {
    // Boundary check: verify this is a valid proxy
    if (proxy_id >= num_proxies_) {
      printf("Error: Invalid proxy_id %u for combiner warp %u\n", proxy_id,
             combiner_warp_id);
      return;
    }

    auto* ring_buffer = ring_buffers_[proxy_id];
    if (ring_buffer == nullptr) {
      printf("Error: Ring buffer for proxy %u is null\n", proxy_id);
      return;
    }

    TransferCmd batch[MAX_BATCH_SIZE];
    uint32_t batch_warp_ids[MAX_BATCH_SIZE];

    while (!(*stop_flag)) {
      uint32_t batch_count = 0;

      // Step 1: Scan and collect requests from assigned warps
      if (proxy_warp_lists_ != nullptr &&
          proxy_warp_lists_[proxy_id].warp_ids != nullptr) {
        // Optimized path: only scan warps assigned to this proxy
        ProxyWarpList const& my_warps = proxy_warp_lists_[proxy_id];
        for (uint32_t i = 0; i < my_warps.count && batch_count < MAX_BATCH_SIZE;
             i++) {
          uint32_t w = my_warps.warp_ids[i];
          auto* record = pub_lists_->get_record(w);
          if (record->request_flag == 1) {
            // Mark as being processed
            record->request_flag = 2;

            batch[batch_count] = record->cmd;
            batch_warp_ids[batch_count] = w;
            batch_count++;
          }
        }
      } else {
        // Fallback path: scan all warps (original behavior)
        for (uint32_t w = 0; w < pub_lists_->num_warps; w++) {
          if (warp_to_proxy_map_[w] == proxy_id) {
            auto* record = pub_lists_->get_record(w);
            if (record->request_flag == 1 && batch_count < MAX_BATCH_SIZE) {
              // Mark as being processed
              record->request_flag = 2;

              batch[batch_count] = record->cmd;
              batch_warp_ids[batch_count] = w;
              batch_count++;
            }
          }
        }
      }

      // Step 2: If we have requests, submit them to ring buffer
      if (batch_count > 0) {
        // Try to reserve space in ring buffer
        uint64_t start_slot;
        bool success = false;

        // Simple spin until we get space
        while (!success) {
          uint64_t current_head = ring_buffer->head;
          uint64_t current_tail = ring_buffer->volatile_tail();
          uint64_t available_space =
              ring_buffer->capacity - (current_head - current_tail);

          if (available_space >= batch_count) {
            // Try to atomically reserve the space
            unsigned long long old_head =
                atomicCAS((unsigned long long*)&ring_buffer->head,
                          (unsigned long long)current_head,
                          (unsigned long long)(current_head + batch_count));

            if (old_head == current_head) {
              start_slot = current_head;
              success = true;
            }
          } else {
            __nanosleep(100);  // Wait a bit if buffer is full
          }
        }

        // Step 3: Write commands to ring buffer and copy payloads if available
        for (uint32_t i = 0; i < batch_count; i++) {
          uint32_t idx = (start_slot + i) & ring_buffer->mask();
          ring_buffer->buf[idx] = batch[i];

          // Copy payload data if available
          auto* record = pub_lists_->get_record(batch_warp_ids[i]);
          if (record->payload_ptr != nullptr && record->payload_size > 0) {
            // If we have payload buffers, copy the data
            if (payload_buffers_ != nullptr &&
                payload_buffers_[proxy_id] != nullptr) {
              // Use circular buffer approach - wrap around when full
              uint32_t write_pos = atomicAdd(&payload_write_ptrs_[proxy_id],
                                             record->payload_ size);
              uint32_t actual_pos = write_pos % payload_buffer_size_;

              // Check if we have enough contiguous space
              if (actual_pos + record->payload_size <= payload_buffer_size_) {
                // Copy payload data to staging buffer
                uint8_t* dst = payload_buffers_[proxy_id] + actual_pos;
                memcpy(dst, record->payload_ptr, record->payload_size);

                // Update command to point to staging buffer
                ring_buffer->buf[idx].bytes = record->payload_size;
              } else {
                // Wrap around - split the copy (simplified: just reset to
                // beginning)
                uint8_t* dst = payload_buffers_[proxy_id];
                memcpy(dst, record->payload_ptr, record->payload_size);
                ring_buffer->buf[idx].bytes = record->payload_size;
                // Reset write pointer to avoid overflow
                atomicExch(&payload_write_ptrs_[proxy_id],
                           record->payload_size);
              }

              // Track payload bytes
              atomicAdd((unsigned long long*)&stats_.total_payload_bytes,
                        record->payload_size);
            }
          }
        }

        // Ensure visibility to CPU
        __threadfence_system();

        // Step 4: Notify completion to all request warps
        for (uint32_t i = 0; i < batch_count; i++) {
          auto* record = pub_lists_->get_record(batch_warp_ids[i]);
          record->request_flag = 0;  // Reset request flag
          __threadfence();
          record->response_flag = 1;  // Signal completion
        }

        // Update statistics
        atomicAdd((unsigned long long*)&stats_.total_requests, batch_count);
        atomicAdd((unsigned long long*)&stats_.total_batches, 1);
        atomicMax((unsigned long long*)&stats_.max_batch_size, batch_count);
      }
    }
  }

  // Get statistics
  __host__ __device__ Stats get_stats() const { return stats_; }

  // Reset statistics
  __host__ __device__ void reset_stats() {
    stats_.total_requests = 0;
    stats_.total_batches = 0;
    stats_.total_combines = 0;
    stats_.total_latency_cycles = 0;
    stats_.max_batch_size = 0;
  }

  // Public accessors for benchmark
  __device__ PublicationList* get_pub_lists() { return pub_lists_; }
  __device__ DeviceToHostCmdBuffer** get_ring_buffers() {
    return ring_buffers_;
  }
  __device__ Stats* get_stats_ptr() { return &stats_; }
  __device__ uint32_t* get_warp_to_proxy_map() { return warp_to_proxy_map_; }
  __device__ uint32_t* get_proxy_to_combiner_map() {
    return proxy_to_combiner_map_;
  }
};

// Invalid combiner marker
constexpr uint32_t INVALID_COMBINER = UINT32_MAX;

// Mapping utilities
__host__ __device__ inline uint32_t get_proxy_for_warp(uint32_t warp_id,
                                                       uint32_t num_proxies) {
  return warp_id % num_proxies;
}

// Get the combiner warp for a proxy (the minimum warp ID mapped to this proxy)
__host__ __device__ inline uint32_t get_combiner_for_proxy(
    uint32_t proxy_id, uint32_t num_warps, uint32_t num_proxies) {
  // For modulo mapping, the combiner is the first warp that maps to this proxy
  // That is: the smallest warp_id where (warp_id % num_proxies == proxy_id)

  if (proxy_id < num_warps) {
    // When we have enough warps, proxy_id itself is the combiner
    return proxy_id;
  } else {
    // No warp maps to this proxy
    return INVALID_COMBINER;
  }
}

// Initialize warp to proxy mapping and build proxy warp lists
__host__ inline void init_warp_mapping(
    uint32_t num_warps, uint32_t num_proxies, uint32_t* warp_to_proxy_map,
    uint32_t* proxy_to_combiner_map,
    ProxyWarpList* proxy_warp_lists = nullptr) {
  // Step 1: Set warp to proxy mapping using modulo
  for (uint32_t w = 0; w < num_warps; w++) {
    warp_to_proxy_map[w] = get_proxy_for_warp(w, num_proxies);
  }

  // Step 2: Initialize combiner map to invalid
  for (uint32_t p = 0; p < num_proxies; p++) {
    proxy_to_combiner_map[p] = INVALID_COMBINER;
  }

  // Step 3: Find the minimum warp ID for each proxy
  for (uint32_t w = 0; w < num_warps; w++) {
    uint32_t proxy_id = warp_to_proxy_map[w];
    // Update combiner if this is the first warp we see for this proxy
    // or if this warp has a smaller ID
    if (proxy_to_combiner_map[proxy_id] == INVALID_COMBINER ||
        w < proxy_to_combiner_map[proxy_id]) {
      proxy_to_combiner_map[proxy_id] = w;
    }
  }

  // Step 4: Build proxy warp lists (if provided)
  if (proxy_warp_lists != nullptr) {
    // Count warps per proxy
    for (uint32_t p = 0; p < num_proxies; p++) {
      uint32_t count = 0;
      for (uint32_t w = 0; w < num_warps; w++) {
        if (warp_to_proxy_map[w] == p) count++;
      }

      // Allocate and fill warp list for this proxy
      proxy_warp_lists[p].count = count;
      if (count > 0) {
        proxy_warp_lists[p].warp_ids = new uint32_t[count];
        uint32_t idx = 0;
        for (uint32_t w = 0; w < num_warps; w++) {
          if (warp_to_proxy_map[w] == p) {
            proxy_warp_lists[p].warp_ids[idx++] = w;
          }
        }
      } else {
        proxy_warp_lists[p].warp_ids = nullptr;
      }
    }
  }

  // Verification: print mapping for debugging - DISABLED for performance
  // printf("Warp to Proxy mapping (num_warps=%u, num_proxies=%u):\n",
  // num_warps, num_proxies); for (uint32_t p = 0; p < num_proxies; p++) {
  //     printf("  Proxy %u: combiner=", p);
  //     if (proxy_to_combiner_map[p] != INVALID_COMBINER) {
  //         printf("warp %u, assigned warps={", proxy_to_combiner_map[p]);
  //         bool first = true;
  //         for (uint32_t w = 0; w < num_warps; w++) {
  //             if (warp_to_proxy_map[w] == p) {
  //                 if (!first) printf(",");
  //                 printf("%u", w);
  //                 first = false;
  //             }
  //         }
  //         printf("}\n");
  //     } else {
  //         printf("NONE (no warps assigned)\n");
  //     }
  // }
}

}  // namespace flat_combining
}  // namespace uccl

#endif  // RING_BUFFER_FC_CUH
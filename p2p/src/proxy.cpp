#include "ring_buffer.cuh"
#include <thread>
#include <vector>
#include <chrono>
#include "proxy.hpp"

#ifdef NO_RDMA
#include <cuda_runtime.h>
void rdma_write_stub(int dst_rank, void* local_dev_ptr, size_t bytes) {
    // Local simulation: do nothing (assume remote node copies).
    (void)dst_rank; (void)local_dev_ptr; (void)bytes;
}
#else
#include <infiniband/verbs.h>
// A minimal wrapper around an RDMA WRITE. Production code should create
// protection domain, queue pair, registered MR, CQ, etc. For brevity we show
// only a conceptual stub; fill with your own ibv_* calls when integrating.
void rdma_write_stub(int, void*, size_t) {
    fprintf(stderr, "[RDMA] real implementation required\n");
}
#endif

void proxy_loop(ProxyCtx ctx) {
    auto& rb = *ctx.rb_host;
    uint64_t tail = 0;
    for (;;) {
        uint64_t head = rb.head.load(std::memory_order_acquire);
        while (tail < head) {
            uint32_t idx = tail & QUEUE_MASK;
            auto& cmd = rb.buf[idx];
            rdma_write_stub(cmd.dst_rank, cmd.src_ptr, cmd.bytes);
            tail++;
            rb.tail.store(tail, std::memory_order_release);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}
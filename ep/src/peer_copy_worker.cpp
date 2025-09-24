#include "peer_copy_worker.hpp"
#include "common.hpp"
#include "peer_copy.cuh"
#include "proxy.hpp"
#include "rdma.hpp"
#include <mutex>

void sync_and_post(PeerWorkerCtx& ctx, CopyRingBuffer& ring,
                   gpuStream_t& stream, int idx) {
  if (ctx.async_memcpy_count > ctx.prev_completed_async_memcpy_count) {
    gpuError_t err = gpuStreamSynchronize(stream);
    if (err != gpuSuccess) {
      fprintf(stderr, "Kernel execution failed: %s\n", gpuGetErrorString(err));
      std::abort();
    }
    remote_send_ack((ProxyCtx*)ring.ctx, ring.ack_qp, ctx.highest_issued_wr_id,
                    ring.ack_mr, ring.ack_buf, idx);
    ctx.prev_completed_async_memcpy_count = ctx.async_memcpy_count;
  }
}

void peer_copy_worker(PeerCopyShared& shared, PeerWorkerCtx& ctx,
                      CopyRingBuffer& ring, int idx) {
  pin_thread_to_cpu(idx + 1 + MAIN_THREAD_CPU_IDX);
  // TODO(MaoZiming): improves pinning.
  printf("Peer copy worker %d started on CPU core %d\n", idx + 1,
         sched_getcpu());
  gpuStream_t stream;
  GPU_RT_CHECK(gpuSetDevice(shared.src_device));
  GPU_RT_CHECK(gpuStreamCreate(&stream));
  CopyTask* d_tasks;
  GPU_RT_CHECK(
      gpuMallocAsync(&d_tasks, RECEIVER_BATCH_SIZE * sizeof(CopyTask), stream));

  while (shared.run.load(std::memory_order_acquire)) {
    CopyTask t;
    int copy_batch_size = 0;
    if (RECEIVER_BATCH_SIZE == 1) {
      if (!ring.pop(t)) {
        continue;
      }
      copy_batch_size = 1;
      ctx.tasks[0] = t;
    } else {
      int n = ring.popN(ctx.tasks, RECEIVER_BATCH_SIZE);
      if (n == 0) {
        continue;
      }
      t = ctx.tasks[0];
      copy_batch_size = n;
    }
    for (int i = 0; i < copy_batch_size; ++i) {
      maybe_enable_peer_access(shared.src_device, ctx.tasks[i].dst_dev);
      ctx.task_wrs[i] = ctx.tasks[i].wr_id;
    }
    ctx.highest_issued_wr_id =
        std::max(ctx.highest_issued_wr_id, ctx.task_wrs[copy_batch_size - 1]);
    // NOTE(MaoZiming): peer_copy.cu has some kernels such as
    // launch_peer_bulk_copy2 that might be good.
    gpuError_t err;
    if (t.dst_ptr) {
      err =
          gpuMemcpyPeerAsync(t.dst_ptr, t.dst_dev, t.src_ptr, shared.src_device,
                             t.bytes * copy_batch_size, stream);
    }
    std::string func_name = "gpuMemcpyPeerAsync";
    if (err != gpuSuccess) {
      fprintf(stderr,
              "%s failed (%s) wr_id=%llu\n"
              "  dst_ptr=%p dst_dev=%d\n"
              "  src_ptr=%p src_dev=%d\n"
              "  size=%zu (bytes)\n"
              "  stream=%p\n",
              func_name.c_str(), gpuGetErrorString(err),
              static_cast<unsigned long long>(t.wr_id), t.dst_ptr, t.dst_dev,
              t.src_ptr, shared.src_device,
              static_cast<size_t>(t.bytes * copy_batch_size), (void*)stream);
      std::abort();
    }
    ctx.async_memcpy_count += copy_batch_size;
    sync_and_post(ctx, ring, stream, idx);
  }
  GPU_RT_CHECK(gpuFreeAsync(d_tasks, stream));
  GPU_RT_CHECK(gpuStreamSynchronize(stream));
  GPU_RT_CHECK(gpuStreamDestroy(stream));
}
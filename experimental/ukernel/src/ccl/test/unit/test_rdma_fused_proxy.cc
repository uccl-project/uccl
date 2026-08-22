#include "backend/rdma_fused_proxy.h"
#include <cassert>
#include <cstdio>
#include <cstdint>

using namespace UKernel::CCL;

int main() {
  uint64_t posted_index = UINT64_MAX;
  size_t posted_count = 0;
  RdmaFusedProxy proxy(
      [&](uint64_t idx) {
        posted_index = idx;
        ++posted_count;
        return true;
      },
      64, 64);

  Cmd put{};
  put.kind = ExecOpKind::Put;
  put.dst_peer = 1;
  put.bytes = 1024;
  void* fake_run = reinterpret_cast<void*>(0x1234);
  uint64_t idx = proxy.pool().alloc(put, fake_run, 7, PutPath::Rdma);
  assert(idx != UINT64_MAX);
  proxy.ring().push_from_host(idx);
  assert(proxy.progress() == 1);
  assert(posted_count == 1);
  assert(posted_index == idx);

  // Pool slot should be released after successful post.
  uint64_t idx2 = proxy.pool().alloc(put, fake_run, 8, PutPath::Rdma);
  assert(idx2 == idx);  // reused the freed slot
  proxy.ring().push_from_host(idx2);
  assert(proxy.progress() == 1);
  assert(posted_count == 2);
  assert(posted_index == idx2);

  // Empty ring -> no-op.
  assert(proxy.progress() == 0);

  // Failed post should NOT release the slot.
  RdmaFusedProxy fail_proxy(
      [&](uint64_t) { return false; },
      8, 8);
  uint64_t fidx = fail_proxy.pool().alloc(put, nullptr, 0, PutPath::Rdma);
  fail_proxy.ring().push_from_host(fidx);
  assert(fail_proxy.progress() == 0);
  assert(fail_proxy.pool().get(fidx).cmd.bytes == 1024);
  // Still allocated: second alloc should get a different slot (or fail if
  // capacity 8 but only one used? Actually not released so second alloc
  // gets another free slot).
  uint64_t fidx2 = fail_proxy.pool().alloc(put, nullptr, 0, PutPath::Rdma);
  assert(fidx2 != fidx);

  std::printf("test_rdma_fused_proxy: OK\n");
  return 0;
}

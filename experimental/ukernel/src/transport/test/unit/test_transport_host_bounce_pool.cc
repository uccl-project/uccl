#include "memory/memory_manager.h"
#include "test.h"
#include "test_utils.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

namespace {

using UKernel::Transport::BounceCpuBufferPool;
using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;
using UKernel::Transport::TestUtil::throws;

void test_pool_reuse_and_registration() {
  std::atomic<int> register_calls{0};
  std::vector<uint64_t> deregistered;

  {
    BounceCpuBufferPool pool(
        [&](uint64_t, void*, size_t) {
          ++register_calls;
          return true;
        },
        [&](uint64_t mr_id) { deregistered.push_back(mr_id); });

    auto lease0 = pool.acquire(1024, false);
    require(lease0.valid(), "first lease should be valid");
    require(lease0.bytes >= 4096, "lease should round up to bucket capacity");
    require(!lease0.uccl_registered,
            "non-UCCL lease should not be pre-registered");
    auto first_ptr = lease0.ptr;
    auto first_id = lease0.mr_id;
    pool.release(lease0);

    auto lease1 = pool.acquire(2048, false);
    require(lease1.ptr == first_ptr, "pool should reuse idle bucket");
    require(lease1.mr_id == first_id, "reused lease should keep MR id");
    pool.release(lease1);

    auto lease2 = pool.acquire(2048, true);
    require(lease2.ptr == first_ptr,
            "registered lease should reuse the same bucket");
    require(lease2.uccl_registered,
            "registered lease should report UCCL registration");
    require(register_calls.load() == 1,
            "register callback should run once for first registration");
    pool.release(lease2);

    auto lease3 = pool.acquire(2048, true);
    require(lease3.uccl_registered,
            "reacquired registered lease should stay registered");
    require(register_calls.load() == 1,
            "registered idle lease should not re-register");
    pool.release(lease3);
  }

  require(deregistered.size() == 1,
          "destructor should deregister one retained registered entry");
}

void test_pool_eviction_and_registration_failures() {
  std::vector<uint64_t> deregistered;

  {
    BounceCpuBufferPool pool(
        [](uint64_t, void*, size_t) { return true; },
        [&](uint64_t mr_id) { deregistered.push_back(mr_id); });

    auto a = pool.acquire(1024, true);
    auto b = pool.acquire(1024, true);
    auto c = pool.acquire(1024, true);
    require(a.valid() && b.valid() && c.valid(),
            "all concurrently acquired leases should be valid");

    auto c_id = c.mr_id;
    pool.release(a);
    pool.release(b);
    pool.release(c);

    require(deregistered.size() == 1,
            "releasing beyond idle cap should evict one entry");
    require(deregistered[0] == c_id,
            "newly released entry should be evicted when idle cap is exceeded");
  }

  require(deregistered.size() == 3,
          "destructor should deregister the remaining retained entries");

  BounceCpuBufferPool failing_pool(
      [](uint64_t, void*, size_t) { return false; }, [](uint64_t) {});
  require(throws([&] { (void)failing_pool.acquire(1024, true); }),
          "failed registration should throw");
}

}  // namespace

void test_transport_host_bounce_pool() {
  run_case("transport unit", "host bounce pool reuse and registration",
           test_pool_reuse_and_registration);
  run_case("transport unit", "host bounce pool eviction and failures",
           test_pool_eviction_and_registration_failures);
}

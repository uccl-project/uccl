#include "memory/shm_manager.h"
#include "test.h"
#include "test_utils.h"

namespace {

using UKernel::Transport::SHMManager;
using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

void test_shm_create_delete_and_map() {
  SHMManager shm;
  auto a = shm.create_local_shm(1024, false);
  auto b = shm.create_local_shm(2048, true);

  require(a.valid && b.valid, "shm items should be valid");
  require(a.bytes == 1024 && b.bytes == 2048, "shm item sizes should match");
  require(a.shm_id != b.shm_id, "shm ids should be unique");
  require(!a.shareable && b.shareable, "shareability should follow creation mode");

  auto mapped = shm.open_remote_shm(b.shm_name);
  require(mapped.valid && mapped.ptr != nullptr, "mapped shm should be valid");

  require(shm.delete_local_shm(a.shm_id),
          "delete local anonymous shm should succeed");
  require(shm.delete_local_shm(b.shm_id), "delete local shared shm should succeed");
  require(shm.close_remote_shm(b.shm_name), "delete mapped shm cache should succeed");
}

}  // namespace

void test_transport_host_bounce_pool() {
  run_case("transport unit", "shm manager create/delete/map",
           test_shm_create_delete_and_map);
}

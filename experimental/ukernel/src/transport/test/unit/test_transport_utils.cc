#include "test.h"
#include "test_utils.h"
#include "utils.h"

using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;
using UKernel::Transport::TestUtil::ScopedEnvVar;

void test_generate_host_id() {
  run_case("transport unit", "generate host id", [] {
    auto id = UKernel::Transport::generate_host_id(false);
    auto id_with_ip = UKernel::Transport::generate_host_id(true);
    require(!id.empty(), "generate_host_id(false) should not be empty");
    require(!id_with_ip.empty(), "generate_host_id(true) should not be empty");

    ScopedEnvVar host_id_override("UHM_HOST_ID_OVERRIDE",
                                  "transport-test-host");
    auto override_id = UKernel::Transport::generate_host_id(false);
    auto override_id_with_ip = UKernel::Transport::generate_host_id(true);

    require(override_id == "transport-test-host",
            "host id override should take precedence");
    require(override_id_with_ip.rfind("transport-test-host", 0) == 0,
            "host id override should be reflected in include_ip path");
  });
}

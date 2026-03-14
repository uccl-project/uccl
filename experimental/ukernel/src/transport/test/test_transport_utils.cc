#include "test.h"
#include "utils.h"
#include <cstdlib>
#include <stdexcept>

namespace {

void require(bool cond, char const* message) {
  if (!cond) {
    throw std::runtime_error(message);
  }
}

}  // namespace

void test_generate_host_id() {
  auto id = UKernel::Transport::generate_host_id(false);
  auto id_with_ip = UKernel::Transport::generate_host_id(true);
  require(!id.empty(), "generate_host_id(false) should not be empty");
  require(!id_with_ip.empty(), "generate_host_id(true) should not be empty");

  ::setenv("UHM_HOST_ID_OVERRIDE", "transport-test-host", 1);
  auto override_id = UKernel::Transport::generate_host_id(false);
  auto override_id_with_ip = UKernel::Transport::generate_host_id(true);
  ::unsetenv("UHM_HOST_ID_OVERRIDE");

  require(override_id == "transport-test-host",
          "host id override should take precedence");
  require(override_id_with_ip.rfind("transport-test-host", 0) == 0,
          "host id override should be reflected in include_ip path");

  std::cout << "Test Utils: generate id " << id << " with ip " << id_with_ip
            << " override " << override_id << std::endl;
}

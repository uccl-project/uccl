#include "test.h"
#include "test_utils.h"
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "core") {
      test_transport_core();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-socket") {
      test_socket_oob();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-socket-meta") {
      test_socket_meta_exchange_multi_threads(
          UKernel::Transport::TestUtil::get_int_arg(argc, argv, "--world-size",
                                                    4));
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-shm") {
      test_shm_oob();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-serde") {
      test_transport_oob_exchangeables();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "host-bounce") {
      test_transport_host_bounce_pool();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "tcp-adapter") {
      test_transport_tcp_adapter();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "utils-host-id") {
      test_generate_host_id();
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "[transport unit test] fatal: " << e.what() << std::endl;
    return 2;
  }

  std::cerr << "Usage:\n"
            << "  test_transport_unit core\n"
            << "  test_transport_unit oob-socket\n"
            << "  test_transport_unit oob-socket-meta [--world-size N]\n"
            << "  test_transport_unit oob-shm\n"
            << "  test_transport_unit oob-serde\n"
            << "  test_transport_unit host-bounce\n"
            << "  test_transport_unit tcp-adapter\n"
            << "  test_transport_unit utils-host-id\n";
  return 1;
}

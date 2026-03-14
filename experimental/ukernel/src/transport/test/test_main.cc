#include "test.h"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

int get_int_arg(int argc, char** argv, char const* key, int def) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key && i + 1 < argc) {
      return std::atoi(argv[i + 1]);
    }
  }
  return def;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "core") {
      test_transport_core();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "communicator") {
      return test_transport_communicator(argc - 1, argv + 1);
    }
    if (argc > 1 && std::string(argv[1]) == "communicator-local") {
      test_transport_communicator_local();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-socket") {
      test_socket_oob();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-socket-meta") {
      test_socket_meta_exchange_multi_threads(
          get_int_arg(argc, argv, "--world-size", 4));
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-redis") {
      test_redis_oob();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-redis-meta") {
      test_redis_meta_exchange_multi_threads(
          get_int_arg(argc, argv, "--world-size", 4));
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "oob-uds") {
      test_uds_oob();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "utils-host-id") {
      test_generate_host_id();
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "[transport test] fatal: " << e.what() << std::endl;
    return 2;
  }

  std::cerr << "Usage:\n"
            << "  test_transport_main core\n"
            << "  test_transport_main communicator --role=server|client --case=basic|batch|poll-release|notifier [--exchanger-ip IP] [--exchanger-port PORT]\n"
            << "  test_transport_main communicator-local\n"
            << "  test_transport_main oob-socket\n"
            << "  test_transport_main oob-socket-meta [--world-size N]\n"
            << "  test_transport_main oob-redis\n"
            << "  test_transport_main oob-redis-meta [--world-size N]\n"
            << "  test_transport_main oob-uds\n"
            << "  test_transport_main utils-host-id\n";
  return 1;
}

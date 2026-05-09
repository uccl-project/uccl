#include "test.h"
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "communicator") {
      return test_transport_communicator(argc - 1, argv + 1);
    }
    if (argc > 1 && std::string(argv[1]) == "communicator-local") {
      test_transport_communicator_local();
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "[transport integration test] fatal: " << e.what()
              << std::endl;
    return 2;
  }

  std::cerr << "Usage:\n"
            << "  test_transport_integration communicator --role=server|client "
               "--case=exchange|ipc-buffer-meta [--exchanger-ip IP] "
               "[--exchanger-port PORT] [--transport auto|ipc|uccl|tcp] "
               "[--gpu GPU] [--peer-gpu GPU]\n"
            << "    IPC multi-GPU example: run both roles with "
               "CUDA_VISIBLE_DEVICES=6,7; defaults are server --gpu=0 and "
               "client --gpu=1.\n"
            << "  test_transport_integration communicator-local\n";
  return 1;
}

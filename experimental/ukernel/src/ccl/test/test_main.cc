#include "test.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc == 1) {
    test_ccl_plan();
    test_ccl_executor();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "ccl-plan") {
    test_ccl_plan();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "ccl-exec") {
    test_ccl_executor();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "ccl-rdma-ag") {
    return test_ccl_rdma_allgather(argc - 1, argv + 1);
  }
  std::cerr << "Usage: test_main [ccl-plan|ccl-exec|ccl-rdma-ag ...]\n";
  return 1;
}

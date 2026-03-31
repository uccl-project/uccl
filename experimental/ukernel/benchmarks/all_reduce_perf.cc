#include "ccl_perf_main.h"
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    return UKernel::CCL::Benchmark::run_perf_main(
        argc, argv, UKernel::CCL::CollectiveKind::AllReduce);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "[all_reduce_perf] fatal: %s\n", e.what());
    return 2;
  }
}

#pragma once

#include "../src/ccl/plan.h"

namespace UKernel {
namespace CCL {
namespace Benchmark {

int run_perf_main(int argc, char** argv, CollectiveKind collective);

}  // namespace Benchmark
}  // namespace CCL
}  // namespace UKernel

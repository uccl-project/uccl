#include "cpu_router_c.h"
#include "cpu_router.hpp"
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
namespace {
thread_local std::string error;
}
extern "C" void* uccl_lite_router_create() {
  try {
    uccl_lite::cpu_router::CpuRouterOptions options;
    if (char const* b = std::getenv("UCCL_LITE_TRANSPORT")) {
      if (std::string(b) == "dma")
        options.backend = uccl_lite::cpu_router::TransportBackend::Dma;
      else if (std::string(b) != "kernel")
        throw std::invalid_argument("unknown transport");
    }
    return new uccl_lite::cpu_router::CpuRouter(options);
  } catch (std::exception const& e) {
    error = e.what();
    return nullptr;
  }
}
extern "C" int uccl_lite_router_enqueue_n(void* router, int operation,
                                          size_t count, int ranks,
                                          void const* const* inputs,
                                          void* const* outputs,
                                          void* const* streams,
                                          int const* devices) {
  try {
    if (!router || operation < 0 || operation > 2 || ranks < 1 ||
        ranks > uccl_lite::cpu_router::kMaxRanks || !inputs || !outputs ||
        !streams || !devices)
      throw std::invalid_argument("invalid router handle/operation");
    std::vector<uccl_lite::cpu_router::RouterPort> ports(ranks);
    for (int rank = 0; rank < ranks; ++rank)
      ports[rank] = {inputs[rank], outputs[rank],
                     static_cast<cudaStream_t>(streams[rank]), devices[rank]};
    auto plan =
        operation == 0
            ? uccl_lite::cpu_router::RouterPlan::allReduce(count, ranks)
        : operation == 1
            ? uccl_lite::cpu_router::RouterPlan::allGather(count, ranks)
            : uccl_lite::cpu_router::RouterPlan::reduceScatter(count, ranks);
    static_cast<uccl_lite::cpu_router::CpuRouter*>(router)->enqueue(plan,
                                                                    ports);
    return 0;
  } catch (std::exception const& e) {
    error = e.what();
    return -1;
  }
}
extern "C" int uccl_lite_router_enqueue(
    void* router, int operation, size_t count, void const* const inputs[8],
    void* const outputs[8], void* const streams[8], int const devices[8]) {
  return uccl_lite_router_enqueue_n(router, operation, count, 8, inputs,
                                    outputs, streams, devices);
}
extern "C" int uccl_lite_router_destroy(void* router) {
  delete static_cast<uccl_lite::cpu_router::CpuRouter*>(router);
  return 0;
}
extern "C" char const* uccl_lite_router_last_error() { return error.c_str(); }
extern "C" int uccl_lite_router_graph_control(void* router, int action) {
  try {
    if (!router) throw std::invalid_argument("null router");
    auto* core = static_cast<uccl_lite::cpu_router::CpuRouter*>(router);
    switch (action) {
      case 0:
        core->beginCapture();
        break;
      case 1:
        core->endCapture();
        break;
      case 2:
        core->beginReplay();
        break;
      case 3:
        core->endReplay();
        break;
      default:
        throw std::invalid_argument("invalid graph action");
    }
    return 0;
  } catch (std::exception const& e) {
    error = e.what();
    return -1;
  }
}

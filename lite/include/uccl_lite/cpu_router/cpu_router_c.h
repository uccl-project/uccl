#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// operation: 0=AllReduce, 1=AllGather, 2=ReduceScatter; count is API element
// count. Streams are opaque CUDA stream pointers. The legacy entry point
// accepts eight ports. The count-aware entry point accepts 1..32 ports in
// arbitrary device order.
void* uccl_lite_router_create(void);
int uccl_lite_router_enqueue(void* router, int operation, size_t count,
                             void const* const inputs[8],
                             void* const outputs[8], void* const streams[8],
                             int const devices[8]);
int uccl_lite_router_enqueue_n(void* router, int operation, size_t count,
                               int ranks, void const* const* inputs,
                               void* const* outputs, void* const* streams,
                               int const* devices);
int uccl_lite_router_destroy(void* router);
// 0=begin capture, 1=end capture, 2=begin replay, 3=end replay.
int uccl_lite_router_graph_control(void* router, int action);
char const* uccl_lite_router_last_error(void);
#ifdef __cplusplus
}
#endif

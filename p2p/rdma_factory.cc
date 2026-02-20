// Must include engine.h to get full Endpoint definition
// Note: This will pull in pybind11 headers, but that's OK for factory compilation
#include "engine.h"

extern "C" {
    void* create_uccl_endpoint_rdma(int num_cpus) {
        return new Endpoint(num_cpus);
    }

    void destroy_uccl_endpoint_rdma(void* ep) {
        delete static_cast<Endpoint*>(ep);
    }
}

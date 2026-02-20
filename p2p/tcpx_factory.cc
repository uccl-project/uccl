#include "nccl_tcpx_endpoint.h"

extern "C" {
    void* create_uccl_endpoint_tcpx(int num_cpus) {
        return new nccl_tcpx::Endpoint(num_cpus);
    }

    void destroy_uccl_endpoint_tcpx(void* ep) {
        delete static_cast<nccl_tcpx::Endpoint*>(ep);
    }
}

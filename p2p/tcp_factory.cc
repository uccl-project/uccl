#include "engine.h"

extern "C" {
    void* create_uccl_endpoint_tcp(int num_cpus) {
        return new Endpoint(num_cpus);
    }

    void destroy_uccl_endpoint_tcp(void* ep) {
        delete static_cast<Endpoint*>(ep);
    }
}

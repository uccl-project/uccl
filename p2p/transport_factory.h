#pragma once

// Simple C-linkage factory interface for transport plugins
extern "C" {
    typedef void* (*create_endpoint_fn)(int num_cpus);
    typedef void (*destroy_endpoint_fn)(void* endpoint);
}

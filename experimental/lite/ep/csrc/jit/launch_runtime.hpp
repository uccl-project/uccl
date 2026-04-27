#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <deep_ep/common/exception.cuh>

#include "../utils/format.hpp"
#include "../utils/system.hpp"
#include "compiler.hpp"
#include "include_parser.hpp"

namespace deep_ep::jit {

struct LaunchArgs {
    std::pair<int, int> grid_dim;
    int num_threads;
    int smem_size;
    int cluster_dim;
    bool cooperative;
    bool pdl_enabled;

    LaunchArgs(const int& grid_dim_x, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1, const bool& cooperative = false, const bool& pdl_enabled = false):
        grid_dim({grid_dim_x, 1}), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim), cooperative(cooperative), pdl_enabled(pdl_enabled) {}

    LaunchArgs(const std::pair<int, int>& grid_dim, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1, const bool& cooperative = false, const bool& pdl_enabled = false):
        grid_dim(grid_dim), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim), cooperative(cooperative), pdl_enabled(pdl_enabled) {}
};

template <typename Derived>
class LaunchRuntime {
public:
    template <typename Args>
    static std::string generate(const Args& args) {
        auto code = Derived::generate_impl(args);

        // NOTES: we require that `generate_impl`'s includes never change
        static std::string include_hash;
        if (include_hash.empty())
            include_hash = include_parser->get_hash_value(code);

        // TODO: optimize string concat performance
        code = fmt::format("// Includes' hash value: {}\n{}", include_hash, code);
        if (get_env<int>("EP_JIT_DEBUG", 0))
            printf("Generated kernel code:\n%s\n", code.c_str());
        return code;
    }

    template <typename Args>
    static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args,
                       const std::optional<at::cuda::CUDAStream>& stream_opt = std::nullopt) {
        const auto kernel = kernel_runtime->kernel;
        const auto stream = stream_opt.value_or(at::cuda::getCurrentCUDAStream());
        const LaunchArgs& launch_args = args.launch_args;

        const dim3& grid_dim = {static_cast<unsigned>(launch_args.grid_dim.first),
                                static_cast<unsigned>(launch_args.grid_dim.second),
                                1};
        const dim3& block_dim = {static_cast<unsigned>(launch_args.num_threads), 1, 1};
        auto cluster_dim = launch_args.cluster_dim;
        auto pdl_enabled = launch_args.pdl_enabled;
#ifdef DISABLE_SM90_FEATURES
        cluster_dim = 1;
        pdl_enabled = false;
#endif
        auto config = construct_launch_config(kernel, stream, launch_args.smem_size,
                                              grid_dim, block_dim, cluster_dim,
                                              launch_args.cooperative, pdl_enabled);

        // Launch in the derived class
        if (get_env<int>("EP_JIT_DEBUG")) {
            printf("Launch kernel with {%d, %d} x %d (cooperative: %d), shared memory: %d bytes, cluster: %d, stream: %ld\n",
                    launch_args.grid_dim.first, launch_args.grid_dim.second, launch_args.num_threads,
                    launch_args.cooperative,
                    launch_args.smem_size, cluster_dim, stream.id());
        }
        Derived::launch_impl(kernel, config, args);
    }
};

} // namespace deep_ep::jit

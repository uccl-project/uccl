#pragma once

#include "../utils/format.hpp"
#include "../utils/system.hpp"
#include "compiler.hpp"
#include "include_parser.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::jit {

struct LaunchArgs {
  std::pair<int, int> grid_dim;
  int num_threads;
  int smem_size;
  int cluster_dim;
  bool cooperative;
  bool pdl_enabled;

  LaunchArgs(int const& grid_dim_x, int const& num_threads,
             int const& smem_size = 0, int const& cluster_dim = 1,
             bool const& cooperative = false, bool const& pdl_enabled = false)
      : grid_dim({grid_dim_x, 1}),
        num_threads(num_threads),
        smem_size(smem_size),
        cluster_dim(cluster_dim),
        cooperative(cooperative),
        pdl_enabled(pdl_enabled) {}

  LaunchArgs(std::pair<int, int> const& grid_dim, int const& num_threads,
             int const& smem_size = 0, int const& cluster_dim = 1,
             bool const& cooperative = false, bool const& pdl_enabled = false)
      : grid_dim(grid_dim),
        num_threads(num_threads),
        smem_size(smem_size),
        cluster_dim(cluster_dim),
        cooperative(cooperative),
        pdl_enabled(pdl_enabled) {}
};

template <typename Derived>
class LaunchRuntime {
 public:
  template <typename Args>
  static std::string generate(Args const& args) {
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
  static void launch(
      std::shared_ptr<KernelRuntime> const& kernel_runtime, Args const& args,
      std::optional<at::cuda::CUDAStream> const& stream_opt = std::nullopt) {
    auto const kernel = kernel_runtime->kernel;
    auto const stream = stream_opt.value_or(at::cuda::getCurrentCUDAStream());
    LaunchArgs const& launch_args = args.launch_args;

    dim3 const& grid_dim = {static_cast<unsigned>(launch_args.grid_dim.first),
                            static_cast<unsigned>(launch_args.grid_dim.second),
                            1};
    dim3 const& block_dim = {static_cast<unsigned>(launch_args.num_threads), 1,
                             1};
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
      printf(
          "Launch kernel with {%d, %d} x %d (cooperative: %d), shared memory: "
          "%d bytes, cluster: %d, stream: %ld\n",
          launch_args.grid_dim.first, launch_args.grid_dim.second,
          launch_args.num_threads, launch_args.cooperative,
          launch_args.smem_size, cluster_dim, stream.id());
    }
    Derived::launch_impl(kernel, config, args);
  }
};

}  // namespace deep_ep::jit

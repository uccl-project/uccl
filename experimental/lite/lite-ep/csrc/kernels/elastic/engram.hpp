#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>
#include <nccl.h>
#include <nccl_device.h>

namespace deep_ep::elastic {

class EngramFetchRuntime final : public jit::LaunchRuntime<EngramFetchRuntime> {
 public:
  struct Args {
    // Templated arguments
    int num_entries_per_rank;
    int hidden;
    int num_ranks;
    int num_qps;

    // Parameters
    ncclDevComm_t nccl_dev_comm;
    ncclWindow_t nccl_window;
    void* storage;
    void* fetched;
    int* indices;
    ncclGinRequest_t* last_gin_requests;
    int num_tokens;

    jit::LaunchArgs launch_args;
  };

  static std::string generate_impl(Args const& args) {
    return fmt::format(R"(
#include <deep_ep/impls/engram_fetch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&engram_fetch_impl<{}, {}, {}, {}, {}>);
}}
)",
                       args.num_qps, args.num_entries_per_rank, args.hidden,
                       args.num_ranks, args.launch_args.num_threads);
  }

  static void launch_impl(jit::KernelHandle const& kernel,
                          jit::LaunchConfigHandle const& config, Args args) {
    EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
        kernel, config, args.nccl_dev_comm, args.nccl_window, args.storage,
        args.fetched, args.indices, args.last_gin_requests, args.num_tokens));
  }
};

static void launch_engram_fetch(ncclDevComm_t const& nccl_dev_comm,
                                ncclWindow_t const& nccl_window, void* storage,
                                void* fetched, int* indices,
                                ncclGinRequest_t* last_gin_requests,
                                int const& num_entries_per_rank,
                                int const& hidden, int const& num_tokens,
                                int const& num_ranks, int const& num_qps,
                                at::cuda::CUDAStream const& stream) {
  constexpr int kNumEngramFetchThreads = 1024;

  // Generate, build and launch
  const EngramFetchRuntime::Args args = {
      .num_entries_per_rank = num_entries_per_rank,
      .hidden = hidden,
      .num_ranks = num_ranks,
      .num_qps = num_qps,
      .nccl_dev_comm = nccl_dev_comm,
      .nccl_window = nccl_window,
      .storage = storage,
      .fetched = fetched,
      .indices = indices,
      .last_gin_requests = last_gin_requests,
      .num_tokens = num_tokens,
      .launch_args = jit::LaunchArgs(num_qps, kNumEngramFetchThreads)};
  auto const code = EngramFetchRuntime::generate(args);
  auto const runtime = jit::compiler->build("engram_fetch", code);
  EngramFetchRuntime::launch(runtime, args, stream);
}

class EngramFetchWaitRuntime final
    : public jit::LaunchRuntime<EngramFetchWaitRuntime> {
 public:
  struct Args {
    // Templated arguments
    int num_ranks;

    ncclDevComm_t nccl_dev_comm;
    ncclWindow_t nccl_window;
    ncclGinRequest_t* last_gin_requests;

    jit::LaunchArgs launch_args;
  };

  static std::string generate_impl(Args const& args) {
    return fmt::format(R"(
#include <deep_ep/impls/engram_fetch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&engram_fetch_wait_impl<{}, {}>);
}}
)",
                       args.num_ranks, args.launch_args.num_threads);
  }

  static void launch_impl(jit::KernelHandle const& kernel,
                          jit::LaunchConfigHandle const& config, Args args) {
    EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config, args.nccl_dev_comm,
                                             args.nccl_window,
                                             args.last_gin_requests));
  }
};

static void launch_engram_fetch_wait(ncclGinRequest_t* last_gin_requests,
                                     ncclDevComm_t const& nccl_dev_comm,
                                     ncclWindow_t const& nccl_window,
                                     int const& num_ranks, int const& num_qps,
                                     at::cuda::CUDAStream const& stream) {
  constexpr int kNumEngramFetchWaitThreads = 1024;

  // Generate, build and launch
  const EngramFetchWaitRuntime::Args args = {
      .num_ranks = num_ranks,
      .nccl_dev_comm = nccl_dev_comm,
      .nccl_window = nccl_window,
      .last_gin_requests = last_gin_requests,
      .launch_args = jit::LaunchArgs(num_qps, kNumEngramFetchWaitThreads)};
  auto const code = EngramFetchWaitRuntime::generate(args);
  auto const runtime = jit::compiler->build("engram_fetch_wait", code);
  EngramFetchWaitRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic

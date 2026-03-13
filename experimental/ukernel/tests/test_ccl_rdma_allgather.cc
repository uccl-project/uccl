#include "executor.h"
#include "rdma_backend.h"
#include "test.h"
#include "transport.h"
#include "../src/compute/task.h"
#include "../src/compute/gpu_rt.h"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Communicator = UKernel::Transport::Communicator;
using CommunicatorConfig = UKernel::Transport::CommunicatorConfig;

constexpr int kWorldSize = 2;
constexpr int kServerRank = 0;
constexpr int kClientRank = 1;
constexpr int kGpu = 0;
constexpr size_t kElemsPerRank = 128;
constexpr size_t kTotalElems = kElemsPerRank * kWorldSize;

std::string get_arg(int argc, char** argv, char const* key, char const* def = "") {
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], key, std::strlen(key)) == 0) {
      char const* p = argv[i] + std::strlen(key);
      if (*p == '=') return std::string(p + 1);
      if (*p == '\0' && i + 1 < argc) return std::string(argv[i + 1]);
    }
  }
  return std::string(def);
}

void fill_rank_segment(std::vector<float>& out, int rank, float base) {
  size_t offset = static_cast<size_t>(rank) * kElemsPerRank;
  for (size_t i = 0; i < kElemsPerRank; ++i) {
    out[offset + i] = base + static_cast<float>(i);
  }
}

bool check_allgather(std::vector<float> const& out) {
  for (size_t i = 0; i < kElemsPerRank; ++i) {
    if (out[i] != 10.0f + static_cast<float>(i)) return false;
    if (out[kElemsPerRank + i] != 20.0f + static_cast<float>(i)) return false;
  }
  return true;
}

int run_role(int rank, int peer_rank) {
  auto cfg = std::make_shared<CommunicatorConfig>();
  cfg->backend = UKernel::Transport::TransportBackend::UCCL;
  auto comm = std::make_shared<Communicator>(kGpu, rank, kWorldSize, cfg);

  if (rank == kServerRank) {
    if (!comm->accept_from(peer_rank)) return 2;
  } else {
    if (!comm->connect_to(peer_rank)) return 2;
  }

  GPU_RT_CHECK(gpuSetDevice(kGpu));
  float* d_output = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d_output, sizeof(float) * kTotalElems));

  std::vector<float> h_output(kTotalElems, 0.0f);
  fill_rank_segment(h_output, rank, rank == 0 ? 10.0f : 20.0f);
  GPU_RT_CHECK(gpuMemcpy(d_output, h_output.data(), sizeof(float) * kTotalElems,
                         gpuMemcpyHostToDevice));

  UKernel::CCL::BufferBindings bindings{};
  bindings.final_output = d_output;
  bindings.registration_bytes = sizeof(float) * kTotalElems;

  UKernel::CCL::CommunicatorRdmaBackend rdma_backend(*comm, peer_rank, bindings);
  UKernel::CCL::ExecutorBackends backends{};
  backends.rdma = &rdma_backend;
  UKernel::CCL::Executor executor(backends);

  UKernel::CCL::CollectiveConfig gather_cfg{};
  gather_cfg.nranks = kWorldSize;
  gather_cfg.rank = rank;
  gather_cfg.channels = 1;
  gather_cfg.bytes_per_rank = sizeof(float) * kElemsPerRank;
  gather_cfg.chunk_bytes = sizeof(float) * kElemsPerRank;
  gather_cfg.requested_cpu_backend = UKernel::Compute::CpuBackendKind::Rdma;
  gather_cfg.device_caps.is_same_node = false;
  gather_cfg.device_caps.supports_rdma = true;

  auto handle = executor.submit_allgather(gather_cfg);
  executor.wait(handle);
  if (executor.status(handle) != UKernel::CCL::CollectiveOpStatus::Completed) {
    std::cerr << "[rank " << rank << "] allgather status failed\n";
    return 3;
  }
  executor.release(handle);

  GPU_RT_CHECK(gpuMemcpy(h_output.data(), d_output, sizeof(float) * kTotalElems,
                         gpuMemcpyDeviceToHost));
  GPU_RT_CHECK(gpuFree(d_output));

  if (!check_allgather(h_output)) {
    std::cerr << "[rank " << rank << "] allgather validation failed\n";
    return 4;
  }

  std::cout << "[rank " << rank << "] RDMA allgather OK\n";
  return 0;
}

}  // namespace

int test_ccl_rdma_allgather(int argc, char** argv) {
  std::string role = get_arg(argc, argv, "--role", "");
  if (role == "server") return run_role(kServerRank, kClientRank);
  if (role == "client") return run_role(kClientRank, kServerRank);

  std::cerr << "Usage:\n"
            << "  ccl-rdma-ag --role=server\n"
            << "  ccl-rdma-ag --role=client\n";
  return 1;
}

#include <cstring>
#include <optional>
#include <vector>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include <nvshmem.h>

namespace deep_ep::nvshmem {

nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

void* alloc(const size_t& size, const size_t& alignment) {
    return nvshmem_align(alignment, size);
}

void free(void* ptr) {
    nvshmem_free(ptr);
}

void barrier(const bool& with_cpu_sync,
             const std::optional<cudaStream_t>& stream_opt = std::nullopt) {
    // Wait all streams to finish on this GPU
    if (with_cpu_sync)
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    // NOTES: this only launches kernels at GPU
    if (stream_opt.has_value()) {
        nvshmemx_barrier_all_on_stream(stream_opt.value());
    } else {
        nvshmem_barrier_all();
    }

    // Let CPU wait
    if (with_cpu_sync)
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

int init(const std::vector<uint8_t>& root_unique_id_val,
         const int& rank,
         const int& num_ranks,
         const int& team_split_stride) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    if (team_split_stride > 0 and num_ranks > team_split_stride) {
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % team_split_stride == 0);
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                                  rank % team_split_stride,
                                                  team_split_stride,
                                                  num_ranks / team_split_stride,
                                                  &cpu_rdma_team_config,
                                                  0,
                                                  &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }

    // Wait all GPUs to get ready
    barrier(true);
    return nvshmem_my_pe();
}

void finalize() {
    barrier(true);
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}

}  // namespace deep_ep::nvshmem

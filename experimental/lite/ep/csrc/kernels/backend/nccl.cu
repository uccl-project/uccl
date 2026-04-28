#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>
#include <utility>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sstream>

#include <cuda.h>
#include <nccl.h>
#include <nccl_device/core.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "api.cuh"
#include "../../utils/lazy_driver.hpp"
#include "../../utils/system.hpp"


namespace deep_ep::nccl {

namespace {

size_t align_up(const size_t value, const size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

std::pair<void*, void*> alloc_host_window(const size_t size, const size_t alignment) {
    CUdevice device;
    CUDA_DRIVER_CHECK(lazy_cuCtxGetDevice(&device));
    int numa_node = 0;
    const auto numa_result = lazy_cuDeviceGetAttribute(&numa_node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device);
    if (numa_result != CUDA_SUCCESS or numa_node < 0)
        numa_node = 0;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = numa_node;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity = 0;
    CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    const auto alloc_size = align_up(size, std::max(alignment, granularity));

    CUmemGenericAllocationHandle handle;
    CUDA_DRIVER_CHECK(lazy_cuMemCreate(&handle, alloc_size, &prop, 0));

    void* ptr = nullptr;
    CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(&ptr), alloc_size, granularity, 0, 0));
    CUDA_DRIVER_CHECK(lazy_cuMemMap(reinterpret_cast<CUdeviceptr>(ptr), alloc_size, 0, handle, 0));

    CUmemAccessDesc access_desc[2] = {};
    access_desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc[0].location.id = device;
    access_desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    access_desc[1].location.id = numa_node;
    access_desc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(reinterpret_cast<CUdeviceptr>(ptr), alloc_size, access_desc, 2));
    CUDA_DRIVER_CHECK(lazy_cuMemRelease(handle));
    std::memset(ptr, 0, alloc_size);
    return {ptr, ptr};
}

void free_host_window(void* ptr) {
    if (ptr == nullptr)
        return;

    size_t size = 0;
    CUDA_DRIVER_CHECK(lazy_cuMemGetAddressRange_v2(nullptr, &size, reinterpret_cast<CUdeviceptr>(ptr)));
    CUDA_DRIVER_CHECK(lazy_cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size));
    CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size));
}

}  // namespace

pybind11::bytearray get_local_unique_id() {
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    std::vector<char> result(sizeof(ncclUniqueId));
    std::memcpy(result.data(), &unique_id, sizeof(ncclUniqueId));
    return {result.data(), result.size()};
}

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx) {
    // Copy unique ID
    ncclUniqueId root_unique_id;
    const auto root_unique_id_str = root_unique_id_bytes.cast<std::string>();
    std::memcpy(&root_unique_id, root_unique_id_str.c_str(), sizeof(ncclUniqueId));

    // Init
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("New NCCL host communicator created (%d/%d)\n", rank_idx, num_ranks);
    return reinterpret_cast<int64_t>(comm);
}

void destroy_nccl_comm(const int64_t& nccl_comm) {
    NCCL_CHECK(ncclCommAbort(reinterpret_cast<ncclComm_t>(nccl_comm)));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("NCCL host communicator aborted\n");
}

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm) {
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    const int num_ranks = ncclTeamWorld(comm).nRanks, num_nvl_ranks = ncclTeamLsa(comm).nRanks;
    if (get_env("EP_FORCE_NO_NVLINK", 0))
        return {num_ranks, 1};
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    return {num_ranks / num_nvl_ranks, num_nvl_ranks};
}

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
}

NCCLSymmetricMemoryContext::NCCLSymmetricMemoryContext(const int64_t& nccl_comm,
                                                       const int& num_ranks, const int& rank_idx,
                                                       const size_t& size, const size_t& alignment,
                                                       const bool& allow_hybrid_mode,
                                                       const int& sl_idx, const int& num_allocated_qps,
                                                       const size_t& host_bounce_size):
    use_host_window(get_env<int>("EP_FORCE_HOST_WINDOW", 0) != 0),
    rank_idx(rank_idx), num_ranks(num_ranks), num_allocated_qps(num_allocated_qps) {
    if (get_env("EP_BUFFER_DEBUG", 0)) {
        int nccl_version;
        NCCL_CHECK(ncclGetVersion(&nccl_version));
        printf("DeepEP initialized with NCCL version: %d.%d.%d (loaded library)\n",
               nccl_version / 10000, (nccl_version % 10000) / 100, nccl_version % 100);
    }

    // Reuse the NCCL communicator
    comm = reinterpret_cast<ncclComm_t>(nccl_comm);

    // Print number of allocated QPs
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("EP NCCL device communicator has %d allocated QPs\n", num_allocated_qps);

    // Query NCCL supported Gin Type
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    const auto gin_type = allow_hybrid_mode ? props.railedGinType : props.ginType;
    EP_HOST_ASSERT(
        gin_type != NCCL_GIN_TYPE_NONE and
        "NCCL GIN is unavailable. This is usually due to a network configuration issue, "
        "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network.");

    // Initialize NCCL device communicator
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {
        reqs.ginContextCount = num_allocated_qps;
        reqs.ginExclusiveContexts = true;
        if (gin_type != NCCL_GIN_TYPE_PROXY)
            reqs.ginQueueDepth = 1024;
        reqs.ginTrafficClass = sl_idx;
        // Customized RDMA barrier needs extra signals
        reqs.ginSignalCount = num_ranks + 2 * 2;
        reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL: NCCL_GIN_CONNECTION_FULL;
    }
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    // Now we know the NVLink domain size
    const auto actual_lsa_size = dev_comm.lsaSize;
    const auto actual_lsa_rank = dev_comm.lsaRank;
    if (get_env("EP_FORCE_NO_NVLINK", 0)) {
        num_nvl_ranks = 1, nvl_rank_idx = 0;
        num_rdma_ranks = num_ranks, rdma_rank_idx = rank_idx;
    } else {
        num_nvl_ranks = actual_lsa_size, nvl_rank_idx = actual_lsa_rank;
        num_rdma_ranks = num_ranks / num_nvl_ranks, rdma_rank_idx = rank_idx / num_nvl_ranks;
        EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0 and nvl_rank_idx == rank_idx % num_nvl_ranks);
        EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);
    }

    // Calculate scaleout/up domain size
    if (allow_hybrid_mode) {
        num_scaleout_ranks = num_rdma_ranks, num_scaleup_ranks = num_nvl_ranks;
        scaleout_rank_idx = rdma_rank_idx, scaleup_rank_idx = nvl_rank_idx;
    } else {
        num_scaleout_ranks = 1, num_scaleup_ranks = num_ranks;
        scaleout_rank_idx = 0, scaleup_rank_idx = rank_idx;
    }
    is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

    // Create main GPU window (always GPU memory)
    NCCL_CHECK(ncclMemAlloc(&raw_window_ptr, size));
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, size, &window, NCCL_WIN_DEFAULT));
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, actual_lsa_rank, &mapped_window_ptr));

    // Create host bounce window (for host-staging mode) — separate registration
    host_window = nullptr;
    host_window_raw_ptr = nullptr;
    host_window_mapped_ptr = nullptr;
    if (use_host_window and host_bounce_size > 0) {
        EP_HOST_ASSERT(get_env("EP_FORCE_NO_NVLINK", 0) != 0 and
                       "EP_FORCE_HOST_WINDOW currently requires EP_FORCE_NO_NVLINK=1");
        std::tie(host_window_raw_ptr, host_window_mapped_ptr) = alloc_host_window(host_bounce_size, alignment);
        // Note: skip ncclCommWindowRegister for now — it interferes with GIN barrier signals
        // Without registration, put_via_host falls back to regular gin.put from GPU
        // NCCL_CHECK(ncclCommWindowRegister(comm, host_window_raw_ptr, host_bounce_size, &host_window, NCCL_WIN_DEFAULT));
        if (get_env("EP_BUFFER_DEBUG", 0))
            printf("EP host bounce window allocated: %zu bytes at %p (no NCCL register)\n", host_bounce_size, host_window_mapped_ptr);
    }

    // Get LSA pointers for all LSA peers
    // TODO: check whether this is correct for network with RDMA
    nvl_window_ptrs.resize(num_nvl_ranks);
    if (get_env("EP_FORCE_NO_NVLINK", 0)) {
        nvl_window_ptrs[0] = mapped_window_ptr;
    } else {
        for (int i = 0; i < num_nvl_ranks; ++ i)
            NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));
    }

    // TODO: push NCCL team to support aligned allocation
    EP_HOST_ASSERT(size % alignment == 0);
    EP_HOST_ASSERT(reinterpret_cast<uint64_t>(raw_window_ptr) % alignment == 0);
    EP_HOST_ASSERT(reinterpret_cast<uint64_t>(mapped_window_ptr) % alignment == 0);
}

void* NCCLSymmetricMemoryContext::get_sym_ptr(void* ptr, const int& dst_rank_idx) const {
    const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(mapped_window_ptr);
    return static_cast<uint8_t*>(nvl_window_ptrs[dst_rank_idx]) + offset;
}

void NCCLSymmetricMemoryContext::finalize() const {
    // Deregister host bounce window
    if (host_window) {
        NCCL_CHECK(ncclCommWindowDeregister(comm, host_window));
        free_host_window(host_window_raw_ptr);
    }

    // Deregister main GPU window and free buffer
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    NCCL_CHECK(ncclMemFree(raw_window_ptr));

    // Destroy device communicator
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
}

}  // namespace deep_ep::nccl

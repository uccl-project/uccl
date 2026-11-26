// Exercise the standalone ring with independent signed inputs, uneven tails,
// delayed ranks and asynchronous generations. Reuse its exact implementation.
#define main uccl_lite_peer_benchmark_main
#include "../../src/collectives/peer_ag_rs.cu"
#undef main

__host__ __device__ float randomValue(size_t i, int rank, int generation) {
  uint32_t x = uint32_t(i) ^ (uint32_t(rank + 1) * 0x9e3779b9u) ^
               (uint32_t(generation + 1) * 0x85ebca6bu);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  return (int(x % 200001) - 100000) / 100003.0f;
}
__global__ void randomFill(float* input, size_t n, int rank, int generation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    input[i] = randomValue(i, rank, generation);
}
__global__ void delayRank() {
  auto begin = clock64();
  while (clock64() - begin < 1000000) {
  }
}
int main() {
  try {
    for (int r = 0; r < 4; ++r) {
      cudaCheck(cudaSetDevice(r));
      for (int p = 0; p < 4; ++p) {
        if (p == r) continue;
        auto status = cudaDeviceEnablePeerAccess(p, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled)
          cudaGetLastError();
        else
          cudaCheck(status);
      }
    }
    for (size_t bytes : {size_t(1056), size_t(262176), (8UL << 20) + 32}) {
      std::array<std::unique_ptr<Device>, 4> owned;
      std::array<Device*, 4> devices{};
      for (int r = 0; r < 4; ++r) {
        owned[r] = std::make_unique<Device>(r, bytes, bytes, true);
        devices[r] = owned[r].get();
        devices[r]->pushGridBlocks = 8;
      }
      MiniHost mini(bytes, 16);
      size_t shard = bytes / 16;
      for (auto backend : {Backend::MiniHost, Backend::PushGrid,
                           Backend::ScatterHierLL, Backend::ScatterGrid}) {
        bool gather =
            backend == Backend::MiniHost || backend == Backend::PushGrid;
        for (int generation = 0; generation < 12; ++generation) {
          for (auto* d : devices) {
            cudaCheck(cudaSetDevice(d->rank));
            randomFill<<<256, 256, 0, d->stream>>>(
                d->input, gather ? shard : shard * 4, d->rank, generation);
            if (d->rank == generation % 4) delayRank<<<1, 1, 0, d->stream>>>();
          }
          if (backend == Backend::MiniHost)
            mini.launch(devices, shard);
          else if (backend == Backend::PushGrid)
            launchPushSerial(devices, Operation::Gather, shard, 8);
          else if (backend == Backend::ScatterGrid)
            launchScatterFence(devices, shard, true);
          else
            launchScatterLL(devices, shard, true);
          // Host-bank credits are checked across four asynchronous generations.
          // LL/fence candidates retain the all-rank eager completion contract.
          if (backend == Backend::MiniHost && generation % 4 != 3) continue;
          synchronize(devices);
          size_t count = gather ? shard * 4 : shard;
          std::vector<float> output(count);
          for (auto* d : devices) {
            cudaCheck(cudaSetDevice(d->rank));
            cudaCheck(cudaMemcpy(output.data(), d->output,
                                 count * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < count; ++i) {
              double expected = 0;
              if (gather)
                expected = randomValue(i % shard, int(i / shard), generation);
              else
                for (int r = 0; r < 4; ++r)
                  expected +=
                      randomValue(size_t(d->rank) * shard + i, r, generation);
              if (!std::isfinite(output[i]) ||
                  std::abs(output[i] - expected) >
                      1e-6 + 1e-6 * std::abs(expected))
                throw std::runtime_error("independent small/medium mismatch");
            }
          }
        }
        std::printf("PASS bytes=%zu backend=%s random/delayed generations=12\n",
                    bytes, backendName(backend));
      }
    }
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

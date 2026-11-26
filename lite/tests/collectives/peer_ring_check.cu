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
    for (size_t bytes :
         {size_t(1008), size_t(262160), (8UL << 20) + 16, (64UL << 20) + 16}) {
      std::array<std::unique_ptr<Device>, 4> owned;
      std::array<Device*, 4> devices{};
      for (int r = 0; r < 4; ++r) {
        owned[r] = std::make_unique<Device>(r, bytes, 0, false, 1UL << 20);
        devices[r] = owned[r].get();
      }
      size_t shard = bytes / 16;
      for (int generation = 0; generation < 12; ++generation) {
        for (auto* d : devices) {
          cudaCheck(cudaSetDevice(d->rank));
          randomFill<<<256, 256, 0, d->stream>>>(d->input, bytes / 4, d->rank,
                                                 generation);
          if (d->rank == generation % 4) delayRank<<<1, 1, 0, d->stream>>>();
        }
        launchScatterRing(devices, shard);
        // Four consecutive changed-input generations without host completion.
        if (generation % 4 != 3) continue;
        synchronize(devices);
        std::vector<float> output(shard);
        for (auto* d : devices) {
          cudaCheck(cudaSetDevice(d->rank));
          cudaCheck(cudaMemcpy(output.data(), d->output, shard * sizeof(float),
                               cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < shard; ++i) {
            double expected = 0;
            for (int r = 0; r < 4; ++r)
              expected +=
                  randomValue(size_t(d->rank) * shard + i, r, generation);
            if (!std::isfinite(output[i]) ||
                std::abs(output[i] - expected) >
                    1e-6 + 1e-6 * std::abs(expected))
              throw std::runtime_error("independent random reduction mismatch");
          }
        }
      }
      std::printf(
          "PASS bytes=%zu signed-random delayed-ranks async-generations=12\n",
          bytes);
    }
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

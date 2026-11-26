// Diagnose completion semantics of the frozen standalone reference unchanged.
#define main uccl_lite_retained_reference_main
#include "peer_ag_rs.cu"
#undef main

__global__ void delayReferenceSource(unsigned long long cycles) {
  auto begin = clock64();
  while (clock64() - begin < cycles) __nanosleep(1024);
}
int main() {
  try {
    constexpr size_t bytes = 64UL << 10, shard = bytes / 16;
    std::array<std::unique_ptr<Device>, kRanks> owned;
    std::array<Device*, kRanks> devices{};
    for (int rank = 0; rank < kRanks; ++rank) {
      cudaCheck(cudaSetDevice(rank));
      for (int peer = 0; peer < kRanks; ++peer) {
        if (rank == peer) continue;
        auto status = cudaDeviceEnablePeerAccess(peer, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled)
          cudaGetLastError();
        else
          cudaCheck(status);
      }
      owned[rank] = std::make_unique<Device>(rank, 1UL << 30, 0, false);
      devices[rank] = owned[rank].get();
    }
    fill(devices, Operation::Gather, shard, 9);
    launchPushSerial(devices, Operation::Gather, shard);
    synchronize(devices);
    verify(devices, Operation::Gather, Backend::PushSerial, shard, 9);
    cudaCheck(cudaSetDevice(3));
    delayReferenceSource<<<1, 1, 0, devices[3]->stream>>>(1);
    cudaCheck(cudaStreamSynchronize(devices[3]->stream));
    constexpr int generation = 10;
    fill(devices, Operation::Gather, shard, generation);
    cudaDeviceProp property{};
    cudaCheck(cudaGetDeviceProperties(&property, 3));
    cudaCheck(cudaSetDevice(3));
    delayReferenceSource<<<1, 1, 0, devices[3]->stream>>>(
        static_cast<unsigned long long>(property.clockRate) * 500);
    launchPushSerial(devices, Operation::Gather, shard);
    int missing = 0;
    std::vector<float> output(bytes / 4);
    for (int rank = 0; rank < 3; ++rank) {
      cudaCheck(cudaSetDevice(rank));
      cudaCheck(cudaStreamSynchronize(devices[rank]->stream));
      cudaCheck(cudaMemcpy(output.data(), devices[rank]->output, bytes,
                           cudaMemcpyDeviceToHost));
      size_t wrong = 0;
      for (size_t i = 0; i < output.size(); ++i) {
        auto source = i / shard;
        float expected =
            float(source + 1 + generation * 4) * float(1 + (i % shard) % 97);
        if (output[i] != expected) ++wrong;
      }
      if (wrong) ++missing;
      std::printf("rank=%d stream_complete=1 wrong_elements=%zu\n", rank,
                  wrong);
    }
    cudaCheck(cudaSetDevice(3));
    auto pending = cudaStreamQuery(devices[3]->stream);
    if (pending != cudaErrorNotReady)
      throw std::runtime_error("delayed source finished before observation");
    cudaGetLastError();
    synchronize(devices);
    verify(devices, Operation::Gather, Backend::PushSerial, shard, generation);
    if (missing != 3)
      throw std::runtime_error(
          "completion gap was not reproduced on all early ranks");
    std::printf(
        "Original source unchanged: local stream completion did not imply "
        "output ready on %d ranks; global completion validated all outputs.\n",
        missing);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

#define UCCL_LITE_RESIDENT_NO_MAIN
#include "../../src/collectives/resident_collectives.cu"

int main() {
  using namespace uccl_lite;
  try {
    constexpr size_t capacity = 1048576;
    std::array<std::unique_ptr<Device>, 4> owned;
    std::array<Device*, 4> devices{};
    std::array<float*, 4> inputs{}, outputs{};
    for (int r = 0; r < 4; ++r) {
      cudaCheck(cudaSetDevice(r));
      for (int p = 0; p < 4; ++p)
        if (p != r) {
          auto status = cudaDeviceEnablePeerAccess(p, 0);
          if (status == cudaErrorPeerAccessAlreadyEnabled)
            cudaGetLastError();
          else
            cudaCheck(status);
        }
      owned[r] = std::make_unique<Device>(r, capacity, 0, false);
      devices[r] = owned[r].get();
      cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&inputs[r]), capacity,
                              cudaHostAllocPortable));
      cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&outputs[r]), capacity,
                              cudaHostAllocPortable));
      // Prepare both copy directions before a persistent grid is active.
      cudaCheck(cudaMemcpyAsync(devices[r]->input, inputs[r], capacity,
                                cudaMemcpyHostToDevice, devices[r]->stream));
      cudaCheck(cudaMemcpyAsync(outputs[r], devices[r]->input, capacity,
                                cudaMemcpyDeviceToHost, devices[r]->stream));
      cudaCheck(cudaStreamSynchronize(devices[r]->stream));
    }
    Resident resident(devices, capacity, true);
    auto check = [&](size_t bytes, int op, int generation) {
      size_t count = bytes / 4, shard = count / 4;
      for (int r = 0; r < 4; ++r) {
        size_t n = op == 0 ? shard : count;
        for (size_t i = 0; i < n; ++i)
          inputs[r][i] = residentValue(i, r, generation);
        cudaCheck(cudaSetDevice(r));
        cudaCheck(cudaMemcpyAsync(devices[r]->input, inputs[r],
                                  n * sizeof(float), cudaMemcpyHostToDevice,
                                  devices[r]->stream));
      }
      synchronize(devices);
      resident.launch(op, count);
      size_t n = op == 1 ? shard : count;
      for (int r = 0; r < 4; ++r) {
        cudaCheck(cudaSetDevice(r));
        cudaCheck(cudaMemcpyAsync(
            outputs[r], op == 2 ? devices[r]->input : devices[r]->output,
            n * sizeof(float), cudaMemcpyDeviceToHost, devices[r]->stream));
      }
      synchronize(devices);
      for (int r = 0; r < 4; ++r)
        for (size_t i = 0; i < n; ++i) {
          double expected = 0;
          if (op == 0)
            expected = inputs[i / shard][i % shard];
          else
            for (int source = 0; source < 4; ++source)
              expected += inputs[source][op == 1 ? size_t(r) * shard + i : i];
          double tolerance = op == 0 ? 0 : 1e-6 + 1e-6 * std::abs(expected);
          if (!std::isfinite(outputs[r][i]) ||
              std::abs(outputs[r][i] - expected) > tolerance)
            throw std::runtime_error(
                "resident independent host-reference mismatch");
        }
    };
    for (int window = 0; window < 3; ++window) {
      resident.start();
      for (size_t bytes :
           {size_t(16), size_t(32), size_t(1008), size_t(2048), size_t(2064),
            size_t(8192), size_t(8208), size_t(262176), capacity})
        for (int op = 0; op < 3; ++op)
          check(bytes, op, window * 100 + op + int(bytes % 97));
      resident.stop();
    }
    resident.epoch = UINT32_MAX - 2;
    resident.start();
    check(2048, 2, 501);
    check(8192, 1, 502);
    bool rejected = false;
    try {
      resident.launch(0, 256);
    } catch (std::runtime_error const&) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("epoch wrap was not rejected");
    resident.stop();
    for (int r = 0; r < 4; ++r) {
      cudaFreeHost(inputs[r]);
      cudaFreeHost(outputs[r]);
    }
    std::puts(
        "PASS resident: 83 mixed operations, packet/raw boundaries, scalar "
        "tails, "
        "start/stop reuse and epoch exhaustion; independent host reference");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

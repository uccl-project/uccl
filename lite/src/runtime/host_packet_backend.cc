// Local transport for the shared strict NCCL ABI. No NCCL implementation is
// linked; construction is CPU-only and prepare is an explicit first-use step.

#include "uccl_lite/multiprocess_collectives.h"
#include "uccl_lite/runtime/collective_backend.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cuda_runtime_api.h>

namespace uccl_lite::runtime {
class HostPacketCollectives : public CollectiveBackend {
 public:
  using Id = std::array<unsigned char, 128>;
  static Id createUniqueId() {
    Id id{};
    std::ifstream random("/dev/urandom", std::ios::binary);
    if (!random.read(reinterpret_cast<char*>(id.data()), id.size()))
      throw std::runtime_error("cannot create local collective unique ID");
    return id;
  }
  HostPacketCollectives(Id const& id, int rank, int device, int ranks,
                        size_t capacity)
      : rank_(rank), device_(device), ranks_(ranks), capacity_(capacity) {
    if (ranks < 2 || ranks > 32 || rank < 0 || rank >= ranks)
      throw std::invalid_argument("invalid host collective rank extent");
    // Use 256 random session bits, within the POSIX name length limit.
    static char const digits[] = "0123456789abcdef";
    name_ = "strict-";
    for (size_t i = 0; i < 32; ++i) {
      name_ += digits[id[i] >> 4];
      name_ += digits[id[i] & 15];
    }
  }
  ~HostPacketCollectives() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device_);
    if (packet_) uccl_lite_mp_destroy(packet_);
    if (input_) cudaFree(input_);
    if (output_) cudaFree(output_);
    if (done_) cudaEventDestroy(done_);
    if (previous >= 0 && previous != device_) cudaSetDevice(previous);
  }
  bool initialized() const override { return packet_ != nullptr; }
  bool idle() const override {
    return !recorded_ || cudaEventQuery(done_) == cudaSuccess;
  }
  void prepare() override {
    check(cudaDeviceSynchronize());
    check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    check(cudaMalloc(&input_, capacity_));
    check(cudaMalloc(&output_, capacity_));
    packet_ = uccl_lite_mp_create(name_.c_str(), ranks_, rank_, capacity_);
    if (!packet_) throw std::runtime_error(uccl_lite_mp_last_error());
  }
  void synchronize() override {
    if (recorded_) check(cudaEventSynchronize(done_));
  }
  void enqueue(NetworkOperation operation, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    int op = static_cast<int>(operation);
    if (!input || !output)
      throw std::invalid_argument("null collective buffer");
    if (recorded_ && stream != lastStream_)
      check(cudaStreamWaitEvent(stream, done_, 0));
    size_t inputBytes = count * 2 * (op == 2 ? ranks_ : 1);
    size_t outputBytes = count * 2 * (op == 1 ? ranks_ : 1);
    auto a = reinterpret_cast<uintptr_t>(input);
    auto b = reinterpret_cast<uintptr_t>(output);
    bool overlap = a < b + outputBytes && b < a + inputBytes;
    // Legal AG aliases its own output segment, whose writes preserve input.
    // RS instead stages only its result (one rank shard of the input), then
    // copies back after the kernel has finished every input read.
    bool legalGather = op == 1 && a == b + size_t(rank_) * inputBytes;
    if ((a & 15) || (op == 1 && overlap && !legalGather)) {
      check(cudaMemcpyAsync(input_, input, inputBytes, cudaMemcpyDeviceToDevice,
                            stream));
      input = input_;
    }
    void* target = ((b & 15) || (op == 2 && overlap)) ? output_ : output;
    if (uccl_lite_mp_enqueue_ordered(packet_, op, input, target, count, stream))
      throw std::runtime_error(uccl_lite_mp_last_error());
    if (target != output)
      check(cudaMemcpyAsync(output, target, outputBytes,
                            cudaMemcpyDeviceToDevice, stream));
    check(cudaEventRecord(done_, stream));
    lastStream_ = stream;
    recorded_ = true;
  }

 private:
  static void check(cudaError_t result) {
    if (result != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(result));
  }
  int rank_, device_, ranks_;
  size_t capacity_;
  std::string name_;
  void* packet_ = nullptr;
  void* input_ = nullptr;
  void* output_ = nullptr;
  cudaEvent_t done_ = nullptr;
  cudaStream_t lastStream_ = nullptr;
  bool recorded_ = false;
};
BackendRegistration hostPacketBackend() {
  return {{"host-packet",
           "UCCL_LITE_LOCAL8_STRICT_V1",
           {8},
           ScalarType::BFloat16,
           false,
           true,
           2,
           32},
          [] { return HostPacketCollectives::createUniqueId(); },
          [](BackendContext const& c) {
            return std::make_unique<HostPacketCollectives>(
                c.id, c.rank, c.device, c.ranks, c.capacity);
          }};
}
}  // namespace uccl_lite::runtime

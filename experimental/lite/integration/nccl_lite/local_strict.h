// Local transport for the shared strict NCCL ABI. No NCCL implementation is
// linked; construction is CPU-only and prepare is an explicit first-use step.
#pragma once

#include "uccl_lite/multiprocess_collectives.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cuda_runtime_api.h>

namespace uccl_lite::nccl_detail {
class LocalCollectives {
 public:
  using Id = std::array<unsigned char, 128>;
  static Id createUniqueId() {
    Id id{};
    std::ifstream random("/dev/urandom", std::ios::binary);
    if (!random.read(reinterpret_cast<char*>(id.data()), id.size()))
      throw std::runtime_error("cannot create local collective unique ID");
    return id;
  }
  LocalCollectives(Id const& id, int rank, int device, int, size_t capacity)
      : rank_(rank), device_(device), capacity_(capacity) {
    // Use 256 random session bits, within the POSIX name length limit.
    static char const digits[] = "0123456789abcdef";
    name_ = "strict-";
    for (size_t i = 0; i < 32; ++i) {
      name_ += digits[id[i] >> 4];
      name_ += digits[id[i] & 15];
    }
  }
  ~LocalCollectives() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device_);
    if (packet_) uccl_lite_mp_destroy(packet_);
    if (input_) cudaFree(input_);
    if (output_) cudaFree(output_);
    if (done_) cudaEventDestroy(done_);
    if (previous >= 0 && previous != device_) cudaSetDevice(previous);
  }
  bool initialized() const { return packet_ != nullptr; }
  bool idle() const {
    return !recorded_ || cudaEventQuery(done_) == cudaSuccess;
  }
  void prepare() {
    check(cudaDeviceSynchronize());
    check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    check(cudaMalloc(&input_, capacity_));
    check(cudaMalloc(&output_, capacity_));
    packet_ = uccl_lite_mp_create(name_.c_str(), 8, rank_, capacity_);
    if (!packet_) throw std::runtime_error(uccl_lite_mp_last_error());
  }
  void synchronize() {
    if (recorded_) check(cudaEventSynchronize(done_));
  }
  void enqueue(int op, void const* input, void* output, size_t count,
               cudaStream_t stream) {
    if (!input || !output)
      throw std::invalid_argument("null collective buffer");
    if (recorded_ && stream != lastStream_)
      check(cudaStreamWaitEvent(stream, done_, 0));
    size_t inputBytes = count * (op == 2 ? 16 : 2);
    size_t outputBytes = count * (op == 1 ? 16 : 2);
    auto a = reinterpret_cast<uintptr_t>(input);
    auto b = reinterpret_cast<uintptr_t>(output);
    bool overlap = a < b + outputBytes && b < a + inputBytes;
    // Legal AG aliases its own output segment, whose writes preserve input.
    // RS instead stages only its result (1/8 of the input), then copies back
    // after the kernel has finished every input read.
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
  int rank_, device_;
  size_t capacity_;
  std::string name_;
  void* packet_ = nullptr;
  void* input_ = nullptr;
  void* output_ = nullptr;
  cudaEvent_t done_ = nullptr;
  cudaStream_t lastStream_ = nullptr;
  bool recorded_ = false;
};
}  // namespace uccl_lite::nccl_detail

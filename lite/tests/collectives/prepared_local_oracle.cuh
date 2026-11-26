#pragma once
// Independent complete-buffer oracle shared by prepared-runtime consumers.
#include "uccl_lite/runtime/prepared_local.hpp"
#include <array>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace uccl_lite_test {
using namespace uccl_lite;
using namespace uccl_lite::runtime;
inline void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
inline unsigned short inputValue(size_t i, int rank, int generation, bool ag) {
  if (ag) return static_cast<unsigned short>(i * 313 + rank * 43 + generation);
  return __bfloat16_as_ushort(__float2bfloat16_rn(
      float(int((i * 5 + rank * 3) % 23) - 11 + generation * 2)));
}
inline void execute(PreparedLocalRuntime& runtime, size_t plan, int rank,
                    int ranks, std::array<cudaStream_t, 2> const& streams) {
  auto const& request = runtime.selection(plan).request;
  int op = int(request.operation);
  size_t count = request.count;
  auto layout = request.layouts[rank];
  size_t inputs = count * (op == 2 ? ranks : 1),
         outputs = count * (op == 1 ? ranks : 1);
  size_t extent = std::max(inputs, outputs) + 256;
  unsigned short *a, *b;
  check(cudaMalloc(&a, extent * 2));
  check(cudaMalloc(&b, extent * 2));
  size_t inputOffset =
      layout.alignment / 2 + (layout.inPlace && op == 1 ? rank * count : 0);
  size_t outputOffset =
      layout.alignment / 2 + (layout.inPlace && op == 2 ? rank * count : 0);
  bool inputB = layout.inPlace && op == 1, outputA = layout.inPlace && op != 1;
  for (int generation = 0; generation < 3; ++generation) {
    std::vector<unsigned short> ha(extent, 0x7f7f), hb(extent, 0x7f7f);
    auto& input = inputB ? hb : ha;
    for (size_t i = 0; i < inputs; ++i)
      input[inputOffset + i] = inputValue(i, rank, generation, op == 1);
    check(cudaMemcpy(a, ha.data(), extent * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, hb.data(), extent * 2, cudaMemcpyHostToDevice));
    auto& expected = outputA ? ha : hb;
    for (size_t i = 0; i < outputs; ++i) {
      if (op == 1)
        expected[outputOffset + i] =
            inputValue(i % count, i / count, generation, true);
      else {
        double result = 0;
        for (int source = 0; source < ranks; ++source)
          result += __bfloat162float(__ushort_as_bfloat16(
              inputValue(op == 2 ? size_t(rank) * count + i : i, source,
                         generation, false)));
        expected[outputOffset + i] =
            __bfloat16_as_ushort(__float2bfloat16_rn(float(result)));
      }
    }
    runtime.enqueue(plan, (inputB ? b : a) + inputOffset,
                    (outputA ? a : b) + outputOffset, streams[generation % 2]);
    runtime.synchronize();
    std::vector<unsigned short> actualA(extent), actualB(extent);
    check(cudaMemcpy(actualA.data(), a, extent * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actualB.data(), b, extent * 2, cudaMemcpyDeviceToHost));
    if (actualA != ha || actualB != hb)
      throw std::runtime_error(
          "prepared runtime independent full-buffer oracle failed");
  }
  cudaGraph_t graph;
  check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
  bool rejected = false;
  try {
    runtime.enqueue(plan, (inputB ? b : a) + inputOffset,
                    (outputA ? a : b) + outputOffset, streams[0]);
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  check(cudaStreamEndCapture(streams[0], &graph));
  check(cudaGraphDestroy(graph));
  if (!rejected) throw std::runtime_error("unqualified capture accepted");
  check(cudaFree(a));
  check(cudaFree(b));
}
}  // namespace uccl_lite_test

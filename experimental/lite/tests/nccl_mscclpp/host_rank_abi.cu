// A variable-rank BF16 client using only public NCCL/MPI/CUDA APIs.
#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void check(ncclResult_t e) {
  if (e != ncclSuccess) throw std::runtime_error(ncclGetErrorString(e));
}
unsigned short bits(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
float value(size_t i, int rank, int generation) {
  return float(int(i % 17) - 8) * (rank + 1) + generation;
}
ncclResult_t invoke(int op, void const* in, void* out, size_t count,
                    ncclComm_t comm, cudaStream_t stream) {
  if (op == 1) return ncclAllGather(in, out, count, ncclBfloat16, comm, stream);
  if (op == 2)
    return ncclReduceScatter(in, out, count, ncclBfloat16, ncclSum, comm,
                             stream);
  return ncclAllReduce(in, out, count, ncclBfloat16, ncclSum, comm, stream);
}
__global__ void queuedProducer(unsigned short* input, size_t count, int rank,
                               int generation, bool gather) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    // AG must preserve every BF16 bit pattern, including NaNs and signed zero.
    input[i] = gather ? static_cast<unsigned short>(i + rank * 191 + generation)
                      : __bfloat16_as_ushort(__float2bfloat16_rn(
                            float(int(i % 17) - 8) * (rank + 1) + generation));
  }
}
void queuedCheck(ncclComm_t comm, int rank, int ranks,
                 std::array<cudaStream_t, 2> const& streams, bool prepared) {
  constexpr int queued = 36;
  size_t stride = 65536UL * ranks + 256, total = queued * stride;
  unsigned short *input, *output;
  check(cudaMalloc(&input, total * 2));
  check(cudaMalloc(&output, total * 2));
  check(cudaMemset(input, 0x5a, total * 2));
  check(cudaMemset(output, 0x5a, total * 2));
  check(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == ranks - 1)
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  for (int call = 0; call < queued; ++call) {
    int op = call % 3 + 1;
    size_t count = (call / 3) % 2 ? 65536 : 64;
    size_t offset = call * stride + (prepared ? 128 : 17 + rank);
    auto stream = streams[call % 2];
    queuedProducer<<<32, 256, 0, stream>>>(
        input + offset, count * (op == 2 ? ranks : 1), rank, call, op == 1);
    check(cudaGetLastError());
    check(invoke(op, input + offset, output + offset, count, comm, stream));
  }
  for (auto stream : streams) check(cudaStreamSynchronize(stream));
  std::vector<unsigned short> actualInput(total), actualOutput(total),
      expectedInput(total, 0x5a5a), expectedOutput(total, 0x5a5a);
  check(
      cudaMemcpy(actualInput.data(), input, total * 2, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(actualOutput.data(), output, total * 2,
                   cudaMemcpyDeviceToHost));
  for (int call = 0; call < queued; ++call) {
    int op = call % 3 + 1;
    size_t count = (call / 3) % 2 ? 65536 : 64;
    size_t offset = call * stride + (prepared ? 128 : 17 + rank);
    for (size_t i = 0; i < count * (op == 2 ? ranks : 1); ++i)
      expectedInput[offset + i] =
          op == 1 ? static_cast<unsigned short>(i + rank * 191 + call)
                  : bits(value(i, rank, call));
    for (size_t i = 0; i < count * (op == 1 ? ranks : 1); ++i) {
      if (op == 1)
        expectedOutput[offset + i] =
            static_cast<unsigned short>(i % count + (i / count) * 191 + call);
      else {
        double sum = 0;
        for (int source = 0; source < ranks; ++source)
          sum += __bfloat162float(__ushort_as_bfloat16(bits(
              value(op == 2 ? size_t(rank) * count + i : i, source, call))));
        expectedOutput[offset + i] = bits(float(sum));
      }
    }
  }
  if (actualInput != expectedInput || actualOutput != expectedOutput)
    throw std::runtime_error("queued producer/consumer/staging reuse mismatch");
  check(cudaFree(input));
  check(cudaFree(output));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    bool prepared = argc > 1 && std::string(argv[1]) == "prepared";
    int devices = 0;
    check(cudaGetDeviceCount(&devices));
    if (ranks < 2 || ranks > devices)
      throw std::runtime_error("test requires distinct local CUDA devices");
    check(cudaSetDevice(rank));
    std::array<cudaStream_t, 2> streams;
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    size_t maxElements = 65536UL * ranks, extent = maxElements + 256;
    unsigned short *a, *b;
    check(cudaMalloc(&a, extent * 2));
    check(cudaMalloc(&b, extent * 2));
    int calls = 0;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      ncclUniqueId id;
      if (!rank) check(ncclGetUniqueId(&id));
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
      ncclComm_t comm;
      check(ncclCommInitRank(&comm, ranks, id, rank));
      int actual = 0;
      check(ncclCommCount(comm, &actual));
      if (actual != ranks)
        throw std::runtime_error("communicator reported a fixed rank count");
      cudaGraph_t graph;
      check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      if (invoke(3, a, b, 64, comm, streams[0]) != ncclInvalidUsage)
        throw std::runtime_error("capture accepted");
      check(cudaStreamEndCapture(streams[0], &graph));
      check(cudaGraphDestroy(graph));
      for (int generation = 0; generation < 2; ++generation)
        for (size_t count : {64UL, 256UL, 4096UL, 65536UL})
          for (int op = 1; op <= 3; ++op)
            for (bool inplace : {false, true}) {
              if (prepared && inplace != (count == 4096 && (rank & 1)))
                continue;
              size_t guard = generation ? size_t(17 + rank) : 32;
              if (prepared)
                guard = count == 256 ? size_t(1u << (rank % 4)) : 128;
              std::vector<unsigned short> ha(extent, 0x5a5a),
                  hb(extent, 0x5a5a);
              size_t inElements = count * (op == 2 ? ranks : 1),
                     outElements = count * (op == 1 ? ranks : 1);
              bool inputB = inplace && op == 1, outputA = inplace && op != 1;
              size_t inputOffset = guard + (inputB ? rank * count : 0),
                     outputOffset =
                         guard + (inplace && op == 2 ? rank * count : 0);
              auto& input = inputB ? hb : ha;
              for (size_t i = 0; i < inElements; ++i)
                input[inputOffset + i] = bits(value(i, rank, generation));
              check(
                  cudaMemcpy(a, ha.data(), extent * 2, cudaMemcpyHostToDevice));
              check(
                  cudaMemcpy(b, hb.data(), extent * 2, cudaMemcpyHostToDevice));
              auto& expected = outputA ? ha : hb;
              for (size_t i = 0; i < outElements; ++i) {
                float result = 0;
                if (op == 1)
                  result = value(i % count, int(i / count), generation);
                else
                  for (int source = 0; source < ranks; ++source)
                    result += __bfloat162float(__ushort_as_bfloat16(
                        bits(value(op == 2 ? size_t(rank) * count + i : i,
                                   source, generation))));
                expected[outputOffset + i] = bits(result);
              }
              auto stream = streams[calls % 2];
              check(invoke(op, (inputB ? b : a) + inputOffset,
                           (outputA ? a : b) + outputOffset, count, comm,
                           stream));
              check(cudaStreamSynchronize(stream));
              std::vector<unsigned short> actualA(extent), actualB(extent);
              check(cudaMemcpy(actualA.data(), a, extent * 2,
                               cudaMemcpyDeviceToHost));
              check(cudaMemcpy(actualB.data(), b, extent * 2,
                               cudaMemcpyDeviceToHost));
              if (actualA != ha || actualB != hb)
                throw std::runtime_error(
                    "BF16 full-buffer/alias/guard mismatch");
              ++calls;
            }
      queuedCheck(comm, rank, ranks, streams, prepared);
      calls += 36;
      check(ncclCommDestroy(comm));
    }
    for (auto stream : streams) check(cudaStreamDestroy(stream));
    check(cudaFree(a));
    check(cudaFree(b));
    if (!rank)
      std::printf(
          "PASS: BF16 strict runtime, %d ranks, %d calls/rank, two lifetimes, "
          "aliases, alignment, guards, queued producers, bitwise AG, "
          "stream changes and capture recovery\n",
          ranks, calls);
    MPI_Finalize();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "rank %d: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

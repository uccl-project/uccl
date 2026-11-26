// This client links only NCCL/MPI/CUDA. Exercise the interposer without its
// API.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(result));
}
float value(size_t index, int rank, int generation) {
  return float(int(index % 17) - 8) * (rank + 1) + generation;
}
void call(int op, float const* input, float* output, size_t count,
          ncclComm_t comm, cudaStream_t stream) {
  if (op == 1)
    check(ncclAllGather(input, output, count, ncclFloat, comm, stream));
  else if (op == 2)
    check(ncclReduceScatter(input, output, count, ncclFloat, ncclSum, comm,
                            stream));
  else
    check(
        ncclAllReduce(input, output, count, ncclFloat, ncclSum, comm, stream));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (world != 8)
      throw std::runtime_error("requires eight ranks on two hosts");
    check(cudaSetDevice(rank % 4));
    cudaStream_t streams[2];
    for (auto& s : streams)
      check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    size_t maxElements = (1 << 18) / 4;
    size_t guard = 33 + rank;
    float *a, *b;
    check(cudaMalloc(&a, (maxElements + 128) * sizeof(float)));
    check(cudaMalloc(&b, (maxElements + 128) * sizeof(float)));
    int calls = 0;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      ncclUniqueId id;
      if (!rank) check(ncclGetUniqueId(&id));
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
      ncclComm_t comm;
      check(ncclGroupStart());
      check(ncclCommInitRank(&comm, world, id, rank));
      check(ncclGroupEnd());
      // Saved empty graphs must not disable later eager routing.
      cudaGraph_t empty;
      cudaGraphExec_t emptyExec;
      check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      for (int op = 1; op <= 3; ++op) call(op, a, b, 0, comm, streams[0]);
      check(cudaStreamEndCapture(streams[0], &empty));
      check(cudaGraphInstantiate(&emptyExec, empty, nullptr, nullptr, 0));
      for (int generation = 0; generation < 3; ++generation) {
        // 384 B is deliberately outside the candidate profile: exercise
        // native calls between routed sizes and collective families.
        for (size_t bytes : {1024UL, 16384UL, 384UL, 262144UL}) {
          for (int op = 1; op <= 3; ++op) {
            size_t count = bytes / (op == 3 ? 4 : 32);
            size_t ni = count * (op == 2 ? 8 : 1);
            size_t no = count * (op == 1 ? 8 : 1);
            bool inplace = (rank + generation) % 2;
            size_t sourceOffset = inplace && op == 1 ? rank * count : 0;
            size_t targetOffset = inplace && op == 2 ? rank * count : 0;
            std::vector<float> input(maxElements + 128, -12345.f);
            std::vector<float> output(input.size(), -12345.f);
            for (size_t i = 0; i < ni; ++i)
              input[guard + sourceOffset + i] = value(i, rank, generation);
            auto stream = streams[calls % 2];
            check(cudaMemcpyAsync(a, input.data(), input.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream));
            if (!inplace)
              check(cudaMemcpyAsync(b, output.data(),
                                    output.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
            float* outputBase = inplace ? a : b;
            call(op, a + guard + sourceOffset,
                 outputBase + guard + targetOffset, count, comm, stream);
            check(cudaMemcpyAsync(output.data(), outputBase,
                                  output.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            for (size_t i = 0; i < output.size(); ++i) {
              float expected = inplace ? input[i] : -12345.f;
              if (i >= guard + targetOffset && i < guard + targetOffset + no) {
                size_t index = i - guard - targetOffset;
                if (op == 1)
                  expected = value(index % count, index / count, generation);
                else {
                  expected = 0;
                  for (int peer = 0; peer < 8; ++peer)
                    expected += value(op == 2 ? rank * count + index : index,
                                      peer, generation);
                }
              }
              if (output[i] != expected)
                throw std::runtime_error("full-buffer/guard mismatch");
            }
            ++calls;
          }
        }
      }
      check(cudaGraphLaunch(emptyExec, streams[1]));
      check(cudaStreamSynchronize(streams[1]));
      check(cudaGraphExecDestroy(emptyExec));
      check(cudaGraphDestroy(empty));
      // A grouped call after activation transfers ownership back to NCCL.
      check(ncclGroupStart());
      call(3, a + guard, b + guard, 256, comm, streams[0]);
      check(ncclGroupEnd());
      check(cudaStreamSynchronize(streams[0]));
      // Native graph capture/replay after the transition must remain valid.
      cudaGraph_t graph;
      cudaGraphExec_t executable;
      check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      call(3, a + guard, b + guard, 256, comm, streams[0]);
      check(cudaStreamEndCapture(streams[0], &graph));
      check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
      for (int i = 0; i < 3; ++i)
        check(cudaGraphLaunch(executable, streams[0]));
      check(cudaStreamSynchronize(streams[0]));
      check(cudaGraphExecDestroy(executable));
      check(cudaGraphDestroy(graph));
      check(ncclCommDestroy(comm));
    }
    check(cudaFree(a));
    check(cudaFree(b));
    for (auto s : streams) check(cudaStreamDestroy(s));
    if (!rank)
      std::printf(
          "ABI full-buffer checks passed: %d calls/rank, two lifetimes\n",
          calls);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

// Per-call timing/preparation protocol matching the recorded UCCL harnesses.
// This complements, and never replaces, unchanged nccl-tests validation.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#ifdef UCCL_LITE_REFERENCE_LOCAL
constexpr int kRanks = 4;
#else
constexpr int kRanks = 8;
#endif
extern "C" ncclResult_t uccl_lite_nccl_synchronize(ncclComm_t);
void checked(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void checked(ncclResult_t s) {
  if (s != ncclSuccess) throw std::runtime_error(ncclGetErrorString(s));
}
__global__ void prepareAg(float* input, size_t count, int rank,
                          int generation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    input[i] = float(rank + 1 + generation * kRanks) * float(1 + i % 97);
}
__global__ void verifyAg(float const* output, size_t count, size_t shard,
                         int generation, unsigned* wrong) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    float expected = float(i / shard + 1 + generation * kRanks) *
                     float(1 + (i % shard) % 97);
    if (output[i] != expected) atomicAdd(wrong, 1);
  }
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if ((argc != 7 && argc != 8) || world != kRanks)
      throw std::invalid_argument(
          "op(ag/ar/rs) layout(out/in) begin end warmups iterations; "
          "configured "
          "ranks required");
    std::string op = argv[1], layout = argv[2];
    bool ag = op == "ag", rs = op == "rs", inplace = layout == "in";
    if ((!ag && !rs && op != "ar") || (!inplace && layout != "out"))
      throw std::invalid_argument("operation/layout");
    size_t begin = std::stoull(argv[3]), end = std::stoull(argv[4]);
    int warmup = std::stoi(argv[5]), iterations = std::stoi(argv[6]);
    size_t allocation = argc == 8 ? std::stoull(argv[7]) : end;
    if (allocation < end || allocation > (size_t(1) << 30) ||
        (allocation & (allocation - 1)))
      throw std::invalid_argument("allocation extent");
    if (begin < 1024 || end > (size_t(1) << 30) || begin > end ||
        (begin & (begin - 1)) || (end & (end - 1)) || warmup < 0 ||
        iterations < 1)
      throw std::invalid_argument("extent/iteration range");
    checked(cudaSetDevice(rank % 4));
    cudaStream_t stream, upload;
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checked(cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking));
    float *a, *b;
    unsigned* wrong;
    checked(cudaMalloc(&a, allocation));
    checked(cudaMalloc(&b, allocation));
    checked(cudaMalloc(&wrong, sizeof(unsigned)));
#ifdef UCCL_LITE_REFERENCE_LOCAL
    cudaEvent_t eventStart = nullptr, eventEnd = nullptr;
    float* referenceScratch = nullptr;
    if (ag || rs) {
      checked(cudaEventCreate(&eventStart));
      checked(cudaEventCreate(&eventEnd));
      checked(cudaMalloc(&referenceScratch, allocation));
    }
#endif
    ncclUniqueId id;
    if (!rank) checked(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    checked(ncclCommInitRank(&comm, world, id, rank));
    if (!rank) {
#ifdef UCCL_LITE_REFERENCE_LOCAL
      std::printf(
          "# local timing=%s; reference AG/RS uses one process/four GPUs; "
          "candidate uses four processes\n",
          ag || rs ? "max-rank CUDA events including enqueue gaps"
                   : "max-rank host enqueue+completion");
#else
      std::printf(
          "# protocol=UCCL per-call preparation; completion includes internal "
          "retirement; fixed pointers; reference AG process layout differs\n");
#endif
      std::printf("# dtype=FP32; warmup=%d iterations=%d max_allocation=%zu\n",
                  warmup, iterations, allocation);
    }
    std::vector<float> hostInput, hostOutput;
    for (size_t bytes = begin; bytes <= end; bytes *= 2) {
      size_t count = bytes / 4, shard = count / kRanks;
      float* output = inplace ? a + (rs ? rank * shard : 0) : b;
      float* input = a + (inplace && ag ? rank * shard : 0);
      hostInput.resize(ag ? shard : count);
      hostOutput.resize(rs ? shard : count);
      double total = 0;
      for (int i = 0; i < warmup + iterations; ++i) {
        int generation = 10 + i;
#ifdef UCCL_LITE_REFERENCE_LOCAL
        if (ag || rs) {
          if (inplace) checked(cudaMemsetAsync(a, 0xff, bytes, stream));
          prepareAg<<<256, 256, 0, stream>>>(input, ag ? shard : count, rank,
                                             generation);
          if (!inplace) checked(cudaMemsetAsync(b, 0xff, bytes, stream));
          checked(cudaMemsetAsync(referenceScratch, 0xff, bytes, stream));
          checked(cudaGetLastError());
          checked(cudaDeviceSynchronize());
        } else
#endif
            if (ag) {
          if (inplace) {
            checked(cudaMemsetAsync(a, 0xff, bytes, stream));
            prepareAg<<<256, 256, 0, stream>>>(input, shard, rank, generation);
          } else {
            prepareAg<<<256, 256, 0, upload>>>(input, shard, rank, generation);
            checked(cudaMemsetAsync(output, 0xff, bytes, stream));
          }
          checked(cudaGetLastError());
          checked(cudaDeviceSynchronize());
        } else {
          for (size_t j = 0; j < count; ++j)
            hostInput[j] =
                float(rank + 1 + generation * kRanks) * float(1 + j % 97);
          checked(cudaMemcpyAsync(input, hostInput.data(), bytes,
                                  cudaMemcpyHostToDevice, stream));
          checked(cudaStreamSynchronize(stream));
        }
        MPI_Barrier(MPI_COMM_WORLD);
#ifdef UCCL_LITE_REFERENCE_LOCAL
        if (ag || rs) checked(cudaEventRecord(eventStart, stream));
#endif
        auto start = std::chrono::steady_clock::now();
        if (ag)
          checked(ncclAllGather(input, output, shard, ncclFloat, comm, stream));
        else if (rs)
          checked(ncclReduceScatter(input, output, shard, ncclFloat, ncclSum,
                                    comm, stream));
        else
          checked(ncclAllReduce(input, output, count, ncclFloat, ncclSum, comm,
                                stream));
#ifdef UCCL_LITE_REFERENCE_LOCAL
        // The original local AR harness times launch + caller-stream
        // completion.
        if (ag || rs) checked(cudaEventRecord(eventEnd, stream));
        checked(cudaStreamSynchronize(stream));
#else
        checked(uccl_lite_nccl_synchronize(comm));
#endif
        auto stop = std::chrono::steady_clock::now();
        double localUs = std::chrono::duration<double, std::micro>(stop - start)
                             .count(),
               maxUs = 0;
#ifdef UCCL_LITE_REFERENCE_LOCAL
        if (ag || rs) {
          float ms;
          checked(cudaEventElapsedTime(&ms, eventStart, eventEnd));
          localUs = double(ms) * 1000;
        }
#endif
        MPI_Allreduce(&localUs, &maxUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        double error = 0;
        if (ag) {
          checked(cudaMemsetAsync(wrong, 0, sizeof(unsigned), stream));
          verifyAg<<<256, 256, 0, stream>>>(output, count, shard, generation,
                                            wrong);
          checked(cudaGetLastError());
          unsigned n;
          checked(cudaMemcpyAsync(&n, wrong, sizeof(n), cudaMemcpyDeviceToHost,
                                  stream));
          checked(cudaStreamSynchronize(stream));
          error = n ? 1 : 0;
        } else {
          checked(cudaMemcpy(hostOutput.data(), output, hostOutput.size() * 4,
                             cudaMemcpyDeviceToHost));
          double factor =
              kRanks * (kRanks + 1) / 2.0 + generation * kRanks * kRanks;
          for (size_t j = 0; j < hostOutput.size(); ++j) {
            double expected = factor * (1 + (j + (rs ? rank * shard : 0)) % 97);
            double e = std::fabs(double(hostOutput[j]) - expected) / expected;
            error = std::max(error, std::isfinite(e) ? e : INFINITY);
          }
        }
        double maxError = 0;
        MPI_Allreduce(&error, &maxError, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        if (maxError > 1e-5)
          throw std::runtime_error("full-output validation failed");
        if (i >= warmup) {
          total += maxUs;
          if (!rank)
            std::printf("sample,%s,%s,%zu,%d,%.6f,%.3e\n", op.c_str(),
                        layout.c_str(), bytes, i - warmup, maxUs, maxError);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      if (!rank) {
        std::printf("result,%s,%s,%zu,%.6f\n", op.c_str(), layout.c_str(),
                    bytes, total / iterations);
        std::fflush(stdout);
      }
      if (bytes > end / 2) break;
    }
    checked(ncclCommDestroy(comm));
    checked(cudaFree(a));
    checked(cudaFree(b));
    checked(cudaFree(wrong));
#ifdef UCCL_LITE_REFERENCE_LOCAL
    if (referenceScratch) checked(cudaFree(referenceScratch));
    if (eventStart) checked(cudaEventDestroy(eventStart));
    if (eventEnd) checked(cudaEventDestroy(eventEnd));
#endif
    checked(cudaStreamDestroy(stream));
    checked(cudaStreamDestroy(upload));
    if (!rank) std::puts("UCCL reference-protocol checks passed");
  } catch (std::exception const& e) {
    std::fprintf(stderr, "rank %d: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

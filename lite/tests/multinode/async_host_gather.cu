#include "../../src/multinode/async_host_gather.cuh"
#include <cstdio>
#include <vector>
#include <mpi.h>
void checked(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (world != 8) throw std::runtime_error("requires eight ranks");
    checked(cudaSetDevice(rank % 4));
    constexpr size_t maximum = ((128UL << 20) + 128) / 4;
    float *a, *b;
    checked(cudaMalloc(&a, (maximum + 128) * 4));
    checked(cudaMalloc(&b, (maximum + 128) * 4));
    cudaStream_t streams[2];
    for (auto& s : streams)
      checked(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    int calls = 0;
    for (int life = 0; life < 2; ++life) {
      auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, 8);
      mscclpp::UniqueId id{};
      if (!rank) id = mscclpp::TcpBootstrap::createUniqueId();
      MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      bootstrap->initialize(id, 120);
      auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
      auto* gather = new uccl_lite::multinode::AsyncHostGather(
          *comm, rank, argc > 1 ? std::stoull(argv[1]) : 262144);
#ifdef UCCL_LITE_HOST_SHARED_EVENT
      cudaEvent_t borrowed;
      checked(cudaEventCreateWithFlags(&borrowed, cudaEventDisableTiming));
#endif
      for (int generation = 0; generation < 3; ++generation) {
        for (size_t bytes :
             {1024UL, 8UL << 20, 128UL, (6UL << 20) + 128, 32UL << 20,
              (128UL << 20) + 128, 4096UL, 512UL << 10, (4UL << 20) - 128,
              (16UL << 20) - 128}) {
          size_t count = bytes / 32, guard = 33 + rank;
          bool inplace = (generation + rank) & 1;
          std::vector<float> input(maximum + 128, -12345.f),
              output(maximum + 128, -12345.f);
          size_t sourceOffset = inplace ? rank * count : 0;
          for (size_t i = 0; i < count; ++i)
            input[guard + sourceOffset + i] =
                float(int(i % 17) - 8) * (rank + 1) + generation;
          float* src = generation & 1 ? b : a;
          float* dst = inplace ? src : (generation & 1 ? a : b);
          auto stream = streams[calls % 2];
          checked(cudaMemcpyAsync(src, input.data(), input.size() * 4,
                                  cudaMemcpyHostToDevice, stream));
          if (!inplace)
            checked(cudaMemcpyAsync(dst, output.data(), output.size() * 4,
                                    cudaMemcpyHostToDevice, stream));
          if (rank == 3 && generation == 1) usleep(10000);
          if (!rank)
            std::fprintf(stderr, "async AG life=%d generation=%d bytes=%zu\n",
                         life, generation, bytes);
          gather->enqueue(src + guard + sourceOffset, dst + guard, count * 4,
                          stream
#ifdef UCCL_LITE_HOST_SHARED_EVENT
                          ,
                          (calls % 2) ? borrowed : nullptr
#endif
          );
          checked(cudaMemcpyAsync(output.data(), dst, output.size() * 4,
                                  cudaMemcpyDeviceToHost, stream));
#ifdef UCCL_LITE_HOST_SHARED_EVENT
          // The owner may advance its borrowed event after the gather. A
          // later gather must preserve that ordered boundary and ownership.
          if (calls % 2) checked(cudaEventRecord(borrowed, stream));
#endif
          checked(cudaStreamSynchronize(stream));
#ifdef UCCL_LITE_HOST_SHARED_EVENT
          if (calls % 10 == 0) gather->synchronize(true);
#endif
          for (size_t i = 0; i < output.size(); ++i) {
            float expected = inplace ? input[i] : -12345.f;
            if (i >= guard && i < guard + 8 * count) {
              size_t j = i - guard;
              expected = float(int((j % count) % 17) - 8) * (j / count + 1) +
                         generation;
            }
            if (output[i] != expected)
              throw std::runtime_error("async AG output/guard mismatch");
          }
          ++calls;
        }
        checked(cudaFree(a));
        checked(cudaMalloc(&a, (maximum + 128) * 4));
      }
      gather->synchronize();
      delete gather;
#ifdef UCCL_LITE_HOST_SHARED_EVENT
      checked(cudaEventDestroy(borrowed));
#endif
      bootstrap->barrier();
    }
    checked(cudaFree(a));
    checked(cudaFree(b));
    for (auto s : streams) checked(cudaStreamDestroy(s));
    if (!rank)
      std::printf("Async host gather passed: %d calls/rank, two lifetimes\n",
                  calls);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "rank %d: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}

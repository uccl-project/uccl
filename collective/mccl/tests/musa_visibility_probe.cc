// Hardware gate: production v11 callbacks followed by an immediate GPU
// consumer.
#include "mccl_net_abi.h"
#include "musa_visibility_kernels.h"
#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <dlfcn.h>
#include <mpi.h>

namespace {
[[noreturn]] void fail(char const* message) {
  std::fprintf(stderr, "visibility probe: %s\n", message);
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort();
}

#define VERIFY(call, success)             \
  do {                                    \
    if ((call) != (success)) fail(#call); \
  } while (0)

// Library callbacks may be nonblocking; a timeout aborts, never recycles a WR.
template <typename Poll>
void await(Poll poll) {
  double const deadline = MPI_Wtime() + 120;
  while (!poll()) {
    if (MPI_Wtime() > deadline) fail("operation timeout; aborting all ranks");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void test_to_completion(mcclNet_v11_t const* net, void* request) {
  await([&] {
    int done = 0;
    VERIFY(net->test(request, &done, nullptr), mcclSuccess);
    return done != 0;
  });
}
}  // namespace

int main(int argc, char** argv) {
  VERIFY(MPI_Init(&argc, &argv), MPI_SUCCESS);
  int rank, ranks, local_rank;
  VERIFY(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
  VERIFY(MPI_Comm_size(MPI_COMM_WORLD, &ranks), MPI_SUCCESS);
  if (argc != 4 || ranks != 2)
    fail("usage: mpirun -np 2 musa_visibility_probe PLUGIN.so BYTES ROUNDS");
  size_t bytes;
  unsigned rounds;
  try {
    size_t end = 0;
    auto b = std::stoull(argv[2], &end);
    if (end != std::string(argv[2]).size() || b == 0 || b > INT_MAX)
      fail("invalid byte count");
    bytes = size_t(b);
    auto r = std::stoull(argv[3], &end);
    if (end != std::string(argv[3]).size() || r == 0 || r > UINT_MAX)
      fail("invalid round count");
    rounds = unsigned(r);
  } catch (std::exception const&) {
    fail("invalid numeric argument");
  }
  MPI_Comm local;
  VERIFY(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                             MPI_INFO_NULL, &local),
         MPI_SUCCESS);
  VERIFY(MPI_Comm_rank(local, &local_rank), MPI_SUCCESS);
  int gpu_count;
  VERIFY(musaGetDeviceCount(&gpu_count), musaSuccess);
  if (local_rank >= gpu_count) fail("one visible GPU per local rank required");
  VERIFY(musaSetDevice(local_rank), musaSuccess);
  char host[MPI_MAX_PROCESSOR_NAME];
  int host_length;
  VERIFY(MPI_Get_processor_name(host, &host_length), MPI_SUCCESS);
  std::printf("visibility rank=%d host=%s visible_gpu=%d bytes=%zu rounds=%u\n",
              rank, host, local_rank, bytes, rounds);
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) fail(dlerror());
  auto const* net =
      static_cast<mcclNet_v11_t const*>(dlsym(library, "mcclNetPlugin_v11"));
  if (!net || !net->init || !net->devices || !net->listen || !net->connect ||
      !net->accept || !net->regMr || !net->deregMr || !net->isend ||
      !net->irecv || !net->iflush || !net->test || !net->closeSend ||
      !net->closeRecv || !net->closeListen || !net->finalize)
    fail("incomplete v11 plugin");
  void* ctx = nullptr;
  mcclNetCommConfig_v11_t config{};
  VERIFY(net->init(&ctx, 1, &config, nullptr, nullptr), mcclSuccess);
  int devices;
  VERIFY(net->devices(&devices), mcclSuccess);
  if (!devices) fail("no network devices");
  alignas(std::max_align_t) std::array<uint8_t, MCCL_NET_HANDLE_MAXSIZE>
      handle{};
  void* listener = nullptr;
  if (rank == 0)
    VERIFY(net->listen(ctx, 0, handle.data(), &listener), mcclSuccess);
  VERIFY(MPI_Bcast(handle.data(), handle.size(), MPI_BYTE, 0, MPI_COMM_WORLD),
         MPI_SUCCESS);
  void* comm = nullptr;
  await([&] {
    if (rank == 0)
      VERIFY(net->accept(listener, &comm, nullptr), mcclSuccess);
    else
      VERIFY(net->connect(ctx, 0, handle.data(), &comm, nullptr), mcclSuccess);
    return comm != nullptr;
  });
  void* allocation = nullptr;
  unsigned* errors = nullptr;
  musaStream_t stream;
  VERIFY(musaStreamCreate(&stream), musaSuccess);
  VERIFY(musaMalloc(&allocation, bytes + 2 * kVisibilityGuard), musaSuccess);
  VERIFY(musaMalloc(reinterpret_cast<void**>(&errors), sizeof(*errors)),
         musaSuccess);
  auto* payload = static_cast<uint8_t*>(allocation) + kVisibilityGuard;
  void* mr = nullptr;
  // Register the aligned allocation, not the deliberately offset payload.
  VERIFY(net->regMr(comm, allocation, bytes + 2 * kVisibilityGuard,
                    MCCL_PTR_MUSA, &mr),
         mcclSuccess);
  for (unsigned iteration = 0; iteration < rounds; ++iteration) {
    unsigned const epoch = iteration + 1;
    if (rank == 1) {
      VERIFY(fill_payload(payload, bytes, epoch, stream), musaSuccess);
    } else {
      VERIFY(musaMemsetAsync(allocation, 0xa5, bytes + 2 * kVisibilityGuard,
                             stream),
             musaSuccess);
    }
    // Before transfer only: finish producer/reset, not a post-receive flush.
    VERIFY(musaStreamSynchronize(stream), musaSuccess);
    void* request = nullptr;
    await([&] {
      if (rank == 1)
        VERIFY(net->isend(comm, payload, bytes, 0, mr, nullptr, &request),
               mcclSuccess);
      else {
        void* data[] = {payload};
        size_t sizes[] = {bytes};
        void* handles[] = {mr};
        int tags[] = {0};
        VERIFY(
            net->irecv(comm, 1, data, sizes, tags, handles, nullptr, &request),
            mcclSuccess);
      }
      return request != nullptr;
    });
    test_to_completion(net, request);
    if (rank == 0) {
      request = nullptr;
      void* data[] = {payload};
      int sizes[] = {int(bytes)};
      void* handles[] = {mr};
      await([&] {
        VERIFY(net->iflush(comm, 1, data, sizes, handles, &request),
               mcclSuccess);
        return request != nullptr;
      });
      test_to_completion(net, request);
      // Do not insert device sync or a payload D2H copy before this consumer.
      VERIFY(musaMemsetAsync(errors, 0, sizeof(*errors), stream), musaSuccess);
      VERIFY(check_payload(static_cast<uint8_t*>(allocation), bytes, epoch,
                           errors, stream),
             musaSuccess);
      unsigned host_errors = 0;
      VERIFY(musaMemcpyAsync(&host_errors, errors, sizeof(host_errors),
                             musaMemcpyDeviceToHost, stream),
             musaSuccess);
      VERIFY(musaStreamSynchronize(stream), musaSuccess);
      if (host_errors) fail("GPU payload/canary mismatch");
    }
    // Consumption ACK prevents the next epoch overwriting the same MR early.
    VERIFY(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
  }
  VERIFY(net->deregMr(comm, mr), mcclSuccess);
  if (rank == 0) {
    VERIFY(net->closeRecv(comm), mcclSuccess);
    VERIFY(net->closeListen(listener), mcclSuccess);
  } else
    VERIFY(net->closeSend(comm), mcclSuccess);
  VERIFY(net->finalize(ctx), mcclSuccess);
  VERIFY(musaFree(allocation), musaSuccess);
  VERIFY(musaFree(errors), musaSuccess);
  VERIFY(musaStreamDestroy(stream), musaSuccess);
  VERIFY(MPI_Comm_free(&local), MPI_SUCCESS);
  std::printf("GPU visibility probe rank=%d: PASS\n", rank);
  dlclose(library);
  VERIFY(MPI_Finalize(), MPI_SUCCESS);
}

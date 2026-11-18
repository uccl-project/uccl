// Capability gate only: public MCCL P2P, MUSA events and nonempty bidirectional
// data.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mccl.h>
#include <mpi.h>
#include <musa_runtime.h>

#define PROBE_CHECK(call, success)                                          \
  do {                                                                      \
    auto const result = (call);                                             \
    if (result != (success)) {                                              \
      std::fprintf(stderr, "%s failed (%d) at %s:%d\n", #call, int(result), \
                   __FILE__, __LINE__);                                     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                         \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

int main(int argc, char** argv) {
  PROBE_CHECK(MPI_Init(&argc, &argv), MPI_SUCCESS);
  int rank, ranks, local_rank;
  PROBE_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
  PROBE_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks), MPI_SUCCESS);
  if (ranks != 2) {
    std::fprintf(stderr, "P2P probe requires exactly two ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm local;
  PROBE_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &local),
              MPI_SUCCESS);
  PROBE_CHECK(MPI_Comm_rank(local, &local_rank), MPI_SUCCESS);
  int devices = 0;
  PROBE_CHECK(musaGetDeviceCount(&devices), musaSuccess);
  if (local_rank >= devices) {
    std::fprintf(stderr, "one visible GPU per local rank is required\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  PROBE_CHECK(musaSetDevice(local_rank), musaSuccess);
  char host[MPI_MAX_PROCESSOR_NAME];
  int length;
  PROBE_CHECK(MPI_Get_processor_name(host, &length), MPI_SUCCESS);
  std::printf("probe rank=%d host=%s visible_gpu=%d\n", rank, host, local_rank);
  mcclUniqueId id;
  if (rank == 0) PROBE_CHECK(mcclGetUniqueId(&id), mcclSuccess);
  PROBE_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD),
              MPI_SUCCESS);
  mcclComm_t comm;
  PROBE_CHECK(mcclCommInitRank(&comm, ranks, id, rank), mcclSuccess);
  musaStream_t producer, network, consumer;
  musaEvent_t ready, received;
  PROBE_CHECK(musaStreamCreate(&producer), musaSuccess);
  PROBE_CHECK(musaStreamCreate(&network), musaSuccess);
  PROBE_CHECK(musaStreamCreate(&consumer), musaSuccess);
  PROBE_CHECK(musaEventCreate(&ready), musaSuccess);
  PROBE_CHECK(musaEventCreate(&received), musaSuccess);
  size_t const send_count = rank == 0 ? 17 : 31;
  size_t const recv_count = rank == 0 ? 31 : 17;
  std::vector<float> input(send_count), output(recv_count, -1);
  for (size_t i = 0; i < send_count; ++i) input[i] = rank * 1000 + i;
  void *send_buffer, *recv_buffer;
  PROBE_CHECK(musaMalloc(&send_buffer, send_count * sizeof(float)),
              musaSuccess);
  PROBE_CHECK(musaMalloc(&recv_buffer, recv_count * sizeof(float)),
              musaSuccess);
  PROBE_CHECK(
      musaMemcpyAsync(send_buffer, input.data(), send_count * sizeof(float),
                      musaMemcpyHostToDevice, producer),
      musaSuccess);
  PROBE_CHECK(musaEventRecord(ready, producer), musaSuccess);
  PROBE_CHECK(musaStreamWaitEvent(network, ready, 0), musaSuccess);
  PROBE_CHECK(mcclGroupStart(), mcclSuccess);
  PROBE_CHECK(
      mcclRecv(recv_buffer, recv_count, mcclFloat, 1 - rank, comm, network),
      mcclSuccess);
  PROBE_CHECK(
      mcclSend(send_buffer, send_count, mcclFloat, 1 - rank, comm, network),
      mcclSuccess);
  PROBE_CHECK(mcclGroupEnd(), mcclSuccess);
  PROBE_CHECK(musaEventRecord(received, network), musaSuccess);
  PROBE_CHECK(musaStreamWaitEvent(consumer, received, 0), musaSuccess);
  PROBE_CHECK(
      musaMemcpyAsync(output.data(), recv_buffer, recv_count * sizeof(float),
                      musaMemcpyDeviceToHost, consumer),
      musaSuccess);
  PROBE_CHECK(musaStreamSynchronize(consumer), musaSuccess);
  for (size_t i = 0; i < recv_count; ++i)
    if (output[i] != float((1 - rank) * 1000 + i)) {
      std::fprintf(stderr, "P2P data mismatch rank=%d index=%zu\n", rank, i);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  PROBE_CHECK(mcclCommDestroy(comm), mcclSuccess);
  PROBE_CHECK(musaFree(send_buffer), musaSuccess);
  PROBE_CHECK(musaFree(recv_buffer), musaSuccess);
  PROBE_CHECK(musaEventDestroy(ready), musaSuccess);
  PROBE_CHECK(musaEventDestroy(received), musaSuccess);
  PROBE_CHECK(musaStreamDestroy(producer), musaSuccess);
  PROBE_CHECK(musaStreamDestroy(network), musaSuccess);
  PROBE_CHECK(musaStreamDestroy(consumer), musaSuccess);
  PROBE_CHECK(MPI_Comm_free(&local), MPI_SUCCESS);
  std::printf("P2P public API / stream-event probe rank=%d: PASS\n", rank);
  PROBE_CHECK(MPI_Finalize(), MPI_SUCCESS);
}

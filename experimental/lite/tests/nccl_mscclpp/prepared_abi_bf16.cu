#include "prepared_abi.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
void check(cudaError_t r) {
  if (r != cudaSuccess) throw std::runtime_error(cudaGetErrorString(r));
}
void nccl(ncclResult_t r) {
  if (r != ncclSuccess) throw std::runtime_error(ncclGetLastError(nullptr));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 2) throw std::invalid_argument("NATIVE_LIBRARY");
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local);
    int localRank, localRanks, visible;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localRanks);
    check(cudaGetDeviceCount(&visible));
    check(cudaSetDevice(visible == 1 ? 0 : localRanks - 1 - localRank));
    MPI_Comm_free(&local);
    ncclUniqueId id;
    if (!rank) nccl(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    nccl(ncclCommInitRank(&comm, ranks, id, rank));
    constexpr size_t count = 257;
    size_t extent = count * ranks + 128;
    unsigned short *a, *b, *c, *d;
    for (auto p : {&a, &b, &c, &d}) check(cudaMalloc(p, extent * 2));
    auto encode = [](float v) {
      return __bfloat16_as_ushort(__float2bfloat16_rn(v));
    };
    std::vector<unsigned short> input(extent, 0x5a5a), empty(extent, 0x5a5a);
    for (size_t i = 0; i < count; ++i) input[1 + i] = encode(rank + 1);
    check(cudaMemcpy(a, input.data(), extent * 2, cudaMemcpyHostToDevice));
    for (auto p : {b, c, d})
      check(cudaMemcpy(p, empty.data(), extent * 2, cudaMemcpyHostToDevice));
    ucclLiteUniqueId shared;
    if (!rank) nccl(ucclLiteGetUniqueId(&shared));
    MPI_Bcast(&shared, sizeof(shared), MPI_BYTE, 0, MPI_COMM_WORLD);
    ucclLiteOptions options;
    nccl(ucclLiteOptionsInit(&options));
    options.nativeLibrary = argv[1];
    options.configurationIdentity = "prepared-ABI-BF16-correctness-only";
    options.capacityBytes = 1UL << 20;
    options.dtype = ncclBfloat16;
    options.flags &= ~(UCCL_LITE_ALLOW_HOST | UCCL_LITE_ALLOW_PEER_PACKET);
    std::vector<ucclLiteRankLayout> layout(ranks, {0, 2});
    ucclLiteRequest requests[3] = {
        {ucclLiteAllReduce, count, layout.data(), size_t(ranks), 0},
        {ucclLiteAllGather, count, layout.data(), size_t(ranks), 0},
        {ucclLiteReduceScatter, count, layout.data(), size_t(ranks), 0}};
    nccl(ucclLiteNcclPrepare(comm, &shared, &options, requests, 3));
    cudaStream_t x, y;
    check(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&y, cudaStreamNonBlocking));
    nccl(ncclAllReduce(a + 1, b + 1, count, ncclBfloat16, ncclSum, comm, x));
    nccl(ncclAllGather(b + 1, d + 1, count, ncclBfloat16, comm, y));
    nccl(
        ncclReduceScatter(d + 1, c + 1, count, ncclBfloat16, ncclSum, comm, x));
    check(cudaStreamSynchronize(x));
    auto verify = [&](unsigned short* p, size_t n, float v) {
      std::vector<unsigned short> actual(extent);
      check(cudaMemcpy(actual.data(), p, extent * 2, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < extent; ++i)
        if (actual[i] != (i >= 1 && i - 1 < n ? encode(v) : 0x5a5a))
          throw std::runtime_error("BF16 ABI full buffer/guard mismatch");
    };
    float sum = ranks * (ranks + 1) / 2;
    verify(a, count, rank + 1);
    verify(b, count, sum);
    verify(d, count * ranks, sum);
    verify(c, count, ranks * sum);
    size_t bytes = 0;
    nccl(ucclLiteNcclReport(comm, nullptr, &bytes));
    std::vector<char> report(bytes);
    nccl(ucclLiteNcclReport(comm, report.data(), &bytes));
    if (std::string(report.data()).find("\"accelerated_calls\":3") ==
        std::string::npos)
      throw std::runtime_error(
          "BF16 fixture failed to exercise three accelerated calls");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(y, cudaStreamCaptureModeGlobal));
    nccl(ncclAllReduce(b + 1, c + 1, count, ncclBfloat16, ncclSum, comm, y));
    check(cudaStreamEndCapture(y, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    nccl(ucclLiteNcclRelease(comm));
    for (int replay = 0; replay < 3; ++replay) {
      check(cudaGraphLaunch(executable, y));
      check(cudaStreamSynchronize(y));
      verify(c, count, ranks * sum);
    }
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(x));
    check(cudaStreamDestroy(y));
    nccl(ncclCommDestroy(comm));
    for (auto p : {a, b, c, d}) check(cudaFree(p));
    if (!rank)
      std::cout << "PASS: BF16 prepared standard ABI; AR/AG/RS scalar tails, "
                   "minimum alignment, full buffers/guards and native capture "
                   "after retirement; delayed-native fixture\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}

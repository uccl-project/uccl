#include "prepared_abi.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <mpi.h>
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(std::string(ncclGetErrorString(result)) + ": " +
                             ncclGetLastError(nullptr));
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks, localRank, localRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local);
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localRanks);
  try {
    require(argc == 4, "MODE MANIFEST ALTERNATE_MANIFEST");
    std::string mode = argv[1];
    bool error = mode == "missing" || mode == "mismatch" || mode == "malformed";
    bool grouped = mode == "group" || (mode == "mixed-group" && !rank) ||
                   mode == "multi-local";
    bool unobserved = mode == "optout" || mode == "nonblocking";
    require(error || grouped || mode == "mixed-group" || unobserved ||
                mode == "no-manifest" || mode == "incompatible" ||
                mode == "fallback",
            "unknown mode");
    setenv("UCCL_LITE_AUTO_INIT", mode == "optout" ? "0" : "1", 1);
    setenv("UCCL_LITE_AUTO_MANIFEST", argv[2], 1);
    if ((mode == "missing" && !rank) || mode == "no-manifest")
      unsetenv("UCCL_LITE_AUTO_MANIFEST");
    if ((mode == "mismatch" || mode == "malformed") && !rank)
      setenv("UCCL_LITE_AUTO_MANIFEST", argv[3], 1);
    int localComms = mode == "multi-local" ? 2 : 1;
    int devices = 0;
    check(cudaGetDeviceCount(&devices));
    require(localComms * localRanks <= devices,
            "insufficient GPUs for unique local ownership");
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      ncclUniqueId id;
      if (!rank) check(ncclGetUniqueId(&id));
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
      std::vector<ncclComm_t> comms(localComms, nullptr);
      std::vector<cudaStream_t> streams(localComms);
      std::vector<float*> buffers(localComms);
      ncclResult_t status = ncclSuccess;
      if (grouped) {
        check(ncclGroupStart());
        check(ncclGroupStart());
      }
      for (int c = 0; c < localComms; ++c) {
        check(cudaSetDevice(localRank * localComms + c));
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = mode == "nonblocking" ? 0 : 1;
        status = ncclCommInitRankConfig(&comms[c], ranks * localComms, id,
                                        rank * localComms + c, &config);
        if (mode == "nonblocking") {
          require(status == ncclSuccess || status == ncclInProgress,
                  "nonblocking init failed");
          auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(60);
          do {
            check(ncclCommGetAsyncError(comms[c], &status));
            if (status != ncclInProgress) break;
            require(std::chrono::steady_clock::now() < deadline,
                    "nonblocking readiness timeout");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          } while (true);
        }
        if (!error) check(status);
      }
      if (grouped) {
        check(ncclGroupEnd());
        status = ncclGroupEnd();
      }
      if (error) {
        require(status == ncclInvalidUsage,
                "all ranks must reject manifest disagreement");
        check(ncclCommAbort(comms[0]));
        MPI_Barrier(MPI_COMM_WORLD);
        continue;
      }
      check(status);
      int participants = ranks * localComms;
      constexpr int guard = 64, count = 64, extent = count + 2 * guard;
      for (int c = 0; c < localComms; ++c) {
        check(cudaSetDevice(localRank * localComms + c));
        size_t bytes = 0;
        auto reportStatus = ucclLiteNcclReport(comms[c], nullptr, &bytes);
        if (unobserved)
          require(reportStatus == ncclInvalidArgument,
                  "unprepared native communicator acquired UCCL state");
        else {
          check(reportStatus);
          std::vector<char> report(bytes);
          check(ucclLiteNcclReport(comms[c], report.data(), &bytes));
          std::string text = report.data();
          require(text.find("\"automatic_init\":true") != std::string::npos,
                  "missing automatic-init diagnostic");
          require(
              text.find("\"accelerated_resources\":false") != std::string::npos,
              "native fallback created accelerator resources");
          if (!rank) std::cout << "INIT " << text << '\n';
        }
        check(cudaStreamCreateWithFlags(&streams[c], cudaStreamNonBlocking));
        check(cudaMalloc(&buffers[c], extent * sizeof(float)));
        std::vector<float> input(extent, -99);
        for (int i = 0; i < count; ++i)
          input[guard + i] = rank * localComms + c + lifetime + i % 5;
        check(cudaMemcpy(buffers[c], input.data(), extent * sizeof(float),
                         cudaMemcpyHostToDevice));
      }
      check(ncclGroupStart());
      for (int c = 0; c < localComms; ++c) {
        check(cudaSetDevice(localRank * localComms + c));
        check(ncclAllReduce(buffers[c] + guard, buffers[c] + guard, count,
                            ncclFloat32, ncclSum, comms[c], streams[c]));
      }
      auto groupStatus = ncclGroupEnd();
      if (mode == "nonblocking")
        require(groupStatus == ncclSuccess || groupStatus == ncclInProgress,
                "nonblocking group failed");
      else
        check(groupStatus);
      for (int c = 0; c < localComms; ++c) {
        check(cudaSetDevice(localRank * localComms + c));
        ncclResult_t ready;
        do {
          check(ncclCommGetAsyncError(comms[c], &ready));
        } while (ready == ncclInProgress);
        check(ready);
        check(cudaStreamSynchronize(streams[c]));
        std::vector<float> output(extent);
        check(cudaMemcpy(output.data(), buffers[c], extent * sizeof(float),
                         cudaMemcpyDeviceToHost));
        for (int i = 0; i < extent; ++i) {
          float expected = i >= guard && i < guard + count
                               ? participants * (participants - 1) / 2 +
                                     participants * (lifetime + (i - guard) % 5)
                               : -99;
          require(output[i] == expected,
                  "native full-buffer/guard reference mismatch");
        }
        check(ncclCommDestroy(comms[c]));
        size_t bytes = 0;
        require(ucclLiteNcclReport(comms[c], nullptr, &bytes) ==
                    ncclInvalidArgument,
                "destroy retained automatic state");
        check(cudaFree(buffers[c]));
        check(cudaStreamDestroy(streams[c]));
      }
    }
    if (!rank)
      std::cout << "PASS automatic initialization " << mode
                << " ranks=" << ranks << " two lifetimes\n";
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

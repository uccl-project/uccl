// Public NCCL/MPI client for the retained two-host/four-GPU network provider.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local);
  int localRank, localRanks;
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localRanks);
  try {
    if (ranks != 8 || localRanks != 4 || argc != 3 ||
        !std::getenv("UCCL_LITE_NETWORK_PROFILE"))
      throw std::invalid_argument(
          "eight ranks, mode and corrupt-profile path required");
    int device = std::getenv("UCCL_LITE_ABI_REVERSE_DEVICES") ? 3 - localRank
                                                              : localRank;
    if (cudaSetDevice(device) != cudaSuccess)
      throw std::runtime_error("CUDA setup failed");
    std::string mode = argv[1];
    if (mode != "mismatch" && mode != "corrupt" && mode != "groups" &&
        mode != "binding" && mode != "rank-map")
      throw std::invalid_argument("invalid rejection mode");
    ncclUniqueId id;
    if (!rank && ncclGetUniqueId(&id) != ncclSuccess)
      throw std::runtime_error("unique ID failed");
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    // The launcher already verified the strict application profile. Change one
    // rank's resource-profile configuration to exercise runtime agreement
    // itself.
    if (rank == ranks - 1) {
      if (mode == "rank-map") {
        auto source = std::getenv("UCCL_LITE_MSCCLPP_PROFILE");
        if (!source) throw std::runtime_error("missing application profile");
        std::ifstream input(source);
        std::vector<std::string> lines;
        for (std::string line; std::getline(input, line);)
          lines.push_back(line);
        if (lines.size() < 3 || lines[0] != "UCCL_LITE_RUNTIME_PROFILE_V3")
          throw std::runtime_error("rank-map rejection needs V3 profile");
        std::istringstream row(lines[2]);
        std::vector<std::string> fields;
        for (std::string word; row >> word;) fields.push_back(word);
        if (fields.size() != 18)
          throw std::runtime_error("invalid test rank-map extent");
        fields[16] = fields[2];
        fields[17] = fields[3];
        lines[2].clear();
        for (auto const& field : fields) lines[2] += field + ' ';
        std::string path = std::string(source) + ".duplicate-binding";
        std::ofstream output(path);
        for (auto const& line : lines) output << line << '\n';
        output.close();
        if (!output)
          throw std::runtime_error("write invalid test profile failed");
        setenv("UCCL_LITE_MSCCLPP_PROFILE", path.c_str(), 1);
      } else if (mode == "binding") {
        if (cudaSetDevice((device + 1) % localRanks) != cudaSuccess)
          throw std::runtime_error("binding mismatch setup failed");
      } else if (mode == "mismatch")
        unsetenv("UCCL_LITE_NETWORK_PROFILE");
      else
        setenv("UCCL_LITE_NETWORK_PROFILE", argv[2], 1);
    }
    ncclComm_t comm = nullptr;
    auto result = ncclCommInitRank(&comm, ranks, id, rank);
    int rejected = result != ncclSuccess && !comm, total = 0;
    MPI_Allreduce(&rejected, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (total != ranks)
      throw std::runtime_error(
          "network preparation was not rejected by every rank");
    if (!rank)
      std::printf(
          "PASS: network preparation rejected on all 8 ranks; mode=%s\n",
          mode.c_str());
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}

#include "../../integration/nccl_lite/host_control.h"
#include <cstdio>
#include <signal.h>
#include <sys/wait.h>

using uccl_lite::nccl_detail::HostControl;
namespace {
unsigned char pattern(int rank, int generation, size_t i) {
  return static_cast<unsigned char>(rank * 31 + generation * 17 + i * 7);
}
void child(std::array<unsigned char, 128> id, int rank, int ranks,
           int scenario) {
  alarm(8);
  bool expectFailure = scenario == 1 || scenario == 2;
  if (scenario == 1 && rank == ranks - 1) id[0] ^= 0x80;
  if (rank == 0) usleep(50000);  // Clients must tolerate a late root.
  try {
    for (int lifetime = 0; lifetime < (scenario ? 1 : 3); ++lifetime) {
      id[127] = static_cast<unsigned char>(lifetime);
      HostControl control("127.0.0.1", id.data(), id.size(), rank, ranks,
                          expectFailure ? 800 : 4000);
      if (scenario == 2 && rank == ranks - 1) return;
      for (int generation = 0; generation < 5; ++generation) {
        size_t bytes = generation % 2 ? 131071 : 33;
        std::vector<unsigned char> local(bytes), all(bytes * ranks);
        for (size_t i = 0; i < bytes; ++i)
          local[i] = pattern(rank, generation, i);
        control.allGather(local.data(), all.data(), bytes);
        for (int r = 0; r < ranks; ++r)
          for (size_t i = 0; i < bytes; ++i)
            if (all[size_t(r) * bytes + i] != pattern(r, generation, i))
              throw std::logic_error("allGather payload mismatch");
        std::vector<unsigned char> broadcast(128, 0);
        if (rank == 0)
          for (size_t i = 0; i < broadcast.size(); ++i)
            broadcast[i] = pattern(7, generation, i);
        control.broadcast(broadcast.data(), broadcast.size());
        for (size_t i = 0; i < broadcast.size(); ++i)
          if (broadcast[i] != pattern(7, generation, i))
            throw std::logic_error("broadcast payload mismatch");
      }
    }
  } catch (std::runtime_error const& error) {
    if (expectFailure) return;
    std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
    _exit(1);
  }
  if (expectFailure) {
    std::fprintf(stderr, "rank %d accepted failed session\n", rank);
    _exit(1);
  }
}
void run(int ranks, int scenario) {
  std::array<unsigned char, 128> id{};
  uint64_t seed = uint64_t(getpid()) * 1099511628211ULL + ranks * 13 + scenario;
  std::memcpy(id.data(), &seed, sizeof(seed));
  int occupied = -1;
  if (scenario == 3) {
    occupied = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(
        uccl_lite::nccl_detail::hostControlInitialPort(id.data(), id.size()));
    if (occupied < 0 ||
        bind(occupied, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) ||
        listen(occupied, ranks))
      throw std::runtime_error("occupied-port fixture failed");
  }
  std::vector<pid_t> children;
  for (int r = 0; r < ranks; ++r) {
    auto pid = fork();
    if (pid < 0) throw std::runtime_error("fork failed");
    if (!pid) {
      if (occupied >= 0) close(occupied);
      child(id, r, ranks, scenario);
      _exit(0);
    }
    children.push_back(pid);
  }
  bool ok = true;
  for (auto pid : children) {
    int status;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
        WEXITSTATUS(status))
      ok = false;
  }
  if (occupied >= 0) close(occupied);
  if (!ok) throw std::runtime_error("CPU control child failed");
}
}  // namespace
int main() {
  try {
    run(4, 0);
    run(8, 0);
    run(4, 1);
    run(4, 2);
    run(4, 3);
    std::puts(
        "PASS: CPU control 4/8 ranks, changing payloads, three lifetimes, "
        "late root, wrong session, disconnected peer and occupied port");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

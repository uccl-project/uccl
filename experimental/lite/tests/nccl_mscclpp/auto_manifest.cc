#include "auto_manifest.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace uccl_lite::integration;
void require(bool ok) {
  if (!ok) throw std::runtime_error("manifest contract failed");
}
int main() {
  char path[] = "build/auto-manifest-XXXXXX";
  int fd = mkstemp(path);
  require(fd >= 0);
  close(fd);
  struct Remove {
    char* path;
    ~Remove() { unlink(path); }
  } remove{path};
  auto load = [&](std::string const& rows, std::string dtype = "float32",
                  std::string flags = "17", size_t count = 1) {
    std::ofstream out(path);
    out << "uccl-lite-auto-prepare-v1\ndtype " << dtype
        << "\ncapacity 4096\nflags " << flags << "\nrequests " << count << '\n'
        << rows;
    out.close();
    return loadAutoManifest(path);
  };
  auto rejects = [&](auto fn) {
    bool rejected = false;
    try {
      fn();
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(rejected);
  };
  for (int ranks : {1, 2, 3, 4, 8, 32}) {
    auto plans = materializeAutoManifest(load("allreduce 64 out 256"), ranks);
    require(plans.size() == 1 && plans[0].layouts.size() == size_t(ranks));
  }
  require(materializeAutoManifest(load("allgather 512 out 256"), 4).empty());
  require(materializeAutoManifest(load("allgather 17 in 256"), 4).empty());
  require(materializeAutoManifest(load("allgather 17 in 4"), 4).size() == 1);
  auto explicitRanks = load("reducescatter 64 ranks 2 0 256 1 4");
  require(materializeAutoManifest(explicitRanks, 2).size() == 1);
  require(materializeAutoManifest(explicitRanks, 4).empty());
  require(materializeAutoManifest(load("allreduce 1 out 2", "bfloat16"), 3)
              .size() == 1);
  require(
      materializeAutoManifest(load("allreduce 1 out 2", "float16"), 3).size() ==
      1);
  rejects([&] { load("allreduce 1 out 2"); });
  rejects([&] { load("allreduce 1 out 3"); });
  rejects([&] { load("allreduce 1 out 512"); });
  rejects([&] { load("allreduce 1 out 4", "float64"); });
  rejects([&] { load("allreduce 1 out 4", "float32", "64"); });
  rejects([&] { load("allreduce 1 out 4", "float32", "17", 1025); });
  rejects([&] { load("allreduce 0 out 4"); });
  rejects([&] { load("allreduce -1 out 4"); });
  rejects([&] { load("allreduce -18446744073709551615 out 4"); });
  rejects([&] { load("allreduce 1 out 4", "float32", "-4294967295"); });
  rejects([&] { load("allreduce 1 ranks 1 -4294967295 4"); });
  rejects([&] { load("allreduce 1 ranks 2 0 4"); });
  rejects([&] { load("allreduce 1 ranks 1 2 4"); });
  rejects([&] { load("allreduce 1 out"); });
  rejects([&] { load("allreduce 1 out 4 trailing"); });
  rejects([&] { load("broadcast 1 out 4"); });
  rejects([&] {
    materializeAutoManifest(
        load("allreduce 1 out 4\nallreduce 1 ranks 2 0 4 0 4", "float32", "17",
             2),
        2);
  });
  rejects([&] { load(std::string(1 << 20, ' ')); });
  rejects([&] { loadAutoManifest("build"); });
  std::cout << "PASS automatic workload manifest: exact layouts, rank "
               "materialization, capacity and malformed inputs\n";
}

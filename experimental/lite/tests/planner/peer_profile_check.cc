#include "uccl_lite/peer_profile.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
using namespace uccl_lite;
void require(bool ok, char const* text) {
  if (!ok) throw std::runtime_error(text);
}
int main() {
  char name[] = "build/peer-profile-XXXXXX";
  try {
    auto dir = mkdtemp(name);
    if (!dir) throw std::runtime_error("temporary directory failed");
    std::string path = std::string(dir) + "/plans";
    Topology t;
    t.identity = "topology";
    t.devices = {0, 1, 2, 3};
    t.peerAccess.resize(16, true);
    Workload w{Collective::AllReduce, ScalarType::BFloat16, 1024, false, 256};
    PeerPlanRecord r{t.identity,
                     "executor",
                     "resource-probe",
                     w,
                     {{{20, 21, 22}, {8, 9, 10}, {5, 6, 7}}},
                     true};
    updatePeerPlanRecord(path, r);
    auto records = loadPeerPlanRecords(path);
    auto match = matchPeerPlanRecord(t, w, "executor", records);
    require(match && match->qualified && match->fromCache &&
                match->family == PeerFamily::Bulk,
            "exact record was not requalified");
    require(!matchPeerPlanRecord(t, w, "new-build", records),
            "stale executor accepted");
    auto moved = t;
    moved.identity += "changed";
    require(!matchPeerPlanRecord(moved, w, "executor", records),
            "stale topology accepted");
    auto other = w;
    other.alignment = 2;
    require(!matchPeerPlanRecord(t, other, "executor", records),
            "alignment mismatch accepted");
    r.timings[2] = {25, 26, 27};
    r.timings[1] = {20, 21, 22};
    updatePeerPlanRecord(path, r);
    require(matchPeerPlanRecord(t, w, "executor", loadPeerPlanRecords(path))
                    ->family == PeerFamily::Direct,
            "new regression did not restore baseline");
    std::vector<pid_t> children;
    for (int i = 1; i <= 3; ++i) {
      auto pid = fork();
      if (pid < 0) throw std::runtime_error("fork failed");
      if (!pid) {
        try {
          auto child = r;
          child.workload.bytes = 1024 * (i + 1);
          updatePeerPlanRecord(path, child);
          _exit(0);
        } catch (...) {
          _exit(1);
        }
      }
      children.push_back(pid);
    }
    for (auto pid : children) {
      int status = 0;
      require(waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
                  WEXITSTATUS(status) == 0,
              "cache writer failed");
    }
    require(loadPeerPlanRecords(path).size() == 4,
            "concurrent writers lost a record");
    auto invalid = r;
    invalid.correct = false;
    bool rejected = false;
    try {
      updatePeerPlanRecord(path, invalid);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(rejected && loadPeerPlanRecords(path).size() == 4,
            "invalid update changed cache");
    invalid = r;
    invalid.timings[2].clear();
    rejected = false;
    try {
      matchPeerPlanRecord(t, w, "executor", {invalid});
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(rejected, "incomplete eligible-family evidence accepted");
    {
      std::ofstream out(path, std::ios::app);
      out << "corrupt";
    }
    rejected = false;
    try {
      loadPeerPlanRecords(path);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(rejected, "trailing corruption accepted");
    std::filesystem::remove_all(dir);
    std::cout << "PASS: exact peer plan identity, requalification, concurrent "
                 "persistence and corruption rejection\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

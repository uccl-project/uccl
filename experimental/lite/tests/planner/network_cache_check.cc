#include "uccl_lite/network_cache.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sys/wait.h>
#include <unistd.h>
using namespace uccl_lite;
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class F>
void reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (std::exception const&) {
    rejected = true;
  }
  require(rejected, "invalid network cache accepted");
}
NetworkPlanRecord record(size_t bytes) {
  NetworkPlanRecord r;
  r.workload = {Collective::AllReduce, ScalarType::Float32, bytes, false, 4};
  r.layouts.resize(8);
  r.rankOrder.resize(8);
  std::iota(r.rankOrder.begin(), r.rankOrder.end(), 0);
  r.rails.resize(8, 1);
  r.candidateSamples.resize(18, 5);
  r.baselineSamples.resize(18, 10);
  r.candidateCorrect = true;
  r.useHierarchy = true;
  return r;
}
int main() {
  char directory[] = "build/network-cache-check-XXXXXX";
  if (!mkdtemp(directory)) return 1;
  std::string path = std::string(directory) + "/cache";
  try {
    require(!loadNetworkPlanCache(path), "absent cache is not a miss");
    NetworkPlanCache c;
    c.identity = "context\nwith quotes \" and \\ backslash";
    NetworkProfile p;
    p.hosts = {{{"a", "ta", "sa"}, {"b", "tb", "sb"}}};
    p.probeIdentity = "probe";
    NetworkPathObservation sample;
    sample.key = {0, 1, "g0", "g1", "nic0", "nic1", 1, 1};
    sample.iterations = 1;
    sample.sizes = {{16, {1, 2, 3}}, {1024, {4, 5, 6}}};
    p.paths.push_back(sample);
    c.calibration = p;
    auto a = record(256), b = record(512);
    c.records = {a, b};
    auto encoded = encodeNetworkPlanCache(c),
         decoded = encodeNetworkPlanCache(decodeNetworkPlanCache(encoded));
    require(encoded == decoded, "cache resource/sample roundtrip changed");
    require(encoded.rfind("uccl-lite-network-plans-v1", 0) == 0,
            "hierarchy-only wire format changed");
    auto direct = a;
    direct.candidate = NetworkCacheCandidate::DirectExchange;
    direct.useHierarchy = false;
    direct.useDirect = true;
    direct.rankOrder.clear();
    direct.rails.clear();
    direct.configurationIdentities.assign(8,
                                          "actual direct topology/transport");
    require(!sameNetworkRequest(a, direct),
            "candidate kinds share a cache key");
    auto mixed = c;
    mixed.records.push_back(direct);
    auto v2 = encodeNetworkPlanCache(mixed);
    require(v2.rfind("uccl-lite-network-plans-v2", 0) == 0 &&
                encodeNetworkPlanCache(decodeNetworkPlanCache(v2)) == v2,
            "mixed candidate roundtrip failed");
    auto bf16 = direct;
    bf16.workload = {Collective::AllGather, ScalarType::BFloat16, 257 * 3 * 2,
                     false, 2};
    bf16.layouts = {{false, 2}, {true, 2}, {false, 2}};
    bf16.configurationIdentities = {"rank0", "rank1", "rank2"};
    auto generic = c;
    generic.calibration.reset();
    generic.records = {bf16};
    require(decodeNetworkPlanCache(encodeNetworkPlanCache(generic))
                    .records[0]
                    .workload.dtype == ScalarType::BFloat16,
            "BF16/unequal extent lost");
    for (int variant = 0; variant < 7; ++variant) {
      auto broken = generic;
      auto& r = broken.records[0];
      if (variant == 0) r.configurationIdentities.pop_back();
      if (variant == 1) r.configurationIdentities[0].clear();
      if (variant == 2) r.useHierarchy = true;
      if (variant == 3) r.useDirect = false;
      if (variant == 4) ++r.workload.bytes;
      if (variant == 5) r.layouts[1].alignment = 256;
      if (variant == 6) r.workload.dtype = ScalarType::Float16;
      reject([&] { encodeNetworkPlanCache(broken); });
    }
    reject([&] { decodeNetworkPlanCache(v2.substr(0, v2.size() - 8)); });
    auto dualPath = path + "-dual";
    auto empty = c;
    empty.records.clear();
    updateNetworkPlanCache(dualPath, empty, a);
    auto directContext = empty;
    directContext.calibration.reset();
    updateNetworkPlanCache(dualPath, directContext, direct);
    require(loadNetworkPlanCache(dualPath)->calibration.has_value(),
            "direct writer discarded concurrent hierarchy calibration");
    require(loadNetworkPlanCache(dualPath)->records.size() == 2,
            "direct update overwrote hierarchy");
    eraseNetworkPlanCacheRecord(dualPath, c.identity, direct);
    require(loadNetworkPlanCache(dualPath)->records.size() == 1 &&
                loadNetworkPlanCache(dualPath)->records[0].candidate ==
                    NetworkCacheCandidate::Hierarchy,
            "direct eviction erased hierarchy");
    auto changed = a;
    changed.layouts[1].inPlace = true;
    require(!sameNetworkRequest(a, changed), "layout key collapsed");
    changed = a;
    changed.workload.collective = Collective::AllGather;
    require(!sameNetworkRequest(a, changed), "operation key collapsed");
    auto bad = c;
    bad.records[0].useHierarchy = false;
    reject([&] { encodeNetworkPlanCache(bad); });
    bad = c;
    bad.records[0].candidateSamples.pop_back();
    reject([&] { encodeNetworkPlanCache(bad); });
    bad = c;
    bad.records[0].candidateSamples[0] = 0;
    reject([&] { encodeNetworkPlanCache(bad); });
    bad = c;
    bad.records[0].rankOrder[1] = 0;
    reject([&] { encodeNetworkPlanCache(bad); });
    bad = c;
    bad.records[0].rails[0] = 8;
    reject([&] { encodeNetworkPlanCache(bad); });
    bad = c;
    bad.records[0].layouts[0].alignment = 3;
    reject([&] { encodeNetworkPlanCache(bad); });
    bad = c;
    bad.records.push_back(a);
    reject([&] { encodeNetworkPlanCache(bad); });
    reject([&] { decodeNetworkPlanCache(encoded + "junk"); });
    reject(
        [&] { decodeNetworkPlanCache(encoded.substr(0, encoded.size() / 2)); });
    auto incompatible = encoded;
    incompatible.replace(0, incompatible.find('\n'),
                         "uccl-lite-network-plans-v9");
    reject([&] { decodeNetworkPlanCache(incompatible); });
    c.records.clear();
    updateNetworkPlanCache(path, c, a);
    updateNetworkPlanCache(path, c, b);
    require(loadNetworkPlanCache(path)->records.size() == 2,
            "update lost another shape");
    a.candidateCorrect = false;
    a.useHierarchy = false;
    updateNetworkPlanCache(path, c, a);
    require(loadNetworkPlanCache(path)->records.size() == 2,
            "replacement appended duplicate");
    for (int child = 0; child < 4; ++child) {
      pid_t pid = fork();
      require(pid >= 0, "fork failed");
      if (!pid) {
        try {
          updateNetworkPlanCache(path, c, record(128 * (child + 10)));
          _exit(0);
        } catch (...) {
          _exit(1);
        }
      }
    }
    for (int child = 0; child < 4; ++child) {
      int status;
      require(
          wait(&status) > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "concurrent writer failed");
    }
    require(loadNetworkPlanCache(path)->records.size() == 6,
            "concurrent updates lost records");
    eraseNetworkPlanCacheRecord(path, "wrong identity", a);
    require(loadNetworkPlanCache(path)->records.size() == 6,
            "wrong-context erase mutated cache");
    eraseNetworkPlanCacheRecord(path, c.identity, a);
    require(loadNetworkPlanCache(path)->records.size() == 5,
            "exact eviction failed");
    auto other = c;
    other.identity = "new context";
    updateNetworkPlanCache(path, other, b);
    auto loaded = loadNetworkPlanCache(path);
    require(loaded->identity == other.identity && loaded->records.size() == 1,
            "contexts mixed");
    NetworkPlanCache bounded;
    bounded.identity = "bounded-context";
    for (size_t i = 0; i < 4096; ++i)
      bounded.records.push_back(record(128 * (i + 1000)));
    updateNetworkPlanCache(path, bounded, record(128 * 10000));
    auto limited = loadNetworkPlanCache(path);
    require(limited->records.size() == 4096 &&
                limited->records.front().workload.bytes == 128 * 1001 &&
                limited->records.back().workload.bytes == 128 * 10000,
            "bounded update failed to retire oldest qualification");
    {
      std::ofstream out(path);
      out << "corrupt";
    }
    reject([&] { updateNetworkPlanCache(path, c, a); });
    {
      std::ifstream in(path);
      std::string content;
      in >> content;
      require(content == "corrupt", "malformed cache silently replaced");
    }
    reject([&] { loadNetworkPlanCache(directory); });
    std::filesystem::remove_all(directory);
    std::cout << "PASS: exact network cache evidence/layout, embedded "
                 "resources, bounded parsing, atomic concurrent merge, context "
                 "replacement and eviction\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    std::filesystem::remove_all(directory);
    return 1;
  }
}

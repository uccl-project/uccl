#pragma once
#include "prepared_abi.h"
#include <string>
#include <vector>
namespace uccl_lite::integration {
struct AutoInitRecord {
  ncclComm_t comm = nullptr;
  bool grouped = false, blocking = true;
  std::string configuration;
};
struct AutoInitCompletion {
  int rank, ranks;
  ucclLiteUniqueId id;
  std::string root;
};
struct AutoInitApi {
  decltype(&ncclCommCount) count;
  decltype(&ncclCommUserRank) rank;
  decltype(&ncclCommCuDevice) device;
  decltype(&ncclAllGather) gather;
  decltype(&ncclGroupStart) groupStart;
  decltype(&ncclGroupEnd) groupEnd;
  decltype(&ucclLiteGetUniqueId) uniqueId;
  decltype(&ucclLiteNcclPrepare) prepare;
  bool (*available)();
  void (*complete)(std::vector<AutoInitCompletion> const&);
  std::string library, implementation;
};
struct AutoInitResult {
  ncclComm_t comm;
  ncclResult_t status;
  std::string reason;
};
bool automaticInitEnabled();
AutoInitRecord autoInitRecord(ncclComm_t comm, ncclConfig_t const* config,
                              bool grouped);
std::vector<AutoInitResult> prepareNativeInitializations(
    std::vector<AutoInitRecord> const& records, AutoInitApi const& api);
}  // namespace uccl_lite::integration

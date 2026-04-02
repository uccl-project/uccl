// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstdlib>
#include <type_traits>

// clang-format off
#include "env.hpp"
#include "errors.hpp"
// clang-format on

#include "logger.hpp"

template <typename T>
T readEnv(std::string const& envName, const T& defaultValue) {
  char const* envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return defaultValue;
  if constexpr (std::is_same_v<T, int>) {
    return atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    return (std::string(envCstr) != "0");
  } else {
    return T(envCstr);
  }
}

template <typename T>
void readAndSetEnv(std::string const& envName, T& env) {
  char const* envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return;
  if constexpr (std::is_same_v<T, int>) {
    env = atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    env = (std::string(envCstr) != "0");
  } else {
    env = std::string(envCstr);
  }
}

template <typename T>
void logEnv(std::string const& envName, const T& env) {
  if (!getenv(envName.c_str())) return;
  INFO(mscclpp::ENV, envName, "=", env);
}

namespace mscclpp {

Env::Env()
    : debug(readEnv<std::string>("MSCCLPP_DEBUG", "")),
      debugSubsys(readEnv<std::string>("MSCCLPP_DEBUG_SUBSYS", "")),
      debugFile(readEnv<std::string>("MSCCLPP_DEBUG_FILE", "")),
      logLevel(readEnv<std::string>("MSCCLPP_LOG_LEVEL", "ERROR")),
      logSubsys(readEnv<std::string>("MSCCLPP_LOG_SUBSYS", "ALL")),
      logFile(readEnv<std::string>("MSCCLPP_LOG_FILE", "")),
      hcaDevices(readEnv<std::string>("MSCCLPP_HCA_DEVICES", "")),
      ibvSo(readEnv<std::string>("MSCCLPP_IBV_SO", "")),
      ibvMode(readEnv<std::string>("MSCCLPP_IBV_MODE", "host")),
      hostid(readEnv<std::string>("MSCCLPP_HOSTID", "")),
      socketFamily(readEnv<std::string>("MSCCLPP_SOCKET_FAMILY", "")),
      socketIfname(readEnv<std::string>("MSCCLPP_SOCKET_IFNAME", "")),
      commId(readEnv<std::string>("MSCCLPP_COMM_ID", "")),
      cacheDir(readEnv<std::string>(
          "MSCCLPP_CACHE_DIR",
          readEnv<std::string>("HOME", "~") + "/.cache/mscclpp")),
      npkitDumpDir(readEnv<std::string>("MSCCLPP_NPKIT_DUMP_DIR", "")),
      cudaIpcUseDefaultStream(
          readEnv<bool>("MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM", false)),
      ncclSharedLibPath(readEnv<std::string>("MSCCLPP_NCCL_LIB_PATH", "")),
      forceNcclFallbackOperation(
          readEnv<std::string>("MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION", "")),
      ncclSymmetricMemory(
          readEnv<bool>("MSCCLPP_NCCL_SYMMETRIC_MEMORY", false)),
      ncclSendRecvStagingBytes(readEnv<int>(
          "MSCCLPP_NCCL_SENDRECV_STAGING_BYTES", 256 * 1024 * 1024)),
      forceDisableNvls(readEnv<bool>("MSCCLPP_FORCE_DISABLE_NVLS", false)),
      disableChannelCache(
          readEnv<bool>("MSCCLPP_DISABLE_CHANNEL_CACHE", false)) {}

std::shared_ptr<Env> env() {
  static std::shared_ptr<Env> globalEnv = std::shared_ptr<Env>(new Env());
  static bool logged = false;
  if (!logged) {
    logged = true;
    // cannot log inside the constructor because of circular dependency
    logEnv("MSCCLPP_DEBUG", globalEnv->debug);
    logEnv("MSCCLPP_DEBUG_SUBSYS", globalEnv->debugSubsys);
    logEnv("MSCCLPP_DEBUG_FILE", globalEnv->debugFile);
    logEnv("MSCCLPP_LOG_LEVEL", globalEnv->logLevel);
    logEnv("MSCCLPP_LOG_SUBSYS", globalEnv->logSubsys);
    logEnv("MSCCLPP_LOG_FILE", globalEnv->logFile);
    logEnv("MSCCLPP_HCA_DEVICES", globalEnv->hcaDevices);
    logEnv("MSCCLPP_IBV_SO", globalEnv->ibvSo);
    logEnv("MSCCLPP_IBV_MODE", globalEnv->ibvMode);
    logEnv("MSCCLPP_HOSTID", globalEnv->hostid);
    logEnv("MSCCLPP_SOCKET_FAMILY", globalEnv->socketFamily);
    logEnv("MSCCLPP_SOCKET_IFNAME", globalEnv->socketIfname);
    logEnv("MSCCLPP_COMM_ID", globalEnv->commId);
    logEnv("MSCCLPP_CACHE_DIR", globalEnv->cacheDir);
    logEnv("MSCCLPP_NPKIT_DUMP_DIR", globalEnv->npkitDumpDir);
    logEnv("MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM",
           globalEnv->cudaIpcUseDefaultStream);
    logEnv("MSCCLPP_NCCL_LIB_PATH", globalEnv->ncclSharedLibPath);
    logEnv("MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION",
           globalEnv->forceNcclFallbackOperation);
    logEnv("MSCCLPP_NCCL_SYMMETRIC_MEMORY", globalEnv->ncclSymmetricMemory);
    logEnv("MSCCLPP_NCCL_SENDRECV_STAGING_BYTES",
           globalEnv->ncclSendRecvStagingBytes);
    logEnv("MSCCLPP_FORCE_DISABLE_NVLS", globalEnv->forceDisableNvls);
    logEnv("MSCCLPP_DISABLE_CHANNEL_CACHE", globalEnv->disableChannelCache);
  }
  return globalEnv;
}

}  // namespace mscclpp

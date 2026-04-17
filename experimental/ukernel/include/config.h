#pragma once

#include <cstdlib>
#include <string>

#define DEFAULT_EXCHANGER_SERVER_IP "0.0.0.0"
#define DEFAULT_EXCHANGER_SERVER_PORT 6979

namespace UKernel {
namespace Transport {

enum class PreferredTransport { Auto, Ipc, Rdma, Uccl, Tcp };

struct CommunicatorConfig {
  std::string exchanger_ip = DEFAULT_EXCHANGER_SERVER_IP;
  int exchanger_port = DEFAULT_EXCHANGER_SERVER_PORT;
  std::string oob_namespace = "default";
  int local_id = -1;
  PreferredTransport preferred_transport = PreferredTransport::Auto;

  static CommunicatorConfig from_env() {
    CommunicatorConfig config;
    config.exchanger_ip =
        getEnvOrDefault("UHM_EXCHANGER_SERVER_IP", DEFAULT_EXCHANGER_SERVER_IP);
    config.exchanger_port = getEnvOrDefault("UHM_EXCHANGER_SERVER_PORT",
                                            DEFAULT_EXCHANGER_SERVER_PORT);
    config.oob_namespace = getEnvOrDefault("UHM_OOB_NAMESPACE", "default");
    config.local_id = getLocalIdOrDefault();
    return config;
  }

 private:
  static int getEnvOrDefault(char const* env_name, int default_val) {
    char const* val = std::getenv(env_name);
    if (val) {
      try {
        return std::stoi(val);
      } catch (...) {
        return default_val;
      }
    }
    return default_val;
  }

  static std::string getEnvOrDefault(char const* env_name,
                                     std::string const& default_val) {
    char const* val = std::getenv(env_name);
    if (val) return std::string(val);
    return default_val;
  }

  static int getLocalIdOrDefault() {
    constexpr char const* kEnvNames[] = {
        "UHM_LOCAL_ID",    "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID", "SLURM_LOCALID",
        "LOCAL_RANK",
    };
    for (char const* env_name : kEnvNames) {
      char const* val = std::getenv(env_name);
      if (!val || val[0] == '\0') continue;
      try {
        return std::stoi(val);
      } catch (...) {
      }
    }
    return -1;
  }
};

}  // namespace Transport

namespace Device {}
}  // namespace UKernel

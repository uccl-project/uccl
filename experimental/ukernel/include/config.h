#pragma once

#include <cstdlib>
#include <string>

#define DEFAULT_EXCHANGER_SERVER_IP "0.0.0.0"
#define DEFAULT_EXCHANGER_SERVER_PORT 6979

namespace UKernel {
namespace Transport {

enum class PreferredTransport { Auto, Ipc, Uccl, Tcp };

struct CommunicatorConfig {
  std::string exchanger_ip;
  int exchanger_port;
  int local_id;
  PreferredTransport preferred_transport;

  CommunicatorConfig()
      : exchanger_ip(getEnvOrDefault("UHM_EXCHANGER_SERVER_IP",
                                     DEFAULT_EXCHANGER_SERVER_IP)),
        exchanger_port(getEnvOrDefault("UHM_EXCHANGER_SERVER_PORT",
                                       DEFAULT_EXCHANGER_SERVER_PORT)),
        local_id(getLocalIdOrDefault()),
        preferred_transport(PreferredTransport::Auto) {}

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

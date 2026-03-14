#pragma once

#include <cstdlib>
#include <string>

#define DEFAULT_EXCHANGER_SERVER_IP "0.0.0.0"
#define DEFAULT_EXCHANGER_SERVER_PORT 6979

namespace UKernel {
namespace Transport {

struct CommunicatorConfig {
  std::string exchanger_ip;
  int exchanger_port;

  CommunicatorConfig()
      : exchanger_ip(getEnvOrDefault("UHM_EXCHANGER_SERVER_IP",
                                     DEFAULT_EXCHANGER_SERVER_IP)),
        exchanger_port(getEnvOrDefault("UHM_EXCHANGER_SERVER_PORT",
                                       DEFAULT_EXCHANGER_SERVER_PORT)) {}

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
};

}  // namespace Transport

namespace Device {}
}  // namespace UKernel

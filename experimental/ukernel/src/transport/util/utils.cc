#include "utils.h"
#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <netdb.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

std::string get_hostname() {
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf)) == 0) {
    return std::string(buf);
  }
  return "unknown";
}

std::string get_primary_ip() {
  char hostname[256] = {0};
  if (gethostname(hostname, sizeof(hostname)) != 0) return "0.0.0.0";

  struct hostent* he = gethostbyname(hostname);
  if (!he || !he->h_addr_list[0]) return "0.0.0.0";

  struct in_addr** addr_list = (struct in_addr**)he->h_addr_list;
  return std::string(inet_ntoa(*addr_list[0]));
}

std::string read_machine_id() {
  std::ifstream f("/etc/machine-id");
  if (!f.is_open()) return "";
  std::string id;
  std::getline(f, id);
  f.close();
  return id;
}

std::string generate_host_id(bool include_ip) {
  if (char const* override_id = std::getenv("UHM_HOST_ID_OVERRIDE")) {
    if (override_id[0] != '\0') {
      std::string id(override_id);
      if (include_ip) {
        id += "-" + get_primary_ip();
      }
      return id;
    }
  }

  std::string id = read_machine_id();
  if (!id.empty()) {
    if (include_ip) {
      id += "-" + get_primary_ip();
    }
    return id;
  }

  id = get_hostname();
  if (include_ip) {
    id += "-" + get_primary_ip();
  }
  return id;
}

}  // namespace Transport
}  // namespace UKernel

#pragma once

#include <arpa/inet.h>
#include <glog/logging.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <netinet/in.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ifaddrs.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace uccl {

#define MAX_IB_DEVS 32
#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)

struct ib_dev {
  char prefix[64];
  int port;
};

static bool match_if(char const* string, char const* ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool match_port(int const port1, int const port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}

static bool match_if_list(char const* string, int port, struct ib_dev* ifList,
                          int listSize, bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i = 0; i < listSize; i++) {
    if (match_if(string, ifList[i].prefix, matchExact) &&
        match_port(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

static inline int parse_interfaces(char const* string, struct ib_dev* ifList,
                                   int maxList) {
  if (!string) return 0;

  char const* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr + 1);
        ifNum++;
        ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++;
        ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

/* Format a string representation of a (struct sockaddr *) socket address using
 * getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
static inline char const* socket_to_string(struct sockaddr* saddr, char* buf) {
  if (buf == NULL || saddr == NULL) return NULL;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) {
    buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void)getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST,
                    service, NI_MAXSERV, NI_NUMERICHOST | NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

static inline char const* socket_to_string(union socketAddress* saddr,
                                           char* buf) {
  return socket_to_string(&saddr->sa, buf);
}

static inline uint16_t socket_to_port(struct sockaddr* saddr) {
  return ntohs(saddr->sa_family == AF_INET
                   ? ((struct sockaddr_in*)saddr)->sin_port
                   : ((struct sockaddr_in6*)saddr)->sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static inline int env_socket_family(void) {
  int family = -1;  // Family selection is not forced, will use first one found
  char* env = getenv("NCCL_SOCKET_FAMILY");
  if (env == NULL) return family;

  LOG(INFO) << "NCCL_SOCKET_FAMILY set by environment to " << env;

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6;  // IPv6
  return family;
}

static int find_interfaces(char const* prefixList, char* names,
                           union socketAddress* addrs, int sock_family,
                           int maxIfNameSize, int maxIfs) {
  char line[SOCKET_NAME_MAXLEN + 1];
  struct ib_dev userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parse_interfaces(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    VLOG(3) << "Found interface " << interface->ifa_name << ":"
            << socket_to_string(interface->ifa_addr, line);

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family) continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    if (!(match_if_list(interface->ifa_name, -1, userIfs, nUserIfs,
                        searchExact) ^
          searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names + i * maxIfNameSize) == 0) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names + found * maxIfNameSize, interface->ifa_name,
              maxIfNameSize);
      // Store the IP address
      int salen =
          (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
      memcpy(addrs + found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

static bool match_subnet(struct ifaddrs local_if, union socketAddress* remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote->sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;                    // IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) {  // Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they
    // are not in the same scope For Global type, this field is 0, so a
    // comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    LOG(ERROR) << "Net : Unsupported address family type";
    return false;
  }
}

static int find_interface_match_subnet(char* ifNames,
                                       union socketAddress* localAddrs,
                                       union socketAddress* remoteAddr,
                                       int ifNameMaxSize, int maxIfs) {
  char line[SOCKET_NAME_MAXLEN + 1];
  char line_a[SOCKET_NAME_MAXLEN + 1];
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    // check against user specified interfaces
    if (!match_subnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen =
        (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
    memcpy(localAddrs + found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames + found * ifNameMaxSize, interface->ifa_name,
            ifNameMaxSize);

    VLOG(3) << "NET : Found interface " << interface->ifa_name << ":"
            << socket_to_string(&(localAddrs[found].sa), line)
            << " in the same subnet as remote address "
            << socket_to_string(&(remoteAddr->sa), line_a);
    found++;
    if (found == maxIfs) break;
  }

  if (found == 0) {
    LOG(ERROR)
        << "Net : No interface found in the same subnet as remote address "
        << socket_to_string(&(remoteAddr->sa), line_a);
  }
  freeifaddrs(interfaces);
  return found;
}

static bool get_socket_addr_from_string(union socketAddress* ua,
                                        char const* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    LOG(ERROR) << "Net : string is null";
    return false;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct ib_dev ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parse_interfaces(ip_port_pair, &ni, 1) != 1) {
      LOG(ERROR) << "Net : No valid <IPv4_or_hostname>:<port> pair found";
      return false;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      LOG(ERROR) << "Net : error encountered when getting address info : "
                 << gai_strerror(rv);
      return false;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;  // IPv4
      // inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);  // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;      // IPv6
      sin6.sin6_port = htons(ni.port);  // port
      sin6.sin6_flowinfo = 0;           // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;           // should be global scope, set to 0
    } else {
      LOG(ERROR) << "Net : unsupported IP family";
      return false;
    }

    freeaddrinfo(p);  // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      LOG(ERROR) << "Net : No valid [IPv6]:port pair found";
      return false;
    }
    bool global_scope =
        (j == -1
             ? true
             : false);  // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair + 1, global_scope ? i - 1 : j - 1);
    strncpy(port_str, ip_port_pair + i + 2, len - i - 1);
    int port = atoi(port_str);
    if (!global_scope)
      strncpy(if_name, ip_port_pair + j + 1,
              i - j - 1);  // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                     // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));  // IP address
    sin6.sin6_port = htons(port);                    // port
    sin6.sin6_flowinfo = 0;  // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id =
        global_scope
            ? 0
            : if_nametoindex(
                  if_name);  // 0 if global scope; intf index if link scope
  }
  return true;
}

static int find_interfaces(char* ifNames, union socketAddress* ifAddrs,
                           int ifNameMaxSize, int maxIfs) {
  static int shownIfName = 0;
  int nIfs = 0;
  // Allow user to force the INET socket family selection
  int sock_family = env_socket_family();
  // User specified interface
  char* env = getenv("UCCL_SOCKET_IFNAME");
  if (!env) env = getenv("NCCL_SOCKET_IFNAME");
  if (env && strlen(env) > 1) {
    LOG(INFO) << "UCCL/NCCL_SOCKET_IFNAME set by environment to " << env;
    // Specified by user : find or fail
    if (shownIfName++ == 0)
      LOG(INFO) << "UCCL/NCCL_SOCKET_IFNAME set to " << env;
    nIfs = find_interfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize,
                           maxIfs);
  } else {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = find_interfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                           maxIfs);
    // else see if we can get some hint from COMM ID
    if (nIfs == 0) {
      char* commId = getenv("NCCL_COMM_ID");
      if (commId && strlen(commId) > 1) {
        LOG(INFO) << "NCCL_COMM_ID set by environment to " << commId;
        // Try to find interface that is in the same subnet as the IP in comm id
        union socketAddress idAddr;
        get_socket_addr_from_string(&idAddr, commId);
        nIfs = find_interface_match_subnet(ifNames, ifAddrs, &idAddr,
                                           ifNameMaxSize, maxIfs);
      }
    }
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0)
      nIfs = find_interfaces("^docker,lo", ifNames, ifAddrs, sock_family,
                             ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0)
      nIfs = find_interfaces("docker", ifNames, ifAddrs, sock_family,
                             ifNameMaxSize, maxIfs);
    if (nIfs == 0)
      nIfs = find_interfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                             maxIfs);
  }
  return nIfs;
}

static bool parse_ip(std::string const& ip, sockaddr_storage* out,
                     socklen_t* outlen, int* family) {
  std::memset(out, 0, sizeof(*out));
  if (inet_pton(AF_INET, ip.c_str(), &((sockaddr_in*)out)->sin_addr) == 1) {
    *family = AF_INET;
    *outlen = sizeof(sockaddr_in);
    return true;
  }
  if (inet_pton(AF_INET6, ip.c_str(), &((sockaddr_in6*)out)->sin6_addr) == 1) {
    *family = AF_INET6;
    *outlen = sizeof(sockaddr_in6);
    return true;
  }
  return false;
}

static bool is_local_ip(std::string const& ip) {
  sockaddr_storage target;
  socklen_t tlen = 0;
  int tfam = 0;
  if (!parse_ip(ip, &target, &tlen, &tfam)) return false;

  ifaddrs* ifs = nullptr;
  if (getifaddrs(&ifs) != 0) return false;

  bool found = false;
  for (auto* it = ifs; it != nullptr; it = it->ifa_next) {
    if (!it->ifa_addr) continue;
    if (it->ifa_addr->sa_family != tfam) continue;

    if (tfam == AF_INET) {
      auto* a = (sockaddr_in*)it->ifa_addr;
      auto* b = (sockaddr_in*)&target;
      if (std::memcmp(&a->sin_addr, &b->sin_addr, sizeof(in_addr)) == 0) {
        found = true;
        break;
      }
    } else if (tfam == AF_INET6) {
      auto* a = (sockaddr_in6*)it->ifa_addr;
      auto* b = (sockaddr_in6*)&target;
      if (std::memcmp(&a->sin6_addr, &b->sin6_addr, sizeof(in6_addr)) == 0) {
        found = true;
        break;
      }
    }
  }
  freeifaddrs(ifs);
  return found;
}

}  // namespace uccl
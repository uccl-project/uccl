// Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "socket.h"
#include "debug.h"
#include "env.hpp"
#include "errors.hpp"
#include "utils.hpp"
#include "utils_internal.hpp"
#include <net/if.h>
#include <fstream>
#include <sstream>
#include <errno.h>
#include <ifaddrs.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

namespace mscclpp {

#define MSCCLPP_SOCKET_SEND 0
#define MSCCLPP_SOCKET_RECV 1

/* Format a string representation of a (union mscclppSocketAddress *) socket
 * address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
char const* SocketToString(union SocketAddress* addr, char* buf,
                           int const numericHostForm /*= 1*/) {
  if (buf == NULL || addr == NULL) return NULL;
  struct sockaddr* saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) {
    buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be
   * determined.)
   */
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  (void)getnameinfo(saddr, sizeof(union SocketAddress), host, NI_MAXHOST,
                    service, NI_MAXSERV, flag);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

// Equivalent with ($ cat /proc/sys/net/ipv4/tcp_fin_timeout)
static int getTcpFinTimeout() {
  std::ifstream ifs("/proc/sys/net/ipv4/tcp_fin_timeout");
  if (!ifs.is_open()) {
    throw SysError("open /proc/sys/net/ipv4/tcp_fin_timeout failed", errno);
  }
  int timeout;
  ifs >> timeout;
  return timeout;
}

static uint16_t socketToPort(union SocketAddress* addr) {
  struct sockaddr* saddr = &addr->sa;
  return ntohs(saddr->sa_family == AF_INET ? addr->sin.sin_port
                                           : addr->sin6.sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static int envSocketFamily(void) {
  int family = -1;  // Family selection is not forced, will use first one found
  std::string const& socketFamily = env()->socketFamily;
  if (socketFamily == "") return family;

  if (socketFamily == "AF_INET")
    family = AF_INET;  // IPv4
  else if (socketFamily == "AF_INET6")
    family = AF_INET6;  // IPv6
  return family;
}

static int findInterfaces(char const* prefixList, char* names,
                          union SocketAddress* addrs, int sock_family,
                          int maxIfNameSize, int maxIfs) {
#ifdef MSCCLPP_ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    TRACE(MSCCLPP_INIT | MSCCLPP_NET, "Found interface %s:%s",
          interface->ifa_name,
          SocketToString((union SocketAddress*)interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family) continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^
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
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                      : sizeof(struct sockaddr_in6);
      memcpy(addrs + found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

static bool matchSubnet(struct ifaddrs local_if, union SocketAddress* remote) {
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
    WARN("Net : Unsupported address family type");
    return false;
  }
}

int FindInterfaceMatchSubnet(char* ifNames, union SocketAddress* localAddrs,
                             union SocketAddress* remoteAddr, int ifNameMaxSize,
                             int maxIfs) {
#ifdef MSCCLPP_ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
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
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                    : sizeof(struct sockaddr_in6);
    memcpy(localAddrs + found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames + found * ifNameMaxSize, interface->ifa_name,
            ifNameMaxSize);

    TRACE(MSCCLPP_INIT | MSCCLPP_NET,
          "NET : Found interface %s:%s in the same subnet as remote address %s",
          interface->ifa_name, SocketToString(localAddrs + found, line),
          SocketToString(remoteAddr, line_a));
    found++;
    if (found == maxIfs) break;
  }

  if (found == 0) {
    WARN("Net : No interface found in the same subnet as remote address %s",
         SocketToString(remoteAddr, line_a));
  }
  freeifaddrs(interfaces);
  return found;
}

void SocketGetAddrFromString(union SocketAddress* ua,
                             char const* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    throw Error("Net : string is null", ErrorCode::InvalidUsage);
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      throw Error("Net : No valid <IPv4_or_hostname>:<port> pair found",
                  ErrorCode::InvalidUsage);
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      std::stringstream ss;
      ss << "Net : error encountered when getting address info : "
         << gai_strerror(rv);
      throw Error(ss.str(), ErrorCode::InvalidUsage);
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
      throw Error("Net : unsupported IP family", ErrorCode::InvalidUsage);
    }

    freeaddrinfo(p);  // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      throw Error("Net : No valid [IPv6]:port pair found",
                  ErrorCode::InvalidUsage);
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
}

int FindInterfaces(char* ifNames, union SocketAddress* ifAddrs,
                   int ifNameMaxSize, int maxIfs, char const* inputIfName) {
  static int shownIfName = 0;
  int nIfs = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  std::string const& socketIfname = env()->socketIfname;
  if (inputIfName) {
    INFO(MSCCLPP_NET, "using iterface %s", inputIfName);
    nIfs = findInterfaces(inputIfName, ifNames, ifAddrs, sock_family,
                          ifNameMaxSize, maxIfs);
  } else if (socketIfname != "") {
    // Specified by user : find or fail
    if (shownIfName++ == 0)
      INFO(MSCCLPP_NET, "MSCCLPP_SOCKET_IFNAME set to %s",
           socketIfname.c_str());
    nIfs = findInterfaces(socketIfname.c_str(), ifNames, ifAddrs, sock_family,
                          ifNameMaxSize, maxIfs);
  } else {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                          maxIfs);
    // else see if we can get some hint from COMM ID
    if (nIfs == 0) {
      std::string const& commId = env()->commId;
      if (commId != "") {
        // Try to find interface that is in the same subnet as the IP in comm id
        union SocketAddress idAddr;
        SocketGetAddrFromString(&idAddr, commId.c_str());
        nIfs = FindInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr,
                                        ifNameMaxSize, maxIfs);
      }
    }
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0)
      nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family,
                            ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0)
      nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family,
                            ifNameMaxSize, maxIfs);
    if (nIfs == 0)
      nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                            maxIfs);
  }
  return nIfs;
}

Socket::Socket(SocketAddress const* addr, uint64_t magic, enum SocketType type,
               uint32_t volatile* abortFlag, int asyncFlag) {
  fd_ = -1;
  acceptFd_ = -1;
  connectRetries_ = 0;
  acceptRetries_ = 0;
  abortFlag_ = abortFlag;
  asyncFlag_ = asyncFlag;
  state_ = SocketStateInitialized;
  magic_ = magic;
  type_ = type;

  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    memcpy(&addr_, addr, sizeof(union SocketAddress));
    family = addr_.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
      char line[SOCKET_NAME_MAXLEN + 1];
      std::stringstream ss;
      ss << "mscclppSocketInit: connecting to address "
         << SocketToString(&addr_, line) << " with family " << family
         << " is neither AF_INET(" << AF_INET << ") nor AF_INET6(" << AF_INET6
         << ")";
      throw Error(ss.str(), ErrorCode::InvalidUsage);
    }
    salen_ = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                 : sizeof(struct sockaddr_in6);

    /* Connect to a hostname / port */
    fd_ = ::socket(family, SOCK_STREAM, 0);
    if (fd_ == -1) {
      throw SysError("socket creation failed", errno);
    }
  } else {
    memset(&addr_, 0, sizeof(union SocketAddress));
  }

  /* Set socket as non-blocking if async or if we need to be able to abort */
  if ((asyncFlag_ || abortFlag_) && fd_ >= 0) {
    int flags = fcntl(fd_, F_GETFL);
    if (flags == -1) {
      throw SysError("fcntl(F_GETFL) failed", errno);
    }
    if (fcntl(fd_, F_SETFL, flags | O_NONBLOCK) == -1) {
      throw SysError("fcntl(F_SETFL) failed", errno);
    }
  }
}

Socket::~Socket() { close(); }

void Socket::bind() {
  if (fd_ == -1) {
    throw Error("file descriptor is -1", ErrorCode::InvalidUsage);
  }

  if (socketToPort(&addr_)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    if (::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                     sizeof(opt)) != 0) {
      throw SysError("::setsockopt(SO_REUSEADDR | SO_REUSEPORT) failed", errno);
    }
#else
    if (::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
      throw SysError("setsockopt(SO_REUSEADDR) failed", errno);
    }
#endif
  }

  int finTimeout = getTcpFinTimeout();
  int retrySecs = finTimeout + 1;
  int remainSecs = retrySecs;

  // addr port should be 0 (Any port)
  while (::bind(fd_, &addr_.sa, salen_) != 0) {
    // upon EADDRINUSE, retry up to for (finTimeout + 1) seconds
    if (errno != EADDRINUSE) {
      throw SysError("bind failed", errno);
    }
    if (remainSecs > 0) {
      INFO(MSCCLPP_INIT,
           "No available ephemeral ports found, will retry after 1 second");
      sleep(1);
      remainSecs--;
    } else {
      throw SysError("No available ephemeral ports found for " +
                         std::to_string(retrySecs) + " seconds",
                     errno);
    }
  }

  /* Get the assigned Port */
  socklen_t size = salen_;
  if (::getsockname(fd_, &addr_.sa, &size) != 0) {
    throw SysError("getsockname failed", errno);
  }
  state_ = SocketStateBound;
}

void Socket::bindAndListen() {
  bind();
#ifdef MSCCLPP_ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
  TRACE(MSCCLPP_INIT | MSCCLPP_NET, "Listening on socket %s",
        SocketToString(&addr_, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in
   * /proc/sys/net/core/somaxconn
   */
  if (::listen(fd_, 16384) != 0) {
    throw SysError("listen failed", errno);
  }
  state_ = SocketStateReady;
}

void Socket::connect(int64_t timeout) {
  Timer timer;
#ifdef MSCCLPP_ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  int const one = 1;

  if (fd_ == -1) {
    throw Error("file descriptor is -1", ErrorCode::InvalidUsage);
  }

  if (state_ != SocketStateInitialized) {
    std::stringstream ss;
    ss << "wrong socket state " << state_;
    if (state_ == SocketStateError)
      throw Error(ss.str(), ErrorCode::RemoteError);
    throw Error(ss.str(), ErrorCode::InternalError);
  }
  TRACE(MSCCLPP_INIT | MSCCLPP_NET, "Connecting to socket %s",
        SocketToString(&addr_, line));

  if (setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)) !=
      0) {
    throw SysError("setsockopt(TCP_NODELAY) failed", errno);
  }

  state_ = SocketStateConnecting;
  do {
    progressState();
    if (timeout > 0 && timer.elapsed() > timeout) {
      throw Error("connect timeout", ErrorCode::Timeout);
    }
  } while (asyncFlag_ == 0 && (abortFlag_ == NULL || *abortFlag_ == 0) &&
           (state_ == SocketStateConnecting ||
            state_ == SocketStateConnectPolling ||
            state_ == SocketStateConnected));

  if (abortFlag_ && *abortFlag_ != 0)
    throw Error("aborted", ErrorCode::Aborted);
}

void Socket::accept(Socket const* listenSocket, int64_t timeout) {
  Timer timer;

  if (listenSocket == NULL) {
    throw Error("listenSocket is NULL", ErrorCode::InvalidUsage);
  }
  if (listenSocket->getState() != SocketStateReady) {
    throw Error("listenSocket is in error state " +
                    std::to_string(listenSocket->getState()),
                ErrorCode::InternalError);
  }

  if (acceptFd_ == -1) {
    fd_ = listenSocket->getFd();
    connectRetries_ = listenSocket->getConnectRetries();
    acceptRetries_ = listenSocket->getAcceptRetries();
    abortFlag_ = listenSocket->getAbortFlag();
    asyncFlag_ = listenSocket->getAsyncFlag();
    magic_ = listenSocket->getMagic();
    type_ = listenSocket->getType();
    addr_ = listenSocket->getAddr();
    salen_ = listenSocket->getSalen();

    acceptFd_ = listenSocket->getFd();
    state_ = SocketStateAccepting;
  }

  do {
    progressState();
    if (timeout > 0 && timer.elapsed() > timeout) {
      throw Error("accept timeout", ErrorCode::Timeout);
    }
  } while (asyncFlag_ == 0 && (abortFlag_ == NULL || *abortFlag_ == 0) &&
           (state_ == SocketStateAccepting || state_ == SocketStateAccepted));

  if (abortFlag_ && *abortFlag_ != 0)
    throw Error("aborted", ErrorCode::Aborted);
}

void Socket::send(void* ptr, int size) {
  int offset = 0;
  if (state_ != SocketStateReady) {
    std::stringstream ss;
    ss << "socket state (" << state_ << ") is not ready";
    throw Error(ss.str(), ErrorCode::InternalError);
  }
  socketWait(MSCCLPP_SOCKET_SEND, ptr, size, &offset);
}

void Socket::recv(void* ptr, int size) {
  int offset = 0;
  if (state_ != SocketStateReady) {
    std::stringstream ss;
    ss << "socket state (" << state_ << ") is not ready";
    throw Error(ss.str(), ErrorCode::InternalError);
  }
  socketWait(MSCCLPP_SOCKET_RECV, ptr, size, &offset);
}

void Socket::recvUntilEnd(void* ptr, int size, int* closed) {
  int offset = 0;
  *closed = 0;
  if (state_ != SocketStateReady) {
    std::stringstream ss;
    ss << "socket state (" << state_ << ") is not ready in recvUntilEnd";
    throw Error(ss.str(), ErrorCode::InternalError);
  }

  int bytes = 0;
  char* data = (char*)ptr;

  do {
    bytes = ::recv(fd_, data + (offset), size - (offset), 0);
    if (bytes == 0) {
      *closed = 1;
      return;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN &&
          state_ != SocketStateClosed) {
        throw SysError("recv until end failed", errno);
      } else {
        bytes = 0;
      }
    }
    (offset) += bytes;
    if (abortFlag_ && *abortFlag_ != 0) {
      throw Error("aborted", ErrorCode::Aborted);
    }
  } while (bytes > 0 && (offset) < size);
}

void Socket::close() {
  if (fd_ >= 0) ::close(fd_);
  state_ = SocketStateClosed;
  fd_ = -1;
}

void Socket::progressState() {
  if (state_ == SocketStateAccepting) {
    tryAccept();
  }
  if (state_ == SocketStateAccepted) {
    finalizeAccept();
  }
  if (state_ == SocketStateConnecting) {
    startConnect();
  }
  if (state_ == SocketStateConnectPolling) {
    pollConnect();
  }
  if (state_ == SocketStateConnected) {
    finalizeConnect();
  }
}

void Socket::tryAccept() {
  socklen_t socklen = sizeof(union SocketAddress);
  fd_ = ::accept(acceptFd_, &addr_.sa, &socklen);
  if (fd_ != -1) {
    state_ = SocketStateAccepted;
  } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
    std::stringstream ss;
    ss << "accept failed (fd " << acceptFd_ << ")";
    throw SysError(ss.str(), errno);
  } else {
    usleep(SLEEP_INT);
    if (++acceptRetries_ % 1000 == 0)
      INFO(MSCCLPP_ALL, "tryAccept: Call to try accept returned %s, retrying",
           strerror(errno));
  }
}

void Socket::finalizeAccept() {
  uint64_t magic;
  enum SocketType type;
  int received = 0;
  socketProgress(MSCCLPP_SOCKET_RECV, &magic, sizeof(magic), &received);
  if (received == 0) return;
  socketWait(MSCCLPP_SOCKET_RECV, &magic, sizeof(magic), &received);
  if (magic != magic_) {
    WARN("finalizeAccept: wrong magic %lx != %lx", magic, magic_);
    ::close(fd_);
    fd_ = -1;
    // Ignore spurious connection and accept again
    state_ = SocketStateAccepting;
    return;
  } else {
    received = 0;
    socketWait(MSCCLPP_SOCKET_RECV, &type, sizeof(type), &received);
    if (type != type_) {
      state_ = SocketStateError;
      ::close(fd_);
      fd_ = -1;
      std::stringstream ss;
      ss << "wrong socket type " << type << " != " << type_;
      throw Error(ss.str(), ErrorCode::InternalError);
    } else {
      state_ = SocketStateReady;
    }
  }
}

void Socket::startConnect() {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = ::connect(fd_, &addr_.sa, salen_);
  if (ret == 0) {
    state_ = SocketStateConnected;
    return;
  } else if (errno == EINPROGRESS) {
    state_ = SocketStateConnectPolling;
    return;
  } else if (errno == ECONNREFUSED || errno == ETIMEDOUT) {
    usleep(SLEEP_INT);
    if (++connectRetries_ % 1000 == 0)
      INFO(MSCCLPP_ALL, "Call to connect returned %s, retrying",
           strerror(errno));
    return;
  } else {
    char line[SOCKET_NAME_MAXLEN + 1];
    state_ = SocketStateError;
    std::stringstream ss;
    ss << "connect to " << SocketToString(&addr_, line) << " failed";
    throw SysError(ss.str(), errno);
  }
}

void Socket::pollConnect() {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = fd_;
  pfd.events = POLLOUT;
  ret = ::poll(&pfd, 1, timeout);
  if (ret == -1) throw SysError("poll failed", errno);
  if (ret == 0) return;

  /* check socket status */
  if ((ret == 1 && (pfd.revents & POLLOUT)) == 0) {
    throw Error("poll failed", ErrorCode::InternalError);
  }
  if (getsockopt(fd_, SOL_SOCKET, SO_ERROR, (void*)&ret, &rlen) == -1) {
    throw SysError("getsockopt failed", errno);
  }

  if (ret == 0) {
    state_ = SocketStateConnected;
  } else if (ret == ECONNREFUSED || ret == ETIMEDOUT) {
    if (++connectRetries_ % 1000 == 0) {
      INFO(MSCCLPP_ALL, "Call to connect returned %s, retrying",
           strerror(errno));
    }
    usleep(SLEEP_INT);

    ::close(fd_);
    fd_ = ::socket(addr_.sa.sa_family, SOCK_STREAM, 0);
    state_ = SocketStateConnecting;
  } else if (ret != EINPROGRESS) {
    state_ = SocketStateError;
    throw Error("connect failed", ErrorCode::SystemError);
  }
}

void Socket::finalizeConnect() {
  int sent = 0;
  socketProgress(MSCCLPP_SOCKET_SEND, &magic_, sizeof(magic_), &sent);
  if (sent == 0) return;
  socketWait(MSCCLPP_SOCKET_SEND, &magic_, sizeof(magic_), &sent);
  sent = 0;
  socketWait(MSCCLPP_SOCKET_SEND, &type_, sizeof(type_), &sent);
  state_ = SocketStateReady;
}

void Socket::socketProgressOpt(int op, void* ptr, int size, int* offset,
                               int block, int* closed) {
  int bytes = 0;
  *closed = 0;
  char* data = (char*)ptr;

  do {
    if (op == MSCCLPP_SOCKET_RECV)
      bytes = ::recv(fd_, data + (*offset), size - (*offset),
                     block ? 0 : MSG_DONTWAIT);
    if (op == MSCCLPP_SOCKET_SEND)
      bytes = ::send(fd_, data + (*offset), size - (*offset),
                     block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);
    if (op == MSCCLPP_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        throw SysError("recv failed", errno);
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
    if (abortFlag_ && *abortFlag_ != 0) {
      throw Error("aborted", ErrorCode::Aborted);
    }
  } while (bytes > 0 && (*offset) < size);
}

void Socket::socketProgress(int op, void* ptr, int size, int* offset) {
  int closed;
  socketProgressOpt(op, ptr, size, offset, 0, &closed);
  if (closed) {
    char line[SOCKET_NAME_MAXLEN + 1];
    throw Error("connection closed by remote peer " +
                    std::string(SocketToString(&addr_, line, 0)),
                ErrorCode::RemoteError);
  }
}

void Socket::socketWait(int op, void* ptr, int size, int* offset) {
  while (*offset < size) socketProgress(op, ptr, size, offset);
}

}  // namespace mscclpp

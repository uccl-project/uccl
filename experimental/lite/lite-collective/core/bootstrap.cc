// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "api.h"
#include "core.hpp"
#include "debug.h"
#include "errors.hpp"
#include "socket.h"
#include "unix_socket.hpp"
#include "utils_internal.hpp"
#include <cstring>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/resource.h>

namespace mscclpp {

static void setFilesLimit() {
  rlimit filesLimit;
  if (getrlimit(RLIMIT_NOFILE, &filesLimit) != 0)
    throw SysError("getrlimit failed", errno);
  filesLimit.rlim_cur = filesLimit.rlim_max;
  if (setrlimit(RLIMIT_NOFILE, &filesLimit) != 0)
    throw SysError("setrlimit failed", errno);
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

struct ExtInfo {
  int rank;
  int nRanks;
  SocketAddress extAddressListenRoot;
  SocketAddress extAddressListen;
};

MSCCLPP_API_CPP void Bootstrap::groupBarrier(std::vector<int> const& ranks) {
  int dummy = 0;
  for (auto rank : ranks) {
    if (rank != this->getRank()) {
      this->send(static_cast<void*>(&dummy), sizeof(dummy), rank, 0);
    }
  }
  for (auto rank : ranks) {
    if (rank != this->getRank()) {
      this->recv(static_cast<void*>(&dummy), sizeof(dummy), rank, 0);
    }
  }
}

MSCCLPP_API_CPP void Bootstrap::send(std::vector<char> const& data, int peer,
                                     int tag) {
  size_t size = data.size();
  send((void*)&size, sizeof(size_t), peer, tag);
  send((void*)data.data(), data.size(), peer, tag);
}

MSCCLPP_API_CPP void Bootstrap::recv(std::vector<char>& data, int peer,
                                     int tag) {
  size_t size;
  recv((void*)&size, sizeof(size_t), peer, tag);
  data.resize(size);
  recv((void*)data.data(), data.size(), peer, tag);
}

struct UniqueIdInternal {
  uint64_t magic;
  union SocketAddress addr;
};
static_assert(sizeof(UniqueIdInternal) <= sizeof(UniqueId),
              "UniqueIdInternal is too large to fit into UniqueId");

class TcpBootstrap::Impl {
 public:
  static UniqueId createUniqueId();
  static UniqueId getUniqueId(UniqueIdInternal const& uniqueId);

  Impl(int rank, int nRanks);
  ~Impl();
  void initialize(UniqueId const& uniqueId, int64_t timeoutSec);
  void initialize(std::string const& ifIpPortTrio, int64_t timeoutSec);
  void establishConnections(int64_t timeoutSec);
  UniqueId getUniqueId() const;
  int getRank();
  int getNranks();
  int getNranksPerNode();
  void allGather(void* allData, int size);
  void broadcast(void* data, int size, int root);
  void send(void* data, int size, int peer, int tag);
  void recv(void* data, int size, int peer, int tag);
  void barrier();
  void close();

 private:
  UniqueIdInternal uniqueId_;
  int rank_;
  int nRanks_;
  int nRanksPerNode_;
  bool netInitialized;
  std::unique_ptr<Socket> listenSockRoot_;
  std::unique_ptr<Socket> listenSock_;
  std::unique_ptr<Socket> ringRecvSocket_;
  std::unique_ptr<Socket> ringSendSocket_;
  std::vector<SocketAddress> peerCommAddresses_;
  std::vector<int> barrierArr_;
  std::unique_ptr<uint32_t> abortFlagStorage_;
  uint32_t volatile* abortFlag_;
  std::thread rootThread_;
  SocketAddress netIfAddr_;
  std::unordered_map<std::pair<int, int>, std::shared_ptr<Socket>, PairHash>
      peerSendSockets_;
  std::unordered_map<std::pair<int, int>, std::shared_ptr<Socket>, PairHash>
      peerRecvSockets_;

  void netSend(Socket* sock, void const* data, int size);
  void netRecv(Socket* sock, void* data, int size);

  std::shared_ptr<Socket> getPeerSendSocket(int peer, int tag);
  std::shared_ptr<Socket> getPeerRecvSocket(int peer, int tag);

  UnixSocketServer& unixSocketServer_;

  static void assignPortToUniqueId(UniqueIdInternal& uniqueId);
  static void netInit(std::string ipPortPair, std::string interface,
                      SocketAddress& netIfAddr);

  void bootstrapCreateRoot();
  void bootstrapRoot();
  void getRemoteAddresses(Socket* listenSock,
                          std::vector<SocketAddress>& rankAddresses,
                          std::vector<SocketAddress>& rankAddressesRoot,
                          int& rank);
  void sendHandleToPeer(int peer,
                        std::vector<SocketAddress> const& rankAddresses,
                        std::vector<SocketAddress> const& rankAddressesRoot);
};

UniqueId TcpBootstrap::Impl::createUniqueId() {
  UniqueIdInternal uniqueId;
  SocketAddress netIfAddr;
  netInit("", "", netIfAddr);
  getRandomData(&uniqueId.magic, sizeof(uniqueId_.magic));
  std::memcpy(&uniqueId.addr, &netIfAddr, sizeof(SocketAddress));
  assignPortToUniqueId(uniqueId);
  return getUniqueId(uniqueId);
}

UniqueId TcpBootstrap::Impl::getUniqueId(UniqueIdInternal const& uniqueId) {
  UniqueId ret;
  std::memcpy(&ret, &uniqueId, sizeof(uniqueId));
  return ret;
}

TcpBootstrap::Impl::Impl(int rank, int nRanks)
    : rank_(rank),
      nRanks_(nRanks),
      nRanksPerNode_(0),
      netInitialized(false),
      peerCommAddresses_(nRanks, SocketAddress()),
      barrierArr_(nRanks, 0),
      abortFlagStorage_(new uint32_t(0)),
      abortFlag_(abortFlagStorage_.get()),
      unixSocketServer_(UnixSocketServer::instance()) {}

UniqueId TcpBootstrap::Impl::getUniqueId() const {
  return getUniqueId(uniqueId_);
}

int TcpBootstrap::Impl::getRank() { return rank_; }

int TcpBootstrap::Impl::getNranks() { return nRanks_; }

void TcpBootstrap::Impl::initialize(UniqueId const& uniqueId,
                                    int64_t timeoutSec) {
  if (!netInitialized) {
    netInit("", "", netIfAddr_);
    netInitialized = true;
  }

  std::memcpy(&uniqueId_, &uniqueId, sizeof(uniqueId_));
  if (rank_ == 0) {
    bootstrapCreateRoot();
  }

  char line[MAX_IF_NAME_SIZE + 1];
  SocketToString(&uniqueId_.addr, line);
  INFO(MSCCLPP_INIT, "rank %d nranks %d - connecting to %s", rank_, nRanks_,
       line);
  establishConnections(timeoutSec);

  unixSocketServer_.start();
  INFO(MSCCLPP_INIT, "rank %d - unix socket server started", rank_);
}

void TcpBootstrap::Impl::initialize(std::string const& ifIpPortTrio,
                                    int64_t timeoutSec) {
  // first check if it is a trio
  int nColons = 0;
  for (auto c : ifIpPortTrio) {
    if (c == ':') {
      nColons++;
    }
  }
  std::string ipPortPair = ifIpPortTrio;
  std::string interface = "";
  if (nColons == 2) {
    // we know the <interface>
    interface = ifIpPortTrio.substr(0, ipPortPair.find_first_of(':'));
    ipPortPair = ifIpPortTrio.substr(ipPortPair.find_first_of(':') + 1);
  }

  if (!netInitialized) {
    netInit(ipPortPair, interface, netIfAddr_);
    netInitialized = true;
  }

  uniqueId_.magic = 0xdeadbeef;
  std::memcpy(&uniqueId_.addr, &netIfAddr_, sizeof(SocketAddress));
  SocketGetAddrFromString(&uniqueId_.addr, ipPortPair.c_str());

  if (rank_ == 0) {
    bootstrapCreateRoot();
  }

  establishConnections(timeoutSec);
  unixSocketServer_.start();
  INFO(MSCCLPP_INIT, "rank %d - unix socket server started", rank_);
}

TcpBootstrap::Impl::~Impl() {
  if (abortFlag_) {
    *abortFlag_ = 1;
  }
  if (rootThread_.joinable()) {
    rootThread_.join();
  }
}

void TcpBootstrap::Impl::getRemoteAddresses(
    Socket* listenSock, std::vector<SocketAddress>& rankAddresses,
    std::vector<SocketAddress>& rankAddressesRoot, int& rank) {
  ExtInfo info;
  SocketAddress zero;
  std::memset(&zero, 0, sizeof(SocketAddress));

  {
    Socket sock(nullptr, MSCCLPP_SOCKET_MAGIC, SocketTypeUnknown, abortFlag_);
    sock.accept(listenSock);
    netRecv(&sock, &info, sizeof(info));
  }

  if (this->nRanks_ != info.nRanks) {
    throw Error("Bootstrap Root : mismatch in rank count from procs " +
                    std::to_string(this->nRanks_) + " : " +
                    std::to_string(info.nRanks),
                ErrorCode::InternalError);
  }

  if (std::memcmp(&zero, &rankAddressesRoot[info.rank],
                  sizeof(SocketAddress)) != 0) {
    throw Error("Bootstrap Root : rank " + std::to_string(info.rank) + " of " +
                    std::to_string(this->nRanks_) + " has already checked in",
                ErrorCode::InternalError);
  }

  // Save the connection handle for that rank
  rankAddressesRoot[info.rank] = info.extAddressListenRoot;
  rankAddresses[info.rank] = info.extAddressListen;
  rank = info.rank;
}

void TcpBootstrap::Impl::sendHandleToPeer(
    int peer, std::vector<SocketAddress> const& rankAddresses,
    std::vector<SocketAddress> const& rankAddressesRoot) {
  int next = (peer + 1) % nRanks_;
  Socket sock(&rankAddressesRoot[peer], uniqueId_.magic, SocketTypeBootstrap,
              abortFlag_);
  sock.connect();
  netSend(&sock, &rankAddresses[next], sizeof(SocketAddress));
}

void TcpBootstrap::Impl::assignPortToUniqueId(UniqueIdInternal& uniqueId) {
  std::unique_ptr<Socket> socket = std::make_unique<Socket>(
      &uniqueId.addr, uniqueId.magic, SocketTypeBootstrap);
  socket->bind();
  uniqueId.addr = socket->getAddr();
}

void TcpBootstrap::Impl::bootstrapCreateRoot() {
  listenSockRoot_ = std::make_unique<Socket>(
      &uniqueId_.addr, uniqueId_.magic, SocketTypeBootstrap, abortFlag_, 0);
  listenSockRoot_->bindAndListen();
  uniqueId_.addr = listenSockRoot_->getAddr();

  rootThread_ = std::thread([this]() {
    try {
      bootstrapRoot();
    } catch (const std::exception& e) {
      if (abortFlag_ && *abortFlag_) return;
      throw e;
    }
  });
}

void TcpBootstrap::Impl::bootstrapRoot() {
  int numCollected = 0;
  std::vector<SocketAddress> rankAddresses(nRanks_, SocketAddress());
  // for initial rank <-> root information exchange
  std::vector<SocketAddress> rankAddressesRoot(nRanks_, SocketAddress());

  std::memset(rankAddresses.data(), 0, sizeof(SocketAddress) * nRanks_);
  std::memset(rankAddressesRoot.data(), 0, sizeof(SocketAddress) * nRanks_);
  setFilesLimit();

  TRACE(MSCCLPP_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    int rank;
    getRemoteAddresses(listenSockRoot_.get(), rankAddresses, rankAddressesRoot,
                       rank);
    ++numCollected;
    TRACE(MSCCLPP_INIT, "Received connect from rank %d total %d/%d", rank,
          numCollected, nRanks_);
  } while (numCollected < nRanks_ && (!abortFlag_ || *abortFlag_ == 0));

  if (abortFlag_ && *abortFlag_) {
    TRACE(MSCCLPP_INIT, "ABORTED");
    return;
  }

  TRACE(MSCCLPP_INIT, "COLLECTED ALL %d HANDLES", nRanks_);

  // Send the connect handle for the next rank in the AllGather ring
  for (int peer = 0; peer < nRanks_; ++peer) {
    sendHandleToPeer(peer, rankAddresses, rankAddressesRoot);
  }

  TRACE(MSCCLPP_INIT, "DONE");
}

void TcpBootstrap::Impl::netInit(std::string ipPortPair, std::string interface,
                                 SocketAddress& netIfAddr) {
  char netIfName[MAX_IF_NAME_SIZE + 1];
  if (!ipPortPair.empty()) {
    if (interface != "") {
      // we know the <interface>
      int ret = FindInterfaces(netIfName, &netIfAddr, MAX_IF_NAME_SIZE, 1,
                               interface.c_str());
      if (ret <= 0)
        throw Error("NET/Socket : No interface named " + interface + " found.",
                    ErrorCode::InternalError);
    } else {
      // we do not know the <interface> try to match it next
      SocketAddress remoteAddr;
      SocketGetAddrFromString(&remoteAddr, ipPortPair.c_str());
      if (FindInterfaceMatchSubnet(netIfName, &netIfAddr, &remoteAddr,
                                   MAX_IF_NAME_SIZE, 1) <= 0) {
        throw Error("NET/Socket : No usable listening interface found",
                    ErrorCode::InternalError);
      }
    }

  } else {
    int ret = FindInterfaces(netIfName, &netIfAddr, MAX_IF_NAME_SIZE, 1);
    if (ret <= 0) {
      throw Error("TcpBootstrap : no socket interface found",
                  ErrorCode::InternalError);
    }
  }

  char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
  std::sprintf(line, " %s:", netIfName);
  SocketToString(&netIfAddr, line + strlen(line));
  INFO(MSCCLPP_INIT, "TcpBootstrap : Using%s", line);
}

#define TIMEOUT(__exp)                                                      \
  do {                                                                      \
    try {                                                                   \
      __exp;                                                                \
    } catch (Error const& e) {                                              \
      if (e.getErrorCode() == ErrorCode::Timeout) {                         \
        throw Error("TcpBootstrap connection timeout", ErrorCode::Timeout); \
      }                                                                     \
      throw;                                                                \
    }                                                                       \
  } while (0);

void TcpBootstrap::Impl::establishConnections(int64_t timeoutSec) {
  const int64_t connectionTimeoutUs = timeoutSec * 1000000;
  Timer timer;
  SocketAddress nextAddr;
  ExtInfo info;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d", rank_, nRanks_);

  auto getLeftTime = [&]() {
    if (connectionTimeoutUs < 0) {
      // no timeout: always return a large number
      return int64_t(1e9);
    }
    int64_t timeout = connectionTimeoutUs - timer.elapsed();
    if (timeout <= 0)
      throw Error("TcpBootstrap connection timeout", ErrorCode::Timeout);
    return timeout;
  };

  info.rank = rank_;
  info.nRanks = nRanks_;

  uint64_t magic = uniqueId_.magic;
  // Create socket for other ranks to contact me
  listenSock_ = std::make_unique<Socket>(&netIfAddr_, magic,
                                         SocketTypeBootstrap, abortFlag_);
  listenSock_->bindAndListen();
  info.extAddressListen = listenSock_->getAddr();

  {
    // Create socket for root to contact me
    Socket lsock(&netIfAddr_, magic, SocketTypeBootstrap, abortFlag_);
    lsock.bindAndListen();
    info.extAddressListenRoot = lsock.getAddr();

    // stagger connection times to avoid an overload of the root
    auto randomSleep = [](int rank) {
      timespec tv;
      tv.tv_sec = rank / 1000;
      tv.tv_nsec = 1000000 * (rank % 1000);
      TRACE(MSCCLPP_INIT, "rank %d delaying connection to root by %ld msec",
            rank, rank);
      (void)nanosleep(&tv, NULL);
    };
    if (nRanks_ > 128) {
      randomSleep(rank_);
    }

    // send info on my listening socket to root
    {
      Socket sock(&uniqueId_.addr, magic, SocketTypeBootstrap, abortFlag_);
      TIMEOUT(sock.connect(getLeftTime()));
      netSend(&sock, &info, sizeof(info));
    }

    // get info on my "next" rank in the bootstrap ring from root
    {
      Socket sock(nullptr, MSCCLPP_SOCKET_MAGIC, SocketTypeUnknown, abortFlag_);
      TIMEOUT(sock.accept(&lsock, getLeftTime()));
      netRecv(&sock, &nextAddr, sizeof(SocketAddress));
    }
  }

  ringSendSocket_ = std::make_unique<Socket>(&nextAddr, magic,
                                             SocketTypeBootstrap, abortFlag_);
  TIMEOUT(ringSendSocket_->connect(getLeftTime()));
  // Accept the connect request from the previous rank in the AllGather ring
  ringRecvSocket_ = std::make_unique<Socket>(nullptr, MSCCLPP_SOCKET_MAGIC,
                                             SocketTypeUnknown, abortFlag_);
  TIMEOUT(ringRecvSocket_->accept(listenSock_.get(), getLeftTime()));

  // AllGather all listen handlers
  peerCommAddresses_[rank_] = listenSock_->getAddr();
  allGather(peerCommAddresses_.data(), sizeof(SocketAddress));

  TRACE(MSCCLPP_INIT, "rank %d nranks %d - DONE", rank_, nRanks_);
}

int TcpBootstrap::Impl::getNranksPerNode() {
  if (nRanksPerNode_ > 0) return nRanksPerNode_;
  int nRanksPerNode = 0;
  bool useIpv4 = peerCommAddresses_[rank_].sa.sa_family == AF_INET;
  for (int i = 0; i < nRanks_; i++) {
    if (useIpv4) {
      if (peerCommAddresses_[i].sin.sin_addr.s_addr ==
          peerCommAddresses_[rank_].sin.sin_addr.s_addr) {
        nRanksPerNode++;
      }
    } else {
      if (std::memcmp(&(peerCommAddresses_[i].sin6.sin6_addr),
                      &(peerCommAddresses_[rank_].sin6.sin6_addr),
                      sizeof(in6_addr)) == 0) {
        nRanksPerNode++;
      }
    }
  }
  nRanksPerNode_ = nRanksPerNode;
  return nRanksPerNode_;
}

void TcpBootstrap::Impl::allGather(void* allData, int size) {
  char* data = static_cast<char*>(allData);
  int rank = rank_;
  int nRanks = nRanks_;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d", rank, nRanks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i = 0; i < nRanks - 1; i++) {
    size_t rSlice = (rank - i - 1 + nRanks) % nRanks;
    size_t sSlice = (rank - i + nRanks) % nRanks;

    // Send slice to the right
    netSend(ringSendSocket_.get(), data + sSlice * size, size);
    // Recv slice from the left
    netRecv(ringRecvSocket_.get(), data + rSlice * size, size);
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nRanks, size);
}

void TcpBootstrap::Impl::broadcast(void* data, int size, int root) {
  int rank = rank_;
  int nRanks = nRanks_;

  if (nRanks == 1) return;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d root %d size %d", rank, nRanks, root,
        size);

  /*Ring-based broadcast, propagate data in nRanks-1 steps*/

  // For ring-based broadcast, propagate data in nRanks-1 steps
  for (int step = 0; step < nRanks - 1; step++) {
    if (rank == root) {
      // Root sends data to next rank in first step
      if (step == 0) {
        netSend(ringSendSocket_.get(), data, size);
      }
    } else {
      // Calculate when this rank should receive data
      int receiveStep = (rank - root - 1 + nRanks) % nRanks;
      if (step == receiveStep) {
        // Receive from previous rank
        netRecv(ringRecvSocket_.get(), data, size);
        // Forward to next rank (if not last step)
        if (step < nRanks - 2) {
          netSend(ringSendSocket_.get(), data, size);
        }
      }
    }
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d root %d size %d - DONE", rank, nRanks,
        root, size);
}

std::shared_ptr<Socket> TcpBootstrap::Impl::getPeerSendSocket(int peer,
                                                              int tag) {
  auto it = peerSendSockets_.find(std::make_pair(peer, tag));
  if (it != peerSendSockets_.end()) {
    return it->second;
  }
  auto sock =
      std::make_shared<Socket>(&peerCommAddresses_[peer], uniqueId_.magic,
                               SocketTypeBootstrap, abortFlag_);
  sock->connect();
  netSend(sock.get(), &rank_, sizeof(int));
  netSend(sock.get(), &tag, sizeof(int));
  peerSendSockets_[std::make_pair(peer, tag)] = sock;
  return sock;
}

std::shared_ptr<Socket> TcpBootstrap::Impl::getPeerRecvSocket(int peer,
                                                              int tag) {
  auto it = peerRecvSockets_.find(std::make_pair(peer, tag));
  if (it != peerRecvSockets_.end()) {
    return it->second;
  }
  for (;;) {
    auto sock = std::make_shared<Socket>(nullptr, MSCCLPP_SOCKET_MAGIC,
                                         SocketTypeUnknown, abortFlag_);
    sock->accept(listenSock_.get());
    int recvPeer, recvTag;
    netRecv(sock.get(), &recvPeer, sizeof(int));
    netRecv(sock.get(), &recvTag, sizeof(int));
    peerRecvSockets_[std::make_pair(recvPeer, recvTag)] = sock;
    if (recvPeer == peer && recvTag == tag) {
      return sock;
    }
    // TODO(chhwang): set an exit condition or timeout
  }
}

void TcpBootstrap::Impl::netSend(Socket* sock, void const* data, int size) {
  sock->send(&size, sizeof(int));
  sock->send(const_cast<void*>(data), size);
}

void TcpBootstrap::Impl::netRecv(Socket* sock, void* data, int size) {
  int recvSize;
  sock->recv(&recvSize, sizeof(int));
  if (recvSize > size) {
    std::stringstream ss;
    ss << "Message truncated : received " << recvSize << " bytes instead of "
       << size;
    throw Error(ss.str(), ErrorCode::InvalidUsage);
  }
  sock->recv(data, std::min(recvSize, size));
}

void TcpBootstrap::Impl::send(void* data, int size, int peer, int tag) {
  auto sock = getPeerSendSocket(peer, tag);
  netSend(sock.get(), data, size);
}

void TcpBootstrap::Impl::recv(void* data, int size, int peer, int tag) {
  auto sock = getPeerRecvSocket(peer, tag);
  netRecv(sock.get(), data, size);
}

void TcpBootstrap::Impl::barrier() {
  allGather(barrierArr_.data(), sizeof(int));
}

void TcpBootstrap::Impl::close() {
  listenSockRoot_.reset(nullptr);
  listenSock_.reset(nullptr);
  ringRecvSocket_.reset(nullptr);
  ringSendSocket_.reset(nullptr);
  peerSendSockets_.clear();
  peerRecvSockets_.clear();
  unixSocketServer_.stop();
  UnixSocketClient::instance().reset();
}

MSCCLPP_API_CPP UniqueId TcpBootstrap::createUniqueId() {
  return Impl::createUniqueId();
}

MSCCLPP_API_CPP TcpBootstrap::TcpBootstrap(int rank, int nRanks) {
  pimpl_ = std::make_unique<Impl>(rank, nRanks);
}

MSCCLPP_API_CPP UniqueId TcpBootstrap::getUniqueId() const {
  return pimpl_->getUniqueId();
}

MSCCLPP_API_CPP int TcpBootstrap::getRank() const { return pimpl_->getRank(); }

MSCCLPP_API_CPP int TcpBootstrap::getNranks() const {
  return pimpl_->getNranks();
}

MSCCLPP_API_CPP int TcpBootstrap::getNranksPerNode() const {
  return pimpl_->getNranksPerNode();
}

MSCCLPP_API_CPP void TcpBootstrap::send(void* data, int size, int peer,
                                        int tag) {
  pimpl_->send(data, size, peer, tag);
}

MSCCLPP_API_CPP void TcpBootstrap::recv(void* data, int size, int peer,
                                        int tag) {
  pimpl_->recv(data, size, peer, tag);
}

MSCCLPP_API_CPP void TcpBootstrap::allGather(void* allData, int size) {
  pimpl_->allGather(allData, size);
}

MSCCLPP_API_CPP void TcpBootstrap::broadcast(void* data, int size, int root) {
  pimpl_->broadcast(data, size, root);
}

MSCCLPP_API_CPP void TcpBootstrap::initialize(UniqueId uniqueId,
                                              int64_t timeoutSec) {
  pimpl_->initialize(uniqueId, timeoutSec);
}

MSCCLPP_API_CPP void TcpBootstrap::initialize(std::string const& ipPortPair,
                                              int64_t timeoutSec) {
  pimpl_->initialize(ipPortPair, timeoutSec);
}

MSCCLPP_API_CPP void TcpBootstrap::barrier() { pimpl_->barrier(); }

MSCCLPP_API_CPP TcpBootstrap::~TcpBootstrap() { pimpl_->close(); }

}  // namespace mscclpp

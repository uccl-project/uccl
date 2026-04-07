// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_UNIX_SOCKET_HPP_
#define MSCCLPP_UNIX_SOCKET_HPP_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace mscclpp {

class UnixSocketServer {
 public:
  static UnixSocketServer& instance();
  static std::string generateSocketPath(int socketId);

  void start();
  void stop();
  int registerFd(int fd);
  void unregisterFd(int fd);
  std::string getSocketPath() const;
  ~UnixSocketServer();

 private:
  int listenUnixSockFd_ = -1;
  std::string listenUnixSockPath_;
  std::thread mainThread_;
  std::unique_ptr<uint32_t> abortFlagStorage_;
  uint32_t volatile* abortFlag_;
  std::mutex mutex_;
  std::unordered_set<int> fdSet_;

  UnixSocketServer();
  void mainLoop(int listenUnixSockFd);
};

class UnixSocketClient {
 public:
  static UnixSocketClient& instance();

  int requestFd(std::string const& socketPath, uint32_t fdId);
  void reset();
  ~UnixSocketClient();

 private:
  std::unordered_map<std::string, int> cachedFds_;
  std::mutex mutex_;

  UnixSocketClient() = default;
  int requestFdInternal(int connFd, uint32_t fdId);
};

}  // namespace mscclpp

#endif  // MSCCLPP_UNIX_SOCKET_HPP_

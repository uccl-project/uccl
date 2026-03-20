// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "unix_socket.hpp"
#include "debug.h"
#include "errors.hpp"
#include <atomic>
#include <cerrno>
#include <cstring>
#include <vector>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace mscclpp {

namespace {

constexpr size_t kUnixPathMax = sizeof(sockaddr_un::sun_path) - 1;

void sendAll(int fd, void const* buffer, size_t size) {
  char const* data = static_cast<char const*>(buffer);
  size_t written = 0;
  while (written < size) {
    ssize_t sent = send(fd, data + written, size - written, 0);
    if (sent < 0) {
      if (errno == EINTR) continue;
      throw SysError("send() on unix socket failed", errno);
    }
    written += static_cast<size_t>(sent);
  }
}

void recvAll(int fd, void* buffer, size_t size) {
  char* data = static_cast<char*>(buffer);
  size_t received = 0;
  while (received < size) {
    ssize_t n = ::recv(fd, data + received, size - received, 0);
    if (n <= 0) {
      if (errno == EINTR) continue;
      throw SysError("recv() on unix socket failed", errno);
    }
    received += static_cast<size_t>(n);
  }
}

void sendStatusAndFd(int sockFd, int status, int fdToSend) {
  struct msghdr msg;
  std::memset(&msg, 0, sizeof(msg));

  struct iovec iov;
  iov.iov_base = &status;
  iov.iov_len = sizeof(status);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  char control[CMSG_SPACE(sizeof(int))];
  if (status == 0) {
    std::memset(control, 0, sizeof(control));
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &fdToSend, sizeof(fdToSend));
  }

  ssize_t bytes = ::sendmsg(sockFd, &msg, 0);
  if (bytes < 0) {
    throw SysError("sendmsg() on unix socket failed", errno);
  }
}

}  // namespace

UnixSocketServer& UnixSocketServer::instance() {
  static UnixSocketServer server;
  return server;
}

std::string UnixSocketServer::generateSocketPath(int socketId) {
  std::string path =
      "/tmp/mscclpp_bootstrap_" + std::to_string(socketId) + ".sock";
  if (path.size() > kUnixPathMax) {
    throw Error("Generated unix socket path is too long: " + path,
                ErrorCode::InternalError);
  }
  return path;
}

void UnixSocketServer::start() {
  *abortFlag_ = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (listenUnixSockFd_ != -1) {
    return;
  }

  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    throw SysError("socket() failed for unix domain socket", errno);
  }

  std::string socketPath = generateSocketPath(getpid());
  unlink(socketPath.c_str());
  sockaddr_un addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    int err = errno;
    ::close(fd);
    throw SysError("bind() failed for unix socket, sock path: " + socketPath,
                   err);
  }

  if (listen(fd, SOMAXCONN) < 0) {
    int err = errno;
    ::close(fd);
    unlink(socketPath.c_str());
    throw SysError("listen() failed for unix socket, sock path: " + socketPath,
                   err);
  }

  listenUnixSockFd_ = fd;
  listenUnixSockPath_ = socketPath;
  mainThread_ = std::thread([this] {
    try {
      this->mainLoop(listenUnixSockFd_);
    } catch (const std::exception& e) {
      if (abortFlag_ && *abortFlag_) {
        return;
      }
      WARN("Unix socket server main loop exited with exception: %s", e.what());
      throw e;
    }
  });
}

void UnixSocketServer::stop() {
  *abortFlag_ = 1;
  if (mainThread_.joinable()) {
    INFO(MSCCLPP_INIT, "Stopping unix socket server");
    mainThread_.join();
  }
  ::close(listenUnixSockFd_);
  listenUnixSockFd_ = -1;
  if (!listenUnixSockPath_.empty()) {
    unlink(listenUnixSockPath_.c_str());
  }
}

int UnixSocketServer::registerFd(int fd) {
  INFO(MSCCLPP_P2P, "Registered fd %d", fd);
  std::lock_guard<std::mutex> lock(mutex_);
  if (fdSet_.count(fd) != 0) {
    throw Error("Fd is already registered: " + std::to_string(fd),
                ErrorCode::InvalidUsage);
  }
  fdSet_.insert(fd);
  return fd;
}

void UnixSocketServer::unregisterFd(int fd) {
  INFO(MSCCLPP_P2P, "Unregistered fd %d", fd);
  std::lock_guard<std::mutex> lock(mutex_);
  fdSet_.erase(fd);
}

void UnixSocketServer::mainLoop(int listenUnixSockFd) {
  std::vector<pollfd> pollFds;
  pollFds.push_back(
      {listenUnixSockFd, POLLIN | POLLERR | POLLHUP | POLLNVAL | POLLRDHUP, 0});
  auto removeClient = [&pollFds](size_t index) {
    if (index == 0 || index >= pollFds.size()) {
      return;
    }
    ::close(pollFds[index].fd);
    pollFds.erase(pollFds.begin() + static_cast<std::ptrdiff_t>(index));
  };

  while (true) {
    if (abortFlag_ && *abortFlag_) {
      break;
    }
    int rc = poll(pollFds.data(), pollFds.size(), 100);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (abortFlag_ && *abortFlag_) {
        break;
      }
      WARN("poll() failed on unix socket server: %s", std::strerror(errno));
      throw SysError("poll() failed on unix socket server", errno);
    }
    if (rc == 0) {
      continue;
    }

    pollfd& listenPfd = pollFds[0];
    if (listenPfd.revents & (POLLERR | POLLHUP | POLLNVAL | POLLRDHUP)) {
      if (abortFlag_ && *abortFlag_) {
        break;
      }
      throw Error(
          std::string("Unexpected event on unix socket listen fd: revents=0x") +
              std::to_string(listenPfd.revents),
          ErrorCode::InternalError);
    }

    if (listenPfd.revents & POLLIN) {
      int connFd = accept(listenUnixSockFd, nullptr, nullptr);
      if (connFd >= 0) {
        pollFds.push_back(
            {connFd, POLLIN | POLLERR | POLLHUP | POLLNVAL | POLLRDHUP, 0});
      } else if (errno != EINTR) {
        if (abortFlag_ && *abortFlag_) {
          break;
        }
        WARN("accept() failed for unix socket: %s", std::strerror(errno));
        throw SysError("accept() failed for unix socket", errno);
      }
    }

    for (size_t idx = 1; idx < pollFds.size();) {
      pollfd& client = pollFds[idx];
      if (client.revents & (POLLERR | POLLHUP | POLLNVAL | POLLRDHUP)) {
        removeClient(idx);
        continue;
      } else if (client.revents & POLLIN) {
        int fd = 0;
        recvAll(client.fd, &fd, sizeof(fd));
        int fdToSend = -1;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          auto it = fdSet_.find(fd);
          if (it == fdSet_.end()) {
            WARN("Requested fd %d not found, size of fdSet_ is %zu", fd,
                 fdSet_.size());
            throw Error("Requested fd not found: " + std::to_string(fd),
                        ErrorCode::InvalidUsage);
          }
          fdToSend = *it;
        }
        sendStatusAndFd(client.fd, 0, fdToSend);
      }
      ++idx;
    }
  }

  for (size_t i = 1; i < pollFds.size(); ++i) {
    ::close(pollFds[i].fd);
  }
}

UnixSocketServer::UnixSocketServer()
    : abortFlagStorage_(new uint32_t(0)), abortFlag_(abortFlagStorage_.get()) {}

std::string UnixSocketServer::getSocketPath() const {
  return listenUnixSockPath_;
}

UnixSocketClient& UnixSocketClient::instance() {
  static UnixSocketClient client;
  return client;
}

void UnixSocketClient::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto const& pair : cachedFds_) {
    ::close(pair.second);
  }
  cachedFds_.clear();
}

UnixSocketClient::~UnixSocketClient() { reset(); }

int UnixSocketClient::requestFd(std::string const& socketPath, uint32_t fdId) {
  INFO(MSCCLPP_P2P, "Requesting fdId %u from unix socket server at %s", fdId,
       socketPath.c_str());

  int connectedFd = -1;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cachedFds_.find(socketPath);
    if (it != cachedFds_.end()) {
      connectedFd = it->second;
    }
  }
  if (connectedFd != -1) {
    return requestFdInternal(connectedFd, fdId);
  }

  connectedFd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (connectedFd < 0) {
    throw SysError("socket() failed for unix domain socket", errno);
  }
  sockaddr_un addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);

  INFO(MSCCLPP_P2P, "Connecting to unix socket server at %s",
       socketPath.c_str());
  if (connect(connectedFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
      0) {
    ::close(connectedFd);
    throw SysError("connect() failed for unix socket to " + socketPath, errno);
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = cachedFds_.emplace(socketPath, connectedFd);
    if (!inserted) {
      ::close(it->second);
      it->second = connectedFd;
    }
  }
  return requestFdInternal(connectedFd, fdId);
}

int UnixSocketClient::requestFdInternal(int connFd, uint32_t fdId) {
  sendAll(connFd, &fdId, sizeof(fdId));

  struct msghdr msg;
  std::memset(&msg, 0, sizeof(msg));

  int response = -1;
  struct iovec iov;
  iov.iov_base = &response;
  iov.iov_len = sizeof(response);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  char control[CMSG_SPACE(sizeof(int))];
  std::memset(control, 0, sizeof(control));
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);

  ssize_t bytes = ::recvmsg(connFd, &msg, 0);
  if (bytes <= 0) {
    throw SysError("recvmsg() on unix socket failed", errno);
  }

  int receivedFd = -1;
  if (response != 0) {
    throw SysError("Failed to request fd from unix socket server", -response);
  }
  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg == nullptr || cmsg->cmsg_level != SOL_SOCKET ||
      cmsg->cmsg_type != SCM_RIGHTS) {
    throw Error("Did not receive file descriptor over unix socket",
                ErrorCode::InternalError);
  }
  std::memcpy(&receivedFd, CMSG_DATA(cmsg), sizeof(receivedFd));
  return receivedFd;
}

}  // namespace mscclpp

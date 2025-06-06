#pragma once

/**
 * @file util_flock.h
 * @brief File locking utility for process synchronization.
 *
 * This file provides a simple interface for file-based locking using
 * POSIX file locks. It allows you to create, lock, unlock, and remove
 * a lock file.
 */

#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>

#define UCCL_LOCKFILE "/tmp/uccl.lockfile"

class ProcessFileLock {
 public:
  explicit ProcessFileLock() : lockfile_(UCCL_LOCKFILE), fd_(-1) {
    fd_ = open(lockfile_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
      throw std::runtime_error("Failed to open lock file");
    }
  }

  void lock() {
    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;

    if (fcntl(fd_, F_SETLKW, &fl) < 0) {
      throw std::runtime_error("Failed to acquire lock");
    }
  }

  bool try_lock() {
    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;

    return fcntl(fd_, F_SETLK, &fl) >= 0;
  }

  void unlock() {
    struct flock fl;
    fl.l_type = F_UNLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;

    if (fcntl(fd_, F_SETLK, &fl) < 0) {
      throw std::runtime_error("Failed to release lock");
    }
  }

  ~ProcessFileLock() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  ProcessFileLock(ProcessFileLock const&) = delete;
  ProcessFileLock& operator=(ProcessFileLock const&) = delete;

 private:
  std::string lockfile_;
  int fd_;
};
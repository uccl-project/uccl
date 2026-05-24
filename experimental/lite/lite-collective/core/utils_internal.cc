// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "utils_internal.hpp"
#include "debug.h"
#include "env.hpp"
#include "errors.hpp"
#include "gpu_utils.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <signal.h>
#include <unistd.h>

// Throw upon SIGALRM.
static void sigalrmTimeoutHandler(int) {
  signal(SIGALRM, SIG_IGN);
  throw mscclpp::Error("Timer timed out", mscclpp::ErrorCode::Timeout);
}

constexpr char HOSTID_FILE[32] = "/proc/sys/kernel/random/boot_id";

static bool matchIf(char const* string, char const* ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(int const port1, int const port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}

namespace mscclpp {

Timer::Timer(int timeout) { set(timeout); }

Timer::~Timer() {
  if (timeout_ > 0) {
    alarm(0);
    signal(SIGALRM, SIG_DFL);
  }
}

int64_t Timer::elapsed() const {
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
      .count();
}

void Timer::set(int timeout) {
  timeout_ = timeout;
  if (timeout > 0) {
    signal(SIGALRM, sigalrmTimeoutHandler);
    alarm(timeout);
  }
  start_ = std::chrono::steady_clock::now();
}

void Timer::reset() { set(timeout_); }

void Timer::print(std::string const& name) {
  auto us = elapsed();
  std::stringstream ss;
  ss << name << ": " << us << " us\n";
  std::cout << ss.str();
}

std::string int64ToBusId(int64_t id) {
  char busId[20];
  std::snprintf(busId, sizeof(busId), "%04lx:%02lx:%02lx.%01lx", (id) >> 20,
                (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
  return std::string(busId);
}

int64_t busIdToInt64(const std::string busId) {
  char hexStr[17];  // Longest possible int64 hex string + null terminator.
  size_t hexOffset = 0;
  for (size_t i = 0; hexOffset < sizeof(hexStr) - 1 && i < busId.length();
       ++i) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else
      break;
  }
  hexStr[hexOffset] = '\0';
  return std::strtol(hexStr, NULL, 16);
}

uint64_t getHash(char const* string, int n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the MSCCLPP_HOSTID env var.
 */
uint64_t computeHostHash(void) {
  const size_t hashLen = 1024;
  char hostHash[hashLen];

  memset(hostHash, 0, hashLen);

  std::string hostName = getHostName(hashLen, '\0');
  strncpy(hostHash, hostName.c_str(), hostName.size());

  std::string hostid = env()->hostid;
  if (hostid != "") {
    strncpy(hostHash, hostid.c_str(), hashLen);
  } else if (hostName.size() < hashLen) {
    std::ifstream file(HOSTID_FILE, std::ios::binary);
    if (file.is_open()) {
      file.read(hostHash + hostName.size(), hashLen - hostName.size());
    }
  }

  // Make sure the string is terminated
  hostHash[sizeof(hostHash) - 1] = '\0';
  TRACE(MSCCLPP_INIT, "unique hostname '%s'", hostHash);
  return getHash(hostHash, strlen(hostHash));
}

uint64_t getHostHash(void) {
  thread_local std::unique_ptr<uint64_t> hostHash =
      std::make_unique<uint64_t>(computeHostHash());
  // avoid crash on static destruction
  if (hostHash == nullptr) {
    hostHash = std::make_unique<uint64_t>(computeHostHash());
  }
  return *hostHash;
}

/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $$ $(readlink /proc/self/ns/pid)
 */
uint64_t computePidHash(void) {
  char pname[1024];
  // Start off with our pid ($$)
  std::snprintf(pname, sizeof(pname), "%ld", (long)getpid());
  int plen = strlen(pname);
  int len =
      readlink("/proc/self/ns/pid", pname + plen, sizeof(pname) - 1 - plen);
  if (len < 0) len = 0;

  pname[plen + len] = '\0';
  TRACE(MSCCLPP_INIT, "unique PID '%s'", pname);

  return getHash(pname, strlen(pname));
}

uint64_t getPidHash(void) {
  thread_local std::unique_ptr<uint64_t> pidHash =
      std::make_unique<uint64_t>(computePidHash());
  // avoid crash on static destruction
  if (pidHash == nullptr) {
    pidHash = std::make_unique<uint64_t>(computePidHash());
  }
  return *pidHash;
}

int parseStringList(char const* string, netIf* ifList, int maxList) {
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

bool matchIfList(char const* string, int port, netIf* ifList, int listSize,
                 bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i = 0; i < listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact) &&
        matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

/* get any bytes of random data from /dev/urandom */
void getRandomData(void* buffer, size_t bytes) {
  if (bytes > 0) {
    const size_t one = 1UL;
    FILE* fp = fopen("/dev/urandom", "r");
    if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one) {
      throw Error("Failed to read random data", ErrorCode::SystemError);
    }
    if (fp) fclose(fp);
  }
}

TokenPool::TokenPool(size_t nToken) : nToken_(nToken) {
#if (CUDA_NVLS_API_AVAILABLE)
  tokens_ = detail::gpuCallocPhysicalShared<uint64_t>(
      nToken,
      detail::getCuAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  MSCCLPP_CUTHROW(cuMemGetAddressRange((CUdeviceptr*)(&baseAddr_), NULL,
                                       (CUdeviceptr)tokens_.get()));
  size_t nElems = (nToken + (UINT64_WIDTH - 1)) / UINT64_WIDTH;
  allocationMap_.resize(nElems, 0);
  tailMask_ =
      (nToken % UINT64_WIDTH) ? ((1UL << (nToken % UINT64_WIDTH)) - 1) : ~0UL;
#else
  throw Error("TokenPool only available on GPUs with NVLS support",
              ErrorCode::InvalidUsage);
#endif
}

std::shared_ptr<uint64_t> TokenPool::getToken() {
  auto deleter = [self = shared_from_this()](uint64_t* token) {
    size_t index = (token - self->baseAddr_) / UINT64_WIDTH;
    size_t bit = (token - self->baseAddr_) % UINT64_WIDTH;
    uint64_t mask = 1UL << bit;
    self->allocationMap_[index] &= ~mask;
  };

  size_t size = allocationMap_.size();
  for (size_t i = 0; i < size; i++) {
    uint64_t ullong = allocationMap_[i].to_ullong();
    uint64_t mask = (i + 1 == size) ? tailMask_ : ~0ULL;
    uint64_t holes = (~ullong) & mask;
    if (!holes) continue;
    for (int bit = 0; bit < UINT64_WIDTH; bit++) {
      if (holes & (1UL << bit)) {
        allocationMap_[i].set(bit);
        INFO(MSCCLPP_ALLOC, "TokenPool allocated token at addr %p",
             baseAddr_ + i * UINT64_WIDTH + bit);
        return std::shared_ptr<uint64_t>(baseAddr_ + i * UINT64_WIDTH + bit,
                                         deleter);
      }
    }
  }
  throw Error("TokenPool is exhausted", ErrorCode::InternalError);
}

}  // namespace mscclpp

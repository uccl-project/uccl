// Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_UTILS_INTERNAL_HPP_
#define MSCCLPP_UTILS_INTERNAL_HPP_

#include "utils.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace mscclpp {

struct Timer {
  std::chrono::steady_clock::time_point start_;
  int timeout_;

  Timer(int timeout = -1);

  ~Timer();

  /// Returns the elapsed time in microseconds.
  int64_t elapsed() const;

  void set(int timeout);

  void reset();

  void print(std::string const& name);
};

// PCI Bus ID <-> int64 conversion functions
std::string int64ToBusId(int64_t id);
int64_t busIdToInt64(const std::string busId);

uint64_t getHash(char const* string, int n);
uint64_t getHostHash();
uint64_t getPidHash();
void getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(char const* string, struct netIf* ifList, int maxList);
bool matchIfList(char const* string, int port, struct netIf* ifList,
                 int listSize, bool matchExact);

template <class T>
inline void hashCombine(std::size_t& hash, const T& v) {
  std::hash<T> hasher;
  hash ^= hasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

struct PairHash {
 public:
  template <typename T, typename U>
  std::size_t operator()(std::pair<T, U> const& x) const {
    std::size_t hash = 0;
    hashCombine(hash, x.first);
    hashCombine(hash, x.second);
    return hash;
  }
};

class TokenPool : public std::enable_shared_from_this<TokenPool> {
 public:
  TokenPool(size_t nTokens);
  std::shared_ptr<uint64_t> getToken();

 private:
  size_t nToken_;
  uint64_t* baseAddr_;
  uint64_t tailMask_;
  std::shared_ptr<uint64_t> tokens_;
  std::vector<std::bitset<UINT64_WIDTH>> allocationMap_;
};

}  // namespace mscclpp

#endif

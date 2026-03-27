#pragma once

#include "../executor.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace UKernel {
namespace CCL {
namespace TestUtil {

[[noreturn]] inline void fail(std::string const& msg) {
  throw std::runtime_error(msg);
}

inline void require(bool cond, std::string const& msg) {
  if (!cond) fail(msg);
}

template <typename Fn>
inline bool throws(Fn&& fn) {
  try {
    fn();
  } catch (...) {
    return true;
  }
  return false;
}

template <typename Fn>
inline void run_case(char const* suite, char const* name, Fn&& fn) {
  std::cout << "[test] " << suite << " " << name << "..." << std::endl;
  fn();
}

inline bool wait_until_terminal(Executor& executor, CollectiveOpHandle handle,
                                std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (executor.poll(handle)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return executor.poll(handle);
}

}  // namespace TestUtil
}  // namespace CCL
}  // namespace UKernel

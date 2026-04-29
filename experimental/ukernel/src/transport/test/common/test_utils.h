#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace UKernel::Transport::TestUtil {

inline void require(bool cond, std::string_view message) {
  if (!cond) {
    throw std::runtime_error(std::string(message));
  }
}

inline std::string get_arg(int argc, char** argv, char const* key,
                           char const* def) {
  std::string key_str(key);
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key_str && i + 1 < argc) {
      return std::string(argv[i + 1]);
    }
    auto const arg = std::string(argv[i]);
    auto const prefix = key_str + "=";
    if (arg.rfind(prefix, 0) == 0) {
      return arg.substr(prefix.size());
    }
  }
  return std::string(def);
}

inline int get_int_arg(int argc, char** argv, char const* key, int def) {
  auto def_str = std::to_string(def);
  return std::stoi(get_arg(argc, argv, key, def_str.c_str()));
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
  std::cout << "[test][" << suite << "] " << name << "..." << std::endl;
  std::forward<Fn>(fn)();
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(char const* key, char const* value) : key_(key) {
    char const* old = std::getenv(key);
    if (old != nullptr) {
      had_old_ = true;
      old_value_ = old;
    }
    if (value != nullptr) {
      ::setenv(key, value, 1);
    } else {
      ::unsetenv(key);
    }
  }

  ~ScopedEnvVar() {
    if (had_old_) {
      ::setenv(key_.c_str(), old_value_.c_str(), 1);
    } else {
      ::unsetenv(key_.c_str());
    }
  }

  ScopedEnvVar(ScopedEnvVar const&) = delete;
  ScopedEnvVar& operator=(ScopedEnvVar const&) = delete;

 private:
  std::string key_;
  bool had_old_ = false;
  std::string old_value_;
};

}  // namespace UKernel::Transport::TestUtil

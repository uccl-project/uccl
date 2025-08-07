#pragma once
#include <string>
#include <cuda_runtime.h>

class EPException : public std::exception {
 private:
  std::string message = {};

 public:
  explicit EPException(char const* name, char const* file, int const line,
                       std::string const& error) {
    message = std::string("Failed: ") + name + " error " + file + ":" +
              std::to_string(line) + " '" + error + "'";
  }

  char const* what() const noexcept override { return message.c_str(); }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                     \
  do {                                                                      \
    cudaError_t e = (cmd);                                                  \
    if (e != cudaSuccess) {                                                 \
      throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                       \
  } while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                     \
  do {                                                           \
    if (not(cond)) {                                             \
      throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    }                                                            \
  } while (0)
#endif
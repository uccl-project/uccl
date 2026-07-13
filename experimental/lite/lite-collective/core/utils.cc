// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "utils.hpp"
#include "errors.hpp"
#include <chrono>
#include <string>
#include <unistd.h>

namespace mscclpp {

std::string getHostName(int maxlen, char const delim) {
  std::string hostname(maxlen + 1, '\0');
  if (gethostname(const_cast<char*>(hostname.data()), maxlen) != 0) {
    throw Error("gethostname failed", ErrorCode::SystemError);
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1))
    i++;
  hostname[i] = '\0';
  return hostname.substr(0, i);
}

}  // namespace mscclpp

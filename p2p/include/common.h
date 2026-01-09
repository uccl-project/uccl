#ifndef COMMON_H
#define COMMON_H

#include <infiniband/verbs.h>
#include <cstdint>
#include <cstring>

typedef uint64_t FlowID;
struct ConnID {
  void* context;
  int sock_fd;
  int dev;
  FlowID flow_id;
};

#endif
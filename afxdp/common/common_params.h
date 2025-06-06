/* This common_user.h is used by userspace programs */
#ifndef __COMMON_PARAMS_H
#define __COMMON_PARAMS_H

#include "common_defines.h"
#include <getopt.h>

struct option_wrapper {
  struct option option;
  char* help;
  char* metavar;
  bool required;
};

void usage(char const* prog_name, char const* doc,
           const struct option_wrapper* long_options, bool full);

void parse_cmdline_args(int argc, char** argv,
                        const struct option_wrapper* long_options,
                        struct config* cfg, char const* doc);

#endif /* __COMMON_PARAMS_H */

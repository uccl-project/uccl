#include "driver.h"
#include <iostream>
#include <glog/logging.h>
#include <signal.h>

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;

  Driver driver;
  driver.start(argc, argv);

  static volatile bool keep_running = true;
  struct sigaction sa;
  sa.sa_handler = [](int){ keep_running = false; };
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGTERM, &sa, NULL);

  while (keep_running) {
    // INSERT_YOUR_CODE
    driver.recv();
  }

  driver.close();
  return 0;
}

#include "transport.h"

#include <glog/logging.h>

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    return 0;
}
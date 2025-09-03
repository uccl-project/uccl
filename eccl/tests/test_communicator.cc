#include <iostream>
#include "test.h"

#include "transport.h"

void test_communicator() {
    std::cout << "test_communicator" << std::endl;
    auto comm = Communicator(0, 0, 1);
    std::cout << "check if ready: " << comm.check_ready() <<std::endl;
    std::cout << "exit test_communicator" << std::endl;
}

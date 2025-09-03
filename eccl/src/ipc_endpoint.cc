#include "transport.h"

IPCEndpoint::IPCEndpoint(std::shared_ptr<Config> config): config_(config) {
}

IPCEndpoint::~IPCEndpoint() {
}

bool IPCEndpoint::connect_to(int uds_fd) {
    return true;
}

bool IPCEndpoint::send_async(std::shared_ptr<CommRequest> creq){
    return true;
}

bool IPCEndpoint::recv_async(std::shared_ptr<CommRequest> creq) {
    return true;
}

bool IPCEndpoint::poll_reqs(std::vector<std::shared_ptr<CommRequest>>& creqs) {
    return true;
}

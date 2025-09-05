#include "transport.h"

IPCEndpoint::IPCEndpoint(std::shared_ptr<Config> config): config_(config) {
}

IPCEndpoint::~IPCEndpoint() {
}

bool IPCEndpoint::connect_to(int rank) {
    return true;
}

bool IPCEndpoint::accept_from(int rank) {
    return true;
}

bool IPCEndpoint::send_async(int to_rank, std::shared_ptr<Request> creq){
    return true;
}

bool IPCEndpoint::recv_async(int from_rank, std::shared_ptr<Request> creq) {
    return true;
}

bool IPCEndpoint::poll_reqs(std::vector<std::shared_ptr<Request>>& creqs) {
    return true;
}

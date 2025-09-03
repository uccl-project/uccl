#include "transport.h"

RDMAEndpoint::RDMAEndpoint(std::shared_ptr<Config> config): config_(config) {
}

RDMAEndpoint::~RDMAEndpoint() {
}

bool RDMAEndpoint::connect_to(std::string server_ip, int port) {
    // exchange conn info first

    // then create qp
    return true;
}

bool RDMAEndpoint::send_async(std::shared_ptr<CommRequest> creq) {
    return true;
}

bool RDMAEndpoint::recv_async(std::shared_ptr<CommRequest> creq) {
    return true;
}

bool RDMAEndpoint::poll_reqs(std::vector<std::shared_ptr<CommRequest>>& creqs) {
    return true;
}

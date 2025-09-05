#include "transport.h"

RDMAEndpoint::RDMAEndpoint(std::shared_ptr<Config> config): config_(config) {
    qp_num = config->qp_count_per_ep;
    lid = 1111;
}

RDMAEndpoint::~RDMAEndpoint() {
}

bool RDMAEndpoint::connect_to(int rank) {
    // exchange conn info first

    // then create qp
    return true;
}

bool RDMAEndpoint::accept_from(int rank) {
    return true;
}

bool RDMAEndpoint::send_async(int to_rank, std::shared_ptr<Request> creq) {
    return true;
}

bool RDMAEndpoint::recv_async(int from_rank, std::shared_ptr<Request> creq) {
    return true;
}

bool RDMAEndpoint::poll_reqs(std::vector<std::shared_ptr<Request>>& creqs) {
    return true;
}

#include "tcpx_engine.h"

thread_local bool inside_python = false;

namespace tcpx {

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus) {
    return;
}

Endpoint::~Endpoint() {
    return;
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
            uint64_t& conn_id) {
    return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id) {
    return true;
}

std::vector<uint8_t> Endpoint::get_metadata() {
    return std::vector<uint8_t>{0};
}

bool Endpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                char* out_buf) {
    return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
    return true;
}

bool Endpoint::dereg(uint64_t mr_id) {
    return true;
}

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                FifoItem const& slot_item, uint64_t* transfer_id) {
    return true;
}

bool Endpoint::send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                size_t size, uint64_t* transfer_id) {
    return true;
}

bool Endpoint::recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                uint64_t* transfer_id) {
    return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
    return true;
}

}
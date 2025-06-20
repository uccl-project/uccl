#include "engine.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

Endpoint::Endpoint(const uint32_t gpu_idx, const uint32_t ncpus)
    : gpu_idx_(gpu_idx), ncpus_(ncpus) {
  std::cout << "Creating Engine with GPU index: " << gpu_idx
            << ", CPUs: " << ncpus << std::endl;

  // Initialize the RDMA endpoint with lazy creation.
  ep_ = new uccl::RDMAEndpoint(NUM_DEVICES, ncpus);

  // Initialize the engine based on the GPU index.
  static std::atomic<uint16_t> listen_port = 10000;
  ep_->initialize_engine_by_dev(gpu_idx_, listen_port);

  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::~Endpoint() {
  std::cout << "Destroying Engine..." << std::endl;
  delete ep_;
  std::cout << "Engine destroyed" << std::endl;
}

bool Endpoint::connect(std::string const& ip_addr, int const& remote_gpu_idx,
                       uint64_t& conn_id) {
  std::cout << "Attempting to connect to " << ip_addr << ":" << remote_gpu_idx
            << std::endl;

  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  uccl::ConnID uccl_conn_id =
      ep_->test_uccl_connect(gpu_idx_, ip_addr, remote_gpu_idx);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};

  return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // For demo purposes, simulate accepted connection
  conn_id = next_conn_id_.fetch_add(1);

  uccl::ConnID uccl_conn_id =
      ep_->test_uccl_accept(gpu_idx_, ip_addr, &remote_gpu_idx);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};

  return true;
}

bool Endpoint::reg_kv(void const* data, size_t size, uint64_t& mr_id) {
  std::cout << "Registering KV, size: " << size << " bytes" << std::endl;

  mr_id = next_mr_id_.fetch_add(1);

  uccl::Mhandle* mhandle;
  ep_->uccl_regmr(gpu_idx_, const_cast<void*>(data), size, 0, &mhandle);

  mr_id_to_mr_[mr_id] = new MR{mr_id, mhandle};

  return true;
}

bool Endpoint::send_kv(uint64_t conn_id, uint64_t mr_id, void const* data,
                       size_t size) {
  std::cout << "Sending KV with mr_id: " << mr_id << ", size: " << size
            << " bytes" << std::endl;
  uccl::ucclRequest ureq;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  ep_->uccl_send_async(
      static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle, data,
      size, &ureq);

  ep_->uccl_poll_ureq(&ureq);

  std::cout << "KV sent successfully" << std::endl;
  return true;
}

bool Endpoint::recv_kv(uint64_t conn_id, uint64_t mr_id, void* data,
                       size_t& size) {
  DCHECK(size && 0xffffffff == 0) << "size must be less than 4GB";

  std::cout << "Receiving KV with mr_id: " << mr_id << std::endl;
  uccl::ucclRequest ureq;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  int size_int = static_cast<int>(size);

  ep_->uccl_recv_async(
      static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
      &data, &size_int, 1, &ureq);

  ep_->uccl_poll_ureq(&ureq);

  size = static_cast<size_t>(size_int);

  std::cout << "KV received successfully, size: " << size << " bytes"
            << std::endl;
  return true;
}

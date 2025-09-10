#include "test.h"
#include "transport.h"
#include "util/gpu_rt.h"
#include <iostream>

void cqpoller_client_thread(std::shared_ptr<Communicator> comm, int peer_rank) {
  if (comm->connect_to(peer_rank)) {
    std::cout << "[CLIENT] Successfully connected to rank " << peer_rank
              << std::endl;
  } else {
    std::cerr << "[CLIENT] Failed to connect to rank " << peer_rank
              << std::endl;
  }
  float* d_data;
  size_t size = 1024 * sizeof(float);
  GPU_RT_CHECK(gpuMalloc((void**)&d_data, size));
  auto mr = comm->reg_mr(d_data, size);
  auto remote_mr = comm->wait_mr_notify(peer_rank);
  std::cout << "[CLIENT] Got remote MR id=" << remote_mr.id << " addr=0x"
            << std::hex << remote_mr.address << " len=" << std::dec
            << remote_mr.length << "\n";
  bool ok = comm->isend(peer_rank, d_data, 0, size, mr.id, remote_mr.id, true);
  if (!ok) {
    std::cerr << "[CLIENT] isend failed\n";
  } else {
    std::cout << "[CLIENT] isend posted\n";
  }

  comm->wait_finish();
  std::cout << "[CLIENT] send completed\n";

  GPU_RT_CHECK(gpuFree(d_data));
}

void cqpoller_server_thread(std::shared_ptr<Communicator> comm, int peer_rank) {
  if (comm->accept_from(peer_rank)) {
    std::cout << "[SERVER] Successfully accepted connection from rank "
              << peer_rank << std::endl;
  } else {
    std::cerr << "[SERVER] Failed to accept connection from rank " << peer_rank
              << std::endl;
  }
  float* d_data;
  size_t size = 1024 * sizeof(float);
  GPU_RT_CHECK(gpuMalloc((void**)&d_data, size));
  auto mr = comm->reg_mr(d_data, size);
  comm->notify_mr(peer_rank, mr);

  // 发起异步 recv
  bool ok = comm->irecv(peer_rank, d_data, 0, size, true);
  if (!ok) {
    std::cerr << "[SERVER] irecv failed\n";
  } else {
    std::cout << "[SERVER] irecv posted\n";
  }

  // 等待完成
  comm->wait_finish();
  std::cout << "[SERVER] recv completed, first 8 bytes="
            << *reinterpret_cast<uint64_t*>(d_data) << "\n";

  GPU_RT_CHECK(gpuFree(d_data));
}

void test_cq_poller() {
  auto comm0 = std::make_shared<Communicator>(
      0, 0, 2);  // gpu_id=0, local_rank=0, world_size=2
  auto comm1 = std::make_shared<Communicator>(
      0, 1, 2);  // gpu_id=0, local_rank=1, world_size=2

  std::thread t_client(cqpoller_client_thread, comm0, 1);
  std::thread t_server(cqpoller_server_thread, comm1, 0);

  t_client.join();
  t_server.join();

  std::cout << "[TEST] RDMA QP connection test finished" << std::endl;
}
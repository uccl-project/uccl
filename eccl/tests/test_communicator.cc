#include "test.h"
#include "transport.h"
#include <iostream>

void client_thread(std::shared_ptr<Communicator> comm, int peer_rank) {
  if (comm->connect_to(peer_rank)) {
    std::cout << "[CLIENT] Successfully connected to rank " << peer_rank
              << std::endl;
  } else {
    std::cerr << "[CLIENT] Failed to connect to rank " << peer_rank
              << std::endl;
  }
}

void server_thread(std::shared_ptr<Communicator> comm, int peer_rank) {
  if (comm->accept_from(peer_rank)) {
    std::cout << "[SERVER] Successfully accepted connection from rank "
              << peer_rank << std::endl;
  } else {
    std::cerr << "[SERVER] Failed to accept connection from rank " << peer_rank
              << std::endl;
  }
}

void test_communicator() {
  auto comm0 = std::make_shared<Communicator>(
      0, 0, 2);  // gpu_id=0, local_rank=0, world_size=2
  auto comm1 = std::make_shared<Communicator>(
      0, 1, 2);  // gpu_id=0, local_rank=1, world_size=2

  std::thread t_client(client_thread, comm0, 1);
  std::thread t_server(server_thread, comm1, 0);

  t_client.join();
  t_server.join();

  std::cout << "[TEST] RDMA QP connection test finished" << std::endl;
}

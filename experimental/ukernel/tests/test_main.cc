#include "test.h"
#include <iostream>

int main(int argc, char** argv) {
  // test_find_best_rdma_for_gpu(0);
  // test_find_best_rdma_for_gpu(2);
  // test_find_best_rdma_for_gpu(3);
  // test_find_best_rdma_for_gpu(5);

  // test_communicator(argc, argv);

  // test_redis_oob();
  // test_uds_oob();

  // test_generate_host_id();

  test_socket_meta_exchange_multi_threads(4);

  // test_cq_poller();

  return 0;
}

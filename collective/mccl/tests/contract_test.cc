#include "mccl_plugin_musa.cc"
#include <cassert>
#include <iostream>

struct Fixture {
  std::shared_ptr<RDMAEndpoint> ep = std::make_shared<RDMAEndpoint>(4);
  McclContext context{ep, 1};
  McclRecvComm comm{};
  ibv_mr mr{};
  Mhandle mh{&mr};
  char buffer[16]{};
  void* data[2]{buffer, buffer};
  void* handles[2]{&mh, &mh};
  int sizes[2]{4, 4};
  size_t recv_sizes[2]{4, 4};

  Fixture() {
    g_endpoint = ep;
    comm.base.context = &context;
    comm.base.conn_id.context = &ep->flow;
    comm.base.request_pool = std::make_shared<McclRequestPool>();
  }

  void pool() {
    uint64_t address;
    while (comm.base.request_pool->alloc_buff(&address) == 0) {
    }
    void* request = this;
    assert(mcclPluginIsend(&comm, buffer, 4, 0, &mh, nullptr, &request) ==
           mcclSuccess);
    assert(request == nullptr);
    request = this;
    assert(mcclPluginIrecv(&comm, 1, data, recv_sizes, nullptr, handles,
                           nullptr, &request) == mcclSuccess);
    assert(request == nullptr);
    request = this;
    assert(mcclPluginIflush(&comm, 1, data, sizes, handles, &request) ==
           mcclSuccess);
    assert(request == nullptr);
    assert(ep->submissions == 0);
  }

  void bounds() {
    auto invalid = [&](auto operation) {
      void* output = this;
      assert(operation(&output) == mcclInvalidArgument);
      assert(output == nullptr);
    };
    invalid([&](void** output) {
      return mcclPluginIsend(&comm, nullptr, 4, 0, &mh, nullptr, output);
    });
    invalid([&](void** output) {
      return mcclPluginIsend(&comm, buffer, 4, 0, nullptr, nullptr, output);
    });
    invalid([&](void** output) {
      return mcclPluginIrecv(&comm, 0, data, recv_sizes, nullptr, handles,
                             nullptr, output);
    });
    invalid([&](void** output) {
      return mcclPluginIflush(&comm, 2, data, sizes, handles, output);
    });
    handles[0] = nullptr;
    invalid([&](void** output) {
      return mcclPluginIrecv(&comm, 1, data, recv_sizes, nullptr, handles,
                             nullptr, output);
    });
    invalid([&](void** output) {
      return mcclPluginIflush(&comm, 1, data, sizes, handles, output);
    });
    handles[0] = &mh;
    recv_sizes[0] = size_t(INT_MAX) + 1;
    invalid([&](void** output) {
      return mcclPluginIrecv(&comm, 1, data, recv_sizes, nullptr, handles,
                             nullptr, output);
    });
    recv_sizes[0] = 4;
    void* request = this;
    assert(mcclPluginIsend(&comm, buffer, size_t(INT_MAX) + 1, 0, &mh, nullptr,
                           &request) == mcclInvalidArgument);
    assert(request == nullptr);
    request = this;
    assert(mcclPluginIrecv(&comm, 2, data, recv_sizes, nullptr, handles,
                           nullptr, &request) == mcclInvalidArgument);
    assert(request == nullptr);
    sizes[0] = -1;
    request = this;
    assert(mcclPluginIflush(&comm, 1, data, sizes, handles, &request) ==
           mcclInvalidArgument);
    assert(request == nullptr);
    assert(ep->submissions == 0);
    assert(ep->flow.qp.pending.empty());
    int done = 123;
    assert(mcclPluginTest(nullptr, &done, nullptr) == mcclInvalidArgument);
    assert(done == 0);
  }
};

void test_mr_cache() {
  RDMAEndpoint ep(4);
  g_musa_mr_support.clear();
  ep.mr_support[0][0] = true;
  ep.mr_support[1][1] = true;
  for (int gpu : {1, 0, 1, 0}) {
    host_gpu = gpu;
    for (int dev : {0, 1}) assert(can_register_musa(&ep, dev) == (gpu == dev));
  }
  for (int gpu : {0, 1})
    for (int dev : {0, 1}) assert(ep.registrations[gpu][dev] == 1);
  assert(host_allocations == 0 && ep.active_mrs == 0);
}

int main(int argc, char** argv) {
  assert(argc == 2);
  std::string name = argv[1];
  Fixture f;
  if (name == "pool")
    f.pool();
  else if (name == "bounds")
    f.bounds();
  else if (name == "mr")
    test_mr_cache();
  else if (name == "cq_poll" || name == "cq_error") {
    f.ep->flow.flow_cq_cnt_ = 1;
    f.ep->flow.cq.fail_poll = name == "cq_poll";
    // A failed CQE must not dereference even a poisoned wr_id.
    f.ep->flow.cq.completions.push_back({IBV_WC_RDMA_READ, 1, 0, 1, 1, 42});
    f.ep->flow.poll_flow_cq();
    return 99;
  } else
    return 2;
  std::cout << "contract " << name << ": PASS\n";
}

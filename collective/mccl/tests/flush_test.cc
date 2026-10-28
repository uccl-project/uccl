#include "mccl_plugin_musa.cc"
#include <cassert>
#include <iostream>

struct FlushFixture {
  std::shared_ptr<RDMAEndpoint> ep = std::make_shared<RDMAEndpoint>(4);
  McclContext context{ep, 123};
  McclRecvComm comm{};
  ibv_mr mr{};
  Mhandle mh{&mr};
  char buffer[16]{};
  void* data[3]{buffer, buffer + 4, buffer + 8};
  void* handles[3]{&mh, &mh, &mh};
  int sizes[3]{4, 8, 0};
  UcclFlow old_flow;

  FlushFixture() {
    g_endpoint = ep;
    comm.base.context = &context;
    comm.base.conn_id.context = &ep->flow;
    comm.base.request_pool = std::make_shared<McclRequestPool>();
  }

  ucclRequest* seed_reused_slot(UcclFlow* previous_flow) {
    auto pool = comm.base.request_pool;
    std::vector<uint64_t> slots;
    uint64_t address;
    while (pool->alloc_buff(&address) == 0) slots.push_back(address);
    auto* req = reinterpret_cast<ucclRequest*>(slots.front());
    req->context = previous_flow;
    req->type = ReqRx;
    req->rc_or_flush_done = true;
    for (auto slot : slots) pool->free_buff(slot);
    return req;
  }

  void run(bool reuse, bool stale, bool noop, bool multiple) {
    auto* reused =
        reuse ? seed_reused_slot(stale ? &old_flow : &ep->flow) : nullptr;
    if (noop) sizes[0] = 0;
    int count = multiple ? 3 : 1;
    auto available = comm.base.request_pool->size();
    void* request = nullptr;
    assert(mcclPluginIflush(&comm, count, data, sizes, handles, &request) ==
           mcclSuccess);
    assert(request != nullptr);
    auto* req = static_cast<ucclRequest*>(request);
    if (reuse) assert(req == reused);
    assert(req->context == &ep->flow);
    assert(req->type == ReqFlush);
    assert(comm.base.request_pool->size() == available - 1);

    bool async = !noop;
#ifdef __HIP_PLATFORM_AMD__
    async = false;
#endif
    if (async) {
      assert(ep->flow.qp.pending.size() == 1);
      // The flag must already be false at ibv_post_send, not cleared later.
      assert(ep->flow.qp.pending_at_post);
      auto expected = reinterpret_cast<uint64_t>(data[multiple ? 1 : 0]);
      assert(ep->flow.qp.pending.front().wr.rdma.remote_addr == expected);
      for (int i = 0; i < 10; ++i) {
        int done = -1;
        assert(mcclPluginTest(request, &done, nullptr) == mcclSuccess);
        assert(done == 0);
        assert(comm.base.request_pool->size() == available - 1);
      }
      assert(old_flow.cq.polls == 0);
      ep->flow.deliver_completions();
    } else {
      assert(ep->flow.qp.pending.empty());
    }
    int done = -1;
    assert(mcclPluginTest(request, &done, nullptr) == mcclSuccess);
    assert(done == 1);
    assert(ep->flow.flow_cq_cnt_ == 0);
    assert(comm.base.request_pool->size() == available);
    assert(ep->flow.cq.completions.empty());
  }
};

int main(int argc, char** argv) {
  assert(argc == 2);
  std::string test = argv[1];
  FlushFixture fixture;
  if (test == "fresh")
    fixture.run(false, false, false, false);
  else if (test == "stale")
    fixture.run(true, true, false, false);
  else if (test == "pending")
    fixture.run(true, false, false, false);
  else if (test == "noop")
    fixture.run(false, false, true, false);
  else if (test == "noop_stale")
    fixture.run(true, true, true, false);
  else if (test == "last")
    fixture.run(true, false, false, true);
  else
    assert(false);
  std::cout << "flush " << test << ": PASS\n";
}

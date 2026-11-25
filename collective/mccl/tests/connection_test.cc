#include "mccl_plugin_musa.cc"
#include <cassert>
#include <chrono>
#include <iostream>

template <typename Poll>
void* await_connection(Poll poll) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  void* result = nullptr;
  while (result == nullptr) {
    assert(poll(&result) == mcclSuccess);
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  return result;
}

void test_connect() {
  ConnectionGate gate;
  auto ep = std::make_shared<RDMAEndpoint>(4);
  ep->gate = &gate;
  McclContext context{ep, 123};
  McclHandle handle{};
  auto poll = [&](void** result) {
    return mcclPluginConnect(&context, 0, &handle, result, nullptr);
  };
  void* result = reinterpret_cast<void*>(1);
  assert(poll(&result) == mcclSuccess && result == nullptr);
  gate.wait_for_worker();
  for (int i = 0; i < 100; ++i) {
    result = reinterpret_cast<void*>(1);
    assert(poll(&result) == mcclSuccess && result == nullptr);
  }
  gate.release();
  auto* comm = static_cast<McclSendComm*>(await_connection(poll));
  assert(comm->base.context == &context);
  assert(comm->base.conn_id.context == &ep->flow);
  assert(comm->base.conn_id.sock_fd == 42);
  assert(comm->base.request_pool != nullptr);
  assert(mcclPluginCloseSend(comm) == mcclSuccess);
}

void test_accept() {
  ConnectionGate gate;
  auto ep = std::make_shared<RDMAEndpoint>(4);
  ep->gate = &gate;
  McclContext context{ep, 123};
  auto* listener = new McclListenComm{};
  listener->context = &context;
  listener->listen_fd = -1;
  auto poll = [&](void** result) {
    return mcclPluginAccept(listener, result, nullptr);
  };
  void* result = reinterpret_cast<void*>(1);
  assert(poll(&result) == mcclSuccess && result == nullptr);
  gate.wait_for_worker();
  assert(mcclPluginCloseListen(listener) == mcclInvalidUsage);
  for (int i = 0; i < 100; ++i) {
    result = reinterpret_cast<void*>(1);
    assert(poll(&result) == mcclSuccess && result == nullptr);
  }
  gate.release();
  auto* comm = static_cast<McclRecvComm*>(await_connection(poll));
  assert(comm->base.context == &context);
  assert(comm->base.conn_id.context == &ep->flow);
  assert(comm->base.conn_id.sock_fd == 43);
  assert(comm->base.request_pool != nullptr);
  assert(comm->remote_ip == "192.0.2.17");
  assert(comm->remote_dev == 7);
  assert(mcclPluginCloseRecv(comm) == mcclSuccess);
  assert(mcclPluginCloseListen(listener) == mcclSuccess);
}

void test_worker_failure(bool accepting) {
  ConnectionGate gate;
  auto ep = std::make_shared<RDMAEndpoint>(4);
  ep->gate = &gate;
  ep->fail_connection = true;
  McclContext context{ep, 123};
  McclHandle handle{};
  auto* listener = new McclListenComm{};
  listener->context = &context;
  listener->listen_fd = -1;
  auto poll = [&](void** result) {
    return accepting ? mcclPluginAccept(listener, result, nullptr)
                     : mcclPluginConnect(&context, 0, &handle, result, nullptr);
  };
  void* result = nullptr;
  assert(poll(&result) == mcclSuccess && result == nullptr);
  gate.wait_for_worker();
  gate.release();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  mcclResult_t status;
  do {
    result = reinterpret_cast<void*>(1);
    status = poll(&result);
    assert(result == nullptr);
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  } while (status == mcclSuccess);
  assert(status == mcclSystemError);
  assert(mcclPluginCloseListen(listener) == mcclSuccess);
}

int main() {
  for (int i = 0; i < 100; ++i) {
    test_connect();
    test_accept();
  }
  test_worker_failure(false);
  test_worker_failure(true);
  std::cout << "connect/accept publication: PASS (100 rounds each)\n";
}

#include "tcp_transport_adapter.h"
#include "test.h"
#include "test_utils.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

namespace {

using UKernel::Transport::TcpTransportAdapter;
using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

void fill_pattern(std::vector<uint8_t>& buf, uint8_t seed) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>((seed + i) & 0xFF);
  }
}

void test_tcp_connect_and_transfer() {
  TcpTransportAdapter server("127.0.0.1", /*local_rank=*/0);
  TcpTransportAdapter client("127.0.0.1", /*local_rank=*/1);

  std::exception_ptr accept_error;
  std::thread accept_thread([&] {
    try {
      TcpTransportAdapter::AcceptedPeer accepted{};
      require(server.accept_from_peer(/*peer_rank=*/1, "127.0.0.1", &accepted),
              "server accept_from_peer failed");
      require(accepted.remote_rank == 1, "accepted peer rank mismatch");
      require(accepted.remote_ip == "127.0.0.1", "accepted peer ip mismatch");
    } catch (...) {
      accept_error = std::current_exception();
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  require(client.connect_to_peer(/*peer_rank=*/0, "127.0.0.1",
                                 server.get_listen_port()),
          "client connect_to_peer failed");
  accept_thread.join();
  if (accept_error) std::rethrow_exception(accept_error);

  require(client.has_send_peer(0), "client send path should be connected");
  require(server.has_recv_peer(1), "server recv path should be connected");

  std::vector<uint8_t> sendbuf(8 * 1024);
  std::vector<uint8_t> recvbuf(sendbuf.size(), 0);
  fill_pattern(sendbuf, 0x33);

  require(server.recv_async(/*peer_rank=*/1, recvbuf.data(), recvbuf.size(),
                            /*request_id=*/41) == 0,
          "server recv_async should succeed");
  require(client.send_async(/*peer_rank=*/0, sendbuf.data(), sendbuf.size(),
                            /*request_id=*/42) == 0,
          "client send_async should succeed");

  require(client.wait_completion(42), "client wait_completion failed");
  require(server.wait_completion(41), "server wait_completion failed");
  require(!client.request_failed(42), "client request should not fail");
  require(!server.request_failed(41), "server request should not fail");
  require(sendbuf == recvbuf, "tcp adapter payload mismatch");

  client.release_request(42);
  server.release_request(41);
}

void test_tcp_invalid_peer_paths() {
  TcpTransportAdapter adapter("127.0.0.1", /*local_rank=*/0);
  std::vector<uint8_t> buf(64, 0);

  require(adapter.send_async(/*peer_rank=*/1, buf.data(), buf.size(),
                             /*request_id=*/1) == -1,
          "send_async without peer should fail");
  require(adapter.recv_async(/*peer_rank=*/1, buf.data(), buf.size(),
                             /*request_id=*/2) == -1,
          "recv_async without peer should fail");
  require(adapter.poll_completion(/*request_id=*/999),
          "poll_completion on unknown request should be treated as done");
  require(!adapter.wait_completion(/*request_id=*/999),
          "wait_completion on unknown request should fail");
  require(!adapter.request_failed(/*request_id=*/999),
          "unknown request should not report failure");
  adapter.release_request(/*request_id=*/999);
}

}  // namespace

void test_transport_tcp_adapter() {
  run_case("transport unit", "tcp adapter connect and transfer",
           test_tcp_connect_and_transfer);
  run_case("transport unit", "tcp adapter invalid peer paths",
           test_tcp_invalid_peer_paths);
}

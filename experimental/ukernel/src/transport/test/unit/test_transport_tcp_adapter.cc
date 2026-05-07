#include "adapter/tcp_adapter.h"
#include "test.h"
#include "test_utils.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <thread>
#include <vector>

namespace {

using UKernel::Transport::PeerConnectSpec;
using UKernel::Transport::PeerConnectType;
using UKernel::Transport::TcpPeerConnectSpec;
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
      PeerConnectSpec spec{};
      spec.peer_rank = 1;
      spec.type = PeerConnectType::Accept;
      spec.detail = TcpPeerConnectSpec{"127.0.0.1", 0};
      require(server.ensure_peer(spec), "server ensure_peer(accept) failed");
    } catch (...) {
      accept_error = std::current_exception();
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  PeerConnectSpec connect_spec{};
  connect_spec.peer_rank = 0;
  connect_spec.type = PeerConnectType::Connect;
  connect_spec.detail =
      TcpPeerConnectSpec{"127.0.0.1", server.get_listen_port()};
  require(client.ensure_peer(connect_spec),
          "client ensure_peer(connect) failed");
  accept_thread.join();
  if (accept_error) std::rethrow_exception(accept_error);

  require(client.has_peer(0), "client duplex path should be connected");
  require(server.has_peer(1), "server duplex path should be connected");

  std::vector<uint8_t> sendbuf(8 * 1024);
  std::vector<uint8_t> recvbuf(sendbuf.size(), 0);
  fill_pattern(sendbuf, 0x33);

  unsigned recv_req = server.recv_async(/*peer_rank=*/1, recvbuf.data(),
                                        recvbuf.size(), /*local_buffer_id=*/0,
                                        /*bounce_provider=*/nullptr);
  require(recv_req != 0, "server recv_async should succeed");
  unsigned send_req =
      client.send_async(/*peer_rank=*/0, sendbuf.data(), sendbuf.size(),
                        /*local_buffer_id=*/0, std::nullopt,
                        /*bounce_provider=*/nullptr);
  require(send_req != 0, "client send_async should succeed");

  require(client.wait_completion(send_req), "client wait_completion failed");
  require(server.wait_completion(recv_req), "server wait_completion failed");
  require(!client.request_failed(send_req), "client request should not fail");
  require(!server.request_failed(recv_req), "server request should not fail");
  require(sendbuf == recvbuf, "tcp adapter payload mismatch");

  client.release_request(send_req);
  server.release_request(recv_req);
}

void test_tcp_invalid_peer_paths() {
  TcpTransportAdapter adapter("127.0.0.1", /*local_rank=*/0);
  std::vector<uint8_t> buf(64, 0);

  require(adapter.send_async(/*peer_rank=*/1, buf.data(), buf.size(),
                             /*local_buffer_id=*/0, std::nullopt,
                             /*bounce_provider=*/nullptr) == 0,
          "send_async without peer should fail");
  require(adapter.recv_async(/*peer_rank=*/1, buf.data(), buf.size(),
                             /*local_buffer_id=*/0,
                             /*bounce_provider=*/nullptr) == 0,
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

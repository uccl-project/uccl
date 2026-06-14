#include "adapter/tcp_adapter.h"
#include "test.h"
#include "test_utils.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace {

using UKernel::Transport::PeerConnectSpec;
using UKernel::Transport::PeerConnectType;
using UKernel::Transport::TcpPeerConnectSpec;
using UKernel::Transport::TcpTransportAdapter;
using UKernel::Transport::TransportAdapter;
using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

void fill_pattern(std::vector<uint8_t>& buf, uint8_t seed) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>((seed + i) & 0xFF);
  }
}

void test_tcp_connect_and_transfer() {
  TcpTransportAdapter server("127.0.0.1", /*local_rank=*/0, /*gpu_id=*/-1);
  TcpTransportAdapter client("127.0.0.1", /*local_rank=*/1, /*gpu_id=*/-1);

  std::exception_ptr accept_error;
  std::thread accept_thread([&] {
    try {
      PeerConnectSpec spec{};
      spec.peer_rank = 1;
      spec.type = PeerConnectType::Accept;
      spec.detail = TcpPeerConnectSpec{"127.0.0.1", 0};
      require(server.ensure_wait_path(spec),
              "server ensure_wait_path(accept) failed");
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
  require(client.ensure_put_path(connect_spec),
          "client ensure_put_path(connect) failed");
  accept_thread.join();
  if (accept_error) std::rethrow_exception(accept_error);

  require(client.has_put_path(0), "client put path should be connected");
  require(server.has_wait_path(1), "server wait path should be connected");
  require(!server.has_put_path(1), "server put path should not be connected");
  require(!client.has_wait_path(0), "client wait path should not be connected");

  PeerConnectSpec wrong_put_spec{};
  wrong_put_spec.peer_rank = 1;
  wrong_put_spec.type = PeerConnectType::Accept;
  wrong_put_spec.detail = TcpPeerConnectSpec{"127.0.0.1", 0};
  require(!server.ensure_put_path(wrong_put_spec),
          "ensure_put_path should reject Accept type");

  PeerConnectSpec wrong_wait_spec{};
  wrong_wait_spec.peer_rank = 0;
  wrong_wait_spec.type = PeerConnectType::Connect;
  wrong_wait_spec.detail =
      TcpPeerConnectSpec{"127.0.0.1", server.get_listen_port()};
  require(!client.ensure_wait_path(wrong_wait_spec),
          "ensure_wait_path should reject Connect type");

  std::vector<uint8_t> sendbuf(8 * 1024);
  std::vector<uint8_t> recvbuf(sendbuf.size(), 0);
  fill_pattern(sendbuf, 0x33);

  TransportAdapter::WaitTarget wait_target;
  wait_target.local_ptr = recvbuf.data();
  wait_target.len = recvbuf.size();
  wait_target.local_buffer_id = 0;
  unsigned recv_req =
      server.wait_signal_async(/*peer_rank=*/1, /*expected_tag=*/0, wait_target,
                              /*comm_rid=*/2);
  require(recv_req != 0, "server wait_signal_async should succeed");
  unsigned send_req =
      client.send_put_async(/*peer_rank=*/0, sendbuf.data(),
                            /*local_buffer_id=*/0,
                            /*remote_ptr=*/nullptr, /*remote_buffer_id=*/0,
                            sendbuf.size(), /*comm_rid=*/1);
  require(send_req != 0, "client send_put_async should succeed");

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  require(sendbuf == recvbuf, "tcp adapter payload mismatch");

  unsigned bad_wait =
      client.wait_signal_async(/*peer_rank=*/0, /*expected_tag=*/123, std::nullopt,
                              /*comm_rid=*/5);
  require(bad_wait == 0, "wait_signal_async without wait path should fail");
  unsigned bad_put =
      server.send_put_async(/*peer_rank=*/1, sendbuf.data(),
                            /*buffer_id=*/0,
                            /*remote_ptr=*/nullptr, /*remote_buffer_id=*/0,
                            sendbuf.size(), /*comm_rid=*/6);
  require(bad_put == 0, "send_put_async without put path should fail");

  unsigned signal_wait =
      server.wait_signal_async(/*peer_rank=*/1, /*expected_tag=*/42, std::nullopt,
                              /*comm_rid=*/3);
  require(signal_wait != 0, "signal wait should enqueue");
  unsigned signal_send = client.send_signal_async(/*peer_rank=*/0, /*tag=*/42,
                                                  /*comm_rid=*/4);
  require(signal_send != 0, "signal send should enqueue");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  unsigned signal_wait_bad =
      server.wait_signal_async(/*peer_rank=*/1, /*expected_tag=*/99, std::nullopt,
                               /*comm_rid=*/7);
  require(signal_wait_bad != 0, "mismatched signal wait should enqueue");
  unsigned signal_send_bad = client.send_signal_async(/*peer_rank=*/0, /*tag=*/100,
                                                      /*comm_rid=*/8);
  require(signal_send_bad != 0, "mismatched signal send should enqueue");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

void test_tcp_invalid_peer_paths() {
  TcpTransportAdapter adapter("127.0.0.1", /*local_rank=*/0, /*gpu_id=*/-1);
  std::vector<uint8_t> buf(64, 0);

  require(adapter.send_put_async(/*peer_rank=*/1, buf.data(),
                                 /*local_buffer_id=*/0,
                                 /*remote_ptr=*/nullptr, /*remote_buffer_id=*/0,
                                 buf.size(), /*comm_rid=*/1) == 0,
          "send_put_async without peer should fail");
  TransportAdapter::WaitTarget wait_target;
  wait_target.local_ptr = buf.data();
  wait_target.len = buf.size();
  wait_target.local_buffer_id = 0;
  require(adapter.wait_signal_async(/*peer_rank=*/1, /*expected_tag=*/0,
                                    std::move(wait_target), /*comm_rid=*/2) == 0,
          "wait_signal_async without peer should fail");
}

}  // namespace

void test_transport_tcp_adapter() {
  run_case("transport unit", "tcp adapter connect and transfer",
           test_tcp_connect_and_transfer);
  run_case("transport unit", "tcp adapter invalid peer paths",
           test_tcp_invalid_peer_paths);
}

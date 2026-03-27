#include "peer_session.h"
#include "test.h"
#include "test_utils.h"

namespace {

using UKernel::Transport::CommunicatorMeta;
using UKernel::Transport::PeerSession;
using UKernel::Transport::PeerSessionManager;
using UKernel::Transport::PeerTransportKind;
using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

void test_peer_session_state_machine() {
  PeerSession session(/*peer_rank=*/3);
  require(session.peer_rank() == 3, "peer rank should be preserved");
  require(!session.has_meta(), "fresh peer session should not have meta");
  require(session.transport_kind() == PeerTransportKind::Unknown,
          "fresh peer session should start with unknown transport");
  require(!session.send_ready(), "fresh peer session send path should be false");
  require(!session.recv_ready(), "fresh peer session recv path should be false");

  CommunicatorMeta meta{};
  meta.host_id = "host-a";
  meta.ip = "10.0.0.1";
  meta.local_id = 7;
  session.set_meta(meta);
  session.set_transport_kind(PeerTransportKind::Ipc);
  session.set_send_ready(true);
  session.set_recv_ready(true);

  require(session.has_meta(), "peer session should expose set meta");
  require(session.meta().host_id == "host-a",
          "peer session should retain host id");
  require(session.transport_kind() == PeerTransportKind::Ipc,
          "peer session transport kind mismatch");
  require(session.send_ready(), "peer session send readiness mismatch");
  require(session.recv_ready(), "peer session recv readiness mismatch");

  session.reset();
  require(!session.has_meta(), "reset should clear meta presence");
  require(session.transport_kind() == PeerTransportKind::Unknown,
          "reset should clear transport kind");
  require(!session.send_ready(), "reset should clear send readiness");
  require(!session.recv_ready(), "reset should clear recv readiness");
}

void test_peer_session_manager_paths() {
  PeerSessionManager manager(/*world_size=*/4);
  require(manager.world_size() == 4, "manager world size mismatch");
  require(manager.get(-1) == nullptr, "negative rank should be rejected");
  require(manager.get(4) == nullptr, "out-of-range rank should be rejected");

  auto* peer1 = manager.get(1);
  require(peer1 != nullptr, "expected valid peer session at rank 1");
  peer1->set_send_ready(true);
  peer1->set_recv_ready(false);
  require(manager.has_peer_send_path(1), "manager send path should reflect peer");
  require(!manager.has_peer_recv_path(1),
          "manager recv path should reflect peer");
  require(!manager.has_peer_send_path(3),
          "untouched peer send path should remain false");
  require(!manager.has_peer_recv_path(7),
          "out-of-range recv path query should be false");
}

}  // namespace

void test_transport_peer_session() {
  run_case("transport unit", "peer session state machine",
           test_peer_session_state_machine);
  run_case("transport unit", "peer session manager paths",
           test_peer_session_manager_paths);
}

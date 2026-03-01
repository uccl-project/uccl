#!/usr/bin/env python3
"""
Phase 1 unit test — UCCL intra-node connection detection via the nixl code path.

Two nixl agents run as separate processes on the same node.  The test replicates
the exact sequence that nixl uses internally:

  getConnInfo -> (out-of-band metadata exchange) -> loadRemoteConnInfo
  registerMem -> (xfer desc exchange) -> prepXfer -> postXfer -> checkXfer

Expected output from uccl_engine.cc:
  uccl_engine_connect: connection to <ip> is intra-node
  uccl_engine_accept:  connection from <ip> is intra-node

A GPU-to-GPU WRITE (client writes ones into server's GPU buffer) is performed
to confirm the connection is fully functional, not just established.
"""

from __future__ import annotations

import sys
import multiprocessing
import torch

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as exc:
    sys.stderr.write("Failed to import NIXL\n")
    raise

BUF_ELEMS = 1024  # floats  (~4 KB)


# ---------------------------------------------------------------------------
# Process bodies
# ---------------------------------------------------------------------------

def server_proc(s_to_c, c_to_s, xfer_pipe, done_pipe):
    """
    Server side:
      - publishes its conn metadata
      - receives client conn metadata, calls add_remote_agent (triggers accept)
      - registers a zeroed receive buffer
      - sends its xfer descs to client
      - waits for client "DONE" signal
      - asserts the buffer was filled with ones by the client WRITE
    """
    config = nixl_agent_config(backends=["UCCL"])
    agent = nixl_agent("server", config)

    # Step 1: exchange connection metadata
    # Server sends first so client can unblock on recv without deadlock
    local_meta = agent.get_agent_metadata()
    s_to_c.send(bytes(local_meta))
    remote_meta = c_to_s.recv()
    agent.add_remote_agent(remote_meta)

    # Step 2: register receive buffer (zeros on GPU — client will overwrite with ones)
    buf = torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
    descs = agent.get_reg_descs([buf])
    reg_descs = agent.register_memory(descs)
    local_xfer_descs = reg_descs.trim()

    # Step 3: send xfer descs to client so it can target this buffer
    xfer_pipe.send(agent.get_serialized_descs(local_xfer_descs))

    # Step 4: wait for client to signal transfer complete
    done_pipe.recv()

    # Step 5: verify data (sync GPU before reading)
    torch.cuda.synchronize()
    assert buf.mean().item() == 1.0, \
        f"[server] FAIL: expected all 1.0, got mean={buf.mean().item()}"
    print("[server] PASS: GPU buffer filled with ones by client WRITE", flush=True)

    agent.deregister_memory(reg_descs)
    agent.remove_remote_agent("client")


def client_proc(s_to_c, c_to_s, xfer_pipe, done_pipe):
    """
    Client side:
      - receives server conn metadata, publishes its own
      - calls add_remote_agent (triggers connect)
      - registers a ones send buffer
      - receives server xfer descs, initializes and posts WRITE
      - polls until DONE, then signals server
    """
    config = nixl_agent_config(backends=["UCCL"])
    agent = nixl_agent("client", config)

    # Step 1: exchange connection metadata
    remote_meta = s_to_c.recv()
    local_meta = agent.get_agent_metadata()
    c_to_s.send(bytes(local_meta))
    agent.add_remote_agent(remote_meta)

    # Step 2: register send buffer (all ones on GPU)
    buf = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
    descs = agent.get_reg_descs([buf])
    reg_descs = agent.register_memory(descs)
    local_xfer_descs = reg_descs.trim()

    # Step 3: receive server xfer descs
    remote_xfer_desc_bytes = xfer_pipe.recv()
    remote_xfer_descs = agent.deserialize_descs(remote_xfer_desc_bytes)

    # Step 4: initialize and post WRITE (client ones -> server buffer)
    handle = agent.initialize_xfer(
        "WRITE", local_xfer_descs, remote_xfer_descs, "server"
    )
    state = agent.transfer(handle)
    assert state != "ERR", "[client] transfer() returned ERR"

    while True:
        state = agent.check_xfer_state(handle)
        assert state != "ERR", "[client] check_xfer_state() returned ERR"
        if state == "DONE":
            break

    # Step 5: signal server
    done_pipe.send("DONE")
    print("[client] PASS: GPU-to-GPU WRITE completed successfully", flush=True)

    agent.release_xfer_handle(handle)
    agent.deregister_memory(reg_descs)
    agent.remove_remote_agent("server")


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------

def test_nixl_intranode():
    print("=== UCCL intra-node detection test (nixl code path) ===", flush=True)
    print("Look for 'is intra-node' in the uccl_engine output below.", flush=True)
    print("", flush=True)

    # Two unidirectional pipes for conn metadata exchange
    s_to_c_send, s_to_c_recv = multiprocessing.Pipe(duplex=False)
    c_to_s_send, c_to_s_recv = multiprocessing.Pipe(duplex=False)
    # Pipe for xfer descriptor delivery (server -> client)
    xfer_send, xfer_recv = multiprocessing.Pipe(duplex=False)
    # Pipe for done signal (client -> server)
    done_send, done_recv = multiprocessing.Pipe(duplex=False)

    srv = multiprocessing.Process(
        target=server_proc,
        args=(s_to_c_send, c_to_s_recv, xfer_send, done_recv),
    )
    cli = multiprocessing.Process(
        target=client_proc,
        args=(s_to_c_recv, c_to_s_send, xfer_recv, done_send),
    )

    srv.start()
    cli.start()
    srv.join()
    cli.join()

    assert srv.exitcode == 0, f"server process exited with code {srv.exitcode}"
    assert cli.exitcode == 0, f"client process exited with code {cli.exitcode}"

    print("", flush=True)
    print("=== test_nixl_intranode PASSED ===", flush=True)


if __name__ == "__main__":
    try:
        test_nixl_intranode()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)

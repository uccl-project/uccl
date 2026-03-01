#!/usr/bin/env python3
"""
Phase 1 unit test — UCCL intra-node connection detection via the nixl code path.

Two nixl agents are created in the same process on the same node.  The test
replicates the exact sequence that nixl uses internally:

  registerMem -> getConnInfo (get_agent_metadata) -> add_remote_agent (loadRemoteConnInfo)
  -> initialize_xfer (prepXfer) -> transfer (postXfer) -> checkXfer

Expected output from uccl_engine.cc:
  uccl_engine_connect: connection to <ip> is intra-node
  uccl_engine_accept:  connection from <ip> is intra-node

A GPU-to-GPU WRITE (client writes ones into server's GPU buffer) confirms the
connection is fully functional, not just established.
"""

from __future__ import annotations

import sys
import torch

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as exc:
    sys.stderr.write("Failed to import NIXL\n")
    raise

BUF_ELEMS = 1024  # floats (~4 KB)


def test_nixl_intranode():
    print("=== UCCL intra-node detection test (nixl code path) ===", flush=True)
    print("Look for 'is intra-node' in the uccl_engine output below.", flush=True)
    print("", flush=True)

    config = nixl_agent_config(backends=["UCCL"])
    server = nixl_agent("server", config)
    client = nixl_agent("client", config)

    # Verify the UCCL backend was actually loaded.  If NIXL_PLUGIN_DIR is not
    # set (or points to a directory that does not contain libplugin_UCCL.so),
    # nixl silently skips the backend and the test would fail with a cryptic
    # "no backends support remote operations" error.
    if "UCCL" not in server.backends or "UCCL" not in client.backends:
        raise RuntimeError(
            "UCCL backend not registered.  Build nixl with the UCCL plugin "
            "and set NIXL_PLUGIN_DIR to the directory containing "
            "libplugin_UCCL.so before running this test."
        )

    # Step 1: register GPU buffers FIRST.
    # get_agent_metadata() serialises both connection info AND the registered
    # memory's backend public data (fifo_item hex for UCCL).  add_remote_agent()
    # then loads both the connection and the remote memory section in one call.
    # If memory is registered after add_remote_agent, the remote section stays
    # empty and createXferReq cannot find any matching backend.
    srv_buf = torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
    cli_buf = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")

    srv_reg = server.register_memory(server.get_reg_descs([srv_buf]))
    cli_reg = client.register_memory(client.get_reg_descs([cli_buf]))

    # Step 2: exchange connection metadata and connect.
    # Both listener threads are already running and blocking in uccl_engine_accept().
    # add_remote_agent triggers loadRemoteConnInfo -> uccl_engine_connect, which
    # unblocks the peer's listener thread -> uccl_engine_accept logs "is intra-node".
    # It also loads the remote memory section (loadRemoteSections), making the
    # remote buffer's fifo_item available for prepXfer.
    server_meta = server.get_agent_metadata()
    client_meta = client.get_agent_metadata()
    server.add_remote_agent(client_meta)
    client.add_remote_agent(server_meta)

    # Step 3: build transfer descriptor lists.
    # The serialised form carries addr/len so nixl can look up the remote buffer
    # in the already-loaded remote section on the client side.
    srv_xfer_descs = srv_reg.trim()
    cli_xfer_descs = cli_reg.trim()

    remote_descs = client.deserialize_descs(server.get_serialized_descs(srv_xfer_descs))

    # Step 4: client WRITEs its ones buffer into the server's zeros buffer
    handle = client.initialize_xfer("WRITE", cli_xfer_descs, remote_descs, "server")
    state = client.transfer(handle)
    assert state != "ERR", "transfer() returned ERR"

    while True:
        state = client.check_xfer_state(handle)
        assert state != "ERR", "check_xfer_state() returned ERR"
        if state == "DONE":
            break

    # Step 5: verify (no cuda.synchronize needed — nixl/uccl guarantees completion)
    assert srv_buf.mean().item() == 1.0, \
        f"FAIL: expected all 1.0 in server buffer, got mean={srv_buf.mean().item()}"

    print("[client] PASS: GPU-to-GPU WRITE completed successfully", flush=True)
    print("[server] PASS: GPU buffer filled with ones", flush=True)

    # Cleanup
    client.release_xfer_handle(handle)
    client.deregister_memory(cli_reg)
    server.deregister_memory(srv_reg)
    client.remove_remote_agent("server")
    server.remove_remote_agent("client")

    print("", flush=True)
    print("=== test_nixl_intranode PASSED ===", flush=True)


if __name__ == "__main__":
    try:
        test_nixl_intranode()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)

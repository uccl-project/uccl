"""Offset-based P2P transfer correctness suite using current ukernel_p2p API."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import torch
import ukernel_p2p as p2p


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def elem_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def build_tensor(rank: int, case_id: int, total_elems: int) -> torch.Tensor:
    base = rank * 1_000_000 + case_id * 10_000
    host = torch.arange(total_elems, dtype=torch.float32) + float(base)
    return host.cuda()


def expected_payload(sender_rank: int, case_id: int, start_elem: int,
                     length: int) -> torch.Tensor:
    base = sender_rank * 1_000_000 + case_id * 10_000
    host = torch.arange(start_elem, start_elem + length, dtype=torch.float32) + float(base)
    return host.cuda()


def verify_segment(tensor: torch.Tensor, recv_off_el: int, payload_el: int,
                   expected: torch.Tensor, guard: float, case_name: str) -> None:
    left = tensor[:recv_off_el]
    mid = tensor[recv_off_el: recv_off_el + payload_el]
    right = tensor[recv_off_el + payload_el:]
    if left.numel() > 0:
        require(torch.all(left == guard).item(), f"{case_name}: left guard corrupted")
    if not torch.equal(mid, expected):
        require(False, f"{case_name}: payload mismatch")
    if right.numel() > 0:
        require(torch.all(right == guard).item(), f"{case_name}: right guard corrupted")


SEND_BASE = 20_000
RECV_BASE = 10_000


def register_bufs(comm, selected: str, peer: int, case_id: int,
                  rank: int, send_buf, recv_buf):
    sid = SEND_BASE + case_id * 10 + rank
    rid = RECV_BASE + case_id * 10 + rank
    rm_rid = RECV_BASE + case_id * 10 + peer
    require(comm.reg_rdma(sid, send_buf, publish=False), "reg_rdma(send)")
    require(comm.reg_rdma(rid, recv_buf, publish=True), "reg_rdma(recv)")
    if selected == "ipc":
        require(comm.reg_ipc(rid, recv_buf, publish=True), "reg_ipc(recv)")
        require(comm.wait_ipc(peer, rm_rid), "wait_ipc(peer recv)")
    elif selected in ("uccl", "rdma"):
        require(comm.wait_mr(peer, rm_rid), "wait_mr(peer recv)")
    return sid, rid, rm_rid


def unreg_bufs(comm, selected: str, case_id: int, rank: int):
    sid = SEND_BASE + case_id * 10 + rank
    rid = RECV_BASE + case_id * 10 + rank
    if selected == "ipc":
        comm.unreg_ipc(rid)
    comm.unreg_rdma(rid)
    comm.unreg_rdma(sid)


def poll_rid(comm, rid, timeout_s: float = 10.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if rid in comm.poll([rid]):
            return
    raise RuntimeError(f"timeout waiting for rid={rid}")


def run_oneway(comm, rank: int, peer: int, selected: str,
               sender_rank: int, case_id: int,
               send_off_el: int, recv_off_el: int,
               payload_el: int, total_el: int,
               case_name: str, tag: int):
    guard = -7777.0 - float(case_id)
    eb = elem_bytes(torch.float32)

    send_buf = build_tensor(rank, case_id, total_el)
    recv_buf = torch.full((total_el,), guard, device="cuda", dtype=torch.float32)
    sid, rid, rm_rid = register_bufs(comm, selected, peer, case_id,
                                      rank, send_buf, recv_buf)
    try:
        if rank == sender_rank:
            req = comm.send_put_async(peer, local_buf=sid,
                                       off=send_off_el * eb,
                                       len=payload_el * eb,
                                       remote_buf=rm_rid,
                                       remote_off=recv_off_el * eb)
            require(req != 0, f"{case_name}: send_put_async failed")
            poll_rid(comm, req)
            comm.signal(peer, tag)
        else:
            req = comm.wait_signal_async(peer, tag)
            require(req != 0, f"{case_name}: wait_signal_async failed")
            poll_rid(comm, req)
            exp = expected_payload(sender_rank, case_id, send_off_el, payload_el)
            verify_segment(recv_buf, recv_off_el, payload_el, exp, guard, case_name)
        require(comm.barrier(f"{case_name}_done", 30000), f"{case_name}: barrier failed")
    finally:
        unreg_bufs(comm, selected, case_id, rank)


def run_bidir(comm, rank: int, peer: int, selected: str):
    case_id = 99
    total_el = 4096
    payload_el = 777
    guard = -9999.0
    eb = elem_bytes(torch.float32)
    tag0, tag1 = 400, 401
    my_tag = tag0 if rank == 0 else tag1
    peer_tag = tag1 if rank == 0 else tag0
    send_off_el = 23 + rank * 11
    recv_off_el = 71 + peer * 13

    send_buf = build_tensor(rank, case_id, total_el)
    recv_buf = torch.full((total_el,), guard, device="cuda", dtype=torch.float32)
    sid, rid, rm_rid = register_bufs(comm, selected, peer, case_id,
                                      rank, send_buf, recv_buf)
    try:
        put_rid = comm.send_put_async(peer, local_buf=sid,
                                       off=send_off_el * eb,
                                       len=payload_el * eb,
                                       remote_buf=rm_rid,
                                       remote_off=recv_off_el * eb)
        require(put_rid != 0, "bidir: send_put_async failed")
        poll_rid(comm, put_rid)
        comm.signal(peer, my_tag)

        wait_rid = comm.wait_signal_async(peer, peer_tag)
        require(wait_rid != 0, "bidir: wait_signal_async failed")
        poll_rid(comm, wait_rid)

        # Peer wrote into our buffer at offset 71 + rank*13.
        my_recv_off_el = 71 + rank * 13
        exp = expected_payload(peer, case_id, 23 + peer * 11, payload_el)
        verify_segment(recv_buf, my_recv_off_el, payload_el, exp, guard,
                       "bidir_offset")
        require(comm.barrier("bidir_done", 30000), "bidir: barrier failed")
    finally:
        unreg_bufs(comm, selected, case_id, rank)


def main() -> None:
    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 2)
    local_rank = env_int("LOCAL_RANK", rank)
    master_addr = env_str("MASTER_ADDR", "127.0.0.1")
    xport = env_int("EXCHANGER_PORT", 29620)
    transport = env_str("TRANSPORT", "auto")

    require(world == 2, "needs WORLD_SIZE=2")
    torch.cuda.set_device(local_rank)

    comm = p2p.Communicator(gpu_id=local_rank, rank=rank, world_size=world,
                            exchanger_ip=master_addr, exchanger_port=xport,
                            transport=transport)
    peer = 1 - rank
    if rank < peer:
        require(comm.connect_peer(peer), "connect_peer failed")
    else:
        require(comm.accept_peer(peer), "accept_peer failed")

    selected = comm.peer_transport(peer)
    print(f"[rank {rank}] transport={selected}")

    if transport != "auto":
        require(selected == transport,
                f"transport mismatch: {transport} vs {selected}")

    cases = [
        ("oneway_full_r0_r1",       0,   0,   0, 1024, 2048, 1, 10),
        ("oneway_offset_r1_r0",     1,  13, 127,  333, 2048, 2, 11),
        ("oneway_offset_r0_r1",     0, 511,   7,  257, 2048, 3, 12),
    ]
    for name, sender, so, ro, pl, tot, cid, tag in cases:
        run_oneway(comm, rank, peer, selected, sender, cid,
                   so, ro, pl, tot, name, tag)
        if rank == 0:
            print(f"[{selected}] {name}: PASS")

    run_bidir(comm, rank, peer, selected)
    if rank == 0:
        print(f"[{selected}] bidir_offset: PASS")
        print(f"[{selected}] all checks passed")


if __name__ == "__main__":
    main()

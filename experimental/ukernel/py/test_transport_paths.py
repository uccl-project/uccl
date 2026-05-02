import os
from dataclasses import dataclass
from typing import List

import torch
import ukernel_p2p as p2p


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def visible_physical_gpu(local_rank: int) -> str:
    vis = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return str(local_rank)
    parts = [p.strip() for p in vis.split(",") if p.strip()]
    if 0 <= local_rank < len(parts):
        return parts[local_rank]
    return "unknown"


def peer_access_capability(local_rank: int, peer_local_rank: int) -> str:
    try:
        fn = getattr(torch.cuda, "device_can_access_peer", None)
        if fn is None:
            fn = getattr(torch.cuda, "can_device_access_peer", None)
        if fn is None:
            return "n/a"
        return "yes" if bool(fn(local_rank, peer_local_rank)) else "no"
    except Exception:
        return "n/a"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def elem_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True)
class OneWayCase:
    name: str
    sender_rank: int
    send_offset_elems: int
    recv_offset_elems: int
    payload_elems: int
    total_elems: int
    case_id: int


def build_tensor_payload(rank: int, case_id: int, total_elems: int) -> torch.Tensor:
    # Deterministic payload to make per-element validation easy.
    base = rank * 1_000_000 + case_id * 10_000
    host = torch.arange(total_elems, dtype=torch.float32) + float(base)
    return host.cuda()


def expected_payload(sender_rank: int, case_id: int, start_elem: int, length: int) -> torch.Tensor:
    base = sender_rank * 1_000_000 + case_id * 10_000
    host = torch.arange(start_elem, start_elem + length, dtype=torch.float32) + float(base)
    return host.cuda()


def verify_segment(
    tensor: torch.Tensor,
    recv_offset_elems: int,
    payload_elems: int,
    expected: torch.Tensor,
    guard_value: float,
    case_name: str,
) -> None:
    left = tensor[:recv_offset_elems]
    mid = tensor[recv_offset_elems : recv_offset_elems + payload_elems]
    right = tensor[recv_offset_elems + payload_elems :]

    if left.numel() > 0:
        require(torch.all(left == guard_value).item(), f"{case_name}: left guard region corrupted")
    if not torch.equal(mid, expected):
        diff = (mid != expected).nonzero()
        if diff.numel() > 0:
            i = int(diff[0].item())
            got_v = float(mid[i].item())
            exp_v = float(expected[i].item())
            raise RuntimeError(
                f"{case_name}: payload mismatch at local_idx={i}, got={got_v}, expected={exp_v}"
            )
        raise RuntimeError(f"{case_name}: payload mismatch")
    if right.numel() > 0:
        require(torch.all(right == guard_value).item(), f"{case_name}: right guard region corrupted")


def setup_peer(comm: p2p.Communicator, rank: int, peer: int) -> None:
    # Keep public connect/accept semantics; adapter handles duplex details internally.
    if rank < peer:
        require(comm.connect_peer(peer), "connect_peer failed")
    else:
        require(comm.accept_peer(peer), "accept_peer failed")


def send_buffer_id(case_id: int, owner_rank: int) -> int:
    return 20_000 + case_id * 10 + owner_rank


def recv_buffer_id(case_id: int, owner_rank: int) -> int:
    return 10_000 + case_id * 10 + owner_rank


def register_case_buffers(
    comm: p2p.Communicator,
    selected_transport: str,
    peer: int,
    case_id: int,
    rank: int,
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
) -> int:
    local_send_id = send_buffer_id(case_id, rank)
    local_recv_id = recv_buffer_id(case_id, rank)
    remote_recv_id = recv_buffer_id(case_id, peer)
    require(comm.reg_rdma(local_send_id, send_buf, publish=False), "reg_rdma(send) failed")
    require(comm.reg_rdma(local_recv_id, recv_buf, publish=True), "reg_rdma(recv) failed")
    if selected_transport == "ipc":
        require(comm.reg_ipc(local_recv_id, recv_buf, publish=True), "reg_ipc(recv) failed")
        require(comm.wait_ipc(peer, remote_recv_id), "wait_ipc(peer recv buffer) failed")
    elif selected_transport == "uccl":
        require(comm.wait_mr(peer, remote_recv_id), "wait_mr(peer recv buffer) failed")
    return remote_recv_id


def unregister_case_buffers(
    comm: p2p.Communicator,
    selected_transport: str,
    case_id: int,
    rank: int,
) -> None:
    local_send_id = send_buffer_id(case_id, rank)
    local_recv_id = recv_buffer_id(case_id, rank)
    if selected_transport == "ipc":
        comm.unreg_ipc(local_recv_id)
    comm.unreg_rdma(local_recv_id)
    comm.unreg_rdma(local_send_id)


def run_oneway_case(
    comm: p2p.Communicator,
    rank: int,
    peer: int,
    selected_transport: str,
    case: OneWayCase,
) -> None:
    guard = -7777.0 - float(case.case_id)
    dtype = torch.float32
    eb = elem_bytes(dtype)

    send_buf = build_tensor_payload(rank, case.case_id, case.total_elems)
    recv_buf = torch.full((case.total_elems,), guard, device="cuda", dtype=dtype)

    send_off_b = case.send_offset_elems * eb
    recv_off_b = case.recv_offset_elems * eb
    payload_b = case.payload_elems * eb
    remote_recv_id = register_case_buffers(
        comm, selected_transport, peer, case.case_id, rank, send_buf, recv_buf
    )

    try:
        if rank == case.sender_rank:
            req = comm.isend(
                peer,
                send_buf,
                offset=send_off_b,
                len=payload_b,
                remote_buffer_id=remote_recv_id,
                remote_offset=recv_off_b,
            )
            require(req != 0, f"{case.name}: isend returned 0")
            require(comm.wait_finish(req), f"{case.name}: wait_finish(isend) failed")
        else:
            req = comm.irecv(peer, recv_buf, offset=recv_off_b, len=payload_b)
            require(req != 0, f"{case.name}: irecv returned 0")
            require(comm.wait_finish(req), f"{case.name}: wait_finish(irecv) failed")
            exp = expected_payload(
                sender_rank=case.sender_rank,
                case_id=case.case_id,
                start_elem=case.send_offset_elems,
                length=case.payload_elems,
            )
            verify_segment(
                recv_buf,
                recv_offset_elems=case.recv_offset_elems,
                payload_elems=case.payload_elems,
                expected=exp,
                guard_value=guard,
                case_name=case.name,
            )
        require(
            comm.barrier(f"{case.name}_done", 30000),
            f"{case.name}: completion barrier failed",
        )
    finally:
        unregister_case_buffers(comm, selected_transport, case.case_id, rank)


def run_bidirectional_offset_case(
    comm: p2p.Communicator, rank: int, peer: int, selected_transport: str
) -> None:
    case_id = 99
    total_elems = 4096
    payload_elems = 777
    send_offset_elems = 23 + rank * 11
    recv_offset_elems = 71 + rank * 13
    guard = -9999.0
    eb = elem_bytes(torch.float32)

    send_buf = build_tensor_payload(rank, case_id, total_elems)
    recv_buf = torch.full((total_elems,), guard, device="cuda", dtype=torch.float32)
    remote_recv_id = register_case_buffers(
        comm, selected_transport, peer, case_id, rank, send_buf, recv_buf
    )

    try:
        recv_req = comm.irecv(
            peer,
            recv_buf,
            offset=recv_offset_elems * eb,
            len=payload_elems * eb,
        )
        require(recv_req != 0, "bidir_offset: irecv returned 0")

        send_req = comm.isend(
            peer,
            send_buf,
            offset=send_offset_elems * eb,
            len=payload_elems * eb,
            remote_buffer_id=remote_recv_id,
            remote_offset=(71 + peer * 13) * eb,
        )
        require(send_req != 0, "bidir_offset: isend returned 0")

        require(
            comm.wait_finish_multi([send_req, recv_req]),
            "bidir_offset: wait_finish_multi failed",
        )

        exp = expected_payload(
            sender_rank=peer,
            case_id=case_id,
            start_elem=23 + peer * 11,
            length=payload_elems,
        )
        verify_segment(
            recv_buf,
            recv_offset_elems=recv_offset_elems,
            payload_elems=payload_elems,
            expected=exp,
            guard_value=guard,
            case_name="bidir_offset",
        )
        require(comm.barrier("bidir_offset_done", 30000), "bidir_offset: completion barrier failed")
    finally:
        unregister_case_buffers(comm, selected_transport, case_id, rank)


def build_cases() -> List[OneWayCase]:
    return [
        OneWayCase(
            name="oneway_full_rank0_to_rank1",
            sender_rank=0,
            send_offset_elems=0,
            recv_offset_elems=0,
            payload_elems=1024,
            total_elems=2048,
            case_id=1,
        ),
        OneWayCase(
            name="oneway_offset_rank1_to_rank0",
            sender_rank=1,
            send_offset_elems=13,
            recv_offset_elems=127,
            payload_elems=333,
            total_elems=2048,
            case_id=2,
        ),
        OneWayCase(
            name="oneway_offset_rank0_to_rank1",
            sender_rank=0,
            send_offset_elems=511,
            recv_offset_elems=7,
            payload_elems=257,
            total_elems=2048,
            case_id=3,
        ),
    ]


def main() -> None:
    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 2)
    local_rank = env_int("LOCAL_RANK", rank)
    master_addr = env_str("MASTER_ADDR", "127.0.0.1")
    exchanger_port = env_int("EXCHANGER_PORT", 29620)
    transport = env_str("TRANSPORT", "auto")

    require(world == 2, "test_transport_paths requires WORLD_SIZE=2")
    torch.cuda.set_device(local_rank)

    comm = p2p.Communicator(
        gpu_id=local_rank,
        rank=rank,
        world_size=world,
        exchanger_ip=master_addr,
        exchanger_port=exchanger_port,
        transport=transport,
    )
    peer = 1 - rank
    setup_peer(comm, rank, peer)

    selected = comm.peer_transport(peer)
    same_host = comm.same_host(peer)
    peer_local_rank = 1 - local_rank
    p2p_cap = peer_access_capability(local_rank, peer_local_rank)
    local_phys = visible_physical_gpu(local_rank)
    peer_phys = visible_physical_gpu(peer_local_rank)
    print(
        f"[rank {rank}] local_rank={local_rank} gpu={local_phys} "
        f"peer_rank={peer} peer_local_rank={peer_local_rank} peer_gpu={peer_phys} "
        f"transport={selected} same_host={same_host} canAccessPeer={p2p_cap}"
    )

    if transport != "auto":
        require(
            selected == transport,
            f"peer transport mismatch: requested={transport}, selected={selected}",
        )

    for case in build_cases():
        run_oneway_case(comm, rank, peer, selected, case)
        if rank == 0:
            print(f"[transport={selected}] {case.name}: PASS")

    run_bidirectional_offset_case(comm, rank, peer, selected)
    if rank == 0:
        print(f"[transport={selected}] bidir_offset: PASS")
        print(f"[transport={selected}] all transport path correctness checks passed")


if __name__ == "__main__":
    main()

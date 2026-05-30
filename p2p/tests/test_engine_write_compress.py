"""
Correctness test for compressed RDMA WRITE (split_only strategy).

Rank 0 (writer) writes a tensor filled with a per-iteration constant.
Rank 1 (receiver) verifies the decompressed result after each iteration.

Each iteration is strictly serialized via a gloo handshake: writer waits for
receiver's "verified" signal before sending the next write. This prevents a
later WriteReqMeta from being processed by the background thread before the
current iteration's verification completes.

Usage (two nodes):
  UCCL_P2P_COMPRESS_STRATEGY=split_only \\
  torchrun --nnodes=2 --nproc_per_node=1 \\
      --node-rank=X --master_addr=<IP> \\
      tests/test_engine_write_compress.py [--size 67108864] [--dtype bfloat16]
"""

from __future__ import annotations
import argparse, os, sys, time
import torch
import torch.distributed as dist
from uccl import p2p

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
_FLOAT_TYPES = {
    torch.float32: p2p.FloatType.kFloat32,
    torch.bfloat16: p2p.FloatType.kBFloat16,
    torch.float16: p2p.FloatType.kFloat16,
}


def make_fill_tensor(n_bytes: int, dtype: torch.dtype, value: float) -> torch.Tensor:
    item_size = torch.tensor(0, dtype=dtype).element_size()
    n_elems = n_bytes // item_size
    return torch.full((n_elems,), value, dtype=dtype, device="cuda").contiguous()


def gloo_send(value: int, dst: int) -> None:
    dist.send(torch.tensor([value], dtype=torch.int32), dst=dst)


def gloo_recv(src: int) -> int:
    t = torch.zeros(1, dtype=torch.int32)
    dist.recv(t, src=src)
    return t.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--size",
        type=int,
        default=64 * 1024 * 1024,
        help="bytes to transfer per iteration (>2 MB for compression)",
    )
    ap.add_argument("--dtype", choices=list(_DTYPES), default="bfloat16")
    ap.add_argument("--iters", type=int, default=128)
    args = ap.parse_args()

    dtype = _DTYPES[args.dtype]
    float_type = _FLOAT_TYPES[dtype]

    strategy = os.environ.get("UCCL_P2P_COMPRESS_STRATEGY", "")
    if strategy not in ("split", "split_only"):
        print(
            f"[WARN] UCCL_P2P_COMPRESS_STRATEGY={strategy!r}; "
            "set to 'split_only' for compression to activate"
        )

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    torch.cuda.set_device(0)

    ep = p2p.Endpoint(local_gpu_idx=0)
    local_meta = bytes(ep.get_metadata())

    # Exchange endpoint metadata
    meta_len = len(local_meta)
    if rank == 0:
        dist.send(torch.ByteTensor(list(local_meta)), dst=1)
        remote_t = torch.zeros(meta_len, dtype=torch.uint8)
        dist.recv(remote_t, src=1)
    else:
        remote_t = torch.zeros(meta_len, dtype=torch.uint8)
        dist.recv(remote_t, src=0)
        dist.send(torch.ByteTensor(list(local_meta)), dst=0)
    remote_meta = bytes(remote_t.tolist())

    if rank == 0:
        ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_meta)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
        assert ok, "connect failed"

        src = make_fill_tensor(args.size, dtype, 1.0)
        ok, mr_id = ep.reg(src.data_ptr(), src.nbytes, floatType=float_type)
        assert ok

        blob_t = torch.zeros(64, dtype=torch.uint8)
        dist.recv(blob_t, src=1)
        fifo_blob = bytes(blob_t.numpy())

        for i in range(args.iters):
            fill_val = float(i + 1)
            src.fill_(fill_val)
            torch.cuda.synchronize()

            ok, tid = ep.write_async(
                conn_id, mr_id, src.data_ptr(), src.nbytes, fifo_blob
            )
            assert ok, f"write_async failed iter={i}"
            while True:
                ok, done = ep.poll_async(tid)
                assert ok
                if done:
                    break

            # poll_async returns when data WCs arrive; WriteReqMeta may still
            # be in the NIC send queue. Sleep so the NIC has time to send it.
            time.sleep(0.5)

            # Signal receiver that WriteReqMeta has been transmitted, then
            # wait for receiver to confirm decompression + verification done.
            # This strict handshake prevents iter N+1 data from arriving at
            # receiver before iter N is verified.
            gloo_send(i, dst=1)
            result = gloo_recv(src=1)
            if result != 0:
                print(f"[Writer] receiver reported FAIL on iter={i}")
                dist.destroy_process_group()
                sys.exit(1)
            print(f"[Writer] iter={i} fill={fill_val} confirmed OK")

        print("[Writer] all iterations complete")

    else:
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "accept failed"
        print(f"[Receiver] accepted from {r_ip}")

        dst = make_fill_tensor(args.size, dtype, 0.0)
        ok, mr_id = ep.reg(dst.data_ptr(), dst.nbytes, floatType=float_type)
        assert ok

        ok, fifo_blob = ep.advertise(conn_id, mr_id, dst.data_ptr(), dst.nbytes)
        assert ok and len(fifo_blob) == 64
        dist.send(torch.ByteTensor(list(fifo_blob)), dst=0)

        for i in range(args.iters):
            fill_val = float(i + 1)

            # Wait for writer's "data ready" signal, then allow time for the
            # background thread to process WriteReqMeta and finish decompression.
            gloo_recv(src=0)
            time.sleep(0.5)
            torch.cuda.synchronize()

            if not torch.all(dst == fill_val):
                bad = (dst != fill_val).nonzero(as_tuple=False)
                idx = bad[0].item()
                print(
                    f"[Receiver] FAIL iter={i} expected={fill_val}: "
                    f"{len(bad)} mismatches, "
                    f"first at [{idx}] got={dst.flatten()[idx].item()}"
                )
                gloo_send(1, dst=0)  # report failure to writer
                dist.destroy_process_group()
                sys.exit(1)

            print(
                f"[Receiver] iter={i} OK — "
                f"{args.size // 1024 // 1024} MB {args.dtype} "
                f"all == {fill_val}"
            )
            gloo_send(0, dst=0)  # report success to writer

        print("[Receiver] all iterations passed ✓")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

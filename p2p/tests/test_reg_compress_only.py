"""
Regression test for Endpoint.reg_compress_only() (split_only compression).

Rank 0 (writer):
  - POSITIVE: registers a large bfloat16 tensor (>= the active compression
    threshold) via reg_compress_only(), writes it with write_async(), and
    waits for the receiver to confirm the decompressed values.
  - NEGATIVE: registers a separate, smaller bfloat16 tensor (below the
    threshold) via reg_compress_only(), then issues a synchronous write()
    with it. This must be rejected outright rather than silently falling
    back to an ordinary (uncompressed) RDMA write, since a compress_only
    handle has no user-buffer MR/lkey to fall back to.

Rank 1 (receiver):
  - Advertises one destination buffer for the positive case and verifies the
    decompressed tensor once the writer signals completion.

This test does not inspect any private C++ state from Python (no binding
exposes P2PMhandle internals for an arbitrary mr_id); it only checks the
public return-value contracts of reg_compress_only()/write_async()/write().
It does NOT prove that the user's buffer never reached ibv_reg_mr() -- that
is verified separately via UCCL_P2P_LOG_LEVEL=INFO UCCL_DEBUG_SUBSYS=RDMA.

Usage (single node, two local ranks):
  UCCL_P2P_COMPRESS_STRATEGY=split_only \\
  torchrun --nnodes=1 --nproc_per_node=2 tests/test_reg_compress_only.py
"""

from __future__ import annotations
import os, sys, time
import torch
import torch.distributed as dist
from uccl import p2p

# Read the same env var the C++ side reads; do not hardcode 16 MiB.
MIN_COMPRESS = int(os.environ.get("UCCL_P2P_MIN_COMPRESS_BYTES", 16 * 1024 * 1024))

_BFLOAT16_ITEM_SIZE = torch.tensor(0, dtype=torch.bfloat16).element_size()


def make_fill_tensor(n_bytes: int, value: float) -> torch.Tensor:
    n_elems = max(1, (n_bytes + _BFLOAT16_ITEM_SIZE - 1) // _BFLOAT16_ITEM_SIZE)
    return torch.full((n_elems,), value, dtype=torch.bfloat16, device="cuda").contiguous()


def gloo_send(value: int, dst: int) -> None:
    dist.send(torch.tensor([value], dtype=torch.int32), dst=dst)


def gloo_recv(src: int) -> int:
    t = torch.zeros(1, dtype=torch.int32)
    dist.recv(t, src=src)
    return t.item()


def main():
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

        # Receiver's destination FIFO metadata for the positive case.
        blob_t = torch.zeros(64, dtype=torch.uint8)
        dist.recv(blob_t, src=1)
        fifo_blob = bytes(blob_t.numpy())

        # --- POSITIVE CASE ---
        big = make_fill_tensor(MIN_COMPRESS, 1.0)
        assert big.nbytes >= MIN_COMPRESS
        torch.cuda.synchronize()

        ok, mr_id = ep.reg_compress_only(
            big.data_ptr(), big.nbytes, p2p.FloatType.kBFloat16
        )
        assert ok, "reg_compress_only() failed for the large (positive-case) tensor"

        ok, tid = ep.write_async(conn_id, mr_id, big.data_ptr(), big.nbytes, fifo_blob)
        assert ok, "write_async() failed to post the compress_only positive-case write"

        done = False
        while not done:
            ok, done = ep.poll_async(tid)
            assert ok, "poll_async() reported failure for the positive-case transfer"

        # poll_async() returns once the data WCs arrive; WriteReqMeta may
        # still be in the NIC send queue. Sleep so the NIC has time to send
        # it, same as test_engine_write_compress.py.
        time.sleep(0.5)

        gloo_send(0, dst=1)
        result = gloo_recv(src=1)
        if result != 0:
            print("[Writer] receiver reported FAIL on positive case")
            dist.destroy_process_group()
            sys.exit(1)
        print("[Writer] positive case confirmed OK")

        # --- NEGATIVE CASE ---
        # A SEPARATE, smaller compress_only handle -- do not reuse `mr_id`
        # with a smaller size, since registered_addr/registered_len
        # validation in endpoint_wrapper.h would reject that for the wrong
        # reason (buffer mismatch) rather than the compression gate.
        small_nbytes = max(_BFLOAT16_ITEM_SIZE, (MIN_COMPRESS // 2 // 8) * 2)
        assert small_nbytes < MIN_COMPRESS, (
            "UCCL_P2P_MIN_COMPRESS_BYTES is too small for the negative case; "
            "pick a larger override"
        )
        small = make_fill_tensor(small_nbytes, 1.0)
        assert small.nbytes < MIN_COMPRESS
        torch.cuda.synchronize()

        ok, small_mr_id = ep.reg_compress_only(
            small.data_ptr(), small.nbytes, p2p.FloatType.kBFloat16
        )
        assert ok, "reg_compress_only() itself must still succeed below the threshold"

        # Reuse the same already-advertised destination FIFO metadata: the
        # compress_only rejection happens client-side, before the remote
        # address/size are ever used on the wire.
        ok = ep.write(conn_id, small_mr_id, small.data_ptr(), small.nbytes, fifo_blob)
        assert not ok, (
            "compress_only write below the compression threshold must be "
            "rejected, not silently sent as an ordinary uncompressed RDMA write"
        )
        print("[Writer] negative case correctly rejected")

        print("[Writer] all cases complete")

    else:
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "accept failed"
        print(f"[Receiver] accepted from {r_ip}")

        dst = make_fill_tensor(MIN_COMPRESS, 0.0)
        ok, mr_id = ep.reg(dst.data_ptr(), dst.nbytes, floatType=p2p.FloatType.kBFloat16)
        assert ok

        ok, fifo_blob = ep.advertise(mr_id, dst.data_ptr(), dst.nbytes)
        assert ok and len(fifo_blob) == 64
        dist.send(torch.ByteTensor(list(fifo_blob)), dst=0)

        # Wait for writer's "data ready" signal, then allow time for the
        # background thread to process WriteReqMeta and finish decompression.
        gloo_recv(src=0)
        time.sleep(0.5)
        torch.cuda.synchronize()

        if not torch.all(dst == 1.0):
            bad = (dst != 1.0).nonzero(as_tuple=False)
            idx = bad[0].item()
            print(
                f"[Receiver] FAIL: {len(bad)} mismatches, "
                f"first at [{idx}] got={dst.flatten()[idx].item()}"
            )
            gloo_send(1, dst=0)
            dist.destroy_process_group()
            sys.exit(1)

        print(f"[Receiver] positive case OK -- {dst.nbytes // 1024 // 1024} MB all == 1.0")
        gloo_send(0, dst=0)

        # Nothing else to check: the negative case is rejected client-side on
        # the writer and never reaches the wire.
        print("[Receiver] all cases passed ✓")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted, terminating…")
        sys.exit(1)

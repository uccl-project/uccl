#!/usr/bin/env python3
"""
Inter-node RDMA + MLU-memory P2P test (torch-free, CNRT).

Single-node self-test (default: spawn server+client on two local MLUs, RDMA over
NIC loopback, run write+read):
  UCCL_P2P_RDMA_DEV=<mlx5 device> python3 tests/xnode_cnrt.py

Real inter-node (run on two machines, --op selects write/read):
  server:  python3 tests/xnode_cnrt.py --role server --bind 192.168.2.248
  client:  python3 tests/xnode_cnrt.py --role client --peer 192.168.2.248 [--op read]

Buffers use plain cnrtMalloc (non peer-able, like torch_mlu tensors); uccl relays
them for Cambricon through a peer-able staging buffer (Plan A), so user buffers
need no special allocation. SIZE=128MB > 32MB single chunk to cover the multi-chunk
path; sparse position markers give full-buffer verification catching chunk mis-order.

The script carries a plain TCP rendezvous (default port 18515) to exchange uccl
metadata / memory descriptors / signals.
"""

import argparse
import ctypes as C
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import p2p

NEUWARE = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
H2D, D2D, D2H = 0, 1, 2
ELEMS = 32 * 1024 * 1024  # 128MB / 4 (> 32MB single chunk -> multi-chunk)
SIZE = ELEMS * 4
SWEEP = [
    (1 << 12, 500),
    (1 << 20, 200),
    (1 << 25, 30),
    (1 << 27, 8),
]  # 4KB/1MB/32MB(single chunk)/128MB(4 chunks)
MARKERS = list(range(0, ELEMS, 1 << 20))  # one position marker per 1M elems


def cnrt_load():
    return C.CDLL(os.path.join(NEUWARE, "lib64", "libcnrt.so"))


def ck(e, n):
    if e != 0:
        raise RuntimeError(f"{n} err={e}")


def cnrt_buf(cnrt, gpu, mode):
    ck(cnrt.cnrtSetDevice(gpu), "setdev")
    # plain cnrtMalloc (non peer-able): uccl relays via peer-able staging.
    p = C.c_void_p(0)
    ck(cnrt.cnrtMalloc(C.byref(p), C.c_size_t(SIZE)), "malloc")
    host = (C.c_float * ELEMS)()  # all zero (calloc, fast)
    if mode == "src":
        for off in MARKERS:
            host[off] = float(
                (off >> 20) + 1
            )  # region index (small int, float32-exact)
    ck(cnrt.cnrtMemcpy(p, host, C.c_size_t(SIZE), C.c_int(H2D)), "h2d")
    return p


def cnrt_read_at(cnrt, gpu, p, elem_off):
    ck(cnrt.cnrtSetDevice(gpu), "setdev")
    ck(cnrt.cnrtSyncDevice(), "sync")
    v = C.c_float(0)
    src = C.c_void_p(p.value + elem_off * 4)
    ck(cnrt.cnrtMemcpy(C.byref(v), src, C.c_size_t(4), C.c_int(D2H)), "d2h")
    return v.value


def verify_markers(cnrt, gpu, p):
    bad = 0
    for off in MARKERS:
        if abs(cnrt_read_at(cnrt, gpu, p, off) - float((off >> 20) + 1)) > 1e-3:
            bad += 1
    return bad


# ── length-prefixed TCP rendezvous ───────────────────────────────────────────
def send_msg(sock, b):
    sock.sendall(struct.pack("!I", len(b)) + b)


def recv_msg(sock):
    n = struct.unpack("!I", _recvn(sock, 4))[0]
    return _recvn(sock, n)


def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise RuntimeError("rendezvous closed")
        buf += c
    return buf


def _rendezvous_server(args):
    ep = p2p.Endpoint(local_gpu_idx=args.gpu)
    meta = ep.get_metadata()
    ip, port, bdf = p2p.Endpoint.parse_metadata(meta)
    print(f"[server] endpoint up. oob meta: ip={ip} port={port} bdf={bdf}")
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((args.bind, args.rv_port))
    lsock.listen(1)
    print(
        f"[server] rendezvous listening on {args.bind}:{args.rv_port}, waiting client..."
    )
    conn, peer = lsock.accept()
    print(f"[server] client connected: {peer}")
    send_msg(conn, meta)
    send_msg(conn, struct.pack("!I", port))
    ok, rip, rgpu, conn_id = ep.accept()
    assert ok, "uccl accept failed"
    print(f"[server] uccl connected: remote_ip={rip} conn_id={conn_id}")
    return ep, conn, lsock, port


def _rendezvous_client(args):
    ep = p2p.Endpoint(local_gpu_idx=args.gpu)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[client] connecting rendezvous {args.peer}:{args.rv_port} ...")
    for _ in range(30):
        try:
            sock.connect((args.peer, args.rv_port))
            break
        except ConnectionRefusedError:
            time.sleep(0.5)
    meta = recv_msg(sock)
    port = struct.unpack("!I", recv_msg(sock))[0]
    print(f"[client] got server metadata, control port={port}")
    ok, conn_id = ep.connect(args.peer, "", port)
    assert ok, "uccl connect failed"
    print(f"[client] uccl connected: conn_id={conn_id}")
    return ep, sock, conn_id


# ── write: server is RDMA-WRITE target, client issues write ──────────────────
def server_write(args, ep, conn, lsock):
    dst = cnrt_buf(cnrt_load(), args.gpu, "dst")  # zeroed, client writes markers
    ok, mr_id = ep.reg(dst.value, SIZE)
    assert ok, "reg(MLU dst) failed"
    ok, adv = ep.advertise(mr_id, dst.value, SIZE)
    assert ok, "advertise failed"
    send_msg(conn, adv)
    assert recv_msg(conn) == b"written"
    bad = verify_markers(cnrt_load(), args.gpu, dst)
    correct = bad == 0
    print(f"[server] verified {len(MARKERS)} markers, {bad} wrong")
    print(
        f"[server] {'PASS' if correct else 'FAIL'}: inter-node multi-chunk RDMA-WRITE"
    )
    send_msg(conn, b"verified" if correct else b"bad")

    print("\n[server] bandwidth run (client drives timing)...")
    for size, _it in SWEEP:
        ok, adv = ep.advertise(mr_id, dst.value, size)
        send_msg(conn, adv)
        assert recv_msg(conn) == b"done"
    print("\n[server] bandwidth (reported by client):\n" + recv_msg(conn).decode())
    return correct


def client_write(args, ep, sock, conn_id):
    src = cnrt_buf(cnrt_load(), args.gpu, "src")
    ok, mr_id = ep.reg(src.value, SIZE)
    assert ok, "reg(MLU src) failed"
    adv = recv_msg(sock)
    ok = ep.write(conn_id, mr_id, src.value, SIZE, adv)
    assert ok, "RDMA write failed"
    send_msg(sock, b"written")
    print(f"[client] server verify: {recv_msg(sock).decode()}")

    lines = [f"{'size':>8} {'iters':>6} {'lat(us)':>10} {'BW(GB/s)':>10}", "-" * 38]
    for size, iters in SWEEP:
        adv = recv_msg(sock)
        for _ in range(3):
            ep.write(conn_id, mr_id, src.value, size, adv)
        t0 = time.perf_counter()
        for _ in range(iters):
            ok = ep.write(conn_id, mr_id, src.value, size, adv)
            assert ok
        t1 = time.perf_counter()
        per = (t1 - t0) / iters
        lines.append(f"{_h(size):>8} {iters:>6} {per*1e6:>10.2f} {size/per/1e9:>10.2f}")
        send_msg(sock, b"done")
    report = "\n".join(lines)
    print("\n[client] bandwidth:\n" + report)
    send_msg(sock, report.encode())
    return True  # write correctness is judged on the server side


# ── read: server is data source (RDMA-READ target), client issues read ───────
def server_read(args, ep, conn, lsock):
    src = cnrt_buf(cnrt_load(), args.gpu, "src")  # source data (markers)
    ok, mr_id = ep.reg(src.value, SIZE)
    assert ok, "reg(MLU src) failed"
    ok, adv = ep.advertise(mr_id, src.value, SIZE)
    assert ok, "advertise failed"
    send_msg(conn, adv)
    assert recv_msg(conn) == b"read_done"
    print("[server] client finished reading.")
    return True


def client_read(args, ep, sock, conn_id):
    dst = cnrt_buf(cnrt_load(), args.gpu, "dst")  # local target (zeroed)
    ok, mr_id = ep.reg(dst.value, SIZE)
    assert ok, "reg(MLU dst) failed"
    adv = recv_msg(sock)
    t0 = time.perf_counter()
    ok = ep.read(conn_id, mr_id, dst.value, SIZE, adv)
    t1 = time.perf_counter()
    assert ok, "RDMA read failed"
    bad = verify_markers(cnrt_load(), args.gpu, dst)
    correct = bad == 0
    bw = SIZE / (t1 - t0)
    print(
        f"[client] read {SIZE >> 20}MB in {(t1 - t0) * 1e3:.2f}ms, {bw / 1e9:.2f} GB/s"
    )
    print(f"[client] verified {len(MARKERS)} markers, {bad} wrong")
    print(f"[client] {'PASS' if correct else 'FAIL'}: inter-node multi-chunk RDMA-READ")
    send_msg(sock, b"read_done")
    return correct


def run_server(args):
    ep, conn, lsock, _ = _rendezvous_server(args)
    ok = (
        server_read(args, ep, conn, lsock)
        if args.op == "read"
        else server_write(args, ep, conn, lsock)
    )
    conn.close()
    lsock.close()
    print("\n[server] done. " + ("all passed" if ok else "failures present"))
    sys.exit(0 if ok else 1)


def run_client(args):
    ep, sock, conn_id = _rendezvous_client(args)
    fn = client_read if args.op == "read" else client_write
    ok = fn(args, ep, sock, conn_id)
    sock.close()
    sys.exit(0 if ok else 1)


def _h(n):
    if n >= 1 << 20:
        return f"{n >> 20}MB"
    if n >= 1 << 10:
        return f"{n >> 10}KB"
    return f"{n}B"


def _proc_entry(args):
    (run_server if args.role == "server" else run_client)(args)


def self_test(args):
    # single-node self-test: spawn server+client on two local MLUs, RDMA over NIC
    # loopback, run write+read.
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    all_ok = True
    for i, op in enumerate(("write", "read")):
        port = args.rv_port + i
        s = argparse.Namespace(
            role="server",
            op=op,
            bind="127.0.0.1",
            peer="",
            rv_port=port,
            gpu=args.server_gpu,
            server_gpu=args.server_gpu,
            client_gpu=args.client_gpu,
        )
        c = argparse.Namespace(
            role="client",
            op=op,
            bind="127.0.0.1",
            peer="127.0.0.1",
            rv_port=port,
            gpu=args.client_gpu,
            server_gpu=args.server_gpu,
            client_gpu=args.client_gpu,
        )
        print(
            f"\n==== self-test op={op} (server gpu{s.gpu} <-> client gpu{c.gpu}) ===="
        )
        srv = mp.Process(target=_proc_entry, args=(s,))
        cli = mp.Process(target=_proc_entry, args=(c,))
        srv.start()
        time.sleep(1)
        cli.start()
        srv.join()
        cli.join()
        ok = srv.exitcode == 0 and cli.exitcode == 0
        all_ok = all_ok and ok
        print(f"==== op={op}: {'PASS' if ok else 'FAIL'} ====")
    print(f"\nself-test summary: {'all passed' if all_ok else 'failures present'}")
    sys.exit(0 if all_ok else 1)


def main():
    ap = argparse.ArgumentParser()
    # no --role: single-node self-test (write+read on two local MLUs);
    # with --role: manual / real inter-node.
    ap.add_argument("--role", choices=["server", "client"])
    ap.add_argument("--op", default="write", choices=["write", "read"])
    ap.add_argument("--bind", default="0.0.0.0", help="server: rendezvous bind IP")
    ap.add_argument("--peer", default="", help="client: server RDMA IP")
    ap.add_argument("--rv-port", type=int, default=18515)
    ap.add_argument("--gpu", type=int, default=0, help="local MLU device index")
    ap.add_argument("--server-gpu", type=int, default=1, help="self-test: server MLU")
    ap.add_argument("--client-gpu", type=int, default=0, help="self-test: client MLU")
    args = ap.parse_args()
    if args.role is None:
        self_test(args)
    elif args.role == "server":
        run_server(args)
    else:
        if not args.peer:
            ap.error("--peer required (server RDMA IP)")
        run_client(args)


if __name__ == "__main__":
    main()

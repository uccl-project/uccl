#!/usr/bin/env python3
"""
跨节点 (inter-node) RDMA + MLU 显存 P2P 测试 (torch-free, CNRT).

两台寒武纪服务器上跑同一个脚本：
  server 端:  python3 tests/xnode_cnrt.py --role server --bind 192.168.2.248
  client 端:  python3 tests/xnode_cnrt.py --role client --peer 192.168.2.248

client 把本地 MLU 显存(填 1.0)经 mlx5 RoCE RDMA-WRITE 到 server 的 MLU 显存,
server cnrtMemcpy 读回校验。验证 uccl 的跨机 RDMA 路径 + cambricon_peer_mem
能否真正把 MLU 显存跨机直传。

脚本自带一个普通 TCP rendezvous (默认端口 18515, 走同一张 RDMA 网) 用于交换
uccl endpoint 元数据 / 显存描述符 / 完成信号, 免去手工 copy。
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
ELEMS = 1024 * 1024            # 4MB / 4
SIZE = ELEMS * 4
SWEEP = [(1 << 12, 1000), (1 << 16, 1000), (1 << 20, 500),
         (1 << 22, 300), (1 << 24, 100), (1 << 26, 50)]


def cnrt_load():
    return C.CDLL(os.path.join(NEUWARE, "lib64", "libcnrt.so"))


def ck(e, n):
    if e != 0:
        raise RuntimeError(f"{n} err={e}")


def cnrt_buf(cnrt, gpu, fill):
    ck(cnrt.cnrtSetDevice(gpu), "setdev")
    p = C.c_void_p(0)
    ck(cnrt.cnrtMalloc(C.byref(p), C.c_size_t(SIZE)), "malloc")
    host = (C.c_float * ELEMS)(*([fill] * ELEMS))
    ck(cnrt.cnrtMemcpy(p, host, C.c_size_t(SIZE), C.c_int(H2D)), "h2d")
    return p


def cnrt_read(cnrt, gpu, p, n=16):
    ck(cnrt.cnrtSetDevice(gpu), "setdev")
    ck(cnrt.cnrtSyncDevice(), "sync")
    host = (C.c_float * n)()
    ck(cnrt.cnrtMemcpy(host, p, C.c_size_t(n * 4), C.c_int(D2H)), "d2h")
    return list(host)


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


# ── server: 被 RDMA-WRITE 的一方 ──────────────────────────────────────────────
def run_server(args):
    cnrt = cnrt_load()
    ep = p2p.Endpoint(local_gpu_idx=args.gpu)
    meta = ep.get_metadata()
    ip, port, bdf = p2p.Endpoint.parse_metadata(meta)
    print(f"[server] endpoint up. oob meta: ip={ip} port={port} bdf={bdf}")

    # rendezvous: 等 client 连上, 把 uccl 元数据发过去
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((args.bind, args.rv_port))
    lsock.listen(1)
    print(f"[server] rendezvous listening on {args.bind}:{args.rv_port}, 等 client...")
    conn, peer = lsock.accept()
    print(f"[server] client 接入: {peer}")
    send_msg(conn, meta)
    send_msg(conn, struct.pack("!I", port))  # 明确告诉 client 控制端口

    # uccl accept (阻塞直到 client connect)
    ok, rip, rgpu, conn_id = ep.accept()
    assert ok, "uccl accept failed"
    print(f"[server] uccl 连接建立: remote_ip={rip} conn_id={conn_id}")

    # ── 正确性: dst 清零, client 写入 1.0, 校验 ──
    dst = cnrt_buf(cnrt, args.gpu, 0.0)
    ok, mr_id = ep.reg(dst.value, SIZE)
    assert ok, "reg(MLU dst) failed"
    ok, adv = ep.advertise(mr_id, dst.value, SIZE)
    assert ok, "advertise failed"
    send_msg(conn, adv)                       # 把显存描述符给 client
    assert recv_msg(conn) == b"written"
    vals = cnrt_read(cnrt, args.gpu, dst)
    correct = all(abs(v - 1.0) < 1e-5 for v in vals)
    print(f"[server] 校验前16个元素: {vals[:8]} ...")
    print(f"[server] {'✅ PASS' if correct else '❌ FAIL'}: 跨机 RDMA-WRITE 数据正确性")
    send_msg(conn, b"verified" if correct else b"bad")

    # ── 带宽: 多次 advertise 同一 buf, client 反复写计时 ──
    print("\n[server] 带宽测试中(client 主导计时)...")
    for size, _it in SWEEP:
        ok, adv = ep.advertise(mr_id, dst.value, size)
        send_msg(conn, adv)
        assert recv_msg(conn) == b"done"
    results = recv_msg(conn).decode()
    print("\n[server] 带宽结果(client 上报):")
    print(results)
    conn.close()
    lsock.close()
    print("\n[server] 完成。" + ("全部通过 ✅" if correct else "存在失败 ❌"))
    sys.exit(0 if correct else 1)


# ── client: 发起 RDMA-WRITE 的一方 ────────────────────────────────────────────
def run_client(args):
    cnrt = cnrt_load()
    ep = p2p.Endpoint(local_gpu_idx=args.gpu)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[client] 连 rendezvous {args.peer}:{args.rv_port} ...")
    for _ in range(30):
        try:
            sock.connect((args.peer, args.rv_port))
            break
        except ConnectionRefusedError:
            time.sleep(0.5)
    meta = recv_msg(sock)
    port = struct.unpack("!I", recv_msg(sock))[0]
    print(f"[client] 收到 server 元数据, 控制端口={port}")

    # uccl connect: 用 server 的 RDMA IP 做控制通道
    ok, conn_id = ep.connect(args.peer, "", port)
    assert ok, "uccl connect failed"
    print(f"[client] uccl 连接建立: conn_id={conn_id}")

    src = cnrt_buf(cnrt, args.gpu, 1.0)
    ok, mr_id = ep.reg(src.value, SIZE)
    assert ok, "reg(MLU src) failed"

    # ── 正确性 ──
    adv = recv_msg(sock)
    ok = ep.write(conn_id, mr_id, src.value, SIZE, adv)
    assert ok, "RDMA write failed"
    send_msg(sock, b"written")
    verdict = recv_msg(sock).decode()
    print(f"[client] server 校验: {verdict}")

    # ── 带宽 ──
    lines = [f"{'size':>8} {'iters':>6} {'lat(us)':>10} {'BW(GB/s)':>10}",
             "-" * 38]
    for size, iters in SWEEP:
        adv = recv_msg(sock)
        # warmup
        for _ in range(5):
            ep.write(conn_id, mr_id, src.value, size, adv)
        t0 = time.perf_counter()
        for _ in range(iters):
            ok = ep.write(conn_id, mr_id, src.value, size, adv)
            assert ok
        t1 = time.perf_counter()
        per = (t1 - t0) / iters
        bw = size / per
        lines.append(f"{_h(size):>8} {iters:>6} {per*1e6:>10.2f} {bw/1e9:>10.2f}")
        send_msg(sock, b"done")
    report = "\n".join(lines)
    print("\n[client] 带宽:\n" + report)
    send_msg(sock, report.encode())
    sock.close()


def _h(n):
    if n >= 1 << 20:
        return f"{n >> 20}MB"
    if n >= 1 << 10:
        return f"{n >> 10}KB"
    return f"{n}B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["server", "client"])
    ap.add_argument("--bind", default="0.0.0.0", help="server: rendezvous 绑定 IP")
    ap.add_argument("--peer", default="", help="client: server 的 RDMA IP")
    ap.add_argument("--rv-port", type=int, default=18515)
    ap.add_argument("--gpu", type=int, default=0, help="本机 MLU device index")
    args = ap.parse_args()
    if args.role == "server":
        run_server(args)
    else:
        if not args.peer:
            ap.error("--peer 必填 (server 的 RDMA IP)")
        run_client(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
命门探针 (torch-free): 复现 uccl rdma_context.cc 对寒武纪 MLU 的注册路径——
用普通 ibv_reg_mr 把一块 cnrtMalloc 的 MLU 显存注册进 mlx5，验证内核
cambricon_peer_mem (PeerDirect) 是否真能让网卡 DMA 寒武纪显存。

成功 => 跨机 MLU P2P 的命门通 (单机即可判定 ~80%)。
失败 => peermem 路也不通，跨机走不了。

    python3 tests/probe_peermem_regmr.py            # 默认 mlx5_0 + MLU dev0
    python3 tests/probe_peermem_regmr.py mlx5_1 1   # 指定网卡/设备
"""
import ctypes as C
import os
import sys

NEUWARE = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
SIZE = 4 << 20  # 4 MB
# uccl 用的 access flags
IBV_ACCESS_LOCAL_WRITE = 1
IBV_ACCESS_REMOTE_WRITE = 2
IBV_ACCESS_REMOTE_READ = 4
ACCESS = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ


def main():
    hca = sys.argv[1] if len(sys.argv) > 1 else "mlx5_0"
    mlu_dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # ---- 1) cnrtMalloc 一块 MLU 显存 ----
    cnrt = C.CDLL(os.path.join(NEUWARE, "lib64", "libcnrt.so"))
    assert cnrt.cnrtSetDevice(mlu_dev) == 0, "cnrtSetDevice failed"
    dptr = C.c_void_p(0)
    assert cnrt.cnrtMalloc(C.byref(dptr), C.c_size_t(SIZE)) == 0, "cnrtMalloc failed"
    print(f"[1] cnrtMalloc OK: MLU dev{mlu_dev} 显存 @ 0x{dptr.value:x}, {SIZE>>20}MB")

    # ---- 2) 打开 mlx5 verbs 设备 + alloc_pd ----
    ibv = C.CDLL("libibverbs.so.1")
    ibv.ibv_get_device_list.restype = C.POINTER(C.c_void_p)
    ibv.ibv_get_device_name.restype = C.c_char_p
    ibv.ibv_get_device_name.argtypes = [C.c_void_p]
    ibv.ibv_open_device.restype = C.c_void_p
    ibv.ibv_open_device.argtypes = [C.c_void_p]
    ibv.ibv_alloc_pd.restype = C.c_void_p
    ibv.ibv_alloc_pd.argtypes = [C.c_void_p]
    ibv.ibv_reg_mr.restype = C.c_void_p
    ibv.ibv_reg_mr.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    ibv.ibv_dereg_mr.argtypes = [C.c_void_p]

    n = C.c_int(0)
    lst = ibv.ibv_get_device_list(C.byref(n))
    assert lst and n.value > 0, "no RDMA devices"
    dev = None
    for i in range(n.value):
        name = ibv.ibv_get_device_name(lst[i]).decode()
        if name == hca:
            dev = lst[i]
    assert dev, f"{hca} not found"
    ctx = ibv.ibv_open_device(dev)
    assert ctx, "ibv_open_device failed"
    pd = ibv.ibv_alloc_pd(ctx)
    assert pd, "ibv_alloc_pd failed"
    print(f"[2] 打开 {hca} + alloc_pd OK")

    # ---- 3) 命门: ibv_reg_mr 注册 MLU 显存 ----
    C.set_errno(0)
    mr = ibv.ibv_reg_mr(pd, dptr, C.c_size_t(SIZE), ACCESS)
    err = C.get_errno()
    if not mr:
        print(f"[3] ❌ ibv_reg_mr(MLU显存) 失败  errno={err} ({os.strerror(err)})")
        print("\n命门: 不通。peermem 路径注册 MLU 显存失败 → 跨机 MLU P2P 走不了。")
        sys.exit(1)

    # 读 struct ibv_mr 的 lkey/rkey (offset: lkey=36, rkey=40 on x86_64)
    buf = (C.c_uint32 * 2).from_address(mr + 36)
    lkey, rkey = buf[0], buf[1]
    print(f"[3] ✅ ibv_reg_mr(MLU显存) 成功! lkey=0x{lkey:x} rkey=0x{rkey:x}")
    print("\n命门: 通! mlx5 经 cambricon_peer_mem 成功注册 MLU 显存。")
    print("=> 网卡可直接 DMA 寒武纪显存，跨机 MLU P2P 的核心前提成立。")

    ibv.ibv_dereg_mr(mr)
    cnrt.cnrtFree(dptr)


if __name__ == "__main__":
    main()

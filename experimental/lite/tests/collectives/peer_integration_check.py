"""Exercise AG and RS through a prepared application bridge, without Torch."""
import argparse
import ctypes as c
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library')
    parser.add_argument('--ranks', type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    cuda = c.CDLL('/usr/local/cuda/lib64/libcudart.so')
    bridge = c.CDLL(args.library)
    ptr = c.c_void_p
    array = ptr * args.ranks
    cuda.cudaMalloc.argtypes = [c.POINTER(ptr), c.c_size_t]
    cuda.cudaMemcpy.argtypes = [ptr, ptr, c.c_size_t, c.c_int]
    cuda.cudaFree.argtypes = [ptr]
    cuda.cudaStreamCreateWithFlags.argtypes = [c.POINTER(ptr), c.c_uint]
    cuda.cudaStreamSynchronize.argtypes = [ptr]
    cuda.cudaStreamDestroy.argtypes = [ptr]
    bridge.uccl_lite_e2e_create.argtypes = [c.c_int]
    bridge.uccl_lite_e2e_create.restype = ptr
    bridge.uccl_lite_e2e_destroy.argtypes = [ptr]
    bridge.uccl_lite_e2e_error.restype = c.c_char_p
    bridge.uccl_lite_e2e_prepare.argtypes = [ptr, c.c_size_t, array, array, array]
    bridge.uccl_lite_e2e_enqueue.argtypes = [ptr, c.c_int, c.c_size_t, array, array, array]

    def check(status):
        if status:
            raise RuntimeError(f'CUDA error {status}')

    def bf16(value):
        return struct.unpack('<I', struct.pack('<f', value))[0] >> 16

    inputs, outputs, streams = array(), array(), array()
    context = None
    try:
        for rank in range(args.ranks):
            check(cuda.cudaSetDevice(rank))
            for target in (inputs, outputs):
                allocation = ptr()
                check(cuda.cudaMalloc(c.byref(allocation), 4096 * args.ranks * 2))
                target[rank] = allocation
            stream = ptr()
            check(cuda.cudaStreamCreateWithFlags(c.byref(stream), 1))
            streams[rank] = stream
        context = bridge.uccl_lite_e2e_create(1)
        if not context:
            raise RuntimeError(bridge.uccl_lite_e2e_error().decode())
        if not bridge.uccl_lite_e2e_enqueue(context, 1, 99, inputs, outputs, streams):
            raise RuntimeError('unprepared integration call accepted')
        for count in (1, 257, 4096):
            if bridge.uccl_lite_e2e_prepare(context, count, inputs, outputs, streams):
                raise RuntimeError(bridge.uccl_lite_e2e_error().decode())
            for op in (1, 2, 1, 2):
                for rank in range(args.ranks):
                    check(cuda.cudaSetDevice(rank))
                    host = (c.c_uint16 * (count * args.ranks))(*([bf16(rank + 1)] * (count * args.ranks)))
                    check(cuda.cudaMemcpy(inputs[rank], host, c.sizeof(host), 1))
                if bridge.uccl_lite_e2e_enqueue(context, op, count, inputs, outputs, streams):
                    raise RuntimeError(bridge.uccl_lite_e2e_error().decode())
                for rank in range(args.ranks):
                    check(cuda.cudaSetDevice(rank))
                    check(cuda.cudaStreamSynchronize(streams[rank]))
                    size = count * (args.ranks if op == 1 else 1)
                    host = (c.c_uint16 * size)()
                    check(cuda.cudaMemcpy(host, outputs[rank], c.sizeof(host), 2))
                    expected = ([bf16(i // count + 1) for i in range(size)] if op == 1
                                else [bf16(args.ranks * (args.ranks + 1) / 2)] * size)
                    if list(host) != expected:
                        raise RuntimeError(f'bridge result mismatch: op={op}, count={count}, rank={rank}')
        print(f'PASS: {args.ranks}-GPU application bridge prepares both AG/RS, rejects unprepared calls, and handles scalar tails')
    finally:
        if context:
            bridge.uccl_lite_e2e_destroy(context)
        for rank in range(args.ranks):
            cuda.cudaSetDevice(rank)
            if streams[rank]:
                cuda.cudaStreamDestroy(streams[rank])
            for target in (inputs, outputs):
                if target[rank]:
                    cuda.cudaFree(target[rank])
        if hasattr(bridge, 'uccl_lite_e2e_finalize') and context:
            bridge.uccl_lite_e2e_finalize()


if __name__ == '__main__':
    main()

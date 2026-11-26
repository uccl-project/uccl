"""Probe BF16 MoE combine with per-row system-scope readiness publication."""

import argparse
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def combine_ready(
    input_ptr,
    output_ptr,
    ready_ptr,
    generation,
    K: tl.constexpr,
    E: tl.constexpr,
    SERIAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    if SERIAL:
        value = tl.full((BLOCK_K,), 0, tl.float32)
        for expert in range(E):
            part = tl.load(
                input_ptr + row * E * K + expert * K + columns,
                mask=columns < K,
                other=0,
            )
            value += part.to(tl.float32)
    else:
        experts = tl.arange(0, E)
        parts = tl.load(
            input_ptr
            + row * E * K
            + experts[:, None] * K
            + columns[None, :],
            mask=columns[None, :] < K,
            other=0,
        )
        value = tl.sum(parts.to(tl.float32), axis=0)
    tl.store(output_ptr + row * K + columns, value.to(tl.bfloat16), columns < K)
    tl.debug_barrier()
    tl.atomic_xchg(ready_ptr + row, generation, sem="release", scope="sys")


@triton.jit
def consume_ready(output_ptr, ready_ptr, copy_ptr, generation, K: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, K)
    ready = tl.atomic_cas(ready_ptr + row, 0, 0, sem="acquire", scope="sys")
    attempts = 0
    while ready != generation and attempts < 1000000:
        ready = tl.atomic_cas(ready_ptr + row, 0, 0, sem="acquire", scope="sys")
        attempts += 1
    value = tl.load(output_ptr + row * K + columns)
    tl.store(copy_ptr + row * K + columns, value)


def time_call(call, repeats=5, iterations=100):
    samples = []
    for _ in range(repeats):
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / iterations)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 296])
    args = parser.parse_args()
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        moe_sum_reduce_torch_compile,
    )

    torch.manual_seed(20261112)
    for rows in args.rows:
        source = torch.randn((rows, 8, 4096), device="cuda", dtype=torch.bfloat16)
        source[::2, ::3] = 0
        expected = torch.empty((rows, 4096), device="cuda", dtype=torch.bfloat16)
        moe_sum_reduce_torch_compile(source, expected, 1.0)
        torch.cuda.synchronize()
        expected_samples = time_call(
            lambda: moe_sum_reduce_torch_compile(source, expected, 1.0)
        )
        for serial in [False, True]:
            output = torch.empty_like(expected)
            flags = torch.zeros(rows, device="cuda", dtype=torch.int32)

            def launch():
                combine_ready[(rows,)](
                    source,
                    output,
                    flags,
                    1,
                    K=4096,
                    E=8,
                    SERIAL=serial,
                    BLOCK_K=4096,
                    num_warps=8,
                )

            launch()
            torch.cuda.synchronize()
            different = (output != expected).sum().item()
            largest = (output.float() - expected.float()).abs().max().item()
            ready = bool((flags == 1).all().item())
            samples = time_call(launch)
            cross_stream_equal = None
            if rows == 8:
                flags.zero_()
                copy = torch.empty_like(output)
                transport_stream = torch.cuda.Stream()
                transport_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(transport_stream):
                    consume_ready[(rows,)](
                        output, flags, copy, 1, K=4096, num_warps=4
                    )
                launch()
                torch.cuda.synchronize()
                cross_stream_equal = bool(torch.equal(copy, output))
            print(
                f"rows={rows} serial={serial} different={different}/{output.numel()} "
                f"max_abs={largest:.6g} ready={ready} "
                f"cross_stream_equal={cross_stream_equal} "
                f"reference_us={statistics.median(expected_samples):.3f} "
                f"combine_ready_us={statistics.median(samples):.3f} "
                f"reference_samples={[round(x, 3) for x in expected_samples]} "
                f"samples={[round(x, 3) for x in samples]}",
                flush=True,
            )


if __name__ == "__main__":
    main()

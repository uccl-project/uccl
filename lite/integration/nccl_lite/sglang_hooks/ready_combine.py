"""BF16 MoE combine with per-row system-scope readiness publication.

This opt-in producer preserves the measured SGLang combine result. Its flags
are prepared for an independent RS consumer; the current integration still
uses SGLang's ordinary stream ordering until that consumer is implemented.
"""

import contextvars
import ctypes
import hashlib
import inspect
import os

import torch
import triton
import triton.language as tl


_ACTIVE = contextvars.ContextVar("uccl_lite_ready_combine", default=None)
_READY_ATTR = "_uccl_lite_combine_ready"
_READY_RS_ATTR = "_uccl_lite_ready_rs"
_EXPECTED_COMBINE = "96c5ca9c1c421c89eec0e40771e514d2b14345cc04d79873f9b48146e835de22"
_EXPECTED_DOWN_KERNEL = "72cee6e057ff8337ccc0a56d76b434999fb71bc453ee82df111f0e71cac7ba50"


@triton.jit
def combine_ready(input_ptr, output_ptr, ready_ptr, generation, K: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, K)
    experts = tl.arange(0, 8)
    parts = tl.load(
        input_ptr + row * 8 * K + experts[:, None] * K + columns[None, :]
    )
    value = tl.sum(parts.to(tl.float32), axis=0).to(tl.bfloat16)
    tl.store(output_ptr + row * K + columns, value)
    tl.debug_barrier()
    tl.atomic_xchg(ready_ptr + row, generation, sem="release", scope="sys")


@triton.jit
def combine_wait_down(
    input_ptr,
    output_ptr,
    counter_ptr,
    ready_ptr,
    expected: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    count = tl.atomic_cas(counter_ptr + row, 0, 0, sem="acquire", scope="gpu")
    while count < expected:
        count = tl.atomic_cas(counter_ptr + row, 0, 0, sem="acquire", scope="gpu")
    columns = tl.arange(0, K)
    experts = tl.arange(0, 8)
    parts = tl.load(
        input_ptr + row * 8 * K + experts[:, None] * K + columns[None, :]
    )
    value = tl.sum(parts.to(tl.float32), axis=0).to(tl.bfloat16)
    tl.store(output_ptr + row * K + columns, value)
    tl.debug_barrier()
    tl.atomic_xchg(ready_ptr + row, 1, sem="release", scope="sys")


def install(model_module):
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe_triton_kernels
    from sglang.srt.utils.common import get_current_device_stream_fast

    original = fused_moe.moe_sum_reduce_torch_compile
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != _EXPECTED_COMBINE:
        print("UCCL Lite ready combine: unsupported SGLang source; hook inactive", flush=True)
        return

    original_forward = model_module.Glm4MoeSparseMoeBlock.forward_normal
    original_shared = model_module.Glm4MoeSparseMoeBlock._forward_shared_experts
    down_counter_enabled = os.environ.get("UCCL_LITE_SGLANG_DOWN_COUNTER") == "1"
    verify_down_counter = os.environ.get("UCCL_LITE_SGLANG_DOWN_COUNTER_VERIFY") == "1"
    down_group_value = os.environ.get("UCCL_LITE_SGLANG_DOWN_GROUP_M")
    down_group = int(down_group_value) if down_group_value else None
    if down_group not in (None, 1, 2, 4, 8):
        raise ValueError("down GROUP_SIZE_M must be 1, 2, 4, or 8")
    down_verified = 0
    if down_counter_enabled:
        from down_gemm_ready_kernel import fused_moe_kernel_ready

        ordinary_kernel = fused_moe_triton_kernels.fused_moe_kernel
        kernel_source = inspect.getsource(ordinary_kernel.fn)
        if hashlib.sha256(kernel_source.encode()).hexdigest() != _EXPECTED_DOWN_KERNEL:
            raise RuntimeError("ready down GEMM requires the pinned SGLang kernel")

        class DownKernelDispatch:
            def __getitem__(self, grid):
                ordinary = ordinary_kernel[grid]
                ready = fused_moe_kernel_ready[grid]

                def launch(*args, **kwargs):
                    if "READY_COUNTER" in kwargs:
                        return ready(*args, **kwargs)
                    return ordinary(*args, **kwargs)

                return launch

        fused_moe_triton_kernels.fused_moe_kernel = DownKernelDispatch()
        ordinary_invoke = fused_moe.invoke_fused_moe_kernel

        def invoke(*args, **kwargs):
            state = _ACTIVE.get()
            if (
                state is None
                or len(args) < 15
                or args[3].shape != (8, 8, 4096)
                or args[13] != 1
                or not kwargs.get("use_fp8_w8a8", False)
                or kwargs.get("block_shape") is not None
            ):
                return ordinary_invoke(*args, **kwargs)
            config = dict(args[14])
            if down_group is not None:
                config["GROUP_SIZE_M"] = down_group
            counter = torch.zeros(8, device=args[3].device, dtype=torch.int32)
            n_tiles = triton.cdiv(4096, config["BLOCK_SIZE_N"])
            n_expert_blocks = triton.cdiv(args[9].shape[0], config["BLOCK_SIZE_M"])
            expert_counter = torch.zeros(
                n_expert_blocks, device=args[3].device, dtype=torch.int32
            )
            expected = 8
            state["down_counter"] = counter
            state["down_expert_counter"] = expert_counter
            state["down_expected"] = expected
            state["down_n_tiles"] = n_tiles
            state["down_output_ptr"] = args[3].data_ptr()
            if down_pipeline_enabled:
                flags = torch.zeros(8, device=args[3].device, dtype=torch.int32)
                initialized = torch.cuda.Event()
                initialized.record(get_current_device_stream_fast())
                state["down_flags"] = flags
                state["down_initialized"] = initialized
            args = list(args)
            config["READY_COUNTER"] = counter
            config["READY_EXPERT_COUNTER"] = expert_counter
            config["READY_GROUP"] = 8
            config["READY_N_TILES"] = n_tiles
            args[14] = config
            return ordinary_invoke(*args, **kwargs)

        fused_moe.invoke_fused_moe_kernel = invoke
        if down_group is not None:
            print(f"UCCL Lite down GEMM GROUP_SIZE_M={down_group}", flush=True)
    ready_rs_enabled = os.environ.get("UCCL_LITE_SGLANG_READY_RS") == "1"
    down_pipeline_enabled = os.environ.get("UCCL_LITE_SGLANG_DOWN_OVERLAP") == "1"
    if down_pipeline_enabled and not (down_counter_enabled and ready_rs_enabled):
        raise RuntimeError("down overlap requires down counters and ready RS")
    if ready_rs_enabled and os.environ.get("UCCL_LITE_SGLANG_RS_SUM2") != "1":
        raise RuntimeError("ready RS requires the RS sum2 hook")
    prepare_call = enqueue_call = None
    if ready_rs_enabled:
        path = os.environ.get("SGLANG_NCCL_SO_PATH")
        if not path or not os.path.isabs(path):
            raise RuntimeError("ready RS requires absolute SGLANG_NCCL_SO_PATH")
        library = ctypes.CDLL(path)
        prepare_call = library.uccl_lite_nccl_prepare_rs_ready
        prepare_call.argtypes = [ctypes.c_void_p]
        prepare_call.restype = ctypes.c_int
        enqueue_call = library.uccl_lite_nccl_reduce_scatter_ready_sum2
        enqueue_call.argtypes = [ctypes.c_void_p] * 6
        enqueue_call.restype = ctypes.c_int
    prepared = False
    transport_stream = None
    combine_stream = None
    pynccl = None
    verify = os.environ.get("UCCL_LITE_SGLANG_READY_COMBINE_VERIFY") == "1"
    verify_limit = int(os.environ.get("UCCL_LITE_SGLANG_VERIFY_CALLS", "1"))
    verified = 0
    different_calls = 0
    pointer_mismatches = 0

    def shared_experts(self, hidden_states):
        output = original_shared(self, hidden_states)
        state = _ACTIVE.get()
        if state is not None and state.get("ready_rs"):
            state["aux"] = output
        return output

    def verify_output(x, out, scale):
        nonlocal verified, different_calls
        if verify and verified < verify_limit:
            reference = torch.empty_like(out)
            original(x, reference, scale)
            different = (out != reference).sum().item()
            verified += 1
            different_calls += different != 0
            if verified == 1 or verified % 100 == 0 or (different and different_calls <= 5):
                print(
                    f"UCCL Lite ready combine verify gpu={torch.cuda.current_device()} "
                    f"call={verified} different={different}/{out.numel()} "
                    f"different_calls={different_calls}",
                    flush=True,
                )

    def combine(x, out, scale):
        nonlocal down_verified
        state = _ACTIVE.get()
        if (
            state is None
            or x.shape != (8, 8, 4096)
            or out.shape != (8, 4096)
            or x.dtype != torch.bfloat16
            or out.dtype != torch.bfloat16
            or not x.is_contiguous()
            or not out.is_contiguous()
            or scale != 1.0
            or torch.cuda.is_current_stream_capturing()
        ):
            return original(x, out, scale)
        if down_counter_enabled:
            if "down_counter" not in state:
                raise RuntimeError("ready down GEMM was not intercepted")
            if verify_down_counter and down_verified < verify_limit:
                values = state["down_counter"].tolist()
                expert_values = state["down_expert_counter"].tolist()
                expected = state["down_expected"]
                down_verified += 1
                if values != [expected] * 8:
                    raise RuntimeError(
                        f"ready down GEMM tile counts {values} != {expected}"
                    )
                if any(v not in (0, state["down_n_tiles"]) for v in expert_values):
                    raise RuntimeError(
                        f"ready down GEMM expert tile counts {expert_values}"
                    )
                if down_verified == 1 or down_verified % 100 == 0:
                    print(
                        f"UCCL Lite down GEMM tile counts gpu={torch.cuda.current_device()} "
                        f"call={down_verified} counts={values} "
                        f"active_expert_blocks={sum(v != 0 for v in expert_values)}",
                        flush=True,
                    )
        if down_pipeline_enabled and state.get("ready_rs") and state.get("aux") is not None:
            if x.data_ptr() != state["down_output_ptr"]:
                raise RuntimeError("down GEMM output moved before ready combine")
            flags = state["down_flags"]
            counter = state["down_counter"]
            initialized = state["down_initialized"]
            current_stream = get_current_device_stream_fast()
            with torch.cuda.stream(combine_stream):
                combine_stream.wait_event(initialized)
                x.record_stream(combine_stream)
                out.record_stream(combine_stream)
                counter.record_stream(combine_stream)
                state["down_expert_counter"].record_stream(combine_stream)
                flags.record_stream(combine_stream)
                combine_wait_down[(8,)](
                    x, out, counter, flags, state["down_expected"],
                    K=4096, num_warps=8
                )
                combine_done = torch.cuda.Event()
                combine_done.record(combine_stream)
            with torch.cuda.stream(transport_stream):
                transport_stream.wait_event(initialized)
                output = torch.empty((1, 4096), device=x.device, dtype=out.dtype)
                out.record_stream(transport_stream)
                state["aux"].record_stream(transport_stream)
                flags.record_stream(transport_stream)
                status = enqueue_call(
                    out.data_ptr(), state["aux"].data_ptr(), output.data_ptr(),
                    flags.data_ptr(), pynccl.comm, transport_stream.cuda_stream,
                )
                if status != 0:
                    raise RuntimeError(f"UCCL Lite down-ready RS enqueue failed: {status}")
                completed = torch.cuda.Event()
                completed.record(transport_stream)
            state["ready_output"] = output
            state["ready_completed"] = completed
            state["combine_completed"] = combine_done
            state["flags"] = flags
            state["output_ptr"] = out.data_ptr()
            if verify and verified < verify_limit:
                current_stream.wait_event(combine_done)
            verify_output(x, out, scale)
            return
        if state.get("ready_rs") and state.get("aux") is not None:
            current_stream = get_current_device_stream_fast()
            with torch.cuda.stream(transport_stream):
                flags = torch.empty(8, device=x.device, dtype=torch.int32)
                flags.zero_()
                output = torch.empty((1, 4096), device=x.device, dtype=out.dtype)
                initialized = torch.cuda.Event()
                initialized.record(transport_stream)
                out.record_stream(transport_stream)
                state["aux"].record_stream(transport_stream)
                status = enqueue_call(
                    out.data_ptr(),
                    state["aux"].data_ptr(),
                    output.data_ptr(),
                    flags.data_ptr(),
                    pynccl.comm,
                    transport_stream.cuda_stream,
                )
                if status != 0:
                    raise RuntimeError(f"UCCL Lite ready RS enqueue failed: {status}")
                completed = torch.cuda.Event()
                completed.record(transport_stream)
            current_stream.wait_event(initialized)
            flags.record_stream(current_stream)
            state["ready_output"] = output
            state["ready_completed"] = completed
        else:
            flags = torch.zeros(8, device=x.device, dtype=torch.int32)
        combine_ready[(8,)](x, out, flags, 1, K=4096, num_warps=8)
        state["flags"] = flags
        state["output_ptr"] = out.data_ptr()
        verify_output(x, out, scale)

    def forward_normal(self, hidden_states, *args, **kwargs):
        nonlocal pointer_mismatches, prepared, transport_stream, combine_stream, pynccl
        if (
            hidden_states.dtype != torch.bfloat16
            or hidden_states.shape != (8, 4096)
            or torch.cuda.is_current_stream_capturing()
        ):
            return original_forward(self, hidden_states, *args, **kwargs)
        use_reduce_scatter = kwargs.get(
            "use_reduce_scatter", args[1] if len(args) > 1 else False
        )
        ready_rs = ready_rs_enabled and use_reduce_scatter
        if ready_rs and not prepared:
            group = get_tp_group()
            pynccl = group.pynccl_comm
            if group.world_size != 8 or pynccl is None:
                raise RuntimeError("ready RS requires TP8 PyNccl")
            torch.cuda.synchronize()
            if prepare_call(pynccl.comm) != 0:
                raise RuntimeError("UCCL Lite ready RS prepare failed")
            transport_stream = torch.cuda.Stream()
            if down_pipeline_enabled:
                combine_stream = torch.cuda.Stream()
            dummy_x = torch.zeros((8, 8, 4096), device="cuda", dtype=torch.bfloat16)
            dummy_out = torch.empty((8, 4096), device="cuda", dtype=torch.bfloat16)
            dummy_flags = torch.zeros(8, device="cuda", dtype=torch.int32)
            combine_ready[(8,)](
                dummy_x, dummy_out, dummy_flags, 1, K=4096, num_warps=8
            )
            if down_pipeline_enabled:
                dummy_counter = torch.full(
                    (8,), 8, device="cuda", dtype=torch.int32
                )
                dummy_flags.zero_()
                with torch.cuda.stream(combine_stream):
                    combine_wait_down[(8,)](
                        dummy_x, dummy_out, dummy_counter, dummy_flags, 8,
                        K=4096, num_warps=8,
                    )
            torch.cuda.synchronize()
            with torch.cuda.stream(transport_stream):
                dummy_a = torch.zeros((8, 4096), device="cuda", dtype=torch.bfloat16)
                dummy_b = torch.zeros_like(dummy_a)
                dummy_result = torch.empty((1, 4096), device="cuda", dtype=torch.bfloat16)
                dummy_ready = torch.ones(8, device="cuda", dtype=torch.int32)
                if (
                    enqueue_call(
                        dummy_a.data_ptr(),
                        dummy_b.data_ptr(),
                        dummy_result.data_ptr(),
                        dummy_ready.data_ptr(),
                        pynccl.comm,
                        transport_stream.cuda_stream,
                    )
                    != 0
                ):
                    raise RuntimeError("UCCL Lite ready RS warmup failed")
                transport_stream.synchronize()
            prepared = True
        state = {"ready_rs": ready_rs}
        token = _ACTIVE.set(state)
        try:
            result = original_forward(self, hidden_states, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)
        if "flags" in state:
            if result.data_ptr() == state["output_ptr"]:
                setattr(result, _READY_ATTR, state["flags"])
                if "ready_output" in state:
                    setattr(
                        result,
                        _READY_RS_ATTR,
                        (state["ready_output"], state["ready_completed"],
                         state["flags"], state["aux"]),
                    )
                    if down_pipeline_enabled:
                        setattr(result, "_uccl_lite_down_state", state)
            else:
                pointer_mismatches += 1
                if "ready_output" in state:
                    raise RuntimeError("ready RS producer output was copied")
                if pointer_mismatches <= 3:
                    print(
                        "UCCL Lite ready combine: final MoE output copied; "
                        "readiness cannot be consumed directly",
                        flush=True,
                    )
        return result

    fused_moe.moe_sum_reduce_torch_compile = combine
    model_module.Glm4MoeSparseMoeBlock._forward_shared_experts = shared_experts
    model_module.Glm4MoeSparseMoeBlock.forward_normal = forward_normal
    print("UCCL Lite SGLang GLM4 MoE ready combine hook installed", flush=True)

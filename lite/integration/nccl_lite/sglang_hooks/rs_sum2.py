"""Opt-in GLM-4.5 EP8 MoE epilogue plus ReduceScatter integration.

This hook is pinned to the installed SGLang GLM4 MoE call boundary. It keeps
SGLang's FP8 expert GEMMs and folds the BF16 routed/shared add into UCCL Lite's
small GPU packet RS upload. Other shapes use SGLang's original implementation.
"""

import ctypes
import hashlib
import inspect
import os

import torch


_AUX_ATTR = "_uccl_lite_rs_sum2_aux"
_MAX_GPU_RS_BYTES = int(os.environ.get("UCCL_LITE_MP_PACKET_RS_GPU_MAX_BYTES", "0"))
_EXPECTED_FORWARD = "fb3c6d0125e99f09ae6b8c1af228a55434e0692bdfea1f175d60e420238f76d3"
_EXPECTED_SCATTER = "2251c137e2b658d96298a29f3dce3de2e7733f5cb3e7132461ad8e73d5bafe4f"


def _source_hash(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def install(model_module):
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers import communicator as comm_module
    from sglang.srt.utils.common import get_current_device_stream_fast

    path = os.environ.get("SGLANG_NCCL_SO_PATH")
    if not path or not os.path.isabs(path):
        raise RuntimeError("RS sum2 requires absolute SGLANG_NCCL_SO_PATH")
    library = ctypes.CDLL(path)
    call = library.uccl_lite_nccl_reduce_scatter_sum2
    call.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    call.restype = ctypes.c_int

    scatter_class = comm_module.CommunicateSummableTensorPairFn
    original_scatter = scatter_class._scatter_hidden_states
    original_forward = model_module.Glm4MoeSparseMoeBlock.forward_normal
    verify_rs = os.environ.get("UCCL_LITE_SGLANG_RS_SUM2_VERIFY") == "1"
    verify_limit = int(os.environ.get("UCCL_LITE_SGLANG_VERIFY_CALLS", "1"))
    verified_rs = 0
    different_rs_calls = 0
    ready_seen = False
    if (
        _source_hash(original_forward) != _EXPECTED_FORWARD
        or _source_hash(original_scatter) != _EXPECTED_SCATTER
    ):
        print("UCCL Lite RS sum2: unsupported SGLang source; hook inactive", flush=True)
        return

    def scatter_hidden_states(
        hidden_states, residual, forward_batch, context, allow_reduce_scatter=False
    ):
        nonlocal verified_rs, different_rs_calls, ready_seen
        auxiliary = getattr(hidden_states, _AUX_ATTR, None)
        if auxiliary is None:
            return original_scatter(
                hidden_states,
                residual,
                forward_batch,
                context,
                allow_reduce_scatter=allow_reduce_scatter,
            )
        if not allow_reduce_scatter or not forward_batch.dp_padding_mode.is_max_len():
            raise RuntimeError("RS sum2 reached an unsupported scatter layout")
        ready_result = getattr(hidden_states, "_uccl_lite_ready_rs", None)
        if ready_result is not None:
            output, completed, flags, ready_aux = ready_result
            if (
                output.shape != (1, 4096)
                or output.dtype != torch.bfloat16
                or ready_aux is not auxiliary
            ):
                raise RuntimeError("ready RS output shape or auxiliary changed")
            stream = get_current_device_stream_fast()
            stream.wait_event(completed)
            output.record_stream(stream)
            if verify_rs and verified_rs < verify_limit:
                summed = torch.add(hidden_states, auxiliary)
                reference = torch.empty_like(output)
                pynccl = get_tp_group().pynccl_comm
                with pynccl.change_state(enable=True, stream=stream):
                    pynccl.reduce_scatter(reference, summed, stream=stream)
                different = (output != reference).sum().item()
                largest = (output.float() - reference.float()).abs().max().item()
                verified_rs += 1
                different_rs_calls += different != 0
                if (
                    verified_rs == 1
                    or verified_rs % 100 == 0
                    or (different and different_rs_calls <= 5)
                ):
                    print(
                        f"UCCL Lite RS ready verify gpu={torch.cuda.current_device()} "
                        f"call={verified_rs} different={different}/{output.numel()} "
                        f"max_abs={largest:.6g} different_calls={different_rs_calls}",
                        flush=True,
                    )
            if not ready_seen:
                print(
                    f"UCCL Lite RS sum2 used {flags.numel()} ready flags "
                    f"on gpu={torch.cuda.current_device()}",
                    flush=True,
                )
                ready_seen = True
            return output, residual
        ready = getattr(hidden_states, "_uccl_lite_combine_ready", None)
        if ready is not None and not ready_seen:
            print(
                f"UCCL Lite RS sum2 received {ready.numel()} combine readiness flags "
                f"on gpu={torch.cuda.current_device()}",
                flush=True,
            )
            ready_seen = True
        output = comm_module.get_local_dp_buffer()
        group = get_tp_group()
        pynccl = group.pynccl_comm
        if (
            pynccl is None
            or group.world_size != 8
            or hidden_states.numel() != output.numel() * 8
            or output.dtype != torch.bfloat16
            or not output.is_contiguous()
        ):
            return original_scatter(
                torch.add(hidden_states, auxiliary),
                residual,
                forward_batch,
                context,
                allow_reduce_scatter=allow_reduce_scatter,
            )
        stream = get_current_device_stream_fast()
        hidden_states.record_stream(stream)
        auxiliary.record_stream(stream)
        with pynccl.change_state(enable=True, stream=stream):
            status = call(
                hidden_states.data_ptr(),
                auxiliary.data_ptr(),
                output.data_ptr(),
                output.numel(),
                pynccl.comm,
                stream.cuda_stream,
            )
            if status == 0 and verify_rs and verified_rs < verify_limit:
                summed = torch.add(hidden_states, auxiliary)
                reference = torch.empty_like(output)
                pynccl.reduce_scatter(reference, summed, stream=stream)
        if status != 0:
            raise RuntimeError(f"UCCL Lite RS sum2 enqueue failed: {status}")
        if verify_rs and verified_rs < verify_limit:
            different = (output != reference).sum().item()
            largest = (output.float() - reference.float()).abs().max().item()
            verified_rs += 1
            different_rs_calls += different != 0
            if (
                verified_rs == 1
                or verified_rs % 100 == 0
                or (different and different_rs_calls <= 5)
            ):
                print(
                    f"UCCL Lite RS sum2 verify gpu={torch.cuda.current_device()} "
                    f"call={verified_rs} different={different}/{output.numel()} "
                    f"max_abs={largest:.6g} different_calls={different_rs_calls}",
                    flush=True,
                )
        return output, residual

    def forward_normal(
        self,
        hidden_states,
        should_allreduce_fusion=False,
        use_reduce_scatter=False,
    ):
        input_bytes = hidden_states.numel() * hidden_states.element_size()
        group = get_tp_group()
        eligible = (
            use_reduce_scatter
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and hidden_states.numel() % 8 == 0
            and 2048 <= input_bytes <= _MAX_GPU_RS_BYTES
            and group.world_size == 8
            and group.pynccl_comm is not None
            and not torch.cuda.is_current_stream_capturing()
            and model_module._is_cuda
            and not model_module._use_aiter
            and not model_module.is_wrapped_method(self.experts.quant_method, "kt_ep")
        )
        if not eligible:
            return original_forward(
                self, hidden_states, should_allreduce_fusion, use_reduce_scatter
            )

        shared_output = self._forward_shared_experts(hidden_states)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)
        if (
            shared_output is None
            or final_hidden_states.dtype != torch.bfloat16
            or not final_hidden_states.is_contiguous()
            or shared_output.dtype != torch.bfloat16
            or not shared_output.is_contiguous()
            or final_hidden_states.shape != shared_output.shape
        ):
            if shared_output is not None:
                final_hidden_states = torch.add(final_hidden_states, shared_output)
            return final_hidden_states

        setattr(final_hidden_states, _AUX_ATTR, shared_output)
        return final_hidden_states

    scatter_class._scatter_hidden_states = staticmethod(scatter_hidden_states)
    model_module.Glm4MoeSparseMoeBlock.forward_normal = forward_normal
    print("UCCL Lite SGLang GLM4 MoE RS sum2 hook installed", flush=True)

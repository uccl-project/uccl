"""Opt-in decode AG plus BF16 GLM4 MoE gate projection.

Only a sparse GLM4 layer's direct forward may defer its one-row/rank AG. The
fused packet kernel fills the gathered buffer and gate logits before the MoE
reads either. All other AG calls retain SGLang's original implementation.
"""

import contextvars
import ctypes
import hashlib
import inspect
import os

import torch


_LOCAL_ATTR = "_uccl_lite_ag_gate_local"
_LOGITS_ATTR = "_uccl_lite_ag_gate_logits"
_ACTIVE = contextvars.ContextVar("uccl_lite_ag_gate_active", default=None)
_EXPECTED_DECODER = "d1efa2ba8a1d42b3ec86ec6a7aa68ad651fbe22b4377f513e8412f4d0f800400"
_EXPECTED_GATHER = "513dac0a096de4d98f0574aad33cd4e191ce55f3dd3eed6821163d260408de8e"
_EXPECTED_BLOCK = "b409636da88b7608021658b3f6cd90e3e0c21c5ed364f145315ddbbba44c2617"
_EXPECTED_GATE = "6c1e813f4836ba9a6eb892ab548af0cb4064b66f043778a7c803b99578979218"
_EXPECTED_DP_GATHER = "96320e530d6b72461a92bbb1d6ff637db16f124b3f81c4ba1ae098406d5ff13a"


def _source_hash(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def install(model_module):
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers import communicator as comm_module
    from sglang.srt.utils.common import get_current_device_stream_fast

    path = os.environ.get("SGLANG_NCCL_SO_PATH")
    if not path or not os.path.isabs(path):
        raise RuntimeError("AG+gate requires absolute SGLANG_NCCL_SO_PATH")
    library = ctypes.CDLL(path)
    call = library.uccl_lite_nccl_all_gather_gemm_128x4096
    call.argtypes = [ctypes.c_void_p] * 6
    call.restype = ctypes.c_int

    decoder = model_module.Glm4MoeDecoderLayer
    block = model_module.Glm4MoeSparseMoeBlock
    gate = model_module.Glm4MoeGate
    original_decoder = decoder.forward
    original_block = block.forward
    original_forward_normal = block.forward_normal
    original_gate = gate.forward
    original_dp_gather = comm_module.dp_gather_partial
    verify_gather = os.environ.get("UCCL_LITE_SGLANG_AG_GATHER_VERIFY") == "1"
    verify_gate = os.environ.get("UCCL_LITE_SGLANG_AG_GATE_VERIFY") == "1"
    verify_limit = int(os.environ.get("UCCL_LITE_SGLANG_VERIFY_CALLS", "1"))
    verified_gather = 0
    different_gather_calls = 0
    verified_gate = 0
    different_gate_calls = 0
    changed_gate_top8_calls = 0
    original_gather_method = (
        comm_module.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
    )
    if (
        _source_hash(original_decoder) != _EXPECTED_DECODER
        or _source_hash(original_gather_method) != _EXPECTED_GATHER
        or _source_hash(original_block) != _EXPECTED_BLOCK
        or _source_hash(original_gate) != _EXPECTED_GATE
        or _source_hash(original_dp_gather) != _EXPECTED_DP_GATHER
    ):
        print("UCCL Lite AG+gate: unsupported SGLang source; hook inactive", flush=True)
        return

    def decoder_forward(self, *args, **kwargs):
        incoming = kwargs.get("hidden_states")
        if incoming is None and len(args) > 1:
            incoming = args[1]
        candidate = (
            incoming is not None
            and incoming.shape[0] == 1
            and self.is_layer_sparse
            and isinstance(self.mlp, block)
            and not self.mlp._enable_a2a_moe
            and not model_module.get_is_capture_mode()
            and not torch.cuda.is_current_stream_capturing()
            and self.mlp.gate.weight.shape == (128, 4096)
            and self.mlp.gate.weight.dtype == torch.bfloat16
            and self.mlp.gate.weight.is_contiguous()
        )
        if not candidate:
            return original_decoder(self, *args, **kwargs)
        state = {"deferred": False}
        token = _ACTIVE.set(state)
        try:
            result = original_decoder(self, *args, **kwargs)
            if state["deferred"]:
                raise RuntimeError("AG+gate deferred AG was not consumed")
            return result
        finally:
            _ACTIVE.reset(token)

    def dp_gather_partial(global_tokens, local_tokens, forward_batch):
        state = _ACTIVE.get()
        if (
            state is None
            or state["deferred"]
            or not forward_batch.dp_padding_mode.is_max_len()
            or global_tokens.dtype != torch.bfloat16
            or local_tokens.dtype != torch.bfloat16
            or global_tokens.numel() != 8 * 4096
            or local_tokens.numel() != 4096
            or not global_tokens.is_contiguous()
            or not local_tokens.is_contiguous()
            or global_tokens.untyped_storage().data_ptr()
            == local_tokens.untyped_storage().data_ptr()
            or get_tp_group().world_size != 8
            or get_tp_group().pynccl_comm is None
        ):
            return original_dp_gather(global_tokens, local_tokens, forward_batch)
        setattr(global_tokens, _LOCAL_ATTR, local_tokens)
        state["deferred"] = True

    def forward_normal(self, hidden_states, *args, **kwargs):
        nonlocal verified_gather, different_gather_calls
        local = getattr(hidden_states, _LOCAL_ATTR, None)
        if local is None:
            return original_forward_normal(self, hidden_states, *args, **kwargs)
        state = _ACTIVE.get()
        if state is None or not state["deferred"]:
            raise RuntimeError("AG+gate missing decoder context")
        gate_logits = torch.empty(
            (8, 128), dtype=torch.bfloat16, device=hidden_states.device
        )
        group = get_tp_group()
        pynccl = group.pynccl_comm
        stream = get_current_device_stream_fast()
        local.record_stream(stream)
        hidden_states.record_stream(stream)
        gate_logits.record_stream(stream)
        with pynccl.change_state(enable=True, stream=stream):
            status = call(
                local.data_ptr(),
                hidden_states.data_ptr(),
                self.gate.weight.data_ptr(),
                gate_logits.data_ptr(),
                pynccl.comm,
                stream.cuda_stream,
            )
            if status == 0 and verify_gather and verified_gather < verify_limit:
                reference_gather = torch.empty_like(hidden_states)
                pynccl.all_gather(reference_gather, local, stream=stream)
        if status != 0:
            raise RuntimeError(f"UCCL Lite AG+gate enqueue failed: {status}")
        if verify_gather and verified_gather < verify_limit:
            different = (hidden_states != reference_gather).sum().item()
            verified_gather += 1
            different_gather_calls += different != 0
            if (
                verified_gather == 1
                or verified_gather % 100 == 0
                or (different and different_gather_calls <= 5)
            ):
                print(
                    f"UCCL Lite AG gather verify gpu={torch.cuda.current_device()} "
                    f"call={verified_gather} different={different}/{hidden_states.numel()} "
                    f"different_calls={different_gather_calls}",
                    flush=True,
                )
        setattr(hidden_states, _LOGITS_ATTR, gate_logits)
        try:
            return original_forward_normal(self, hidden_states, *args, **kwargs)
        finally:
            delattr(hidden_states, _LOGITS_ATTR)
            delattr(hidden_states, _LOCAL_ATTR)
            state["deferred"] = False

    def gate_forward(self, hidden_states):
        nonlocal verified_gate, different_gate_calls, changed_gate_top8_calls
        logits = getattr(hidden_states, _LOGITS_ATTR, None)
        if logits is not None:
            if verify_gate and verified_gate < verify_limit:
                reference = original_gate(self, hidden_states)
                different = (logits != reference).sum().item()
                largest = (logits.float() - reference.float()).abs().max().item()
                ids = torch.topk(logits.float(), k=8).indices.sort(dim=-1).values
                ref_ids = (
                    torch.topk(reference.float(), k=8).indices.sort(dim=-1).values
                )
                changed_rows = (ids != ref_ids).any(dim=-1).sum().item()
                verified_gate += 1
                different_gate_calls += different != 0
                changed_gate_top8_calls += changed_rows != 0
                if (
                    verified_gate == 1
                    or verified_gate % 100 == 0
                    or (changed_rows and changed_gate_top8_calls <= 5)
                ):
                    print(
                        f"UCCL Lite AG+gate verify gpu={torch.cuda.current_device()} "
                        f"call={verified_gate} different={different}/{logits.numel()} "
                        f"max_abs={largest:.6g} changed_top8_rows={changed_rows} "
                        f"different_calls={different_gate_calls} "
                        f"changed_top8_calls={changed_gate_top8_calls}",
                        flush=True,
                    )
            return logits
        return original_gate(self, hidden_states)

    decoder.forward = decoder_forward
    comm_module.dp_gather_partial = dp_gather_partial
    block.forward_normal = forward_normal
    gate.forward = gate_forward
    print("UCCL Lite SGLang GLM4 MoE AG+gate hook installed", flush=True)

"""Full Granite inference through a test-only, single-device GIN receiver."""
import argparse
import asyncio
import ctypes
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
if TYPE_CHECKING:
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast

MODEL = "ibm-granite/granite-3.1-1b-a400m-base"
REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
STAT_NAMES = ("calls", "writes", "signals", "quiet", "payload_bytes")


class GinTransport:
    def __init__(self, library: Path, device: int, queues: int, maximum: int):
        self.library = ctypes.CDLL(str(library.resolve()))
        self.library.model_transport_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
        self.library.model_transport_create.restype = ctypes.c_void_p
        self.library.model_transport_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.library.model_transport_copy.restype = None
        self.library.model_transport_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        self.library.model_transport_stats.restype = None
        self.library.model_transport_destroy.argtypes = [ctypes.c_void_p]
        self.library.model_transport_destroy.restype = None
        self.handle = self.library.model_transport_create(device, queues, maximum)
        assert self.handle

    def copy(self, source: torch.Tensor, groups: int) -> torch.Tensor:
        assert source.is_cuda and source.is_contiguous()
        destination = torch.empty_like(source)
        self.library.model_transport_copy(self.handle, source.data_ptr(),
            destination.data_ptr(), source.numel() * source.element_size(), groups,
            torch.cuda.current_stream(source.device).cuda_stream)
        return destination

    def stats(self) -> dict[str, int]:
        values = (ctypes.c_uint64 * len(STAT_NAMES))()
        self.library.model_transport_stats(self.handle, values)
        return dict(zip(STAT_NAMES, values))

    def close(self) -> None:
        self.library.model_transport_destroy(self.handle)


@dataclass
class BatchOutput:
    tokens: list[list[int]]
    logits_sha256: str


@dataclass
class Request:
    index: int
    started: float
    result: asyncio.Future[list[int]]


class Inference:
    def __init__(self, weights: Path, prompt_tokens: int, new_tokens: int, device: int):
        from transformers import AutoTokenizer, GraniteMoeForCausalLM

        self.device = device
        self.new_tokens = new_tokens
        self.transport: GinTransport | None = None
        self.model = GraniteMoeForCausalLM.from_pretrained(
            weights, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True).to(device).eval()
        assert self.model.config.hidden_size == 1024
        assert self.model.config.num_hidden_layers == 24
        assert self.model.config.num_local_experts == 32
        self.tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
        prompts = ["Explain how a GPU runs many requests concurrently. ",
                   "Write a short example of batching for machine learning. ",
                   "Describe how a mixture of experts routes tokens. ",
                   "Why must a sender wait before reusing a buffer? "]
        self.inputs = [cast(list[int], self.tokenizer.encode(text * prompt_tokens))[:prompt_tokens]
                       for text in prompts]
        assert all(len(tokens) == prompt_tokens for tokens in self.inputs)
        # This test pins the v4.57 Granite tuple contract, without modifying its
        # router, experts, weights, attention, or cache implementation.
        for layer in self.model.model.layers:
            layer.block_sparse_moe.register_forward_pre_hook(self.dispatch)
            layer.block_sparse_moe.register_forward_hook(self.combine)

    def dispatch(self, module: torch.nn.Module,
                 inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        assert len(inputs) == 1
        if self.transport is None:
            return inputs
        hidden = inputs[0].contiguous()
        return (self.transport.copy(hidden, min(hidden.shape[0], 64)),)

    def combine(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...],
                output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transport is None:
            return output
        hidden, router_logits = output
        return self.transport.copy(hidden.contiguous(), min(hidden.shape[0], 64)), router_logits

    @torch.inference_mode()
    def generate(self, indices: list[int]) -> BatchOutput:
        torch.cuda.set_device(self.device)
        tokens = torch.tensor([self.inputs[index % len(self.inputs)] for index in indices],
                              dtype=torch.long, device=self.device)
        generated: list[torch.Tensor] = []
        cache = None
        digest = hashlib.sha256()
        for step in range(self.new_tokens):
            result = cast("MoeCausalLMOutputWithPast", self.model(
                input_ids=tokens, past_key_values=cache, use_cache=True,
                logits_to_keep=1, return_dict=True))
            logits = result.logits
            assert logits is not None and bool(torch.isfinite(logits).all())
            digest.update(logits.detach().to(torch.float32).cpu().numpy().tobytes())
            tokens = logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(tokens)
            cache = result.past_key_values
            assert cache is not None
        return BatchOutput(torch.cat(generated, dim=1).cpu().tolist(), digest.hexdigest())


async def request_wave(inference: Inference, batch_size: int, concurrency: int,
                       executor: ThreadPoolExecutor) -> dict[str, object]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Request] = asyncio.Queue()
    latencies: list[float] = []
    batch_hashes: list[str] = []
    actual_batches: list[int] = []

    async def client(index: int) -> list[int]:
        request = Request(index, time.perf_counter(), loop.create_future())
        await queue.put(request)
        tokens = await request.result
        latencies.append((time.perf_counter() - request.started) * 1000)
        return tokens

    started = time.perf_counter()
    clients = [asyncio.create_task(client(index)) for index in range(concurrency)]
    await asyncio.sleep(0)  # Admit the whole finite wave before batching.
    peak_pending = queue.qsize()
    assert peak_pending == concurrency
    while not queue.empty():
        requests = [queue.get_nowait() for _ in range(min(batch_size, queue.qsize()))]
        indices = [request.index for request in requests]
        output = await loop.run_in_executor(executor, inference.generate, indices)
        actual_batches.append(len(requests))
        batch_hashes.append(output.logits_sha256)
        for request, tokens in zip(requests, output.tokens, strict=True):
            request.result.set_result(tokens)
        await asyncio.sleep(0)
    tokens = await asyncio.gather(*clients)
    elapsed = time.perf_counter() - started
    assert len(tokens) == concurrency and all(len(row) == inference.new_tokens for row in tokens)
    ordered = sorted(latencies)
    return {"elapsed_ms": elapsed * 1000, "requests_per_second": concurrency / elapsed,
            "output_tokens_per_second": concurrency * inference.new_tokens / elapsed,
            "latency_median_ms": ordered[len(ordered) // 2],
            "latency_p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
            "actual_batches": actual_batches, "peak_pending_requests": peak_pending,
            "tokens": tokens, "batch_logits_sha256": batch_hashes}


def native_gate(transport: GinTransport, device: int) -> None:
    # Uneven partitions and decreasing/increasing group counts catch lifecycle
    # errors which fixed-size model batches would miss.
    for groups in (8, 1, 64, 4, 32, 2, 64):
        source = torch.arange(groups * 32 + 13, dtype=torch.int32, device=device)
        received = transport.copy(source, groups)
        assert torch.equal(source, received)
    print(json.dumps({"test": "model_transport_native", "cases": 7,
                      "scope": "CUDA/FIFO/local receiver, no network", "pass": True}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--arm", choices=("scalar", "v1", "v2"), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--queues", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.batch_size <= 64 and args.concurrency >= args.batch_size
    assert args.concurrency % args.batch_size == 0 and args.new_tokens > 0
    assert args.rounds > args.warmup >= 0 and args.prompt_tokens > 0
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261005)
    torch.use_deterministic_algorithms(True)
    maximum = max(2 ** 20, args.batch_size * args.prompt_tokens * 1024 * 2)
    transport = GinTransport(args.library, args.device, args.queues, maximum)
    native_gate(transport, args.device)
    if args.native_only:
        transport.close()
        return
    import transformers

    assert transformers.__version__ == "4.57.1", "use the pinned private Transformers runtime"
    assert args.weights is not None
    inference = Inference(args.weights, args.prompt_tokens, args.new_tokens, args.device)
    print(json.dumps({"test": "model_configuration", "model": MODEL, "revision": REVISION,
        "arm": args.arm, "layers": 24, "experts": 32, "dtype": "bfloat16",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(args.device),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest()}), flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        reference = asyncio.run(request_wave(inference, args.batch_size, args.concurrency, executor))
        inference.transport = transport
        for round_index in range(args.rounds):
            before = transport.stats()
            result = asyncio.run(request_wave(inference, args.batch_size, args.concurrency, executor))
            assert result["tokens"] == reference["tokens"], "generation differs from native model"
            assert result["batch_logits_sha256"] == reference["batch_logits_sha256"], "logits changed"
            after = transport.stats()
            commands = {name: after[name] - before[name] for name in STAT_NAMES}
            logical_signals = args.concurrency * args.new_tokens * 24 * 2
            assert commands["calls"] == args.concurrency // args.batch_size * args.new_tokens * 24 * 2
            assert commands["signals"] == logical_signals and commands["writes"] == logical_signals * 32
            assert commands["quiet"] == logical_signals * args.queues * (32 if args.arm == "scalar" else 1)
            assert commands["payload_bytes"] == args.concurrency * (args.prompt_tokens + args.new_tokens - 1) * 1024 * 2 * 24 * 2
            print(json.dumps({"test": "model_e2e", "arm": args.arm, "round": round_index,
                "warmup": round_index < args.warmup, "batch_size": args.batch_size,
                "concurrency": args.concurrency, "prompt_tokens": args.prompt_tokens,
                "new_tokens": args.new_tokens, "layers": 24, "commands": commands,
                "reference": {"tokens": reference["tokens"],
                              "batch_logits_sha256": reference["batch_logits_sha256"]},
                "scope": "single-device full model; test receiver", "pass": True, **result}), flush=True)
    inference.transport = None
    transport.close()


if __name__ == "__main__":
    main()

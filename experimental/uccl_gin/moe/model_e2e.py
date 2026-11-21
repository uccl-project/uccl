"""Full Qwen3 generation with queued clients, forward HT and real expert MMA."""
import argparse
import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Literal, cast

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from expert_mma import ExpertMma
from ht_bridge import HtBridge
from qwen_moe import ACTIVE_WORKER, QwenHtMoe, Worker
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Qwen3MoeForCausalLM
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeMLP,
    Qwen3MoeSparseMoeBlock,
)

Backend = Literal['torch', 'torch-ht', 'cutlass-ht', 'cutlass-ht-sorted']
ROUTING_ORDERS: tuple[tuple[Backend, ...], ...] = (
    ('torch', 'torch-ht', 'cutlass-ht-sorted', 'cutlass-ht'),
    ('torch-ht', 'cutlass-ht', 'torch', 'cutlass-ht-sorted'),
    ('cutlass-ht', 'cutlass-ht-sorted', 'torch-ht', 'torch'),
    ('cutlass-ht-sorted', 'torch', 'cutlass-ht', 'torch-ht'),
)


@dataclass
class BatchOutput:
    tokens: list[list[int]]
    logits: list[torch.Tensor]
    logits_sha256: str
    cuda_span_ms: float


class Inference:
    def __init__(self, weights: Path, mma_library: Path, ht_library: Path,
                 batch: int, prompt: int, new_tokens: int, ctas: int):
        started = time.perf_counter()
        self.model = Qwen3MoeForCausalLM.from_pretrained(
            weights, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
        print(json.dumps({'stage': 'checkpoint_loaded_cpu', 'elapsed_s': time.perf_counter() - started}), flush=True)
        # Materialize mmap-backed weights on the CPU before the H2D copies.
        # GH200 first-touch copies of individual mmap views otherwise stall.
        for parameter in self.model.parameters():
            parameter.data = parameter.pin_memory()
        print(json.dumps({'stage': 'weights_pinned_cpu', 'elapsed_s': time.perf_counter() - started}), flush=True)
        self.model.to('cuda', non_blocking=True).eval()
        torch.cuda.synchronize()
        print(json.dumps({'stage': 'weights_ready_gpu', 'elapsed_s': time.perf_counter() - started,
                          'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                          'cuda_reserved_bytes': torch.cuda.memory_reserved()}), flush=True)
        config = self.model.config
        assert (config.hidden_size, config.num_hidden_layers, config.num_experts, config.num_experts_per_tok) == (2048, 48, 128, 8)
        for layer in self.model.model.layers:
            assert isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
            layer.mlp = QwenHtMoe(layer.mlp)
        tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
        prompts = ('Explain batching and communication in a GPU server. ',
                   'Write a short example about scheduling many requests. ',
                   'Describe how mixture of experts routes tokens. ',
                   'Why must a sender wait before reusing a buffer? ')
        self.inputs = [cast(list[int], tokenizer.encode(text * prompt))[:prompt] for text in prompts]
        assert all(len(row) == prompt for row in self.inputs)
        self.new_tokens = new_tokens
        stream = torch.cuda.Stream()
        free_before, _ = torch.cuda.mem_get_info()
        with torch.cuda.stream(stream):
            self.worker = Worker(stream, HtBridge(ht_library, stream, batch * prompt), ExpertMma(mma_library, ctas))
            value = torch.empty((batch * prompt, 2048), device='cuda', dtype=torch.bfloat16)
            expert = cast(Qwen3MoeMLP, self.model.model.layers[0].mlp.original.experts[0])
            self.resources = {'communication': self.worker.transport.resources(),
                              'expert_gate': self.worker.mma.query(value, expert.gate_proj.weight),
                              'expert_down': self.worker.mma.query(
                                  torch.empty((batch * prompt, 768), device='cuda', dtype=torch.bfloat16), expert.down_proj.weight)}
        stream.synchronize()
        self.context_initialized_bytes = free_before - torch.cuda.mem_get_info()[0]

    @torch.inference_mode()
    def generate(self, indices: list[int], backend: Backend) -> BatchOutput:
        torch.cuda.set_device(0)
        self.worker.routing = 'sorted' if backend == 'cutlass-ht-sorted' else 'where'
        self.worker.backend = 'cutlass-ht' if backend == 'cutlass-ht-sorted' else backend
        context = ACTIVE_WORKER.set(self.worker)
        with torch.cuda.stream(self.worker.stream), sdpa_kernel(SDPBackend.MATH):
            tokens = torch.tensor([self.inputs[i % len(self.inputs)] for i in indices], device='cuda', dtype=torch.long)
            cache = None
            outputs, generated = [], []
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(self.new_tokens):
                result = cast(MoeCausalLMOutputWithPast, self.model(
                    input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True))
                cache = result.past_key_values
                logits = result.logits[:, -1, :].contiguous()
                tokens = logits.argmax(dim=-1, keepdim=True)
                outputs.append(logits.detach()); generated.append(tokens)
            all_tokens = torch.cat(generated, dim=-1)
            stop.record(); stop.synchronize()
            elapsed = start.elapsed_time(stop)
            logits_cpu = [value.cpu() for value in outputs]
            token_rows = cast(list[list[int]], all_tokens.cpu().tolist())
        ACTIVE_WORKER.reset(context)
        digest = hashlib.sha256()
        for value in logits_cpu: digest.update(value.view(torch.uint8).numpy().tobytes())
        return BatchOutput(token_rows, logits_cpu, digest.hexdigest(), elapsed)

    def close(self) -> None:
        self.worker.transport.close(); self.worker.mma.close()
        assert self.worker.transport.active_allocations() == 0


def validate(actual: BatchOutput, expected: BatchOutput, backend: Backend) -> tuple[float, float]:
    assert actual.tokens == expected.tokens, 'greedy generation differs from native model'
    maxima, relative = [], []
    for output, reference in zip(actual.logits, expected.logits, strict=True):
        exact = backend in ('torch', 'torch-ht')
        torch.testing.assert_close(output, reference, rtol=0 if exact else .03, atol=0 if exact else .125)
        difference = output.float() - reference.float()
        maxima.append(difference.abs().max().item())
        relative.append((torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference.float())).item())
    assert max(relative) < .01
    return max(maxima), max(relative)


async def wave(inference: Inference, batch: int, concurrency: int, backend: Backend,
               executor: ThreadPoolExecutor, reference: list[BatchOutput] | None,
               routing_reference: list[BatchOutput] | None = None) -> tuple[dict, list[BatchOutput]]:
    loop = asyncio.get_running_loop()
    pending: asyncio.Queue[tuple[int, float, asyncio.Future[list[int]]]] = asyncio.Queue()
    latencies = []

    async def client(index: int) -> list[int]:
        future: asyncio.Future[list[int]] = loop.create_future()
        started = time.perf_counter()
        await pending.put((index, started, future))
        result = await future
        latencies.append((time.perf_counter() - started) * 1000)
        return result

    started = time.perf_counter()
    clients = [asyncio.create_task(client(i)) for i in range(concurrency)]
    await asyncio.sleep(0)
    assert pending.qsize() == concurrency
    batches, outputs, errors = [], [], []
    before = inference.worker.transport.stats()
    while not pending.empty():
        requests = [pending.get_nowait() for _ in range(min(batch, pending.qsize()))]
        result = await loop.run_in_executor(executor, inference.generate, [r[0] for r in requests], backend)
        if reference is not None: errors.append(validate(result, reference[len(outputs)], backend))
        if routing_reference is not None:
            expected = routing_reference[len(outputs)]
            assert result.tokens == expected.tokens and result.logits_sha256 == expected.logits_sha256
        outputs.append(result); batches.append(len(requests))
        for request, tokens in zip(requests, result.tokens, strict=True): request[2].set_result(tokens)
    generated = await asyncio.gather(*clients)
    elapsed = time.perf_counter() - started
    after = inference.worker.transport.stats()
    calls = 0 if backend == 'torch' else concurrency // batch * inference.new_tokens * 48
    assert after['dispatch_calls'] - before['dispatch_calls'] == calls
    assert after['combine_calls'] - before['combine_calls'] == calls
    assert after['live_handle'] == after['dispatched'] == 0
    assert after['callback_active_allocations'] == before['callback_active_allocations']
    ordered = sorted(latencies)
    report = {'backend': backend, 'batch_size': batch, 'concurrency': concurrency,
              'actual_batches': batches, 'peak_pending_requests': concurrency, 'active_macro_batches': 1,
              'host_wave_with_full_logit_validation_ms': elapsed * 1000,
              'cuda_span_sum_ms': sum(r.cuda_span_ms for r in outputs),
              'requests_per_second': concurrency / elapsed,
              'output_tokens_per_second': concurrency * inference.new_tokens / elapsed,
              'latency_median_ms': ordered[len(ordered) // 2],
              'latency_p95_ms': ordered[min(len(ordered)-1, int(len(ordered)*.95))],
              'tokens': generated, 'batch_logits_sha256': [r.logits_sha256 for r in outputs],
              'max_abs_logit_error': max((e[0] for e in errors), default=0),
              'max_relative_rms_logit_error': max((e[1] for e in errors), default=0),
              'routing_bitwise_reference_verified': routing_reference is not None,
              'dispatch_calls': calls, 'combine_calls': calls, 'status': 'PASS'}
    report['callback_active_allocations'] = after['callback_active_allocations']
    return report, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--mma-library', type=Path, required=True)
    parser.add_argument('--ht-library', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--concurrency', type=int, default=128)
    parser.add_argument('--prompt-tokens', type=int, default=128)
    parser.add_argument('--new-tokens', type=int, default=2)
    parser.add_argument('--ctas', type=int, default=132)
    parser.add_argument('--session', type=int, default=0)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--routing-ablation', action='store_true')
    args = parser.parse_args()
    assert 0 < args.batch_size <= 64 and args.concurrency >= args.batch_size
    assert args.concurrency % args.batch_size == 0 and 0 < args.batch_size * args.prompt_tokens <= 8192
    assert args.new_tokens > 0 and args.warmups >= 0 and args.repeats > 0
    torch.cuda.set_device(0); torch.manual_seed(20261013); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    inference = Inference(args.weights, args.mma_library, args.ht_library,
                          args.batch_size, args.prompt_tokens, args.new_tokens, args.ctas)
    receipt = json.loads((args.weights/'download-complete.json').read_text())
    print(json.dumps({'test': 'full_qwen_model_configuration', 'model': receipt['model'], 'revision': receipt['revision'],
                      'layers': 48, 'experts': 128, 'topk': 8, 'hidden': 2048, 'physical_gpus': 1, 'nodes': 1,
                      'prompt_tokens': args.prompt_tokens, 'new_tokens': args.new_tokens, 'session': args.session,
                      'device': torch.cuda.get_device_name(0), 'device_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                      'runtime_versions': inference.worker.transport.versions(), 'torch': str(torch.__version__),
                      'attention_backend': 'math', 'deterministic_algorithms': True,
                      'expert_cta_target': args.ctas, 'scope': 'full generation; one-node/one-GPU forward HT loopback; queued clients'}), flush=True)
    print(json.dumps({'test': 'full_qwen_resource_budget', 'resources': inference.resources,
                      'active_macro_batches': 1, 'context_initialized_bytes': inference.context_initialized_bytes,
                      'scope': 'maximum single-expert rows and initialized context; not peak group memory or a co-residency guarantee'}), flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        baseline, reference = asyncio.run(wave(inference, args.batch_size, args.concurrency, 'torch', executor, None))
        print(json.dumps({'test': 'full_qwen_model_reference', **baseline}), flush=True)
        replay, _ = asyncio.run(wave(inference, args.batch_size, args.concurrency, 'torch', executor, reference))
        print(json.dumps({'test': 'full_qwen_native_replay', **replay}), flush=True)
        routing_reference = None
        if args.routing_ablation:
            report, routing_reference = asyncio.run(wave(inference, args.batch_size, args.concurrency,
                                                        'cutlass-ht', executor, reference))
            print(json.dumps({'test': 'full_qwen_routing_reference', **report}), flush=True)
        order = (ROUTING_ORDERS[args.session % 4] if args.routing_ablation else
                 cast(tuple[Backend, ...], tuple(permutations(('torch', 'torch-ht', 'cutlass-ht')))[args.session % 6]))
        for round_index in range(args.warmups + args.repeats):
            for backend in order:
                report, _ = asyncio.run(wave(inference, args.batch_size, args.concurrency, backend, executor, reference,
                                            routing_reference if backend.startswith('cutlass-ht') else None))
                print(json.dumps({'test': 'full_qwen_model_e2e', 'round': round_index,
                                  'warmup': round_index < args.warmups, **report}), flush=True)
    inference.close()
    print(json.dumps({'test': 'full_qwen_model_e2e_summary', 'status': 'PASS',
                      'sessions_in_process': 1, 'final_callback_allocations': 0,
                      'weights_cleanup': 'after all required model runs and archived verification'}), flush=True)


if __name__ == '__main__':
    main()

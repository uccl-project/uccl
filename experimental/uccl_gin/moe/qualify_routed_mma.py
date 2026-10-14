"""Native Qwen MoE oracle for routing, BF16 aggregation and real GEMMs."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from expert_mma import ExpertMma
from ht_bridge import HtBridge
from qwen_moe import ACTIVE_WORKER, QwenHtMoe, Worker
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mma-library', type=Path, required=True)
    parser.add_argument('--ht-library', type=Path, required=True)
    parser.add_argument('--ctas', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261013)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_default_dtype(torch.bfloat16)
    config = Qwen3MoeConfig.from_pretrained(args.config, local_files_only=True)
    original = Qwen3MoeSparseMoeBlock(config).to('cuda').eval()
    torch.set_default_dtype(torch.float32)
    stream = torch.cuda.current_stream()
    worker = Worker(stream, HtBridge(args.ht_library, stream, 8192), ExpertMma(args.mma_library, args.ctas))
    model = QwenHtMoe(original)
    modes: tuple[tuple[Literal['torch-ht', 'cutlass-ht'], Literal['where', 'sorted']], ...] = (
        ('torch-ht', 'where'), ('torch-ht', 'sorted'),
        ('cutlass-ht', 'where'), ('cutlass-ht', 'sorted'))
    cases = 0
    print(json.dumps({'test': 'routed_mma_configuration', 'device': torch.cuda.get_device_name(0),
                      'scope': 'seeded random Qwen3 MoE weights, one layer, native oracle; no trained model',
                      'ctas': args.ctas, 'communication': worker.transport.resources(),
                      'libraries': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (args.mma_library, args.ht_library)}}), flush=True)
    for batch, length in ((1, 1), (64, 1), (64, 128), (64, 1), (64, 128)):
        hidden = torch.randn(batch, length, 2048, device='cuda', dtype=torch.bfloat16)
        expected, expected_router = original(hidden)
        context = ACTIVE_WORKER.set(worker)
        old_outputs = {}
        for backend, routing in modes:
            worker.backend, worker.routing = backend, routing
            actual, router = model(hidden)
            torch.testing.assert_close(router, expected_router, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0 if backend == 'torch-ht' else .03,
                                       atol=0 if backend == 'torch-ht' else .01)
            if routing == 'where':
                old_outputs[backend] = actual
            else:
                torch.testing.assert_close(actual, old_outputs[backend], rtol=0, atol=0)
            difference = actual.float() - expected.float()
            relative = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected.float())
            assert relative.item() < .01
            stats = worker.transport.stats()
            assert stats['live_handle'] == stats['dispatched'] == 0
            assert stats['dispatch_calls'] == stats['combine_calls']
            cases += 1
            print(json.dumps({'test': 'routed_mma_oracle', 'case': cases, 'backend': backend,
                              'routing': routing, 'batch': batch, 'sequence': length,
                              'max_abs_error': difference.abs().max().item(), 'relative_rms_error': relative.item(),
                              'sorted_bitwise_matches_where': routing == 'sorted',
                              'stats': stats, 'status': 'PASS'}), flush=True)
        ACTIVE_WORKER.reset(context)
    worker.transport.close(); worker.mma.close()
    assert worker.transport.active_allocations() == 0
    print(json.dumps({'test': 'routed_mma_summary', 'cases': cases,
                      'final_callback_allocations': 0, 'status': 'PASS'}), flush=True)


if __name__ == '__main__':
    main()

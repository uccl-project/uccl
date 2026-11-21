"""Actual Qwen3 layer weights: native, forward-HT and CUTLASS-HT output oracles."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import torch
from expert_mma import ExpertMma
from ht_bridge import HtBridge
from qwen_moe import ACTIVE_WORKER, QwenHtMoe, Worker
from safetensors import safe_open
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock


def load_layer(weights: Path) -> Qwen3MoeSparseMoeBlock:
    config = Qwen3MoeConfig.from_pretrained(weights, local_files_only=True)
    layer = Qwen3MoeSparseMoeBlock(config)
    prefix = 'model.layers.0.mlp.'
    index = json.loads((weights/'model.safetensors.index.json').read_text())['weight_map']
    names = {name: shard for name, shard in index.items() if name.startswith(prefix)}
    tensors = {}
    for shard in sorted(set(names.values())):
        with safe_open(weights/shard, framework='pt', device='cpu') as source:
            for name, target in names.items():
                if target == shard:
                    tensors[name[len(prefix):]] = source.get_tensor(name)
    layer.load_state_dict(tensors, strict=True, assign=True)
    return layer.to(device='cuda', dtype=torch.bfloat16).eval()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--mma-library', type=Path, required=True)
    parser.add_argument('--ht-library', type=Path, required=True)
    parser.add_argument('--ctas', type=int, default=18)
    args = parser.parse_args()
    torch.manual_seed(20261013)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_device(0)
    layer = load_layer(args.weights)
    stream = torch.cuda.current_stream()
    worker = Worker(stream, HtBridge(args.ht_library, stream, 8192), ExpertMma(args.mma_library, args.ctas))
    candidate = QwenHtMoe(layer)
    print(json.dumps({'test': 'actual_qwen_layer_configuration', 'layers_executed': 1,
                      'layer': 0, 'hidden': 2048, 'experts': 128, 'topk': 8,
                      'runtime_versions': worker.transport.versions(),
                      'communication_resources': worker.transport.resources(),
                      'weight_revision': json.loads((args.weights/'download-complete.json').read_text())['revision']}), flush=True)
    for batch, length in ((8, 16), (32, 64), (64, 128)):
        hidden = torch.randn(batch, length, 2048, device='cuda', dtype=torch.bfloat16)
        reference, router = cast(tuple[torch.Tensor, torch.Tensor], layer(hidden))
        context = ACTIVE_WORKER.set(worker)
        modes: tuple[Literal['torch-ht', 'cutlass-ht'], ...] = ('torch-ht', 'cutlass-ht')
        for mode in modes:
            worker.backend = mode
            output, actual_router = candidate(hidden)
            torch.testing.assert_close(actual_router, router, rtol=0, atol=0)
            torch.testing.assert_close(output, reference, rtol=0 if mode == 'torch-ht' else .03,
                                       atol=0 if mode == 'torch-ht' else .01)
            difference = (output.float()-reference.float()).abs()
            relative_rms = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference.float())
            assert relative_rms.item() < .01
            stats = worker.transport.stats()
            assert stats['dispatch_calls'] == stats['combine_calls'] and stats['live_handle'] == stats['dispatched'] == 0
            print(json.dumps({'test': 'actual_qwen_moe_layer', 'backend': mode, 'batch': batch,
                              'sequence': length, 'tokens': batch*length, 'max_abs_error': difference.max().item(),
                              'mean_abs_error': difference.mean().item(), 'relative_rms_error': relative_rms.item(),
                              'stats': stats, 'status': 'PASS'}), flush=True)
        ACTIVE_WORKER.reset(context)
    worker.transport.close(); worker.mma.close()
    print(json.dumps({'test': 'actual_qwen_layer_summary', 'cases': 6, 'status': 'PASS',
                      'physical_gpus': 1, 'nodes': 1, 'scope': 'trained MoE layer/forward HT; full model generation still separate',
                      'libraries': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (args.mma_library, args.ht_library)}}), flush=True)


if __name__ == '__main__':
    main()

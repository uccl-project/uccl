"""Qualify real BF16 expert GEMMs and BF16 SwiGLU against independent arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from expert_mma import ExpertMma
from torch.nn import functional as F


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--ctas', type=int, default=18)
    parser.add_argument('--maximum-rows', type=int, default=8192)
    args = parser.parse_args()
    torch.manual_seed(20261013)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    mma = ExpertMma(args.library, args.ctas)
    cases = 0
    for rows in (1, 31, 128, 129, 512, 2048, 8192):
        if rows > args.maximum_rows:
            continue
        for outputs, hidden in ((768, 2048), (2048, 768)):
            value = (torch.randn(rows, hidden, device='cuda') * .125).to(torch.bfloat16)
            weight = (torch.randn(outputs, hidden, device='cuda') * .125).to(torch.bfloat16)
            expected = F.linear(value.float(), weight.float()).to(torch.bfloat16)
            actual = mma.linear(value, weight)
            torch.testing.assert_close(actual, expected, rtol=.01, atol=.002)
            resources = mma.query(value, weight)
            assert 0 < resources['blocks'] <= args.ctas
            error = (actual.float() - expected.float()).abs()
            print(json.dumps({'test': 'expert_mma', 'rows': rows, 'outputs': outputs, 'hidden': hidden,
                              'ctas_requested': args.ctas, 'resources': resources,
                              'max_abs_error': error.max().item(), 'mean_abs_error': error.mean().item(),
                              'status': 'PASS'}), flush=True)
            cases += 1
    for count in (1, 127, 128, 513, 4097):
        gate = (torch.randn(count, device='cuda') * 4).to(torch.bfloat16)
        up = torch.randn(count, device='cuda').to(torch.bfloat16)
        expected = F.silu(gate) * up
        actual = mma.swiglu(gate, up)
        torch.testing.assert_close(actual, expected, rtol=.01, atol=.002)
        print(json.dumps({'test': 'expert_swiglu', 'elements': count,
                          'max_abs_error': (actual.float() - expected.float()).abs().max().item(),
                          'status': 'PASS'}), flush=True)
        cases += 1
    mma.close()
    print(json.dumps({'test': 'expert_mma_summary', 'cases': cases, 'status': 'PASS',
                      'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
                      'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
                      'device': torch.cuda.get_device_name(0), 'scope': 'real expert arithmetic diagnostic; no model E2E/timing claim'}), flush=True)


if __name__ == '__main__':
    main()

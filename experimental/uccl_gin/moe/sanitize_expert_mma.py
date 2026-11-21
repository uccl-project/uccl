"""CPU arithmetic oracle and explicit tensor teardown for GPU sanitizers."""
import argparse
import json
from pathlib import Path

import torch
from expert_mma import ExpertMma
from torch.nn import functional as F


@torch.inference_mode()
def qualify(library: Path) -> None:
    torch.set_num_threads(8); torch.manual_seed(20261013)
    mma = ExpertMma(library, 18)
    for rows in (1, 31, 128, 129, 512):
        for outputs, hidden in ((768, 2048), (2048, 768)):
            value = (torch.randn(rows, hidden)*.125).to(torch.bfloat16)
            weight = (torch.randn(outputs, hidden)*.125).to(torch.bfloat16)
            reference = F.linear(value.float(), weight.float()).to(torch.bfloat16)
            value_gpu, weight_gpu = value.cuda(), weight.cuda()
            output = mma.linear(value_gpu, weight_gpu)
            actual = output.cpu()
            torch.testing.assert_close(actual, reference, rtol=.01, atol=.002)
            resources = mma.query(value_gpu, weight_gpu)
            assert 0 < resources['blocks'] <= 18
            print(json.dumps({'test': 'sanitized_expert_mma', 'rows': rows, 'outputs': outputs,
                              'hidden': hidden, 'resources': resources, 'reference_device': 'cpu',
                              'max_abs_error': (actual.float()-reference.float()).abs().max().item(),
                              'status': 'PASS'}), flush=True)
    for count in (1, 127, 128, 513, 4097):
        gate = (torch.randn(count)*4).to(torch.bfloat16)
        up = torch.randn(count).to(torch.bfloat16)
        actual = mma.swiglu(gate.cuda(), up.cuda()).cpu()
        reference = F.silu(gate)*up
        torch.testing.assert_close(actual, reference, rtol=.01, atol=.002)
        print(json.dumps({'test': 'sanitized_swiglu', 'elements': count,
                          'max_abs_error': (actual.float()-reference.float()).abs().max().item(),
                          'status': 'PASS'}), flush=True)
    mma.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    qualify(args.library)
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0
    print(json.dumps({'test': 'sanitized_expert_summary', 'cases': 15,
                      'torch_active_bytes': 0, 'torch_reserved_bytes': 0, 'status': 'PASS'}), flush=True)

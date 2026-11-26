#!/usr/bin/env python3
"""Compare every trained BF16 parameter between ABBA processes, untimed."""
import argparse
import json
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument('--prefix',required=True)
p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
p.add_argument('--directory',type=Path,default=Path('/dev/shm/uccl-lite-e2e-validation'))
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(1)
results=[]
for base in sorted(a.directory.glob(f'{a.prefix}_00_nccl_training_node*_parameters.pt')):
    ref=torch.load(base,weights_only=True,mmap=True)
    for phase,backend in ((1,'peer'),(2,'peer'),(3,'nccl')):
        matches=list(a.directory.glob(base.name.replace('00_nccl',f'{phase:02d}_*')))
        assert len(matches)==1, f'expected one phase {phase} candidate for {base.name}'
        candidate=matches[0]
        values=torch.load(candidate,weights_only=True,mmap=True)
        assert len(values)==len(ref)
        maximum=0.;relative_l2=0.;squared=0.;ref_squared=0.;elements=0
        for x,y in zip(values,ref):
            x=x.to(a.device).float();y=y.to(a.device).float();d=x-y
            maximum=max(maximum,d.abs().max().item())
            squared+=d.square().sum().item();ref_squared+=y.square().sum().item();elements+=x.numel()
            torch.testing.assert_close(x,y,atol=.002,rtol=.02)
        results.append({'reference':base.name,'candidate':candidate.name,'elements':elements,'max_abs_diff':maximum,'relative_l2':(squared/ref_squared)**.5})
assert results,'no parameter files'
a.out.write_text(json.dumps({'status':'PASS','comparison_device':a.device,'atol':.002,'rtol':.02,'comparisons':results},indent=2)+'\n')

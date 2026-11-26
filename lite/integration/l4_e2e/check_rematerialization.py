#!/usr/bin/env python3
"""Check every SGD-updated block parameter with/without rematerialization."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from benchmark import Collectives, Model, synchronize

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(1)
c=Collectives('nccl')
m=Model(a.checkpoint,c)
initial=[v.detach().clone() for v in m.parameters]
g=torch.Generator().manual_seed(12345)
t=torch.randint(m.config['vocab_size'],(8,33),generator=g)
tokens=[v.to(r)for r,v in enumerate(t.chunk(4))]
reference=None
histories=[]
maximum=0.
elements=0
for enabled in (False,True):
    with torch.no_grad():
        for v,original in zip(m.parameters,initial):v.copy_(original)
    m.checkpoint_blocks=enabled
    opt=torch.optim.SGD(m.parameters,lr=.001,foreach=True)
    opt.zero_grad(set_to_none=True)
    logits,_=m.forward([v[:,:-1]for v in tokens])
    losses=[F.cross_entropy(v.flatten(0,1).float(),t[:,1:].flatten())for v,t in zip(logits,tokens)]
    torch.autograd.backward(losses,[torch.full((),.25,device=r)for r in range(4)])
    opt.step();synchronize()
    histories.append([v.item()for v in losses])
    if not enabled:
        reference=[v.detach().clone()for v in m.parameters]
    else:
        for v,ref in zip(m.parameters,reference):
            torch.testing.assert_close(v,ref,atol=.002,rtol=.02)
            maximum=max(maximum,(v.float()-ref.float()).abs().max().item())
            elements+=v.numel()
assert histories[0]==histories[1]
a.out.write_text(json.dumps({'status':'PASS','losses':histories,'parameter_elements':elements,
    'max_parameter_abs_diff':maximum,'atol':.002,'rtol':.02},indent=2)+'\n')
c.close();c.lib.uccl_lite_e2e_finalize()

#!/usr/bin/env python3
"""Paired large-message chunk sweep; direct P2P controls remain unchunked."""
import os
import pathlib
import subprocess
import sys
root=pathlib.Path(sys.argv[1])
variants=('uccl_fp32_reference,hybrid_grouped_1m,hybrid_grouped_2m,'
          'hybrid_grouped_4m,hybrid_grouped_8m,hybrid_grouped_16m,'
          'p2p_xor_batch_ce_fixed4,p2p_memop_onewait_fixed4')
env=dict(os.environ,UCCL_CE_COORDINATOR_CPU='4')
for sweep,powers in [('forward',range(23,31)),('reverse',range(30,22,-1))]:
    out=root/sweep
    out.mkdir(parents=True,exist_ok=True)
    for power in powers:
        size=1<<power
        print(sweep,size,flush=True)
        with (out/f'{size}.csv').open('w') as log,(out/f'{size}.stderr').open('w') as err:
            subprocess.run(['build/ce_allgather',str(size),'10',variants],env=env,
                           stdout=log,stderr=err,check=True,timeout=240)
    subprocess.run([sys.executable,'scripts/summarize_ce_allgather.py',str(out)],check=True)
out=root/'tails'
out.mkdir(exist_ok=True)
for size in [33554448,67108880]:
    with (out/f'{size}.csv').open('w') as log,(out/f'{size}.stderr').open('w') as err:
        subprocess.run(['build/ce_allgather',str(size),'5',variants],env=env,
                       stdout=log,stderr=err,check=True,timeout=120)

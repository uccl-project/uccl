#!/usr/bin/env python3
"""Fresh paired medium-message CE/memop trials, prepared scopes kept explicit."""
import os
import pathlib
import subprocess
import sys
root=pathlib.Path(sys.argv[1])
variants=('uccl_fp32_reference,hybrid_grouped_1m,p2p_xor_batch_ce,'
          'p2p_xor_batch_ce_fixed4,p2p_memop,p2p_memop_fixed4,'
          'p2p_memop_events,p2p_memop_onewait,p2p_memop_onewait_fixed4,'
          'p2p_memop_armed_ce,p2p_memop_armed_sm')
env=dict(os.environ,UCCL_CE_COORDINATOR_CPU='4')
for sweep,powers in [('forward',range(17,24)),('reverse',range(23,16,-1))]:
    out=root/sweep
    out.mkdir(parents=True,exist_ok=True)
    for power in powers:
        size=1<<power
        print(sweep,size,flush=True)
        with (out/f'{size}.csv').open('w') as log,(out/f'{size}.stderr').open('w') as err:
            subprocess.run(['build/ce_allgather',str(size),'20',variants],env=env,
                           stdout=log,stderr=err,check=True,timeout=120)
    subprocess.run([sys.executable,'scripts/summarize_ce_allgather.py',str(out)],check=True)
out=root/'tails'
out.mkdir(exist_ok=True)
for size in [131088,8388624]:
    with (out/f'{size}.csv').open('w') as log,(out/f'{size}.stderr').open('w') as err:
        subprocess.run(['build/ce_allgather',str(size),'5',variants],env=env,
                       stdout=log,stderr=err,check=True,timeout=120)
out=root/'delayed'
out.mkdir(exist_ok=True)
for size in [131072,8388608]:
    with (out/f'{size}.csv').open('w') as log,(out/f'{size}.stderr').open('w') as err:
        subprocess.run(['build/ce_allgather',str(size),'5',
                        'p2p_memop,p2p_memop_fixed4,p2p_memop_events,p2p_memop_onewait,p2p_memop_onewait_fixed4'],
                       env=dict(env,UCCL_CE_DELAY_INPUT_RANK='3'),
                       stdout=log,stderr=err,check=True,timeout=120)

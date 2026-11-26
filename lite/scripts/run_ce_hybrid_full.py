#!/usr/bin/env python3
"""Fresh paired four-L4 FP32 sweeps; no historical latency substitution."""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
variants = ('uccl_fp32_reference,packet_sm,nccl,batch_ce_4m,'
            'hybrid_grouped_1m,hybrid_grouped_4m')
for sweep, sizes in [('forward', range(10, 31)), ('reverse', range(30, 9, -1))]:
    out = root / sweep
    out.mkdir(exist_ok=True)
    for power in sizes:
        size = 1 << power
        print(sweep, size, flush=True)
        with (out / f'{size}.csv').open('w') as log, (out / f'{size}.stderr').open('w') as err:
            subprocess.run(['build/ce_allgather', str(size), '10', variants],
                           stdout=log, stderr=err, check=True, timeout=180)
    subprocess.run([sys.executable, 'scripts/summarize_ce_allgather.py', str(out)], check=True)
# Additional scalar/tail checks around reference policy boundaries.
out = root / 'tails'
out.mkdir(exist_ok=True)
for size in [1008, 1048592, 16777232]:
    with (out / f'{size}.csv').open('w') as log, (out / f'{size}.stderr').open('w') as err:
        subprocess.run(['build/ce_allgather', str(size), '5', variants],
                       stdout=log, stderr=err, check=True, timeout=60)

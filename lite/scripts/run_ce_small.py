#!/usr/bin/env python3
"""Screen and independently confirm serialized eager single-packet schedules."""
import pathlib
import subprocess
import sys
root = pathlib.Path(sys.argv[1])
screen = ('uccl_fp32_reference,hybrid_grouped_1m,sp_gpu_32,sp_cpu_32,'
          'sp_cpu_16,sp_cpu_8,sp_cpu_4,sp_cpu_1,sp_threads_32,'
          'sp_threads_16,sp_threads_8,sp_threads_4,sp_threads_1')
confirm = ('uccl_fp32_reference,hybrid_grouped_1m,sp_gpu_32,sp_cpu_32,'
           'sp_cpu_8,sp_threads_8')
for sweep, sizes, iterations, variants in [
    ('screen', [1 << p for p in range(10,24)], 5, screen),
    ('confirm', [1 << p for p in range(23,9,-1)], 20, confirm),
    ('tails', [1008,1048592,4194320], 5, confirm),
]:
    out = root / sweep
    out.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        print(sweep, size, flush=True)
        with (out / f'{size}.csv').open('w') as log, (out / f'{size}.stderr').open('w') as err:
            subprocess.run(['build/ce_allgather', str(size), str(iterations), variants],
                           stdout=log, stderr=err, check=True, timeout=90)
    subprocess.run([sys.executable, 'scripts/summarize_ce_allgather.py', str(out)], check=True)

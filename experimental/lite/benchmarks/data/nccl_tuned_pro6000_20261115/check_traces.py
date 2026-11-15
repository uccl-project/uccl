#!/usr/bin/env python3
"""Verify actual NCCL selection for every frozen schedule key."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
schedule = {(r['operation'], r['bytes']): r for r in json.loads((root / 'schedule.json').read_text())}
names = {'AllReduce': 'allreduce', 'AllGather': 'allgather', 'ReduceScatter': 'reducescatter'}
observed = {}
for run in json.loads((root / 'trace_runs.json').read_text()):
    assert run['correctness_pass'] and run['returncode'] == 0
    text = (root / (run['name'] + '_debug.log')).read_text()
    edges = sorted(set(line.split(' via ', 1)[1] for line in text.splitlines()
                       if 'Channel ' in line and ' -> ' in line and ' via ' in line))
    assert not any(edge.startswith(('SHM/', 'NET/')) for edge in edges), edges
    selections = re.findall(r'(AllReduce|AllGather|ReduceScatter)( \[Copy Engine\])?: (\d+) Bytes -> (.*)', text)
    for operation, ce, count, detail in selections:
        op = names[operation]
        size = int(count) * (1 if op == 'allreduce' else 2)
        key = op, size
        assert run['op'] == op and run['begin'] <= size <= run['end']
        assert schedule[key]['configuration'] == run['config']
        if not ce:
            assert edges and all(edge.startswith('P2P/') for edge in edges)
        else:
            assert 'cudaMemcpy' in detail and run['environment']['NCCL_CTA_POLICY'] == '2'
        observed[key] = dict(operation=op, bytes=size, configuration=run['config'],
                              copy_engine=bool(ce), selection=detail, channel_transports=edges,
                              log=run['name'] + '_debug.log')
assert set(observed) == set(schedule), (set(schedule) - set(observed))
(root / 'transport.json').write_text(json.dumps(list(observed.values()), indent=2) + '\n')
print('Verified actual NCCL selection for all 63 scheduled cases')

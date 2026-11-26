#!/usr/bin/env python3
"""Check ascending trace coverage and P2P memcpy transport selection."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
results = []
aliases = {'AllReduce': 'allreduce', 'AllGather': 'allgather', 'ReduceScatter': 'reducescatter'}
for run in json.loads((root / 'trace_runs.json').read_text()):
    assert run['correctness_pass'] and run['size_order'] == 'ascending'
    text = (root / (run['name'] + '_debug.log')).read_text()
    edges = sorted(set(line.split(' via ', 1)[1] for line in text.splitlines()
                       if 'Channel ' in line and ' -> ' in line and ' via ' in line))
    assert edges and all(edge.startswith('P2P/') for edge in edges)
    memcpy = run['config'] != 'default'
    assert all(('/CE' in edge) == memcpy for edge in edges)
    selections = re.findall(r'(AllReduce|AllGather|ReduceScatter): (\d+) Bytes -> (.*)', text)
    observed = {}
    for operation, count, detail in selections:
        op = aliases[operation]
        assert op == run['op']
        size = int(count) * (1 if op == 'allreduce' else 2)
        observed[size] = detail
    assert set(observed) == {1 << i for i in range(10, 31)}
    assert '[Copy Engine]' not in text  # This is transport CE, not zero-CTA CE.
    results.append(dict(operation=run['op'], configuration=run['config'],
                        channel_transports=edges, selections=observed,
                        log=run['name'] + '_debug.log'))
assert len(results) == 6
(root / 'transport.json').write_text(json.dumps(results, indent=2) + '\n')
print('Verified all six full ascending transport traces')

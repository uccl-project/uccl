#!/usr/bin/env python3
"""Validate sweeps and plot paired latency/throughput panels like 8xL4."""
import collections
import csv
import math
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use('Agg')

from figure_style import configure
from plot_4l4_collectives import plot, size_name

root = Path(__file__).resolve().parents[1]
data = root / 'benchmarks/data/small_medium_4l4_20261113'
new = {'mini-host', 'scatter-hier-ll', 'push-grid', 'scatter-grid'}

def save(name, rows):
    with (data/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def read(log, operations, sizes, iterations, fixed=None, baseline="nccl"):
    calls = collections.defaultdict(lambda: collections.defaultdict(dict))
    for line in (data/log).read_text().splitlines():
        if line.startswith('# sample,'):
            _, op, size, backend, run, call, us = line.split(',')
            value = float(us)
            assert math.isfinite(value) and value > 0
            sample = calls[op, int(size), backend][int(run)]
            assert int(call) not in sample
            sample[int(call)] = value
    assert {(o,s) for o,s,_ in calls} == {(o,s) for o in operations for s in sizes}
    candidates, selected = [], []
    for op in operations:
        for size in sizes:
            stats = {}
            for (operation, extent, backend), runs in calls.items():
                if (operation, extent) != (op,size): continue
                assert set(runs) == {1,2,3}
                assert all(sorted(v) == list(range(1, iterations+1)) for v in runs.values())
                means = [mean(v.values()) for v in runs.values()]
                stats[backend] = (median(means), min(means), max(means))
                candidates.append(dict(operation=op, bytes=size, backend=backend,
                                       median_us=median(means), min_us=min(means), max_us=max(means)))
            assert baseline in stats
            best = fixed or min((b for b in stats if b!=baseline), key=lambda b:stats[b][0])
            prior = min((stats[b][0] for b in stats if b!=baseline and b not in new))
            selected.append(dict(operation=op, bytes=size, baseline=baseline, uccl_path=best,
                                 uccl_median_us=stats[best][0], uccl_min_us=stats[best][1],
                                 uccl_max_us=stats[best][2], nccl_median_us=stats[baseline][0],
                                 nccl_min_us=stats[baseline][1], nccl_max_us=stats[baseline][2],
                                 prior_best_us=prior, speedup=stats[baseline][0]/stats[best][0]))
    print(log, 'validated timed calls:', sum(len(v) for runs in calls.values() for v in runs.values()))
    return candidates, selected

agc, ag = read('ag_final.log', ['all_gather'], [16<<i for i in range(27)], 10)
rsc, rs = read('rs_final.log', ['reduce_scatter'], [16<<i for i in range(27)], 10)
rc, resident = read('resident_doorbell_final.log', ['all_gather','reduce_scatter','all_reduce'],
                    [16<<i for i in range(17)], 20, 'resident')
save('ag_candidates.csv', agc); save('ag_selected.csv', ag)
save('rs_candidates.csv', rsc); save('rs_selected.csv', rs)
save('resident_candidates.csv', rc); save('resident_selected.csv', resident)
gc, graph_control = read('resident_graph_final.log', ['all_gather','reduce_scatter','all_reduce'],
                         [16<<i for i in range(17)], 20, 'resident', 'nccl-graph')
save('resident_graph_candidates.csv', gc); save('resident_graph_selected.csv', graph_control)
with (data/'ar_selected.csv').open() as f:
    ar = list(csv.DictReader(f))
for row in ar:
    row['bytes'] = int(row['bytes'])
    for key in ['uccl_median_us','nccl_median_us','uccl_min_us','uccl_max_us','nccl_min_us','nccl_max_us']:
        row[key] = float(row[key])
assert [r['bytes'] for r in ar] == [1024<<i for i in range(21)]



def cell(row):
    a,b=row['uccl_median_us'],row['nccl_median_us']
    return f'{a:,.2f} / {b:,.2f}'+(' †' if a>b else '')
lookup = [{r['bytes']:r for r in rows} for rows in [ar,ag,rs]]
lines=['| Full bytes/GPU | AllReduce | AllGather | ReduceScatter |', '| ---: | ---: | ---: | ---: |']
for n in [1024<<i for i in range(21)]:
    lines.append('| '+size_name(n)+' | '+' | '.join(cell(v[n]) for v in lookup)+' |')
(data/'ordinary_table.md').write_text('\n'.join(lines)+'\n')
rlookup={op:{r['bytes']:r for r in resident if r['operation']==op} for op in ['all_reduce','all_gather','reduce_scatter']}
lines=['| Full bytes/GPU | Resident AR / NCCL | Resident AG / NCCL | Resident RS / NCCL |', '| ---: | ---: | ---: | ---: |']
for n in [16,1024,16384,65536,262144,1048576]:
    lines.append('| '+size_name(n)+' | '+' | '.join(cell(rlookup[op][n]) for op in rlookup)+' |')
(data/'resident_table.md').write_text('\n'.join(lines)+'\n')
print('AG medium improvements:')
for row in ag:
    if row['bytes'] in [2**20,2**21,2**22,2**23,2**24]: print(row)

# The README displays only these two-panel views. Older focused/resident SVGs
# remain archived; resident timings do not cover the throughput range.
configure()
plot(ar, 'AllReduce', '4 processes, host timer', 1.5)
plot(ag, 'AllGather', '1 process, eager GPU events', .75)
plot(rs, 'ReduceScatter', '1 process, eager GPU events', .75)

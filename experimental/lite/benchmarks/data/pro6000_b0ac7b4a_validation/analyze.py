"""Reconstruct full-run medians and compare peer/joint with NCCL default."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'results'
OPS = ['allreduce', 'allgather', 'reducescatter']
SIZES = [2**i for i in range(10, 31)]
NAMES = ['nccl-forward','peer-forward','joint-forward','joint-reverse','peer-reverse','nccl-reverse']

def gm(values):
    return math.exp(mean(math.log(v) for v in values))

def main():
    assert json.loads((RAW/'COMPLETE.json').read_text())['complete']
    cfg = json.loads((ROOT/'config.json').read_text())
    provenance = json.loads((RAW/'provenance.json').read_text())
    assert cfg['revision'] == cfg['verified_remote_head'] == provenance['config']['revision']
    times, routes = {}, {}
    for name in NAMES:
        status = json.loads((RAW/(name+'.status.json')).read_text())
        assert status['returncode'] == 0
        log = (RAW/(name+'.log')).read_text()
        assert 'correctness: PASS' in log
        if name.startswith('nccl'):
            tr = json.loads((RAW/(name+'.transport.json')).read_text())
            assert tr['p2p_only'] and tr['lines'] and all('via P2P' in x for x in tr['lines'])
        with (RAW/(name+'.samples.csv')).open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6300
        groups, selected = {}, {}
        for r in rows:
            key = r['operation'], int(r['bytes'])
            ri = int(r['repeat']), int(r['iteration'])
            assert key[0] in OPS and key[1] in SIZES and 0 <= ri[0] < 5 and 0 <= ri[1] < 20
            group = groups.setdefault(key, {})
            assert ri not in group
            v = float(r['slowest_rank_us'])
            assert math.isfinite(v) and v > 0
            group[ri] = v
            selected.setdefault(key, set()).add(r['selected_backend'])
        assert len(groups) == 63 and all(len(g) == 100 for g in groups.values())
        assert all(len(v) == 1 for v in selected.values())
        times[name] = {k: median(median(g[r,i] for i in range(20)) for r in range(5)) for k,g in groups.items()}
        routes[name] = {k: next(iter(v)) for k,v in selected.items()}
        printed = list(csv.DictReader([s for s in log.splitlines() if s.startswith('operation,') or any(s.startswith(op+',') for op in OPS)]))
        assert len(printed) == 63
        for r in printed:
            assert r['correct'] == '1'
            assert math.isclose(float(r['median_us']), times[name][r['operation'],int(r['bytes'])], rel_tol=2e-8)
    cases = []
    for mode in ['peer','joint']:
        for op in OPS:
            for size in SIZES:
                k = op,size
                nf,nr = [times['nccl-'+order][k] for order in ['forward','reverse']]
                cf,cr = [times[mode+'-'+order][k] for order in ['forward','reverse']]
                cases.append(dict(backend=mode,operation=op,bytes=size,forward_us=cf,reverse_us=cr,
                                  nccl_forward_us=nf,nccl_reverse_us=nr,forward_speedup=nf/cf,reverse_speedup=nr/cr,
                                  central_speedup=math.sqrt(nf/cf*nr/cr),forward_route=routes[mode+'-forward'][k],
                                  reverse_route=routes[mode+'-reverse'][k]))
    with (ROOT/'cases.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(cases[0]),lineterminator='\n');w.writeheader();w.writerows(cases)
    scores = {}
    for mode in ['peer','joint']:
        scores[mode] = {}
        for region,lo,hi in [('full',1024,1<<30),('serving',8<<10,16<<20)]:
            scores[mode][region] = {}
            for op in ['all']+OPS:
                rs = [r for r in cases if r['backend']==mode and lo<=r['bytes']<=hi and (op=='all' or r['operation']==op)]
                scores[mode][region][op] = dict(cases=len(rs),forward=gm(r['forward_speedup'] for r in rs),
                                              reverse=gm(r['reverse_speedup'] for r in rs),
                                              central=gm(r['central_speedup'] for r in rs),
                                              wins_both=sum(r['forward_speedup']>1 and r['reverse_speedup']>1 for r in rs))
    result = dict(valid=True,measured_revision=cfg['revision'],baseline='NCCL default, verified P2P on every logged data edge',
                  scope='2 PRO6000 GPUs; BF16 out-of-place; 63 equal-weight cases, 1 KiB–1 GiB; prepared eager, slowest-rank CUDA events including submission gaps; no E2E/graphs',
                  protocol=dict(warmup=100,repeats=5,iterations=20,primary_invocations=37800,calibration='Separate process/seed; 3 warmups, 3 x5 per backend/case',
                                execution_order=NAMES,source='Exact export of verified remote commit; initial incomplete export excluded before performance'),
                  scores=scores,route_counts={name:{route:sum(v==route for v in rs.values()) for route in set(rs.values())} for name,rs in routes.items()},
                  historical_joint_score=1.5742135529636239,
                  files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.iterdir() if p.is_file()})
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'scores':scores,'routes':result['route_counts']},indent=2))

if __name__ == '__main__':
    main()

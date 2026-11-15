"""Verify packaged measurements and recompute aggregates; Python standard library only."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def gm(values):
    return math.exp(mean(math.log(value) for value in values))


def rows(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    for name, digest in provenance['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    router = rows('router-shm-cases.csv')
    assert len(router) == 252
    result = {'router': {}, 'pro6000_joint': {}, 'l41_peer_serving': {}}
    for node in ['l41', 'l48']:
        result['router'][node] = {}
        for order in ['forward', 'reverse']:
            cases = [r for r in router if r['node'] == node and r['order'] == order]
            assert len(cases) == 63
            result['router'][node][order] = {
                'paired_gain': gm(float(r['incumbent_us']) / float(r['router_us']) for r in cases),
                'nccl_before': gm(float(r['nccl_before_us']) / float(r['router_us']) for r in cases),
                'nccl_after': gm(float(r['nccl_after_us']) / float(r['router_us']) for r in cases),
            }
    pro = rows('pro6000-joint-cases.csv')
    assert len(pro) == 63 and all(r['selected'] == 'peer' for r in pro)
    for ref in ['nccl_default_us', 'nccl_shm_us']:
        result['pro6000_joint'][ref] = gm(float(r[ref]) / float(r['joint_us']) for r in pro)
    peer = rows('l41-peer-cases.csv')
    assert len(peer) == 63
    for op in ['all', 'allreduce', 'allgather', 'reducescatter']:
        cases = [r for r in peer if 8192 <= int(r['bytes']) <= 16777216
                 and (op == 'all' or r['operation'] == op)]
        result['l41_peer_serving'][op] = {
            ref: gm(math.sqrt(float(r[ref + '_forward_us']) * float(r[ref + '_reverse_us'])
                             / float(r['peer_forward_us']) / float(r['peer_reverse_us'])) for r in cases)
            for ref in ['nccl', 'default']
        }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

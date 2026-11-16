"""Small fixed-configuration NCCL search; never invokes a UCCL backend."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent
OUT = ROOT / 'build/nccl_multinode_tuned_20261116'
MPI = '/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun'
BINARY = OUT / 'uccl_lite_nccl_2x4l4_baseline'
REMOTE = '10.10.55.1'
CONFIGS = {
    'sm_default': {},
    'sm_ring_simple': {'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple'},
    'sm_p2p_ring_simple': {'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple',
                           'NCCL_P2P_LEVEL': 'SYS'},
    'sm_p2p_ring_simple_c4': {'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple',
                              'NCCL_P2P_LEVEL': 'SYS', 'NCCL_MIN_NCHANNELS': '4',
                              'NCCL_MAX_NCHANNELS': '4'},
    'ce_ring_simple': {},
    'ce_c4': {'NCCL_MIN_NCHANNELS': '4', 'NCCL_MAX_NCHANNELS': '4'},
    'ce_c8_buffer8m': {'NCCL_MIN_NCHANNELS': '8', 'NCCL_MAX_NCHANNELS': '8',
                      'NCCL_BUFFSIZE': '8388608'},
}
for name, settings in CONFIGS.items():
    settings.update(NCCL_P2P_USE_CUDA_MEMCPY='1' if name.startswith('ce_') else '0',
                    NCCL_SHM_USE_CUDA_MEMCPY='0')
    if name.startswith('ce_'):
        settings.update(NCCL_P2P_LEVEL='SYS', NCCL_ALGO='Ring', NCCL_PROTO='Simple')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sizes(op):
    return [1 << p for p in range(20 if op == 'allgather' else 10, 31)]


def samples(path, op, warm):
    values = {phase: {} for phase in ('warmup', 'timed')}
    for line in path.read_text().splitlines():
        fields = line.split(',')
        if len(fields) != 6 or fields[0] != 'nccl':
            continue
        phase, size, latency = fields[1], int(fields[3]), float(fields[4])
        assert int(fields[5]) == 0 and math.isfinite(latency) and latency > 0, path
        values[phase].setdefault(size, []).append(latency)
    for phase, count in [('warmup', warm), ('timed', 10)]:
        assert sorted(values[phase]) == sizes(op), path
        assert all(len(v) == count for v in values[phase].values()), path
    return {str(size): statistics.mean(v) for size, v in values['timed'].items()}


def idle():
    query = ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader']
    for command in [query, ['ssh', REMOTE, shlex.join(query)]]:
        active = subprocess.check_output(command, text=True).strip()
        if active:
            raise RuntimeError('GPU occupied: ' + active)


def run(stage, op, config, repeat):
    name = f'{stage}_{op}_{config}_{repeat}'
    record = OUT / f'{name}.json'
    if record.exists():
        value = json.loads(record.read_text())
        assert value['returncode'] == 0, record
        return value
    stdout = OUT / f'{name}.stdout'
    assert not stdout.exists(), f'Preserve partial run before resuming: {stdout}'
    idle()
    count = 2 if op == 'allgather' else 4
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('NCCL_', 'UCCL_', 'CUDA_VISIBLE_DEVICES'))}
    env.update(CONFIGS[config], NCCL_SOCKET_IFNAME='ibp55s0f0',
               NCCL_IB_HCA='mlx5_0,mlx5_1', CUDA_VISIBLE_DEVICES='0,1,2,3',
               NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,ENV,GRAPH,P2P,SHM,NET',
               NCCL_DEBUG_FILE=str(OUT / (name + '.debug.%h.%p.log')))
    command = [MPI, '--host', f'10.10.55.2:{count},10.10.55.1:{count}',
               '-np', str(count * 2), '--map-by', f'ppr:{count}:node',
               '--bind-to', 'none', '--mca', 'pml', 'ob1', '--mca', 'btl',
               'self,vader,tcp', '--mca', 'btl_tcp_if_include', 'ibp55s0f0']
    for key in sorted(env):
        if key.startswith(('NCCL_', 'CUDA_VISIBLE_DEVICES')):
            command.extend(['-x', key])
    warm = 5 if stage == 'calibrate' else 10
    command += [str(BINARY), '--op', op, '--gpus-per-process',
                '2' if op == 'allgather' else '1', '-b', str(sizes(op)[0]),
                '-e', str(1 << 30), '-w', str(warm), '-n', '10']
    started = datetime.now(timezone.utc).isoformat()
    print(name + ': starting', flush=True)
    with stdout.open('w') as out, (OUT / f'{name}.stderr').open('w') as err:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                timeout=300)
    subprocess.run(['scp', REMOTE + ':' + str(OUT / (name + '.debug.*.log')),
                    str(OUT) + '/'], check=True, stdout=subprocess.DEVNULL)
    value = dict(stage=stage, operation=op, config=config, repeat=repeat,
                 command=command, environment={k: v for k, v in env.items()
                                               if k.startswith(('NCCL_', 'CUDA_'))},
                 started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
                 returncode=result.returncode, warmups=warm, iterations=10)
    if result.returncode == 0:
        value['means_us'] = samples(stdout, op, warm)
    dump(record, value)
    assert result.returncode == 0, record
    evidence = '\n'.join(p.read_text() for p in OUT.glob(name + '.debug.*.log'))
    assert 'NCCL version 2.28.7' in evidence, name
    if config.startswith('ce_'):
        assert '/CE' in evidence and 'via P2P/' in evidence, 'CE transport missing: ' + name
    else:
        assert '/CE' not in evidence, 'unexpected CE transport: ' + name
    print(name + ': PASS', flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['calibrate', 'calibrate-ce', 'select', 'evaluate'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'calibrate':
        dump(DATA / 'candidates.json', CONFIGS)
        jobs = [(op, config) for op in ('allreduce', 'allgather', 'reducescatter')
                for config in CONFIGS if op == 'allgather' or config.startswith('sm_')]
        random.Random(20261116).shuffle(jobs)
        first = ('allgather', 'ce_ring_simple')
        jobs.remove(first)
        jobs.insert(0, first)
        for repeat in (1, 2):
            for op, config in jobs if repeat == 1 else reversed(jobs):
                run('calibrate', op, config, repeat)
    elif args.stage == 'calibrate-ce':
        # Follow-up request: extend CE to AR/RS without rerunning existing curves.
        jobs = [(op, config) for op in ('allreduce', 'reducescatter')
                for config in CONFIGS if config.startswith('ce_')]
        random.Random(20261117).shuffle(jobs)
        for repeat in (1, 2):
            for op, config in jobs if repeat == 1 else reversed(jobs):
                run('calibrate', op, config, repeat)
    elif args.stage == 'select':
        schedule, scores = [], []
        for op in ('allreduce', 'allgather', 'reducescatter'):
            for family in ('sm', 'ce'):
                candidates = []
                for config in CONFIGS:
                    if not config.startswith(family + '_'):
                        continue
                    records = [json.loads((OUT / f'calibrate_{op}_{config}_{r}.json').read_text())
                               for r in (1, 2)]
                    assert all(r['returncode'] == 0 for r in records)
                    score = math.exp(statistics.mean(math.log(float(r['means_us'][str(n)]))
                                                     for r in records for n in sizes(op)))
                    entry = dict(operation=op, family=family, config=config, geometric_mean_us=score)
                    scores.append(entry)
                    candidates.append(entry)
                winner = min(candidates, key=lambda x: x['geometric_mean_us'])
                schedule.append(dict(winner, environment=CONFIGS[winner['config']]))
        dump(DATA / 'calibration_scores.json', scores)
        dump(DATA / 'schedule.json', schedule)
        print(json.dumps(schedule, indent=2))
    else:
        schedule = json.loads((DATA / 'schedule.json').read_text())
        for repeat in (1, 2, 3):
            for entry in schedule if repeat != 2 else reversed(schedule):
                run('evaluate', entry['operation'], entry['config'], repeat)


if __name__ == '__main__':
    main()

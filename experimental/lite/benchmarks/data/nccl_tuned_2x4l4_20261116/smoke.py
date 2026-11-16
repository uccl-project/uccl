"""Check legacy AG process layout and a two-GPU/process non-power-of-two tail."""
import json
import os
import subprocess

from run import BINARY, CONFIGS, DATA, MPI, OUT

records = []
for gpus, size, config in [(1, 1048576, 'sm_default'), (2, 6291968, 'ce_ring_simple')]:
    count = 4 // gpus
    name = f'smoke_ag_gpp{gpus}'
    stdout = OUT / (name + '.stdout')
    assert not stdout.exists(), stdout
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NCCL_', 'UCCL_'))}
    env.update(CONFIGS[config], NCCL_SOCKET_IFNAME='ibp55s0f0',
               NCCL_IB_HCA='mlx5_0,mlx5_1', NCCL_DEBUG='WARN', CUDA_VISIBLE_DEVICES='0,1,2,3')
    command = [MPI, '--host', f'10.10.55.2:{count},10.10.55.1:{count}', '-np', str(count * 2),
               '--map-by', f'ppr:{count}:node', '--bind-to', 'none', '--mca', 'pml', 'ob1',
               '--mca', 'btl', 'self,vader,tcp', '--mca', 'btl_tcp_if_include', 'ibp55s0f0']
    for key in sorted(env):
        if key.startswith(('NCCL_', 'CUDA_VISIBLE_DEVICES')):
            command.extend(['-x', key])
    command += [str(BINARY), '--op', 'allgather', '--gpus-per-process', str(gpus),
                '-b', str(size), '-e', str(size), '-w', '5', '-n', '10']
    with stdout.open('w') as out, (OUT / (name + '.stderr')).open('w') as err:
        subprocess.run(command, env=env, stdout=out, stderr=err, timeout=180, check=True)
    samples = [line.split(',') for line in stdout.read_text().splitlines() if line.startswith('nccl,')]
    assert len(samples) == 15 and sum(f[1] == 'timed' for f in samples) == 10
    assert all(int(f[3]) == size and int(f[5]) == 0 for f in samples)
    records.append(dict(name=name, command=command, settings=CONFIGS[config],
                        checked_warmups=5, checked_timed_calls=10))
(DATA / 'smoke_validation.json').write_text(json.dumps(records, indent=2) + '\n')

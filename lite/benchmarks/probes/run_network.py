"""Run the two-host network probe after observing idle GPUs on both hosts.

This does not reserve the cluster or assert exclusive CPU/fabric access. It
records resource observations and rejects a run if another GPU process appears.
It never stops another user's process.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import time

LOCAL_HOSTS = {platform.node(), 'localhost', '127.0.0.1', '::1',
               *subprocess.check_output(['hostname', '-I'], text=True).split()}

def remote(host, arguments):
    if host in LOCAL_HOSTS:
        return subprocess.check_output(arguments, text=True, timeout=15)
    return subprocess.check_output(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host,
         shlex.join(arguments)], text=True, timeout=15)


def observation(host):
    apps = remote(host, ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    devices = remote(host, ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'])
    return dict(host=host, time=time.time(),
                apps=list(csv.reader(apps.splitlines(), skipinitialspace=True)),
                devices=list(csv.reader(devices.splitlines(), skipinitialspace=True)))


def idle(record, ranks):
    return not record['apps'] and len(record['devices']) >= ranks and all(
        int(row[2]) == 0 and int(row[3]) == 0 for row in record['devices'])


def foreign_apps(records, binary, owned):
    foreign = []
    for record in records:
        known = owned.setdefault(record['host'], set())
        for row in record['apps']:
            if row[2] == binary:
                known.add(row[1])
            elif row[2] in ('[No data]', '[Not Found]', '[N/A]') and row[1] in known:
                # NVML may still report this allocation after our rank exits,
                # when /proc can no longer resolve its executable name.
                continue
            else:
                foreign.append((record['host'], row))
    return foreign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hosts', required=True, help='local-root,remote host addresses')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('build/uccl_lite_calibrate_network'))
    parser.add_argument('--mpirun', required=True)
    parser.add_argument('--iface', required=True)
    parser.add_argument('--hcas', default='mlx5_0,mlx5_1')
    parser.add_argument('--peer-hcas', help='optional different remote enumeration')
    parser.add_argument('--ranks-per-host', type=int, default=4)
    parser.add_argument('--max-bytes', type=int, default=64 << 20)
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--job-seconds', type=int, default=300)
    args = parser.parse_args()
    hosts = args.hosts.split(',')
    if len(hosts) != 2 or hosts[0] == hosts[1] or not 1 <= args.ranks_per_host <= 32 or args.wait_seconds < 0 or args.job_seconds < 1:
        parser.error('two distinct hosts, 1–32 ranks/host and valid deadlines required')
    names = [remote(host, ['hostname']).strip() for host in hosts]
    if names[0] != platform.node() or names[0] == names[1]:
        parser.error('first host must be this machine; second must be a distinct host')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out / 'uccl_lite_calibrate_network'
    shutil.copy2(args.binary.resolve(strict=True), binary)
    manifest = dict(hosts=hosts, hostnames=names, ranks_per_host=args.ranks_per_host,
                    max_bytes=args.max_bytes, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    scope='isolated probe flow; sender host posting and CQ completion; receiver visibility/validation outside timing',
                    dtype='uint32 raw payload', message_bytes=[16, 1 << 20, args.max_bytes],
                    baseline=None, warmups=2, iterations=5, repetitions=3,
                    isolated_probe_flows=1, concurrent_probe_flows=2 * args.ranks_per_host,
                    concurrent_scope='max-rank cohort including MPI start/end barriers; every sender and cohort duration retained',
                    concurrent_message_bytes=[1 << 20, args.max_bytes],
                    best_variant_envelope=False, status='waiting')
    manifest['hardware'] = [dict(host=host,
        gpus=remote(host, ['nvidia-smi', '--query-gpu=index,name,uuid,pci.bus_id,driver_version', '--format=csv,noheader']),
        cpu=remote(host, ['lscpu'])) for host in hosts]
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    # Each rank derives topology/local rank itself; this wrapper only selects
    # the explicit experiment's HCA enumeration and process placement.
    wrapper = out / 'launch.py'
    wrapper.write_text(
        'import os, platform\n'
        + 'os.environ["MSCCLPP_HCA_DEVICES"] = (' + repr(args.peer_hcas or args.hcas)
        + ' if platform.node() == ' + repr(names[1]) + ' else ' + repr(args.hcas) + ')\n'
        + 'os.execv(' + repr(str(binary)) + ', ' + repr([str(binary), str(out / 'network.profile'), str(args.max_bytes)]) + ')\n')
    remote(hosts[1], ['mkdir', '-p', str(out)])
    subprocess.run(['rsync', '-a', '-s', str(out) + '/', hosts[1] + ':' + str(out) + '/'], check=True)
    deadline = time.monotonic() + args.wait_seconds
    quiet = 0
    with (out / 'resources.jsonl').open('w') as resources:
        while quiet < 2:
            records = [observation(host) for host in hosts]
            resources.write(json.dumps(dict(phase='before', records=records)) + '\n'); resources.flush()
            ready = all(idle(record, args.ranks_per_host) for record in records)
            quiet = quiet + 1 if ready else 0
            if quiet == 2:
                break
            if not ready and time.monotonic() >= deadline:
                manifest['status'] = 'resources-busy'; save()
                raise RuntimeError('GPU resources are occupied; no probe was launched')
            print('Observed idle GPUs; confirming' if ready else 'Waiting for idle GPUs on both hosts', flush=True)
            time.sleep(2 if ready else min(15, max(1, deadline - time.monotonic())))
        env = os.environ.copy()
        env.update(MSCCLPP_SOCKET_IFNAME=args.iface, MSCCLPP_IBV_MODE='host-no-atomic')
        placement = ','.join(host + ':' + str(args.ranks_per_host) for host in hosts)
        command = [args.mpirun, '--host', placement, '-np', str(2 * args.ranks_per_host),
                   '--map-by', f'ppr:{args.ranks_per_host}:node', '--bind-to', 'none',
                   '--timeout', str(args.job_seconds), '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
                   '--mca', 'btl_tcp_if_include', args.iface, '--mca', 'oob_tcp_if_include', args.iface,
                   '-x', 'MSCCLPP_SOCKET_IFNAME', '-x', 'MSCCLPP_IBV_MODE', 'python3', '-B', str(wrapper)]
        manifest.update(status='running', command=command, started=time.time()); save()
        contaminated = False
        owned = {}
        print('Launching isolated network probe', flush=True)
        with (out / 'probe.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    records = [observation(host) for host in hosts]
                    resources.write(json.dumps(dict(phase='during', records=records)) + '\n'); resources.flush()
                    foreign = foreign_apps(records, str(binary), owned)
                    if foreign:
                        contaminated = True
                        manifest['foreign_gpu_processes'] = foreign
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                        break
                except (subprocess.SubprocessError, OSError) as error:
                    # Monitoring failure cannot establish isolation, even if the
                    # probe itself subsequently passes its numerical oracle.
                    contaminated = True
                    resources.write(json.dumps(dict(phase='monitor-error', error=str(error))) + '\n'); resources.flush()
                time.sleep(1)
            status = process.wait()
        manifest.update(status='contaminated' if contaminated else 'passed' if status == 0 else 'failed',
                        returncode=status, ended=time.time()); save()
        if contaminated and (out / 'network.profile').exists():
            (out / 'network.profile').rename(out / 'network.contaminated.profile')
        if contaminated or status:
            raise RuntimeError('network probe did not produce an isolated successful result; see manifest/log')
        print('PASS: network probe and resource monitoring; profile=' + str(out / 'network.profile'))


if __name__ == '__main__':
    main()

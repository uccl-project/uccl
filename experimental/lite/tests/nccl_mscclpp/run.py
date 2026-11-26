#!/usr/bin/env python3
"""Run unchanged MPI nccl-tests on two four-GPU hosts, with route assertions.

This is a candidate evaluation, not automatic deployment admission. Artifacts
are staged into a new build directory at the same path on both hosts.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import mscclpp_profile
import runtime_profile


def checked(log, operation, routed):
    if not re.search(r'# Out of bounds values\s*:\s*0\s+OK', log):
        raise RuntimeError('missing nccl-tests correctness summary')
    ranks = {int(x) for x in re.findall(r'#\s+Rank\s+(\d+)\s+Group', log)}
    if ranks != set(range(8)):
        raise RuntimeError('expected eight nccl-tests ranks')
    counters = re.findall(r'UCCL Lite MSCCL\+\+: rank=(\d+) AG=(\d+) RS=(\d+) AR=(\d+) fallback=(\d+)', log)
    if routed and (len(counters) != 8 or {int(x[0]) for x in counters} != set(range(8)) or
                   any(int(x[operation]) == 0 for x in counters)):
        raise RuntimeError('test did not route UCCL on every rank')
    if routed is False and any(sum(map(int, x[1:4])) for x in counters):
        raise RuntimeError('fallback test unexpectedly routed UCCL')
    rows = {}
    for line in log.splitlines():
        fields = line.split()
        if len(fields) == 13 and fields[0].isdigit():
            if fields[8] != '0' or fields[12] != '0':
                raise RuntimeError('incorrect nccl-tests output')
            rows[int(fields[0])] = [float(fields[5]), float(fields[9])]
    if not rows:
        raise RuntimeError('no measured rows')
    return {'rows_us': rows, 'route_counters': counters}


def checked_strict(log, operation):
    result = checked(log, operation, False)
    loaded = re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0', log)
    fallback = re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)', log)
    if sorted(map(int, loaded)) != list(range(8)) or sorted((int(r), int(n)) for r,n in fallback) != [(r,0) for r in range(8)]:
        raise RuntimeError('strict native-library/fallback audit is incomplete')
    routes = re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', log)
    expected = {(rank, operation, size) for rank in range(8) for size in result['rows_us']}
    actual = {(int(rank), int(op), int(size)) for rank,op,size,count in routes if int(count)>0}
    if actual != expected or len(routes) != len(expected):
        raise RuntimeError('not every size executed UCCL on every rank')
    result['strict_routes'] = routes
    result['native_nccl_loaded'] = False
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strict', action='store_true', default=True, help='mandatory: never execute native NCCL in candidate jobs')
    p.add_argument('--hosts', required=True, help='ordered node0,node1 SSH/MPI hosts')
    p.add_argument('--iface', required=True)
    p.add_argument('--hcas', default='mlx5_0,mlx5_1')
    p.add_argument('--peer-hcas', help='ABI-only diagnostic: override MSCCL++ HCA order on the second host')
    p.add_argument('--devices', help='ABI-only CUDA_VISIBLE_DEVICES order on the first host (and default for the second)')
    p.add_argument('--peer-devices', help='ABI-only CUDA_VISIBLE_DEVICES order on the second host')
    p.add_argument('--expect-network-order', help='assert an eight-entry execution-to-logical rank permutation')
    p.add_argument('--rank-by', choices=('slot', 'node'), default='slot', help='ABI-only global MPI rank ordering')
    p.add_argument('--reverse-rank-devices', action='store_true', help='ABI-only assign CUDA devices in reverse local-rank order')
    p.add_argument('--mpirun', required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--library', type=Path, default=ROOT / 'build/nccl_lite/libuccl_lite_nccl_strict.so')
    p.add_argument('--legacy-profile', action='store_true', help='use the historical machine-specific profile format for old library artifacts')
    p.add_argument('--network-profile', type=Path, help='optional measured endpoint-path profile')
    p.add_argument('--expect-network-plan', choices=('matched', 'physical'), help='assert measured-profile result in the ABI job')
    p.add_argument('--expect-network-group', choices=('matched', 'absent'), help='assert exact concurrent flow evidence in the ABI job')
    p.add_argument('--network-rejections', type=Path, help='NCCL-only all-rank network preparation rejection client')
    p.add_argument('--network-group-mismatch', type=Path, help='valid profile differing only in concurrent evidence, for one-rank rejection')
    p.add_argument('--abi-extended', action='store_true', help='include the 32-MiB multi-piece ABI case')
    p.add_argument('--abi-boundary512', action='store_true', help='include the 512-KiB ABI boundary case; requires the matching ABI binary and --abi-extended')
    p.add_argument('--abi-serial-packet', action='store_true', help='add 32/64/128-KiB mixed ABI cases and delayed small AG; requires --abi-boundary512 and its matching binary')
    p.add_argument('--abi-medium64', action='store_true', help='add the 64-MiB mixed ABI boundary; requires --abi-serial-packet and its matching binary')
    p.add_argument('--abi-medium128', action='store_true', help='add the 128-MiB mixed ABI boundary; requires --abi-medium64 and its matching binary')
    p.add_argument('--abi-only', action='store_true')
    p.add_argument('--job-timeout', type=int, default=900)
    p.add_argument('--abi', type=Path, help='optional NCCL-only ABI stress executable')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--max-bytes', type=int, default=1 << 20,
                   help='power-of-two complete extent, up to 1 GiB')
    p.add_argument('--shapes', type=Path, help='fixed JSON [operation, count] candidate rows')
    p.add_argument('--rounds', type=int, default=1, help='independent ABBA rounds')
    p.add_argument('--performance-only', action='store_true', help='omit additional ABI/graph/misalignment jobs; retain nccl-tests correctness checking in every sweep')
    p.add_argument('--max-regression', type=float, help='fail if any median candidate/native ratio exceeds 1 + this fraction')
    args = p.parse_args()
    if args.expect_network_plan and not (args.network_profile and args.abi):
        p.error('--expect-network-plan requires --network-profile and --abi')
    if args.expect_network_group and not (args.network_profile and args.abi):
        p.error('--expect-network-group requires --network-profile and --abi')
    if args.network_rejections and not args.network_profile:
        p.error('--network-rejections requires --network-profile')
    if args.network_group_mismatch and not args.network_rejections:
        p.error('--network-group-mismatch requires --network-rejections')
    placement = bool(args.peer_hcas or args.devices or args.peer_devices)
    explicit_ranks = args.rank_by != 'slot' or args.reverse_rank_devices
    if args.legacy_profile and (args.devices or args.peer_devices or explicit_ranks):
        p.error('device permutations require the runtime profile format')
    if (placement or explicit_ranks) and not args.abi_only:
        p.error('HCA/device placement overrides require --abi-only')
    expected_order = None
    if args.expect_network_order:
        try:
            expected_order = tuple(map(int, args.expect_network_order.split(',')))
        except ValueError:
            p.error('--expect-network-order requires integer ranks')
        if sorted(expected_order) != list(range(8)) or not args.abi:
            p.error('--expect-network-order requires a permutation of 0..7 and --abi')
    if args.abi_boundary512 and not args.abi_extended:
        p.error('--abi-boundary512 requires --abi-extended')
    if args.abi_serial_packet and not args.abi_boundary512:
        p.error('--abi-serial-packet requires --abi-boundary512')
    if args.abi_medium64 and not args.abi_serial_packet:
        p.error('--abi-medium64 requires --abi-serial-packet')
    if args.abi_medium128 and not args.abi_medium64:
        p.error('--abi-medium128 requires --abi-medium64')
    if (args.abi_only or args.abi_extended) and not args.abi:
        p.error('--abi-only requires the strict ABI executable via --abi')
    if args.job_timeout < 1 or args.warmup < 0 or args.iterations < 1:
        p.error('job-timeout/iterations must be positive and warmup nonnegative')
    if args.max_bytes < 1024 or args.max_bytes > 1 << 30 or args.max_bytes & (args.max_bytes - 1) or args.rounds < 1:
        p.error('max-bytes must be a power of two in 1 KiB–1 GiB; rounds must be positive')
    if args.max_regression is not None and (not math.isfinite(args.max_regression) or args.max_regression < 0):
        p.error('max-regression must be finite and nonnegative')
    hosts = args.hosts.split(',')
    if len(hosts) != 2 or hosts[0] == hosts[1]:
        p.error('two distinct ordered hosts required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'validation_driver.py')
    real = args.real_nccl.resolve(strict=True)
    library = out / args.library.name
    shutil.copy2(args.library, library)
    if args.network_profile:
        shutil.copy2(args.network_profile, out / 'network.profile')
    if args.network_rejections:
        shutil.copy2(args.network_rejections, out / 'network_prepare_rejection')
        (out / 'network-corrupt.profile').write_text('deliberately invalid network resource profile\n')
    if args.network_group_mismatch:
        shutil.copy2(args.network_group_mismatch, out / 'network-groups-different.profile')
    (out / 'strict').mkdir()
    (out / 'strict/libnccl.so.2').symlink_to('../' + library.name)
    dependencies = subprocess.check_output(['ldd', str(library)], text=True)
    if re.search(r'libnccl\.so', dependencies):
        raise RuntimeError('strict candidate depends on native NCCL')
    (out / 'dependencies.txt').write_text(dependencies)
    profile_script = 'mscclpp_profile.py' if args.legacy_profile else 'runtime_profile.py'
    for name in ['mscclpp_profile.py', 'runtime_profile.py', 'dropin_profile.py']:
        shutil.copy2(ROOT / 'integration/nccl_lite' / name, out / name)
    operations = [('all_gather', 1), ('reduce_scatter', 2), ('all_reduce', 3)]
    for op, _ in operations:
        shutil.copy2(args.tests / (op + '_perf'), out / (op + '_perf'))
    if args.abi:
        if args.max_bytes < (128 if args.abi_medium128 else 64 if args.abi_medium64 else 32 if args.abi_extended else 16) << 20:
            p.error('strict ABI test profile does not cover its largest case')
        shutil.copy2(args.abi, out / 'abi_check')
    if placement:
        peer_name = subprocess.check_output(['ssh', hosts[1], 'hostname'], text=True).strip()
        setup = "import os, platform, sys\npeer = platform.node() == " + repr(peer_name) + "\n"
        if args.peer_hcas:
            setup += "if peer:\n    os.environ['MSCCLPP_HCA_DEVICES'] = " + repr(args.peer_hcas) + "\n"
        if args.devices or args.peer_devices:
            setup += "devices = " + repr(args.peer_devices or args.devices) + " if peer else " + repr(args.devices) + "\n"
            setup += "if devices is not None:\n    os.environ['CUDA_VISIBLE_DEVICES'] = devices\n"
        (out / 'placement_launcher.py').write_text(setup
            + "os.execv(sys.executable, [sys.executable, '-B', "
            + repr(str(out / profile_script)) + ", *sys.argv[1:]])\n")
    subprocess.run(['ssh', hosts[1], 'mkdir -p ' + shlex.quote(str(out))], check=True)
    subprocess.run(['rsync', '-a', str(out) + '/', f'{hosts[1]}:{out}/'], check=True)
    env = os.environ.copy()
    if env.get('LD_PRELOAD'):
        p.error('unset LD_PRELOAD before validation')
    env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT=socket.gethostbyname(hosts[0]),
               NCCL_SOCKET_IFNAME=args.iface, NCCL_IB_HCA=args.hcas,
               MSCCLPP_SOCKET_IFNAME=args.iface, MSCCLPP_HCA_DEVICES=args.hcas, MSCCLPP_IBV_MODE='host-no-atomic',
               LD_LIBRARY_PATH=str(real.parent) + ':' + env.get('LD_LIBRARY_PATH', ''))
    exported = ['UCCL_LITE_MSCCLPP_CONTROL_ROOT', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA', 'MSCCLPP_SOCKET_IFNAME',
                'MSCCLPP_IBV_MODE', 'MSCCLPP_HCA_DEVICES', 'LD_LIBRARY_PATH']
    if args.reverse_rank_devices:
        env['UCCL_LITE_ABI_REVERSE_DEVICES'] = '1'
        exported.append('UCCL_LITE_ABI_REVERSE_DEVICES')
    if args.network_profile:
        env['UCCL_LITE_NETWORK_PROFILE'] = str(out / 'network.profile')
        exported.append('UCCL_LITE_NETWORK_PROFILE')
    def mpi(hosts, n, mapping):
        command = [args.mpirun, '--timeout', str(args.job_timeout), '--host', hosts, '-np', str(n), '--map-by', mapping,
                   '--rank-by', args.rank_by,
                   '--bind-to', 'none', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
                   '--mca', 'btl_tcp_if_include', args.iface,
                   '--mca', 'oob_tcp_if_include', args.iface]
        for key in exported:
            command += ['-x', key]
        return command
    launcher = ['python3', '-B', str(out / ('placement_launcher.py' if placement else profile_script))]
    options = ['--library', str(library), '--real-nccl', str(real)]
    if args.legacy_profile:
        options.append('--strict')
    records = []
    for i, host in enumerate(hosts):
        value = subprocess.check_output(mpi(host, 1, 'slot') + launcher + ['probe'] + options,
                                        env=env, text=True, timeout=60)
        record = json.loads(value)
        records.append(record)
        (out / f'host{i}.json').write_text(value)
    shapes = [(op, (1 << power) // (4 if op == 3 else 32))
              for _, op in operations for power in range(10, args.max_bytes.bit_length())]
    if args.shapes:
        shapes = json.loads(args.shapes.read_text())
    expected_shapes = {(op, (1 << power) // (4 if op == 3 else 32)) for _, op in operations for power in range(10, args.max_bytes.bit_length())}
    if {tuple(x) for x in shapes} != expected_shapes:
        p.error('strict testing requires all three operations at every tested size')
    (out / 'shapes.json').write_text(json.dumps(shapes, indent=2) + '\n')
    profile = out / 'candidate.tsv'
    if args.legacy_profile:
        mscclpp_profile.write_candidate(profile, records, shapes, strict=True)
    else:
        rank_map = None
        if explicit_ranks:
            rank_map = []
            for rank in range(8):
                host, device = (rank % 2, rank // 2) if args.rank_by == 'node' else (rank // 4, rank % 4)
                rank_map.append([host, 3 - device if args.reverse_rank_devices else device])
        runtime_profile.write_candidate(profile, records, shapes, 'float32',
                                        runtime_profile.catalog(library), 'network-hierarchy', rank_map)
    subprocess.run(['rsync', '-a', str(profile), f'{hosts[1]}:{out}/'], check=True)
    prefix = mpi(','.join(h + ':4' for h in hosts), 8, 'ppr:4:node')
    manifest = {'scope': 'strict UCCL-only candidate; native NCCL absent from candidate processes',
                'hardware': 'two hosts, four participating GPUs per host; FP32 SUM',
                'hardware_details': [dict(hostname=r['hostname'], physical_inventory=r['details']['gpus'], participating_uuids=r['uuids']) for r in records],
                'timing': 'nccl-tests maximum-rank host enqueue/completion batch mean; both layouts',
                'baseline': 'native NCCL, identical environment and test binaries',
                'selection': 'one library with topology-derived rails and retained algorithm policy; no native fallback or per-size best-of measurement envelope',
                'repetitions': f'{args.rounds} native/candidate/candidate/native rounds; {2 * args.rounds} jobs per backend',
                'warmup': args.warmup, 'iterations': args.iterations,
                'max_bytes': args.max_bytes, 'shapes': shapes,
                'max_regression': args.max_regression,
                'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                'commands': [], 'results': {}}
    if args.network_profile:
        manifest['network_profile_sha256'] = hashlib.sha256((out / 'network.profile').read_bytes()).hexdigest()
        manifest['network_model_scope'] = 'isolated directed path medians plus shared physical bounds; no concurrent throughput qualification'
    if args.abi_only:
        manifest.update(timing='not measured; correctness-only ABI check',
                        baseline='none; native NCCL is not run in ABI-only mode',
                        repetitions='one ABI job, two communicator lifetimes',
                        warmup=None, iterations=None)
    def run(label, op, number, backend, extra=(), routes=True, missing=False):
        command = prefix.copy()
        if backend == 'candidate':
            command += launcher + ['launch'] + options + ['--profile', str(out / 'missing.tsv' if missing else profile), '--']
        else:
            command += ['env', 'LD_PRELOAD=' + str(real)]
        command += [str(out / (op + '_perf')), '-b', '1K', '-e', str(args.max_bytes), '-f', '2',
                    '-g', '1', '-d', 'float', '-w', str(args.warmup), '-n', str(args.iterations),
                    '-c', '1', '-a', '3', *extra]
        manifest['commands'].append({'label': label, 'argv': command})
        print('Running', label, flush=True)
        with (out / (label + '.log')).open('w') as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900)
        log = (out / (label + '.log')).read_text()
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; see its log')
        manifest['results'][label] = checked_strict(log, number) if backend == 'candidate' else checked(log, number, False)
        expected = {1 << i for i in range(10, args.max_bytes.bit_length())}
        rows = manifest['results'][label]['rows_us']
        if set(rows) != expected or any(not math.isfinite(t) or t <= 0 for row in rows.values() for t in row):
            raise RuntimeError(f'{label}: incomplete size coverage or invalid timing')
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def reject(label, op, extra, message, missing=False):
        command = prefix + launcher + ['launch'] + options + ['--profile', str(out / 'missing.tsv' if missing else profile), '--', str(out / (op + '_perf')), '-b', '1K', '-e', '1K', '-g', '1', '-d', 'float', '-w', '1', '-n', '1', '-c', '1', *extra]
        manifest['commands'].append({'label': label, 'argv': command})
        with (out / (label + '.log')).open('w') as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=90)
        log = (out / (label + '.log')).read_text()
        if result.returncode == 0 or message not in log:
            raise RuntimeError(label + ': expected strict rejection was not observed')
        manifest.setdefault('rejections', {})[label] = {'exit_code': result.returncode, 'reason': message}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for op, number in operations:
        if args.abi_only:
            break
        for i, backend in enumerate(['native', 'candidate', 'candidate', 'native'] * args.rounds):
            run(f'{op}_{i}_{backend}', op, number, backend,
                routes=any(row[0] == number for row in shapes))
        if args.performance_only:
            continue
        run(op + '_unaligned', op, number, 'candidate', ['-u', '1'])
        # Eager warmups complete; external capture must reject before queuing work.
        reject(op + '_graph_rejected', op, ['-G', '1'], 'external graph capture unsupported')
    if not args.performance_only and not args.abi_only:
        reject('product_rejected', 'all_reduce', ['-o', 'prod'], 'outside the strict profile')
        reject('missing_profile_rejected', 'all_reduce', [], 'profile/backend communicator mismatch', missing=True)
        reject('group_rejected', 'all_reduce', ['-m', '2'], 'grouped collectives unsupported')
    if args.abi:
        command = prefix + launcher + ['launch'] + options + ['--profile', str(profile), '--', str(out / 'abi_check')]
        manifest['commands'].append({'label': 'abi', 'argv': command})
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('Running abi', flush=True)
        with (out / 'abi.log').open('w') as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
        log = (out / 'abi.log').read_text()
        from collections import Counter
        routes = re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', log)
        abi_sizes = [1024, 16384, 262144, 1048576, 8388608, 16777216] + ([33554432] if args.abi_extended else [])
        if args.abi_boundary512:
            abi_sizes.append(524288)
        if args.abi_serial_packet:
            abi_sizes += [32768, 65536, 131072]
        if args.abi_medium64:
            abi_sizes.append(67108864)
        if args.abi_medium128:
            abi_sizes.append(134217728)
        abi_calls = len(abi_sizes) * 3 * 3 * 2
        expected = Counter({(rank, op, size, 3): 2 for rank in range(8) for op in [1,2,3]
                            for size in abi_sizes})
        loaded = re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0', log)
        if (result.returncode or f'Strict ABI full-buffer checks passed: {abi_calls} calls/rank, two lifetimes' not in log
                or Counter(tuple(map(int, row)) for row in routes) != expected
                or any(loaded.count(str(rank)) != 2 for rank in range(8))):
            raise RuntimeError('strict ABI correctness or exact per-lifetime route assertion failed')
        manifest['results']['abi'] = {'full_buffer_calls_per_rank': abi_calls, 'lifetimes': 2, 'native_nccl_loaded': False}
        if expected_order:
            mappings = re.findall(r'UCCL Lite network rank mapping: rank=(\d+) execution_rank=(\d+).*?order=([0-9,]+)', log)
            actual = Counter((int(r), int(e), tuple(map(int, order.rstrip(',').split(',')))) for r, e, order in mappings)
            expected = Counter({(r, expected_order.index(r), expected_order): 2 for r in range(8)})
            if actual != expected:
                raise RuntimeError('network execution-rank mapping assertion failed')
            manifest['results']['abi']['execution_to_logical_order'] = expected_order
        if args.expect_network_plan:
            plans = re.findall(r'UCCL Lite measured network plan: rank=(\d+) matched=(\d+)', log)
            expected_match = int(args.expect_network_plan == 'matched')
            if Counter((int(rank), int(match)) for rank, match in plans) != Counter({(rank, expected_match): 2 for rank in range(8)}):
                raise RuntimeError('measured-network plan assertion failed')
            manifest['results']['abi']['measured_network_plan'] = args.expect_network_plan
        if args.expect_network_group:
            groups = re.findall(r'UCCL Lite measured network plan: rank=(\d+).*?group_matched=(\d+)', log)
            match = int(args.expect_network_group == 'matched')
            if Counter((int(rank), int(value)) for rank, value in groups) != Counter({(rank, match): 2 for rank in range(8)}):
                raise RuntimeError('concurrent-network group assertion failed')
            manifest['results']['abi']['concurrent_network_group'] = args.expect_network_group
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.network_rejections:
        modes = ['mismatch', 'corrupt'] + (['groups'] if args.network_group_mismatch else [])
        if explicit_ranks:
            modes += ['binding', 'rank-map']
        for mode in modes:
            negative_profile = out / ('network-groups-different.profile' if mode == 'groups' else 'network-corrupt.profile')
            command = prefix + launcher + ['launch'] + options + ['--profile', str(profile), '--',
                str(out / 'network_prepare_rejection'), mode, str(negative_profile)]
            manifest['commands'].append({'label': 'network_' + mode, 'argv': command})
            with (out / ('network_' + mode + '.log')).open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode or 'PASS: network preparation rejected on all 8 ranks; mode=' + mode not in (out / ('network_' + mode + '.log')).read_text():
                raise RuntimeError('network preparation rejection failed: ' + mode)
            if mode == 'binding' and 'profile/topology/capability consensus failed' not in (out / ('network_' + mode + '.log')).read_text():
                raise RuntimeError('rank binding rejection did not exercise the frontend gate')
            if mode == 'rank-map' and 'profile/backend communicator mismatch' not in (out / ('network_' + mode + '.log')).read_text():
                raise RuntimeError('duplicate rank map did not exercise profile parsing rejection')
            manifest['results']['network_' + mode] = {'all_ranks_rejected': True}
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.abi_only:
        print('Strict ABI checks passed', flush=True)
        return
    ratios = {}
    for op, _ in operations:
        order = ['native', 'candidate', 'candidate', 'native'] * args.rounds
        runs = [manifest['results'][f'{op}_{i}_{backend}']['rows_us']
                for i, backend in enumerate(order)]
        ratios[op] = {size: [statistics.median([r[size][layout] for r, b in zip(runs, order) if b == 'candidate']) /
                             statistics.median([r[size][layout] for r, b in zip(runs, order) if b == 'native'])
                             for layout in range(2)] for size in runs[0]}
    (out / 'candidate_over_nccl.json').write_text(json.dumps(ratios, indent=2) + '\n')
    print('Correctness and route assertions passed. Ratios > 1 mean slower than NCCL.', flush=True)
    if args.max_regression is not None:
        failures = [{'operation': op, 'bytes': size, 'layout': layout, 'ratio': ratio}
                    for op, rows in ratios.items() for size, pair in rows.items()
                    for layout, ratio in zip(['out-of-place', 'in-place'], pair)
                    if ratio > 1 + args.max_regression]
        verdict = {'threshold': args.max_regression, 'passed': not failures,
                   'rows': sum(len(rows) * 2 for rows in ratios.values()), 'failures': failures}
        (out / 'performance_verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
        print(f'Performance gate: {len(failures)}/{verdict["rows"]} rows failed', flush=True)
        if failures:
            sys.exit(2)


if __name__ == '__main__':
    main()

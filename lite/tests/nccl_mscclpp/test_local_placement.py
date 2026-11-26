"""CPU-placement admission checks; no CUDA workload or timing changes."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import local_numa_profile as placement


class PlacementChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addresses = ['0000:4a:00.0', '0000:61:00.0', '0000:ca:00.0', '0000:e1:00.0']
        for rank, address in enumerate(self.addresses):
            p = self.root / 'bus/pci/devices' / address / 'numa_node'
            p.parent.mkdir(parents=True)
            p.write_text(str(rank // 2))
        for node, cpus in [(0, '0,2,4,6'), (1, '1,3,5,7')]:
            p = self.root / 'devices/system/node' / f'node{node}' / 'cpulist'
            p.parent.mkdir(parents=True)
            p.write_text(cpus)
        self.details = dict(launch_affinity=[0, 1, 2, 3],
            gpus='\n'.join(f'GPU-{r}, NVIDIA L4, {a}' for r, a in enumerate(self.addresses)),
            cpus='\n'.join(f'{r},{r},{r % 2},{r % 2}' for r in range(8)),
            adapter_launcher_sha256=placement.base.common.sha256(placement.base.__file__))

    def test_allowed_cpu_intersection(self):
        nodes, masks = placement.placement(self.details, self.root)
        self.assertEqual(nodes, [0, 0, 1, 1])
        self.assertEqual(masks, [[0, 2], [0, 2], [1, 3], [1, 3]])

    def test_no_remote_numa_fallback(self):
        self.details['launch_affinity'] = [0, 2]
        with self.assertRaisesRegex(ValueError, 'no allowed'):
            placement.placement(self.details, self.root)

    def test_unknown_node_refused(self):
        (self.root / 'bus/pci/devices' / self.addresses[2] / 'numa_node').write_text('-1')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            placement.placement(self.details, self.root)

    def test_changed_placement_invalidates_profile(self):
        base = dict(hostname='test', details=self.details, identity='unused',
                    versions=[12080, 12080, 22807], uuids=['0' * 32] * 4)
        original_placement = placement.placement
        with mock.patch.object(placement.base, 'record', side_effect=lambda *a: copy.deepcopy(base)), \
             mock.patch.object(placement, 'placement', side_effect=lambda d: original_placement(d, self.root)):
            before = placement.record(Path('native'), Path('candidate'))
            path = self.root / 'profile.tsv'
            placement.base.write_candidate(path, [before], [(1, 64)])
            placement.validate_profile(before, path)
            self.details['launch_affinity'].extend([4, 5])
            after = placement.record(Path('native'), Path('candidate'))
            self.assertNotEqual(before['identity'], after['identity'])
            with self.assertRaisesRegex(ValueError, 'stale'):
                placement.validate_profile(after, path)

    def test_audit_requires_all_actual_binding_markers(self):
        nodes, masks = placement.placement(self.details, self.root)
        d = dict(self.details, cpu_placement='numa-local', gpu_numa_nodes=nodes,
                 rank_cpu_affinity=masks,
                 numa_launcher_sha256=placement.base.common.sha256(placement.__file__))
        r = dict(hostname='test', details=d,
                 identity=hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(),
                 versions=[12080, 12080, 22807], uuids=['0' * 32] * 4)
        (self.root / 'identity.json').write_text(json.dumps(r))
        placement.base.write_candidate(self.root / 'profile.tsv', [r], [(1, 64)])
        (self.root / 'numa_launcher.py').write_bytes(Path(placement.__file__).read_bytes())
        (self.root / 'base_launcher.py').write_bytes(Path(placement.base.__file__).read_bytes())
        command = ['python3', '-B', str(placement.launcher_path('numa'))]
        lines = [f'UCCL Lite CPU placement: rank={rank} cpus=' + ','.join(map(str, mask)) + '\n'
                 for rank, mask in enumerate(masks)]
        placement.validate_run(self.root, 'numa', command, ''.join(lines))
        with self.assertRaisesRegex(ValueError, 'missing'):
            placement.validate_run(self.root, 'numa', command, ''.join(lines[:-1]))


if __name__ == '__main__':
    unittest.main()

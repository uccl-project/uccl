"""Physical-core placement invariants; no CUDA execution or timing changes."""
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
import local_core_profile as core
import local_cpu_profile as dispatch


class CorePlacementChecks(unittest.TestCase):
    def setUp(self):
        self.nodes = [0, 0, 1, 1]
        self.details = dict(launch_affinity=list(range(16)), cpus='\n'.join(
            f'{cpu},{cpu % 8},{cpu % 2},{cpu % 2}' for cpu in range(16)))

    def test_distinct_cores_keep_allowed_siblings(self):
        self.assertEqual(core.core_masks(self.details, self.nodes),
                         [[4, 12], [6, 14], [5, 13], [7, 15]])

    def test_two_core_groups_are_disjoint(self):
        self.assertEqual(core.core_masks(self.details, self.nodes, cores_per_rank=2),
                         [[0, 2, 8, 10], [4, 6, 12, 14], [1, 3, 9, 11], [5, 7, 13, 15]])

    def test_two_core_groups_refuse_insufficient_capacity(self):
        self.details['launch_affinity'] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            core.core_masks(self.details, self.nodes, cores_per_rank=2)

    def test_restricted_parent_mask(self):
        self.details['launch_affinity'] = [0, 1, 2, 3]
        self.assertEqual(core.core_masks(self.details, self.nodes), [[0], [2], [1], [3]])

    def test_refuses_sibling_oversubscription(self):
        self.details['launch_affinity'] = [0, 8, 1, 9]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            core.core_masks(self.details, self.nodes)

    def test_unknown_topology_refused(self):
        self.details['cpus'] = self.details['cpus'].replace('0,0,0,0', '0,-1,0,0')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            core.core_masks(self.details, self.nodes)

    def test_sysfs_snapshot_disagreement_refused(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            core.core_masks(self.details, self.nodes, [[0], [0], [1], [1]])

    def test_identity_markers_and_snapshots(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as temp:
            directory = Path(temp)
            details = dict(self.details, adapter_launcher_sha256=core.base.common.sha256(core.base.__file__))
            base_record = dict(hostname='test', details=details, identity='unused',
                               versions=[12090, 12090, 22807], uuids=['0' * 32] * 4)
            available = [[r for r in range(16) if r % 2 == node] for node in self.nodes]
            with mock.patch.object(core.base, 'record', side_effect=lambda *a: copy.deepcopy(base_record)), \
                 mock.patch.object(core.numa, 'placement', return_value=(self.nodes, available)):
                record = dispatch.make_record('core', Path('native'), Path('candidate'), directory)
            (directory / 'identity.json').write_text(json.dumps(record))
            core.base.write_candidate(directory / 'profile.tsv', [record], [(1, 64)])
            command = ['python3', '-B', str(dispatch.launcher_path('core'))]
            log = ''.join(f'UCCL Lite CPU placement: rank={r} cpus=' + ','.join(map(str, mask)) + '\n'
                          for r, mask in enumerate(record['details']['rank_cpu_affinity']))
            dispatch.validate_run(directory, 'core', command, log)
            with self.assertRaisesRegex(ValueError, 'marker'):
                dispatch.validate_run(directory, 'core', command, log.splitlines(keepends=True)[0])
            (directory / 'core_launcher.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                dispatch.validate_run(directory, 'core', command, log)

    def test_old_launch_modes_keep_paths(self):
        for mode in ['unbound', 'numa']:
            self.assertEqual(dispatch.launcher_path(mode), core.numa.launcher_path(mode))


if __name__ == '__main__':
    unittest.main()

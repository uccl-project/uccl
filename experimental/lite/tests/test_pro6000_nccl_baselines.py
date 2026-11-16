"""Guard default/tuned separation, frozen selection, and transport evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pro6000_nccl_baselines as baseline


class BaselineEvidence(unittest.TestCase):
    def test_selects_sm_or_ce_using_all_calibration_rounds(self):
        configs = {
            name: dict(op="allreduce", backend="nccl", args={}, env=env)
            for name, env in (
                ("default", {}), ("sm", {"NCCL_P2P_USE_CUDA_MEMCPY": "0"}),
                ("ce", {"NCCL_P2P_USE_CUDA_MEMCPY": "1"}))
        }
        records = [dict(job=dict(candidate=c, round=r),
                        rows={str(n): (1 if c == "default" else
                              (5 if n <= 1 << 20 else 20) if c == "sm" else 10)
                              for n in baseline.SIZES})
                   for c in configs for r in range(3)]
        selected = baseline.choose_allreduce(records, configs)
        self.assertEqual(selected["1024"], "sm")
        self.assertEqual(selected[str(1 << 30)], "ce")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            baseline.choose_allreduce(records[:-1], configs)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            baseline.choose_allreduce(records + records[:1], configs)

    def test_defaults_are_unconfigured_and_validation_is_frozen(self):
        root = Path(__file__).resolve().parents[1]
        ar = json.loads((root / "benchmarks/data/pro6000_allreduce_ce_ring_cpp_20261116/frozen-policy.json").read_text())
        ag = json.loads((root / "benchmarks/data/pro6000_ag_rs_cpp_20261116/policy.json").read_text())
        configs, selected = baseline.make_catalog(ar, ag)
        selected["allreduce"]["nccl"] = {str(n): "allreduce-nccl-sm-auto" for n in baseline.SIZES}
        policy = dict(selected=selected)
        for op in baseline.OPS:
            self.assertEqual(configs[op + "-nccl-default"]["env"], {})
        jobs = baseline.jobs_for_policy(policy, "validation")
        for op in baseline.OPS:
            for backend in ("ours", "nccl", "default"):
                for size, candidate in selected[op][backend].items():
                    matches = [j for j in jobs if j["candidate"] == candidate and j["begin"] <= int(size) <= j["end"]]
                    self.assertEqual(sorted(j["round"] for j in matches), list(range(5)))
        self.assertTrue(all(configs[j["candidate"]]["backend"] == "nccl"
                            for j in baseline.jobs_for_policy(policy, "diagnostic")))

    def test_requires_dispatch_and_actual_transport_on_every_rank(self):
        job = dict(name="test", stage="diagnostic")
        config = dict(op="allreduce", env={"NCCL_P2P_USE_CUDA_MEMCPY": "1"})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for rank in range(8):
                (folder / f"test.nccl.host.{rank}.log").write_text(
                    "Channel 00 : 0 -> 1 via P2P/CUMEM/CE\n"
                    + ("AllReduce: 1024 Bytes -> Algo RING proto SIMPLE\n" if rank == 0 else ""))
            evidence = baseline.check_transport(folder, job, config, [1024])
            self.assertEqual(evidence, json.loads(json.dumps(evidence)))
            with self.assertRaisesRegex(ValueError, "per-operation"):
                baseline.check_transport(folder, {**job, "stage": "validation"}, config, [1024])
            with self.assertRaisesRegex(ValueError, "Missing per-size"):
                baseline.check_transport(folder, job, config, [1024, 2048])
            bad = folder / "test.nccl.host.7.log"
            bad.write_text("Channel 00 : 0 -> 1 via SHM/direct/direct\n")
            with self.assertRaisesRegex(ValueError, "different transport"):
                baseline.check_transport(folder, job, config, [1024])
            bad.unlink()
            with self.assertRaisesRegex(ValueError, "rank logs"):
                baseline.check_transport(folder, job, config, [1024])


if __name__ == "__main__":
    unittest.main()

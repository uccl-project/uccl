"""Performance claims must reject bad ranks, wrong traffic, and changed epochs."""
import copy
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pro6000_ag_rs import parse, sha
from summarize_pro6000_ag_rs import audit, select_policy, SIZES, write_json


class EvidenceChecks(unittest.TestCase):
    def log(self):
        rows = ["# backend=NCCL op=allgather ranks=8 dtype=FP32 out-of-place total-bytes",
                "RESULT 1024 8.000000 0.112000 0"]
        rows += [f"RANK 1024 {r} {r + 1:.6f} 0" for r in range(8)]
        rows += [f"AFFINITY rank={r} tid={100+r} before_cpus=192 matched_cpus=4" for r in range(8)]
        return "\n".join(rows)

    def test_require_complete_correct_slowest_rank(self):
        self.assertEqual(parse(self.log(), [1024], 8), {1024: 8.0})
        for bad in (
            self.log().replace("RANK 1024 7 8.000000 0", ""),
            self.log().replace("RANK 1024 3 4.000000 0", "RANK 1024 3 4.000000 1"),
            self.log().replace("RANK 1024 3 4.000000 0", "RANK 1024 3 nan 0"),
            self.log().replace("RESULT 1024 8.000000 0.112000 0", "RESULT 1024 4.000000 0.224000 0"),
            self.log() + "\nRANK 1024 0 1.000000 0",
            self.log().replace("AFFINITY rank=7", "AFFINITY rank=8"),
        ):
            with self.assertRaises(ValueError):
                parse(bad, [1024], 8)

    def test_reject_allreduce_bandwidth_factor(self):
        with self.assertRaises(ValueError):
            parse(self.log().replace("0.112000", "0.224000"), [1024], 8)

    def fixture(self, root):
        source = root / "source/kernel.cu"
        source.parent.mkdir()
        source.write_text("source snapshot\n")
        job = {"name": "case", "backend": "nccl", "op": "allgather", "args": {"bytes": 1024},
               "env": {"NCCL_P2P_USE_CUDA_MEMCPY": "1", "NCCL_SHM_DISABLE": "1"}}
        manifest = {"ranks": 8, "identities": {"kernel.cu": sha(source)}, "jobs": [job]}
        (root / "case.log").write_text(self.log())
        record = {"job": job, "returncode": 0, "rows": {"1024": 8.0}, "env": {}, "log_sha256": sha(root / "case.log")}
        for path, data in (("manifest.json", manifest), ("results.json", [record]), ("case.json", record),
                           ("completed.json", {"identities": manifest["identities"], "jobs": 1})):
            (root / path).write_text(json.dumps(data))
        for rank in range(8):
            text = "Channel 00/0 : 0 -> 1 via P2P/CUMEM/CE\n"
            if rank == 0:
                text += "AllGather: 128 Bytes -> Algo RING proto SIMPLE\n"
            (root / f"case.nccl.host.{rank}.log").write_text(text)

    def test_transport_scope_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertEqual(audit(root)[2]["nccl_rank_logs"], 8)
            path = root / "case.nccl.host.7.log"
            original = path.read_text()
            for broken in (original.replace("P2P/CUMEM/CE", "SHM/direct/direct"),
                           original.replace("/CE", "")):
                path.write_text(broken)
                with self.assertRaises(ValueError):
                    audit(root)
            path.write_text(original)
            (root / "case.nccl.host.0.log").write_text(original + "AllGather: 1024 Bytes -> Algo RING proto SIMPLE\n")
            with self.assertRaises(ValueError):
                audit(root)  # NCCL dispatch bytes are per shard, not full tensor.
            (root / "case.nccl.host.0.log").write_text(original + "AllGather: 128 Bytes -> Algo RING proto SIMPLE\n")
            (root / "source/kernel.cu").write_text("changed source\n")
            with self.assertRaises(ValueError):
                audit(root)

    def test_initialization_only_requires_matching_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace"
            trace.mkdir()
            self.fixture(trace)
            manifest = json.loads((trace / "manifest.json").read_text())
            manifest.update(libraries={}, hardware="fixture", topology="fixture", base_env={}, cpu_topology="fixture")
            (trace / "manifest.json").write_text(json.dumps(manifest))
            record = json.loads((trace / "case.json").read_text())
            record.update(command=["binary", "--bytes", "1024"], env={"NCCL_DEBUG": "INFO",
                          "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,TUNING", "NCCL_DEBUG_FILE": "trace"})
            (trace / "case.json").write_text(json.dumps(record))
            (trace / "results.json").write_text(json.dumps([record]))
            quiet = Path(directory) / "quiet"
            shutil.copytree(trace, quiet)
            tool = quiet / "control.py"
            tool.write_text("control source\n")
            manifest.update(logging="initialization-only", diagnostic_validation_sha256=sha(trace / "results.json"),
                            control_tool=tool.name, control_tool_sha256=sha(tool))
            (quiet / "manifest.json").write_text(json.dumps(manifest))
            record["env"]["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,GRAPH"
            record["env"]["NCCL_DEBUG_FILE"] = "quiet"
            (quiet / "case.json").write_text(json.dumps(record))
            (quiet / "results.json").write_text(json.dumps([record]))
            with self.assertRaises(ValueError):
                audit(quiet, trace)  # Still contains per-operation dispatch messages.
            (quiet / "case.nccl.host.0.log").write_text("Channel 00/0 : 0 -> 1 via P2P/CUMEM/CE\n")
            self.assertEqual(audit(quiet, trace)[2]["nccl_rank_logs"], 8)
            with self.assertRaises(ValueError):
                audit(quiet)  # Cannot infer the protocol from a silent timing log.
            record["env"]["NCCL_NTHREADS"] = "128"
            (quiet / "case.json").write_text(json.dumps(record))
            (quiet / "results.json").write_text(json.dumps([record]))
            with self.assertRaises(ValueError):
                audit(quiet, trace)  # Only logging may change.

    def calibration(self):
        records = []
        for op in ("allgather", "reducescatter"):
            for backend in ("ours", "nccl"):
                for repeat in range(3):
                    job = {"stage": "calibration", "candidate": op + backend, "backend": backend,
                           "op": op, "round": repeat, "args": {"seed": repeat}}
                    records.append({"job": job, "rows": {str(s): 8.0 for s in SIZES}})
        return records

    def test_reject_incomplete_or_mixed_calibration(self):
        baseline = self.calibration()
        self.assertEqual(len(select_policy({"ranks": 8}, baseline, 3)[1]["allgather"]["ours"]), 21)
        mutations = []
        mutations.append(baseline[:-1])
        bad = copy.deepcopy(baseline)
        bad[0]["job"]["args"]["blocks"] = 32
        mutations.append(bad)
        bad = copy.deepcopy(baseline)
        bad[0]["rows"].pop(str(1 << 30))
        mutations.append(bad)
        bad = copy.deepcopy(baseline)
        bad[0]["job"]["stage"] = "validation"
        mutations.append(bad)
        for records in mutations:
            with self.assertRaises(ValueError):
                select_policy({"ranks": 8}, records, 3)

    def test_refuse_overwriting_frozen_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            write_json(path, {"frozen": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json(path, {"frozen": False})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

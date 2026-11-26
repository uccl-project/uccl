"""Fail-closed evidence checks: a bad rank or non-CE edge cannot become a win."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pro6000_allreduce import parse, placement_audit, transport


class EvidenceChecks(unittest.TestCase):
    def log(self):
        rows = ["RESULT 1024 8.000000 0.224000 0"]
        rows += [f"RANK 1024 {r} {r + 1:.6f} 0" for r in range(8)]
        return "\n".join(rows)

    def test_slowest_rank(self):
        self.assertEqual(parse(self.log(), [1024]), {1024: 8.0})
        with self.assertRaises(ValueError):
            parse(
                self.log().replace(
                    "RESULT 1024 8.000000 0.224000", "RESULT 1024 4.000000 0.448000"
                ),
                [1024],
            )

    def test_reject_bad_or_missing_rank(self):
        for log in (
            self.log().replace("RANK 1024 7 8.000000 0", ""),
            self.log().replace("RANK 1024 3 4.000000 0", "RANK 1024 3 4.000000 1"),
            self.log().replace("RANK 1024 3 4.000000 0", "RANK 1024 3 nan 0"),
            self.log() + "\nRANK 1024 0 1.000000 0",
        ):
            with self.assertRaises(ValueError):
                parse(log, [1024])
        with self.assertRaises(ValueError):
            parse(self.log(), [1024, 2048])

    def test_reject_original_ce_reuse_failure(self):
        # The first validation epoch returned a nonzero combined bad-check count.
        log = self.log().replace(
            "RESULT 1024 8.000000 0.224000 0", "RESULT 1024 8.000000 0.224000 8128"
        )
        with self.assertRaises(ValueError):
            parse(log, [1024])

    def test_actual_transport_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            line = "Channel 00/0 : 0 -> 1 via P2P/CUMEM/CE\nAllReduce: 1024 Bytes -> Algo RING proto SIMPLE\n"
            for rank in range(8):
                (root / f"case.{rank}.log").write_text(line)
            self.assertEqual(transport(root, "case", [1024])["rank_logs"], 8)
            (root / "case.3.log").write_text(line.replace("/CE", ""))
            with self.assertRaises(ValueError):
                transport(root, "case", [1024])
            (root / "case.3.log").write_text(line.replace("SIMPLE", "LL"))
            with self.assertRaises(ValueError):
                transport(root, "case", [1024])
            (root / "case.3.log").write_text(line)
            with self.assertRaises(ValueError):
                transport(root, "case", [1024, 2048])

    def test_require_matched_threads_on_every_rank(self):
        rows = [
            f"AFFINITY rank={r} tid={100+r} before_cpus=192 matched_cpus=4"
            for r in range(8)
        ]
        log = "\n".join(rows)
        self.assertEqual(placement_audit(log, 4)["matched_cpu_count"], 4)
        for bad in (
            "\n".join(rows[:-1]),
            log.replace("matched_cpus=4", "matched_cpus=192"),
            log + "\n" + rows[0],
        ):
            with self.assertRaises(ValueError):
                placement_audit(bad, 4)


if __name__ == "__main__":
    unittest.main()

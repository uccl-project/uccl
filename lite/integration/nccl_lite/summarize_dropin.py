"""Check complete serving ABBA trials and combine them with stock training."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-dir", type=Path, required=True)
    p.add_argument("--serving-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.serving_dir / "manifest.json").read_text())
    assert len(manifest["phases"]) == 4 and all(r["complete"] for r in manifest["phases"])
    rows = [json.loads(x) for x in (args.serving_dir / "samples.jsonl").read_text().splitlines()]
    result = {"training": json.loads((args.training_dir / "summary.json").read_text()), "serving": {}}
    for case in sorted({r["case"] for r in rows}):
        samples = [r for r in rows if r["case"] == case]
        assert len(samples) == 12, (case, len(samples))
        assert len({r["prompt_sha256"] for r in samples}) == 1, "prompt mismatch"
        metrics = {}
        for metric in ("whole_batch_s", "last_ttft_s", "request_median_ttft_s", "request_median_tpot_ms"):
            if any(r.get(metric) is None for r in samples):
                continue
            values = {}
            for backend in ("nccl", "dropin"):
                jobs = {}
                for run_id in sorted({r["run_id"] for r in samples if r["backend"] == backend}):
                    repeats = [r[metric] for r in samples if r["run_id"] == run_id]
                    assert len(repeats) == 3 and all(x > 0 for x in repeats)
                    jobs[run_id] = statistics.median(repeats)
                assert len(jobs) == 2
                values[backend] = {"job_medians": jobs, "median": statistics.median(jobs.values())}
            values["latency_change_pct"] = 100 * (values["dropin"]["median"] / values["nccl"]["median"] - 1)
            values["no_regression_5pct"] = values["latency_change_pct"] <= 5
            values["all_job_medians_within_5pct"] = max(values["dropin"]["job_medians"].values()) <= 1.05 * min(values["nccl"]["job_medians"].values())
            metrics[metric] = values
        result["serving"][case] = {
            "metrics": metrics,
            "identical_output_text": len({r["output_sha256"] for r in samples}) == 1,
            "output_hashes": sorted({r["output_sha256"] for r in samples}),
            "output_hashes_by_backend": {
                backend: sorted({r["output_sha256"] for r in samples if r["backend"] == backend})
                for backend in ("nccl", "dropin")},
        }
    result["no_regression_5pct"] = all(x["no_regression_5pct"] for x in result["training"].values()) and all(
        m["no_regression_5pct"] for case in result["serving"].values() for m in case["metrics"].values())
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"training": {k: v["latency_change_pct"] for k, v in result["training"].items()},
                      "serving": {k: v["metrics"]["whole_batch_s"]["latency_change_pct"] for k, v in result["serving"].items()},
                      "no_regression_5pct": result["no_regression_5pct"]}, indent=2))
    if not result["no_regression_5pct"]:
        raise SystemExit("A measured metric regressed by more than 5%; do not promote this result")


if __name__ == "__main__":
    main()

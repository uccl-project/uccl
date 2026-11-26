#!/usr/bin/env python3
"""Validate complete paired trials and summarize medians without Torch or GPUs."""
import argparse
import json
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    trials = manifest["trials"]
    require(len(trials) == 8 and all(t["complete"] and t["returncode"] == 0 for t in trials),
            "expected eight completed ABBA serving/training jobs")
    records = {t["label"]: json.loads((args.data / (t["label"] + ".json")).read_text()) for t in trials}
    for phase, backend in enumerate(["nccl", "peer", "peer", "nccl"]):
        for kind in ("serving", "training"):
            require(f"{phase:02d}_{backend}_{kind}" in records, "missing ABBA trial")
    reference = next(iter(records.values()))
    for name, record in records.items():
        for field in ("torch", "runtime_nccl_version", "gpu", "checkpoint_config", "trainable_parameters",
                      "complete_block_bytes", "compute"):
            require(record[field] == reference[field], f"{name}: {field} changed")
        require(record["collective_oracle"] == "PASS", "collective oracle failed")
        require(record["reference_check"]["four_token_kv_decode_equal"], "reference decode failed")
        require(record["reference_check"]["max_logit_abs_diff"] == 0 and
                record["reference_check"]["greedy_tokens_equal"], "reference forward differs")
        require(not record["arguments"]["compile_blocks"], "formal result must use eager execution")
        require(len(record["mapped_nccl"]) == 1, "multiple real NCCL libraries mapped")
        for size in record["complete_block_bytes"]:
            require(4 * 2**20 <= size <= 64 * 2**20, "block message outside target interval")
        expected_backend = name.split("_")[1]
        require(record["arguments"]["backend"] == expected_backend, "backend label mismatch")
    summary = {"hardware": reference["gpu"], "torch": reference["torch"],
               "runtime_nccl_version": reference["runtime_nccl_version"],
               "trainable_parameters": reference["trainable_parameters"],
               "complete_block_bytes": reference["complete_block_bytes"],
               "compute": reference["compute"], "serving": {}, "training": {}}
    for kind in ("serving", "training"):
        selected = [(name, d) for name, d in records.items() if name.endswith(kind)]
        cases = [row["case"] for row in selected[0][1]["rows"]]
        require(all([r["case"] for r in d["rows"]] == cases for _, d in selected), "case lists changed")
        for case in cases:
            rows = {name: next(r for r in d["rows"] if r["case"] == case) for name, d in selected}
            config = selected[0][1]["arguments"]
            for name, data in selected:
                for field in ("warmup", "repeats", "seed", "compile_blocks"):
                    require(data["arguments"][field] == config[field], f"{field} changed")
                row = rows[name]
                samples = row["samples_s"]
                require(len(samples) == config["repeats"] and min(samples) > 0, "bad sample count")
                require(statistics.median(samples) == row["median_s"], "median differs from raw data")
                histogram = data["histogram_including_warmups"]
                multiplier = data["arguments"]["warmup"] + data["arguments"]["repeats"]
                layers = data["checkpoint_config"]["n_layer"]
                forwards = sum(c[2] for c in data["arguments"]["cases"]) if kind == "serving" else len(data["rows"])
                expected_ag = multiplier * layers * forwards
                expected_rs = 0 if kind == "serving" else expected_ag
                require(sum(v for k, v in histogram.items() if k.startswith("AG:")) == expected_ag,
                        "AG count differs from model schedule")
                require(sum(v for k, v in histogram.items() if k.startswith("RS:")) == expected_rs,
                        "RS count differs from model schedule")
                require(all(int(k.split(":")[1]) in data["complete_block_bytes"] for k in histogram),
                        "unexpected application message size")
                require(sum(v for k, v in histogram.items() if k.startswith("AG:")) > 0, "missing AG")
                if kind == "training":
                    require(sum(v for k, v in histogram.items() if k.startswith("RS:")) > 0, "missing RS")
            result = {"samples_per_trial": config["repeats"], "warmups_per_case": config["warmup"]}
            for backend in ("nccl", "peer"):
                medians = [r["median_s"] for name, r in rows.items() if f"_{backend}_" in name]
                require(len(medians) == 2, "expected two independent trials per backend")
                result[backend + "_trial_medians_s"] = medians
                result[backend + "_s"] = statistics.median(medians)
            result["latency_reduction_pct"] = 100 * (1 - result["peer_s"] / result["nccl_s"])
            result["throughput_ratio"] = result["nccl_s"] / result["peer_s"]
            values = list(rows.values())
            if kind == "serving":
                require(len({r["prompt_sha256"] for r in values}) == 1, "prompts changed")
                hashes = {h for r in values for h in r["response_sha256"]}
                require(len(hashes) == 1, "generated output changed across repeats/backends")
                require(all(len(r["response_sha256"]) == config["repeats"] for r in values), "missing responses")
                result["all_generated_tokens_identical"] = True
            else:
                require(all(r["parameter_sha256"] == values[0]["parameter_sha256"] for r in values),
                        "full trained parameter hashes differ across trials/backends")
                require(all(r["losses"] == values[0]["losses"] for r in values), "training losses differ")
                result["all_trainable_parameters_bitwise_identical"] = True
                result["all_losses_identical"] = True
            summary[kind][case] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

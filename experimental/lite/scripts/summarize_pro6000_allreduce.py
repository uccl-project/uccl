#!/usr/bin/env python3
"""Audit frozen policies and every final rank, then report every size and loss."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import tarfile

from run_pro6000_allreduce import SIZES, parse, sha, transport


if not __debug__:
    raise RuntimeError("Run the evidence audit without Python -O")


def human(size):
    if size >= 1 << 30:
        return f"{size >> 30} GiB"
    if size >= 1 << 20:
        return f"{size >> 20} MiB"
    return f"{size >> 10} KiB"


def audit(root, output):
    cal = root / "calibration"
    val = root / "validation"
    frozen = json.loads((val / "frozen.json").read_text())
    start = json.loads((val / "validation-start.json").read_text())
    complete = json.loads((val / "validation-complete.json").read_text())
    identity = sha(val / "frozen.json")
    assert identity == start["frozen_sha256"] == complete["frozen_sha256"]
    assert frozen["created"] <= start["started"] <= complete["finished"]
    assert frozen["platform"]["dtype"] == "FP32"
    assert frozen["platform"]["operation"] == "in-place SUM"
    assert frozen["platform"]["warmups"] == 10
    assert frozen["platform"]["continuous_calls"] == 20
    assert (
        hashlib.sha256(
            json.dumps(frozen["platform"], sort_keys=True).encode()
        ).hexdigest()
        == frozen["platform_sha256"]
    )
    for name, digest in frozen["calibration_hashes"].items():
        assert sha(cal / name) == digest, name
    for relative, digest in frozen["identities"].items():
        if relative.startswith("build/"):
            continue
        assert sha(cal / "measured_source" / relative) == digest
        assert sha(val / "measured_source" / relative) == digest
    runs = {}
    counts = dict(
        calibration_pass=0,
        calibration_fail=0,
        validation_pass=0,
        output_checks=0,
        rank_checks=0,
        nccl_rank_logs=0,
    )
    chronology = []
    for directory, phase in ((cal, "calibrate"), (val, "validation")):
        for path in sorted(directory.glob(phase + "-*.json")):
            if path.name in ("validation-start.json", "validation-complete.json"):
                continue
            rec = json.loads(path.read_text())
            log = directory / (rec["label"] + ".txt")
            assert sha(log) == rec["log_sha256"], log
            assert rec["identities"] == frozen["identities"]
            assert rec["platform_sha256"] == frozen["platform_sha256"]
            assert rec["environment"]["UCCL_LITE_PLACEMENT"] == frozen["placement"]
            if phase == "calibrate":
                assert rec["finished"] <= frozen["created"]
            else:
                assert rec["started"] >= start["started"]
                assert rec["finished"] <= complete["finished"]
            if rec["status"] != "pass":
                assert (
                    phase == "calibrate"
                ), "Failed final validation cannot be replaced"
                counts["calibration_fail"] += 1
                continue
            sizes = [s for s in SIZES if rec["begin"] <= s <= rec["end"]]
            values = parse(log.read_text(), sizes)
            assert {str(k): v for k, v in values.items()} == rec["values"]
            assert "warmups=10 continuous=20" in log.read_text()
            if rec["backend"] == "nccl":
                actual = transport(directory / "debug", rec["label"], sizes)
                assert actual["hashes"] == rec["transport"]["hashes"]
                assert json.loads(json.dumps(actual)) == rec["transport"]
                counts["nccl_rank_logs"] += actual["rank_logs"]
            if phase == "calibrate":
                counts["calibration_pass"] += 1
            else:
                counts["validation_pass"] += 1
                # Nine independently checked calls and the complete 30-call batch.
                counts["output_checks"] += 10 * len(sizes)
                counts["rank_checks"] += 80 * len(sizes)
                chronology.append(rec)
            runs[rec["label"]] = rec
    chronology.sort(key=lambda r: r["started"])
    assert all(
        a["finished"] <= b["started"] for a, b in zip(chronology, chronology[1:])
    )
    rows = []
    for size in SIZES:
        row = dict(bytes=size, size=human(size))
        for backend in ("ours", "nccl"):
            name = frozen["policy"][backend][str(size)]
            selected = [runs[f"calibrate-{i}-{backend}-{name}"] for i in range(1, 4)]
            candidates = []
            configs = frozen[
                "ours_candidates" if backend == "ours" else "nccl_candidates"
            ]
            for candidate in configs:
                records = [
                    runs.get(f"calibrate-{i}-{backend}-{candidate}")
                    for i in range(1, 4)
                ]
                if all(r and str(size) in r["values"] for r in records):
                    candidates.append(
                        (median(r["values"][str(size)] for r in records), candidate)
                    )
            # Reconstruct the frozen choice from calibration only, preserving ties.
            best = min(candidates, key=lambda pair: pair[0])
            assert name == best[1], (size, backend, name, best)
            final = [
                runs[f"validation-{i}-{backend}-{name}"]["values"][str(size)]
                for i in range(1, 6)
            ]
            row[backend + "_config"] = name
            row[backend + "_calibration_us"] = median(
                r["values"][str(size)] for r in selected
            )
            row[backend + "_median_us"] = median(final)
            row[backend + "_min_us"] = min(final)
            row[backend + "_max_us"] = max(final)
            row[backend + "_bus_gbs"] = size * 1.75 / (median(final) * 1000)
            for i, value in enumerate(final, 1):
                row[f"{backend}_run{i}_us"] = value
        ratios = [row[f"nccl_run{i}_us"] / row[f"ours_run{i}_us"] for i in range(1, 6)]
        row["median_speedup"] = row["nccl_median_us"] / row["ours_median_us"]
        row["paired_wins"] = sum(r > 1 for r in ratios)
        row["min_paired_speedup"] = min(ratios)
        row["max_paired_speedup"] = max(ratios)
        row["separated_range_win"] = row["ours_max_us"] < row["nccl_min_us"]
        rows.append(row)
    expected = {
        f"validation-{i}-{b}-{n}"
        for i in range(1, 6)
        for b in ("ours", "nccl")
        for n in set(frozen["policy"][b].values())
    }
    assert {r["label"] for r in chronology} == expected
    # Verify backend order reversal at the start of each independent round.
    for i in range(1, 6):
        records = [r for r in chronology if r["label"].startswith(f"validation-{i}-")]
        assert records[0]["backend"] == ("ours" if i % 2 else "nccl")
    failed_epochs = []
    previous = root / "failed-epoch1"
    if previous.exists():
        incident = json.loads((previous / "incident.json").read_text())
        old_frozen = json.loads((previous / "validation" / "frozen.json").read_text())
        assert old_frozen["identities"] != frozen["identities"]
        for name, digest in old_frozen["calibration_hashes"].items():
            assert sha(previous / "calibration" / name) == digest
        failures = []
        for path in (previous / "validation").glob("validation-*.json"):
            if path.name in ("validation-start.json", "validation-complete.json"):
                continue
            record = json.loads(path.read_text())
            log = path.with_suffix(".txt")
            assert sha(log) == record["log_sha256"]
            assert record["identities"] == old_frozen["identities"]
            if record["status"] == "fail":
                bad_rows = [
                    line
                    for line in log.read_text().splitlines()
                    if line.startswith("RESULT ") and int(line.split()[-1])
                ]
                failures.append(
                    dict(
                        label=record["label"],
                        reason=record["reason"],
                        bad_rows=bad_rows,
                    )
                )
            else:
                parse(
                    log.read_text(),
                    [s for s in SIZES if record["begin"] <= s <= record["end"]],
                )
        assert failures, "The retained failed epoch must not be presented as successful"
        failed_epochs.append(
            dict(directory="failed-epoch1", incident=incident, failures=failures)
        )
    profiles = []
    profile_root = root / "exploration" / "profile-final"
    for path in sorted(profile_root.glob("profile-*.json")):
        record = json.loads(path.read_text())
        log = profile_root / (record["label"] + ".txt")
        assert (
            record["status"] == "pass" and record["identities"] == frozen["identities"]
        )
        assert sha(log) == record["log_sha256"]
        values = parse(log.read_text(), [1 << 30])
        counters = []
        for line in log.read_text().splitlines():
            if line.startswith("PROFILE "):
                counters.append(dict(field.split("=") for field in line.split()[1:]))
        assert {(int(r["rank"]), int(r["phase"])) for r in counters} == {
            (rank, phase) for rank in range(8) for phase in range(6)
        }
        local = sum(int(r["local_cycles"]) for r in counters)
        transfer = sum(int(r["transfer_ack_cycles"]) for r in counters)
        profiles.append(
            dict(
                config=record["config"],
                profiled_us=values[1 << 30],
                local_phase_cycles=local,
                transfer_phase_cycles=transfer,
                local_phase_percent=100 * local / (local + transfer),
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    if profiles:
        with (output / "profiles.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(profiles[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(profiles)
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    losses = [r["size"] for r in rows if r["paired_wins"] != 5]
    overlapping = [r["size"] for r in rows if not r["separated_range_win"]]
    conclusion = dict(
        frozen_sha256=identity,
        counts=counts,
        retained_failed_epochs=failed_epochs,
        median_wins=sum(r["median_speedup"] > 1 for r in rows),
        five_of_five_wins=21 - len(losses),
        separated_range_wins=21 - len(overlapping),
        remaining_losing_or_inconsistent_sizes=losses,
        nonseparated_sizes=overlapping,
        all_sizes_reproducibly_faster=not losses and not overlapping,
    )
    (output / "summary.json").write_text(json.dumps(conclusion, indent=2) + "\n")
    lines = [
        "# Frozen-policy FP32/C++ AllReduce validation",
        "",
        "Eight RTX PRO 6000 GPUs; in-place SUM; 10 warmups then 20 continuous calls; "
        "host timing through final stream synchronization; slowest rank. Centers are medians "
        "of five independent run means. Per-size choices come only from three-run calibration.",
        "",
        "| Size | UCCL Lite (µs) | NCCL CE (µs) | Speedup | Paired wins | UCCL worst < NCCL best |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['ours_median_us']:.2f} | {r['nccl_median_us']:.2f} | "
            f"{r['median_speedup']:.3f}× | {r['paired_wins']}/5 | "
            f"{'yes' if r['separated_range_win'] else 'no'} |"
        )
    lines += [
        "",
        "Remaining sizes without five paired wins: "
        + (", ".join(losses) or "none")
        + ".",
        "",
        "Observed ranges are not confidence intervals. No final-run or final-variant selection is used.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    print(json.dumps(conclusion, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    if root.is_file():
        checksums = root.parent / "checksums.json"
        if checksums.exists():
            assert sha(root) == json.loads(checksums.read_text())[root.name]
        extracted = args.output / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(root) as archive:
            for member in archive.getmembers():
                destination = (extracted / member.name).resolve()
                if not destination.is_relative_to(extracted.resolve()) or not (
                    member.isfile() or member.isdir()
                ):
                    raise ValueError("Unsafe archive member")
            archive.extractall(extracted)
        root = extracted
    audit(root, args.output)


if __name__ == "__main__":
    main()

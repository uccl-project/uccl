"""Validate and archive the frozen four-L4 NCCL CE/SM measurement schedule."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent
OUT = ROOT / "build/nccl_tuned_4l4_20261115"


def write_csv(name, rows):
    with (DATA / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    calibration_hashes = json.loads((OUT / "calibrate_hashes.json").read_text())
    evaluation_hashes = json.loads((OUT / "evaluate_hashes.json").read_text())
    assert calibration_hashes == evaluation_hashes, "measured binaries changed"
    assert calibration_hashes == json.loads((OUT / "confirm_hashes.json").read_text())
    for name, digest in calibration_hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    schedule = json.loads((OUT / "schedule.json").read_text())
    assert len(schedule) == 126
    final_schedule = json.loads((OUT / "final_schedule.json").read_text())
    final_by_key = {(r["operation"], r["bytes"], r["family"]): r for r in final_schedule}
    assert len(final_by_key) == 126
    records = []
    transports = []
    samples = []
    for path in sorted(OUT.glob("*.json")):
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or "returncode" not in record:
            continue
        records.append(record)
        if record["returncode"] != 0:
            continue
        name = record["name"]
        debug = "\n".join(p.read_text() for p in OUT.glob(name + ".debug.*.log"))
        channels = [line for line in debug.splitlines() if " via " in line and "Channel " in line]
        assert channels, name
        paths = collections.Counter(line.split(" via ")[1].strip() for line in channels)
        if record["config"].startswith("ce_"):
            assert all("P2P/" in p and "/CE" in p for p in paths), (name, paths)
            assert record["environment"]["NCCL_PROTO"] == "Simple"
        else:
            assert all("/CE" not in p for p in paths), (name, paths)
        transports.append(dict(run=name, paths=dict(paths)))
        counts = collections.Counter()
        values = collections.defaultdict(list)
        for line in (OUT / (name + ".csv")).read_text().splitlines():
            fields = line.split(",")
            if record["operation"] == "allreduce" and line.startswith("nccl,"):
                assert int(fields[5]) == 0, name
                if fields[1] != "timed":
                    continue
                size, call, us = int(fields[3]), int(fields[2]), float(fields[4])
            elif record["operation"] != "allreduce" and line.startswith("# sample,"):
                assert fields[3] == "nccl"
                size, call, us = int(fields[2]), int(fields[5]), float(fields[6])
            else:
                continue
            counts[size] += 1
            values[size].append(us)
            samples.append(dict(run=name, stage=record["stage"], configuration=record["config"],
                                operation=record["operation"], repeat=record["repeat"],
                                bytes=size, call=call, latency_us=us))
        assert dict(counts) == {1 << p: 10 for p in range(10, 31)}, name
        for size, latencies in values.items():
            assert statistics.mean(latencies) == record["means_us"][str(size)]
    cases = {}
    initial_cases = {}
    selection = []
    for selected in schedule:
        op, size, family = selected["operation"], selected["bytes"], selected["family"]
        scores = {}
        for candidate in json.loads((OUT / "candidates.json").read_text()):
            if not candidate.startswith(family + "_"):
                continue
            trials = [r for r in records if r["stage"] == "calibrate" and
                      r["operation"] == op and r["config"] == candidate]
            if len(trials) == 2 and all(r["returncode"] == 0 for r in trials):
                scores[candidate] = statistics.median(r["means_us"][str(size)] for r in trials)
        eligible = dict(scores)
        if family == "sm":
            controls = {r["repeat"]: r["means_us"][str(size)] for r in records
                        if r["stage"] == "calibrate" and r["operation"] == op and r["config"] == "sm_default"}
            for candidate, score in scores.items():
                if candidate == "sm_default":
                    continue
                trials = [r for r in records if r["stage"] == "calibrate" and
                          r["operation"] == op and r["config"] == candidate]
                if (score > scores["sm_default"] * 0.97 or
                    not all(r["means_us"][str(size)] < controls[r["repeat"]] for r in trials)):
                    del eligible[candidate]
        assert selected["config"] == min(eligible, key=eligible.get)
        assert selected["calibration_us"] == scores[selected["config"]]
        initial_runs = [r for r in records if r["stage"] == "evaluate" and
                        r["operation"] == op and r["config"] == selected["config"]]
        assert len(initial_runs) == 3 and all(r["returncode"] == 0 for r in initial_runs)
        initial_us = [r["means_us"][str(size)] for r in initial_runs]
        initial = initial_cases.setdefault((op, size), dict(operation=op, bytes=size))
        initial.update({family + "_configuration": selected["config"],
                        family + "_median_us": statistics.median(initial_us),
                        family + "_min_us": min(initial_us), family + "_max_us": max(initial_us)})
        final = final_by_key[op, size, family]
        expected_config = selected["config"]
        if family == "sm" and expected_config != "sm_default":
            own = sorted([r for r in records if r["stage"] == "evaluate" and
                          r["operation"] == op and r["config"] == expected_config], key=lambda r: r["repeat"])
            controls = sorted([r for r in records if r["stage"] == "evaluate" and
                               r["operation"] == op and r["config"] == "sm_default"], key=lambda r: r["repeat"])
            assert len(own) == len(controls) == 3
            a = [r["means_us"][str(size)] for r in own]
            b = [r["means_us"][str(size)] for r in controls]
            if statistics.median(a) > statistics.median(b) * .97 or not all(x < y for x, y in zip(a, b)):
                expected_config = "sm_default"
        assert final["config"] == expected_config
        stage = "confirm" if family == "sm" and expected_config == "sm_default" else "evaluate"
        matches = [r for r in records if r["stage"] == stage and
                   r["operation"] == op and r["config"] == expected_config]
        assert len(matches) == 3 and {r["repeat"] for r in matches} == {0, 1, 2}
        assert all(r["returncode"] == 0 for r in matches)
        us = [r["means_us"][str(size)] for r in sorted(matches, key=lambda r: r["repeat"])]
        row = cases.setdefault((op, size), dict(operation=op, bytes=size))
        row.update({family + "_configuration": expected_config,
                    family + "_median_us": statistics.median(us),
                    family + "_min_us": min(us), family + "_max_us": max(us)})
        selection.append(dict(**final, measurement_stage=stage, run1_us=us[0], run2_us=us[1], run3_us=us[2]))
    for (op, size), row in cases.items():
        controls = [r for r in records if r["stage"] == "confirm" and
                    r["operation"] == op and r["config"] == "sm_default"]
        assert len(controls) == 3 and all(r["returncode"] == 0 for r in controls)
        us = [r["means_us"][str(size)] for r in controls]
        row.update(default_median_us=statistics.median(us), default_min_us=min(us), default_max_us=max(us))
        prior = [r["means_us"][str(size)] for r in records if r["stage"] == "evaluate" and
                 r["operation"] == op and r["config"] == "sm_default"]
        initial_cases[op, size].update(default_median_us=statistics.median(prior),
                                      default_min_us=min(prior), default_max_us=max(prior))
    write_csv("initial_cases.csv", [initial_cases[key] for key in sorted(initial_cases)])
    write_csv("cases.csv", [cases[key] for key in sorted(cases)])
    write_csv("selected_runs.csv", selection)
    write_csv("samples.csv", samples)
    calibration = []
    for r in records:
        if r["stage"] != "calibrate" or r["returncode"] != 0:
            continue
        for size, us in r["means_us"].items():
            calibration.append(dict(configuration=r["config"], operation=r["operation"],
                                    repeat=r["repeat"], bytes=int(size), mean_us=us))
    write_csv("calibration.csv", calibration)
    (DATA / "transport.json").write_text(json.dumps(transports, indent=2) + "\n")
    for name in ("candidates.json", "schedule.json", "final_schedule.json", "environment.json", "calibrate_hashes.json", "evaluate_hashes.json", "confirm_hashes.json"):
        shutil.copyfile(OUT / name, DATA / name)
    shutil.copyfile(ROOT / "src/collectives/peer_ag_rs.cu", OUT / "peer_original.cu")
    shutil.copyfile(ROOT / "benchmarks/collectives/nccl_4l4_mpi_baseline.cu", OUT / "ar_original.cu")
    with tarfile.open(DATA / "measurements.tar.gz", "w:gz") as archive:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.suffix in (".log", ".csv", ".stderr", ".json", ".cu"):
                archive.add(p, arcname=p.name)
    summary = dict(completed_sweeps=len(records),
                   passed_sweeps=sum(r["returncode"] == 0 for r in records),
                   timed_calls=len(samples),
                   calibration_timed_calls=sum(s["stage"] == "calibrate" for s in samples),
                   evaluation_timed_calls=sum(s["stage"] == "evaluate" for s in samples),
                   confirmation_timed_calls=sum(s["stage"] == "confirm" for s in samples),
                   selected_final_timed_calls=126 * 3 * 10,
                   sm_fallback_points=sum(r["config"] != r["calibrated_configuration"] for r in final_schedule),
                   measurement_archive_sha256=hashlib.sha256((DATA / "measurements.tar.gz").read_bytes()).hexdigest())
    (DATA / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build all six overviews exclusively from fresh full-range GPU/SHM sweeps."""
import csv
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent
LITE = DATA.parents[2]
SOURCE = "benchmarks/data/full_refresh_8l4_20261114/summary.csv"
RS_SOURCE = "benchmarks/data/rs_recovery_8l4_20261114/summary.csv"
OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter")


def main():
    with (LITE / SOURCE).open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(mode, op, 1 << power) for mode in ("gpu", "shm")
                for op in OPERATIONS for power in range(10, 31)}
    actual = {(r["testbed"], r["collective"], int(r["bytes"])) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("full refresh requires all 126 unique workload keys")
    validation = json.loads((LITE / RS_SOURCE).with_name("validation.json").read_text())
    if not (validation["all_target_sweeps_beat_previous"] and
            validation["all_target_sweeps_beat_nccl"]):
        raise ValueError("RS recovery has not passed paired confirmation")
    with (LITE / RS_SOURCE).open() as stream:
        rs = list(csv.DictReader(stream))
    rs_keys = {(r["testbed"], r["collective"], int(r["bytes"])) for r in rs}
    if rs_keys != {key for key in expected if key[1] == "reduce_scatter"} or len(rs) != 42:
        raise ValueError("RS recovery requires all 42 unique workload keys")
    rows = [r for r in rows if r["collective"] != "reduce_scatter"] + rs
    result = []
    for row in rows:
        is_rs = row["collective"] == "reduce_scatter"
        if row["status"] != ("fresh-rs-recovery" if is_rs else "fresh-full-sweep"):
            raise ValueError("historical or unconfirmed point in full refresh")
        for field in ("uccl_median_us", "nccl_median_us"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid measured latency")
        result.append(dict(testbed=row["testbed"], collective=row["collective"],
                           bytes=row["bytes"], placement=row["placement"],
                           uccl_median_us=row["uccl_median_us"],
                           nccl_median_us=row["nccl_median_us"],
                           uccl_source=RS_SOURCE if is_rs else SOURCE,
                           nccl_source=RS_SOURCE if is_rs else SOURCE,
                           variant=row["variant"], status=row["status"]))
    with (DATA / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()

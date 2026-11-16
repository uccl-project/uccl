"""Strict CPU-only comparison of two reviewed finite-bits diagnostic runs.

Usage: compare_finite_bits.py INITIAL_RUN C03_RUN OUTPUT_JSON
Exit 0: no observed bit difference for this corpus. Exit 1: bit mismatch or
rank disagreement. Exit 2: invalid/incomplete provenance or inputs. None of
these outcomes establishes model-quality equivalence or changes its policy.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


TEST_SHA256 = "4d7842cfeec03efffa497874a562f8b84be8e90016234b70735069a81f6b2492"
REMOTE_ROOT = "/home/wxzheng/eat-pcieccl-vllm-pro6000-20261115-074910"
EXPECTED_ARGV = [REMOTE_ROOT + "/env/.venv/bin/python",
                 REMOTE_ROOT + "/controller/finite_bitwise_compatibility.py"]
PINNED = {
    "initial": {
        "source_sha256": "aec0856c1657fcce6e3dca1dcf979a472938a182eab239457399a8028c53f364",
        "binary_sha256": "84d27c951f0c4bfccc377eeeffeab046932e0ebfa99a800741f90f0968c7b318",
        "binary_name": "_pcieccl_C.cpython-312-x86_64-linux-gnu.so",
    },
    "c03": {
        "source_sha256": "ce751fa67339d5163171721a350c09d8edc14876a2e89cb808349ba984f75175",
        "binary_sha256": "89ebe0c1de86d93f6d92e709fcc2dbf753b46579e5fd588c8e35ae5bd1dc243b",
        "binary_name": "_pcieccl_C.cpython-312-x86_64-linux-gnu.so",
    },
}
BASE_SEED = 2026111501
SIZES = (128, 6144, 12288, 24576, 32768, 32896, 49152, 65408,
         65536, 65664, 98304, 196608, 393216, 1048576)
INVOCATIONS = (("eager", 0), ("graph", 1), ("graph", 2), ("graph", 3))
KEY_FIELDS = ("bytes", "scalar_offset", "dtype", "rank", "mode", "generation")
EXPECTED_KEYS = {
    (size, offset, dtype, rank, mode, generation)
    for size, offset, dtype, rank, (mode, generation)
    in itertools.product(SIZES, (0, 1), ("bf16", "fp32"), range(8), INVOCATIONS)
}
HEADER_FIELDS = {"type", "protocol", "test_sha256", "source_sha256", "binary_sha256",
                 "python", "torch", "base_seed", "byte_order", "cases_per_rank",
                 "checked_collectives_per_rank", "scored"}
ROW_FIELDS = {"type", "case", "rank", "dtype", "mode", "generation", "bytes",
              "scalar_offset", "pair_seed", "rank_seed", "input_sha256",
              "output_sha256", "input_preserved", "boundary_guard_pass",
              "output_finite", "status"}
COMPLETION_FIELDS = {"type", "rank", "checked_collectives", "scored"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON field: " + key)
        value[key] = item
    return value


def strict_json(text):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=invalid_constant)


def read_json(path):
    value = strict_json(Path(path).read_text(encoding="utf-8"))
    require(type(value) is dict, f"Expected JSON object: {path}")
    return value


def sha_field(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def exact_int(value, expected):
    return type(value) is int and value == expected


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def seed_for(size, dtype, rank, offset, generation, scope):
    text = f"{BASE_SEED}/{size}/{dtype}/{rank}/{offset}/{generation}/{scope}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def key_record(key):
    return dict(zip(KEY_FIELDS, key))


def validate_header(header, snapshot):
    require(set(header) == HEADER_FIELDS, "Unexpected or missing header fields")
    expected = {"type": "finite_bits_header", "protocol": "finite-bits-v1",
                "test_sha256": TEST_SHA256, "python": "3.12.14",
                "torch": "2.11.0+cu130", "byte_order": "little"}
    for key, value in expected.items():
        require(type(header[key]) is str and header[key] == value,
                "Header identity mismatch: " + key)
    for key in ("source_sha256", "binary_sha256"):
        require(header[key] == PINNED[snapshot][key], "Header snapshot mismatch: " + key)
    for key, value in (("base_seed", BASE_SEED), ("cases_per_rank", 56),
                       ("checked_collectives_per_rank", 224)):
        require(exact_int(header[key], value), "Invalid header count/seed: " + key)
    require(header["scored"] is False, "Diagnostic header must be unscored")


def validate_row(row):
    require(set(row) == ROW_FIELDS, "Unexpected or missing result-row fields")
    for field in ("bytes", "scalar_offset", "rank", "generation", "pair_seed", "rank_seed"):
        require(type(row[field]) is int, "Result integer field has wrong type: " + field)
    for field in ("dtype", "mode", "case", "status"):
        require(type(row[field]) is str, "Result string field has wrong type: " + field)
    key = tuple(row[field] for field in KEY_FIELDS)
    require(key in EXPECTED_KEYS, "Unexpected result key: " + repr(key))
    require(row["case"] == f"bytes{row['bytes']}-offset{row['scalar_offset']}",
            "Case label disagrees with result key")
    require(row["status"] == "pass", "Result row did not pass")
    for field in ("input_preserved", "boundary_guard_pass", "output_finite"):
        require(row[field] is True, "Result invariant failed: " + field)
    for field in ("input_sha256", "output_sha256"):
        require(sha_field(row[field]), "Malformed result digest: " + field)
    for field, rank, scope in (("pair_seed", row["rank"] // 2, "pair"),
                               ("rank_seed", row["rank"], "rank")):
        expected = seed_for(row["bytes"], row["dtype"], rank,
                            row["scalar_offset"], row["generation"], scope)
        require(row[field] == expected, "Seed does not match reviewed generator: " + field)
    return key


def load_run(folder, snapshot):
    folder = Path(folder)
    result_path = folder / "result.json"
    command_path = folder / "finite-bits.command.json"
    log_path = folder / "finite-bits.log"
    hashes_before = {p.name: digest(p) for p in (result_path, command_path, log_path)}
    result = read_json(result_path)
    require(result.get("status") == "complete" and result.get("snapshot") == snapshot,
            "Wrapper did not complete the expected snapshot")
    require(result.get("scored") is False and result.get("test") == "finite-bits",
            "Wrong wrapper diagnostic identity")
    require(result.get("test_sha256") == TEST_SHA256, "Wrapper test hash mismatch")
    require(result.get("identity") == PINNED[snapshot], "Wrapper snapshot identity mismatch")
    require(not result.get("error"), "Wrapper reports an error")
    for field in ("started", "finished"):
        require(finite_number(result.get(field)),
                "Invalid wrapper timestamp: " + field)
    require(result["finished"] >= result["started"], "Wrapper timestamps reversed")
    require(not (folder / "external-gpu-work.json").exists(), "External GPU work was recorded")
    command = read_json(command_path)
    require(exact_int(command.get("returncode"), 0), "Diagnostic subprocess failed or incomplete")
    require(exact_int(command.get("timeout_seconds"), 900), "Unexpected supervision timeout")
    argv = command.get("argv")
    require(type(argv) is list and len(argv) == 2 and all(type(x) is str for x in argv),
            "Unexpected diagnostic command")
    require(argv == EXPECTED_ARGV, "Diagnostic interpreter or script path differs from pinned command")
    for field in ("started", "finished"):
        require(finite_number(command.get(field)), "Invalid command timestamp: " + field)
    require(result["started"] <= command["started"] <= command["finished"] <= result["finished"],
            "Command timestamps are reversed or outside wrapper execution")
    rows, completions, headers = {}, {}, []
    nonblank = 0
    with log_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            nonblank += 1
            try:
                require(line.endswith("\n"), "Truncated final log line")
                require(len(line.encode()) <= 4096, "Oversized diagnostic log line")
                value = strict_json(line)
                require(type(value) is dict, "Diagnostic line is not an object")
                kind = value.get("type")
                if kind == "finite_bits_header":
                    validate_header(value, snapshot)
                    headers.append(value)
                    require(len(headers) == 1, "Duplicate diagnostic header")
                elif kind == "finite_bits":
                    key = validate_row(value)
                    require(key not in rows, "Duplicate result key: " + repr(key))
                    rows[key] = value
                elif kind == "finite_bits_complete":
                    require(set(value) == COMPLETION_FIELDS, "Unexpected completion fields")
                    rank = value["rank"]
                    require(type(rank) is int and 0 <= rank < 8, "Invalid completion rank")
                    require(rank not in completions, "Duplicate rank completion")
                    require(exact_int(value["checked_collectives"], 224)
                            and value["scored"] is False, "Invalid completion count/status")
                    completions[rank] = value
                else:
                    raise ValueError("Unknown diagnostic JSON type")
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{log_path.name}:{line_number}: {error}") from error
    require(len(headers) == 1, "Missing diagnostic header")
    missing, extra = EXPECTED_KEYS - rows.keys(), rows.keys() - EXPECTED_KEYS
    require(not missing and not extra,
            f"Incomplete Cartesian row set: missing={len(missing)}, extra={len(extra)}; "
            f"first_missing={list(sorted(missing))[:3]}")
    require(set(completions) == set(range(8)), "Missing rank completion")
    for rank in range(8):
        require(sum(key[3] == rank for key in rows) == 224, "Per-rank result count mismatch")
    require(nonblank == 1801, "Unexpected total diagnostic record count")
    hashes_after = {p.name: digest(p) for p in (result_path, command_path, log_path)}
    require(hashes_before == hashes_after, "Input run artifacts changed while parsing")
    return {"snapshot": snapshot, "folder": str(folder.resolve()), "header": headers[0],
            "artifact_sha256": hashes_after, "rows": rows, "result_rows": len(rows),
            "rank_completions": len(completions)}


def rank_disagreements(run):
    groups = {}
    for key, row in run["rows"].items():
        group = key[:3] + key[4:]
        groups.setdefault(group, {})[key[3]] = row["output_sha256"]
    return [{"case_key": dict(zip(("bytes", "scalar_offset", "dtype", "mode", "generation"), group)),
             "output_sha256_by_rank": hashes}
            for group, hashes in sorted(groups.items()) if len(set(hashes.values())) != 1]


def finish_report(report, code):
    report["validation_error_count"] = len(report["validation_errors"])
    report["input_provenance_mismatch_count"] = len(report["input_provenance_mismatches"])
    report["output_mismatch_count"] = len(report["output_mismatches"])
    report["rank_disagreement_case_counts"] = {
        name: len(items) for name, items in report["rank_disagreements"].items()}
    return report, code


def compare(initial_folder, c03_folder):
    report = {"protocol": "finite-bits-comparison-v1", "scored": False,
              "expected_rows_per_run": 1792, "expected_calls_per_rank": 224,
              "test_sha256": TEST_SHA256, "expected_snapshot_identities": PINNED,
              "expected_command_argv": EXPECTED_ARGV,
              "validation_errors": [], "input_provenance_mismatches": [],
              "output_mismatches": [], "rank_disagreements": {},
              "limitation": "A match means no observed bit difference for this finite corpus. It is not a quality proof, cannot override the GSM8K policy, and can share a common error. A mismatch alone does not identify the wrong scalar or explain a model-quality change."}
    runs = {}
    for snapshot, folder in (("initial", initial_folder), ("c03", c03_folder)):
        try:
            runs[snapshot] = load_run(folder, snapshot)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
            report["validation_errors"].append({"snapshot": snapshot, "error": str(error)})
    report["runs"] = {name: {k: v for k, v in run.items() if k != "rows"}
                      for name, run in runs.items()}
    if report["validation_errors"]:
        report.update(status="invalid_or_incomplete", compared_output_rows=0)
        return finish_report(report, 2)
    first, second = runs["initial"], runs["c03"]
    for name, run in runs.items():
        report["rank_disagreements"][name] = rank_disagreements(run)
    for field in HEADER_FIELDS - {"source_sha256", "binary_sha256"}:
        if first["header"][field] != second["header"][field]:
            report["validation_errors"].append({"field": field, "error": "Header mismatch across runs"})
    for key in sorted(EXPECTED_KEYS):
        a, b = first["rows"][key], second["rows"][key]
        fields = [field for field in ("input_sha256", "pair_seed", "rank_seed") if a[field] != b[field]]
        if fields:
            report["input_provenance_mismatches"].append(
                {"key": key_record(key), "fields": fields,
                 "initial": {field: a[field] for field in fields},
                 "c03": {field: b[field] for field in fields}})
    if report["validation_errors"] or report["input_provenance_mismatches"]:
        report.update(status="invalid_input_provenance", compared_output_rows=0)
        return finish_report(report, 2)
    for key in sorted(EXPECTED_KEYS):
        a, b = first["rows"][key], second["rows"][key]
        if a["output_sha256"] != b["output_sha256"]:
            report["output_mismatches"].append({"key": key_record(key),
                "initial_output_sha256": a["output_sha256"], "c03_output_sha256": b["output_sha256"]})
    mismatch = bool(report["output_mismatches"] or any(report["rank_disagreements"].values()))
    report.update(status="bitwise_mismatch" if mismatch else "no_observed_bit_difference",
                  compared_output_rows=1792,
                  output_mismatch_count=len(report["output_mismatches"]),
                  rank_disagreement_case_counts={name: len(items) for name, items in report["rank_disagreements"].items()})
    return finish_report(report, 1 if mismatch else 0)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.link(temporary, path)  # Atomic creation, refuses an existing output.
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_run", type=Path)
    parser.add_argument("c03_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite comparison output")
    report, code = compare(args.initial_run, args.c03_run)
    report["comparator_sha256"] = digest(__file__)
    write_once(args.output, report)
    print(json.dumps({"status": report["status"], "compared_output_rows": report["compared_output_rows"],
                      "output_mismatch_count": len(report["output_mismatches"]),
                      "validation_error_count": len(report["validation_errors"]), "output": str(args.output)}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()

"""The explicit test inventory and capability/change based CI target determinator."""

import argparse
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TIERS = {"host", "build", "integration", "performance"}
GLOBAL = [
    "ci/**",
    ".github/workflows/**",
    "include/**",
    "build.sh",
    "setup.py",
    "pyproject.toml",
    "requirements*",
    ".gitmodules",
]


def read_catalog(root=ROOT):
    return (
        json.loads((root / "ci/config/tests.json").read_text()),
        json.loads((root / "ci/config/targets.json").read_text()),
    )


def tracked_files(root=ROOT):
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    return sorted(
        p for p in set(output.decode().split("\0")) - {""} if (root / p).exists()
    )


def test_sources(files):
    """Discover candidates, never implicitly opt a new test into privileged CI."""
    for name in files:
        p = Path(name)
        if "thirdparty" in p.parts or p.suffix not in {
            ".py",
            ".cc",
            ".cpp",
            ".cu",
            ".c",
            ".sh",
        }:
            continue
        if (
            re.search(r"(^test_|_test$)", p.stem)
            or "tests" in p.parts
            or "test" in p.parts
            or "benchmarks" in p.parts
            or p.stem.startswith(("benchmark", "bench_"))
            or p.name.startswith(("run_nccl_test", "run_rccl_test", "run-nccl-tests"))
        ):
            yield name


def compatible(test, target):
    return (
        test.get("nodes", 1) <= max(1, len(target["hosts"]))
        and test.get("gpus", 0) <= target["gpus"]
        and set(test.get("requires", [])).issubset(target["capabilities"])
    )


def target_interface(target, host):
    """Resolve by SSH alias, not rank: --hosts may reorder a hardware pool."""
    if "interfaces" in target:
        if host not in target["interfaces"]:
            raise ValueError(f"No network interface configured for host {host!r}")
        return target["interfaces"][host]
    return target["interface"]


def validate(catalog, targets, files=None):
    errors, ids, covered = [], set(), set()
    for name, target in targets.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            errors.append(f"Invalid target ID: {name}")
        if not target["enabled"] and not target.get("reason"):
            errors.append(f"{name}: disabled target needs a reason")
        if "interfaces" in target:
            interfaces = target["interfaces"]
            if not isinstance(interfaces, dict) or set(interfaces) != set(
                target["hosts"]
            ):
                errors.append(f"{name}: interfaces must cover exactly the target hosts")
            elif any(
                not isinstance(value, str) or not value for value in interfaces.values()
            ):
                errors.append(f"{name}: interface names must be nonempty strings")
    for test in catalog["tests"]:
        name = test["id"]
        if name in ids or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name):
            errors.append(f"Duplicate/invalid test ID: {name}")
        ids.add(name)
        if not isinstance(test.get("enabled"), bool):
            errors.append(f"{name}: enabled must be explicit")
        if type(test.get("nodes", 1)) is not int or test.get("nodes", 1) < 1:
            errors.append(f"{name}: nodes must be a positive integer")
        if type(test.get("gpus", 0)) is not int or test.get("gpus", 0) < 0:
            errors.append(f"{name}: gpus must be a nonnegative integer")
        if not test.get("sources"):
            errors.append(f"{name}: sources must be explicit")
        covered.update(test.get("sources", []))
        if test.get("tier") not in TIERS:
            errors.append(f"{name}: unknown tier")
        if not test.get("enabled") and not test.get("reason"):
            errors.append(f"{name}: disabled test needs a reason")
        if test.get("enabled"):
            if test.get("tier") == "host" and test.get("targets") != ["cpu"]:
                errors.append(f"{name}: host suites must use only the cpu target")
            if (
                not test.get("command")
                or not test.get("paths")
                or not test.get("targets")
            ):
                errors.append(f"{name}: enabled tests need command, paths and targets")
            if not 0 < test.get("timeout", 0) <= 3600:
                errors.append(f"{name}: timeout must be between 1 and 3600 seconds")
        for target_name in test.get("targets", []):
            if target_name not in targets:
                errors.append(f"{name}: unknown target {target_name}")
            elif not compatible(test, targets[target_name]):
                errors.append(f"{name}: requirements incompatible with {target_name}")
    for entry in catalog.get("support", []):
        if not entry.get("reason"):
            errors.append(f"Support entry requires a reason: {entry}")
        covered.update(entry["sources"])
    if files is not None:
        existing = set(files)
        errors.extend(
            f"Unregistered test source: {p}"
            for p in sorted(set(test_sources(files)) - covered)
        )
        errors.extend(
            f"Registered source does not exist: {p}" for p in sorted(covered - existing)
        )
    if errors:
        raise ValueError("\n".join(errors))


def changed_files(base, head="HEAD", root=ROOT):
    # A missing base (first push, force push, shallow checkout) fails open to full coverage.
    if not base or set(base) == {"0"}:
        return None
    try:
        return (
            subprocess.check_output(
                ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"],
                cwd=root,
                stderr=subprocess.PIPE,
            )
            .decode()
            .split("\0")[:-1]
        )
    except subprocess.CalledProcessError:
        return None


def matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def plan(
    catalog,
    targets,
    changed=None,
    tiers=None,
    only=None,
    exclude=None,
    target_names=None,
):
    tiers = set(tiers or ["host", "build", "integration"])
    if not tiers <= TIERS:
        raise ValueError(f"Unknown tiers: {tiers - TIERS}")
    names = {t["id"] for t in catalog["tests"]}
    for pattern in (only or []) + (exclude or []):
        if not any(fnmatch.fnmatchcase(n, pattern) for n in names):
            raise ValueError(f"Test selector matches nothing: {pattern}")
    requested = set(target_names or targets)
    if requested - targets.keys():
        raise ValueError(f"Unknown targets: {requested - targets.keys()}")
    global_change = changed is None or any(matches(p, GLOBAL) for p in changed)
    # Unknown source changes must not silently remove coverage.
    known_patterns = [p for t in catalog["tests"] for p in t.get("paths", [])]
    if changed is not None:
        for p in changed:
            if Path(p).suffix not in {
                ".md",
                ".rst",
                ".png",
                ".svg",
                ".jpg",
            } and not matches(p, known_patterns + GLOBAL):
                global_change = True
    matrix, decisions = {}, []
    for test in catalog["tests"]:
        name = test["id"]
        reason = "selected"
        if not test["enabled"]:
            reason = "disabled: " + test["reason"]
        elif only and not matches(name, only):
            reason = "not requested"
        elif exclude and matches(name, exclude):
            reason = "explicitly excluded"
        elif test["tier"] not in tiers:
            reason = "different tier"
        elif (
            not only
            and not global_change
            and not any(matches(p, test["paths"] + test["sources"]) for p in changed)
        ):
            reason = "unaffected"
        eligible = [
            n
            for n in test.get("targets", [])
            if n in requested and targets[n]["enabled"] and compatible(test, targets[n])
        ]
        if reason == "selected" and not eligible:
            reason = "no enabled compatible target"
        if reason == "selected":
            for n in eligible:
                matrix.setdefault(n, []).append(name)
        decisions.append(
            {
                "id": name,
                "reason": reason,
                "targets": eligible if reason == "selected" else [],
            }
        )
    if only and not matrix:
        raise ValueError(
            "Requested tests have no enabled compatible target in the selected tiers"
        )
    return {
        "schema": 1,
        "changed": changed,
        "full": global_change,
        "include": [
            {"target": n, "tests": tests} for n, tests in sorted(matrix.items())
        ],
        "decisions": decisions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "list", "plan"])
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--tiers", default="host,build,integration")
    parser.add_argument("--tests", default="")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--targets", default="")
    parser.add_argument("--output")
    args = parser.parse_args()
    catalog, targets = read_catalog()
    validate(catalog, targets, tracked_files())
    if args.action == "validate":
        print(
            f"Registered {len(catalog['tests'])} tests; every first-party test source is accounted for"
        )
        return
    if args.action == "list":
        for t in catalog["tests"]:
            print(
                f"{t['id']:48} {t['tier']:12} {'enabled' if t['enabled'] else 'disabled':8} {','.join(t.get('targets', []))} {t.get('reason', '')}"
            )
        return
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    result = plan(
        catalog,
        targets,
        None if args.full else changed_files(args.base, args.head),
        split(args.tiers),
        split(args.tests),
        split(args.exclude),
        split(args.targets),
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output, end="")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write("matrix=" + json.dumps({"include": result["include"]}) + "\n")
            stream.write(f"has_jobs={str(bool(result['include'])).lower()}\n")
            hardware = [job for job in result["include"] if job["target"] != "cpu"]
            stream.write("hardware=" + json.dumps({"include": hardware}) + "\n")
            stream.write(f"has_hardware={str(bool(hardware)).lower()}\n")
            stream.write(
                f"has_cpu={str(any(j['target'] == 'cpu' for j in result['include'])).lower()}\n"
            )


if __name__ == "__main__":
    main()

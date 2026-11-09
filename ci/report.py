"""Small Markdown summaries; detailed logs and JUnit stay in artifacts."""

import argparse
import json
from pathlib import Path


def report_plan(path):
    data = json.loads(path.read_text())
    print("### Selected suites\n")
    print("| Target | Suites |\n|---|---:|")
    for job in data["include"]:
        print(f"| {job['target']} | {len(job['tests'])} |")
    if not data["include"]:
        print("No affected enabled suites.")
    print(
        "\nThe plan artifact records the reason for every selection and exclusion. "
        "PR hardware execution requires a maintainer dispatch."
    )


def report_results(root):
    print(
        "### Results\n\n| Target | Passed ranks | Failed ranks | Skipped ranks | Infrastructure |\n|---|---:|---:|---:|---|"
    )
    found = False
    failed_suites = []
    skip_reasons = []
    timings = []
    for path in sorted(root.glob("*/results.json")):
        found = True
        data = json.loads(path.read_text())
        results = [r for r in data["results"] if not r.get("not_participating")]
        failed = [r for r in results if r["exit_code"]]
        skipped = [r for r in results if r.get("skip_reason")]
        for name in dict.fromkeys(r["id"] for r in results):
            ranks = [r for r in results if r["id"] == name]
            timings.append(
                (
                    path.parent.name,
                    name,
                    (
                        "SKIP"
                        if any(r.get("skip_reason") for r in ranks)
                        else ("FAIL" if any(r["exit_code"] for r in ranks) else "PASS")
                    ),
                    max(r["seconds"] for r in ranks),
                )
            )
        print(
            f"| {path.parent.name} | {len(results)-len(failed)-len(skipped)} | {len(failed)} | {len(skipped)} | {'FAILED' if data.get('infrastructure_error') else 'OK'} |"
        )
        for reason in dict.fromkeys(r["skip_reason"] for r in skipped):
            skip_reasons.append(f"- Skipped `{path.parent.name}`: {reason}")
        for result in failed:
            failed_suites.append(
                f"- `{path.parent.name}/{result['id']}` node {result['node']}: exit {result['exit_code']}; see artifact log."
            )
    if failed_suites:
        print("\n" + "\n".join(failed_suites))
    if skip_reasons:
        print("\n" + "\n".join(skip_reasons))
    if not found:
        print("\nNo results were produced; inspect the job failure.")
    if timings:
        print(
            "\n### Suite timings\n\nTime is the slowest participating rank, including builds in the suite command but excluding shared environment/dependency setup.\n"
        )
        print("| Target | Suite | Result | Seconds |\n|---|---|---|---:|")
        for target, name, status, seconds in timings:
            print(f"| {target} | `{name}` | {status} | {seconds:.3f} |")
    performance = root / "performance.json"
    if performance.exists():
        data = json.loads(performance.read_text())
        print("\n### Performance\n")
        for metric in data["measurements"]:
            limit = metric.get("threshold")
            bar = (
                ""
                if limit is None
                else f"; required {'>=' if metric['direction'] == 'higher' else '<='} {limit:.4g} {metric['unit']}"
            )
            print(
                f"- `{metric['key']}`: {metric['value']:.4g} {metric['unit']} ({metric['status']}{bar})"
            )
        for failure in data["failures"]:
            print(f"- FAILED: {failure}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "results"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    (report_plan if args.mode == "plan" else report_results)(args.path)

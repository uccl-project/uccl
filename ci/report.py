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
        "### Results\n\n| Target | Passed ranks | Failed ranks | Infrastructure |\n|---|---:|---:|---|"
    )
    found = False
    failed_suites = []
    for path in sorted(root.glob("*/results.json")):
        found = True
        data = json.loads(path.read_text())
        results = [r for r in data["results"] if not r.get("not_participating")]
        failed = [r for r in results if r["exit_code"]]
        print(
            f"| {path.parent.name} | {len(results)-len(failed)} | {len(failed)} | {'FAILED' if data.get('infrastructure_error') else 'OK'} |"
        )
        for result in failed:
            failed_suites.append(
                f"- `{path.parent.name}/{result['id']}` node {result['node']}: exit {result['exit_code']}; see artifact log."
            )
    if failed_suites:
        print("\n" + "\n".join(failed_suites))
    if not found:
        print("\nNo results were produced; inspect the job failure.")
    performance = root / "performance.json"
    if performance.exists():
        data = json.loads(performance.read_text())
        print("\n### Performance\n")
        for metric in data["measurements"]:
            print(
                f"- `{metric['key']}`: {metric['value']:.4g} {metric['unit']} ({metric['status']})"
            )
        for failure in data["failures"]:
            print(f"- FAILED: {failure}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "results"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    (report_plan if args.mode == "plan" else report_results)(args.path)

#!/usr/bin/env python3
"""Render one cross-target Markdown report for a CI run from its artifacts.

Artifacts come from jobs that may run pull-request code, so every string read
from them is untrusted. Only the catalog, target list and command-line arguments
reach the report as-is. Suite IDs must match ID_RE; log excerpts are redacted
and fenced; skip reasons are mapped to fixed labels and never echoed.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,80}$")
MAX_REPORT = 60000  # Small enough to post as a GitHub comment (65,536 chars).
MAX_NOTES = 3
MAX_DETAILS = 25
MAX_FILE = 16 << 20
LOG_TAIL = 256 << 10

GROUPS = {
    "ep": "EP",
    "p2p": "P2P",
    "collective": "Collective",
    "mccl": "Collective",
    "nixl": "NIXL",
}
GROUP_ORDER = ["EP", "P2P", "Collective", "NIXL", "Other"]

PASS, FAIL, CRASH, TIMEOUT = "pass", "fail", "crash", "timeout"
BUSY, UNREACHABLE, SKIPPED, INFRA, MISSING = (
    "busy",
    "unreachable",
    "skipped",
    "infra",
    "missing",
)
FAILURES = (FAIL, CRASH, TIMEOUT)
EXECUTED = (PASS,) + FAILURES
NOTE_LABELS = {
    INFRA: "Infrastructure error",
    UNREACHABLE: "Node unreachable",
    BUSY: "GPU busy",
    MISSING: "No results uploaded",
    SKIPPED: "Skipped",
}
NOT_RUN_LABELS = {
    BUSY: "GPU busy",
    UNREACHABLE: "node unreachable",
    SKIPPED: "skipped",
    INFRA: "infrastructure error",
    MISSING: "no results uploaded",
}
LEGEND = (
    "✅ Pass · ❌ Failure · 🟡 Also failing on main · ⏱ Timeout · "
    "⚠️ Infrastructure / unreachable · ⏸ GPU busy · ❔ No results · — Not run"
)

ERROR_PATTERNS = [
    re.compile(r"^(?:E\s+)?[A-Za-z_][\w.]*(?:Error|Exception|Interrupt)\b(?::|\(|$)"),
    re.compile(r"^FAILED\s"),
    re.compile(
        r"(?:\berror:|\bERROR\b|\bFatal\b|fatal error|Segmentation fault|Aborted|"
        r"terminate called|CUDA error|NCCL WARN|Assertion .* failed|No space left)"
    ),
]


def code(text):
    """Inline code for untrusted text, safe inside table cells."""
    text = html.escape(str(text)[:200]).replace("|", "&#124;").replace("\n", " ")
    return f"<code>{text}</code>"


def fence(lines):
    body = "\n".join(lines).replace("<!--", "<!-​-")
    ticks = max([3] + [len(m) + 1 for m in re.findall(r"`+", body)])
    return f"{'`' * ticks}text\n{body}\n{'`' * ticks}"


def label(target):
    return "CPU (GitHub)" if target == "cpu" else target.upper()


def fmt_seconds(value):
    if value is None:
        return ""
    return f"{value:.0f} s" if value >= 10 else f"{value:.1f} s"


class Redactor:
    def __init__(self, targets):
        names = set()
        for config in targets.values():
            names.update(config.get("hosts", []))
            names.update(config.get("interfaces", {}))
            names.update(config.get("interfaces", {}).values())
            if config.get("interface"):
                names.add(config["interface"])
        names = [n for n in names if isinstance(n, str) and n]
        names.sort(key=len, reverse=True)
        # Whole names only, so a host called "l40" leaves "l40s" alone.
        alternation = "|".join(re.escape(n) for n in names) or "(?!)"
        self.hosts = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")
        self.rules = [
            (re.compile(r"/[^\s'\"]*/uccl-ci/runs/[^/\s'\"]+/(?:src/)?"), ""),
            (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
            (re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}\b"), "<ip>"),
            (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
            (re.compile(r"/home/[^/\s'\"]+"), "/home/<user>"),
            (re.compile(r"/(?:ssd|data|scratch|mnt)\d*/[^\s'\"]*"), "<path>"),
            (re.compile(r"\benx[0-9a-f]{12}\b"), "<iface>"),
            (re.compile(r"@(?=[A-Za-z0-9])"), "@​"),
            (re.compile(r"[\x00-\x08\x0b-\x1f\x7f]"), ""),
        ]

    def __call__(self, line):
        line = self.hosts.sub("<host>", line)
        for pattern, repl in self.rules:
            line = pattern.sub(repl, line)
        return line[:240]


def read_json(path):
    try:
        if path.stat().st_size > MAX_FILE:
            return None
        return json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError):
        return None


def read_tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - LOG_TAIL))
            return stream.read().decode(errors="replace").splitlines()
    except OSError:
        return []


def excerpt(lines, redact):
    lines = [l.rstrip() for l in lines if l.strip()]
    for pattern in ERROR_PATTERNS:
        hits = [l.strip() for l in lines if pattern.search(l.strip())]
        if hits:
            return [redact(l) for l in hits[-3:]], True
    return [redact(l.strip()) for l in lines[-2:]], False


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def skip_kind(reason):
    reason = str(reason)
    if "SSH unavailable" in reason:
        return UNREACHABLE
    if (
        "GPU process" in reason
        or "GPU utilization" in reason
        or "became busy" in reason
    ):
        return BUSY
    return SKIPPED


def suite_status(ranks):
    """Collapse per-rank results; precedence: skip > timeout > crash > fail > pass."""
    ranks = [r for r in ranks if not r.get("not_participating")]
    if not ranks:
        return {"status": SKIPPED}
    skips = [skip_kind(r["skip_reason"]) for r in ranks if r.get("skip_reason")]
    if skips:
        return {"status": UNREACHABLE if UNREACHABLE in skips else skips[0]}
    seconds = max(
        [s for s in (as_float(r.get("seconds")) for r in ranks) if s is not None],
        default=None,
    )
    codes = [(as_int(r.get("exit_code")), as_int(r.get("node"))) for r in ranks]
    for status, test in (
        (TIMEOUT, lambda c: c == 124),
        (CRASH, lambda c: c < 0),
        (FAIL, lambda c: c != 0),
    ):
        hit = [(c, n) for c, n in codes if test(c)]
        if hit:
            return {
                "status": status,
                "code": hit[0][0],
                "node": hit[0][1],
                "seconds": seconds,
            }
    return {"status": PASS, "seconds": seconds}


def failure_reason(info):
    after = (
        f" after {fmt_seconds(info.get('seconds'))}"
        if info.get("seconds") is not None
        else ""
    )
    if info["status"] == TIMEOUT:
        return f"timed out{after}"
    if info["status"] == CRASH:
        try:
            name = signal.Signals(-info["code"]).name
        except ValueError:
            name = f"signal {-info['code']}"
        return f"killed by {name}{after}"
    return f"exit {info['code']}{after}"


def target_dir(root, target):
    for path in sorted(root.glob(f"ci-*/{target}/results.json")):
        if path.parent.parent.name in ("ci-cpu", f"ci-gpu-{target}"):
            return path.parent
    return None


def find_log(tdir, node, name):
    for pattern in (f"node-{node}/{name}", f"*/node-{node}/{name}"):
        for path in sorted(tdir.glob(pattern)):
            if path.is_file() and not path.is_symlink():
                return path
    return None


def load_run(root, targets):
    """Return {target: {"suites": {id: info}, "infra": str|None, "dir": Path|None}}."""
    run = {}
    invalid = 0
    for target in targets:
        tdir = target_dir(root, target)
        if tdir is None:
            continue
        data = read_json(tdir / "results.json")
        entry = {"suites": {}, "infra": None, "dir": tdir}
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            entry["infra"] = "results unreadable"
            run[target] = entry
            continue
        if data.get("infrastructure_error"):
            match = re.search(
                r"node-(\d+)/environment\.log", str(data["infrastructure_error"])
            )
            entry["infra"] = {"node": as_int(match.group(1)) if match else None}
        by_id = {}
        for result in data["results"]:
            if not isinstance(result, dict) or not ID_RE.match(
                str(result.get("id", ""))
            ):
                invalid += 1
                continue
            by_id.setdefault(result["id"], []).append(result)
        entry["suites"] = {name: suite_status(ranks) for name, ranks in by_id.items()}
        run[target] = entry
    return run, invalid


def load_plan(root, targets):
    data = read_json(root / "ci-plan" / "ci-plan.json")
    if not isinstance(data, dict) or not isinstance(data.get("include"), list):
        return None
    plan = {}
    for item in data["include"]:
        if not isinstance(item, dict) or item.get("target") not in targets:
            continue
        tests = [
            t for t in item.get("tests", []) if isinstance(t, str) and ID_RE.match(t)
        ]
        plan.setdefault(item["target"], []).extend(dict.fromkeys(tests))
    return plan


def load_baseline(root, targets):
    """Latest executed result per (target, suite) across main runs, newest first."""
    seen, runs = {}, []
    if root is None or not root.is_dir():
        return seen, runs
    for run_dir in sorted(
        (p for p in root.iterdir() if p.name.isdigit()), key=lambda p: -int(p.name)
    ):
        run, _ = load_run(run_dir, targets)
        used = False
        for target, entry in run.items():
            for name, info in entry["suites"].items():
                if info["status"] in EXECUTED and (target, name) not in seen:
                    seen[(target, name)] = (info["status"], run_dir.name)
                    used = True
        if used:
            runs.append(run_dir.name)
    return seen, runs


def group_of(name):
    return GROUPS.get(name.split(".", 1)[0], "Other")


def render(args):
    all_targets = json.loads((args.config / "targets.json").read_text())
    targets = {k: v for k, v in all_targets.items() if v.get("enabled", True)}
    catalog = {
        t["id"] for t in json.loads((args.config / "tests.json").read_text())["tests"]
    }
    redact = Redactor(all_targets)
    root = args.artifacts
    run_url = args.run_url
    jobs = (
        json.loads(os.environ.get(args.jobs_env, "{}") or "{}") if args.jobs_env else {}
    )

    plan = load_plan(root, targets)
    if plan is None:
        rows = [
            f"| {k} | {html.escape(str(v.get('result')))} |"
            for k, v in jobs.items()
            if isinstance(v, dict)
        ]
        return "\n".join(
            [
                f"**[FAILED]** UCCL CI [run]({run_url}): planning failed, so no suites ran.",
                "",
            ]
            + (["| Job | Result |", "|---|---|"] + rows if rows else [])
        )

    # Pull-request runs plan lab targets but skip the GPU job until a maintainer
    # dispatches the run, so those targets were not expected to report.
    gpu_job = jobs.get("gpu")
    undispatched = []
    if isinstance(gpu_job, dict) and gpu_job.get("result") == "skipped":
        undispatched = [t for t in targets if t in plan and t != "cpu"]
        for target in undispatched:
            del plan[target]

    run, invalid = load_run(root, list(plan))
    baseline, baseline_runs = load_baseline(args.baseline, list(plan))

    rows, details, not_run, perf_rows = (
        [],
        {"New": [], "Old": [], "Not compared": [], "Infra": []},
        [],
        [],
    )
    executed = passed = planned_total = perf_failed = unbaselined = 0
    perf_notes = []
    new_suites = set()
    for target in [t for t in targets if t in plan]:
        planned = plan[target]
        entry = run.get(target)
        cells, notes = {}, {}
        for name in planned:
            planned_total += 1
            if name not in catalog:
                new_suites.add(name)
            if entry is None:
                info = {"status": MISSING}
            elif entry["infra"]:
                info = entry["suites"].get(name) or {"status": INFRA}
            else:
                info = entry["suites"].get(name) or {"status": SKIPPED}
            status = info["status"]
            group = group_of(name)
            cell = cells.setdefault(group, {"pass": 0, "total": 0, "icons": set()})
            cell["total"] += 1
            if status in EXECUTED:
                executed += 1
            if status == PASS:
                passed += 1
                cell["pass"] += 1
                cell["icons"].add("✅")
                continue
            if status in FAILURES:
                base = baseline.get((target, name))
                novelty = (
                    "Not compared"
                    if base is None
                    else ("Old" if base[0] in FAILURES else "New")
                )
                icon = (
                    "🟡" if novelty == "Old" else ("⏱" if status == TIMEOUT else "❌")
                )
                cell["icons"].add(icon)
                reason = failure_reason(info)
                suffix = " (new suite)" if name in new_suites else ""
                notes.setdefault(novelty, []).append(f"{code(name)}{suffix} ({reason})")
                log = find_log(entry["dir"], info["node"], f"{name}.log")
                text, matched = excerpt(read_tail(log), redact) if log else ([], False)
                details[novelty].append(
                    (
                        f"- {code(name)} — {label(target)} node {info['node']}, {reason}"
                        + (
                            f"; main last executed it in run #{base[1]}" if base else ""
                        ),
                        ("" if matched else "last output: ", text),
                    )
                )
                continue
            cell["icons"].add({BUSY: "⏸", MISSING: "❔"}.get(status, "⚠️"))
            not_run.append((target, status, name))
        name_label = label(target)
        row_cells = []
        groups_present = [
            g
            for g in GROUP_ORDER
            if any(group_of(n) == g for t in plan for n in plan[t])
        ]
        for group in groups_present:
            cell = cells.get(group)
            if not cell:
                row_cells.append("—")
                continue
            order = ["❌", "⏱", "🟡", "⚠️", "❔", "⏸", "✅"]
            icon = next(i for i in order if i in cell["icons"])
            row_cells.append(f"{icon} {cell['pass']}/{cell['total']}")
        missing = [s for t2, s, n in not_run if t2 == target]
        note_parts = []
        for novelty in ("New", "Old", "Not compared"):
            items = notes.get(novelty, [])
            shown = items[:MAX_NOTES]
            if shown:
                more = (
                    f"<br>… and {len(items) - len(shown)} more"
                    if len(items) > len(shown)
                    else ""
                )
                note_parts.append(
                    "<br>".join(f"**{novelty}:** {i}" for i in shown) + more
                )
        for kind in (INFRA, UNREACHABLE, BUSY, MISSING, SKIPPED):
            count = missing.count(kind)
            if count:
                note_parts.append(
                    f"**{NOTE_LABELS[kind]}:** {count} suite{'s' if count > 1 else ''} not run"
                )
        if entry and isinstance(entry["infra"], dict):
            node = entry["infra"]["node"]
            log = (
                find_log(entry["dir"], node, "environment.log")
                if node is not None
                else None
            )
            text, _ = excerpt(read_tail(log), redact) if log else ([], False)
            details["Infra"].append(
                (
                    f"- {name_label}: environment setup failed{'' if node is None else f' on node {node}'} before any suite ran",
                    ("", text[-1:]),
                )
            )
        rows.append(
            f"| {name_label} | "
            + " | ".join(row_cells)
            + f" | {'<br>'.join(note_parts) or '—'} |"
        )

        perf = (
            read_json(root / f"ci-gpu-{target}" / "performance.json")
            if target != "cpu"
            else None
        )
        if isinstance(perf, dict):
            # ci/performance.py emits pass, regression (hard gate), warning (soft gate) or unbaselined.
            measurements = [
                m for m in perf.get("measurements", []) if isinstance(m, dict)
            ][:200]
            unbaselined += sum(
                1 for m in measurements if m.get("status") == "unbaselined"
            )
            invalid_metrics = max(
                0,
                len(perf.get("failures", []) or [])
                - sum(1 for m in measurements if m.get("status") == "regression"),
            )
            if invalid_metrics and not (entry and entry["infra"]):
                perf_failed += invalid_metrics
                perf_notes.append(
                    f"{name_label}: {invalid_metrics} benchmark measurement(s) missing or invalid"
                )
            for metric in measurements:
                status = metric.get("status")
                if status not in ("regression", "warning"):
                    continue
                perf_failed += status == "regression"
                value = as_float(metric.get("value"))
                perf_rows.append(
                    f"| {name_label} | {code(metric.get('key', '?'))} | {'' if value is None else f'{value:.4g}'} {code(metric.get('unit', ''))} "
                    f"| {'required' if status == 'regression' else 'advisory'} | {'❌ regression' if status == 'regression' else '🟡 warning'} |"
                )

    failures = (
        sum(len(details[k]) for k in ("New", "Old", "Not compared")) + perf_failed
    )
    infra = bool(details["Infra"]) or any(s in (INFRA, MISSING) for _, s, _ in not_run)
    word = (
        "FAILED"
        if failures
        else ("INFRA ERROR" if infra else ("INCOMPLETE" if not_run else "SUCCESS"))
    )
    head = f"**[{word}]** UCCL CI [run]({run_url}) — {passed}/{executed} executed suites passed"
    if not planned_total:
        head = f"**[{word}]** UCCL CI [run]({run_url}) — no suites were selected"
    if not_run:
        head += f"; {len(not_run)} of {planned_total} planned suites did not run"
    lines = [head, ""]
    meta = []
    if args.sha and re.fullmatch(r"[0-9a-f]{7,40}", args.sha):
        meta.append(f"commit `{args.sha[:12]}`")
    if baseline_runs:
        links = ", ".join(
            f"[#{r}]({args.repo_url}/actions/runs/{r})" for r in baseline_runs[:5]
        )
        meta.append(
            f"compared with main runs {links} (latest executed result per suite; selections differ)"
        )
    elif failures:
        meta.append(
            "no main baseline available, so failures are not classified as new or old"
        )
    if undispatched:
        meta.append(
            f"GPU job skipped, so {', '.join(label(t) for t in undispatched)} did not run"
            " (PR GPU execution requires a maintainer dispatch)"
        )
    lines += [" · ".join(meta), ""]
    if not planned_total:
        return "\n".join(lines + [f"Logs: [run artifacts]({run_url})."])

    groups_present = [
        g for g in GROUP_ORDER if any(group_of(n) == g for t in plan for n in plan[t])
    ]
    lines += [
        "### Suites",
        "",
        "| Target | " + " | ".join(groups_present) + " | Notes |",
        "|---|" + "---|" * len(groups_present) + "---|",
        *rows,
        "",
        LEGEND,
    ]
    if invalid:
        lines += [
            "",
            f"{invalid} result record(s) with malformed suite IDs were ignored.",
        ]
    if perf_rows or perf_notes:
        lines += ["", "### Performance gates not passing", ""]
        if perf_rows:
            lines += [
                "| Target | Metric | Value | Gate | Status |",
                "|---|---|---|---|---|",
                *perf_rows[:30],
            ]
        lines += [f"- {note}" for note in perf_notes]
    if unbaselined:
        lines += [
            "",
            f"{unbaselined} performance metric(s) have no reviewed baseline yet and were not gated.",
        ]

    detail_lines = []
    sections = [
        ("New", "New relative to main"),
        ("Old", "Also failing on main"),
        ("Not compared", "Not compared (no executed result on main)"),
        ("Infra", "Infrastructure"),
    ]
    for key, title in sections:
        items = details[key]
        if not items:
            continue
        detail_lines += ["", f"#### {title}"]
        for text, (prefix, excerpt_lines) in items[:MAX_DETAILS]:
            detail_lines.append(text)
            if excerpt_lines:
                block = [prefix + excerpt_lines[0]] + excerpt_lines[1:]
                detail_lines += ["  " + l for l in fence(block).splitlines()]
        if len(items) > MAX_DETAILS:
            detail_lines.append(
                f"- … and {len(items) - MAX_DETAILS} more; see the run artifacts."
            )
    if not_run:
        detail_lines += ["", "#### Planned but not run"]
        for target in [t for t in targets if t in plan]:
            for kind in (INFRA, UNREACHABLE, BUSY, MISSING, SKIPPED):
                names = [n for t2, s, n in not_run if t2 == target and s == kind]
                if names:
                    shown = ", ".join(code(n) for n in names[:12])
                    more = f" and {len(names) - 12} more" if len(names) > 12 else ""
                    detail_lines.append(
                        f"- {label(target)} ({NOT_RUN_LABELS[kind]}): {shown}{more}"
                    )
    if detail_lines:
        lines += [
            "",
            "<details>",
            "<summary>Failure and skip details</summary>",
            *detail_lines,
            "",
            "</details>",
        ]
    lines += ["", f"Full logs, JUnit and timings: [run artifacts]({run_url})."]

    body = "\n".join(lines)
    if len(body) > MAX_REPORT:
        cut = body[: MAX_REPORT - 200].rsplit("\n", 1)[0]
        body = (
            cut
            + f"\n\n… report truncated; see the [run artifacts]({run_url}).\n"
            + ("</details>" if "<details>" in cut else "")
        )
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "ci" / "config")
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--repo-url", default="https://github.com/uccl-project/uccl")
    parser.add_argument("--sha", default="")
    parser.add_argument("--jobs-env", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    repo = r"https://github\.com/[\w.-]+/[\w.-]+"
    if not re.fullmatch(repo + r"/actions/runs/\d+(/attempts/\d+)?", args.run_url):
        parser.error("--run-url must be a GitHub Actions run URL")
    if not re.fullmatch(repo, args.repo_url):
        parser.error("--repo-url must be a GitHub repository URL")
    body = render(args)
    if args.out:
        args.out.write_text(body + "\n")
    else:
        print(body)


if __name__ == "__main__":
    main()

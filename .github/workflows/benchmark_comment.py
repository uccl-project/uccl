#!/usr/bin/env python3
"""Create and update PR benchmark status comments from GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def request(method: str, path: str, payload: dict | None = None) -> dict:
    token = os.environ["GITHUB_TOKEN"]
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{api_url}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode(errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {details}") from exc
    return json.loads(body.decode()) if body else {}


def write_output(name: str, value: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    with open(output, "a", encoding="utf-8") as f:
        f.write(f"{name}={value}\n")


def status_label(status: str) -> str:
    labels = {
        "queued": "queued",
        "running": "running",
        "success": "passed",
        "failure": "failed",
        "cancelled": "cancelled",
        "skipped": "skipped",
        "dispatch-failed": "failed to dispatch",
    }
    return labels.get(status, status)


def build_body(args: argparse.Namespace) -> str:
    marker = args.marker or f"run-benchmark-{args.benchmark}-{args.pr}-{int(time.time())}"
    lines = [
        f"<!-- {marker} -->",
        f"### Benchmark `{args.benchmark}` {status_label(args.status)}",
        "",
        f"PR: #{args.pr}",
        f"Commit: `{args.sha}`",
    ]

    if args.requester:
        lines.append(f"Requested by: @{args.requester}")
    if args.result:
        lines.append(f"Result: `{args.result}`")
    if args.workflow_url:
        lines.append(f"Workflow run: {args.workflow_url}")
    elif args.workflow:
        repo_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{args.repo}"
        lines.append(f"Workflow: {repo_url}/actions/workflows/{args.workflow}")
    if args.error:
        lines.extend(["", f"Error: {args.error}"])
    elif args.status in {"queued", "running"}:
        lines.extend(["", "Results will be posted here when it finishes."])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["create", "update"], required=True)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--comment-id", default="")
    parser.add_argument("--marker", default="")
    parser.add_argument("--requester", default="")
    parser.add_argument("--workflow", default="")
    parser.add_argument("--workflow-url", default="")
    parser.add_argument("--result", default="")
    parser.add_argument("--error", default="")
    args = parser.parse_args()

    if not args.repo:
        raise SystemExit("--repo or GITHUB_REPOSITORY is required")

    owner_repo = args.repo
    marker = args.marker or f"run-benchmark-{args.benchmark}-{args.pr}-{int(time.time())}"
    args.marker = marker
    body = build_body(args)

    if args.mode == "create":
        comment = request(
            "POST",
            f"/repos/{owner_repo}/issues/{args.pr}/comments",
            {"body": body},
        )
        write_output("comment_id", str(comment["id"]))
        write_output("reply_marker", marker)
        return

    if not args.comment_id:
        raise SystemExit("--comment-id is required for update mode")
    request(
        "PATCH",
        f"/repos/{owner_repo}/issues/comments/{args.comment_id}",
        {"body": body},
    )


if __name__ == "__main__":
    main()

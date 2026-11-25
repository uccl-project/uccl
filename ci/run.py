"""Run registered suites locally or over SSH, with a per-machine lease and rank barriers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from catalog import ROOT, read_catalog, tracked_files, validate, plan
from idle import skip_reason


class TargetSkipped(Exception):
    pass


class WorkerError(RuntimeError):
    def __init__(self, message, results):
        super().__init__(message)
        self.results = list(results)


def skip_results(names, definitions, target, reason):
    return [
        dict(
            id=name,
            tier=definitions[name]["tier"],
            target=target,
            node=node,
            exit_code=0,
            seconds=0,
            skip_reason=reason,
        )
        for name in names
        for node in range(definitions[name].get("nodes", 1))
    ]


def remote_preflight(ssh, hosts, target, root):
    """Probe every selected machine before uploading source or building anything."""
    states = {}
    script = (ROOT / "ci/idle.py").read_text()
    for host in hosts:
        command = shlex.join(
            [
                "python3",
                "-",
                "--root",
                root,
                "--gpus",
                str(target["gpus"]),
                "--capabilities",
                ",".join(target["capabilities"]),
            ]
        )
        try:
            result = subprocess.run(
                [*ssh, host, command],
                input=script,
                text=True,
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 255:
                states[host] = dict(
                    state="unavailable",
                    reason=f"SSH unavailable: {result.stderr.strip()}",
                )
            else:
                result.check_returncode()
                states[host] = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            states[host] = dict(
                state="unavailable", reason="SSH admission probe timed out"
            )
        if skip_reason(states[host]):
            break
    return states


def unreachable_hosts(ssh, hosts):
    """Distinguish lost SSH access from a worker/build/test error."""
    unavailable = {}
    for host in hosts:
        try:
            result = subprocess.run(
                [*ssh, host, "true"], text=True, capture_output=True, timeout=20
            )
            if result.returncode == 255:
                unavailable[host] = result.stderr.strip() or "SSH exited with code 255"
        except subprocess.TimeoutExpired:
            unavailable[host] = "SSH connection timed out"
        except OSError:
            continue  # A broken local SSH executable is not an offline lab node.
    return unavailable


def mark_skipped(results, selected, definitions, target, reason, unavailable=False):
    # Lost artifacts cannot establish a pass. Preserve any already observed failure.
    if unavailable:
        for result in results:
            if result["exit_code"] in {0, 125} and not result.get("not_participating"):
                if result["exit_code"]:
                    result["interrupted_exit_code"] = result["exit_code"]
                result["exit_code"] = 0
                result["skip_reason"] = reason
    present = {(r["id"], r["node"]) for r in results}
    results.extend(
        r
        for r in skip_results(selected, definitions, target, reason)
        if (r["id"], r["node"]) not in present
    )


def receive_all(workers, event, timeout=2700):
    pending, messages = set(range(len(workers))), {}
    deadline = time.monotonic() + timeout
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Workers did not reach {event}")
        for index in list(pending):
            try:
                message = workers[index].events.get_nowait()
            except queue.Empty:
                continue
            if message.get("event") != event:
                raise RuntimeError(f"Expected {event}, got {message}")
            if message.get("skip_reason"):
                raise TargetSkipped(message["skip_reason"])
            messages[index] = message
            pending.remove(index)
        if pending:
            time.sleep(0.05)
    return [messages[index] for index in range(len(workers))]


def check(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


class Worker:
    def __init__(self, command, stderr):
        self.errors = open(stderr, "w")
        self.child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            text=True,
            bufsize=1,
        )
        self.events = queue.Queue()

        def read():
            try:
                for line in self.child.stdout:
                    try:
                        self.events.put(json.loads(line))
                    except ValueError:
                        self.events.put(
                            {
                                "event": "fatal",
                                "error": "Invalid worker protocol: " + line,
                            }
                        )
            finally:
                self.events.put({"event": "fatal", "error": "Worker connection closed"})

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def receive(self, event, timeout=2700):
        message = self.events.get(timeout=timeout)
        if message["event"] != event:
            raise RuntimeError(f"Expected {event}, got {message}")
        return message

    def send(self, message):
        self.child.stdin.write(json.dumps(message) + "\n")
        self.child.stdin.flush()

    def close(self):
        try:
            self.child.stdin.close()  # EOF cancels the worker's owned child group.
        except (BrokenPipeError, OSError):
            pass
        try:
            self.child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.child.terminate()
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        self.errors.close()
        self.reader.join(timeout=1)
        self.child.stdout.close()


def archive_source(destination):
    files = tracked_files()
    with tarfile.open(destination, "w:gz") as tar:
        for name in files:
            path = ROOT / name
            if path.is_file() or path.is_symlink():
                tar.add(path, arcname=name, recursive=False)
            elif path.is_dir():  # initialized, pinned Git submodule

                def source_only(info):
                    return (
                        None
                        if any(
                            p == ".git" or p == ".venv" for p in Path(info.name).parts
                        )
                        else info
                    )

                tar.add(path, arcname=name, filter=source_only)


def collect_results(workers, name, timeout):
    """Cancel this suite's other ranks promptly when one rank fails."""
    pending, results = set(range(len(workers))), {}
    deadline = time.monotonic() + timeout
    cancelled = False
    while pending:
        if time.monotonic() >= deadline:
            raise WorkerError(f"Workers did not finish {name}", results.values())
        for index in list(pending):
            try:
                result = workers[index].events.get_nowait()
            except queue.Empty:
                continue
            if result.get("event") != "result" or result.get("id") != name:
                raise WorkerError(
                    f"Expected result for {name}, got {result}", results.values()
                )
            pending.remove(index)
            results[index] = result
            if (
                result["exit_code"] not in {0, 125} or result.get("skip_reason")
            ) and not cancelled:
                cancelled = True
                for peer in pending:
                    try:
                        workers[peer].send(
                            {
                                "cancel": name,
                                **(
                                    {"skip_reason": result["skip_reason"]}
                                    if result.get("skip_reason")
                                    else {}
                                ),
                            }
                        )
                    except BrokenPipeError:
                        # The final result may already be queued after worker exit.
                        pass
        if pending:
            time.sleep(0.05)
    values = [results[i] for i in range(len(workers))]
    busy = next((r["skip_reason"] for r in values if r.get("skip_reason")), None)
    if busy:
        for result in values:
            result.update(
                interrupted_exit_code=result["exit_code"], exit_code=0, skip_reason=busy
            )
    return values


def write_summary(results, output, error=None):
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps({"results": results, "infrastructure_error": error}, indent=2)
    )
    suite = ET.Element("testsuite", name="uccl-ci")
    for result in results:
        if result.get("not_participating"):
            continue
        case = ET.SubElement(
            suite,
            "testcase",
            name=result["id"],
            classname=f"{result['target']}.node{result['node']}",
            time=str(result["seconds"]),
        )
        if result.get("skip_reason"):
            ET.SubElement(case, "skipped", message=result["skip_reason"])
        elif result["exit_code"]:
            ET.SubElement(
                case,
                "failure",
                message=str(result.get("error") or f"exit {result['exit_code']}"),
            )
        ET.SubElement(case, "system-out").text = result.get("log", "")
    if error:
        case = ET.SubElement(suite, "testcase", name="infrastructure", classname="ci")
        ET.SubElement(case, "error", message=error)
    suite.set("tests", str(len(suite)))
    suite.set("failures", str(len(suite.findall("testcase/failure"))))
    suite.set("errors", str(len(suite.findall("testcase/error"))))
    suite.set("skipped", str(len(suite.findall("testcase/skipped"))))
    ET.ElementTree(suite).write(
        output / "junit.xml", encoding="unicode", xml_declaration=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--tests", default="")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--tiers", default="cpu,build,integration")
    parser.add_argument("--ssh-config")
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--hosts",
        help="Explicit SSH aliases for local validation (for example CPU tests on a Linux server)",
    )
    parser.add_argument("--root", default="/var/lib/uccl-ci")
    parser.add_argument("--output", default="ci-results")
    parser.add_argument("--run-id", default="local-" + uuid.uuid4().hex[:12])
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="Keep remote source/build trees for debugging",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error(
            "run-id must contain only letters, numbers, underscores and hyphens"
        )
    catalog, targets = read_catalog()
    validate(catalog, targets, tracked_files())
    if args.target not in targets or not targets[args.target]["enabled"]:
        parser.error("Unknown or disabled target")
    target = targets[args.target]
    split = lambda value: list(filter(None, value.split(",")))
    selected_plan = (
        json.loads(Path(args.plan).read_text())
        if args.plan
        else plan(
            catalog,
            targets,
            tiers=split(args.tiers),
            only=split(args.tests),
            exclude=split(args.exclude),
            target_names=[args.target],
        )
    )
    selected = next(
        (
            job["tests"]
            for job in selected_plan["include"]
            if job["target"] == args.target
        ),
        [],
    )
    if not selected:
        parser.error("No tests selected; refusing to report an empty run as passing")
    output = Path(args.output).resolve() / args.target
    if output.exists() and any(output.iterdir()):
        parser.error(
            f"Result directory is not empty: {output}; use a fresh --output path"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(selected_plan, indent=2))
    source = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
    }
    (output / "source.json").write_text(json.dumps(source, indent=2))
    hosts = (
        (split(args.hosts) if args.hosts else target["hosts"]) if not args.local else []
    )
    definitions = {test["id"]: test for test in catalog["tests"]}
    if any(definitions[name].get("nodes", 1) > max(1, len(hosts)) for name in selected):
        parser.error("Selected suites need more nodes than this invocation provides")
    hosts = hosts[: max(definitions[name].get("nodes", 1) for name in selected)]
    ssh_options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if args.ssh_config:
        ssh_options += ["-F", str(Path(args.ssh_config).resolve())]
    ssh = ["ssh", *ssh_options]
    remote = f"{args.root}/runs/{args.run_id}-{args.target}"
    workers, results, error, staged = [], [], None, []
    unavailable, skipped = {}, None
    tmpdir = tempfile.TemporaryDirectory(prefix="uccl-ci-")
    local = None
    try:
        tmp = tmpdir.name
        admission = remote_preflight(ssh, hosts, target, args.root)
        (output / "admission.json").write_text(json.dumps(admission, indent=2))
        unavailable.update(
            {
                h: s["reason"]
                for h, s in admission.items()
                if s["state"] == "unavailable"
            }
        )
        busy = [
            f"{host}: {skip_reason(state)}"
            for host, state in admission.items()
            if skip_reason(state)
        ]
        if busy:
            raise TargetSkipped("; ".join(busy))
        archive = Path(tmp) / "source.tar.gz"
        archive_source(archive)
        source["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        (output / "source.json").write_text(json.dumps(source, indent=2))
        # Each node receives its own checkout; no shared home/NFS assumptions.
        for index, host in enumerate(hosts or [None]):
            if host:
                command = f"set -eu; umask 077; mkdir {shlex.quote(remote)}; mkdir {shlex.quote(remote + '/src')}; tar -xzf - -C {shlex.quote(remote + '/src')}"
                with open(archive, "rb") as stream:
                    check([*ssh, host, command], stdin=stream)
                staged.append(host)
                worker_cmd = f"cd {shlex.quote(remote + '/src')} && exec python3 ci/worker.py --target {shlex.quote(args.target)} --node {index} --host {shlex.quote(host)} --root {shlex.quote(args.root)}"
                command = [*ssh, host, worker_cmd]
            else:
                local = Path(tmp) / "run" / "src"
                local.mkdir(parents=True)
                with tarfile.open(archive) as tar:
                    tar.extractall(local, filter="data")
                command = [
                    sys.executable,
                    str(local / "ci/worker.py"),
                    "--target",
                    args.target,
                    "--root",
                    str(Path(tmp) / "state"),
                    "--local",
                ]
            workers.append(Worker(command, output / f"node-{index}.ssh.log"))
        ready = receive_all(workers, "ready")
        for worker in workers:
            worker.send({"tests": selected, "master_addr": ready[0]["address"]})
        receive_all(workers, "initialized")
        for index, name in enumerate(selected):
            available = receive_all(workers, "available")
            if any(r["id"] != name for r in available):
                raise RuntimeError("Workers disagree about test ordering")
            for worker in workers:
                worker.send({"action": "prepare", "id": name})
            prepared = receive_all(workers, "prepared")
            if any(p["id"] != name for p in prepared):
                raise RuntimeError("Workers disagree about test ordering")
            action = "run" if all(p["ok"] for p in prepared) else "skip"
            for worker in workers:
                worker.send({"action": action, "port": 24000 + index % 10000})
            current = collect_results(workers, name, timeout=3700)
            results.extend(current)
            if any(r.get("skip_reason") for r in current):
                raise TargetSkipped(
                    next(r["skip_reason"] for r in current if r.get("skip_reason"))
                )
            status = "FAIL" if any(r["exit_code"] for r in current) else "PASS"
            print(f"{args.target}: {name}: {status}", flush=True)
        for worker in workers:
            worker.receive("complete", timeout=30)
    except TargetSkipped as exc:
        skipped = str(exc)
    except (Exception, KeyboardInterrupt) as exc:
        results.extend(getattr(exc, "results", []))
        if (
            isinstance(exc, subprocess.CalledProcessError)
            and exc.returncode == 255
            and isinstance(exc.cmd, (list, tuple))
            and exc.cmd[0] == "ssh"
            and exc.cmd[-2] in hosts
        ):
            unavailable[exc.cmd[-2]] = "SSH command disconnected (exit 255)"
        if not isinstance(exc, KeyboardInterrupt):
            unavailable.update(
                unreachable_hosts(ssh, [h for h in hosts if h not in unavailable])
            )
        if unavailable:
            skipped = "; ".join(
                f"{h}: SSH unavailable: {why}" for h, why in unavailable.items()
            )
        else:
            error = str(exc) or type(exc).__name__
            print(f"CI infrastructure error: {error}", file=sys.stderr)
    finally:
        for worker in workers:
            worker.close()
        for host in staged:
            if host in unavailable:
                continue  # Preserve the remote workspace for later recovery.
            destination = output / host
            destination.mkdir(exist_ok=True)
            result = subprocess.run(
                [
                    "rsync",
                    "-az",
                    "-e",
                    shlex.join(ssh),
                    f"{host}:{remote}/results/",
                    str(destination),
                ]
            )
            if result.returncode:
                unavailable.update(unreachable_hosts(ssh, [host]))
                if host in unavailable:
                    skipped = f"{host}: SSH unavailable during artifact collection: {unavailable[host]}"
                elif not error:
                    error = f"Could not collect results from {host}"
            elif result.returncode == 0 and not args.keep_remote:
                # Only delete the unique run directory we created and successfully archived.
                cleanup = subprocess.run(
                    [*ssh, host, f"rm -rf -- {shlex.quote(remote)}"]
                )
                if cleanup.returncode == 255:
                    unavailable[host] = "SSH unavailable during workspace cleanup"
                    skipped = f"{host}: {unavailable[host]}"
                elif cleanup.returncode and not error:
                    error = f"Could not clean the collected workspace on {host}"
        if local and (local.parent / "results").exists():
            shutil.copytree(local.parent / "results", output, dirs_exist_ok=True)
        tmpdir.cleanup()
        if skipped:
            mark_skipped(
                results, selected, definitions, args.target, skipped, bool(unavailable)
            )
            print(f"{args.target}: SKIPPED: {skipped}", flush=True)
        write_summary(results, output, error)
    return int(bool(error) or any(r["exit_code"] for r in results))


if __name__ == "__main__":
    sys.exit(main())

"""One non-root worker per node; JSON lines on stdin/stdout coordinate each test."""

import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

from catalog import ROOT, read_catalog
from process import run


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def receive():
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("controller disconnected")
        message = json.loads(line)
        if "cancel" not in message:  # A peer may have failed just after we finished.
            return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--root", default="/var/lib/uccl-ci")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if os.geteuid() == 0:
        raise RuntimeError("CI must run as a non-root user")
    catalog, targets = read_catalog()
    target = targets[args.target]
    ci_root = Path(args.root).resolve()
    ci_root.mkdir(parents=True, exist_ok=True)
    artifacts = ROOT.parent / "results" / f"node-{args.node}"
    artifacts.mkdir(parents=True, exist_ok=True)
    lock = open(ci_root / "hardware.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another CI run holds this machine's lease") from exc
    filesystems = {}
    for directory in (ci_root, ci_root / "cache", ROOT):
        directory.mkdir(parents=True, exist_ok=True)
        fs = subprocess.check_output(
            ["findmnt", "-T", str(directory), "-no", "FSTYPE"], text=True
        ).strip()
        if fs.startswith("nfs") or fs == "cifs":
            raise RuntimeError(f"CI requires local storage: {directory} uses {fs}")
        filesystems[str(directory)] = fs
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    env.update(target.get("env", {}))
    env.update(
        {
            "CI_ROOT": str(ci_root),
            "CI_REPO": str(ROOT),
            "CI_TARGET": args.target,
            "CI_ARTIFACTS": str(artifacts),
            "CI_NODE": str(args.node),
            # Some lab images share /etc/machine-id. Every process on this
            # physical node gets one identity; different nodes must not use IPC.
            "UHM_HOST_ID_OVERRIDE": f"uccl-ci-{args.target}-{args.node}",
            "CI_JOBS": str(min(8, os.cpu_count() or 2)),
            "PYTHONUNBUFFERED": "1",
        }
    )
    metadata = {
        "host": socket.gethostname(),
        "uid": os.getuid(),
        "filesystem": fs,
        "filesystems": filesystems,
        "target": args.target,
        "node": args.node,
        "source": str(ROOT),
    }
    if target["gpus"]:
        gpu_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        if len(gpu_info.strip().splitlines()) < target["gpus"]:
            raise RuntimeError("Fewer GPUs than declared in targets.json")
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            raise RuntimeError(
                "GPU is busy; refusing to interrupt an existing workload"
            )
        metadata["gpus"] = gpu_info
        env.update(
            {
                "CUDA_HOME": target["cuda"],
                "CUDA_PATH": target["cuda"],
                "SM": target["sm"],
                # Command-line make variables override legacy Makefile := paths,
                # and propagate into recursive makes without mixing toolkits.
                "MAKEFLAGS": f"CUDA_HOME={target['cuda']} CUDA_PATH={target['cuda']} SM={target['sm']}",
                "TORCH_CUDA_ARCH_LIST": str(int(target["sm"]) // 10)
                + "."
                + target["sm"][-1],
                "UCCL_SOCKET_IFNAME": target["interface"],
                "NCCL_SOCKET_IFNAME": target["interface"],
                "GLOO_SOCKET_IFNAME": target["interface"],
            }
        )
        interfaces = json.loads(
            subprocess.check_output(
                ["ip", "-j", "-4", "addr", "show", "dev", target["interface"]]
            )
        )
        address = next(
            a["local"]
            for i in interfaces
            for a in i["addr_info"]
            if a["family"] == "inet"
        )
    else:
        address = "127.0.0.1"
    lock_name = (
        "host" if not target["gpus"] else ("cu128" if args.target == "l4" else "cu130")
    )
    lockfile = ROOT / f"ci/requirements-{lock_name}.lock"
    fingerprint = hashlib.sha256(
        lockfile.read_bytes() + sys.version.encode()
    ).hexdigest()[:20]
    env["CI_ENV_DIR"] = str(ci_root / "cache" / f"{lock_name}-{fingerprint}")
    env["CI_LOCK"] = str(lockfile)
    env["CI_DEPS"] = str(ci_root / "cache" / "dependencies")
    env["PATH"] = (
        env["CI_ENV_DIR"]
        + "/bin:"
        + env.get("CUDA_HOME", "/usr")
        + "/bin:"
        + env["PATH"]
    )
    env["PYTHONPATH"] = str(ROOT)
    env["PIP_CACHE_DIR"] = str(ci_root / "cache" / "pip")
    (artifacts / "machine.json").write_text(json.dumps(metadata, indent=2))
    with open(artifacts / "toolchain.txt", "w") as stream:
        for command in (
            ["uname", "-a"],
            ["g++", "--version"],
            ["ldd", "--version"],
            ["dpkg-query", "-W"],
            *(
                [[env["CUDA_HOME"] + "/bin/nvcc", "--version"]]
                if target["gpus"]
                else []
            ),
        ):
            subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
    initial = run(
        ["bash", "ci/setup.sh", "environment"],
        artifacts / "environment.log",
        1800,
        env,
        True,
    )
    if initial["exit_code"]:
        emit("fatal", **initial)
        return 1
    os.environ.update(env)
    emit("ready", address=address, artifacts=str(artifacts), metadata=metadata)
    request = receive()
    selected = {t["id"]: t for t in catalog["tests"]}
    built, broken, results = set(), {}, []
    p2p_profile = None
    for name in request["tests"]:
        test = selected[name]
        if not test["enabled"] or args.target not in test["targets"]:
            raise RuntimeError(f"Test {name} is not enabled on {args.target}")
        participating = args.node < test.get("nodes", 1)
        setup_error = None
        if participating:
            for component in test.get("setup", []):
                if component in {"p2p", "compression"} and component != p2p_profile:
                    built.discard(component)
                if component in broken:
                    setup_error = broken[component]
                    break
                if component not in built:
                    result = run(
                        ["bash", "ci/setup.sh", component],
                        artifacts / f"build-{component}.log",
                        2400,
                        env,
                        True,
                    )
                    if result["exit_code"]:
                        broken[component] = setup_error = result
                        break
                    built.add(component)
                    if component in {"p2p", "compression"}:
                        p2p_profile = component
        emit(
            "prepared",
            id=name,
            ok=setup_error is None,
            participating=participating,
            error=setup_error,
        )
        message = receive()
        if message.get("action") != "run" or setup_error:
            result = {
                "exit_code": 125,
                "error": setup_error or "peer setup failed",
                "seconds": 0,
            }
        elif not participating:
            result = {"exit_code": 0, "seconds": 0, "not_participating": True}
        else:
            test_env = {
                **env,
                **test.get("env", {}),
                "MASTER_ADDR": request["master_addr"],
                "MASTER_PORT": str(message["port"]),
                "EXCHANGER_PORT": str(message["port"] + 10000),
                "CI_NODES": str(test.get("nodes", 1)),
                "CI_GPUS": str(max(1, test.get("gpus", 1))),
                "CI_TEST_ID": name,
                "CI_TEST_JUNIT": str(artifacts / f"{name}.pytest.xml"),
            }
            result = run(
                [
                    "bash",
                    "-euo",
                    "pipefail",
                    "-c",
                    "source ci/env.sh\n" + test["command"],
                ],
                artifacts / f"{name}.log",
                test["timeout"],
                test_env,
                True,
                name,
            )
            if result["exit_code"] == 0:
                log_text = Path(result["log"]).read_text(errors="replace")
                for pattern in test.get("failure_patterns", []):
                    if re.search(pattern, log_text):
                        result.update(
                            exit_code=1, error=f"Fixture reported failure: {pattern}"
                        )
                        break
            junit = Path(test_env["CI_TEST_JUNIT"])
            if junit.exists():
                cases = ET.parse(junit).getroot().findall(".//testcase")
                result["cases"] = len(cases)
                result["skipped"] = sum(c.find("skipped") is not None for c in cases)
                if result["exit_code"] == 0 and (
                    not cases or result["skipped"] == len(cases)
                ):
                    result.update(
                        exit_code=1,
                        error="All selected pytest cases were skipped or no cases ran",
                    )
        result.update(id=name, tier=test["tier"], node=args.node, target=args.target)
        results.append(result)
        (artifacts / "results.json").write_text(json.dumps(results, indent=2))
        emit("result", **result)
    emit("complete")
    return int(any(r["exit_code"] for r in results))


if __name__ == "__main__":
    # Raising lets process.run's finally terminate the owned child group.
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as error:
        emit("fatal", error=str(error), traceback=traceback.format_exc())
        sys.exit(1)

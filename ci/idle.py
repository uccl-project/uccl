"""Conservative lab admission checks; never clear an unrelated workload."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

# Login/session plumbing is not a submitted job. Do not inspect or log
# command-line secrets. CPU-only workloads are detected from sampled activity.
SESSION_PROCESSES = {
    "bash",
    "sh",
    "dash",
    "zsh",
    "fish",
    "sshd",
    "sshd-session",
    "sshd-auth",
    "systemd",
    "(sd-pam)",
    "dbus-daemon",
    "dbus-broker",
    "dbus-broker-lau",
    "tmux: server",
    "tmux: client",
    "screen",
    "login",
}
ROOT_SERVICES = SESSION_PROCESSES | {
    "containerd",
    "containerd-shim",
    "dockerd",
    "cron",
    "crond",
    "systemd-journal",
    "systemd-udevd",
    "systemd-logind",
    "NetworkManager",
    "irqbalance",
    "rsyslogd",
    "nvidia-persiste",
    "nvidia-fabricma",
    "nv-hostengine",
    "multipathd",
    "landscape-sysin",  # Ubuntu login/MOTD status helper (comm is truncated).
}


def processes(proc=Path("/proc")):
    result = {}
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            status = (path / "status").read_text()
            uid = int(re.search(r"^Uid:\s+\d+\s+(\d+)", status, re.M)[1])
            result[int(path.name)] = dict(
                pid=int(path.name),
                ppid=int(fields[1]),
                pgid=int(fields[2]),
                uid=uid,
                start=fields[19],
                ticks=int(fields[11]) + int(fields[12]),
                state=fields[0],
                kernel=bool(int(fields[6]) & 0x00200000),
                name=stat[stat.find("(") + 1 : stat.rfind(")")],
            )
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited during the snapshot.
    return result


def owned_processes(snapshot, pid=None):
    pid = os.getpid() if pid is None else pid
    own = {pid}
    while True:
        children = {p for p, info in snapshot.items() if info["ppid"] in own}
        if children <= own:
            break
        own |= children
    while pid in snapshot and snapshot[pid]["ppid"] not in own:
        pid = snapshot[pid]["ppid"]
        own.add(pid)
    return own


def user_jobs(snapshot, own, previous=None, elapsed=0):
    jobs = []
    for pid, p in snapshot.items():
        if pid in own or p["kernel"] or p["state"] == "Z":
            continue
        before = (previous or {}).get(pid)
        active = p["state"] == "D"
        if before and before["start"] == p["start"] and elapsed > 0:
            # One process consuming >=20% of one CPU is enough; do not divide
            # by node core count and miss a single-core job on a large server.
            active |= (p["ticks"] - before["ticks"]) / os.sysconf(
                "SC_CLK_TCK"
            ) / elapsed >= 0.2
        else:
            active |= p["state"] == "R"
        # Generic interpreters and notebook servers can be idle services.
        # Explicit distributed launchers still reserve a node while waiting.
        launcher = re.match(r"^(torchrun|pt_elastic|mpirun|mpiexec|ray::)", p["name"])
        user_job = (
            (1000 <= p["uid"] < 65534 or p["uid"] == os.getuid())
            and p["name"] not in SESSION_PROCESSES
            and (active or launcher)
        )
        root_job = (
            p["uid"] == 0 and p["name"] not in ROOT_SERVICES and (active or launcher)
        )
        if user_job or root_job:
            jobs.append(f"PID {pid} ({p['name']}, uid {p['uid']})")
    return jobs


def query(command):
    return subprocess.check_output(
        command, text=True, stderr=subprocess.PIPE, timeout=10
    )


def gpu_jobs(target, own, settle=True):
    if not target.get("gpus"):
        # A host-only suite may be directed to a lab GPU node with --hosts.
        if shutil.which("nvidia-smi"):
            target = dict(gpus=1, capabilities=["cuda"])
        elif shutil.which("rocm-smi"):
            target = dict(gpus=1, capabilities=["rocm"])
        elif Path("/sys/module/nvidia").exists() or Path("/sys/module/amdgpu").exists():
            raise ValueError("GPU driver exists but activity tooling is missing")
        else:
            return []
    caps = target.get("capabilities", [])
    reasons, pids, usage = [], set(), []
    if "cuda" in caps:
        # XML includes graphics and compute contexts, not just utilization.
        devices = ET.fromstring(query(["nvidia-smi", "-q", "-x"])).findall("gpu")
        for device in devices:
            contexts = device.find("processes")
            if contexts is None:
                raise ValueError("NVIDIA process information unavailable")
            for process in contexts.findall("process_info"):
                if process.findtext("type") != "G":
                    pids.add(int(process.findtext("pid")))
            usage.append(
                float(device.findtext("utilization/gpu_util").replace("%", ""))
            )
    elif "rocm" in caps:
        # Retain the legacy AMD helper's KFD process check, without pkill or sudo.
        output = query(["rocm-smi", "--showpidgpus"])
        pids = set(map(int, re.findall(r"\bPID\s+(\d+)\s+is using", output)))
        if not pids and "No KFD PIDs currently running" not in output:
            raise ValueError("Unrecognized ROCm process information")
        devices = json.loads(query(["rocm-smi", "--showuse", "--json"]))
        usage = [
            float(v["GPU use (%)"]) for k, v in devices.items() if k.startswith("card")
        ]
    else:
        raise ValueError("No activity probe for this accelerator")
    if len(usage) < target["gpus"] or any(not 0 <= value <= 100 for value in usage):
        raise ValueError("GPU activity information is incomplete")
    reasons.extend(f"GPU process PID {pid}" for pid in sorted(pids - own))
    # Our currently running CI child explains utilization; foreign contexts
    # are still checked above. Before launch, even unattributed activity skips.
    if not pids.intersection(own) and any(value > 0 for value in usage):
        if settle and not reasons:
            # Utilization samples can still include the CI kernel that just
            # exited. Recheck once; a live foreign context never gets this grace.
            time.sleep(1.1)
            return gpu_jobs(target, own, settle=False)
        reasons.append("GPU utilization is nonzero")
    return reasons


def record_process(pid, env):
    if not env or not env.get("CI_ROOT") or not env.get("CI_REPO"):
        return None
    snapshot = processes()
    if pid not in snapshot:
        return None
    directory = Path(env["CI_ROOT"]) / "owned-processes"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = directory / f"{pid}.json"
    data = {**snapshot[pid], "run": env["CI_REPO"]}
    record.write_text(json.dumps(data))
    record.chmod(0o600)
    return record


def cleanup_stale(root):
    """Caller holds the machine lease. Reap only recorded, same-UID CI groups."""
    cleaned = []
    for record in (root / "owned-processes").glob("*.json"):
        if record.stat().st_uid != os.getuid():
            raise ValueError("CI process record has an unexpected owner")
        data = json.loads(record.read_text())
        run = Path(data["run"]).resolve()
        if not run.is_relative_to((root / "runs").resolve()):
            raise ValueError("CI process record points outside CI run storage")
        snapshot = processes()
        current = snapshot.get(data["pid"])
        if current is None or current["start"] != data["start"]:
            record.unlink()  # Exited, or PID was reused: never signal it.
            continue
        if current["uid"] != os.getuid() or current["pgid"] != current["pid"]:
            raise ValueError("Cannot verify abandoned CI process ownership")
        members = [p for p in snapshot.values() if p["pgid"] == current["pgid"]]
        if any(p["uid"] != os.getuid() for p in members):
            raise ValueError("CI process group has an unexpected owner")
        identities = {p["pid"]: p["start"] for p in members}
        try:
            os.killpg(current["pgid"], signal.SIGTERM)
        except ProcessLookupError:
            record.unlink()
            continue
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            remaining = [
                p
                for p in processes().values()
                if p["pgid"] == current["pgid"] and p["state"] != "Z"
            ]
            if not remaining:
                break
            time.sleep(0.1)
        remaining = [p for p in processes().values() if p["pgid"] == current["pgid"]]
        if remaining:
            # A surviving original member proves the process group has not
            # been recycled, even when its leader already exited on SIGTERM.
            if not any(
                identities.get(p["pid"]) == p["start"] for p in remaining
            ) or any(p["uid"] != os.getuid() for p in remaining):
                raise ValueError("Cannot verify surviving CI process group")
            try:
                os.killpg(current["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        cleaned.append(current["pid"])
        record.unlink()
    return cleaned


def probe(target, root, lease=True, cleanup=False):
    """cleanup=True also reaps stale groups under the worker's existing lease."""
    root = Path(root)
    lock = None
    cleaned = []
    try:
        if lease:
            lock = open(root / "hardware.lock", "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {
                    "state": "busy",
                    "reason": "Another CI run holds the machine lease",
                }
        if lease or cleanup:
            cleaned = cleanup_stale(root)
        mounts = Path("/proc/mounts").read_text()
        if re.search(r"(?:^|,)hidepid=(?!0(?:,|\s))", mounts):
            raise ValueError("Process visibility is restricted by hidepid")
        previous = processes()
        started = time.monotonic()
        time.sleep(0.25)
        snapshot = processes()
        elapsed = time.monotonic() - started
        own = owned_processes(snapshot)
        # GPU queries may block or wait for utilization to settle. They are
        # outside the CPU sampling interval and must not dilute CPU activity.
        reasons = gpu_jobs(target, own) + user_jobs(snapshot, own, previous, elapsed)
        return {
            "state": "busy" if reasons else "idle",
            "reason": "; ".join(reasons[:12]),
            "cleaned_ci_pids": cleaned,
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        subprocess.SubprocessError,
        ET.ParseError,
    ) as error:
        # Unknown activity is never permission to disturb a machine.
        return {
            "state": "unavailable",
            "reason": f"Cannot verify idle node: {type(error).__name__}",
            "cleaned_ci_pids": cleaned,
        }
    finally:
        if lock:
            lock.close()


def skip_reason(result):
    return result["reason"] if result["state"] != "idle" else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--capabilities", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            probe(
                dict(gpus=args.gpus, capabilities=args.capabilities.split(",")),
                args.root,
            )
        )
    )

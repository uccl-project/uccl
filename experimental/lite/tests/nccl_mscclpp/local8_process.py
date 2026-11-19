"""Guard local GPU measurements; terminate only the measurement's process tree."""
import os
import json
from pathlib import Path
import signal
import subprocess
import time


def gpu_users():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(x) for x in text.split()}


def descendant(pid, root):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        try:
            status = Path(f'/proc/{pid}/status').read_text()
        except FileNotFoundError:
            return True  # Exited between the GPU and process snapshots.
        pid = int(next(line.split()[1] for line in status.splitlines()
                       if line.startswith('PPid:')))
    return False


def process_tree(root):
    # Track launchers before their CUDA contexts appear. During MPI error
    # teardown a rank can be reparented before NVML reports its final context.
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    parents = {int(pid): int(parent) for pid, parent in
               (line.split() for line in rows.splitlines())}
    result = {root}
    while True:
        children = {pid for pid, parent in parents.items() if parent in result}
        if children <= result:
            return result
        result.update(children)


def cpu_list(text):
    result = set()
    for group in text.strip().split(','):
        parts = list(map(int, group.split('-')))
        result.update(range(parts[0], parts[-1] + 1))
    return result


def worker_affinity(users):
    """Check real OS affinity after all eight ranks finish preparation."""
    if len(users) != 8:
        return None
    primary = {6 * rank + offset for rank in range(8) for offset in (4, 5)}
    reserved = set()
    for cpu in primary:
        reserved.update(cpu_list(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text()))
    roles = {'uccl_rs_worker', 'uccl_ar_lane'}
    processes, seen = [], set()
    try:
        for pid in sorted(users):
            threads = []
            for task in Path(f'/proc/{pid}/task').iterdir():
                name = (task / 'comm').read_text().strip()
                status = (task / 'status').read_text().splitlines()
                cpus = cpu_list(next(line.split(':', 1)[1] for line in status
                                     if line.startswith('Cpus_allowed_list:')))
                threads.append(dict(tid=int(task.name), name=name, cpus=sorted(cpus)))
            if {t['name'] for t in threads if t['name'] in roles} != roles:
                return None
            processes.append(dict(pid=pid, threads=threads))
    except (FileNotFoundError, ProcessLookupError):
        return None
    for process in processes:
        for thread in process['threads']:
            cpus = set(thread['cpus'])
            if thread['name'] in roles:
                if len(cpus) != 1 or not cpus <= primary or seen & cpus:
                    raise RuntimeError('packet worker/helper CPU placement mismatch')
                seen.update(cpus)
            elif cpus & reserved:
                raise RuntimeError('application/CUDA thread can preempt a reserved worker core')
    if seen != primary:
        raise RuntimeError('incomplete worker/helper placement')
    return dict(reserved_cpus=sorted(reserved), processes=processes)


def wait(child, timeout, log_path=None, affinity_path=None):
    start = time.monotonic()
    owned = set()
    affinity_recorded = False
    try:
        while child.poll() is None:
            owned.update(process_tree(child.pid))
            users = gpu_users()
            foreign = {pid for pid in users if pid not in owned and not descendant(pid, child.pid)}
            if foreign:
                raise RuntimeError(f'foreign GPU work during measurement: {sorted(foreign)}')
            owned.update(users)
            if affinity_path and not affinity_recorded and log_path and Path(log_path).read_text().count(' isolated CPU=') >= 8:
                snapshot = worker_affinity(users)
                if snapshot:
                    Path(affinity_path).write_text(json.dumps(snapshot, indent=2) + '\n')
                    affinity_recorded = True
            if time.monotonic() - start > timeout:
                raise TimeoutError('measurement timeout')
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        # Open MPI can return immediately after signaling failed ranks. Their
        # CUDA contexts may take a moment to disappear from NVML. Wait only
        # for ranks observed as our descendants, never for unrelated GPU work.
        deadline = time.monotonic() + 10
        while users := gpu_users():
            if users - owned or time.monotonic() >= deadline:
                raise RuntimeError('GPU processes remain after measurement')
            time.sleep(.2)
        return child.returncode
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise

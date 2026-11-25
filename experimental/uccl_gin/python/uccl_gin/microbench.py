from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


_ENV_EXPORTS = (
    "LD_LIBRARY_PATH",
    "NCCL_NET_PLUGIN",
    "FI_PROVIDER",
    "FI_EFA_USE_DEVICE_RDMA",
    "OFI_NCCL_FORCE_NUM_RAILS",
    "NCCL_SOCKET_IFNAME",
    "LOCAL_WORLD_SIZE",
)


@dataclass(frozen=True)
class MicrobenchConfig:
    root: Path
    hosts: str
    local_world_size: int = 8
    sizes: Sequence[int] = (1024, 4096, 65536)
    iters: int = 10
    warmup: int = 2
    timeout_s: int = 120
    mpirun: str = "/opt/amazon/openmpi/bin/mpirun"

    @classmethod
    def from_env(cls, root: Path) -> "MicrobenchConfig":
        hosts = os.environ.get("UCCL_GIN_MPI_HOSTS")
        if not hosts:
            raise ValueError("set UCCL_GIN_MPI_HOSTS=node0:8,node1:8")
        sizes = tuple(
            int(x) for x in os.environ.get("UCCL_GIN_TEST_SIZES", "1024,4096,65536").split(",")
            if x
        )
        return cls(
            root=root,
            hosts=hosts,
            local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "8")),
            sizes=sizes,
            iters=int(os.environ.get("UCCL_GIN_TEST_ITERS", "10")),
            warmup=int(os.environ.get("UCCL_GIN_TEST_WARMUP", "2")),
            timeout_s=int(os.environ.get("UCCL_GIN_TEST_TIMEOUT", "120")),
            mpirun=os.environ.get("UCCL_GIN_MPIRUN", "/opt/amazon/openmpi/bin/mpirun"),
        )

    @property
    def binary(self) -> Path:
        return self.root / "build" / "uccl_gin_microbench"

    def command(self, env: Mapping[str, str] | None = None) -> list[str]:
        env = env or os.environ
        if not self.binary.exists():
            raise FileNotFoundError(f"missing binary: {self.binary}")
        cmd = [
            self.mpirun,
            "--host",
            self.hosts,
            "-np",
            str(self.local_world_size * 2),
            "-npernode",
            str(self.local_world_size),
        ]
        for key in _ENV_EXPORTS:
            if key in env:
                cmd.extend(["-x", key])
        cmd.extend(
            [
                str(self.binary),
                "--sizes",
                ",".join(str(x) for x in self.sizes),
                "--iters",
                str(self.iters),
                "--warmup",
                str(self.warmup),
            ]
        )
        return cmd


@dataclass(frozen=True)
class MicrobenchResult:
    command: Sequence[str]
    returncode: int
    stdout: str

    def assert_correct(self) -> None:
        if self.returncode != 0:
            raise AssertionError(self.stdout)
        required = ("all correctness PASS", "UCCL-put/add", "UCCL-tail/q",
                     "UCCL-put+q", "UCCL-red_add counter")
        missing = [needle for needle in required if needle not in self.stdout]
        if missing:
            raise AssertionError(
                "microbench output missing required correctness markers: "
                + ", ".join(missing)
                + "\n"
                + self.stdout
            )


def run_microbench(config: MicrobenchConfig, env: Mapping[str, str] | None = None) -> MicrobenchResult:
    env = dict(env or os.environ)
    cmd = config.command(env)
    proc = subprocess.run(
        cmd,
        cwd=config.root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=config.timeout_s,
    )
    return MicrobenchResult(command=cmd, returncode=proc.returncode, stdout=proc.stdout)


def main() -> int:
    root = Path(os.environ.get("UCCL_GIN_ROOT", Path.cwd())).resolve()
    result = run_microbench(MicrobenchConfig.from_env(root))
    print(result.stdout)
    result.assert_correct()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

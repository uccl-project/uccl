"""Explicit two-core-per-rank placement using the common local launch boundary."""
from pathlib import Path
import local_core_profile as core
import local_numa_profile as numa


def record(real, library):
    return core.record(real, library, cores_per_rank=2, launcher=Path(__file__))


def make_record(real, library, output):
    return core.make_record(real, library, output, cores_per_rank=2, launcher=Path(__file__))


def validate_run(directory, command, log):
    return core.validate_run(directory, command, log, cores_per_rank=2, launcher=Path(__file__))


if __name__ == '__main__':
    numa.record = record
    numa.main()

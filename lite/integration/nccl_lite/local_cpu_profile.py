"""Select an explicit local CPU placement; existing launch modes are unchanged."""
from pathlib import Path
import local_core_profile as core
import local_core2_profile as core2
import local_numa_profile as numa


def launcher_path(mode):
    if mode == 'core2':
        return Path(core2.__file__).resolve()
    return Path(core.__file__).resolve() if mode == 'core' else numa.launcher_path(mode)


def make_record(mode, real, library, output):
    if mode == 'core2':
        return core2.make_record(real, library, output)
    if mode == 'core':
        return core.make_record(real, library, output)
    return numa.make_record(mode, real, library, output)


def validate_run(directory, mode, command, log):
    if mode == 'core2':
        return core2.validate_run(directory, command, log)
    if mode == 'core':
        return core.validate_run(directory, command, log)
    return numa.validate_run(directory, mode, command, log)

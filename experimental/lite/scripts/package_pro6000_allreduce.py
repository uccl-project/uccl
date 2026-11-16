#!/usr/bin/env python3
"""Deterministically archive collected evidence, excluding generated binaries."""
import argparse
import gzip
from pathlib import Path
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("xb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in sorted(a.input.rglob("*")):
                if path.is_dir():
                    continue
                if path.is_symlink() or path.suffix in (
                    ".o",
                    ".so",
                    ".a",
                    ".d",
                    ".pyc",
                ):
                    raise ValueError("Unexpected generated artifact: " + str(path))
                info = archive.gettarinfo(str(path), str(path.relative_to(a.input)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
print(a.output)

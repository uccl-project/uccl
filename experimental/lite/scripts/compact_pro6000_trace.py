#!/usr/bin/env python3
"""Preserve exact GPU kernel/payload rows without huge CUDA API polling traces."""
import argparse
import json
from pathlib import Path
import sqlite3

from analyze_pro6000_timeline import file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Refusing to overwrite compact trace")
    source = sqlite3.connect(f"file:{a.input.resolve()}?mode=ro", uri=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = sqlite3.connect(a.output)
    manifest = dict(
        source_name=a.input.name, source_sha256=file_sha(a.input), tables={}
    )
    for table in (
        "StringIds",
        "TARGET_INFO_GPU",
        "DIAGNOSTIC_EVENT",
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
    ):
        schema = source.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()[0]
        out.execute(schema)
        query = "SELECT * FROM " + table
        if table == "CUPTI_ACTIVITY_KIND_MEMCPY":
            query += " WHERE srcDeviceId BETWEEN 0 AND 7 AND dstDeviceId BETWEEN 0 AND 7 AND srcDeviceId!=dstDeviceId AND bytes>8"
        columns = len(source.execute("PRAGMA table_info(" + table + ")").fetchall())
        out.executemany(
            "INSERT INTO "
            + table
            + " VALUES ("
            + ",".join("?" for _ in range(columns))
            + ")",
            source.execute(query),
        )
        manifest["tables"][table] = dict(
            query=query, rows=out.execute("SELECT count(*) FROM " + table).fetchone()[0]
        )
    out.commit()
    out.close()
    manifest["output_sha256"] = file_sha(a.output)
    a.output.with_suffix(".projection.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

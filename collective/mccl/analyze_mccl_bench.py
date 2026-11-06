#!/usr/bin/env python3
"""Analyze one paired manifest; never infer hardware acceptance from logs."""

import argparse
import json

from mccl_bench import summarize, validate_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=argparse.FileType("r"))
    args = parser.parse_args()
    with args.manifest as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or manifest.get("state") == "failed":
        parser.error("unsupported or failed run manifest")
    try:
        expected = {
            (collective, backend, run)
            for collective in manifest["collectives"]
            for backend in ("native", "uccl")
            for run in range(manifest["repeats"])
        }
        actual = [
            (record["collective"], record["backend"], record["run"])
            for record in manifest["records"]
        ]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("manifest is missing complete paired runs")
        for record in manifest["records"]:
            validate_coverage(
                record["rows"], manifest["expected_sizes"][record["collective"]]
            )
        comparison = summarize(manifest["records"])
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "state": "unverified",
                "note": "paired bench aggregates, not per-rank tails or a hardware approval",
                "source": manifest["source"],
                "plugin_sha256": manifest["plugin_sha256"],
                "comparison": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

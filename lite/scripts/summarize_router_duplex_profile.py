#!/usr/bin/env python3
"""Summarize simultaneous H2D/D2H activity from Nsight Systems SQLite."""

import argparse
import csv
import sqlite3


def summarize(database, label):
    connection = sqlite3.connect(database)
    rows = []
    devices = [row[0] for row in connection.execute(
        "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "WHERE copyKind IN (1,2) ORDER BY deviceId"
    )]
    for device in devices:
        endpoints = []
        copies = {1: 0, 2: 0}
        byte_count = {1: 0, 2: 0}
        for start, end, kind, size in connection.execute(
            "SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE deviceId=? AND copyKind IN (1,2)", (device,)
        ):
            endpoints.extend(((start, kind, 1), (end, kind, -1)))
            copies[kind] += 1
            byte_count[kind] += size
        endpoints.sort(key=lambda point: (point[0], point[2]))
        assert endpoints
        active = {1: 0, 2: 0}
        duration = {1: 0, 2: 0, "both": 0}
        previous = endpoints[0][0]
        for timestamp, kind, delta in endpoints:
            elapsed = timestamp - previous
            for direction in (1, 2):
                if active[direction]:
                    duration[direction] += elapsed
            if active[1] and active[2]:
                duration["both"] += elapsed
            active[kind] += delta
            previous = timestamp
        assert not active[1] and not active[2]
        rows.append((label, device, copies[2], copies[1], byte_count[2],
                     byte_count[1], duration[2] / 1e6, duration[1] / 1e6,
                     duration["both"] / 1e6,
                     duration["both"] / duration[2]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("profiles", nargs="+", help="label=SQLite path")
    args = parser.parse_args()
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("profile", "gpu", "dtoh_copies", "htod_copies",
                         "dtoh_bytes", "htod_bytes", "dtoh_active_ms",
                         "htod_active_ms", "duplex_ms", "dtoh_overlap_fraction"))
        for profile in args.profiles:
            label, database = profile.split("=", 1)
            writer.writerows(summarize(database, label))


if __name__ == "__main__":
    main()

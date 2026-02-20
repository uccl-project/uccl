#!/usr/bin/env python3
import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MSG_RE = re.compile(
    r"\[RDMA MSG dispatch\]\s+th=(?P<th>\d+)\s+src=(?P<src>\d+)\s+dst=(?P<dst>\d+)\s+msg_bytes=(?P<msg>\d+)"
)
POST_RE = re.compile(
    r"\[RDMA POST dispatch-only\]\s+th=(?P<th>\d+)\s+src=(?P<src>\d+)\s+dst=(?P<dst>\d+)\s+dispatch_wrs=(?P<wrs>\d+)\s+dispatch_bytes=(?P<bytes>\d+)"
)


def parse_message_sizes(path: Path):
    data = []
    source_mode = None

    for line in path.read_text().splitlines():
        m = MSG_RE.search(line)
        if m:
            source_mode = "msg_bytes"
            data.append((int(m.group("dst")), int(m.group("msg"))))

    if data:
        return data, source_mode

    # Fallback for older logs: use per-post average when per-message logs are absent.
    for line in path.read_text().splitlines():
        m = POST_RE.search(line)
        if not m:
            continue
        wrs = int(m.group("wrs"))
        if wrs <= 0:
            continue
        source_mode = "dispatch_bytes/dispatch_wrs"
        msg_size = int(round(int(m.group("bytes")) / wrs))
        data.append((int(m.group("dst")), msg_size))

    return data, source_mode


def percentile(values, p):
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, p))


def format_stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "n": arr.size,
        "min": int(arr.min()),
        "p50": int(percentile(arr, 50)),
        "p90": int(percentile(arr, 90)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot RDMA dispatch message-size breakdown from logs."
    )
    parser.add_argument(
        "--w-batch",
        default="ep/bench/logs/w_batch_data.txt",
        help="Path to with-batch log file.",
    )
    parser.add_argument(
        "--wo-batch",
        default="ep/bench/logs/wo_batch_data.txt",
        help="Path to without-batch log file.",
    )
    parser.add_argument(
        "--out",
        default="ep/bench/logs/dispatch_message_size_breakdown.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=14352,
        help="Bucket width in bytes for message-size histogram.",
    )
    args = parser.parse_args()

    w_path = Path(args.w_batch)
    wo_path = Path(args.wo_batch)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    series = {}
    modes = {}
    by_dst = {}
    for name, path in [("w_batch", w_path), ("wo_batch", wo_path)]:
        parsed, mode = parse_message_sizes(path)
        if not parsed:
            raise ValueError(f"No parseable dispatch entries found in {path}")
        modes[name] = mode
        sizes = [v for _, v in parsed]
        series[name] = sizes
        dst_map = defaultdict(list)
        for dst, v in parsed:
            dst_map[dst].append(v)
        by_dst[name] = {d: float(np.mean(vs)) for d, vs in sorted(dst_map.items())}

    # Bucketize by fixed byte range and render side-by-side bars.
    bucket_size = max(1, int(args.bucket_size))
    max_size = max(max(series["w_batch"]), max(series["wo_batch"]))
    num_buckets = int(math.ceil(max_size / bucket_size))
    bins = np.arange(0, (num_buckets + 1) * bucket_size + 1, bucket_size)
    hist_wo, _ = np.histogram(series["wo_batch"], bins=bins)
    hist_w, _ = np.histogram(series["w_batch"], bins=bins)
    bucket_indices = np.arange(len(bins) - 1)
    bar_width = 0.82
    bucket_labels = [f"{int(bins[i])//1024}-{int(bins[i+1])//1024}KB" for i in range(len(bins) - 1)]

    fig = plt.figure(figsize=(16, 10), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.2)
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, :])

    ax_top_left.bar(
        bucket_indices,
        hist_wo,
        width=bar_width,
        color="#ff7f0e",
        alpha=0.85,
    )
    ax_top_left.set_title(f"without batch message-size buckets (total WR={len(series['wo_batch'])})")
    ax_top_left.set_xlabel("message-size bucket")
    ax_top_left.set_ylabel("count")
    ax_top_left.set_xticks(bucket_indices)
    ax_top_left.set_xticklabels(bucket_labels, rotation=30, ha="right")
    ax_top_left.grid(axis="y", alpha=0.25)
    ax_top_left.set_xlim(-0.5, len(bucket_labels) - 0.5)
    ax_top_left.text(
        0.98,
        0.95,
        f"Total WR: {len(series['wo_batch'])}",
        transform=ax_top_left.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    ax_top_right.bar(
        bucket_indices, hist_w, width=bar_width, color="#1f77b4", alpha=0.85
    )
    ax_top_right.set_title(f"with batch message-size buckets (total WR={len(series['w_batch'])})")
    ax_top_right.set_xlabel("message-size bucket")
    ax_top_right.set_ylabel("count")
    ax_top_right.set_xticks(bucket_indices)
    ax_top_right.set_xticklabels(bucket_labels, rotation=30, ha="right")
    ax_top_right.grid(axis="y", alpha=0.25)
    ax_top_right.set_xlim(-0.5, len(bucket_labels) - 0.5)
    ax_top_right.text(
        0.98,
        0.95,
        f"Total WR: {len(series['w_batch'])}",
        transform=ax_top_right.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    all_dsts = sorted(set(by_dst["w_batch"]) | set(by_dst["wo_batch"]))
    x = np.arange(len(all_dsts))
    width = 0.38
    vals_w = [by_dst["w_batch"].get(d, np.nan) for d in all_dsts]
    vals_wo = [by_dst["wo_batch"].get(d, np.nan) for d in all_dsts]
    ax_bottom.bar(x - width / 2, vals_wo, width=width, label="without_batch", color="#ff7f0e")
    ax_bottom.bar(x + width / 2, vals_w, width=width, label="with_batch", color="#1f77b4")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([str(d) for d in all_dsts])
    ax_bottom.set_title("Average message size by dst rank")
    ax_bottom.set_xlabel("dst rank")
    ax_bottom.set_ylabel("avg bytes/message")
    ax_bottom.grid(axis="y", alpha=0.25)
    ax_bottom.legend()

    fig.suptitle(
        f"Dispatch message-size breakdown (without batch WR={len(series['wo_batch'])}, with batch WR={len(series['w_batch'])})"
    )
    fig.savefig(out_path)

    print(f"Saved figure: {out_path}")
    for name in ["w_batch", "wo_batch"]:
        st = format_stats(series[name])
        print(
            f"{name}: mode={modes[name]}, n={st['n']}, min={st['min']}, "
            f"p50={st['p50']}, p90={st['p90']}, max={st['max']}, mean={st['mean']:.1f}"
        )


if __name__ == "__main__":
    main()

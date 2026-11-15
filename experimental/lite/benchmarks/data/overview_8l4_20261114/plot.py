"""Plot six four-series, two-panel eight-L4 collective overviews."""
import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA.parents[2] / "scripts"))
from figure_style import clean_axis, configure, save

OPERATIONS = (("all_reduce", "allreduce", 14 / 8),
              ("all_gather", "allgather", 7 / 8),
              ("reduce_scatter", "reducescatter", 7 / 8))


def load():
    with (DATA / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["testbed"], row["collective"], int(row["bytes"])): row for row in rows}
    expected = {(mode, operation, 1 << power) for mode in ("gpu", "shm")
                for operation, *_ in OPERATIONS for power in range(10, 31)}
    if len(result) != len(rows) or set(result) != expected:
        raise ValueError("incomplete eight-L4 overview")
    tuned_path = DATA.parent / "nccl_tuned_testbeds12_20261115" / "summary.csv"
    with tuned_path.open() as stream:
        tuned_rows = list(csv.DictReader(stream))
    tuned = {(r["testbed"], r["collective"], int(r["bytes"]), r["engine"]): r
             for r in tuned_rows}
    expected_tuned = {(*key, engine) for key in expected for engine in ("ce", "sm")}
    if len(tuned) != len(tuned_rows) or set(tuned) != expected_tuned:
        raise ValueError("incomplete independently validated NCCL CE/SM curves")
    for key in expected:
        result[key] = dict(result[key])
        for engine in ("ce", "sm"):
            for statistic in ("median_us", "min_run_us", "max_run_us"):
                result[key][f"nccl_{engine}_{statistic}"] = tuned[(*key, engine)][statistic]
    return result


def make_figure(data, mode, operation, factor):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.74, bottom=0.27, wspace=0.28)
    title = dict(all_reduce="AllReduce", all_gather="AllGather",
                 reduce_scatter="ReduceScatter")[operation]
    testbed = 1 if mode == "gpu" else 2
    fig.suptitle(f"Testbed {testbed}: 8×L4 · {title}", y=0.98,
                 fontsize=15, fontweight="bold")
    for ax, sizes, ylabel, ticks, throughput in (
        (axes[0], [1 << power for power in range(10, 21)], "Latency (µs)",
         [1 << power for power in (10, 12, 14, 16, 18, 20)], False),
        (axes[1], [1 << power for power in range(20, 31)], "Bus throughput (GB/s)",
         [1 << power for power in (20, 22, 24, 26, 28, 30)], True),
    ):
        clean_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ax.set_xticks(ticks, [f"{size >> 10}K" if size < 1 << 20 else
                             f"{size >> 20}M" if size < 1 << 30 else "1G"
                             for size in ticks])
        ax.set_xlabel("Complete tensor bytes per rank")
        ax.set_ylabel(ylabel)
        upper = 0.0
        for field, label, color, marker, linestyle in (
            ("nccl_median_us", "NCCL (default)", "#D8A17A", "s", (0, (5, 2.5))),
            ("nccl_sm_median_us", "NCCL SM (tuned)", "#82B4A3", "D", (0, (2, 1.5))),
            ("nccl_ce_median_us", "NCCL CE (tuned)", "#B899C8", "^", (0, (6, 2, 1, 2))),
            ("uccl_median_us", "UCCL Lite", "#0068B5", "o", "-"),
        ):
            latencies = [float(data[mode, operation, size][field]) for size in sizes]
            if any(not math.isfinite(us) or us <= 0 for us in latencies):
                raise ValueError(f"invalid {mode} {operation} timing")
            values = [factor * size / us / 1000 if throughput else us
                      for size, us in zip(sizes, latencies)]
            upper = max(upper, *values)
            if field in ("nccl_ce_median_us", "nccl_sm_median_us"):
                low_field = field.replace("median_us", "min_run_us")
                high_field = field.replace("median_us", "max_run_us")
                bounds = [[float(data[mode, operation, size][f])
                           for f in (low_field, high_field)] for size in sizes]
                bounds = [[factor * size / us / 1000 if throughput else us
                           for us in pair] for size, pair in zip(sizes, bounds)]
                low, high = [min(pair) for pair in bounds], [max(pair) for pair in bounds]
                upper = max(upper, *high)
                ax.fill_between(sizes, low, high, color=color, alpha=0.08, linewidth=0)
            is_uccl = field == "uccl_median_us"
            ax.plot(sizes, values, color=color, linewidth=3.6 if is_uccl else 2.2,
                    marker=marker, linestyle=linestyle, markersize=6.5 if is_uccl else 5,
                    markeredgecolor="white", markeredgewidth=0.7, label=label,
                    zorder=4 if is_uccl else 2)
        ax.set_ylim(0, upper * 1.13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False,
               fontsize=10.5, handlelength=2.4, columnspacing=1.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--testbed", choices=("gpu", "shm", "all"), default="all")
    parser.add_argument("--op", choices=("all_reduce", "all_gather", "reduce_scatter", "all"), default="all")
    args = parser.parse_args()
    configure()
    data = load()
    for mode in (("gpu", "shm") if args.testbed == "all" else (args.testbed,)):
        for operation, name, factor in OPERATIONS:
            if args.op not in ("all", operation):
                continue
            fig = make_figure(data, mode, operation, factor)
            prefix = "collectives-8l4-" + ("shm-" if mode == "shm" else "")
            save(fig, prefix + name + "-overview", args.preview)


if __name__ == "__main__":
    main()

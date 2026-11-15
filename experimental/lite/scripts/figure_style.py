"""Shared, deterministic style for UCCL Lite's data figures."""

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#205E9E"
TEAL = "#148D91"
AMBER = "#BF6C36"
PURPLE = "#8064A6"
BASELINE = "#8999A8"
SECOND_BASELINE = "#B3BEC8"
INK = "#172A3A"
MUTED = "#607486"
GRID = "#E6EDF2"


def configure(black_axes: bool = False) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": "#000000" if black_axes else MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "text.color": INK,
            "xtick.color": "#000000" if black_axes else MUTED,
            "ytick.color": "#000000" if black_axes else MUTED,
            "legend.labelcolor": "#000000" if black_axes else INK,
            "svg.fonttype": "none",
            "svg.hashsalt": "uccl-lite-figures-v2",
        }
    )


def heading(fig, title: str, subtitle: str, left: float = 0.095) -> None:
    fig.text(left, 0.965, title, va="top", fontsize=16, fontweight="bold", color=INK)
    fig.text(left, 0.89, subtitle, va="top", fontsize=10.5, color=MUTED)


def clean_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(axis="both", which="both", length=0, pad=7)


def metric_direction(ax, higher: bool = False) -> None:
    label = "↑ Higher is better" if higher else "↓ Lower is better"
    ax.text(1, 1.06, label, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, fontweight="bold", color="#0068B5", clip_on=False,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#EAF3FA",
                  "edgecolor": "none"})


def figure_metric_direction(fig, higher: bool = False, y: float = 0.90) -> None:
    """Label a figure whose panels all show the same metric direction."""
    label = "↑ Higher is better" if higher else "↓ Lower is better"
    fig.text(0.98, y, label, ha="right", va="bottom", fontsize=12,
             fontweight="bold", color="#0068B5",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": "#EAF3FA",
                   "edgecolor": "none"})


def collective_panels():
    """Keep labels readable when three collective figures share a README row."""
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 6.0))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.85, bottom=0.10,
                        hspace=0.72)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, min_n_ticks=3))
    metric_direction(axes[0])
    metric_direction(axes[1], higher=True)
    return fig, axes


def collective_legend(fig, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=11.5, handlelength=1.6, columnspacing=0.8,
               handletextpad=0.5)


def save(fig, name: str, preview: bool) -> None:
    output = ROOT / "docs/assets" / f"{name}.svg"
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None}, facecolor="white")
    output.write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    if preview:
        folder = ROOT / "build/figures"
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", dpi=160, facecolor="white")
        fig.savefig(
            folder / f"{name}.pdf",
            metadata={"CreationDate": None, "ModDate": None},
            facecolor="white",
        )
    plt.close(fig)
    print(output.relative_to(ROOT))

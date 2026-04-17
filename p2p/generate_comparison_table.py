#!/usr/bin/env python3
"""Generate a neat comparison table image for UCX vs UCCL benchmark results."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from the comparison
data = {
    "UCX": [
        {
            "Block Size": "128 KB",
            "Batch": 10000,
            "B/W (GB/s)": 11.06,
            "Avg Lat (μs)": 11.9,
            "Avg Post (μs)": 1099.6,
            "P99 Post (μs)": 1347.0,
            "Avg Tx (μs)": 117427.8,
            "P99 Tx (μs)": 117656.0,
        },
        {
            "Block Size": "256 KB",
            "Batch": 10000,
            "B/W (GB/s)": 11.02,
            "Avg Lat (μs)": 23.8,
            "Avg Post (μs)": 1097.4,
            "P99 Post (μs)": 1347.0,
            "Avg Tx (μs)": 236848.4,
            "P99 Tx (μs)": 255820.0,
        },
        {
            "Block Size": "512 KB",
            "Batch": 10000,
            "B/W (GB/s)": 11.53,
            "Avg Lat (μs)": 45.5,
            "Avg Post (μs)": 1130.5,
            "P99 Post (μs)": 1890.0,
            "Avg Tx (μs)": 453670.9,
            "P99 Tx (μs)": 562845.0,
        },
    ],
    "UCCL": [
        {
            "Block Size": "128 KB",
            "Batch": 10000,
            "B/W (GB/s)": 12.24,
            "Avg Lat (μs)": 10.7,
            "Avg Post (μs)": 1480.3,
            "P99 Post (μs)": 1754.0,
            "Avg Tx (μs)": 105613.7,
            "P99 Tx (μs)": 105783.0,
        },
        {
            "Block Size": "256 KB",
            "Batch": 10000,
            "B/W (GB/s)": 12.37,
            "Avg Lat (μs)": 21.2,
            "Avg Post (μs)": 1480.6,
            "P99 Post (μs)": 1882.0,
            "Avg Tx (μs)": 210450.1,
            "P99 Tx (μs)": 210678.0,
        },
        {
            "Block Size": "512 KB",
            "Batch": 10000,
            "B/W (GB/s)": 12.53,
            "Avg Lat (μs)": 41.9,
            "Avg Post (μs)": 1391.9,
            "P99 Post (μs)": 1695.0,
            "Avg Tx (μs)": 417144.4,
            "P99 Tx (μs)": 417266.0,
        },
    ],
}

# Colors
UCX_COLOR = "#4A90A4"  # Muted teal
UCCL_COLOR = "#E07B39"  # Warm orange
HEADER_BG = "#1E2A38"  # Dark navy
HEADER_TEXT = "#FFFFFF"  # White
ROW_LIGHT = "#F8F9FA"  # Light gray
ROW_DARK = "#E9ECEF"  # Slightly darker gray
TEXT_COLOR = "#2C3E50"  # Dark text
BORDER_COLOR = "#CBD5E0"  # Light border

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)

# Column headers and widths
columns = [
    "Framework",
    "Block Size",
    "Batch",
    "B/W\n(GB/s)",
    "Avg Lat\n(μs)",
    "Avg Post\n(μs)",
    "P99 Post\n(μs)",
    "Avg Tx\n(μs)",
    "P99 Tx\n(μs)",
]
col_widths = [1.3, 1.2, 1.0, 1.2, 1.2, 1.3, 1.3, 1.5, 1.5]
col_positions = [0.5]
for w in col_widths[:-1]:
    col_positions.append(col_positions[-1] + w)

# Title
ax.text(
    7,
    9.5,
    "UCX vs UCCL Performance Comparison",
    fontsize=18,
    fontweight="bold",
    ha="center",
    va="center",
    color=HEADER_BG,
    fontfamily="DejaVu Sans",
)
ax.text(
    7,
    9.0,
    "Batch Size: 10,000 | RDMA Point-to-Point Transfer",
    fontsize=11,
    ha="center",
    va="center",
    color="#6C757D",
    fontfamily="DejaVu Sans",
)

# Header row
header_y = 8.2
header_height = 0.6

# Header background
header_rect = mpatches.FancyBboxPatch(
    (0.3, header_y - header_height / 2),
    13.4,
    header_height,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor=HEADER_BG,
    edgecolor=HEADER_BG,
    linewidth=0,
)
ax.add_patch(header_rect)

# Header text
for i, (col, pos) in enumerate(zip(columns, col_positions)):
    ax.text(
        pos + col_widths[i] / 2,
        header_y,
        col,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        color=HEADER_TEXT,
        fontfamily="DejaVu Sans",
    )

# Data rows
row_height = 0.55
start_y = 7.4
row_idx = 0

for framework, rows in data.items():
    framework_color = UCX_COLOR if framework == "UCX" else UCCL_COLOR

    for i, row_data in enumerate(rows):
        y_pos = start_y - row_idx * row_height

        # Alternating row background
        bg_color = ROW_LIGHT if row_idx % 2 == 0 else ROW_DARK
        row_rect = mpatches.FancyBboxPatch(
            (0.3, y_pos - row_height / 2),
            13.4,
            row_height,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=bg_color,
            edgecolor=BORDER_COLOR,
            linewidth=0.5,
        )
        ax.add_patch(row_rect)

        # Framework indicator (colored bar on left)
        indicator = mpatches.FancyBboxPatch(
            (0.3, y_pos - row_height / 2),
            0.08,
            row_height,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=framework_color,
            edgecolor=framework_color,
        )
        ax.add_patch(indicator)

        # Data values
        values = [
            framework,
            row_data["Block Size"],
            f"{row_data['Batch']:,}",
            f"{row_data['B/W (GB/s)']:.2f}",
            f"{row_data['Avg Lat (μs)']:.1f}",
            f"{row_data['Avg Post (μs)']:,.1f}",
            f"{row_data['P99 Post (μs)']:,.1f}",
            f"{row_data['Avg Tx (μs)']:,.1f}",
            f"{row_data['P99 Tx (μs)']:,.1f}",
        ]

        for j, (val, pos) in enumerate(zip(values, col_positions)):
            # Framework column gets special styling
            if j == 0:
                fontweight = "bold"
                color = framework_color
            else:
                fontweight = "normal"
                color = TEXT_COLOR

            ax.text(
                pos + col_widths[j] / 2,
                y_pos,
                val,
                fontsize=9,
                ha="center",
                va="center",
                color=color,
                fontweight=fontweight,
                fontfamily="DejaVu Sans",
            )

        row_idx += 1

# Legend / Key metrics summary
summary_y = 3.8

# Summary box background
summary_rect = mpatches.FancyBboxPatch(
    (0.5, summary_y - 1.5),
    13.0,
    2.0,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor="#F1F5F9",
    edgecolor=BORDER_COLOR,
    linewidth=1,
)
ax.add_patch(summary_rect)

ax.text(
    7,
    summary_y + 0.2,
    "Key Insights",
    fontsize=12,
    fontweight="bold",
    ha="center",
    va="center",
    color=HEADER_BG,
    fontfamily="DejaVu Sans",
)

# Insights
insights = [
    ("↑ Bandwidth", "UCCL achieves 8-12% higher throughput"),
    ("↓ Latency", "UCCL shows 8-10% lower average latency"),
    ("↓ Tx Time", "UCCL reduces total transfer time by 8-26%"),
]

for i, (metric, desc) in enumerate(insights):
    x_base = 1.5 + i * 4.3
    ax.text(
        x_base,
        summary_y - 0.5,
        metric,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="center",
        color=UCCL_COLOR,
        fontfamily="DejaVu Sans",
    )
    ax.text(
        x_base,
        summary_y - 1.0,
        desc,
        fontsize=9,
        ha="left",
        va="center",
        color=TEXT_COLOR,
        fontfamily="DejaVu Sans",
    )

# Legend
legend_y = 1.8
ax.add_patch(
    mpatches.Rectangle(
        (5.5, legend_y - 0.15), 0.3, 0.3, facecolor=UCX_COLOR, edgecolor="none"
    )
)
ax.text(
    6.0,
    legend_y,
    "UCX",
    fontsize=10,
    ha="left",
    va="center",
    color=TEXT_COLOR,
    fontfamily="DejaVu Sans",
)
ax.add_patch(
    mpatches.Rectangle(
        (7.5, legend_y - 0.15), 0.3, 0.3, facecolor=UCCL_COLOR, edgecolor="none"
    )
)
ax.text(
    8.0,
    legend_y,
    "UCCL",
    fontsize=10,
    ha="left",
    va="center",
    color=TEXT_COLOR,
    fontfamily="DejaVu Sans",
)

# Footer
ax.text(
    7,
    1.0,
    "Benchmark: P2P Transfer | Batch Size: 10,000",
    fontsize=8,
    ha="center",
    va="center",
    color="#9CA3AF",
    fontfamily="DejaVu Sans",
    style="italic",
)

plt.tight_layout()
plt.savefig(
    "/Users/pravein/praveinResearch/uccl/p2p/comparison_table.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.savefig(
    "/Users/pravein/praveinResearch/uccl/p2p/comparison_table.pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print("✓ Saved comparison_table.png and comparison_table.pdf")
plt.show()

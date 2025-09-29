import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (Throughput in GB/s)
# ---------------------------
systems = ["UCCL-EP", "DeepEP", "UCCL-EP (EFA)"]

throughput_gbps = [42.5, 27.8, 10.9]

idx = np.arange(len(systems))

# ---------------------------
# Plot settings
# ---------------------------
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 3,
        "lines.markersize": 10,
    }
)

# ---------------------------
# Plot Throughput
# ---------------------------
plt.figure(figsize=(9, 6))
bars = plt.bar(idx, throughput_gbps, color=["#1f77b4", "#ff7f0e", "#1f77b4"])
plt.xticks(idx, systems, rotation=0)
plt.xlabel("System")
plt.ylabel("Throughput (GB/s)")
plt.grid(True, axis="y", linestyle="--", alpha=0.6)

# Annotate bars
for rect, val in zip(bars, throughput_gbps):
    h = rect.get_height()
    plt.text(
        rect.get_x() + rect.get_width() / 2.0,
        h,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=14,
    )

plt.tight_layout()
plt.savefig("throughput_comparison.png", dpi=300)

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (Latency in µs, incl. GEMM)
# ---------------------------
systems = [
    "UCCL-EP",
    "DeepEP",
    "UCCL-EP (EFA)"
]

lat_min_us = [507.94, 650.24, 1793.86]
lat_max_us = [540.83, 846.6, 2205.38]

# Compute average
lat_avg_us = np.mean([lat_min_us, lat_max_us], axis=0)

idx = np.arange(len(systems))

# ---------------------------
# Plot settings
# ---------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 3,
    "lines.markersize": 10
})

# ---------------------------
# Plot Average Latency
# ---------------------------
plt.figure(figsize=(9, 6))
bars = plt.bar(idx, lat_avg_us, color=["#1f77b4", "#ff7f0e", "#1f77b4"])
plt.xticks(idx, systems, rotation=0)
plt.xlabel("System")
plt.ylabel("Average Latency (µs)")
plt.grid(True, axis="y", linestyle="--", alpha=0.6)

# Annotate bars
for rect, val in zip(bars, lat_avg_us):
    h = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, h, f"{val:.2f}",
             ha="center", va="bottom", fontsize=14)

plt.tight_layout()
plt.savefig("latency_avg_comparison.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (avg latency in µs)
# ---------------------------
ep_labels = [2, 4, 8, 16]  # x-axis labels
x = np.arange(len(ep_labels))  # bar positions

preprocess = [1378.0, 1993.0, 3310.2, 6130.6]
all2all    = [229.2, 230.4, 232.0, 327.2]
reorganize = [7511.6, 7877.5, 8523.1, 9303.3]

# ---------------------------
# Plot settings
# ---------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

# ---------------------------
# Stacked Bar Plot
# ---------------------------
bar_width = 0.6
fig, ax = plt.subplots(figsize=(8,6))

p1 = ax.bar(x, preprocess, bar_width, label="Preprocess")
p2 = ax.bar(x, all2all, bar_width, bottom=preprocess, label="All2All")
p3 = ax.bar(x, reorganize, bar_width,
            bottom=np.array(preprocess) + np.array(all2all),
            label="Reorganize")

ax.set_xlabel("EP (Number of GPUs)")
ax.set_ylabel("Latency (µs)")
ax.set_xticks(x)
ax.set_xticklabels(ep_labels)

ax.legend()
ax.grid(True, linestyle="--", alpha=0.6, axis="y")
plt.tight_layout()

# Save / Show
plt.savefig("latency_breakdown_stacked.png", dpi=300)
plt.show()

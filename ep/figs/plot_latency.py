import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (Throughput in GB/s)
# ---------------------------
gpus = [16, 24, 32]  # 2, 3, 4 nodes

uccl_dispatch_tp = [2.788, 1.131, 0.7224]
uccl_combine_tp = [4.249, 1.941, 1.325]
pplx_dispatch_tp = [1.027, 0.809, 0.693]
pplx_combine_tp = [1.840, 1.324, 1.098]

# Averages
uccl_avg_tp = np.mean([uccl_dispatch_tp, uccl_combine_tp], axis=0)
pplx_avg_tp = np.mean([pplx_dispatch_tp, pplx_combine_tp], axis=0)

# ---------------------------
# Data (Latency in µs)
# ---------------------------
uccl_dispatch_lat = [2694, 6643, 10399]
uccl_combine_lat = [3421, 7491, 10968]
pplx_dispatch_lat = [7446, 9363, 10924]
pplx_combine_lat = [9841, 11095, 13380]

# Averages
uccl_avg_lat = np.mean([uccl_dispatch_lat, uccl_combine_lat], axis=0)
pplx_avg_lat = np.mean([pplx_dispatch_lat, pplx_combine_lat], axis=0)

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
# Plot Throughput
# ---------------------------
plt.figure(figsize=(8,6))
plt.plot(gpus, uccl_avg_tp, marker="o", label="UCCL-EP (Avg)")
plt.plot(gpus, pplx_avg_tp, marker="s", label="Pplx (Avg)")
plt.xlabel("Number of GPUs")
plt.ylabel("Throughput (GB/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("throughput_vs_gpus_avg.png", dpi=300)
plt.show()

# ---------------------------
# Plot Latency
# ---------------------------
plt.figure(figsize=(8,6))
plt.plot(gpus, uccl_avg_lat, marker="o", label="UCCL-EP (Avg)")
plt.plot(gpus, pplx_avg_lat, marker="s", label="Pplx (Avg)")
plt.xlabel("Number of GPUs")
plt.ylabel("Latency (µs)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("latency_vs_gpus_avg.png", dpi=300)
plt.show()

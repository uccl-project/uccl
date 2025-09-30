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

# Averages for sparse systems
uccl_avg_tp = np.mean([uccl_dispatch_tp, uccl_combine_tp], axis=0)
pplx_avg_tp = np.mean([pplx_dispatch_tp, pplx_combine_tp], axis=0)

# Relative throughput (normalize to best per GPU count)
throughput_matrix = np.vstack([uccl_avg_tp, pplx_avg_tp])
baseline_tp = np.max(throughput_matrix, axis=0)  # best system for each GPU count
uccl_rel_tp = uccl_avg_tp / baseline_tp
pplx_rel_tp = pplx_avg_tp / baseline_tp

# ---------------------------
# Data (Latency in µs)
# ---------------------------
uccl_dispatch_lat = [2694, 6643, 10399]
uccl_combine_lat = [3421, 7491, 10968]
pplx_dispatch_lat = [7446, 9363, 10924]
pplx_combine_lat = [9841, 11095, 13380]

# Averages for sparse systems
uccl_avg_lat = np.mean([uccl_dispatch_lat, uccl_combine_lat], axis=0)
pplx_avg_lat = np.mean([pplx_dispatch_lat, pplx_combine_lat], axis=0)

# Relative latency (normalize to best per GPU count)
latency_matrix = np.vstack([uccl_avg_lat, pplx_avg_lat])
baseline_lat = np.min(latency_matrix, axis=0)  # best (lowest) system per GPU count
uccl_rel_lat = uccl_avg_lat / baseline_lat
pplx_rel_lat = pplx_avg_lat / baseline_lat

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
# Plot Relative Throughput
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(gpus, uccl_rel_tp, marker="o", label="UCCL-EP")
plt.plot(gpus, pplx_rel_tp, marker="s", label="Pplx")
plt.xlabel("Number of GPUs")
plt.ylabel("Relative Throughput (×)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("throughput_vs_gpus_relative.png", dpi=300)
plt.show()

# ---------------------------
# Plot Relative Latency
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(gpus, uccl_rel_lat, marker="o", label="UCCL-EP")
plt.plot(gpus, pplx_rel_lat, marker="s", label="Pplx")
plt.xlabel("Number of GPUs")
plt.ylabel("Relative Latency (×)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("latency_vs_gpus_relative.png", dpi=300)
plt.show()

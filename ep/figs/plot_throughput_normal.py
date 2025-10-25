import matplotlib.pyplot as plt
import numpy as np

# Data
gpus = [16, 24, 32]  # EP=16, 24, 32 corresponds to 2, 3, 4 nodes/GPUs
uccl_dispatch = [2.788, 1.131, 0.7224]
# uccl_combine = [4.249, 1.941, 1.325]
pplx_dispatch = [1.027, 0.809, 0.693]
# pplx_combine = [1.840, 1.324, 1.098]
# torch_all2all = [40.876, 47.280, 42.444]
# nvshmem = [18.142, 11.340, 11.016]

# Take averages for sparse systems
# uccl_avg = np.mean([uccl_dispatch, uccl_combine], axis=0)
# pplx_avg = np.mean([pplx_dispatch, pplx_combine], axis=0)
uccl_avg = uccl_dispatch
pplx_avg = pplx_dispatch

# Font settings for readability
plt.rcParams.update(
    {
        "font.size": 18,  # base font size
        "axes.labelsize": 20,  # axis labels
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 3,
        "lines.markersize": 10,
    }
)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(gpus, uccl_avg, marker="o", label="UCCL-EP")
plt.plot(gpus, pplx_avg, marker="s", label="Pplx")
# plt.plot(gpus, torch_all2all, marker="^", label="Torch all-to-all (Dense)")
# plt.plot(gpus, nvshmem, marker="v", label="NVSHMEM (Dense)")

# Labels and grid
plt.xlabel("Number of GPUs")
plt.ylabel("Throughput (GB/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save or show
plt.savefig("throughput_vs_gpus_all.png", dpi=300)
plt.show()

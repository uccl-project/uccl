import re
import matplotlib.pyplot as plt

log_file = "node1_dispatch.log"

# --- Regex patterns ---
tuning_re = re.compile(
    r"\[tuning\s+(dispatch|combine)\].*BW:\s*([\d.]+)\s*GB/s\s*\(RDMA\)", re.IGNORECASE)
post_re = re.compile(
    r"\[\d+\.\d+s\s*\|\s*\+([\d.]+)ms\].*Posting\s+(\d+)\s+RDMA writes", re.IGNORECASE)

# --- Storage ---
bw_dispatch, rate_dispatch = [], []
bw_combine, rate_combine = [], []

# --- State ---
current_type = None
current_bw = None
current_writes = []
current_durations = []

with open(log_file, "r") as f:
    for line in f:
        t_match = tuning_re.search(line)
        p_match = post_re.search(line)

        if t_match:
            # Finalize previous block
            if current_bw is not None and current_durations:
                total_time_ms = sum(current_durations)
                total_writes = sum(current_writes)
                avg_rate = (total_writes / total_time_ms) if total_time_ms > 0 else 0
                if current_type == "dispatch":
                    bw_dispatch.append(current_bw)
                    rate_dispatch.append(avg_rate)
                elif current_type == "combine":
                    bw_combine.append(current_bw)
                    rate_combine.append(avg_rate)

            # Start new block
            current_type = t_match.group(1).lower()
            current_bw = float(t_match.group(2))
            current_writes = []
            current_durations = []

        elif p_match and current_bw is not None:
            duration_ms = float(p_match.group(1))
            writes = int(p_match.group(2))
            current_durations.append(duration_ms)
            current_writes.append(writes)

# Finalize last block
if current_bw is not None and current_durations:
    total_time_ms = sum(current_durations)
    total_writes = sum(current_writes)
    avg_rate = (total_writes / total_time_ms) if total_time_ms > 0 else 0
    if current_type == "dispatch":
        bw_dispatch.append(current_bw)
        rate_dispatch.append(avg_rate)
    elif current_type == "combine":
        bw_combine.append(current_bw)
        rate_combine.append(avg_rate)

# --- Scatter plot ---
plt.figure(figsize=(8, 5))
plt.scatter(bw_dispatch, rate_dispatch, s=80, alpha=0.8, edgecolors="black",
            color="royalblue", label="dispatch")
plt.scatter(bw_combine, rate_combine, s=80, alpha=0.8, edgecolors="black",
            color="red", label="combine")
plt.xlabel("RDMA Bandwidth (GB/s)")
plt.ylabel("Average number of RDMA Writes per ms")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("debug.png", dpi=200)
plt.show()
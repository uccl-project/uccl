import re
import matplotlib.pyplot as plt

# === Input log file ===
log_file = "node1_dispatch.log"

# === Regex patterns ===
tuning_re = re.compile(
    r"\[tuning\s+(dispatch|combine)\].*NVL chunk\s+(\d+),\s*RDMA chunk\s+(\d+).*?"
    r"BW:\s*([\d.]+)\s*GB/s\s*\(RDMA\)",
    re.IGNORECASE,
)

mixed_re = re.compile(
    r"\[\d+\.\d+s\s*\|\s*\+([\d.]+)ms\].*\[post_gpu_commands_mixed\]"
    r".*Posting\s+(\d+)\s+RDMA writes",
    re.IGNORECASE,
)

cmd_re = re.compile(
    r"\[.*\].*\[post_rdma_async_batched\].*cmd\.bytes:\s*(\d+)",
    re.IGNORECASE,
)

# === Storage ===
bw_dispatch, y_dispatch, label_dispatch = [], [], []
bw_combine, y_combine, label_combine = [], [], []

# === State ===
current_type = None
current_bw = None
current_label = None
durations, num_writes, cmd_bytes = [], [], []

# === Parse file ===
with open(log_file, "r") as f:
    for line in f:
        t_match = tuning_re.search(line)
        m_match = mixed_re.search(line)
        c_match = cmd_re.search(line)

        if t_match:
            # finalize previous tuning block
            if current_bw is not None and durations and cmd_bytes:
                total_time_ms = sum(durations)
                total_writes = sum(num_writes)
                avg_writes_per_ms = (
                    total_writes / total_time_ms if total_time_ms > 0 else 0
                )
                avg_bytes_kb = sum(cmd_bytes) / len(cmd_bytes) / 1024.0
                y_value = avg_writes_per_ms * avg_bytes_kb

                if current_type == "dispatch":
                    bw_dispatch.append(current_bw)
                    y_dispatch.append(y_value)
                    label_dispatch.append(current_label)
                elif current_type == "combine":
                    bw_combine.append(current_bw)
                    y_combine.append(y_value)
                    label_combine.append(current_label)

            # start new tuning block
            current_type = t_match.group(1).lower()
            nvl_chunk = t_match.group(2)
            rdma_chunk = t_match.group(3)
            current_bw = float(t_match.group(4))
            current_label = f"({nvl_chunk},{rdma_chunk})"
            durations, num_writes, cmd_bytes = [], [], []

        elif m_match and current_bw is not None:
            durations.append(float(m_match.group(1)))
            num_writes.append(int(m_match.group(2)))

        elif c_match and current_bw is not None:
            cmd_bytes.append(int(c_match.group(1)))

# finalize last tuning block
if current_bw is not None and durations and cmd_bytes:
    total_time_ms = sum(durations)
    total_writes = sum(num_writes)
    avg_writes_per_ms = total_writes / total_time_ms if total_time_ms > 0 else 0
    avg_bytes_kb = sum(cmd_bytes) / len(cmd_bytes) / 1024.0
    y_value = avg_writes_per_ms * avg_bytes_kb

    if current_type == "dispatch":
        bw_dispatch.append(current_bw)
        y_dispatch.append(y_value)
        label_dispatch.append(current_label)
    elif current_type == "combine":
        bw_combine.append(current_bw)
        y_combine.append(y_value)
        label_combine.append(current_label)

# === Plot ===
plt.figure(figsize=(9, 6))
plt.scatter(
    bw_dispatch,
    y_dispatch,
    s=90,
    alpha=0.85,
    edgecolors="black",
    color="royalblue",
    label="dispatch",
)
plt.scatter(
    bw_combine,
    y_combine,
    s=90,
    alpha=0.85,
    edgecolors="black",
    color="red",
    label="combine",
)

# optional labels
# for x, y, label in zip(bw_dispatch, y_dispatch, label_dispatch):
#     plt.text(x, y * 1.02, label, color="navy", fontsize=9, ha="center")
# for x, y, label in zip(bw_combine, y_combine, label_combine):
#     plt.text(x, y * 1.02, label, color="darkred", fontsize=9, ha="center")

plt.xlabel("RDMA Bandwidth (GB/s)")
plt.ylabel("Average (Writes/ms × cmd.bytes) [KB/ms]")
plt.title(
    "Effective RDMA Throughput vs RDMA Bandwidth\n(labeled by NVL/RDMA chunk)"
)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("debug.png", dpi=200)
plt.show()
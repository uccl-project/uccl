import matplotlib.pyplot as plt

tokens = [128, 256, 512, 1024, 2048, 4096]

# uccl_ep_dispatch = [244.44, 322.68, 406.69, 552.94, 797.31, 1222.0]
# uccl_ep_combine  = [573.8, 951.78, 1428.0, 2253.0, 3541.0, 6518.0]

# pplx_dispatch = [212.4, 385.0, 728, 1085, 1793, 3212]
# pplx_combine  = [237.9, 408, 751, 1424, 2754, 5387]

uccl_ep_dispatch = [200.25, 288.93, 430.19, 660.09, 1113.00, 2026.00]
uccl_ep_combine  = [ 519.51, 763.02, 1176.00, 1929.00, 3039.00, 5377.00]

pplx_dispatch = [228.6, 399.4, 744.9, None, None, None]
pplx_combine  = [331.4, 591.3, 1103.4, None, None, None]


def align_tokens(tokens_list, series):
    return tokens_list[:len(series)]

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

# Dispatch
x_dispatch = align_tokens(tokens, uccl_ep_dispatch)
plt.figure()
plt.plot(x_dispatch, uccl_ep_dispatch, marker="o", label="UCCL-EP")
plt.plot(align_tokens(tokens, pplx_dispatch), pplx_dispatch, marker="s", label="pplx")
plt.xlabel("Number of tokens")
plt.ylabel("Latency (µs)")
plt.grid(True, which="both", linestyle="--", linewidth=1, alpha=0.5)
plt.legend()
plt.title("Dispatch (EP=32, EFA)")
plt.tight_layout()
plt.savefig("dispatch_latency_vs_tokens_pplx_uccl.png", dpi=300)

# Combine
x_combine = align_tokens(tokens, uccl_ep_combine)
plt.figure()
plt.plot(x_combine, uccl_ep_combine, marker="o", label="UCCL-EP")
plt.plot(align_tokens(tokens, pplx_combine), pplx_combine, marker="s", label="pplx")
plt.xlabel("Number of tokens")
plt.ylabel("Latency (µs)")
plt.grid(True, which="both", linestyle="--", linewidth=1, alpha=0.5)
plt.legend()
plt.title("Combine (EP=32, EFA)")
plt.tight_layout()
plt.savefig("combine_latency_vs_tokens_pplx_uccl.png", dpi=300)

plt.show()

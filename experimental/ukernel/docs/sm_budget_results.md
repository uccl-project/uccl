# SM-budget experiment results (2026-09-04)

Redo of the shim-vs-native suite under the *fewer SMs than native* rule:
every shim cell runs at `b <= B` where `B` = native coll channels at that
placement; AllToAll is reported as a pure CE/IPC path with **0 SM**.
HEAD `14391f89`, medians of 3, nccl-tests `-n 10 -w 2`, validation on,
all cells 0 wrong. Raw logs: `/tmp/b300_final`, `/tmp/b300_rotation`,
`/tmp/b300_hostprof`, `/tmp/l40s_final`, `/tmp/l40s_small_final`,
`/tmp/l40s_rotation` on the testbeds.

## SM budgets (native coll channels, measured)

| platform | S2 | S4 | S8 | X4 | X8 | X16 |
|---|---:|---:|---:|---:|---:|---:|
| B300 | 32 | 32 | 32 | — | — | — |
| L40S | 4 | 4 | 2 | 2 | 2 | 2 |

## Best block counts `b*`

| collective | B300 np2/np4/np8 | L40S (per placement) |
|---|---|---|
| AllReduce (fused on B300) | 32 / 32 / 28 | S2,S4: 1 (4 at ≤4M); S8: 1 (2 at ≤4M); X*: 1 (2 at ≤16M) |
| ReduceScatter | 32 / 28 / 32 | S2: 4; S4: 2; S8/X4/X8: 1 (2 at ≤4M); X16: 2 |
| AllGather | 28 / 32 / 24 | 1 everywhere (2 at ≤4M neutral) |
| AllToAll | CE, 0 SM | CE, 0 SM |

Small messages (1M/4M) on L40S are faster at the budget max (`b=4` on
S2/S4, `b=2` on S8/X*) for AllReduce/ReduceScatter; the tables below use
that per-size `b`.

## B300 — busbw GB/s (median of 3)

### AllReduce (fused), shim b=32/32/28 vs native 32ch

| size | sh2 | nat2 | sh4 | nat4 | sh8 | nat8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 6.61 | 53.97 | 5.02 | 65.72 | 2.83 | 53.78 |
| 4M | 27.15 | 96.81 | 17.54 | 141.38 | 11.19 | 124.62 |
| 16M | 81.95 | 288.24 | 62.42 | 246.96 | 38.84 | 270.05 |
| 64M | 219.46 | 434.51 | 180.53 | 539.79 | 109.87 | 419.74 |
| 256M | 341.14 | 508.64 | 317.23 | 596.40 | 247.10 | 651.84 |

### ReduceScatter, shim b=32/28/32 vs native

| size | sh2 | nat2 | sh4 | nat4 | sh8 | nat8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 3.36 | 30.75 | 1.97 | 42.12 | 1.18 | 37.33 |
| 4M | 10.77 | 68.19 | 7.42 | 94.29 | 3.63 | 97.21 |
| 16M | 25.04 | 199.37 | 21.20 | 212.30 | 13.77 | 242.56 |
| 64M | 48.56 | 331.61 | 48.09 | 444.99 | 43.34 | 413.19 |
| 256M | 133.54 | 406.92 | 143.82 | 529.19 | 146.46 | 593.09 |

### AllGather, shim b=28/32/24 vs native

| size | sh2 | nat2 | sh4 | nat4 | sh8 | nat8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 4.68 | 32.17 | 3.27 | 44.57 | 1.78 | 35.98 |
| 4M | 17.97 | 72.05 | 14.24 | 108.45 | 7.59 | 108.24 |
| 16M | 36.73 | 201.20 | 37.19 | 205.46 | 20.76 | 283.94 |
| 64M | 42.87 | 335.65 | 42.13 | 451.35 | 44.51 | 411.24 |
| 256M | 193.20 | 414.36 | 172.31 | 534.01 | 170.66 | 585.63 |

### AllToAll (CE path, 0 SM), shim vs native

| size | sh2 | nat2 | sh4 | nat4 | sh8 | nat8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 4.7 | 14.3 | 5.6 | 19.7 | 4.3 | 20.5 |
| 4M | 22.7 | 53.4 | 25.0 | 71.7 | 22.3 | 82.8 |
| 16M | 47.4 | 127.7 | 41.6 | 189.9 | 38.2 | 204.3 |
| 64M | 48.6 | 231.8 | 52.6 | 343.7 | 47.9 | 368.2 |
| 256M | 229.6 | 312.4 | 212.8 | 456.2 | 188.7 | 516.6 |

### Rotation A/B at 256M (CE path), medians: rot0 ascending, rot1 default

| np | rot0 | rot1 |
|---:|---:|---:|
| 2 | 264.5 | 211.7 |
| 4 | 215.6 | 192.7 |
| 8 | 187.8 | 206.6 |

NVSwitch is mixed: rotation hurts np2/np4 (-20%/-11%) and helps np8 (+10%).

## L40S — busbw GB/s (median of 3, per-size b*)

### AllReduce (unfused CE/IPC), shim vs native

| size | S2 sh/nat | S4 sh/nat | S8 sh/nat | X4 sh/nat | X8 sh/nat | X16 sh/nat |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1M | 8.31/14.48 | 5.68/15.13 | 2.92/9.37 | 4.99/11.19 | 3.09/5.11 | 1.41/5.06 |
| 4M | 19.96/20.18 | 13.03/21.27 | 6.77/14.53 | 8.73/13.54 | 7.49/14.41 | 4.62/12.07 |
| 16M | 23.45/21.14 | 23.91/21.85 | 12.90/14.27 | 11.49/13.73 | 11.49/14.49 | 9.70/15.01 |
| 64M | 24.46/21.39 | 24.61/21.68 | 15.22/14.16 | 11.53/13.60 | 11.52/14.31 | 11.77/14.46 |
| 256M | 25.61/21.48 | 25.64/21.72 | 16.02/14.43 | 10.97/13.67 | 11.06/14.43 | 10.96/14.64 |

### ReduceScatter, shim vs native

| size | S2 sh/nat | S4 sh/nat | S8 sh/nat | X4 sh/nat | X8 sh/nat | X16 sh/nat |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1M | 6.05/11.18 | 4.38/12.71 | 2.33/8.95 | 3.97/9.65 | 2.41/8.27 | 1.23/5.92 |
| 4M | 12.90/17.33 | 10.55/19.53 | 5.66/13.78 | 7.51/12.45 | 6.37/13.61 | 3.94/12.44 |
| 16M | 20.21/19.13 | 20.68/21.10 | 10.66/13.52 | 11.10/12.82 | 11.30/13.88 | 8.20/14.22 |
| 64M | 23.30/19.60 | 23.62/21.08 | 14.56/13.57 | 11.30/12.70 | 11.50/13.77 | 11.60/14.07 |
| 256M | 24.98/19.87 | 24.97/21.21 | 15.77/13.62 | 10.99/12.79 | 11.08/13.76 | 11.09/13.98 |

### AllGather (CE + local publish copy), shim b=1 vs native

| size | S2 sh/nat | S4 sh/nat | S8 sh/nat | X4 sh/nat | X8 sh/nat | X16 sh/nat |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1M | 8.42/10.92 | 6.11/12.22 | 3.18/8.49 | 5.75/9.37 | 3.60/7.96 | 1.74/5.82 |
| 4M | 17.48/17.90 | 14.46/19.59 | 8.01/13.95 | 9.84/12.67 | 8.77/13.95 | 5.71/11.70 |
| 16M | 21.83/19.98 | 21.11/21.36 | 13.40/14.29 | 11.19/13.25 | 11.29/14.65 | 11.44/15.22 |
| 64M | 23.00/20.55 | 23.82/21.41 | 14.42/14.29 | 11.46/13.18 | 11.33/14.56 | 11.69/14.88 |
| 256M | 25.20/20.75 | 25.47/21.49 | 15.37/14.39 | 10.99/13.16 | 11.30/14.52 | 11.21/14.86 |

### AllToAll (CE path, 0 SM), shim vs native

| size | S2 sh/nat | S4 sh/nat | S8 sh/nat | X4 sh/nat | X8 sh/nat | X16 sh/nat |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1M | 8.1/10.2 | 8.4/12.7 | 4.6/4.6 | 6.3/6.8 | 4.9/4.9 | 1.9/3.4 |
| 4M | 16.4/18.1 | 16.7/17.9 | 6.0/5.9 | 8.4/9.4 | 5.8/6.5 | 2.0/4.3 |
| 16M | 21.8/22.3 | 14.2/20.3 | 6.5/6.8 | 9.2/10.8 | 6.2/6.6 | 2.1/4.4 |
| 64M | 23.7/23.6 | 15.4/20.3 | 6.7/6.6 | 8.6/11.3 | 6.2/6.4 | 2.0/4.6 |
| 256M | 23.8/24.0 | 15.6/20.0 | 6.6/7.0 | 9.0/11.3 | 6.1/6.4 | 2.0/4.7 |

### Rotation A/B at 256M (CE path), medians

| np | rot0 | rot1 |
|---:|---:|---:|
| 2 | 23.8 | 23.8 |
| 4 | 15.2 | 15.6 |
| 8 | 6.5 | 6.7 |

## B300 host orchestration (1M AllReduce fused@b*, n=30/w=10, per-collective medians)

HostProf stage totals divided by 40 collectives (µs/collective):

| ranks | enq | sig (flag/signal drain) | tpt (put drain) | dev (device drain) |
|---:|---:|---:|---:|---:|
| 2 | 0.2 | 6.8 | 2.1 | 5.7 |
| 4 | 0.4 | 19.5 | 7.7 | 45.0 |
| 8 | 0.6 | 53.5 | 23.3 | 109.4 |

The host cost scales steeply with rank count while enqueue/dispatch stays
~0.2-0.6µs per collective: the small-message floor is signal/device drain
on the host, not dispatch.

## Key takeaways

- B300: fused AllReduce stays at (or, at np8, below) the native budget —
  `b* = 32/32/28` — and reaches 0.67/0.53/0.38 of native at 256 MiB.
  ReduceScatter and AllGather need 28-32 blocks at 256 MiB, so the
  "fewer SMs" margin on NVSwitch is in AllToAll (0 SM) and in the
  np8 AllReduce cell (28 < 32), not in large RS/AG.
- L40S: PCIe is the bottleneck; AllReduce reaches best performance at
  `b=1` for ≥16 MiB but small messages prefer the budget max. Fusion
  remains negative on this platform (see l40s_measurements.md).
- AllToAll is measured end-to-end as a 0-SM CE/IPC path on both
  platforms; rotation is neutral/positive on L40S and mixed on B300
  NVSwitch (hurts 2/4 ranks, helps 8).

# Single-node 4×L4 ReduceScatter and AllReduce serial-write trial (2026-11-11)

Both `mibura-sky-test-02` (owner) and `l40` (`mibura-sky-test-01`, test node) ran the same ReduceScatter source and NCCL 2.28.7 library SHA; drivers differ. See `hardware_software.txt` for topology, versions and hashes. The main ReduceScatter SVG uses only l40 paired values. All tests are FP32, out of place for ReduceScatter, with full bytes per GPU.

## ReduceScatter

Command on both hosts: `peer_ag_rs --rs-only -b 1024 -e 1073741824 -w 20 -n 10 -r 3`. The same harness rotates the UCCL candidates and NCCL. Every warmup and timed call checks all output elements. The selected CSV chooses the fastest correct non-CPU-router UCCL median per size, without consulting NCCL. Values below are slowest-GPU per-call CUDA-event microseconds, median of three run means.

Each source GPU uses one CTA to write destination shards sequentially. Through 1 MiB, the fused path publishes peer stores with a system fence, signals per-source flags, waits for all remote sources, and reduces locally inside that same kernel. At 8 KiB, an LL16 packet variant avoids the whole-block fence. The source-side write plus separate reduction path wins at 2–8 MiB. The prior peer-copy path remains selected at 16 MiB and above. Four GPUs write disjoint source regions in each destination scratch. This is an eager path: the harness completes each call before reusing its scratch and generation flags.

| Full input bytes/GPU | l40 UCCL path | l40 UCCL (µs) | l40 NCCL (µs) |
| ---: | --- | ---: | ---: |
| 1 KiB | pull | 16.86 | 28.84 |
| 2 KiB | pull | 14.49 | 26.36 |
| 4 KiB | pull | 14.34 | 26.42 |
| 8 KiB | scatter-ll | 16.52 | 26.82 |
| 16 KiB | scatter-fence | 20.42 | 27.97 |
| 32 KiB | scatter-fence | 21.76 | 29.43 |
| 64 KiB | scatter-fence | 22.94 | 32.52 |
| 128 KiB | scatter-fence | 28.83 | 39.11 |
| 256 KiB | scatter-fence | 36.38 | 52.37 |
| 512 KiB | scatter-fence | 51.37 | 67.72 |
| 1 MiB | scatter-fence | 85.57 | 90.48 |
| 2 MiB | push-serial | 139.41 | 141.51 |
| 4 MiB | push-serial | 232.91 | 240.23 |
| 8 MiB | push-serial | 424.61 | 436.88 |
| 16 MiB | peer | 796.65 | 826.52 |
| 32 MiB | peer-noself | 1497.27 | 1534.89 |
| 64 MiB | peer-noself | 3062.25 | 3037.89 |
| 128 MiB | peer-noself | 6149.39 | 5976.54 |
| 256 MiB | peer-noself | 12226.54 | 11814.05 |
| 512 MiB | peer-noself | 24365.85 | 23465.84 |
| 1 GiB | peer-noself | 48615.87 | 47068.22 |

The selected l40 path wins at 16 of 21 sizes. At 32 KiB it takes 21.76 versus 29.43 µs; at 512 KiB, 51.37 versus 67.73 µs; at 1 MiB, 85.57 versus 90.48 µs. The owner host reproduces those gains. The five 64 MiB–1 GiB points still lose; at 1 GiB, 48.62 versus 47.07 ms. The l40 32-KiB full sweep has one high NCCL run; `l40_32k_recheck.log` confirms 21.50 versus 29.08 µs without that outlier.

`rs_crossover_all.log`, `rs_ll_serial_sweep.log`, and `rs_fence_ll_serial_sweep.log` retain the candidate/crossover trials. The scalar fused-fence prototype and LL packets above 32 KiB regress; vectorized peer stores and reduction fix the 32 KiB–1 MiB range. `rs_fence_profile.nsys-rep` is a raw diagnostic timeline. `owner_rs_tail_1008.log`, `l40_rs_tail_1008.log`, `owner_rs_tail_262160.log` and `l40_rs_tail_262160.log` verify non-power-of-two tails; the 262160-byte scalar fallback is correct but slower than NCCL. AG regression logs show the previously selected AG path remains correct and fast.

`owner_rs_full.log` and `l40_rs_full.log` are the accepted raw sweeps; `owner_samples.csv` and `l40_samples.csv` contain 2,310 timed calls per host. Candidate and selected CSVs retain all measured paths. `scripts/summarize_rs_serial.py` validates every size, repetition, call count, path and run-level median. `scripts/plot_optimized_collectives.py` regenerates the l40 figure in the established eight-L4 style.

## AllReduce negative result

`ar_serial_prototype.cu` archives two unpromoted native AllReduce candidates. A6 sequentially writes a full copy to each peer, uses one system-scoped completion flag per source, then reduces in place. A7 sends LL16 packets one peer at a time before local reduction. Both passed correctness in the native MSCCL++ harness. They were screened against the existing A1/A2 algorithms with the same native host-clock method on both four-L4 hosts; no NCCL number is juxtaposed with these native timings.

At 32 KiB, the owner sweep showed A6 26.21 µs versus the best existing A1/A2 path 27.09 µs, but three independent rechecks changed the winner. On l40 A6 took 27.35 µs versus A2 26.72 µs. At 64 KiB, A6 took 34.03 versus A1 29.41 µs on the owner and 35.19 versus 29.65 µs on l40. A7 was slower at 32 KiB (38.94 and 39.57 µs). The serial methods therefore have no repeatable AllReduce gain; the shipping AllReduce source and figure remain unchanged. `ar_serial_ll_sweep.log`, `l40_ar_serial_sweep.log`, `ar_32k_repeat_*.log` and thread-count logs retain the negative measurements.

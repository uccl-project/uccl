# 8xL4 AllGather packet tuning (2026-11-11)

This is one UCCL-Lite host-packet router data set for both the 8xL4 GPU and
8xL4 SHM figures. Both use the same AllGather router path. The SHM figure
compares these UCCL-Lite values with the archived NCCL SHM control; those
two curves were measured in separate runs.

The [screening data](screening/summary.csv) compares 512 KiB, 2 MiB, and
4 MiB packets in two opposite-order sweeps. The 2 MiB choice improves most
large sizes while keeping smaller AllGather calls on 512 KiB packets. The
full paired [ascending](run1.csv) and [descending](run2.csv) runs compare
this setting with NCCL 2.28.9. Each size gets 20 warmups and three
alternating repetitions of 10 timed calls per backend. All 2,520 timed
calls were checked on every GPU; [run 1](run1.log) and [run 2](run2.log)
end in correctness PASS. The [summary](summary.csv) takes the median of
repetition medians within each run, then the median of the two runs.

At 1 GiB, tuned UCCL-Lite takes 166.6 ms and NCCL takes 191.2 ms in the
normal 8xL4 GPU testbed. The SHM NCCL control is recorded separately in
[the SHM data](../shm_8l4_20261111/README.md).

Reproduce the two full runs from `experimental/lite` with `--gpus 8
--fallback nccl --op allgather --large-ag-packet-bytes 2097152 --begin 1024
--end 1073741824 --warmup 20 --iterations 10 --repeats 3 --calibrate`
and `NCCL_NET=Socket`; add `--reverse` on run 2. Use distinct `--samples`
paths, then run `scripts/summarize_ag_packet_8l4.py`.

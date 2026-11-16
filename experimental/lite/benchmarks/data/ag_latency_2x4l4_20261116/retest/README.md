# Recheck of the 256/512-KiB NCCL spikes

No smoothing or removal of slow samples. Both nodes were GPU-idle before the
recheck; binaries, FP32 data, eight GPUs, interfaces and timing scope are
unchanged. Every collective output is checked in every run.

## Controlled observations

- `raw/`: three fresh single-size jobs for each of default/SM, at
  128/256/512/1024 KiB, 20 warmups and 100 measured calls each. At 512 KiB,
  default run means are 2054/1623/2099 µs; SM means are 2180/2172/1896 µs.
  The spikes are reproducible, not a few isolated outliers.
- `probes/`: two jobs per setting/size at 256/512 KiB. 100 warmups and
  switching to eight MPI processes/one GPU each do not remove the spikes.
  Forcing Ring alone does not remove them either. Forcing SIMPLE gives
  123–126 µs at 512 KiB; forcing Ring+SIMPLE gives 125–126 µs.
- `protocol_trace/`: separate diagnostic logging (excluded from plotted
  timings) confirms **RING + LL at 128/256/512 KiB**, **RING + SIMPLE at
  1 MiB**. Byte counts in NCCL's log are per-GPU contributions, one eighth
  of the complete output bytes plotted. This links the slowdown to the LL
  path on these hosts; it does not claim a general NCCL correctness bug.

## Final replacement curves

`refresh/` remeasures the **entire 1 KiB–1 MiB default and SM latency curves**,
with **100 warmups + 200 timed calls per size/job, three fresh jobs**, reversing
mode order in round 2. All 13,200 timed calls pass full output checks.

Default remains genuinely unforced and retains its real spikes. The latency
panel's SM tuned configuration now uses one fixed `NCCL_PROTO=Simple`, with
P2P/SHM copy-engine paths disabled, at **every size**; it is not a per-size
best-of envelope. At 256/512 KiB, median run means are:

| Configuration | 256 KiB | 512 KiB |
| --- | ---: | ---: |
| NCCL default | 646.56 µs | 1938.56 µs |
| NCCL SM, fixed SIMPLE | 101.23 µs | 119.50 µs |

The original summary, run means, figure inputs and figure are retained under
`initial/`; original raw logs remain in the parent `raw/` directory. Only the
two complete latency series are replaced. UCCL Lite, CE and the entire
historical throughput panel remain unchanged. The plotted samples therefore
use 20 warmups/50 calls for UCCL/CE and 100/200 for default/SM, three jobs each.

From the Lite root, after building/staging the same binaries:

```bash
python3 benchmarks/data/ag_latency_2x4l4_20261116/retest/refresh.py \
  --out build/ag_latency_recheck
```

Run `run.py` for isolated-size checks, `probe.py` for the controlled settings,
and `trace_protocol.py` only for protocol identification. Diagnostic/selection
samples are not mixed into final plotted timing.

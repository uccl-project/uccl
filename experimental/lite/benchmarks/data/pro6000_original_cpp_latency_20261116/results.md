# Original PCIeCCL C++ auto-path latency on eight PRO6000 GPUs

FP32; original unchanged C++ source; each run uses 10 warmups and 20 continuous timed calls.
Values are the median and observed range of five independent run means, not confidence intervals.
The PRO5000 historical column uses a separate host/toolchain/window and is not an EAT before/after comparison.

| Bytes/rank | Auto path | PRO6000 median (µs) | Five-run range (µs) | Historical PRO5000 (µs) |
| ---: | --- | ---: | ---: | ---: |
| 1024 | LL | 10.37 | 10.04–10.98 | 7.86 |
| 2048 | LL | 10.79 | 9.84–10.90 | 7.72 |
| 4096 | LL | 10.95 | 10.09–11.02 | 8.29 |
| 8192 | LL | 11.82 | 10.62–12.14 | 8.08 |
| 16384 | LL | 12.10 | 11.82–12.64 | 9.34 |
| 32768 | LL | 14.55 | 14.03–15.20 | 11.62 |
| 65536 | PUT | 32.40 | 31.79–33.02 | 29.34 |
| 131072 | PUT | 33.97 | 33.18–34.51 | 30.42 |
| 262144 | PUT | 36.74 | 36.32–37.73 | 35.51 |
| 524288 | PUT | 48.27 | 48.10–48.94 | 49.07 |
| 1048576 | PUT | 72.87 | 72.34–74.12 | 78.01 |

All 195 size/mode pre-timing correctness checks report zero maximum relative error.
The original timed loop does not validate every invocation; this behavior is preserved.
See [README.md](README.md) for hardware, timing boundaries, source identities and reproduction.

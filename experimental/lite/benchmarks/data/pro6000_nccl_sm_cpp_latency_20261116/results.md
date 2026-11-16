# Matched FP32/C++ batch latency on eight RTX PRO6000 GPUs

In-place SUM; 10 warmups and 20 continuous timed calls; C++ host batch timer; slowest rank.
Independent calibration: three sweeps/config. Validation: five fresh sweeps/config; medians below.
The tuned curve uses per-size choices frozen before validation. Original PCIeCCL is freshly rerun.

| Size | NCCL default SM (µs) | NCCL tuned SM (µs) | Original PCIeCCL auto (µs) | Selected NCCL config |
| ---: | ---: | ---: | ---: | --- |
| 1 KiB | 25.47 | 18.01 | 9.86 | ring-ll |
| 2 KiB | 25.64 | 18.26 | 9.75 | ring-ll |
| 4 KiB | 59.66 | 18.43 | 10.06 | ring-ll |
| 8 KiB | 61.31 | 18.42 | 10.59 | ring-ll |
| 16 KiB | 79.97 | 18.68 | 11.48 | ring-ll |
| 32 KiB | 64.82 | 19.41 | 13.87 | ring-ll |
| 64 KiB | 74.05 | 19.97 | 32.08 | ring-ll |
| 128 KiB | 76.53 | 22.89 | 33.44 | ring-ll |
| 256 KiB | 82.40 | 32.75 | 36.38 | ring-ll-ch2 |
| 512 KiB | 112.48 | 54.02 | 48.26 | ring-ll |
| 1024 KiB | 172.92 | 70.42 | 73.40 | ring-simple-ch8 |

All accepted size/run pre-timing checks reported zero maximum relative error.
See [README.md](README.md) for the frozen policies, raw records, limitations and reproduction.

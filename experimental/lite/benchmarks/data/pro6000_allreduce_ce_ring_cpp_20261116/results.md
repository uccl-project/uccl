# Frozen-policy FP32/C++ AllReduce validation

Eight RTX PRO 6000 GPUs; in-place SUM; 10 warmups then 20 continuous calls; host timing through final stream synchronization; slowest rank. Centers are medians of five independent run means. Per-size choices come only from three-run calibration.

| Size | UCCL Lite (µs) | NCCL CE (µs) | Speedup | Paired wins | UCCL worst < NCCL best |
| --- | ---: | ---: | ---: | ---: | :---: |
| 1 KiB | 9.10 | 186.90 | 20.538× | 5/5 | yes |
| 2 KiB | 9.16 | 184.84 | 20.179× | 5/5 | yes |
| 4 KiB | 9.19 | 185.24 | 20.158× | 5/5 | yes |
| 8 KiB | 9.29 | 185.52 | 19.972× | 5/5 | yes |
| 16 KiB | 9.70 | 188.57 | 19.445× | 5/5 | yes |
| 32 KiB | 11.33 | 191.25 | 16.878× | 5/5 | yes |
| 64 KiB | 12.99 | 193.72 | 14.911× | 5/5 | yes |
| 128 KiB | 16.11 | 202.78 | 12.585× | 5/5 | yes |
| 256 KiB | 26.58 | 215.69 | 8.115× | 5/5 | yes |
| 512 KiB | 48.26 | 218.69 | 4.532× | 5/5 | yes |
| 1 MiB | 75.42 | 244.69 | 3.244× | 5/5 | yes |
| 2 MiB | 126.03 | 300.72 | 2.386× | 5/5 | yes |
| 4 MiB | 229.17 | 356.92 | 1.557× | 5/5 | yes |
| 8 MiB | 392.71 | 481.78 | 1.227× | 5/5 | yes |
| 16 MiB | 627.27 | 744.45 | 1.187× | 5/5 | yes |
| 32 MiB | 1173.44 | 1291.66 | 1.101× | 5/5 | yes |
| 64 MiB | 2299.65 | 2430.40 | 1.057× | 5/5 | yes |
| 128 MiB | 4545.92 | 4854.28 | 1.068× | 5/5 | yes |
| 256 MiB | 9034.35 | 9441.03 | 1.045× | 5/5 | yes |
| 512 MiB | 18085.15 | 18671.42 | 1.032× | 5/5 | yes |
| 1 GiB | 36179.72 | 37070.46 | 1.025× | 5/5 | yes |

Remaining sizes without five paired wins: none.

Observed ranges are not confidence intervals. No final-run or final-variant selection is used.

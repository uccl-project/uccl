# Frozen-policy FP32/C++ AllReduce validation

Eight RTX PRO 6000 GPUs; in-place SUM; 10 warmups then 20 continuous calls; host timing through final stream synchronization; slowest rank. Centers are medians of five independent run means. Per-size choices come only from three-run calibration.

| Size | UCCL Lite (µs) | NCCL CE (µs) | Speedup | Paired wins | UCCL worst < NCCL best |
| --- | ---: | ---: | ---: | ---: | :---: |
| 1 KiB | 9.38 | 185.76 | 19.811× | 5/5 | yes |
| 2 KiB | 9.38 | 183.79 | 19.590× | 5/5 | yes |
| 4 KiB | 9.27 | 184.45 | 19.893× | 5/5 | yes |
| 8 KiB | 9.30 | 185.91 | 19.988× | 5/5 | yes |
| 16 KiB | 9.69 | 188.95 | 19.496× | 5/5 | yes |
| 32 KiB | 11.39 | 190.74 | 16.743× | 5/5 | yes |
| 64 KiB | 13.29 | 193.51 | 14.556× | 5/5 | yes |
| 128 KiB | 16.62 | 201.84 | 12.147× | 5/5 | yes |
| 256 KiB | 26.56 | 216.68 | 8.158× | 5/5 | yes |
| 512 KiB | 48.35 | 218.72 | 4.524× | 5/5 | yes |
| 1 MiB | 74.49 | 244.81 | 3.286× | 5/5 | yes |
| 2 MiB | 126.14 | 300.03 | 2.378× | 5/5 | yes |
| 4 MiB | 230.76 | 357.14 | 1.548× | 5/5 | yes |
| 8 MiB | 395.35 | 483.14 | 1.222× | 5/5 | yes |
| 16 MiB | 716.48 | 745.41 | 1.040× | 5/5 | yes |
| 32 MiB | 1366.90 | 1293.02 | 0.946× | 0/5 | no |
| 64 MiB | 2719.99 | 2429.96 | 0.893× | 0/5 | no |
| 128 MiB | 5523.45 | 4856.92 | 0.879× | 0/5 | no |
| 256 MiB | 11115.49 | 9450.62 | 0.850× | 0/5 | no |
| 512 MiB | 22339.48 | 18675.58 | 0.836× | 0/5 | no |
| 1 GiB | 44768.66 | 37079.87 | 0.828× | 0/5 | no |

Remaining sizes without five paired wins: 32 MiB, 64 MiB, 128 MiB, 256 MiB, 512 MiB, 1 GiB.

Observed ranges are not confidence intervals. No final-run or final-variant selection is used.

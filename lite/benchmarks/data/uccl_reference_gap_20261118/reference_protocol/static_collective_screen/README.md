# Static operation specialization: mixed results

Two four-L4 hosts, FP32 AG and SUM AR/RS, fixed 1-GiB allocations, 32–512 KiB,
both layouts, three repetitions with counterbalanced variant order. 20 warmups;
AG 30 / AR 10 / RS 10 timed calls. Full GPU and internal send retirement is included.
Each value is a median of three maximum-rank per-call means. Baseline is the
fixed output-overlap/ordered candidate, not native NCCL. No best-per-size curve
is assembled.

The candidate compiles separate AG/AR/RS kernels from the exact same bodies,
arguments, bank rules and synchronization. It passes 144 mixed ABI calls/rank
and all 36 screen jobs with full-output, exact route and native-absence checks.
Specialized RS kernels have zero stack in resource metadata, but the 512-KiB
RS latency gap remains. AG does not improve consistently. Packet AR 32/64 KiB
improves in both layouts, while larger AR shapes do not; this suggests a
narrower code-generation experiment, not promotion of the entire variant.
The full zero-regression goal remains unmet and the flag stays unselected.

| Operation | Bytes | Layout | Generic (us) | Specialized (us) | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| ag | 32768 | out | 27.715 | 27.059 | 0.976316 |
| ag | 32768 | in | 26.849 | 27.059 | 1.007846 |
| ag | 65536 | out | 30.100 | 29.628 | 0.984322 |
| ag | 65536 | in | 29.677 | 30.044 | 1.012360 |
| ag | 131072 | out | 37.051 | 36.598 | 0.987784 |
| ag | 131072 | in | 36.813 | 37.140 | 1.008861 |
| ag | 262144 | out | 43.630 | 42.270 | 0.968828 |
| ag | 262144 | in | 41.590 | 42.792 | 1.028904 |
| ag | 524288 | out | 51.823 | 52.163 | 1.006567 |
| ag | 524288 | in | 51.334 | 51.868 | 1.010406 |
| ar | 32768 | out | 38.137 | 37.591 | 0.985686 |
| ar | 32768 | in | 38.140 | 37.185 | 0.974940 |
| ar | 65536 | out | 46.303 | 44.906 | 0.969829 |
| ar | 65536 | in | 45.575 | 44.530 | 0.977066 |
| ar | 131072 | out | 55.794 | 56.397 | 1.010809 |
| ar | 131072 | in | 56.690 | 56.602 | 0.998449 |
| ar | 262144 | out | 74.871 | 75.655 | 1.010467 |
| ar | 262144 | in | 73.853 | 74.909 | 1.014296 |
| ar | 524288 | out | 116.710 | 117.380 | 1.005739 |
| ar | 524288 | in | 116.521 | 116.810 | 1.002475 |
| rs | 32768 | out | 29.329 | 30.150 | 1.027979 |
| rs | 32768 | in | 29.151 | 30.045 | 1.030664 |
| rs | 65536 | out | 31.815 | 31.744 | 0.997750 |
| rs | 65536 | in | 31.399 | 31.798 | 1.012720 |
| rs | 131072 | out | 37.193 | 36.449 | 0.980020 |
| rs | 131072 | in | 36.203 | 36.803 | 1.016570 |
| rs | 262144 | out | 48.420 | 48.022 | 0.991780 |
| rs | 262144 | in | 48.829 | 48.037 | 0.983788 |
| rs | 524288 | out | 80.370 | 80.197 | 0.997846 |
| rs | 524288 | in | 80.179 | 80.740 | 1.006989 |

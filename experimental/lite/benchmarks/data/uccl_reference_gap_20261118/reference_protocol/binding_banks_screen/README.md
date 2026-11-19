# Allocation-aware graph bindings and 64-MiB double buffering

Two four-L4 hosts, FP32 SUM AR/RS, fixed 1-GiB allocations, both layouts,
20 warmups/10 timed calls, three counterbalanced repetitions. RS covers 8–128 MiB;
AR covers 8–64 MiB. Values are medians of maximum-rank per-call means with full
GPU and internal send retirement. The fixed descriptor-cache candidate is the
control. No per-size winner curve or native NCCL fallback is used.

The binding cache checks CUDA's process-unique allocation ID as well as the actual
source pointer. Only legacy-IPC-capable allocations are eligible; unknown kinds
continue to rebind UCCL's memcpy nodes on every call. Driver faults propagate.
An address reused for a different allocation invalidates the cache before updating
all affected nodes. The design uses [CUDA allocation ID guarantees](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__UNIFIED.html),
not pointer equality or cached input contents. Targets are private communicator
allocations whose lifetime already covers the graphs.

The separate buffer experiment extends existing alternating RS banks through 64 MiB,
retaining the 32-bit offset bound check. The combined candidate also includes the
previously measured 8-MiB remote-first schedule. All three pass the strengthened
216-call/rank ABI, retaining every old case and adding 64 MiB plus a delayed rank.
Each also passes 48 queued calls/rank with new GPU input each time, full output/guards,
offset changes and allocation recycling. Every rank actually reuses a VA and exercises
both descriptor branches. Binding-cache candidates record 40 hits, 8 updates and 2
recycled-allocation invalidations/rank in that queue test. All 48 performance jobs pass
full-output, exact route and native-absence checks.

The 64-MiB bank extension and binding cache improve several points, but the combination
still has remaining gaps against the original reference. No full zero-regression
admission or default release promotion is claimed; focused comparisons do not replace
the final paired full curve.

| Variant | RS bytes | Layout | Median(us) |
| --- | ---: | --- | ---: |
| uccl_graph_descriptor_cache | 8388608 | out | 512.009 |
| uccl_graph_descriptor_cache | 8388608 | in | 517.457 |
| uccl_graph_descriptor_cache | 16777216 | out | 911.558 |
| uccl_graph_descriptor_cache | 16777216 | in | 910.389 |
| uccl_graph_descriptor_cache | 33554432 | out | 1827.562 |
| uccl_graph_descriptor_cache | 33554432 | in | 1840.142 |
| uccl_graph_descriptor_cache | 67108864 | out | 3489.758 |
| uccl_graph_descriptor_cache | 67108864 | in | 3482.261 |
| uccl_graph_descriptor_cache | 134217728 | out | 6743.473 |
| uccl_graph_descriptor_cache | 134217728 | in | 6735.944 |
| uccl_graph_binding_cache | 8388608 | out | 503.653 |
| uccl_graph_binding_cache | 8388608 | in | 511.065 |
| uccl_graph_binding_cache | 16777216 | out | 909.007 |
| uccl_graph_binding_cache | 16777216 | in | 907.245 |
| uccl_graph_binding_cache | 33554432 | out | 1825.226 |
| uccl_graph_binding_cache | 33554432 | in | 1825.430 |
| uccl_graph_binding_cache | 67108864 | out | 3474.977 |
| uccl_graph_binding_cache | 67108864 | in | 3481.986 |
| uccl_graph_binding_cache | 134217728 | out | 6735.842 |
| uccl_graph_binding_cache | 134217728 | in | 6726.452 |
| uccl_rs_medium64 | 8388608 | out | 519.810 |
| uccl_rs_medium64 | 8388608 | in | 515.609 |
| uccl_rs_medium64 | 16777216 | out | 918.280 |
| uccl_rs_medium64 | 16777216 | in | 914.403 |
| uccl_rs_medium64 | 33554432 | out | 1835.783 |
| uccl_rs_medium64 | 33554432 | in | 1843.028 |
| uccl_rs_medium64 | 67108864 | out | 3456.771 |
| uccl_rs_medium64 | 67108864 | in | 3456.159 |
| uccl_rs_medium64 | 134217728 | out | 6746.267 |
| uccl_rs_medium64 | 134217728 | in | 6741.484 |
| uccl_graph_banks_combo | 8388608 | out | 506.302 |
| uccl_graph_banks_combo | 8388608 | in | 494.090 |
| uccl_graph_banks_combo | 16777216 | out | 913.435 |
| uccl_graph_banks_combo | 16777216 | in | 908.277 |
| uccl_graph_banks_combo | 33554432 | out | 1822.691 |
| uccl_graph_banks_combo | 33554432 | in | 1823.398 |
| uccl_graph_banks_combo | 67108864 | out | 3452.070 |
| uccl_graph_banks_combo | 67108864 | in | 3458.850 |
| uccl_graph_banks_combo | 134217728 | out | 6740.325 |
| uccl_graph_banks_combo | 134217728 | in | 6739.724 |

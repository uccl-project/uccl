# Graph setup and RS 8-MiB overlap experiments

Two four-L4 hosts, FP32 SUM AR/RS, fixed 1-GiB allocations, both layouts,
20 warmups/10 timed calls, three counterbalanced repetitions. Graph experiments
cover 8–64 MiB for AR and RS; remote-first RS covers 4–16 MiB. Values are medians
of maximum-rank per-call means including GPU and internal send retirement.
The fixed shared-host-event library is the control; no per-size best variant curve.
All three candidates pass 198 mixed ABI calls/rank and all 48 focused jobs pass
full-output, exact route and zero-native-fallback checks.

The descriptor cache retains only two raw GPU pointer values after its writer
has completed. Input contents are always read anew, and CUDA memcpy nodes are
still rebound every call. It has modest/layout-dependent benefits; the graph GPU
input-copy replacement is substantially slower and is rejected. Remote-first RS
reduces and sends the other node's half before reducing its own half, overlapping
IB transfer with local work; source regions are disjoint, arithmetic order and final
send retirement unchanged. RS 8 MiB improves from 502.392/501.429 us to 495.875/487.423 us
(out/in), but full-range final qualification is still required.

The original asynchronous nccl-tests probe passes with 53 descriptor reuses/rank
and no pending-writer branch hits. The added NCCL-only graph_queue client then
validates 48 queued calls/rank with fresh GPU-produced input and full output/guards,
changes offsets, and frees/reallocates input. Every rank sees one VA reuse and both
cache branches: 16 descriptor reuses and 27 pending-writer cases. Original memcpy
node rebinding remains active. Logs and hashes are preserved in queue_check/.

| Case | Variant | Bytes | Layout | Median(us) |
| --- | --- | ---: | --- | ---: |
| graph_rs | uccl_shared_host_event | 16777216 | out | 921.041 |
| graph_rs | uccl_shared_host_event | 16777216 | in | 931.817 |
| graph_rs | uccl_shared_host_event | 33554432 | out | 1876.981 |
| graph_rs | uccl_shared_host_event | 33554432 | in | 1839.638 |
| graph_rs | uccl_shared_host_event | 67108864 | out | 3482.938 |
| graph_rs | uccl_shared_host_event | 67108864 | in | 3488.288 |
| graph_rs | uccl_graph_descriptor_cache | 16777216 | out | 922.832 |
| graph_rs | uccl_graph_descriptor_cache | 16777216 | in | 912.304 |
| graph_rs | uccl_graph_descriptor_cache | 33554432 | out | 1831.103 |
| graph_rs | uccl_graph_descriptor_cache | 33554432 | in | 1839.538 |
| graph_rs | uccl_graph_descriptor_cache | 67108864 | out | 3484.357 |
| graph_rs | uccl_graph_descriptor_cache | 67108864 | in | 3474.831 |
| graph_rs | uccl_rs_graph_sm_input | 16777216 | out | 965.359 |
| graph_rs | uccl_rs_graph_sm_input | 16777216 | in | 966.885 |
| graph_rs | uccl_rs_graph_sm_input | 33554432 | out | 1954.090 |
| graph_rs | uccl_rs_graph_sm_input | 33554432 | in | 1945.427 |
| graph_rs | uccl_rs_graph_sm_input | 67108864 | out | 3747.431 |
| graph_rs | uccl_rs_graph_sm_input | 67108864 | in | 3739.562 |
| overlap_rs | uccl_shared_host_event | 8388608 | out | 502.392 |
| overlap_rs | uccl_shared_host_event | 8388608 | in | 501.429 |
| overlap_rs | uccl_rs8_remote_first | 8388608 | out | 495.875 |
| overlap_rs | uccl_rs8_remote_first | 8388608 | in | 487.423 |

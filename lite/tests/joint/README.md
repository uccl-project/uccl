# Frozen joint checks

The setup-frozen `benchmarks/joint/joint_bench.cu --selftest` exercises the public reusable peer/router selector with independent host expected values and asynchronous caller-stream producers/reuse. The benchmark performs full-output, input-preservation and guard checks on every ordinary invocation as well. See `benchmarks/joint/README.md` for exact numeric and timing contracts.

Run the joint selftest and existing `router_policy_check`, `dispatcher_check`, `cpu_bf16_check`, `router_core_check`, router eager and router graph checks on both supported nodes under the shared evaluation lock. A passing short benchmark is not a substitute for these checks or the held-out full-size evaluation. The standalone NCCL reference links no mutable implementation objects.

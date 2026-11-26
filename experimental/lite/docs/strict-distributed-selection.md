# Strict distributed selection without native NCCL

The common `make nccl-runtime` library now registers
`prepared-distributed-fp32` and `prepared-distributed` (BF16). For supported
multi-host layouts, `runtime_profile.py candidate` selects this automatic
provider instead of choosing a machine-specific implementation. Single-host
defaults and explicitly named legacy providers retain their existing contracts.

This remains an explicit candidate experiment: identity-matched profiles name
the exact operation/count/rank layouts before communicator initialization.
It is not a claim that the resulting runtime beats native NCCL.

## Selection and preparation

The provider uses general direct exchange as its native-free reference.
It discovers participants, peer access and active GPU–NIC paths using the same
executor as the native-backed portfolio. Each declared reference request gets
two warmups and three timed, fully checked calls on private guarded buffers.
This includes BF16, scalar tails and requests too small for hierarchy.

The shared `PreparedNetworkRuntime` then qualifies eligible hierarchy requests
against that reference with three ABBA rounds and the unchanged greater-than-3%
median/nonoverlapping-range rule. These are measurements against UCCL direct
exchange, **not NCCL**. The reference stays available for requests selecting it.
Its placement/transport identity is checked again on subsequent preparation.
Reference buffers/registrations are part of both measured configurations; only
the hierarchy's resources are added/removed in its paired trial.

All discovery, private validation, timing and transport initialization occur at
prepare. Enqueue uses the prepared request table. The executor and strict ABI
have no native NCCL dependency, create no native communicator, and keep rejecting
unsupported operations, groups and external graph capture. No native fallback
is introduced.

## Capability-driven profiles

The public catalog exposes host-count, total-rank and per-host-rank bounds,
plus `extent_multiple`. The generic distributed provider allows 2–32 total
ranks over 2–32 hosts, with positive per-host counts. Those are implementation
capacity bounds, not evidence that every physical deployment has been tested.

Byte granularity is now an implementation property. Generic distributed
requests allow scalar tails; the retained hierarchy and other legacy profiles
still require their recorded 128-byte extent granularity. Capability matching
does not bypass runtime transport checks or measured candidate selection.

V2 profiles retain host-major rank order. V3 profiles explicitly map logical
ranks to host/device pairs and support interleaving and reverse device order.
These versions require one consistent visible-device inventory per host.
[V4 per-rank profiles](strict-rank-profiles.md) now support isolated and reordered
CUDA views by binding logical ranks to UUIDs and individual probe identities.

## Identity probing

`--real-nccl` is optional for `runtime_profile.py probe` and `launch`. Without
it, the record marks `reference_kind=no-native-reference` and contains no native
artifact hash. CUDA driver/runtime and strict ABI compatibility versions remain
checked. Supplying a native reference file still binds its hash into the
identity, without loading it. Probe and launch must use the same choice.

For example, after building and setting the same launch environment on each
participating host:

```sh
python3 -B integration/nccl_lite/runtime_profile.py probe \
  --library build/nccl_runtime/libuccl_lite_nccl_runtime.so --output build/host.json
```

Generate the profile from the ordered host records and exact workload file,
optionally passing `--rank-map`, then launch through the same script. Multi-host
control still requires the root IPv4 setting described in `integration/nccl_lite/RUNTIME.md`.
`make strict-distributed-check` builds the MPI/CUDA correctness client; it does
not itself launch GPU tests.

Final evidence is in `build/strict-distributed-release/`: eleven strict runs
pass across the two recorded Xeon Silver 4410Y/four-L4 hosts, including 1+1,
2+2, 3+1, 3+3 and 4+4 layouts, FP32/BF16, minimal one-element profiles, scalar
tails, mixed aliases and two lifetimes. A wrong UUID is collectively rejected
before backend setup. The existing four-GPU strict FP32 checker still passes.
Two-host/four-rank BF16 memcheck reports zero device errors; only known
capability-probe API reports are suppressed. All successful strict runs verify
that no native NCCL image or fallback appears.

Eight-rank FP32 tests observe actual hierarchy selection with 18 reference and
18 hierarchy samples per eligible request. These use maximum-rank host completion,
three ABBA rounds, two warmups/three timed calls per block; setup and verification
are excluded. Preliminary reference health timings can include resources needed
by earlier plans and are not used to select hierarchy. There is no NCCL speedup
or best-variant-curve claim. Four current native-backed compatibility runs also
pass separately. Source/binary fingerprints are in
`build/strict-distributed-artifacts.sha256`.

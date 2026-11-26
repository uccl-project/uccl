# Strict per-rank GPU identity profiles

V4 profiles bind each logical rank to its owned GPU UUID and its own probed
software/visibility identity. They support different CUDA visibility on the
same host, including one visible GPU per process and different ordinal orders.
The strict library still loads no native NCCL implementation.

The distributed prepared providers explicitly advertise V4 in the capability
catalog. Older providers continue to accept their existing versions; V4 cannot
be used to bypass a provider's complete-view assumptions. V1–V3 retain their
host-record and device-index semantics.

## Probe and launch

For each participating process, use its actual visibility and owned **local**
CUDA ordinal. For example, logical rank zero owning physical GPU 3 with only
that GPU visible can produce its record as follows:

```sh
CUDA_VISIBLE_DEVICES=3 python3 -B integration/nccl_lite/runtime_profile.py probe \
  --library build/nccl_runtime/libuccl_lite_nccl_runtime.so \
  --rank 0 --device 0 --output build/rank0.json
```

Collect one record per logical rank. The candidate command takes those records
in logical rank order and derives host populations from them:

```sh
python3 -B integration/nccl_lite/runtime_profile.py candidate \
  --library build/nccl_runtime/libuccl_lite_nccl_runtime.so \
  --rank-records build/rank0.json build/rank1.json \
  --dtype float32 --shapes build/shapes.json --profile build/runtime.profile
```

Do not combine `--rank-records` with `--rank-map`. UUID ownership already supplies
the binding. At launch, pass the same process-specific `--rank`, `--device`,
visibility and environment used for its probe. The application must initialize
its communicator on that GPU. Optional native reference artifact handling and
the root control address follow the existing runtime launcher contract.

## Agreement and execution

The profile stores a provider and host populations, followed by one participant
row per logical rank: host index/name, identity hash, CUDA driver/runtime and
ABI compatibility versions, and owned GPU UUID. Exact request/layout rows use
the existing prepared format.

The builder verifies rank order, unique owned GPUs, the selected UUID within
the recorded visible inventory, and the identity's ownership binding. Launch
verifies the selected rank's complete record. During communicator initialization,
the C++ adapter independently checks current device UUID, versions, per-rank
identity and global profile agreement. Rank votes carry UUIDs; remote CUDA
ordinals are not compared with local ordinals.

Each process reconstructs only the participating GPUs visible in its own
inventory. Direct exchange may use IB where reciprocal IPC visibility is absent.
Hierarchy is considered only when every rank has a complete and consistent
per-host topology view. A uniform full view can still select hierarchy through
the existing checked timing comparison. Missing visibility does not become an
assumed peer edge or a fabricated bandwidth value.

This changes identity binding, not performance admission: exact workloads still
undergo direct-reference validation and eligible hierarchy comparison during
prepare. No discovery, transport creation or tuning is added to enqueue.

Validation is recorded in `build/strict-rank-view-verified/`: ten V4 cases pass,
covering two-host 2+2 and 4+4 isolated views, 4+4 reordered views, mixed 3+1 views,
uniform 4+4 views, four profile-rejection cases and two-host/four-rank FP32
memcheck with zero device errors. Tests check all AR/AG/RS inputs, outputs and
guards, mixed aliases, scalar tails and two lifetimes. Only known MSCCL++
capability-probe API reports are suppressed in memcheck. Native NCCL is not
loaded by strict runs. Four legacy and four native-backed compatibility cases
also pass. Source/staged hashes are in `build/strict-rank-view-artifacts.sha256`.

The recorded hardware is two Xeon Silver 4410Y/four-L4 hosts. Direct-reference
checks retain three timed observations after two warmups; hierarchy comparison
uses three ABBA rounds and 18/18 samples with maximum-rank host completion.
Setup/verification are outside timing. This is not a speedup claim over NCCL
or a best-variant curve, and does not validate larger physical deployments.

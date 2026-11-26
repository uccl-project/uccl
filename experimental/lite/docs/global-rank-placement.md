# Logical ranks and physical participants

The network registry now derives host membership from actual host identities and
GPU ownership from each participant's current CUDA device. It no longer requires
logical rank blocks to identify hosts or `global_rank % 4` to identify a GPU.
Each host must still contribute four distinct GPUs in two supported NUMA pairs.
Duplicate GPU ownership, inconsistent local inventory, or the wrong actual host
grouping rejects before transport initialization.

The execution plan groups the original root's host first and applies the NUMA
pair order within each host. Original logical ranks remain the application's
source/shard indices. AG/RS permutation and completion handling preserve those
indices, including aliases. NIC planning receives the resulting physical GPU
pairs. A stable inventory order lets unchanged hardware measurements remain
usable when only logical rank ownership changes.

## Profile binding

`UCCL_LITE_RUNTIME_PROFILE_V3` adds an explicit logical-rank map while retaining
hardware/software identity and exact workload checks. The map indexes host
records and CUDA-visible device ordinals; those host indices are not NUMA IDs.
The candidate tool accepts it through `--rank-map`. The frontend checks every
rank's actual hostname, CUDA device and UUID, rejects duplicate/out-of-range
bindings, and agrees on profile contents before transport setup.

V3 supports both explicit providers and automatic prepared providers. Automatic
requests keep their existing per-logical-rank alias/alignment descriptors and
independent qualification. The catalog advertises supported profile versions;
the Python tool does not emit V3 for an older frontend. V1/V2 and historical
headers retain their canonical host-contiguous binding semantics.

The NCCL-facing runtime still requires explicit candidate/workload profiles.
This is not profile-free automatic network algorithm admission, native fallback
or support for arbitrary host/rank counts.

## Validation

The direct-registry checker now uses MPI shared-host discovery for local CUDA
assignment. It checks canonical, interleaved (`--rank-by node`) and reversed GPU
assignment on two real four-L4 hosts, with complete FP32 buffers/guards, mixed
aliases, three data generations, two lifetimes and complete AG/RS extents through
32 MiB. The asserted interleaved execution map is `0,2,4,6,1,3,5,7`; reverse
assignment with stable inventories produces `6,4,2,0,7,5,3,1`. Both retain matching
path/cohort evidence. Logs are under `build/network-global-ranks/`.

`build/network-global-ranks-one-host-rejection.log` rejects eight processes on
one four-GPU host instead of inferring two hosts from rank count. The direct
duplicate-device case rejects all eight ranks before transport setup.
`build/rank-inventory-memcheck-final/run.log` validates interleaved logical hosts
and reverse GPU assignment with zero memory errors on all eight ranks. That
instrumented run covers 4096 elements, mixed aliases and two lifetimes; it is
correctness evidence, not performance measurement.

The public NCCL ABI tests cover the same assignments, including per-rank wrong
actual devices and a duplicate map introduced after launcher verification. They
retain the independent full-buffer reference and all route/identity assertions.
Final logs are `build/rank-inventory-final-{interleaved,reverse,v1,legacy}/`:
each passes 126 calls/rank through 32 MiB over two lifetimes on two four-GPU
hosts. The V3 cases also assert all-rank frontend rejection for wrong actual
devices and malformed duplicate maps. Stable distributed inventories preserve
matching isolated-path and concurrent-cohort evidence in both V3 cases.
The single-host FP32 prepared client also checks V3 reversed ownership and V2
compatibility through 65536 elements, with numerical reductions and bitwise AG.
Its final logs are `build/rank-inventory-local-final/{v3,v2}/run.log`.
CPU catalog tests cover valid maps, lengths, bounds, duplicate/type rejection and
unsupported frontend versions. Validation uses the documented dual-NUMA Xeon
Silver 4410Y/L4 systems; no new performance curve is claimed.

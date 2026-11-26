import json,csv,shutil,hashlib,time
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_default_release';measured=root/'build/uccl_rs16_graph512';out=root/'benchmarks/data/uccl_reference_gap_20261118/reference_protocol/qualified_release';out.mkdir(exist_ok=False)
audit=json.loads((base/'completion_audit.json').read_text());assert audit['passed']
for p in base.iterdir():
 if p.is_file() and p.suffix in ['.json','.log','.py','.sh']:shutil.copy2(p,out/p.name)
for sub in ['abi']:
 for p in (base/sub).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.log','.csv','.py','.txt','.tsv']:
   q=out/sub/p.relative_to(base/sub);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
policy=root/'build/uccl_policy_graph512'
for p in policy.iterdir():
 if p.is_file() and p.suffix in ['.json','.log']:
  q=out/'policy_build'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for name in ['elf_runtime_identity.py','verify_graph512_policy.py','finalize_graph512_default.py']:shutil.copy2(root/'build'/name,out/name)
for name in ['result.json','dry_run.log']:shutil.copy2(root/'build/policy_graph512_recipe_check'/name,out/'policy_build'/('recipe_'+name))
for name in ['batch_raw_log_audit.json','batch_checker_rejection_cases.json','full_diagnostic.json','environment.json','build_command.json']:
 shutil.copy2(measured/name,out/('measured_'+name))
for name in ['raw_log_audit.json','checker_rejection_cases.json','comparison.csv']:
 shutil.copy2(measured/'paired'/name,out/('paired_'+name))
deps=json.loads((base/'source_dependencies.json').read_text())
for name,digest in deps['local'].items():
 p=root/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 q=out/'source'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for name in ['integration/nccl_lite/MSCCLPP.md','integration/nccl_lite/POLICY.md','integration/nccl_lite/README.md','README.md']:
 p=root/name;q=out/'source'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
rows=list(csv.DictReader((measured/'paired/comparison.csv').open()))
text='''# Qualified strict UCCL Lite NCCL replacement

**All126/126 aligned size/layout points pass the zero-slowdown criterion versus
the frozen original README UCCL Lite implementations.** All9 unchanged nccl-tests
sweeps pass numerical, route and native-absence checks. The default Makefile policy
is now `graph512`, and the default library has been rebuilt and independently
validated. This is measured parity on the recorded configuration, not a guarantee
for other hardware, API modes or future runs.

Hardware: two hosts with four NVIDIA L4s each, two NUMA groups/host,
mlx5_0/mlx5_1 and ibp55s0f0. Dtype: FP32 copy for AG and FP32 SUM for AR/RS.
Coverage: every power of two from1KiB through1GiB, both legal out-of-place and
in-place layouts, one GPU/process, eager execution. Size is the complete AG
output, complete RS input, or AR buffer. Exact GPU UUIDs, CPU affinity, topology,
driver/dependency hashes and relevant environment are recorded in the host files.

Baseline: SHA-verified original README UCCL binaries and original process layouts
(AG four MPI processes/two GPUs each; AR/RS eight processes). The replacement uses
eight processes/one GPU each. Each candidate layout is compared with the original
recorded reference layout. The original AG native-NCCL control is disabled.
Neither candidate nor default-library runs load or execute native NCCL.

Three paired rounds alternate reference/out/in ordering. Warmups/timed calls are
AG20/30, AR20/10 and RS10/10. Each point is the median of three maximum-rank
per-call means with fixed pointers, matching data preparation and full GPU plus
internal send retirement. Preparation, MPI barriers and full-output verification
are outside timing. No samples are trimmed and no tolerance is added. One fixed
library supplies the entire curve; it is not a best-variant-per-size envelope.
Both hosts have no active nsys sessions before or after the campaign.

| Operation | Passing points | Worst candidate/original latency ratio |
| --- | ---: | ---: |
'''
for op in ['ag','ar','rs']:text+=f"| {op.upper()} | 42/42 | {audit['paired']['worst_latency_ratio'][op]:.9f} |\n"
text+='''
![Full paired comparison](../graph512_full/paired_ratios.png)

The9 stock nccl-tests jobs use20 warmups,30 timed iterations and3 repetitions per
operation, covering both layouts. Their batch timing comparison is diagnostic;
the separate aligned27-job campaign supplies the per-call admission above. The
independent auditors reconstruct all values from raw logs, verify binary identities,
finite positive timings, every sample index and exact rank/size routes. Eight
corrupted-log/manifest fixtures are rejected. All earlier slower results remain
in the neighboring experiment directories and progress record.

## Default artifact and build

```bash
make -j2 nccl-mscclpp-strict \\
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl \\
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
```

This selects `UCCL_LITE_NCCL_POLICY=graph512` without an override and produces
`build/nccl_lite/libuccl_lite_nccl_strict.so`, with SONAME `libnccl.so.2` and alias
`build/nccl_lite/strict/libnccl.so.2`. The old default artifacts were backed up
before rebuilding. The library has no native NCCL dependency, symbol forwarding
or fallback path; unsupported operations return errors.

'''
text+=f"Measured library SHA-256: `{audit['measured_library_sha256']}`.\n\nDefault library SHA-256: `{audit['default_library_sha256']}`.\n\nLoaded-runtime identity: `{audit['loaded_runtime_sha256']}`.\n\n"
text+='''The raw hashes differ because NVCC records a temporary source filename and ELF
build-ID metadata. The identity comparison retains every allocated section,
layout/program header, dynamic symbol/relocation, constant, code and CUDA fatbin;
only the GNU build-ID descriptor is masked, and non-allocated debug/symbol tables
are excluded. The default and measured loaded code/data match. Code/dynamic-data
mutation self-checks reject differences. This comparison does not alter any timing.
Default-build host records match the measured records except for the library hash.

The explicit policy rebuild verifies no-op behavior and policy-switch invalidation,
and builds the affected backend/component/network-check targets. The canonical
default library separately passes234 mixed ABI calls/rank over two lifetimes,
including delayed peers, mixed layouts, nonaligned pointers, stream changes,
allocation recycling and unsupported-mode checks. The measured candidate also
passes48 queued mixed RS/AR calls/rank with full outputs/guards and actual address
reuse. After rebuilding, the default library passes another unchanged nccl-tests
sweep for each operation over the full range; these are correctness/routing smoke
runs, not a replacement performance campaign.

## Use the verified placement/profile

The identity-checked launcher and profile for the canonical default were staged
on both recorded hosts under `build/uccl_default_release/abi`. The exact three
stock-test commands used for the default-library smoke can be repeated with:

```bash
bash build/uccl_default_release/reproduce_nccl_tests.sh
```

The script uses the strict library and unchanged test executables. `--real-nccl`
is only an API/provenance input to the launcher; strict launch removes the native
runtime path from its control variables and rejects a process that loads native
NCCL. Keep the recorded profile with its matching library/launcher and placement.
Other placements, dtypes, counts, caller graph capture, grouped data submission
and concurrent communicators are outside this qualification.

[Build, launcher and API scope](../../../../integration/nccl_lite/MSCCLPP.md)
are documented separately. Full measurement logs and exact commands are in the
[fixed-library report](../graph512_full/README.md); the machine-readable
[completion audit](completion_audit.json), default ABI/smoke logs, policy-build
records and complete local source snapshot are included here. External SDK and
MSCCL++ dependencies are hashed, not vendored.
'''
# This report is six directories below lite; use the correct relative usage link.
text=text.replace('../../../../integration/nccl_lite/MSCCLPP.md','../../../../../integration/nccl_lite/MSCCLPP.md')
(out/'README.md').write_text(text)
print(out)

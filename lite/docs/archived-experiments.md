# Archived experiments and measurement provenance

The working tree keeps current runtime documentation, buildable benchmark
harnesses, README figures and their required measurement inputs. Raw run logs,
trace dumps, old figure variants, frozen source copies, packaged wheels and
superseded progress reports were removed from the source distribution.

The [complete pre-cleanup tree](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite) retains those records at commit
`cea1faf6473fc76b5bd192db025f5291169d3b58`. Links to removed evidence resolve to that immutable revision.
Restore a historical runner and its inputs together when reproducing an old
experiment; its configuration and timings do not qualify current source.

| Archived material | Interpretation retained |
| --- | --- |
| Raw benchmark and integration runs | Hardware, dtype, timing scope, repetitions, correctness and negative results belong to each recorded configuration. |
| Frozen four/eight-rank joint fixtures and CE trials | Historical candidates; current joint harness and independent reference remain buildable. |
| vLLM/PCIeCCL serving experiment | Quality acceptance was not met; it does not establish a quality-preserving serving improvement for UCCL Lite. |
| Early strict-runtime and optimization journals | Superseded by the [current capability matrix](topology-runtime-status.md); failures and regressions remain in Git history. |
| Older plots and integration wheel | Historical presentation/package artifacts, not current binaries or calibration evidence. |

Current results still have explicit limits. CPU-router benefit depends on the
exact size, layout and topology; it is not a universal win over NCCL. Direct
cost-model observations include sizeable prediction errors; the model never
replaces measured admission. The [calibration report](direct-model-calibration.md)
retains those errors and its measurement scope.

See [benchmark inputs and reproduction](../benchmarks/README.md),
[current runtime capabilities](topology-runtime-status.md), and
[validation](validation.md). New run outputs and binary artifacts belong in
`build/`, outside source directories.

## Local-only records

Uncommitted local measurements were moved to the ignored
`build/cleanup-local-archive/` directory rather than discarded. These files
were never part of the published Git archive. References to local-only GPU
reduction/overlap trials and the local strict comparison report lead here;
the associated design notes retain their conclusions and limitations.

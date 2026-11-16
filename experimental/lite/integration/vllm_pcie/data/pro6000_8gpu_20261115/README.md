# Archived eight-GPU PCIeCCL/vLLM experiment, 2026-11-15

See [the experiment report](../../README.md) for methods, figures and the failed
quality acceptance. This is a historical external experiment, not validation
of a current UCCL Lite runtime change.

| Location | Contents |
| --- | --- |
| `results/final/` | Original final comparisons, report, completion/selection identities, quality/finite-input summaries and experimental review patches. |
| `results/final-validation/` | NCCL and c03 raw case metrics, manifests, parity and first quality samples. |
| `results/final-original/` | Original PCIeCCL raw case metrics, manifest and parity. |
| `results/quality-repeats/` | Planned sequence, all repeat result records and quality counts, including the failed gate. Large fingerprints stay in the source archive. |
| `results/finite-bitwise/` | Raw finite-input output logs, command records, results and comparison. |
| `results/eat/` | All ten configurations and eighteen scored evaluations, including rejected candidates. |
| `results/nccl/`, `results/pcieccl/`, `results/noise/` | Initial reproduction metrics and unchanged-baseline noise measurements (lossless `.json.gz`); excluded from final figures. |
| `forced-p2p/` | Fresh default/P2P case metrics, transport logs, manifests, runtime records, audit, comparison and original scripts. |
| `controller/`, `task/`, root scripts | Historical configurations, quality/retention protocol, summarizers, finite-input diagnostic and benchmark entry points. `task/` contains only selected documentation/scripts, not a vendored source tree. |
| `provenance.json` | Original archive path, SHA-256 and size for every copied file, and hashes of omitted large quality fingerprints. |

The source experiment is pinned to `8cf173918d7e46fdd9734fac4f7292e310439837`.
The selected c03 source SHA-256 is
`ce751fa67339d5163171721a350c09d8edc14876a2e89cb808349ba984f75175`;
the final source commit is `5dbd0666a626a44eeaee55680f846425fc87baa5`.
These are external provenance identities, not claims about ancestors of this
repository branch. The patches remain review artifacts and are not applied.

Original reports/audits reference some artifacts kept only in the source
archive, including the old figures, full source tree, server logs, startup
failure records and large fingerprints. Their original hash maps are preserved
unchanged; `provenance.json` identifies the files actually packaged here.
Checkpoints, dependencies, extensions and compiled artifacts are excluded.

Run `make vllm-pcie-figures` from `experimental/lite` to verify the packaged
hashes and raw serving data and regenerate the four README-style figures.

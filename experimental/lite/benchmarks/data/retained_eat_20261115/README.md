# Retained EAT measurements

See [the optimization document](../../../docs/retained-optimizations.md) for
implementation details, source identities, negative results, timing scope and
reproduction commands.

- `router-shm-cases.csv`: two full paired router validations on each L4 node.
- `pro6000-joint-cases.csv`: official full joint/P2P validation and its default
  and forced-SHM NCCL controls.
- `l41-peer-cases.csv`: two full pure-peer, NCCL P2P and default/SHM runs.
- `l41-peer-comparison.svg`: plot of those l41 measurements, including losses.
- `provenance.json`: historical source IDs and packaged data hashes.
- `integration-validation.json` and `validation-logs/`: correctness checks of
  this integration on the newer upstream base, separate from historical timings.
- `summarize.py`: dependency-free hash verification and aggregate calculation.

These data are per-case medians, not all individual timed samples. The historical
experimental source hashes are provenance identifiers, not Git ancestors of the
integration branch. No new integrated-branch performance or E2E result is claimed.

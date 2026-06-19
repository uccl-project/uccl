## Benchmark CI

Benchmark workflows run in three cases:

- A maintainer comments `/run-benchmark <target>` on a PR.
- An internal PR has the `run-benchmark` label.
- A commit is pushed to `main`, including by merging a PR.

Supported slash-command targets:

| Target | What it runs |
| --- | --- |
| `gh200` | Builds UCCL for NV GH200, then runs EP inter-node latency and NIXL P2P tests. |
| `gb10` | Builds UCCL for NV GB10, then runs EP inter-node latency and NIXL P2P tests. |
| `amd` | Builds UCCL for AMD MI325x, then runs P2P plus EP high-throughput and low-latency tests. |
| `amd-zip` | Builds UCCL for AMD MI325x with `USE_DIETGPU=1`, then runs P2P compression benchmarks. |
| `l4` | Builds and verifies UCCL on NV L4 host. |

Only collaborators with `write`, `maintain`, or `admin` permission can use `/run-benchmark`. External PRs should use this maintainer-approved path because label-triggered benchmark runs are restricted to internal PRs.

Slash-command benchmark runs post a follow-up PR comment with the workflow URL and final result. Label-triggered PR runs and push-to-`main` runs do not post PR comments.

### Permissions

`run-benchmark.yml` needs:

```yaml
permissions:
  actions: write
  contents: read
  issues: write
  pull-requests: write
```

Benchmark jobs that run PR code should keep only `contents: read`. Comment-only jobs may request `issues: write` and `pull-requests: write`.

`issue_comment` workflows use the version on the default branch, so changes to slash-command behavior must be merged to `main` before they take effect.

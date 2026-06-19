## Benchmark CI

Benchmark workflows run in three cases:

- A maintainer comments `/run-benchmark <target>` on a PR.
- An internal PR has the `run-benchmark` label.
- A commit is pushed to `main`, including by merging a PR.

Supported slash-command targets:

```text
l4
amd
amd-zip
gb10
gh200
```

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

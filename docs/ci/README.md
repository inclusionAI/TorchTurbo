# TorchTurbo CI

Current architecture and local validation commands:
[`scripts/ci/README.md`](../../scripts/ci/README.md).

The execution path is:

```text
lint -> Producer -> PlanArtifact -> Consumer -> JobResult -> Gate
```

- `scripts/ci/config/catalog.yml` declares images, venvs, runner lanes, pytest
  facts, and model E2E cases.
- `scripts/ci/config/policies.yml` selects the pull-request or nightly scope.
- `.github/workflows/ascend-pr.yml` is the main PR and reusable workflow entry.
- `.github/workflows/ascend-manual.yml` reduces the normal Plan to one exact job.
- `.github/workflows/ascend-nightly.yml` reuses the main workflow with the
  nightly policy.

## What does this PR do?

<!-- Provide a concise description of what this PR aims to achieve. -->

## Related Issue

<!-- Link the GitHub issue this PR addresses, e.g. "Fixes #123". Search open issues/PRs first to make sure this is not a duplicate. -->

Fixes #(issue)

## Type of Change

<!-- Select ONE that best matches your PR. This must match the prefix used in the PR title (required by CONTRIBUTING.md). -->

- [ ] 🐛 `fix`: Bug fix
- [ ] ✨ `feat`: New feature
- [ ] 📝 `docs`: Documentation update
- [ ] 🎨 `style`: Code style change (no functional change)
- [ ] ♻️ `refactor`: Refactoring
- [ ] ⚡ `enhance`: Existing-feature improvement (e.g. performance)
- [ ] ✅ `test`: Test addition or update
- [ ] 🔧 `build`: Build system or dependency change
- [ ] 👷 `ci`: CI change
- [ ] 🔧 `chore`: Chore
- [ ] ⏪ `revert`: Revert

**Breaking change?** If this PR changes any public API (config fields, CLI arguments, function signatures, etc.), check the box and describe the impact below.

- [ ] 💥 This PR introduces a breaking change

<!-- What breaks, and how should users migrate? -->

## Implementation Details

<!-- For non-trivial changes, briefly describe the design and list the specific changes by module. -->

## Test

<!-- How did you verify this change? -->
<!-- For changes covered by tests, paste the command and result, e.g. `pytest tests/<module>/`. -->
<!-- For changes that cannot be covered by CI (e.g. distributed training results on specific hardware), describe the experiments and show results (training curves, throughput, etc.). -->
<!-- If no test is needed, explain why. -->

```bash
# Paste test commands and outputs here
```

## Future Work

<!-- List known limitations, TODOs, or follow-up items intentionally left out of this PR's scope. -->
<!-- Open a GitHub issue for each follow-up and link it here so the work is tracked rather than lost. -->
<!-- Delete this section if nothing is planned. -->

## Checklist Before Submitting

- [ ] I have read the [Contributing Guide](https://github.com/inclusionAI/TorchTurbo/blob/main/CONTRIBUTING.md).
- [ ] The PR title starts with one of the required prefixes: `fix`, `feat`, `docs`, `style`, `refactor`, `enhance`, `test`, `build`, `ci`, `chore`, or `revert`.
- [ ] Commit messages use the same prefixes (e.g. `feat: add HSDP support`).
- [ ] I have run the relevant tests (`pytest tests/...`) and they pass, or explained in **Test** why they cannot.
- [ ] Documentation and examples are updated if this PR changes any API, config field, or workflow.
- [ ] My branch is up to date with `upstream/main` (`git fetch upstream && git rebase upstream/main`), and I push only to my fork.

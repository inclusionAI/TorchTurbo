# TorchTurbo Repository Guidelines

## Overview

This document is the top-level reference for any AI agent working in this repository.
Read it first.
For the project overview and directory layout, see `README.md`.

## Core Principles

1. **Prefer PyTorch-native training techniques.** Core TorchTurbo training infrastructure and parallelism code should rely on PyTorch libraries where practical. Techniques with moderate-to-large complexity belong in their proper upstream repo (pytorch/pytorch for parallelisms, pytorch/data for data loaders, etc.).

2. **Investigate root cause before patching.** Don't land band-aid fixes. Understand *why* something fails before proposing a solution. If a change seems to help but you can't explain why, dig deeper.

3. **Reuse over duplication.** Before writing new code, check if existing implementations already handle the case. Unify similar code paths across models rather than creating per-model wrappers. If upstream (torchao, PyTorch) already provides functionality, use it.

4. **Protect battle-tested code paths.** Be cautious changing converged behavior. Flag potential silent breakage of existing user code or checkpoints. When in doubt, ask.

5. **Audit all callsites.** When changing shared code (common model components, config fields, distributed utilities), check and update every callsite.

6. **No speculative defensive checks.** Don't add checks, casts, fallbacks, or conversions "just in case." Only validate explicit contracts, user-facing configuration, or invariants whose failure would otherwise be silent or unclear.

7. **Use project skills.** Before starting any task, list `.agents/skills/*/SKILL.md` and use the matching skill if present.

## Code Style

### Readability and Structure

1. **Blend in.** When modifying code, read the surrounding context and match the style of neighboring code (naming, spacing, patterns, idioms).
2. **Write beginner-friendly code.** Avoid deeply nested expressions or chained logic. Break complex expressions into clear, named intermediate steps.
3. **Explicit over implicit.** Inline logic where possible; avoid hiding behavior behind unnecessary layers of indirection.
4. **Optimize for the common path.** Prefer a small implementation for cases that exist today over a general solution for hypothetical edge cases.
5. **Keep the surface area small.** Every function, parameter, branch, and configuration option must justify its maintenance cost. When two approaches work, prefer the one with less code.
6. **Type hints** are required on all public API signatures (functions, methods, class attributes exposed in `__init__.py`).
7. **Avoid `hasattr` and `getattr`.** Their use is almost always a symptom of overly defensive programming or a disguised version check (e.g., "this attribute was added in version X"). Instead, either drop the conditional entirely or express the version check explicitly with a version comparison. There is nearly always a cleaner alternative.

### Naming

1. Names must be **accurate, descriptive, and reflect actual scope**. Don't use "toy/test/temp" in production names; put that context in docstrings instead.
2. Follow upstream conventions: match PyTorch naming where applicable.
3. Use the `num_` prefix for counts (e.g. `num_expert_groups`, not `n_expert_groups`) when not directly matching an upstream API.

### Comments and Documentation

1. **When to comment** Comment when the code is complex or when useful context is not apparent from the code itself, such as intent, rationale, assumptions, constraints, invariants, edge cases, or workarounds. **Write comments in English**.
2. **Comment focus** Write comments about the code's role, purpose, and behavior in the overall system, not the history or intent of the change that introduced or modified it.
3. **ASCII only** in newly added or rewritten code comments and docstrings. Don't introduce Unicode characters (e.g. smart quotes, em dashes, arrows, non-ASCII letters) in comments or docstrings you add or change.
   Use ASCII equivalents: `->` for arrows, `<-` for left arrows, `<->` for bidirectional arrows, `--` for em dashes. Leave preexisting Unicode in untouched comments alone; only enforce this for the comments and docstrings you are adding or rewriting.
4. **Docstrings** follow Google style. Put descriptions in docstrings, not in names.

## Git & PR Conventions

1. **Branch names** use the format `<type>/<short-desc>` (e.g. `fix/checkpoint-header-dtype`, `feat/fsdp2-optimizer-cpu-offload`).
2. - **Commit messages** follow Conventional Commits: `type(scope): description` e.g. `fix(engine): validate config files`.
   - Keep each commit focused; PR titles follow the same format.
   - Keep commit messages concise. Use short sentences.
   - Wrap commit message body lines at 72 characters. Indent continued
     bullet lines by two spaces.
   - Describe only the core implementation idea.
   - Valid types: `feat` `fix` `docs` `style` `refactor` `perf` `test` `build` `ci` `chore` `revert`.
   - Under `## Summary`, list only the important code changes, with one `- ` bullet per change. Do not put effects, benefits, performance results, or motivation there.
   - Use `## Issue` or `## Root cause` only when the problem or its cause
     needs explanation. State the problem and cause directly in simple
     sentences.
   - Do not add `## Validation`, `## Testing`, or similar sections to
     commit messages. Put test commands, results, and validation limits
     in the task response instead.
   - Write commit messages in proper English, with care and punctuation.
3. Titles must be <= 80 characters.
4. **Don't commit unless asked.** Do not commit unless the user explicitly requests it.
5. **Never** commit credentials, local model paths, checkpoints, or generated training outputs.

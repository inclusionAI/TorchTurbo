---
name: torchturbo-add-config
description: Add or update a TorchTurbo user-facing configuration surface across EngineConfig, YAML, TurboArgSpec, CLI, environment variables, examples, and tests. Use this whenever work mentions adding or changing a setting, knob, parameter, option, config field, config path, args.py entry, --tt_* flag, TORCH_TURBO_* variable, or scripts/runs/config_example.yaml, even when the user names only one of those surfaces.
metadata:
  short-description: Add a TorchTurbo configuration surface
---

# TorchTurbo configuration surface integration

Use this skill to introduce or change a TorchTurbo configuration surface. A configuration surface is the complete set of ways a user can supply one setting: `EngineConfig`/YAML, `TorchTurboArgs`, `TurboArgSpec`, `--tt_*` CLI, `TORCH_TURBO_*` environment variable, the example config, and tests.

This skill is not for unrelated `EngineConfig` refactors or for debugging an existing option. It applies when the user wants to expose new behavior to users or change where an existing setting lives.

Start from the repository root and read `AGENTS.md`. Before editing, inspect:

- `src/torchturbo/engine/args.py`
  - `TorchTurboArgs` for values that adapters can seed and runtime code can consume
  - `TurboArgSpec` for the registry that drives parser and config input behavior
  - `parse_torchturbo_args()` for explicit-CLI handling and runtime-only exclusions
- `src/torchturbo/engine/config.py`
  - the target nested `EngineConfig` dataclass
  - `build_from_args()` and its four-layer merge
  - `_validate_config_file()`, which uses spec paths to decide which YAML keys are valid
  - the relevant `_build_*_config()` helper
- `tests/engine/test_args.py`
- `tests/engine/test_engine_config_loading.py`
- `scripts/runs/config_example.yaml`
- every callsite when changing an existing field or shared configuration group

## Get explicit confirmation first

Do not edit configuration code before presenting a complete design and receiving the user's confirmation. This prevents quietly changing the public interface or adding redundant inputs.

Use this template:

```markdown
## Proposed TorchTurbo config surface

- Setting and behavior:
- Default:
- File-backed EngineConfig path: `<dotted.path>` / runtime-only
- Target dataclass and field:
- CLI: yes/no; name: `--tt_<name>`; type/choices/action:
- Environment variable: yes/no; name: `TORCH_TURBO_<NAME>`; reason:
- TorchTurboArgs field: yes/no; field/default:
- Example YAML location: `<path and nested location>`
- Focused tests:
- Compatibility or migration concerns:
```

Give a concrete recommendation for every line. Wait for confirmation. If the user already supplied every decision in the current conversation, restate the completed checklist and still ask them to confirm it before editing.

## Input exposure policy

Use these defaults when making a recommendation:

1. **YAML / EngineConfig:** A normal training-behavior setting is file-backed. Put it under the most specific existing nested group. Do not create a flat root key when a semantically cohesive nested group exists.
2. **CLI:** A normal user-facing training setting gets a `--tt_*` flag. Omit the flag only for uncommon model- or deployment-specific configuration whose primary interface is a config file.
3. **ENV:** Do not add a `TORCH_TURBO_*` variable by default. Recommend one only for an operational switch, compatibility with an existing variable, or a deployment context where editing the launcher command or YAML is impractical. Explain the maintenance cost: another supported name, another precedence layer, and another test surface.
4. **Runtime-only:** A runtime-only input has `config_path=None`. It is not an `EngineConfig` or YAML value. It is consumed directly from `TorchTurboArgs` (for example profiling controls, log level, or plugin paths). `parse_torchturbo_args()` deliberately excludes these destinations from `config_overrides` so they cannot leak into `EngineConfig`. Recommend runtime-only only when the input controls how a run is launched or observed rather than configuring the engine itself.
5. **`args_name`:** For a new normal config-backed CLI setting, add a matching `TorchTurboArgs` field and set `args_name`; this makes defaults and adapter seeding explicit. ENV-only specs generally omit `args_name` unless an adapter must seed the value. Existing CLI specs may omit `args_name` when the config builder only needs the explicit CLI override and no args-level default exists; do not copy that exception without explaining it and getting confirmation.
6. **Compatibility:** Renaming or moving a public key, flag, or environment variable is a migration. Call it out in the design checklist and ask whether compatibility is required.

Prefer one canonical nested YAML location. Avoid exposing the same new setting at two YAML paths unless backward compatibility explicitly requires it.

## Implementation workflow

After confirmation:

1. **Define the canonical config model.**
   - Add or change the nested config dataclass in `src/torchturbo/engine/config.py`, near its related config types.
   - Keep optionality explicit. Use `None` when absence must preserve a value from YAML or another layer.
   - Add the field to `EngineConfig` at the appropriate nested position.
   - Update the relevant `_build_*_config()` helper. Add a new helper only for a genuinely new cohesive section.
   - Audit and update every `EngineConfig` consumer affected by the new field.
2. **Add `TorchTurboArgs` support when needed.**
   - Match the surrounding field order and documentation style.
   - Use `None` for an optional override; use a concrete default only when the args layer should always provide that base value.
   - Document units, valid values, and when the setting is ignored.
3. **Register `TurboArgSpec`.**
   - Place it with related specs; do not append unrelated settings to the end.
   - Use stable snake_case `name`, a canonical `config_path`, the confirmed CLI name, and (only if approved) the confirmed environment name.
   - Set `type` for every non-string parsed input. Use `string_to_bool` for booleans.
   - Use `choices` only when the valid values are fixed. Put user-facing constraints and defaults in `help`.
   - Use `action="append"` for repeatable list flags. Remember that parser behavior and `TorchTurboArgs` typing differ for append flags.
   - Leave `config_path` unset for runtime-only inputs. Never invent a YAML path merely to make validation pass.
4. **Check parsing and merge behavior.**
   - `config_overrides` keys are argparse destinations derived from `cli_name`, not `config_path`.
   - A spec without `args_name` can still contribute an explicit CLI override if it has `config_path`; verify that this is intentional.
   - Runtime-only CLI destinations must remain excluded from `config_overrides`.
   - Confirm `_validate_config_file()` accepts the intended YAML path. File-only nested dataclass fields may require the existing dataclass-schema allowlist.
5. **Update `scripts/runs/config_example.yaml`.**
   - Every normal file-backed setting must appear in this example.
   - Place it in the matching nested section, near related fields.
   - Use the confirmed default and a short comment explaining valid values and behavior.
   - Keep the example loadable; `test_config_example_loads` is not a substitute for schema validation.
   - Do not add ENV-only or runtime-only controls to this YAML.
6. **Add focused tests.**

## Focused test policy

Current tests do not exhaustively test every one of the registered argument specs with every precedence combination. Do not create that duplicated matrix for a normal new setting. Test the behavior that this new surface actually adds.

Use the minimum useful set:

- **Parser:** One focused test in `tests/engine/test_args.py` when CLI parsing, typing, boolean forms, choices, append behavior, defaults, or runtime-only exclusion could fail.
- **File-backed config:** One focused test in `tests/engine/test_engine_config_loading.py` that loads the canonical YAML path and asserts the built nested `EngineConfig` value.
- **CLI override:** If a CLI is added, test that the explicit flag reaches the intended nested config value. Include a different YAML value when the mapping or precedence is nontrivial.
- **ENV override:** If an environment variable is added, test that it reaches the intended path. Use a different YAML value when the environment/file boundary could regress.
- **Validation:** Add or update unknown-key or invalid-value coverage when introducing a new nested block, changing accepted keys, or relying on dataclass-based validation.
- **Runtime-only:** Assert that the parsed field is available to its consumer and that its CLI destination is absent from `config_overrides`.
- **Example:** `tests/engine/test_engine_config_loading.py::test_config_example_loads` should continue to pass after the YAML update.

Only add all four generic precedence tests for a new setting when the change also modifies shared precedence logic. Otherwise, rely on the existing generic tests and test the new mapping once or twice.

Run the focused tests before claiming completion. At minimum, run the touched test files, and include the specific tests related to the new surface. If dependencies or hardware make a test unavailable, say which tests were not run and why.

## Final review checklist

Before reporting completion, verify:

- The user confirmed every proposed input surface and canonical path.
- `EngineConfig`, YAML, `TurboArgSpec`, `TorchTurboArgs`, parser behavior, and consumers agree.
- No redundant ENV variable or CLI flag was introduced without justification.
- Runtime-only inputs do not enter `config_overrides`.
- `scripts/runs/config_example.yaml` is updated only for file-backed settings and remains loadable.
- Focused tests cover the new mapping and any special parser behavior.
- No unrelated formatting or behavior changes were introduced.

"""Run integration cases: prepare assets, launch a framework, then compare metrics.

A suite module (``base_suite.py`` for instance) answers "which cases belong to this
suite, and how is each one judged"; the case YAML answers "what model and what
run configuration". This runner joins the two, executes them, and reports.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from gen_init_checkpoint import REPO_ROOT, asset_path, generate_checkpoint

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.consumer.runtime.model_dataset import (  # noqa: E402
    allow_download_from_env,
    resolve_model_dataset_args,
)

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
TEST_CASES_ROOT = INTEGRATION_ROOT / "test_cases"

MODES = {"train"}
TEST_TYPES = {"regular", "frozen-start"}
# Each test type names whether the run starts from a freshly initialized
# checkpoint. Types that start from an existing checkpoint add their own entry
# here when the runner learns to fetch one.
GENERATES_CHECKPOINT = {"regular": True, "frozen-start": False}
CASE_MAPPING_KEYS = {
    "MODEL_ARGS",
    "TRAIN_ARGS",
    "ENGINE_CONFIG",
    "LAUNCH_ARGS",
    "ENV_VARS",
}
CASE_KEYS = CASE_MAPPING_KEYS | {"UPDATE_GOLDEN"}


@dataclass(frozen=True)
class TestCaseSpec:
    """One case's pass/fail semantics; its model and run config live in the case YAML."""

    test_name: str
    """Case directory under ``test_cases``: ``<model>/<case>``."""
    test_descr: str
    """Human-readable one-liner printed in the run header and on failure."""
    framework: str
    mode: str
    test_type: str
    compare: bool
    metrics: dict[str, dict[str, float]]
    ngpu: int = 2
    platform: str = "npu"
    targets: tuple[str, ...] = ()

    @property
    def config_path(self) -> Path:
        """Return the case YAML this spec reads."""
        return TEST_CASES_ROOT / self.test_name / "config.yaml"

    @property
    def output_name(self) -> str:
        """Return the flat name used for the run output directory."""
        return self.test_name.replace("/", "_")


def load_yaml(path: Path, required: set[str]) -> dict[str, Any]:
    """Read a YAML mapping and require exactly the expected top-level keys."""
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise TypeError(f"{path} must contain a mapping")
    missing = required - set(document)
    extra = set(document) - required
    if missing or extra:
        raise ValueError(f"{path}: missing={sorted(missing)}, unexpected={sorted(extra)}")
    return document


def load_case(spec: TestCaseSpec) -> dict[str, Any]:
    """Merge a suite spec with its case YAML into the flat case the run consumes."""
    case = load_yaml(spec.config_path, CASE_KEYS)
    for name in CASE_MAPPING_KEYS:
        if not isinstance(case[name], dict):
            raise TypeError(f"{spec.config_path}: {name} must be a mapping")
    case.update(
        {
            "FRAMEWORK": spec.framework,
            "MODE": spec.mode,
            "TEST_TYPE": spec.test_type,
            "COMPARE": spec.compare,
            "METRICS": spec.metrics,
        }
    )
    if case["MODE"] not in MODES:
        raise ValueError(f"MODE must be one of {sorted(MODES)}; got {case['MODE']!r}")
    if case["TEST_TYPE"] not in TEST_TYPES:
        raise ValueError(
            f"TEST_TYPE must be one of {sorted(TEST_TYPES)}; got {case['TEST_TYPE']!r}"
        )
    if not isinstance(case["COMPARE"], bool) or not isinstance(
        case["UPDATE_GOLDEN"], bool
    ):
        raise TypeError("COMPARE and UPDATE_GOLDEN must be booleans")
    return case


def validate_training(case: dict[str, Any], model_key: str) -> None:
    """Check runner modes and explicit training inputs."""
    args = case["TRAIN_ARGS"]
    if not args.get("dataset"):
        raise ValueError("TRAIN_ARGS.dataset must explicitly name training data")
    if not args.get(model_key) and not args.get("tokenizer"):
        raise ValueError(f"TRAIN_ARGS.{model_key} or TRAIN_ARGS.tokenizer is required")
    if not GENERATES_CHECKPOINT[case["TEST_TYPE"]] and not args.get(model_key):
        raise ValueError(
            f"TEST_TYPE={case['TEST_TYPE']} requires TRAIN_ARGS.{model_key}"
        )


def cli_args(args: dict[str, Any]) -> list[str]:
    """Translate scalar, list, and JSON-valued training arguments without a shell."""
    result = []
    for name, value in args.items():
        if value is None:
            continue
        result.append(f"--{name}")
        if isinstance(value, dict):
            result.append(json.dumps(value))
        else:
            values = value if isinstance(value, list) else [value]
            result.extend(
                str(item).lower() if isinstance(item, bool) else str(item)
                for item in values
            )
    return result


def run_case(
    spec: TestCaseSpec,
    golden_path: Path,
    *,
    output_dir: Path,
    skip_install: bool,
    forbid_update_golden: bool,
) -> Path:
    """Run one isolated case; never reuse logs or mutate input assets."""
    case = load_case(spec)
    if forbid_update_golden and case["UPDATE_GOLDEN"]:
        raise ValueError(f"{spec.test_name}: CI must not update golden values")
    framework = case["FRAMEWORK"]
    if framework == "swift":
        model_key = "model"
        run_framework = run_swift_training
    elif framework == "llamafactory":
        model_key = "model_name_or_path"
        run_framework = run_llamafactory_training
    else:
        raise NotImplementedError(f"Training is not implemented for {framework}")
    validate_training(case, model_key)
    if not skip_install:
        install_dependencies(spec)
    if case["COMPARE"] and not case["UPDATE_GOLDEN"] and not golden_path.is_file():
        raise FileNotFoundError(
            f"No {golden_path.name}; set UPDATE_GOLDEN to establish a baseline"
        )
    env = os.environ.copy()
    env.update(
        {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in case["ENV_VARS"].items()
        }
    )
    # Apply environment before importing torch/tokenizers.
    os.environ.update(env)
    args, resolved_inputs = resolve_model_dataset_args(
        case["TRAIN_ARGS"],
        framework=framework,
        root=INTEGRATION_ROOT,
        allow_download=allow_download_from_env(env),
    )
    resolved_case = {**case, "TRAIN_ARGS": args}
    from transformers import AutoTokenizer

    if args.get(model_key):
        model_path = Path(args[model_key])
        tokenizer_path = Path(args["tokenizer"]) if args.get("tokenizer") else model_path
    else:
        model_path = None
        tokenizer_path = Path(args["tokenizer"])
    AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / spec.output_name / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(case, sort_keys=False))
    if resolved_inputs:
        (run_dir / "resolved_inputs.json").write_text(
            json.dumps({"inputs": list(resolved_inputs)}, indent=2) + "\n",
            encoding="utf-8",
        )
    engine_path = run_dir / "tt_config.yaml"
    engine_path.write_text(
        yaml.safe_dump({"engine_config": case["ENGINE_CONFIG"]}, sort_keys=False)
    )
    if model_path is None:
        model_path = generate_checkpoint(
            resolved_case, run_dir / "initial_checkpoint", engine_path
        )
    args.pop("tokenizer", None)
    args.update(
        {
            "tt_config": str(engine_path),
            "output_dir": str(run_dir / "training"),
            "logging_dir": str(run_dir / "tensorboard"),
        }
    )
    env["TENSORBOARD_LOGGING_DIR"] = str(run_dir / "tensorboard")
    run_framework(args, model_path, case["LAUNCH_ARGS"], env, run_dir)
    if case["COMPARE"] or case["UPDATE_GOLDEN"]:
        compare_metrics(run_dir, golden_path, case["UPDATE_GOLDEN"])
    return run_dir


def run_swift_training(
    args: dict[str, Any],
    model_path: Path,
    launch_args: dict[str, Any],
    env: dict[str, str],
    run_dir: Path,
) -> None:
    """Prepare Swift file datasets and environment, then launch SFT."""
    args["model"] = str(model_path)
    dataset_values = (
        args["dataset"] if isinstance(args["dataset"], list) else [args["dataset"]]
    )
    args["dataset"] = [str(asset_path(value)) for value in dataset_values]
    pythonpath = str(REPO_ROOT / "src")
    if env.get("SWIFT_SRC_DIR"):
        pythonpath += f":{env['SWIFT_SRC_DIR']}"
    if env.get("PYTHONPATH"):
        pythonpath += f":{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    launch_training(
        [str(REPO_ROOT / "scripts/runs/run_swift.py"), "sft"],
        args,
        launch_args,
        env,
        run_dir,
    )


def run_llamafactory_training(
    args: dict[str, Any],
    model_path: Path,
    launch_args: dict[str, Any],
    env: dict[str, str],
    run_dir: Path,
) -> None:
    """Prepare LLaMA-Factory dataset directory and environment, then launch training."""
    args["model_name_or_path"] = str(model_path)
    if args.get("dataset_dir"):
        args["dataset_dir"] = str(asset_path(args["dataset_dir"]))
    source_dir = env.get("TORCHTURBO_SRC_DIR") or str(REPO_ROOT / "src")
    pythonpath = f"{source_dir}:/opt/MindSpeed"
    if env.get("PYTHONPATH"):
        pythonpath += f":{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    launch_training(
        [str(REPO_ROOT / "scripts/runs/run_lf.py")], args, launch_args, env, run_dir
    )


def launch_training(
    entry_args: list[str],
    args: dict[str, Any],
    launch_args: dict[str, Any],
    env: dict[str, str],
    run_dir: Path,
) -> None:
    """Launch distributed training and stream output to the console and training.log."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *cli_args(launch_args),
        *entry_args,
        *cli_args(args),
    ]
    print(f"Run directory: {run_dir}\n{shlex.join(command)}", flush=True)
    with (
        (run_dir / "training.log").open("w") as log,
        subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process,
    ):
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def compare_metrics(run_dir: Path, golden_path: Path, update_golden: bool) -> None:
    """Delegate the pass/fail verdict to compare_metrics.py; its exit code decides."""
    compare_script = Path(__file__).resolve().parent / "compare_metrics.py"
    command = [sys.executable, str(compare_script), str(run_dir), str(golden_path)]
    if update_golden:
        command.append("--update-golden")
    subprocess.run(command, check=True)


def select_cases(
    args: argparse.Namespace, specs: list[TestCaseSpec]
) -> list[TestCaseSpec]:
    """Keep the specs this machine, this GPU count, and the filters allow."""
    selected = []
    for spec in specs:
        if args.test_name != "all" and spec.test_name != args.test_name:
            continue
        if spec.platform != args.platform:
            print(
                f"Skip {spec.test_name}: declared platform is {spec.platform}, "
                f"runner platform is {args.platform}",
                flush=True,
            )
            continue
        if spec.ngpu > args.ngpu:
            print(
                f"Skip {spec.test_name}: needs {spec.ngpu} devices, {args.ngpu} available",
                flush=True,
            )
            continue
        selected.append(spec)
    return selected


def case_manifest(spec: TestCaseSpec) -> dict[str, Any]:
    """Describe one case for CI without running it."""
    case = load_case(spec)
    golden_path = spec.config_path.parent / f"golden_values_{spec.platform}.json"
    manifest = {
        "case_id": spec.test_name,
        "description": spec.test_descr,
        "framework": spec.framework,
        "platform": spec.platform,
        "ranks": spec.ngpu,
        "mode": spec.mode,
        "test_type": spec.test_type,
        "compare": spec.compare,
        "update_golden": case["UPDATE_GOLDEN"],
        "config": spec.config_path.relative_to(REPO_ROOT).as_posix(),
        "golden": golden_path.relative_to(REPO_ROOT).as_posix(),
    }
    if spec.targets:
        manifest["targets"] = list(spec.targets)
    return manifest


def install_dependencies(spec: TestCaseSpec) -> None:
    """Run the dependency installer this case needs for its model and framework."""
    # TEMPORARY: the installer chain sources deps/platform.sh, which exits
    # unless it finds an NPU or a GPU. CPU cases run against a hand-prepared
    # environment instead.
    if spec.platform == "cpu":
        print(f"Skip dependency install on {spec.platform}: prepare the env manually", flush=True)
        return
    model_dir = spec.test_name.partition("/")[0]
    script_name = (
        "install_swift_deps.sh"
        if spec.framework == "swift"
        else "install_lf_deps.sh"
    )
    script = REPO_ROOT.parent / "internal/scripts/runs" / model_dir / script_name
    print(f"Installing dependencies: {script}", flush=True)
    subprocess.run(["bash", str(script)], cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        required=True,
        help="Suite module name under test_suite/, e.g. base_suite",
    )
    parser.add_argument("--platform", help="Platform this run targets")
    parser.add_argument("--ngpu", type=int, help="Devices available")
    parser.add_argument("--test_name", default="all", help="One case name, or all")
    parser.add_argument(
        "--list-cases", action="store_true", help="List suite cases as JSON"
    )
    parser.add_argument(
        "--json", action="store_true", help="Use JSON output with --list-cases"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Use the caller's prepared environment instead of installing dependencies",
    )
    parser.add_argument(
        "--forbid-update-golden",
        action="store_true",
        help="Fail before training if the case would rewrite its golden baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INTEGRATION_ROOT / "outputs",
        help="Directory for isolated case run outputs",
    )
    args = parser.parse_args()
    # Import the selected suite from the integration test_suite directory.
    sys.path.insert(0, str(INTEGRATION_ROOT / "test_suite"))
    suite = importlib.import_module(args.suite)
    if args.list_cases:
        payload = {
            "schema_version": 1,
            "suite": args.suite,
            "cases": [case_manifest(spec) for spec in suite.TEST_CASES],
        }
        if args.json or not sys.stdout.isatty():
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for case in payload["cases"]:
                print(f"{case['case_id']}\t{case['platform']}\t{case['ranks']}")
        return
    if not args.platform or args.ngpu is None:
        parser.error("--platform and --ngpu are required unless --list-cases is used")
    if args.ngpu < 1:
        raise ValueError("--ngpu must be positive")
    specs = select_cases(args, suite.TEST_CASES)
    if not specs:
        raise ValueError("No case matches the requested suite and filters")
    for spec in specs:
        golden_path = (
            spec.config_path.parent / f"golden_values_{spec.platform}.json"
        )
        print(f"\n===== {spec.test_descr} ({spec.test_name}) =====", flush=True)
        run_case(
            spec,
            golden_path,
            output_dir=args.output_dir,
            skip_install=args.skip_install,
            forbid_update_golden=args.forbid_update_golden,
        )


if __name__ == "__main__":
    main()

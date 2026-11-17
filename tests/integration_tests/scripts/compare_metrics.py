"""Extract per-step TensorBoard metrics and compare or explicitly update golden."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

SNAPSHOT_KEYS = {"METRICS", "TRAIN_ARGS"}


def read_snapshot(run_dir: Path) -> dict[str, Any]:
    """Read the merged case the runner wrote before training started."""
    snapshot = yaml.safe_load((run_dir / "config.yaml").read_text())
    if not isinstance(snapshot, dict) or not SNAPSHOT_KEYS <= set(snapshot):
        raise ValueError(f"Snapshot must include {sorted(SNAPSHOT_KEYS)}")
    return snapshot


def validate_metrics(metrics: dict[str, Any], case: dict[str, Any]) -> None:
    """Require exactly the configured finite, non-empty metric curves."""
    if set(metrics) != set(case["METRICS"]):
        raise ValueError(
            f"Metric tags must equal {sorted(case['METRICS'])}; got {sorted(metrics)}"
        )
    for tag, points in metrics.items():
        if not points:
            raise ValueError(f"{tag}: no steps were recorded")
        for step, value in points.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{tag} step={step}: expected finite scalar, got {value!r}"
                )


def extract_metrics(
    event_dir: Path, case: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Read only this run's event files, retaining all steps and detecting conflicts."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util

    metrics: dict[str, dict[str, float]] = {tag: {} for tag in case["METRICS"]}
    event_files = sorted(event_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {event_dir}")
    for path in event_files:
        events = EventAccumulator(
            str(path),
            size_guidance={"scalars": 0, "tensors": 0},
            purge_orphaned_data=False,
        )
        events.Reload()
        for tag, points in metrics.items():
            values = []
            # The framework logs under its own namespace; accept either spelling.
            for event_tag in (tag, f"train/{tag}"):
                if event_tag in events.Tags()["scalars"]:
                    values.extend(
                        (event.step, event.value) for event in events.Scalars(event_tag)
                    )
                if event_tag in events.Tags()["tensors"]:
                    for event in events.Tensors(event_tag):
                        tensor = tensor_util.make_ndarray(event.tensor_proto)
                        if tensor.size != 1:
                            raise ValueError(
                                f"{event_tag} step={event.step}: expected scalar tensor"
                            )
                        values.append((event.step, float(tensor.item())))
            for step, value in values:
                key = str(step)
                if key in points and points[key] != value:
                    raise ValueError(
                        f"Conflicting duplicate {tag} step={step}: {points[key]} vs {value}"
                    )
                points[key] = value
    validate_metrics(metrics, case)
    return metrics


def compare_metrics(
    actual: dict[str, dict[str, float]],
    golden: dict[str, dict[str, float]],
    case: dict[str, Any],
) -> list[str]:
    """Return every point outside the configured absolute-plus-relative tolerance."""
    validate_metrics(actual, case)
    validate_metrics(golden, case)
    failures = []
    for tag, tolerance in case["METRICS"].items():
        if set(tolerance) != {"atol", "rtol"}:
            raise ValueError(f"{tag}: specify atol and rtol")
        atol, rtol = tolerance["atol"], tolerance["rtol"]
        if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
            raise ValueError(f"{tag}: tolerances must be finite and nonnegative")
        actual_steps = set(actual[tag])
        golden_steps = set(golden[tag])
        if actual_steps != golden_steps:
            missing = sorted(golden_steps - actual_steps, key=int)
            extra = sorted(actual_steps - golden_steps, key=int)
            failures.append(
                f"{tag}: missing steps={missing}, unexpected steps={extra}"
            )
            continue
        for step in sorted(actual_steps, key=int):
            value, reference = actual[tag][step], golden[tag][step]
            error = abs(value - reference)
            allowed = atol + rtol * abs(reference)
            if error > allowed:
                failures.append(
                    f"{tag} step={step}: actual={value:.12g}, golden={reference:.12g}, "
                    f"abs_error={error:.12g}, allowed={allowed:.12g}"
                )
    return failures


def process_run(run_dir: Path, golden_path: Path, update_golden: bool = False) -> None:
    """Compare or update golden after the caller confirms training succeeded."""
    case = read_snapshot(run_dir)
    actual = extract_metrics(run_dir / "tensorboard", case)
    # Validate tolerances even when updating rather than comparing.
    compare_metrics(actual, actual, case)
    if update_golden:
        train_args = case["TRAIN_ARGS"]
        max_steps = train_args.get("max_steps")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("Updating golden requires an explicit positive TRAIN_ARGS.max_steps")
        if (
            train_args.get("logging_strategy", "steps") != "steps"
            or train_args.get("logging_steps") != 1
        ):
            raise ValueError("Updating golden requires logging_strategy=steps and logging_steps=1")
        expected_steps = {str(step) for step in range(1, max_steps + 1)}
        for tag, points in actual.items():
            actual_steps = set(points)
            if actual_steps != expected_steps:
                missing = sorted(expected_steps - actual_steps, key=int)
                extra = sorted(actual_steps - expected_steps, key=int)
                raise ValueError(
                    f"{tag}: missing steps={missing}, unexpected steps={extra}"
                )
    result = {"metrics": actual, "case": case}
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (run_dir / "actual.json").write_text(serialized)
    if update_golden:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=golden_path.parent, prefix=".golden-", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            try:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary_path, golden_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"Golden updated: {golden_path} (baseline update, not a regression pass)")
        return
    golden = json.loads(golden_path.read_text())
    failures = compare_metrics(actual, golden["metrics"], case)
    if failures:
        raise ValueError("Regression failed:\n" + "\n".join(failures))
    print(
        f"Regression passed: {len(actual)} metrics x {len(next(iter(actual.values())))} steps"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Output directory of a successful run")
    parser.add_argument(
        "golden", type=Path, help="golden_values_<platform>.json of the case"
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Write the measured metrics as the new baseline instead of comparing",
    )
    args = parser.parse_args()
    process_run(args.run_dir.resolve(), args.golden.resolve(), args.update_golden)


if __name__ == "__main__":
    main()

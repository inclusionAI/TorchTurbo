#!/usr/bin/env python3
"""Derive per-file pytest facts from source without importing test modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

DEVICE_MARKERS = frozenset({"cpu", "npu", "gpu"})
MULTI_DEVICE_MARKERS = frozenset({"multi_npu", "multi_gpu"})
MULTI_DEVICE_TARGETS = {"multi_npu": "npu", "multi_gpu": "gpu"}
REQUIRED_MARKERS = ("cpu", "npu", "multi_npu", "gpu", "multi_gpu", "integration")
KNOWN_MARKERS = frozenset(REQUIRED_MARKERS)
MODEL_PACKAGES = frozenset({"base", "modules", "registry", "weight_converter"})
FRAMEWORK_PACKAGES = {"llamafactory": "lf", "swift": "swift"}


def test_parts(path: str) -> tuple[str, ...]:
    """Return layout-independent parts below the test root."""
    parts = Path(path).parts
    if parts[:2] == ("tests", "unit_tests"):
        return parts[2:]
    if parts[:1] == ("tests",):
        return parts[1:]
    return parts


@dataclass(frozen=True)
class TestFacts:
    path: str
    devices: frozenset[str]
    device_processes: tuple[tuple[str, int], ...]
    families: frozenset[str]
    frameworks: frozenset[str]
    integration: bool
    process: str

    def processes_for(self, device: str) -> int:
        return dict(self.device_processes).get(device, 1)


@dataclass(frozen=True)
class ScanError:
    path: str
    message: str


def registered_models(root: Path) -> set[str]:
    """Collect model names registered by source decorators."""
    names: set[str] = set()
    for path in sorted((root / "src" / "torchturbo" / "models").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if not ((isinstance(func, ast.Name) and func.id == "register_model")
                        or (isinstance(func, ast.Attribute) and func.attr == "register_model")):
                    continue
                values = decorator.args[:1] + [
                    item.value for item in decorator.keywords if item.arg == "name"
                ]
                names.update(value.value for value in values
                             if isinstance(value, ast.Constant) and isinstance(value.value, str))
    return names


def _markers(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        result: set[str] = set()
        for item in node.elts:
            result |= _markers(item)
        return result
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute) and node.attr in KNOWN_MARKERS:
        base = node.value
        if (isinstance(base, ast.Attribute) and base.attr == "mark"
                and isinstance(base.value, ast.Name) and base.value.id == "pytest"):
            return {node.attr}
    return set()


def _module_markers(tree: ast.Module) -> set[str]:
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is not None and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in targets):
            result |= _markers(value)
    return result


def _imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules, {name for name in modules if name.split(".", 1)[0] in FRAMEWORK_PACKAGES}


def _literal_int(call: ast.Call, name: str) -> int | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, int) else None
    return None


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_string(node: ast.AST) -> str:
    return node.value.strip().lower() if isinstance(node, ast.Constant) and isinstance(
        node.value, str
    ) else ""


def _normalized_device(value: str) -> str:
    return "gpu" if value == "cuda" else value


def _device_evidence(tree: ast.AST) -> set[str]:
    """Return devices with direct tensor/runtime evidence below ``tree``."""
    configured_devices: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func).rsplit(".", 1)[-1]
        if name not in {"SimpleNamespace", "Namespace"}:
            continue
        for keyword in node.keywords:
            if keyword.arg == "device":
                configured_devices.update(id(child) for child in ast.walk(keyword.value))

    devices: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in configured_devices:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = ([alias.name for alias in node.names]
                       if isinstance(node, ast.Import) else [node.module or ""])
            for module in modules:
                if module.split(".", 1)[0] == "fla_npu":
                    devices.add("npu")
            continue
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if _dotted_name(base).rsplit(".", 1)[-1] == "CpuRuntime":
                    devices.add("cpu")
            continue
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        name = dotted.rsplit(".", 1)[-1]
        if name in {"npu", "cuda"} and dotted.count(".") >= 1:
            devices.add(_normalized_device(name))
        elif name in {"to", "device", "init_device_mesh"} and node.args:
            device = _normalized_device(_literal_string(node.args[0]))
            if device in DEVICE_MARKERS:
                devices.add(device)
        elif name in {"require_npu", "require_npus"}:
            devices.add("npu")
        elif dotted == "CpuRuntime":
            devices.add("cpu")
        for keyword in node.keywords:
            if keyword.arg == "device":
                device = _normalized_device(_literal_string(keyword.value))
                if device in DEVICE_MARKERS:
                    devices.add(device)
    return devices


def _module_device_guards(tree: ast.Module) -> set[str]:
    """Return devices required by module-level runtime guards."""
    devices: set[str] = set()
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
        if not isinstance(value, ast.Call):
            continue
        name = _dotted_name(value.func).rsplit(".", 1)[-1]
        if name in {"require_npu", "require_npus"}:
            devices.add("npu")
        elif name == "importorskip" and value.args:
            module = _literal_string(value.args[0]).split(".", 1)[0]
            if module in {"torch_npu", "fla_npu"}:
                devices.add("npu")
    return devices


def _parametrized_processes(tree: ast.AST) -> int:
    """Read world-size values from pytest parametrization, if present."""
    required = 1
    names = {"world_size", "nprocs", "num_processes", "rank_size"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _dotted_name(node.func).rsplit(
                ".", 1)[-1] != "parametrize" or len(node.args) < 2:
            continue
        try:
            parameters = ast.literal_eval(node.args[0])
            values = ast.literal_eval(node.args[1])
        except (ValueError, SyntaxError):
            continue
        if isinstance(parameters, str):
            parameters = [part.strip() for part in parameters.split(",")]
        if not isinstance(parameters, (list, tuple)):
            continue
        indexes = [index for index, name in enumerate(parameters)
                   if isinstance(name, str) and name in names]
        if not indexes or not isinstance(values, (list, tuple)):
            continue
        for row in values:
            row = row if isinstance(row, (list, tuple)) else (row,)
            for index in indexes:
                if index < len(row) and isinstance(row[index], int):
                    required = max(required, row[index])
    return required


def _multiprocess_rank(tree: ast.Module) -> int:
    required = 1
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else "")
        if name not in {"spawn", "init_process_group", "require_npus"}:
            continue
        if name == "require_npus":
            size = node.args[0].value if node.args and isinstance(
                node.args[0], ast.Constant
            ) and isinstance(node.args[0].value, int) else None
        else:
            key = "world_size" if name == "init_process_group" else "nprocs"
            size = _literal_int(node, key)
        required = max(required, size if size is not None else 2)
    return max(required, _parametrized_processes(tree))


def _process_unit(path: str, families: set[str]) -> str:
    parts = test_parts(path)
    if parts[:1] == ("models",):
        if len(parts) == 2:
            return "layer:tests/models"
        if len(parts) >= 3 and parts[1] not in MODEL_PACKAGES:
            return f"family:{parts[1]}"
    if families:
        return "import:" + ",".join(sorted(families))
    return f"layer:{'/'.join(Path(path).parts[:-1])}"


def scan_tests(root: Path, models: set[str]) -> tuple[dict[str, TestFacts], list[ScanError]]:
    """Scan tests/ and validate marker/runtime evidence for each file."""
    facts: dict[str, TestFacts] = {}
    errors: list[ScanError] = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError) as exc:
            errors.append(ScanError(rel, f"cannot parse test file: {exc}"))
            continue
        module_markers = _module_markers(tree)
        tests = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name.startswith("test")]
        devices: set[str] = set()
        multi_devices: set[str] = set()
        device_processes: dict[str, int] = {}
        integration = "integration" in module_markers
        module_guards = _module_device_guards(tree)
        for test in tests:
            markers = module_markers | set().union(
                *(_markers(item) for item in test.decorator_list)
            )
            test_devices = set(markers & DEVICE_MARKERS)
            test_multi = markers & MULTI_DEVICE_MARKERS
            test_devices.update(MULTI_DEVICE_TARGETS[name] for name in test_multi)
            integration |= "integration" in markers
            if not test_devices:
                errors.append(ScanError(rel, f"{test.name} has no device marker"))
            elif "cpu" in test_devices and len(test_devices) > 1:
                errors.append(ScanError(rel, f"{test.name} mixes cpu with accelerator markers"))
            evidence = _device_evidence(test) | module_guards
            if evidence and test_devices.isdisjoint(evidence):
                errors.append(ScanError(
                    rel,
                    f"{test.name} markers {sorted(test_devices)} disagree with "
                    f"code evidence {sorted(evidence)}",
                ))
            devices |= test_devices
            multi_devices |= {MULTI_DEVICE_TARGETS[name] for name in test_multi}
            test_processes = _multiprocess_rank(test)
            for device in test_devices:
                device_processes[device] = max(
                    device_processes.get(device, 1), test_processes
                )
        if not tests:
            continue
        processes = _multiprocess_rank(tree)
        evidence = _device_evidence(tree)
        if evidence and devices.isdisjoint(evidence):
            errors.append(ScanError(
                rel,
                f"device markers {sorted(devices)} disagree with code evidence "
                f"{sorted(evidence)}",
            ))
        if len(devices) == 1:
            only_device = next(iter(devices))
            device_processes[only_device] = max(
                device_processes.get(only_device, 1), processes
            )
        for device in sorted(multi_devices):
            if device_processes.get(device, 1) < 2:
                errors.append(ScanError(rel, f"multi_{device} marker needs at least two processes"))
        for device in sorted(devices - {"cpu"} - multi_devices):
            if device_processes.get(device, 1) >= 2:
                errors.append(ScanError(
                    rel,
                    f"{device} facts need multi_{device}, found "
                    f"{device_processes[device]} processes",
                ))
        modules, external = _imports(tree)
        families = {name.split(".")[2] for name in modules
                    if name.startswith("torchturbo.models.")
                    and len(name.split(".")) > 2
                    and name.split(".")[2] in models
                    and name.split(".")[2] not in MODEL_PACKAGES}
        frameworks = {FRAMEWORK_PACKAGES[name.split(".", 1)[0]] for name in external}
        parts = test_parts(rel)
        if parts[:2] == ("adapters", "llamafactory_v0"):
            frameworks.add("lf")
        if parts[:2] == ("adapters", "msswift"):
            frameworks.add("swift")
        facts[rel] = TestFacts(
            path=rel,
            devices=frozenset(devices),
            device_processes=tuple(sorted(device_processes.items())),
            families=frozenset(families),
            frameworks=frozenset(frameworks),
            integration=integration,
            process=_process_unit(rel, families),
        )
    return facts, errors

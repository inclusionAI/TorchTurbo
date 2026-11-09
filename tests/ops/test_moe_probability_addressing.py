"""The private probability kernel must widen strided storage offsets."""

import ast
from pathlib import Path

import pytest
import torch


def _functions():
    source = Path(__file__).parents[2] / "src/torchturbo/ops/providers/npu/_kernels/moe_probability.py"
    tree = ast.parse(source.read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


@pytest.mark.parametrize("name", ["_scatter_probabilities", "_gather_probability_gradients"])
def test_probability_kernel_widens_assignment_before_stride_products(name):
    function = _functions()[name]
    assignments = [node for node in function.body if isinstance(node, ast.Assign)]
    assignment = next(node.value for node in assignments if any(isinstance(target, ast.Name) and target.id == "assignment" for target in node.targets))
    assert isinstance(assignment, ast.Call) and isinstance(assignment.func, ast.Attribute)
    assert assignment.func.attr == "to" and ast.unparse(assignment.args[0]) == "tl.int64"


def test_probability_backward_widens_packed_row_before_gradient_stride():
    function = _functions()["_gather_probability_gradients"]
    packed_row = next(node.value for node in function.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "packed_row" for target in node.targets))
    assert isinstance(packed_row, ast.Call) and isinstance(packed_row.func, ast.Attribute)
    assert packed_row.func.attr == "to" and ast.unparse(packed_row.args[0]) == "tl.int64"
    # A tiny logical tensor can refer to a valid large storage offset. Int32
    # row*stride wraps, while the widened product preserves the address.
    row = torch.tensor([3], dtype=torch.int32)
    stride = 1_073_741_824
    assert int(row.to(torch.int64).mul(stride)) == 3_221_225_472
    assert int(row.mul(stride)) != 3_221_225_472

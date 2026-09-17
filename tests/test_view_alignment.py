"""The node-benchmark view lists must agree, and this must be checkable anywhere.

Checkpoint selection scores the validation logit ensemble over
`rcps_checkpoint_views`; the reported test number is the ensemble over
`rcps_multiscale_views`.  If the two lists ever diverge, the selected epoch is
optimal for a quantity that never appears in the paper.  The guard is loaded
from its file directly so that this file imports neither `jepa_compare` nor
torch, and therefore runs in a bare environment.

    python -m pytest tests/test_view_alignment.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import yaml

_PATH = Path(__file__).resolve().parents[1] / "jepa_compare" / "view_alignment.py"
_SPEC = importlib.util.spec_from_file_location("view_alignment", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_TORCH_BEFORE = "torch" in sys.modules
_SPEC.loader.exec_module(_MODULE)
_TORCH_AFTER = "torch" in sys.modules
require_aligned_views = _MODULE.require_aligned_views

_CONFIGS = sorted(
    (Path(__file__).resolve().parents[1] / "configs").glob("node_comparison*.yaml")
)


def test_the_guard_itself_needs_no_torch() -> None:
    """Loading the guard must not pull torch in, however it is reached.

    Comparing before with after rather than asserting `"torch" not in
    sys.modules` keeps the test meaningful when some earlier test in the same
    session has already imported torch.
    """
    assert _TORCH_AFTER == _TORCH_BEFORE


def test_mismatched_view_lists_are_refused() -> None:
    with pytest.raises(ValueError, match="rcps_checkpoint_views must equal"):
        require_aligned_views(("prediction", "encoder"), ("hop2", "hop8"))


def test_missing_reported_views_are_refused() -> None:
    with pytest.raises(ValueError, match="rcps_checkpoint_views must equal"):
        require_aligned_views(("prediction", "encoder"), ())


def test_order_matters() -> None:
    with pytest.raises(ValueError, match="rcps_checkpoint_views must equal"):
        require_aligned_views(("prediction", "encoder"), ("encoder", "prediction"))


@pytest.mark.parametrize("views", [("prediction", "encoder"), ("hop2", "hop8")])
def test_aligned_view_lists_pass(views) -> None:
    require_aligned_views(views, list(views))


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_shipped_node_configs_are_aligned(path: Path) -> None:
    training = yaml.safe_load(path.read_text()).get("training", {}) or {}
    if "rcps_multiscale_views" not in training:
        pytest.skip(f"{path.name} configures no DyGJEPA views")
    require_aligned_views(
        training.get("rcps_checkpoint_views", ()),
        training.get("rcps_multiscale_views", ()),
    )

"""Proposition 1: pair multiplicity is not identifiable from the snapshot view.

Two event streams are built over four nodes so that, in the single bin, they
agree on the set of interacting ordered pairs, on every node's outgoing and
incoming interaction counts, and on the multiset of link features -- while their
per-pair multiplicities differ by k-1.  The converter must then produce
bit-identical structural and attribute views, and a differing event view.

Run with:  python -m pytest tests/test_proposition1_snapshot_identifiability.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "prepare_dyglib_homogeneous",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_dyglib_homogeneous.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

K = 4
BINS = 6          # the converter requires at least six bins
CONSTRUCTION = 2 * K + 2   # events of the construction: 2 heavy pairs x K, 2 light
TOTAL = CONSTRUCTION * BINS  # equal-event split then puts the whole construction in bin 0


def _stream(repeat_diagonal: bool) -> list[tuple[int, int]]:
    """Bin-0 events; the two streams swap which pair group repeats K times."""
    u1, u2, v1, v2 = 1, 2, 3, 4  # DyGLib uses one-based ids with row-zero padding
    heavy = [(u1, v1), (u2, v2)] if repeat_diagonal else [(u1, v2), (u2, v1)]
    light = [(u1, v2), (u2, v1)] if repeat_diagonal else [(u1, v1), (u2, v2)]
    events = [pair for pair in heavy for _ in range(K)] + list(light)
    return events


def _write_dataset(root: Path, events: list[tuple[int, int]]) -> Path:
    """Materialise a minimal DyGLib-format dataset directory."""
    root.mkdir(parents=True, exist_ok=True)
    # Equal-event bins: pad so that bin 0 holds exactly the construction and the
    # remaining bins hold identical filler in both streams.
    filler = [(1, 3)] * (TOTAL - CONSTRUCTION)
    all_events = events + filler
    n = len(all_events)
    with (root / "ml_synth.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "u", "i", "ts", "label", "idx"])
        for i, (u, v) in enumerate(all_events):
            # Bin 0 events share ascending timestamps; fillers come strictly later.
            ts = float(i) if i < len(events) else float(1000 + i)
            writer.writerow([i, u, v, ts, 0, i + 1])
    np.save(root / "ml_synth.npy", np.ones((n + 1, 2), dtype=np.float32))
    np.save(root / "ml_synth_node.npy", np.zeros((5, 2), dtype=np.float32))
    return root


def test_snapshot_views_coincide_while_multiplicities_differ(tmp_path: Path) -> None:
    outputs = []
    for name, diagonal in (("a", True), ("b", False)):
        data_dir = _write_dataset(tmp_path / f"raw_{name}", _stream(diagonal))
        out = tmp_path / f"{name}.npz"
        _MODULE.convert(data_dir, out, event_bins=BINS, seed=0)
        outputs.append(np.load(out))

    left, right = outputs
    bins = int(left["num_snapshots"]) if "num_snapshots" in left else BINS

    for t in range(bins):
        np.testing.assert_array_equal(
            left[f"edges_{t}"], right[f"edges_{t}"],
            err_msg=f"structural view A_{t} differs",
        )
    np.testing.assert_allclose(
        left["features"], right["features"], rtol=0, atol=0,
        err_msg="attribute view X_t differs",
    )
    np.testing.assert_array_equal(
        left["active"], right["active"], err_msg="active mask differs"
    )

    def multiplicity(archive, pair):
        queries = archive["queries_0"]
        return int(np.sum((queries[0] == pair[0]) & (queries[1] == pair[1])))

    assert multiplicity(left, (1, 3)) == K
    assert multiplicity(right, (1, 3)) == 1
    assert multiplicity(left, (1, 3)) - multiplicity(right, (1, 3)) == K - 1

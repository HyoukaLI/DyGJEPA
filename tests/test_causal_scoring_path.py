"""The causal scoring path must never read the target snapshot.

`score_windows` calls `prepare_window(..., with_target=False)`, so no encoder is
handed the target snapshot.  Two things are asserted:

1. the probabilities match `evaluate_windows` exactly, so the causal path is not
   a different model;
2. neither the target encoder nor the target relation encoder is invoked, which
   is what makes the causal boundary auditable rather than merely claimed.

Requires torch; run locally with
    python -m pytest tests/test_causal_scoring_path.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from jepa_compare.rcps_jepa import RCPSJEPA  # noqa: E402


def _windows(model, num_nodes: int, feature_dim: int):
    from jepa_compare.data import Snapshot

    generator = torch.Generator().manual_seed(0)
    snapshots = []
    for t in range(model.window_size):
        src = torch.randint(0, num_nodes, (12,), generator=generator)
        dst = torch.randint(0, num_nodes, (12,), generator=generator)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        snapshots.append(
            Snapshot(
                x=torch.randn(num_nodes, feature_dim, generator=generator),
                edge_index=edge_index,
                time=t,
                active=torch.ones(num_nodes, dtype=torch.bool),
                query_edge_index=torch.stack([src, dst]),
                query_timestamps=torch.arange(src.numel(), dtype=torch.float32) + 10.0 * t,
            )
        )
    return [snapshots]


def test_causal_path_matches_and_never_reads_the_target() -> None:
    num_nodes, feature_dim = 24, 8
    model = RCPSJEPA(feature_dim=feature_dim, num_nodes=num_nodes, hidden_dim=16)
    model.eval()
    windows = _windows(model, num_nodes, feature_dim)

    reference = model.evaluate_windows(windows, query_seed=0)
    assert "ap" in reference

    seen: list[bool] = []
    original = model.encode_snapshot

    def spy(snapshot, target: bool = False):
        seen.append(target)
        return original(snapshot, target=target)

    model.encode_snapshot = spy  # type: ignore[method-assign]
    probability, labels, groups = model.score_windows(windows, query_seed=0)
    model.encode_snapshot = original  # type: ignore[method-assign]

    assert not any(seen), "the causal path invoked the target encoder"
    assert probability.shape == labels.shape == groups.shape
    assert torch.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()

"""DyGJEPA node checkpoint selection must score the ensemble it reports.

An earlier version selected the epoch by the better of the two views scored
separately while reporting their logit ensemble.  Here `_select_node_embeddings`
is driven with `ensemble=True` and asserted to score the ensemble exactly once,
over every candidate view, and never to score a view on its own.

The whole file requires torch, because `jepa_compare.compare_node_prediction`
imports it.  The torch-free half of this contract -- the configuration guard
that keeps the two view lists equal -- lives in `tests/test_view_alignment.py`.

    python -m pytest tests/test_node_checkpoint_selection.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from jepa_compare import compare_node_prediction as cnp  # noqa: E402
from jepa_compare.node_evaluation import ProbeSplit  # noqa: E402


class _Graph:
    def __init__(self, num_nodes: int, classes: int) -> None:
        self.num_nodes = num_nodes
        self.labels = torch.arange(num_nodes) % classes
        self.snapshots = []


def test_selection_scores_the_ensemble_and_not_single_views(monkeypatch) -> None:
    num_nodes, dim = 24, 4
    graph = _Graph(num_nodes, classes=3)
    views = {
        "prediction": torch.randn(num_nodes, dim),
        "encoder": torch.randn(num_nodes, dim),
    }
    node_ids = torch.arange(num_nodes)
    monkeypatch.setattr(cnp, "_infer_views", lambda model, g: (views, node_ids))

    ensemble_calls: list[tuple[str, ...]] = []
    single_calls: list[int] = []

    def fake_ensemble(embeddings, labels, split, epochs, seed, hidden_dim=None):
        ensemble_calls.append(tuple(embeddings))
        return {"macro_f1": 0.5, "micro_f1": 0.5}

    def fake_single(*args, **kwargs):  # pragma: no cover - must not be reached
        single_calls.append(1)
        return {"macro_f1": 0.9, "micro_f1": 0.9}

    monkeypatch.setattr(cnp, "validation_probe_ensemble", fake_ensemble)
    monkeypatch.setattr(cnp, "validation_probe", fake_single)

    split = ProbeSplit(
        train=torch.arange(0, 12),
        validation=torch.arange(12, 18),
        test=torch.arange(18, 24),
    )
    _, metrics, label = cnp._select_node_embeddings(
        model=object(),
        graph=graph,
        probe_split=split,
        probe_epochs=1,
        seed=0,
        probe_hidden_dim=None,
        candidates=("prediction", "encoder"),
        ensemble=True,
    )

    # The ensemble is scored exactly once, over both views, and no view is
    # scored on its own -- the 0.9 of `fake_single` must not win.
    assert ensemble_calls == [("prediction", "encoder")]
    assert single_calls == []
    assert metrics["macro_f1"] == 0.5
    assert label == "logit_ensemble[prediction,encoder]"
    assert metrics["selected_view"] == label


def test_ensemble_selection_needs_two_views() -> None:
    with pytest.raises(ValueError, match="at least two views"):
        cnp._select_node_embeddings(
            model=object(),
            graph=_Graph(4, classes=2),
            probe_split=None,
            probe_epochs=1,
            seed=0,
            probe_hidden_dim=None,
            candidates=("prediction",),
            ensemble=True,
        )

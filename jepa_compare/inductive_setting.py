"""DyGLib's inductive (new-node) link-prediction setting on snapshot data.

DyGLib (``get_link_prediction_data``) draws 10% of all nodes among those that
interact after the training period (``random.seed(2020)``), removes every
training edge that touches one of them, trains on what is left, selects the
checkpoint on the ordinary (transductive) validation set and finally reports
the *inductive* metrics: validation/test events with at least one endpoint
that never appears in the (reduced) training data, scored against random
destinations drawn from those events' own destinations (validation seed 1,
test seed 3).  Baseline papers built on DyGLib (DyGFormer, DyG-Mamba, ...)
report this as the "inductive" table.

This module maps the recipe onto the snapshot protocol of the comparison:

* the training period is every snapshot of the training windows; the
  candidate new nodes are the endpoints of query events of the validation /
  test target snapshots;
* :func:`remove_nodes` builds the reduced training snapshots (query events,
  message edges, features and activity of the sampled nodes removed);
* :func:`restrict_queries` builds the evaluation targets (only the query
  events that touch a new node are kept; the snapshot structure is untouched
  because context encoders and neighbour samplers see the full graph at
  evaluation time, exactly as DyGLib's ``full_neighbor_sampler`` does).

Everything here is deterministic dataset state; models are unchanged.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .data import Snapshot
from .link_prediction import TemporalWindowSplit
from .temporal_event_utils import unique_snapshots

DEFAULT_NEW_NODE_RATIO = 0.1
DEFAULT_NEW_NODE_SEED = 2020
# DyGLib seeds: validation/test transductive samplers 0/2, new-node ones 1/3.
INDUCTIVE_VALIDATION_QUERY_SEED = 1
INDUCTIVE_TEST_QUERY_SEED = 3


def _event_endpoints(snapshot: Snapshot) -> tuple[Tensor, Tensor]:
    if snapshot.query_edge_index is None:
        raise ValueError("the inductive setting requires query_edge_index events")
    return snapshot.query_edge_index[0].long(), snapshot.query_edge_index[1].long()


def event_touches(snapshot: Snapshot, node_mask: Tensor) -> Tensor:
    """Boolean mask over the snapshot's query events touching ``node_mask``."""
    sources, destinations = _event_endpoints(snapshot)
    mask = node_mask.to(sources.device)
    return mask[sources] | mask[destinations]


def sample_new_nodes(
    candidates: Tensor, num_total_nodes: int, ratio: float, seed: int
) -> Tensor:
    """DyGLib's draw: ``random.sample(sorted(candidates), int(ratio * |V|))``."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("new_node_ratio must be in (0, 1)")
    count = int(ratio * num_total_nodes)
    pool = sorted(int(node) for node in torch.unique(candidates).tolist())
    if count > len(pool):
        raise ValueError(
            f"cannot hold out {count} new nodes: only {len(pool)} nodes interact "
            "after the training period"
        )
    rng = random.Random(seed)
    return torch.tensor(sorted(rng.sample(pool, count)), dtype=torch.long)


def remove_nodes(snapshot: Snapshot, node_mask: Tensor) -> Snapshot:
    """Training view of a snapshot without the held-out nodes.

    Query events and message edges touching a held-out node are dropped, the
    node's feature row is zeroed and it is marked inactive, so nothing about it
    reaches a model during training.
    """
    mask = node_mask.to(snapshot.x.device)
    keep_events = ~event_touches(snapshot, mask)
    edge_keep = ~(mask[snapshot.edge_index[0].long()] | mask[snapshot.edge_index[1].long()])
    x = snapshot.x.clone()
    x[mask] = 0.0
    active = snapshot.active.clone()
    active[mask] = False
    return Snapshot(
        x=x,
        edge_index=snapshot.edge_index[:, edge_keep],
        active=active,
        time=snapshot.time,
        query_edge_index=snapshot.query_edge_index[:, keep_events],
        query_timestamps=(
            None if snapshot.query_timestamps is None else snapshot.query_timestamps[keep_events]
        ),
        query_features=(
            None if snapshot.query_features is None else snapshot.query_features[keep_events]
        ),
        query_labels=(
            None if snapshot.query_labels is None else snapshot.query_labels[keep_events]
        ),
    )


def restrict_queries(snapshot: Snapshot, node_mask: Tensor) -> Snapshot:
    """Evaluation view of a snapshot: only query events touching ``node_mask``.

    The graph structure, features and activity are the full ones; only the
    positives to score change (DyGLib's ``new_node_val_data`` /
    ``new_node_test_data``).
    """
    keep = event_touches(snapshot, node_mask)
    return Snapshot(
        x=snapshot.x,
        edge_index=snapshot.edge_index,
        active=snapshot.active,
        time=snapshot.time,
        query_edge_index=snapshot.query_edge_index[:, keep],
        query_timestamps=(
            None if snapshot.query_timestamps is None else snapshot.query_timestamps[keep]
        ),
        query_features=(
            None if snapshot.query_features is None else snapshot.query_features[keep]
        ),
        query_labels=None if snapshot.query_labels is None else snapshot.query_labels[keep],
    )


def destination_pool(snapshots: Sequence[Snapshot]) -> Tensor:
    """Unique destinations of the given (already restricted) target events."""
    destinations = [
        snapshot.query_edge_index[1].detach().cpu().long()
        for snapshot in snapshots
        if snapshot.query_edge_index is not None and snapshot.query_edge_index.numel()
    ]
    if not destinations:
        raise ValueError("no new-node events: the inductive pool is empty")
    return torch.unique(torch.cat(destinations), sorted=True)


def _replace_targets(
    windows: Sequence[Sequence[Snapshot]], replacement: dict[int, Snapshot]
) -> list[list[Snapshot]]:
    """Windows whose *target* snapshot is swapped for its restricted view."""
    return [[*window[:-1], replacement[window[-1].time]] for window in windows]


@dataclass
class InductiveSetting:
    """Everything the driver needs for one dataset under the inductive setting."""

    ratio: float
    seed: int
    sampled_nodes: Tensor            # the held-out 10% (DyGLib new_test_node_set)
    new_node_mask: Tensor            # every node absent from the reduced training data
    train_windows: list[list[Snapshot]]          # reduced training windows
    full_train_windows: list[list[Snapshot]]     # the original ones (evaluation histories)
    train_snapshots: list[Snapshot]              # unique reduced training snapshots
    history_snapshots: list[Snapshot]            # reduced train + full val/test, in time order
    validation_windows: list[list[Snapshot]]     # full context, new-node-only targets
    test_windows: list[list[Snapshot]]
    validation_pool: Tensor
    test_pool: Tensor
    validation_query_seed: int = INDUCTIVE_VALIDATION_QUERY_SEED
    test_query_seed: int = INDUCTIVE_TEST_QUERY_SEED
    summary: dict = field(default_factory=dict)

    @property
    def new_node_ids(self) -> np.ndarray:
        return torch.nonzero(self.new_node_mask, as_tuple=False).flatten().cpu().numpy()

    def final_split(self, split: TemporalWindowSplit) -> TemporalWindowSplit:
        """The split whose event counts the coverage check must reproduce."""
        return TemporalWindowSplit(
            train=split.train,
            validation=self.validation_windows,
            test=self.test_windows,
        )


def build_inductive_setting(
    snapshots: Sequence[Snapshot],
    split: TemporalWindowSplit,
    *,
    ratio: float = DEFAULT_NEW_NODE_RATIO,
    seed: int = DEFAULT_NEW_NODE_SEED,
) -> InductiveSetting:
    """Hold out new nodes and derive the reduced training / restricted evaluation views."""
    num_nodes = int(snapshots[0].x.shape[0])
    train_snapshots = unique_snapshots(split.train)
    train_times = {snapshot.time for snapshot in train_snapshots}
    eval_targets = unique_snapshots([*split.validation, *split.test], targets_only=True)

    all_events = [torch.cat(_event_endpoints(s)) for s in snapshots if s.query_edge_index is not None]
    num_total_unique = int(torch.unique(torch.cat(all_events)).numel())
    candidates = torch.cat([torch.cat(_event_endpoints(s)) for s in eval_targets])
    sampled = sample_new_nodes(candidates.cpu(), num_total_unique, ratio, seed)
    sampled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    sampled_mask[sampled] = True

    reduced = {s.time: remove_nodes(s, sampled_mask) for s in train_snapshots}
    seen = torch.zeros(num_nodes, dtype=torch.bool)
    for snapshot in reduced.values():
        sources, destinations = _event_endpoints(snapshot)
        seen[sources.cpu()] = True
        seen[destinations.cpu()] = True
    # DyGLib new_node_set: every node that never interacts in the reduced
    # training data (the sampled ones plus nodes that naturally appear later).
    new_node_mask = ~seen
    assert bool(new_node_mask[sampled].all())

    train_windows = [[reduced[s.time] for s in window] for window in split.train]
    restricted = {s.time: restrict_queries(s, new_node_mask) for s in eval_targets}
    validation_windows = _replace_targets(split.validation, restricted)
    test_windows = _replace_targets(split.test, restricted)
    validation_targets = unique_snapshots(validation_windows, targets_only=True)
    test_targets = unique_snapshots(test_windows, targets_only=True)
    history_snapshots = [
        reduced[s.time] if s.time in train_times else s for s in snapshots
    ]

    def count(targets: Sequence[Snapshot]) -> int:
        return int(sum(t.query_edge_index.shape[1] for t in targets))

    full_validation = unique_snapshots(split.validation, targets_only=True)
    full_test = unique_snapshots(split.test, targets_only=True)
    removed = sum(
        int(s.query_edge_index.shape[1]) - int(reduced[s.time].query_edge_index.shape[1])
        for s in train_snapshots
    )
    summary = {
        "setting": "inductive",
        "new_node_ratio": ratio,
        "new_node_seed": seed,
        "num_nodes": num_nodes,
        "num_interacting_nodes": num_total_unique,
        "sampled_new_nodes": int(sampled.numel()),
        "new_nodes": int(new_node_mask.sum()),
        "training_events_removed": removed,
        "training_events_kept": count(list(reduced.values())),
        "validation_events": count(validation_targets),
        "validation_events_total": count(full_validation),
        "test_events": count(test_targets),
        "test_events_total": count(full_test),
        "validation_pool": int(destination_pool(validation_targets).numel()),
        "test_pool": int(destination_pool(test_targets).numel()),
    }
    return InductiveSetting(
        ratio=ratio,
        seed=seed,
        sampled_nodes=sampled,
        new_node_mask=new_node_mask,
        train_windows=train_windows,
        full_train_windows=list(split.train),
        train_snapshots=[reduced[s.time] for s in train_snapshots],
        history_snapshots=history_snapshots,
        validation_windows=validation_windows,
        test_windows=test_windows,
        validation_pool=destination_pool(validation_targets),
        test_pool=destination_pool(test_targets),
        summary=summary,
    )


def normalize_setting(value: object) -> str:
    setting = str(value or "transductive").lower()
    if setting not in {"transductive", "inductive"}:
        raise ValueError(
            f"link.setting must be 'transductive' or 'inductive', got {value!r}"
        )
    return setting

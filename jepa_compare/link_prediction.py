from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor

from .data import Snapshot


PAIR_STAT_DIM = 8
NODE_EVENT_DIM = 6
_NEIGHBOR_CACHE: dict[tuple[int, int, bool], list[set[int]]] = {}
_CANONICAL_PAIR_CACHE: dict[tuple[int, int, bool], Tensor] = {}
_NODE_TRANSITION_CACHE: dict[tuple[int, int, bool, str, torch.dtype], Tensor] = {}


@dataclass(frozen=True)
class LinkQueries:
    pairs: Tensor
    labels: Tensor
    group_ids: Tensor
    timestamps: Tensor | None = None


@dataclass(frozen=True)
class TemporalWindowSplit:
    train: list[list[Snapshot]]
    validation: list[list[Snapshot]]
    test: list[list[Snapshot]]


def sliding_windows(snapshots: Sequence[Snapshot], window_size: int) -> list[list[Snapshot]]:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if len(snapshots) < window_size:
        raise ValueError("not enough snapshots for one window")
    return [list(snapshots[end - window_size + 1 : end + 1]) for end in range(window_size - 1, len(snapshots))]


def temporal_window_split(
    snapshots: Sequence[Snapshot],
    window_size: int,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> TemporalWindowSplit:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("validation_ratio must be non-negative and leave a test split")
    windows = sliding_windows(snapshots, window_size)
    if len(windows) < 3:
        raise ValueError("at least three sliding windows are required for train/validation/test")
    train_end = max(1, int(len(windows) * train_ratio))
    validation_end = max(train_end + 1, int(len(windows) * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, len(windows) - 1)
    return TemporalWindowSplit(
        train=windows[:train_end],
        validation=windows[train_end:validation_end],
        test=windows[validation_end:],
    )


def canonical_pairs(edge_index: Tensor, num_nodes: int, undirected: bool = True) -> Tensor:
    cache_key = (id(edge_index), num_nodes, undirected)
    cached = _CANONICAL_PAIR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if edge_index.numel() == 0:
        pairs = torch.empty(0, 2, dtype=torch.long, device=edge_index.device)
        _CANONICAL_PAIR_CACHE[cache_key] = pairs
        return pairs
    src, dst = edge_index
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if undirected:
        lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
        keys = lo * num_nodes + hi
    else:
        keys = src * num_nodes + dst
    keys = torch.unique(keys, sorted=True)
    pairs = torch.stack([keys // num_nodes, keys % num_nodes], dim=-1)
    _CANONICAL_PAIR_CACHE[cache_key] = pairs
    return pairs


def _pair_keys(pairs: Tensor, num_nodes: int, undirected: bool) -> Tensor:
    if undirected:
        lo = torch.minimum(pairs[:, 0], pairs[:, 1])
        hi = torch.maximum(pairs[:, 0], pairs[:, 1])
        return lo * num_nodes + hi
    return pairs[:, 0] * num_nodes + pairs[:, 1]


def sample_link_queries(
    target: Snapshot,
    previous: Snapshot | None,
    negative_ratio: float = 1.0,
    max_positive: int | None = None,
    seed: int = 42,
    new_edges_only: bool = True,
    undirected: bool = True,
    bipartite_source_count: int | None = None,
) -> LinkQueries:
    """Sample deterministic positive and negative candidate links.

    For cumulative snapshot datasets, ``new_edges_only`` removes edges already
    present in the final context snapshot. If no new edge exists, the function
    falls back to all target edges so small smoke-test graphs remain usable.
    """
    device = target.edge_index.device
    num_nodes = target.x.shape[0]
    if bipartite_source_count is not None:
        if undirected:
            raise ValueError("bipartite link prediction requires undirected=False")
        if not 0 < bipartite_source_count < num_nodes:
            raise ValueError("bipartite_source_count does not split the node space")

    def valid_pairs(snapshot: Snapshot) -> tuple[Tensor, Tensor]:
        if snapshot.query_edge_index is None:
            pairs = canonical_pairs(snapshot.edge_index, num_nodes, undirected)
            timestamps = torch.full(
                (pairs.shape[0],),
                float(snapshot.time),
                dtype=snapshot.x.dtype,
                device=device,
            )
        else:
            src, dst = snapshot.query_edge_index
            keep = src != dst
            src, dst = src[keep], dst[keep]
            if snapshot.query_timestamps is None:
                timestamps = torch.full(
                    (src.shape[0],),
                    float(snapshot.time),
                    dtype=snapshot.x.dtype,
                    device=device,
                )
            else:
                timestamps = snapshot.query_timestamps[keep].to(device=device)
            if undirected:
                src, dst = torch.minimum(src, dst), torch.maximum(src, dst)
            pairs = torch.stack([src, dst], dim=-1)
        if bipartite_source_count is not None and pairs.numel():
            partition_keep = (
                (pairs[:, 0] < bipartite_source_count)
                & (pairs[:, 1] >= bipartite_source_count)
            )
            pairs = pairs[partition_keep]
            timestamps = timestamps[partition_keep]
        return pairs, timestamps

    positives, positive_timestamps = valid_pairs(target)
    if new_edges_only and previous is not None and positives.numel():
        previous_pairs, _ = valid_pairs(previous)
        previous_keys = set(_pair_keys(previous_pairs.cpu(), num_nodes, undirected).tolist())
        keep = torch.tensor(
            [key not in previous_keys for key in _pair_keys(positives.cpu(), num_nodes, undirected).tolist()],
            dtype=torch.bool,
            device=device,
        )
        new_positives = positives[keep]
        if new_positives.numel():
            positives = new_positives
            positive_timestamps = positive_timestamps[keep]
    if positives.numel() == 0:
        raise ValueError("target snapshot has no positive link candidates")

    generator = torch.Generator().manual_seed(seed)
    if max_positive is not None and positives.shape[0] > max_positive:
        order = torch.randperm(positives.shape[0], generator=generator)[:max_positive]
        positives = positives[order.to(device)]
        positive_timestamps = positive_timestamps[order.to(device)]

    positive_count = positives.shape[0]
    negative_count = max(1, int(round(positive_count * negative_ratio)))
    forbidden = set(
        _pair_keys(valid_pairs(target)[0].cpu(), num_nodes, undirected).tolist()
    )
    negative_pairs: list[tuple[int, int]] = []
    negative_groups: list[int] = []
    used_by_group: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(1_000, negative_count * 100)
    while len(negative_pairs) < negative_count and attempts < max_attempts:
        group = len(negative_pairs) % positive_count
        u = int(positives[group, 0].item())
        if bipartite_source_count is None:
            v = int(torch.randint(num_nodes, (1,), generator=generator).item())
        else:
            v = int(
                torch.randint(
                    bipartite_source_count,
                    num_nodes,
                    (1,),
                    generator=generator,
                ).item()
            )
        attempts += 1
        if u == v:
            continue
        if undirected and u > v:
            u, v = v, u
        key = u * num_nodes + v
        if key in forbidden or (group, key) in used_by_group:
            continue
        used_by_group.add((group, key))
        negative_pairs.append((u, v))
        negative_groups.append(group)
    if len(negative_pairs) < negative_count:
        raise RuntimeError("unable to sample enough negative links")
    negatives = torch.tensor(negative_pairs, dtype=torch.long, device=device)
    pairs = torch.cat([positives, negatives], dim=0)
    labels = torch.cat(
        [torch.ones(positives.shape[0], device=device), torch.zeros(negatives.shape[0], device=device)]
    )
    group_ids = torch.cat(
        [
            torch.arange(positive_count, dtype=torch.long, device=device),
            torch.tensor(negative_groups, dtype=torch.long, device=device),
        ]
    )
    negative_timestamps = positive_timestamps[
        torch.tensor(negative_groups, dtype=torch.long, device=device)
    ]
    timestamps = torch.cat([positive_timestamps, negative_timestamps], dim=0)
    order = torch.randperm(pairs.shape[0], generator=generator).to(device)
    return LinkQueries(
        pairs[order], labels[order], group_ids[order], timestamps[order]
    )


def neighbor_sets(snapshot: Snapshot, undirected: bool = True) -> list[set[int]]:
    num_nodes = snapshot.x.shape[0]
    cache_key = (id(snapshot.edge_index), num_nodes, undirected)
    cached = _NEIGHBOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    neighbors = [set() for _ in range(num_nodes)]
    if snapshot.edge_index.numel() == 0:
        _NEIGHBOR_CACHE[cache_key] = neighbors
        return neighbors
    src, dst = snapshot.edge_index.detach().cpu()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        neighbors[u].add(v)
        if undirected:
            neighbors[v].add(u)
    _NEIGHBOR_CACHE[cache_key] = neighbors
    return neighbors


def relation_context_nodes(
    context: Sequence[Snapshot],
    pairs: Tensor,
    budget: int,
    decay: float = 0.8,
    bridge_weight: float = 1.0,
    undirected: bool = True,
) -> tuple[Tensor, Tensor]:
    """Select a pair-conditioned temporal context with a fixed node budget."""
    if budget < 2:
        raise ValueError("subgraph budget must be at least two")
    adjacency = [neighbor_sets(snapshot, undirected) for snapshot in context]
    pair_list = pairs.detach().cpu().tolist()
    rows: list[list[int]] = []
    masks: list[list[bool]] = []
    for u, v in pair_list:
        score: dict[int, float] = {u: float("inf"), v: float("inf")}
        for age, neighbors in enumerate(reversed(adjacency)):
            weight = decay**age
            nu, nv = neighbors[u], neighbors[v]
            for node in nu | nv:
                endpoint_hits = float(node in nu) + float(node in nv)
                bridge_bonus = bridge_weight if node in nu and node in nv else 0.0
                score[node] = score.get(node, 0.0) + weight * (endpoint_hits + bridge_bonus)
        ranked = sorted(score, key=lambda node: (-score[node], node))[:budget]
        valid = [True] * len(ranked)
        if len(ranked) < budget:
            ranked.extend([u] * (budget - len(ranked)))
            valid.extend([False] * (budget - len(valid)))
        rows.append(ranked)
        masks.append(valid)
    return (
        torch.tensor(rows, dtype=torch.long, device=pairs.device),
        torch.tensor(masks, dtype=torch.bool, device=pairs.device),
    )


def pair_statistics(
    snapshot: Snapshot,
    pairs: Tensor,
    context_nodes: Tensor | None = None,
    undirected: bool = True,
) -> Tensor:
    """Return direct, degree, bridge, activity, and coverage statistics."""
    neighbors = neighbor_sets(snapshot, undirected)
    num_nodes = snapshot.x.shape[0]
    active = snapshot.active.detach().cpu()
    rows = []
    pair_list = pairs.detach().cpu().tolist()
    local_nodes = None if context_nodes is None else context_nodes.detach().cpu().tolist()
    for index, (u, v) in enumerate(pair_list):
        nu, nv = neighbors[u], neighbors[v]
        common = nu & nv
        union = nu | nv
        direct = float(v in nu)
        degree_scale = max(1, num_nodes - 1)
        common_scale = max(1.0, (len(nu) * len(nv)) ** 0.5)
        jaccard = len(common) / max(1, len(union))
        if local_nodes is None:
            coverage = len(union) / degree_scale
        else:
            selected = set(local_nodes[index])
            coverage = len(union & selected) / max(1, len(selected))
        rows.append(
            [
                direct,
                len(nu) / degree_scale,
                len(nv) / degree_scale,
                len(common) / common_scale,
                jaccard,
                float(active[u]),
                float(active[v]),
                coverage,
            ]
        )
    return torch.tensor(rows, dtype=snapshot.x.dtype, device=pairs.device)


def temporal_pair_increments(
    context: Sequence[Snapshot],
    pairs: Tensor,
    context_nodes: Tensor,
    undirected: bool = True,
) -> Tensor:
    states = torch.stack(
        [pair_statistics(snapshot, pairs, context_nodes, undirected) for snapshot in context], dim=1
    )
    increments = torch.empty_like(states)
    increments[:, 0] = states[:, 0]
    if states.shape[1] > 1:
        increments[:, 1:] = states[:, 1:] - states[:, :-1]
    start_time, end_time = context[0].time, context[-1].time
    scale = max(1, end_time - start_time)
    time_steps = [1.0 / scale]
    time_steps.extend((context[i].time - context[i - 1].time) / scale for i in range(1, len(context)))
    time_channel = torch.tensor(time_steps, dtype=states.dtype, device=states.device)
    time_channel = time_channel.view(1, -1, 1).expand(states.shape[0], -1, -1)
    return torch.cat([time_channel, increments], dim=-1)


def node_transition_statistics(
    current: Snapshot,
    previous: Snapshot | None,
    undirected: bool = True,
) -> Tensor:
    """Encode one structural transition for every node.

    The six channels are log-normalized degree, bounded signed degree change,
    edge-arrival ratio, edge-removal ratio, neighbor persistence, and node
    activity.  Change channels use the local before/after union rather than
    ``num_nodes`` so their scale does not vanish on large sparse graphs.
    """
    num_nodes = current.x.shape[0]
    cache_key = (
        id(current.edge_index),
        0 if previous is None else id(previous.edge_index),
        undirected,
        str(current.x.device),
        current.x.dtype,
    )
    cached = _NODE_TRANSITION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    current_neighbors = neighbor_sets(current, undirected)
    previous_neighbors = (
        [set() for _ in range(num_nodes)]
        if previous is None
        else neighbor_sets(previous, undirected)
    )
    degree_scale = math.log1p(max(1, num_nodes - 1))
    rows = []
    current_active = current.active.detach().cpu()
    for node, (now, before) in enumerate(zip(current_neighbors, previous_neighbors)):
        arrivals = now - before
        removals = before - now
        retained = now & before
        union_size = max(1, len(now | before))
        rows.append(
            [
                math.log1p(len(now)) / degree_scale,
                (len(now) - len(before)) / max(1, len(now) + len(before)),
                len(arrivals) / union_size,
                len(removals) / union_size,
                len(retained) / union_size,
                float(current_active[node]),
            ]
        )
    statistics = torch.tensor(rows, dtype=current.x.dtype, device=current.x.device)
    _NODE_TRANSITION_CACHE[cache_key] = statistics
    return statistics


def temporal_node_increments(
    context: Sequence[Snapshot],
    undirected: bool = True,
) -> Tensor:
    """Return time-augmented relational event increments for every node.

    Shape: ``[num_nodes, context_steps, NODE_EVENT_DIM + 1]``.  The first
    channel is elapsed time; the remaining channels describe local relation
    changes.  These vectors are path increments and can therefore be composed
    online with Chen's identity by :func:`truncated_signature`.
    """
    if not context:
        raise ValueError("at least one context snapshot is required")
    transitions = []
    for index, snapshot in enumerate(context):
        previous = None if index == 0 else context[index - 1]
        transitions.append(node_transition_statistics(snapshot, previous, undirected))
    events = torch.stack(transitions, dim=1)
    start_time, end_time = context[0].time, context[-1].time
    scale = max(1, end_time - start_time)
    time_steps = [1.0 / scale]
    time_steps.extend(
        (context[index].time - context[index - 1].time) / scale
        for index in range(1, len(context))
    )
    time_channel = torch.tensor(time_steps, dtype=events.dtype, device=events.device)
    time_channel = time_channel.view(1, -1, 1).expand(events.shape[0], -1, -1)
    return torch.cat([time_channel, events], dim=-1)


def neighbor_mean_embeddings(
    snapshot: Snapshot,
    embeddings: Tensor,
    undirected: bool = True,
) -> Tensor:
    """Pool relation-neighbor embeddings without constructing a dense graph."""
    num_nodes = embeddings.shape[0]
    if snapshot.edge_index.numel() == 0:
        return torch.zeros_like(embeddings)
    if undirected:
        pairs = canonical_pairs(snapshot.edge_index, num_nodes, undirected=True)
        src = torch.cat([pairs[:, 0], pairs[:, 1]])
        dst = torch.cat([pairs[:, 1], pairs[:, 0]])
    else:
        src, dst = snapshot.edge_index
    pooled = torch.zeros_like(embeddings)
    pooled.index_add_(0, dst, embeddings[src])
    degree = torch.bincount(dst, minlength=num_nodes).to(embeddings.dtype).clamp_min_(1.0)
    return pooled / degree.unsqueeze(-1)


def new_neighbor_mean_embeddings(
    current: Snapshot,
    previous: Snapshot,
    embeddings: Tensor,
    undirected: bool = True,
) -> tuple[Tensor, Tensor]:
    """Pool embeddings of neighbors that appear only in ``current``.

    Returns a dense ``[num_nodes, hidden_dim]`` target and a boolean mask that
    marks nodes with at least one new neighbor.  Sorted edge keys and
    ``searchsorted`` avoid a dense adjacency matrix and keep the operation
    practical for DBLP-scale cumulative snapshots.
    """
    num_nodes = embeddings.shape[0]
    current_pairs = canonical_pairs(current.edge_index, num_nodes, undirected)
    previous_pairs = canonical_pairs(previous.edge_index, num_nodes, undirected)
    current_keys = _pair_keys(current_pairs, num_nodes, undirected)
    previous_keys = _pair_keys(previous_pairs, num_nodes, undirected)
    if previous_keys.numel() == 0:
        is_previous = torch.zeros_like(current_keys, dtype=torch.bool)
    else:
        positions = torch.searchsorted(previous_keys, current_keys)
        valid = positions < previous_keys.numel()
        safe_positions = positions.clamp_max(previous_keys.numel() - 1)
        is_previous = valid & (previous_keys[safe_positions] == current_keys)
    new_pairs = current_pairs[~is_previous]
    pooled = torch.zeros_like(embeddings)
    degree = torch.zeros(num_nodes, dtype=embeddings.dtype, device=embeddings.device)
    if new_pairs.numel() == 0:
        return pooled, degree.bool()
    if undirected:
        src = torch.cat([new_pairs[:, 0], new_pairs[:, 1]])
        dst = torch.cat([new_pairs[:, 1], new_pairs[:, 0]])
    else:
        src, dst = new_pairs[:, 0], new_pairs[:, 1]
    pooled.index_add_(0, dst, embeddings[src])
    degree = torch.bincount(dst, minlength=num_nodes).to(embeddings.dtype)
    mask = degree > 0
    pooled = pooled / degree.clamp_min(1.0).unsqueeze(-1)
    return pooled, mask


def binary_average_precision(labels: Tensor, probabilities: Tensor) -> float:
    labels = labels.detach().float()
    probabilities = probabilities.detach().float().to(labels.device)
    positives = int(labels.sum().item())
    if positives == 0:
        return float("nan")
    order = torch.argsort(probabilities, descending=True, stable=True)
    ranked = labels[order]
    precision = ranked.cumsum(0) / torch.arange(
        1, ranked.numel() + 1, device=ranked.device
    )
    return float(precision[ranked.bool()].mean().item())


def binary_roc_auc(labels: Tensor, probabilities: Tensor) -> float:
    labels = labels.detach().float()
    probabilities = probabilities.detach().float().to(labels.device)
    positive_count = int((labels == 1).sum().item())
    negative_count = int((labels == 0).sum().item())
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    # Mann-Whitney rank statistic is exactly equivalent to pairwise AUC while
    # requiring O(n), rather than O(n_positive * n_negative), intermediate
    # memory. Average ranks preserve the conventional 0.5 credit for ties.
    order = torch.argsort(probabilities, stable=True)
    sorted_scores = probabilities[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    average_ranks = (
        starts.to(probabilities.dtype) + 1.0 + ends.to(probabilities.dtype)
    ) / 2.0
    ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = ranks[sorted_labels == 1].sum()
    correction = positive_count * (positive_count + 1) / 2
    auc = (positive_rank_sum - correction) / (positive_count * negative_count)
    return float(auc.item())


def grouped_ranking_metrics(
    labels: Tensor,
    probabilities: Tensor,
    group_ids: Tensor,
    recall_k: int = 10,
) -> tuple[float, float]:
    """Return MRR and Recall@K for one-positive candidate groups."""
    labels = labels.detach().float()
    probabilities = probabilities.detach().float().to(labels.device)
    group_ids = group_ids.detach().long().to(labels.device)
    if labels.numel() == 0:
        return float("nan"), float("nan")
    _, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    positive_counts = torch.zeros(
        group_count, dtype=labels.dtype, device=labels.device
    )
    positive_counts.scatter_add_(0, inverse, labels)
    if not torch.all(positive_counts == 1):
        raise ValueError("each ranking group must contain exactly one positive")

    # Stable sorts preserve the original candidate order for tied scores.
    score_order = torch.argsort(probabilities, descending=True, stable=True)
    group_order = torch.argsort(inverse[score_order], stable=True)
    ordered_rows = score_order[group_order]
    ordered_groups = inverse[ordered_rows]
    positions = torch.arange(
        ordered_rows.numel(), device=labels.device, dtype=torch.long
    )
    group_start_markers = torch.where(
        torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=labels.device),
                ordered_groups[1:] != ordered_groups[:-1],
            ]
        ),
        positions,
        torch.zeros_like(positions),
    )
    group_starts = torch.cummax(group_start_markers, dim=0).values
    ranks = positions - group_starts + 1
    positive_ranks = ranks[labels[ordered_rows] == 1].float()
    return (
        float(positive_ranks.reciprocal().mean().item()),
        float((positive_ranks <= recall_k).float().mean().item()),
    )
